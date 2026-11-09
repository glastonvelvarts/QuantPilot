"""Recommendation engine: which model, quantization and runtime to use on this PC, and why.

The CLI (`ocq recommend`) and the TUI (Check page) both call this module, so they always agree.
It does not size anything itself: memory, layer split and speed come from fit_at, the same code
`ocq feasibility` uses. This module adds what to do with those rows:

1. candidates: every model x quantization x runtime x cache type worth trying
2. hard limits: a candidate that cannot run here is rejected, with the reason
3. ranking: the rest are ordered by what the workload cares about most
4. explanation: why the winner won, why the others lost, and how sure we are
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path

from ocq.engine import calibration_fits_in_ram
from ocq.model_analyzer import quant_options, resolve_model
from ocq.model_size import kv_value_bytes
from ocq.plan import free_disk_gb
from ocq.quantizers.llama_cpp import TAG as LLAMA_CPP_TAG
from ocq.quantizers.llama_cpp import converter_architectures
from ocq.types import (
    VLLM,
    HardwareProfile,
    ModelSnapshot,
    OcqError,
    QuantAlgorithm,
    QuantOption,
)


class Objective(str, Enum):
    """What a workload can care about. Each one is measured on its own, never added up."""

    QUALITY = "quality"  # how good the answers are (estimated until phase 7 measures it)
    SPEED = "speed"  # tokens per second for one conversation
    CONTEXT = "context"  # how much text fits in one conversation
    MEMORY = "memory"  # how much room is left for other programs
    THROUGHPUT = "throughput"  # conversations served at the same time

    def __str__(self) -> str:
        return self.value


@dataclass(frozen=True)
class WorkloadProfile:
    """What the user wants to do, in their words, and what that means for the ranking."""

    key: str
    label: str
    # Most important first. Candidates are compared on the first objective, and the next one
    # only breaks near-ties (see the ranking step).
    priorities: tuple[Objective, ...]
    # Tokens of context the workload needs when the user does not say.
    context: int
    # Conversations served at the same time.
    conversations: int = 1


WORKLOADS: dict[str, WorkloadProfile] = {
    profile.key: profile
    for profile in (
        WorkloadProfile(
            "chat",
            "General Chat",
            (Objective.QUALITY, Objective.SPEED, Objective.MEMORY),
            context=8192,
        ),
        WorkloadProfile(
            "coding",
            "Coding",
            (Objective.QUALITY, Objective.CONTEXT, Objective.SPEED),
            context=32768,
        ),
        WorkloadProfile(
            "rag",
            "RAG (documents)",
            (Objective.CONTEXT, Objective.QUALITY, Objective.SPEED),
            context=32768,
        ),
        WorkloadProfile(
            "agentic",
            "Agentic",
            (Objective.QUALITY, Objective.SPEED, Objective.CONTEXT),
            context=32768,
        ),
        WorkloadProfile(
            "reasoning",
            "Reasoning",
            (Objective.QUALITY, Objective.SPEED, Objective.CONTEXT),
            context=32768,
        ),
        WorkloadProfile(
            "long-context",
            "Long Context",
            (Objective.CONTEXT, Objective.QUALITY, Objective.MEMORY),
            context=131072,
        ),
        WorkloadProfile(
            "throughput",
            "High Throughput",
            (Objective.THROUGHPUT, Objective.MEMORY, Objective.QUALITY),
            context=8192,
            conversations=8,
        ),
    )
}
DEFAULT_WORKLOAD = "chat"


@dataclass(frozen=True)
class RecommendationRequest:
    """What the user asked for. Empty fields mean: work it out."""

    workload: WorkloadProfile = WORKLOADS[DEFAULT_WORKLOAD]
    # Models to consider (HF repo ids, local paths, ollama:name:tag). Empty: the models on this
    # PC plus the curated list of popular repos.
    models: tuple[str, ...] = ()
    # Tokens of context. 0: the workload's own.
    context: int = 0
    # Cache types to try. Empty: f16 and q8_0.
    kv_caches: tuple[str, ...] = ()

    @property
    def context_tokens(self) -> int:
        return self.context or self.workload.context


@dataclass(frozen=True)
class Candidate:
    """One way to run one model: a row of fit_at at one context and cache type."""

    model_ref: str  # what the user (or discovery) named it
    model_id: str  # the model it resolved to
    option: QuantOption  # memory, run mode, layer split and speed, from fit_at
    context: int
    kv_cache: str
    local: bool  # already on this PC; False means it has to be downloaded first
    model: ModelSnapshot = field(compare=False, repr=False)  # the facts the row was sized from
    # QuantPilot has to make this file first. False for files as published or as on disk.
    quantize: bool = False
    tags: frozenset[str] = frozenset()  # what the model is made for (CODER, REASONING)

    @property
    def runtime(self) -> str:
        return self.option.runtime

    @property
    def name(self) -> str:
        return f"{self.model_id} {self.option.label} · {self.runtime}"


@dataclass(frozen=True)
class Check:
    """The outcome of one hard limit for one candidate."""

    name: str  # memory, runtime, architecture, context, quantization
    passed: bool
    text: str  # plain words, shown when the check fails


@dataclass(frozen=True)
class Rejection:
    """A candidate that cannot run here, and every limit it broke."""

    candidate: Candidate
    failed: tuple[Check, ...]


class ReasonKind(str, Enum):
    GOOD = "good"  # ✓ a point in its favour
    TRADEOFF = "tradeoff"  # △ something given up for it
    BAD = "bad"  # ✗ why it lost or cannot run


@dataclass(frozen=True)
class Reason:
    kind: ReasonKind
    text: str


class ConfidenceLevel(str, Enum):
    LOW = "Low"
    MEDIUM = "Medium"
    HIGH = "High"
    VERY_HIGH = "Very High"

    def __str__(self) -> str:
        return self.value


@dataclass(frozen=True)
class Evidence:
    """One thing the recommendation rests on, and whether it was measured or guessed."""

    text: str  # "hardware measured", "no runs of this model on this PC yet"
    known: bool


@dataclass(frozen=True)
class Confidence:
    level: ConfidenceLevel
    evidence: tuple[Evidence, ...] = ()


@dataclass(frozen=True)
class Recommendation:
    """A candidate that can run here, with what it scores and why."""

    candidate: Candidate
    # Each objective on its own scale (0 worst, 1 best among the candidates). Used to order
    # them, never shown as a number.
    scores: dict[Objective, float] = field(default_factory=dict)
    reasons: tuple[Reason, ...] = ()
    warnings: tuple[str, ...] = ()
    confidence: Confidence = Confidence(ConfidenceLevel.LOW)


@dataclass(frozen=True)
class RecommendationResult:
    request: RecommendationRequest
    ranked: tuple[Recommendation, ...]  # best first
    rejected: tuple[Rejection, ...]
    # (model, why) for models whose files could not be read: offline, gated, broken.
    unread: tuple[tuple[str, str], ...] = ()

    @property
    def best(self) -> Recommendation | None:
        return self.ranked[0] if self.ranked else None

    @property
    def alternatives(self) -> tuple[Recommendation, ...]:
        return self.ranked[1:]


# Candidates (6B) ----------------------------------------------------------------------------

CODER = "coder"  # made for writing code
REASONING = "reasoning"  # thinks before it answers


@dataclass(frozen=True)
class CatalogEntry:
    repo: str
    tags: frozenset[str] = frozenset()


def entry(repo: str, *tags: str) -> CatalogEntry:
    return CatalogEntry(repo, frozenset(tags))


# Popular repos to suggest besides the models on this PC, from small to large, so every PC gets
# something. All ungated (no license to accept first), and all read by ocq inspect with every
# fact known, checked 2026-11-09. Tags only where the name or the model card says so.
CATALOG: tuple[CatalogEntry, ...] = (
    entry("Qwen/Qwen3.5-0.8B"),
    entry("LiquidAI/LFM3-1.7B"),
    entry("Qwen/Qwen3.5-2B"),
    entry("ibm-granite/granite-4.2-3b"),
    entry("nvidia/NVIDIA-Nemotron-3-Nano-4B-BF16", REASONING),
    entry("Qwen/Qwen3.5-4B"),
    entry("Qwen/Qwen2.5-Coder-7B-Instruct", CODER),
    entry("ibm-granite/granite-4.2-8b"),
    entry("Qwen/Qwen3.5-9B"),
    entry("openai/gpt-oss-20b", REASONING),
    entry("google/gemma-4-26B-A4B-it"),
    entry("Qwen/Qwen3.8-27B"),
    entry("Qwen/Qwen3-Coder-30B-A3B-Instruct", CODER),
    entry("nvidia/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-BF16", REASONING),
    entry("zai-org/GLM-4.7-Flash"),
    entry("google/gemma-4-31B-it"),
    entry("allenai/Olmo-3.1-32B-Think", REASONING),
    entry("Qwen/Qwen3.6-35B-A3B"),
    entry("openai/gpt-oss-120b", REASONING),
    entry("nvidia/NVIDIA-Nemotron-3-Super-120B-A12B-BF16", REASONING),
)
CATALOG_TAGS = {e.repo.lower(): e.tags for e in CATALOG}

DEFAULT_KV_CACHES = ("f16", "q8_0")
RESOLVE_WORKERS = 6  # Hub requests at once: each repo is a few small header reads


@dataclass(frozen=True)
class ModelSource:
    """A model to consider and where it comes from."""

    ref: str  # what resolve_model reads: a path, ollama:name:tag or a Hub repo id
    label: str  # how to name it
    local: bool
    tags: frozenset[str] = frozenset()


def model_sources(
    request: RecommendationRequest, extra_dirs: list[Path] | None = None
) -> list[ModelSource]:
    """The models the user named, or else the ones on this PC plus the catalog."""
    from ocq.discovery import find_models

    found = [model for model in find_models(extra_dirs) if model.complete and readable(model)]
    by_ref = {model.ref.lower(): model for model in found}
    if request.models:
        sources = []
        for ref in request.models:
            model = by_ref.get(ref.lower())
            if model is not None:
                sources.append(local_source(model))
            else:
                local = ref.startswith("ollama:") or Path(ref).expanduser().exists()
                sources.append(
                    ModelSource(ref, ref, local, CATALOG_TAGS.get(ref.lower(), frozenset()))
                )
        return sources
    sources = [local_source(model) for model in found]
    sources += [
        ModelSource(e.repo, e.repo, False, e.tags) for e in CATALOG if e.repo.lower() not in by_ref
    ]
    return sources


def readable(model) -> bool:
    """Formats QuantPilot reads the facts of (ocq models shows the others with a "?")."""
    from ocq.discovery import GGUF, MLX_LAYOUT, SAFETENSORS

    return model.format in (GGUF, SAFETENSORS, MLX_LAYOUT)


def local_source(model) -> ModelSource:
    # The copy on disk, not the Hub: no network, and exactly the files on this PC.
    ref = model.ref if model.ref.startswith("ollama:") else str(model.path)
    return ModelSource(ref, model.label, True, CATALOG_TAGS.get(model.ref.lower(), frozenset()))


def expand(
    source: ModelSource,
    snapshot: ModelSnapshot,
    hardware: HardwareProfile,
    context: int,
    kv_caches: tuple[str, ...],
) -> list[Candidate]:
    """Every row of quant_options (formats x runtimes) for every cache type.

    A cache type the row's runtime does not have (transformers keeps 16 bits, vLLM has no
    q8_0 below compute capability 8.9) would repeat the f16 row, so it is left out.
    """
    published = bool(snapshot.gguf_files)  # a GGUF repo or file: rows are files as they are
    candidates: list[Candidate] = []
    for kv_cache in kv_caches:
        for option in quant_options(snapshot, hardware, context, kv_cache):
            if kv_cache != "f16" and same_cache(kv_cache, option.runtime, hardware):
                continue
            candidates.append(
                Candidate(
                    model_ref=source.ref,
                    model_id=source.label,
                    option=option,
                    context=context,
                    kv_cache=kv_cache,
                    local=source.local,
                    model=snapshot,
                    quantize=not published and option.row_kind != "stored",
                    tags=source.tags,
                )
            )
    return candidates


def same_cache(kv_cache: str, runtime: str, hardware: HardwareProfile) -> bool:
    fp8 = hardware.vllm_fp8_kv_cache
    return kv_value_bytes(kv_cache, runtime, fp8) == kv_value_bytes("f16", runtime, fp8)


def generate_candidates(
    request: RecommendationRequest,
    hardware: HardwareProfile,
    extra_dirs: list[Path] | None = None,
) -> tuple[list[Candidate], list[tuple[str, str]]]:
    """(candidates, unread): every way to run every model, and the models that could not be
    read with the reason. Hub repos are read a few at a time."""
    sources = model_sources(request, extra_dirs)
    context = request.context_tokens
    kv_caches = request.kv_caches or DEFAULT_KV_CACHES

    def read(source: ModelSource) -> ModelSnapshot | str:
        try:
            return resolve_model(source.ref, None, None, context, "f16", use_llmfit=False)
        except OcqError as exc:
            return str(exc)
        except (OSError, ValueError, KeyError) as exc:  # a broken file must not hide the others
            return f"could not read it ({type(exc).__name__})"

    with ThreadPoolExecutor(RESOLVE_WORKERS) as pool:
        snapshots = list(pool.map(read, sources))

    candidates: list[Candidate] = []
    unread: list[tuple[str, str]] = []
    for source, snapshot in zip(sources, snapshots, strict=True):
        if isinstance(snapshot, str):
            unread.append((source.label, snapshot))
            continue
        try:
            candidates += expand(source, snapshot, hardware, context, kv_caches)
        except OcqError as exc:
            unread.append((source.label, str(exc)))
    return candidates, unread


# Hard limits (6C) ---------------------------------------------------------------------------
# A candidate that breaks one cannot run on this PC, whatever it scores, so it is rejected
# before ranking, with every reason in plain words. Each rule is one QuantPilot already
# applies elsewhere (fit_at, the quantizers' checks, the plan's warnings); none is new.

MEMORY = "memory"
RUNTIME = "runtime"
CONTEXT = "context"
ARCHITECTURE = "architecture"
QUANTIZATION = "quantization"
DISK = "disk"

CALIBRATED = (QuantAlgorithm.AWQ.value, QuantAlgorithm.GPTQ.value, QuantAlgorithm.NVFP4.value)
NEEDS_NVIDIA = (QuantAlgorithm.BNB.value, QuantAlgorithm.NVFP4.value)  # to run, not to make


def is_bnb(candidate: Candidate) -> bool:
    stored = str(candidate.model.facts.quantization.value or "")
    return candidate.option.row_kind == QuantAlgorithm.BNB.value or (
        candidate.option.row_kind == "stored" and stored.startswith("BNB")
    )


def supports_runtime(candidate: Candidate, hardware: HardwareProfile) -> Check:
    """The runtime runs on this PC at all."""
    runtime, row_kind = candidate.runtime, candidate.option.row_kind
    problem = ""
    if runtime == VLLM and not hardware.has_cuda:
        where = "a Mac" if hardware.apple_silicon else "this PC"
        problem = f"vLLM needs an NVIDIA GPU, and {where} has none."
    elif runtime == VLLM and hardware.os_name == "Windows":
        problem = "vLLM does not run on Windows itself, only in WSL2 or on Linux."
    elif (is_bnb(candidate) or row_kind in NEEDS_NVIDIA) and not hardware.has_cuda:
        name = "bitsandbytes" if is_bnb(candidate) else "NVFP4"
        problem = f"{name} needs an NVIDIA GPU, and this PC has none."
    elif candidate.model.is_mlx and not hardware.apple_silicon:
        problem = "MLX models run only on Apple Silicon Macs."
    return Check(RUNTIME, not problem, problem or f"{runtime} runs on this PC")


def fits_memory(candidate: Candidate) -> Check:
    option = candidate.option
    used = f"{option.memory_required_gb:.1f} GB"
    if option.runnable:
        return Check(MEMORY, True, f"needs {used} of {option.memory_available_gb:.1f} GB free")
    if option.note:  # vLLM's own reason, such as its share of the GPU not being free
        return Check(MEMORY, False, option.note)
    return Check(
        MEMORY, False, f"Needs {used}; only {option.memory_available_gb:.1f} GB is free for it."
    )


def supports_context(candidate: Candidate) -> Check:
    longest = candidate.model.context_length
    if longest and candidate.context > longest:
        return Check(
            CONTEXT,
            False,
            f"Holds up to {longest:,} tokens of context; {candidate.context:,} were asked for.",
        )
    return Check(CONTEXT, True, f"holds {candidate.context:,} tokens of context")


def supports_architecture(candidate: Candidate, gguf_architectures: frozenset[str] | None) -> Check:
    """Making a GGUF needs llama.cpp's converter to know the model's architecture. Files
    that already exist were made by a llama.cpp that knew it."""
    architecture = candidate.model.architecture
    if not (candidate.quantize and candidate.option.row_kind == "gguf"):
        return Check(ARCHITECTURE, True, "")
    if gguf_architectures is None:
        return Check(ARCHITECTURE, True, "not checked: llama.cpp's converter is not on this PC")
    if architecture not in gguf_architectures:
        return Check(
            ARCHITECTURE,
            False,
            f"llama.cpp ({LLAMA_CPP_TAG}) cannot convert {architecture} models to GGUF yet.",
        )
    return Check(ARCHITECTURE, True, f"llama.cpp converts {architecture}")


def supports_quantization(candidate: Candidate, hardware: HardwareProfile) -> Check:
    """QuantPilot can make the file on this PC."""
    row_kind = candidate.option.row_kind
    if not candidate.quantize:
        return Check(QUANTIZATION, True, "")
    if row_kind == QuantAlgorithm.AWQ.value and not hardware.has_cuda:
        return Check(QUANTIZATION, False, "Making an AWQ file needs an NVIDIA GPU.")
    if row_kind in CALIBRATED and not calibration_fits_in_ram(hardware, candidate.model):
        return Check(
            QUANTIZATION,
            False,
            f"Making it loads the whole original model ({candidate.model.original_gb:.1f} GB) "
            f"into RAM; {hardware.available_ram_gb:.1f} GB is free.",
        )
    return Check(QUANTIZATION, True, "")


def disk_needed_gb(candidate: Candidate) -> float | None:
    """Disk for the download and for making the file. None when the original size is unknown."""
    original = candidate.model.original_gb
    made = candidate.option.weights_gb if candidate.quantize else 0.0
    if candidate.quantize and candidate.option.row_kind == "gguf":
        made += original or 0.0  # the full-precision GGUF made first (as the plan counts it)
    if candidate.local:
        return made
    if candidate.quantize or candidate.option.row_kind == "stored":
        return None if original is None else original + made
    return candidate.option.weights_gb  # a published GGUF file


def fits_disk(candidate: Candidate, free_gb: float) -> Check:
    needed = disk_needed_gb(candidate)
    if needed is None:
        return Check(DISK, True, "not checked: the size of the original files is unknown")
    if needed > free_gb:
        what = "download it" if not candidate.quantize else "download and quantize it"
        if candidate.local:
            what = "quantize it"
        return Check(
            DISK, False, f"Needs {needed:.1f} GB of disk to {what}; {free_gb:.1f} GB is free."
        )
    return Check(DISK, True, f"needs {needed:.1f} GB of disk" if needed else "")


def check_candidate(
    candidate: Candidate,
    hardware: HardwareProfile,
    disk_free_gb: float,
    gguf_architectures: frozenset[str] | None,
) -> list[Check]:
    """Every hard limit for one candidate. When the runtime cannot run here, that is the only
    answer: the memory figures assume a runtime that runs."""
    runtime = supports_runtime(candidate, hardware)
    if not runtime.passed:
        return [runtime]
    return [
        runtime,
        fits_memory(candidate),
        supports_context(candidate),
        supports_architecture(candidate, gguf_architectures),
        supports_quantization(candidate, hardware),
        fits_disk(candidate, disk_free_gb),
    ]


def apply_limits(
    candidates: list[Candidate],
    hardware: HardwareProfile,
    disk_free_gb: float | None = None,
    gguf_architectures: frozenset[str] | None | bool = True,
) -> tuple[list[tuple[Candidate, list[Check]]], list[Rejection]]:
    """(allowed with the checks they passed, rejected). disk_free_gb defaults to the disk
    models are saved on, gguf_architectures (True) to what the installed converter knows."""
    if disk_free_gb is None:
        disk_free_gb = free_disk_gb(Path.home())
    if gguf_architectures is True:
        gguf_architectures = converter_architectures()
    allowed: list[tuple[Candidate, list[Check]]] = []
    rejected: list[Rejection] = []
    for candidate in candidates:
        checks = check_candidate(candidate, hardware, disk_free_gb, gguf_architectures)
        failed = tuple(check for check in checks if not check.passed)
        if failed:
            rejected.append(Rejection(candidate, failed))
        else:
            allowed.append((candidate, checks))
    return allowed, rejected
