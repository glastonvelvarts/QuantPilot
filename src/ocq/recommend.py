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

import math
import re
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field, replace
from enum import Enum
from pathlib import Path

from ocq.engine import calibration_fits_in_ram
from ocq.measurements import MEASURED, runs_of
from ocq.model_analyzer import fit_at, quant_options, resolve_model
from ocq.model_size import kv_value_bytes
from ocq.plan import free_disk_gb
from ocq.quantizers.llama_cpp import TAG as LLAMA_CPP_TAG
from ocq.quantizers.llama_cpp import converter_architectures
from ocq.types import (
    VLLM,
    FactKind,
    FitLevel,
    HardwareProfile,
    ModelSnapshot,
    OcqError,
    QuantAlgorithm,
    QuantOption,
    RunMode,
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
    # The best one: why it won. Any other: why it lost to the best one.
    reasons: tuple[Reason, ...] = ()
    warnings: tuple[str, ...] = ()
    confidence: Confidence = Confidence(ConfidenceLevel.LOW)
    # Each objective in its own unit (effective parameters, tok/s, tokens, 1/GB).
    values: dict[Objective, float] = field(default_factory=dict)
    checks: tuple[Check, ...] = ()  # the hard limits it passed
    # Where it dropped out while the best one was picked: LOST_ON_DRAWBACKS, an Objective's
    # value, or LOST_ON_TIE. Empty for the best one.
    lost_on: str = ""


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
        """The next best of each other model and format: the same file with another cache
        type is the same choice, so only its best ranked version is listed."""
        seen = set()
        distinct = []
        for recommendation in self.ranked:
            key = (recommendation.candidate.model_ref, recommendation.candidate.option.label)
            if key not in seen:
                seen.add(key)
                distinct.append(recommendation)
        return tuple(distinct[1:])


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
            return f"could not read it: {exc or type(exc).__name__}"

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


# Ranking (6D) -------------------------------------------------------------------------------
# No single weighted score. Candidates are first split into tiers (fits comfortably, fast enough
# to read along), then picked one at a time: keep those within a tolerance of the best on the
# workload's first objective, of those the ones within a tolerance of the best on the second,
# and so on; the tie-breaks decide what is left. So a later objective only ever chooses among
# candidates that are as good as the best on every earlier one.

# Quality: an estimate until phase 7 measures it. Two published sources:
# 1. What each GGUF type costs, llama.cpp b11429 (tools/quantize/quantize.cpp, "+x ppl @
#    Llama-3-8B"; tools/perplexity/README.md: Llama 3 8B BF16 perplexity 6.225194).
LLAMA3_8B_PPL = 6.225194
LLAMA3_8B_PARAMETERS = 8.03e9
GGUF_PPL_INCREASE = {
    "Q8_0": 0.0026,
    "Q6_K": 0.0217,
    "Q5_K_M": 0.0569,
    "Q5_K_S": 0.1049,
    "Q5_1": 0.1062,
    "Q5_0": 0.1316,
    "Q4_K_M": 0.1754,
    "Q4_K_S": 0.2689,
    "Q4_1": 0.4511,
    "Q4_0": 0.4685,
    "Q3_K_L": 0.5562,
    "Q3_K_M": 0.6569,
    "Q3_K_S": 1.6321,
    "Q2_K_S": 3.1836,
    "Q2_K": 3.5199,
}
# 2. How much perplexity is worth in parameters, measured by the same tool on the same text:
#    llama.cpp's README at ffb06a3 (2023-05-30) has LLaMA 7B (6.7B parameters) at 5.9066 and
#    13B (13.0B) at 5.2543 in F16. Within a family, ln(perplexity) falls by SIZE_SLOPE for
#    every e-fold of parameters, so a type that adds dL to ln(perplexity) leaves a model as
#    good as one with exp(-dL / SIZE_SLOPE) of its parameters. (The same table: 13B at Q4_0,
#    5.3860, still beats 7B at F16.) The share is applied to models of every size; bigger
#    models lose less to quantization, so this is on the safe side for them.
SIZE_SLOPE = math.log(5.9066 / 5.2543) / math.log(13.0 / 6.7)
# Keeping less than this share of its quality, a model is worse than a smaller one at a
# milder type (Q3_K_S and Q2_K on Llama-3-8B).
DEGRADED = 0.5

# Formats with no llama.cpp measurement, rated as the GGUF type closest to how they quantize.
FULL = "full"
AS_RELEASED = "as released"
CALIBRATED_4BIT = "Q4_K_S"  # AWQ, GPTQ, NVFP4: 4-bit groups with importance-aware scales
PLAIN_4BIT = "Q4_0"  # bitsandbytes NF4, MLX: 4-bit groups rounded to nearest
BITS_AS_GGUF = {2: "Q2_K", 3: "Q3_K_S", 4: PLAIN_4BIT, 5: "Q5_0", 6: "Q6_K", 8: "Q8_0"}

# Tiers. Reading speed: about 238 words a minute (Brysbaert 2019), 4 words a second, at
# about 1.3 tokens a word. Slower than that, a chat makes people wait for every line.
READING_SPEED_TPS = 6.0
COMFORTABLE = (FitLevel.PERFECT, FitLevel.GOOD)  # Marginal: over 85% of the free memory
CONTEXT_ROOM = 4
SLOW = "slower than reading speed"
TIGHT = "uses most of the free memory"
DEGRADED_FORMAT = "keeps less than half of the model's quality"
SPECIALIST = "made for code"

LOST_ON_DRAWBACKS = "drawbacks"
LOST_ON_TIE = "tie"

# Differences smaller than these are within what the estimates can tell apart.
TOLERANCE = {
    Objective.QUALITY: 0.05,  # within a model the losses are measured: Q8_0 and Q6_K tie
    Objective.SPEED: 0.10,
    Objective.CONTEXT: 0.0,  # already in whole steps (doubling)
    Objective.MEMORY: 0.10,
    Objective.THROUGHPUT: 0.10,
}


def quality_kept(gguf_type: str) -> float:
    """Share of a model's quality a GGUF type keeps (FULL and AS_RELEASED keep all of it)."""
    if gguf_type in (FULL, AS_RELEASED):
        return 1.0
    increase = GGUF_PPL_INCREASE[gguf_type]
    loss_added = math.log((LLAMA3_8B_PPL + increase) / LLAMA3_8B_PPL)
    return math.exp(-loss_added / SIZE_SLOPE)


def rated_as(candidate: Candidate) -> str:
    """The GGUF type a candidate's quality is read from, FULL or AS_RELEASED."""
    option = candidate.option
    label = option.label.removesuffix(" (as stored)").strip()
    upper = label.upper()
    if option.row_kind == "stored" and not candidate.model.is_quantized:
        return FULL  # the original weights
    if upper in GGUF_PPL_INCREASE:
        return upper
    if upper in ("F32", "F16", "BF16"):
        return FULL
    if upper.startswith("MXFP4"):
        return AS_RELEASED  # gpt-oss was trained for and released in MXFP4
    bits = re.search(r"(\d+)-BIT|FP(\d+)|Q(\d)|INT(\d+)", upper)
    width = int(next(group for group in bits.groups() if group)) if bits else 0
    if width >= 8:
        return "Q8_0"
    if width == 4 and upper.startswith(("AWQ", "GPTQ", "NVFP4", "MODELOPT", "IQ4")):
        return CALIBRATED_4BIT
    # No width in the name (an unknown method): the most common, 4-bit.
    return BITS_AS_GGUF.get(width, "Q2_K" if width else PLAIN_4BIT)


def effective_parameters(model: ModelSnapshot) -> float:
    """Parameters to compare models by. A mixture of experts counts as the geometric mean of
    its total and active parameters, a rule of thumb (30B with 3B active is about a 9.5B)."""
    total = model.facts.parameters.value or model.params_b * 1e9
    active = model.facts.active_parameters
    if model.is_moe and active.known and active.value:
        return math.sqrt(total * active.value)
    return float(total)


def estimated_quality(candidate: Candidate) -> float:
    """Effective parameters the candidate is worth: its size times the share its format keeps."""
    return effective_parameters(candidate.model) * quality_kept(rated_as(candidate))


def refit(candidate: Candidate, hardware: HardwareProfile, context: int) -> QuantOption:
    """The candidate's row at another context: the same call quant_options makes."""
    option = candidate.option
    return fit_at(
        option.label,
        option.weights_gb,
        candidate.model,
        hardware,
        context,
        use_measurements=option.row_kind == "gguf",
        row_kind=option.row_kind,
        kv_cache=candidate.kv_cache,
        runtime=option.runtime,
    )


def longest_context(candidate: Candidate, hardware: HardwareProfile, limit: int) -> int:
    """The longest context that still fits, doubling from the one asked for, up to limit."""
    best, context = candidate.context, candidate.context
    while context < limit:
        context = min(context * 2, limit)
        if not refit(candidate, hardware, context).runnable:
            break
        best = context
    return best


def objective_values(
    candidate: Candidate, workload: WorkloadProfile, hardware: HardwareProfile
) -> dict[Objective, float]:
    """Each objective as a number where more is better."""
    option = candidate.option
    values = {
        Objective.QUALITY: estimated_quality(candidate),
        Objective.SPEED: option.estimated_tps,
        Objective.MEMORY: 1 / max(option.memory_required_gb, 0.01),
    }
    wanted = set(workload.priorities)
    if Objective.CONTEXT in wanted:
        # Room for more text, counted up to CONTEXT_ROOM times what was asked: past that, more
        # room does not change what the workload can do.
        limit = min(
            candidate.model.context_length or candidate.context, CONTEXT_ROOM * candidate.context
        )
        values[Objective.CONTEXT] = float(longest_context(candidate, hardware, limit))
    if Objective.THROUGHPUT in wanted:
        # Decoding reads the weights once per step for every conversation in the batch, so
        # while memory speed is the limit, tokens per second grow with the conversations
        # that fit (an upper bound: compute becomes the limit at some point). The
        # conversations share one cache (llama.cpp's unified cache, vLLM's pages).
        conversations = 1  # the hard limits already checked one
        while conversations < workload.conversations:
            cells = candidate.context * (conversations + 1)
            if not refit(candidate, hardware, cells).runnable:
                break
            conversations += 1
        values[Objective.THROUGHPUT] = conversations * option.estimated_tps
    return values


def drawbacks(candidate: Candidate, workload: WorkloadProfile) -> list[str]:
    """What makes a candidate a poor pick whatever it scores. Candidates with fewer come first."""
    found = []
    if workload.conversations == 1 and candidate.option.estimated_tps < READING_SPEED_TPS:
        found.append(SLOW)
    if candidate.option.fit_level not in COMFORTABLE:
        found.append(TIGHT)
    if quality_kept(rated_as(candidate)) < DEGRADED:
        found.append(DEGRADED_FORMAT)
    if CODER in candidate.tags and workload.key != "coding":
        found.append(SPECIALIST)
    return found


def tier(candidate: Candidate, workload: WorkloadProfile) -> int:
    return len(drawbacks(candidate, workload))


def tie_break(candidate: Candidate, values: dict[Objective, float], workload: WorkloadProfile):
    """What decides between candidates equally good on every objective: a model made for the
    workload, the default cache, files already here, no quantizing, then quality and speed."""
    made_for = {"coding": CODER, "reasoning": REASONING}.get(workload.key)
    return (
        made_for is not None and made_for not in candidate.tags,
        candidate.kv_cache != "f16",
        not candidate.local,
        candidate.quantize,
        -values[Objective.QUALITY],
        -values[Objective.SPEED],
    )


def within(value: float, best: float, tolerance: float) -> bool:
    return value >= best * (1 - tolerance) if best > 0 else value >= best


def rank(
    allowed: list[Candidate], workload: WorkloadProfile, hardware: HardwareProfile
) -> list[Recommendation]:
    """Best first. Each Recommendation's scores are its values over the best of them (0 to 1)."""
    values = {id(c): objective_values(c, workload, hardware) for c in allowed}
    remaining = list(allowed)
    ranked: list[Candidate] = []
    lost_on: dict[int, str] = {}  # where each one dropped out while the best one was picked
    while remaining:
        first = not ranked
        best_tier = min(tier(c, workload) for c in remaining)
        pool = [c for c in remaining if tier(c, workload) == best_tier]
        if first:
            lost_on |= {id(c): LOST_ON_DRAWBACKS for c in remaining if c not in pool}
        for objective in workload.priorities:
            best = max(values[id(c)][objective] for c in pool)
            kept = [c for c in pool if within(values[id(c)][objective], best, TOLERANCE[objective])]
            if first:
                lost_on |= {id(c): objective.value for c in pool if c not in kept}
            pool = kept
        winner = min(pool, key=lambda c: tie_break(c, values[id(c)], workload))
        if first:
            lost_on |= {id(c): LOST_ON_TIE for c in pool if c is not winner}
        ranked.append(winner)
        remaining.remove(winner)

    tops = {
        objective: max((values[id(c)][objective] for c in allowed), default=0.0)
        for objective in (values[id(allowed[0])] if allowed else {})
    }
    return [
        Recommendation(
            c,
            scores={
                objective: (value / tops[objective] if tops[objective] else 0.0)
                for objective, value in values[id(c)].items()
            },
            values=values[id(c)],
            lost_on=lost_on.get(id(c), ""),
        )
        for c in ranked
    ]


def recommend(
    request: RecommendationRequest,
    hardware: HardwareProfile,
    extra_dirs: list[Path] | None = None,
    disk_free_gb: float | None = None,
) -> RecommendationResult:
    """The whole engine: candidates, hard limits, ranking, explanations. The CLI and the TUI
    call this."""
    candidates, unread = generate_candidates(request, hardware, extra_dirs)
    allowed, rejected = apply_limits(candidates, hardware, disk_free_gb)
    checks = {id(candidate): tuple(passed) for candidate, passed in allowed}
    ranked = rank([candidate for candidate, _ in allowed], request.workload, hardware)
    ranked = [replace(r, checks=checks[id(r.candidate)]) for r in ranked]
    ranked = explain(ranked, request.workload, hardware)
    return RecommendationResult(request, tuple(ranked), tuple(rejected), tuple(unread))


# Explanations and confidence (6E) -----------------------------------------------------------
# Never a score: what the pick does on this PC, what it gives up, why each alternative lost,
# and what the advice rests on (measured or estimated).

GOOD, TRADEOFF, BAD = ReasonKind.GOOD, ReasonKind.TRADEOFF, ReasonKind.BAD
MADE_FOR = {"coding": (CODER, "Made for code"), "reasoning": (REASONING, "Made for reasoning")}
KNOWN_KINDS = (FactKind.EXACT, FactKind.CROSS_CHECKED)
TRADEOFFS_SHOWN = 2


def tokens(count: float) -> str:
    count = int(count)
    return f"{count // 1024}K" if count >= 1024 and count % 1024 == 0 else f"{count:,}"


def percent(share: float) -> str:
    return f"{share:.0%}"


def short_name(candidate: Candidate, other: Candidate) -> str:
    """How to name a candidate next to another: only what differs."""
    if candidate.model_ref != other.model_ref:
        return f"{candidate.model_id} {candidate.option.label}"
    if candidate.option.label == other.option.label:
        return f"the {candidate.kv_cache} cache"
    if candidate.kv_cache != other.kv_cache:
        return f"{candidate.option.label} with the {candidate.kv_cache} cache"
    return candidate.option.label


def option_name(candidate: Candidate) -> str:
    return f"{candidate.option.label} in {candidate.runtime}, {candidate.kv_cache} cache"


def fit_reason(candidate: Candidate, hardware: HardwareProfile) -> Reason:
    option = candidate.option
    used = f"{option.memory_required_gb:.1f} of {option.memory_available_gb:.1f} GB free"
    mode = option.run_mode
    kind = GOOD if option.fit_level in COMFORTABLE else TRADEOFF
    if mode == RunMode.GPU and hardware.unified_memory:
        text = f"Fits in memory: {used}"
    elif mode == RunMode.GPU:
        cards = f" across {option.gpu_count} GPUs" if option.gpu_count > 1 else ""
        text = f"Fits entirely in GPU memory{cards}: {used}"
    elif mode == RunMode.GPU_EXPERTS_RAM:
        text = f"Fits on the GPU with the experts of {option.experts_in_ram} layers in RAM"
    elif mode == RunMode.GPU_OFFLOAD:
        kind = TRADEOFF
        layers = candidate.model.num_layers + 1  # llama.cpp counts the output layer too
        on_gpu = f"{option.gpu_layers} of {layers} layers" if option.gpu_layers else "part"
        text = f"Runs partly from RAM: {on_gpu} on the GPU"
    elif hardware.has_gpu and not hardware.unified_memory:
        kind = TRADEOFF
        text = f"Runs from RAM on the CPU, too big for the GPU: {used}"
    else:
        text = f"Runs from RAM: {used}"
    if option.fit_level not in COMFORTABLE:
        text += ", most of the free memory"
    return Reason(kind, text)


def speed_reason(candidate: Candidate) -> Reason:
    option = candidate.option
    runs = runs_of(candidate.model.model_id, option.label, candidate.runtime)
    if runs:
        low, high = min(r["tps"] for r in runs), max(r["tps"] for r in runs)
        span = f"{low:.0f}" if round(low) == round(high) else f"{low:.0f}–{high:.0f}"
        text = f"Measured {span} tok/s in {len(runs)} run{'s' * (len(runs) > 1)} on this PC"
    elif option.speed_basis == MEASURED:
        text = f"About {option.estimated_tps:.0f} tok/s, from earlier runs on this PC"
    else:
        text = f"Up to {option.estimated_tps:.0f} tok/s, as fast as this PC's memory allows"
    slow = option.estimated_tps < READING_SPEED_TPS
    return Reason(TRADEOFF if slow else GOOD, text + (", slower than reading" if slow else ""))


def quality_reason(candidate: Candidate) -> Reason:
    rating = rated_as(candidate)
    kept = quality_kept(rating)
    label = candidate.option.label
    if rating == FULL:
        text = "Full quality: the original weights"
    elif rating == AS_RELEASED:
        text = "Full quality: the format the model was released in"
    elif rating == label.upper():
        text = (
            f"{label} keeps about {percent(kept)} of the model's quality (llama.cpp's measurements)"
        )
    else:
        text = f"Keeps about {percent(kept)} of the model's quality (estimated as {rating})"
    return Reason(BAD if kept < DEGRADED else GOOD, text)


def difference(
    objective: Objective, candidate: Recommendation, other: Recommendation, exact: bool = False
) -> tuple[int, str]:
    """(1 better, -1 worse, 0 about the same) and how, for candidate against other. exact:
    any difference counts, not only one beyond the objective's tolerance."""
    mine, theirs = candidate.values.get(objective), other.values.get(objective)
    if mine is None or theirs is None:
        return 0, ""
    tolerance = 0.0 if exact else TOLERANCE[objective]
    if (
        theirs * (1 - tolerance) <= mine <= theirs / (1 - tolerance)
        if tolerance
        else mine == theirs
    ):
        return 0, ""
    better = mine > theirs
    a, b = candidate.candidate, other.candidate
    if objective == Objective.QUALITY and a.model_ref == b.model_ref:
        kept = (quality_kept(rated_as(a)), quality_kept(rated_as(b)))
        word = "more" if better else "less"
        text = f"keeps {word} of the model's quality ({percent(kept[0])} vs {percent(kept[1])})"
    elif objective == Objective.QUALITY:
        word = "stronger" if better else "weaker"
        text = f"estimated {word}: like a {mine / 1e9:.1f}B model vs {theirs / 1e9:.1f}B"
    elif objective == Objective.SPEED:
        word = "faster" if better else "slower"
        text = f"{abs(mine / theirs - 1):.0%} {word} ({mine:.0f} vs {theirs:.0f} tok/s)"
    elif objective == Objective.CONTEXT:
        word = "more" if better else "less"
        text = f"room for {word} context ({tokens(mine)} vs {tokens(theirs)} tokens)"
    elif objective == Objective.MEMORY:
        word = "less" if better else "more"
        gb = (a.option.memory_required_gb, b.option.memory_required_gb)
        text = f"uses {word} memory ({gb[0]:.1f} vs {gb[1]:.1f} GB)"
    else:
        word = "more" if better else "less"
        text = f"serves {word} at once ({mine:.0f} vs {theirs:.0f} tok/s in all)"
    return (1 if better else -1), text


def tie_reason(candidate: Candidate, best: Candidate, workload: WorkloadProfile) -> str:
    """Why the best one won a tie, from the first tie-break that tells them apart."""
    made_for = MADE_FOR.get(workload.key, (None, ""))[0]
    if made_for and (made_for in best.tags) != (made_for in candidate.tags):
        return f"the pick is {MADE_FOR[workload.key][1].lower()}"
    if candidate.kv_cache != best.kv_cache:
        return "the pick uses the default (f16) cache"
    if candidate.local != best.local:
        return "the pick is already on this PC"
    if candidate.quantize != best.quantize:
        return "the pick needs no quantizing"
    return "the pick is a little better on quality or speed"


def winner_reasons(
    best: Recommendation,
    alternatives: tuple[Recommendation, ...],
    workload: WorkloadProfile,
    hardware: HardwareProfile,
) -> tuple[Reason, ...]:
    candidate = best.candidate
    reasons = [fit_reason(candidate, hardware), speed_reason(candidate)]
    room = best.values.get(Objective.CONTEXT, candidate.context)
    text = f"Holds the {tokens(candidate.context)} tokens of context asked for"
    if room > candidate.context:
        text += f", room for {tokens(room)}"
    reasons.append(Reason(GOOD, text))
    reasons.append(quality_reason(candidate))
    tag, words = MADE_FOR.get(workload.key, (None, ""))
    if tag and tag in candidate.tags:
        reasons.append(Reason(GOOD, words))
    if SPECIALIST in drawbacks(candidate, workload):
        reasons.append(Reason(TRADEOFF, "Made for code, not for this kind of work"))
    if not candidate.local:
        size = disk_needed_gb(replace(candidate, quantize=False))
        download = f" (about {size:.1f} GB)" if size else ""
        reasons.append(Reason(TRADEOFF, f"Not on this PC yet: download it first{download}"))
    if candidate.quantize:
        reasons.append(
            Reason(TRADEOFF, f"QuantPilot makes the {candidate.option.label} file first")
        )
    # What the best ranked alternatives that gain something would give, and what they cost.
    shown = 0
    for other in alternatives:
        if shown == TRADEOFFS_SHOWN:
            break
        gains, costs = [], []
        for objective in workload.priorities:
            if other.lost_on == objective.value:
                costs.append(difference(objective, other, best, exact=True)[1])
                continue
            sign, how = difference(objective, other, best)
            (gains if sign > 0 else costs if sign < 0 else []).append(how)
        if gains and costs:
            name = short_name(other.candidate, candidate)
            if name.startswith("the "):  # "the q8_0 cache": a sentence start, not a model name
                name = name.capitalize()
            reasons.append(Reason(TRADEOFF, f"{name}: {', '.join(gains)}, but {', '.join(costs)}"))
            shown += 1
    return tuple(reasons)


def loser_reasons(
    other: Recommendation, best: Recommendation, workload: WorkloadProfile
) -> tuple[Reason, ...]:
    """Why other ranks below best, replaying where it dropped out of the ranking: its
    drawbacks, or the priority it fell short on (with what it does better before that), or
    the tie-break."""
    mine, theirs = drawbacks(other.candidate, workload), drawbacks(best.candidate, workload)
    reasons = [Reason(BAD, capital(d)) for d in mine if d not in theirs]
    for objective in workload.priorities:
        if other.lost_on == objective.value:
            _, how = difference(objective, other, best, exact=True)
            reasons.append(Reason(BAD, capital(how)))
            return tuple(reasons)
        sign, how = difference(objective, other, best)
        if sign > 0:
            reasons.append(Reason(GOOD, capital(how)))
    if other.lost_on == LOST_ON_TIE:
        why = tie_reason(other.candidate, best.candidate, workload)
        reasons.append(Reason(TRADEOFF, f"As good as the pick here; {why}"))
    return tuple(reasons)


def capital(text: str) -> str:
    return text[:1].upper() + text[1:]


def confidence(recommendation: Recommendation, hardware: HardwareProfile) -> Confidence:
    """What the advice rests on. Medium: the model read from its files, speed an upper bound.
    High: speed from runs on this PC. Very High: runs of this very setup. Low: a fact the
    advice depends on is estimated, or a check could not be made."""
    candidate, option = recommendation.candidate, recommendation.candidate.option
    facts = candidate.model.facts
    evidence: list[Evidence] = []

    sized = facts.parameters.kind in KNOWN_KINDS
    evidence.append(
        Evidence("Model size read from its files" if sized else "Model size estimated", sized)
    )
    cache_note = facts.kv_cache.note or ""
    context_known = (
        option.context_known
        and facts.kv_cache.kind in KNOWN_KINDS
        and "unknown layer kind" not in cache_note
    )
    if context_known:
        context_text = "Context memory read from its files"
    elif "unknown size" in cache_note:
        context_text = "Context memory estimated: some layers keep a state of unknown size"
    elif "unknown layer kind" in cache_note:
        context_text = "Context memory estimated: some layers are of an unknown kind"
    else:
        context_text = "Context memory estimated"
    evidence.append(Evidence(context_text, context_known))
    moe_known = not candidate.model.is_moe or facts.active_parameters.known
    if candidate.model.is_moe:
        evidence.append(
            Evidence(
                "Weights used per token read from its files"
                if moe_known
                else "Weights used per token unknown",
                moe_known,
            )
        )
    gpu = option.run_mode in (RunMode.GPU, RunMode.GPU_EXPERTS_RAM) and not hardware.unified_memory
    card = hardware.primary_gpu
    bandwidth = card.measured_bandwidth_gbs if gpu and card else hardware.measured_ram_bandwidth_gbs
    evidence.append(
        Evidence(
            "Memory speed measured on this PC" if bandwidth else "Memory speed from the specs",
            bool(bandwidth),
        )
    )
    runs = runs_of(candidate.model.model_id, option.label, candidate.runtime)
    measured = option.speed_basis == MEASURED
    if runs:
        evidence.append(
            Evidence(f"{len(runs)} run{'s' * (len(runs) > 1)} of this setup on this PC", True)
        )
    else:
        evidence.append(
            Evidence(
                "Speed from earlier runs on this PC" if measured else "Speed is an upper bound",
                measured,
            )
        )
    evidence.append(Evidence("Quality estimated from size and format (not measured yet)", False))
    unchecked = [check for check in recommendation.checks if check.text.startswith("not checked")]
    for check in unchecked:
        evidence.append(Evidence(f"{check.name.capitalize()} {check.text}", False))

    if not (sized and context_known and moe_known) or unchecked:
        level = ConfidenceLevel.LOW
    elif runs:
        level = ConfidenceLevel.VERY_HIGH
    elif measured:
        level = ConfidenceLevel.HIGH
    else:
        level = ConfidenceLevel.MEDIUM
    return Confidence(level, tuple(evidence))


def explain(
    ranked: list[Recommendation], workload: WorkloadProfile, hardware: HardwareProfile
) -> list[Recommendation]:
    """Reasons, warnings and confidence for every ranked candidate."""
    if not ranked:
        return []
    best = ranked[0]
    shown = RecommendationResult(RecommendationRequest(workload), tuple(ranked), ())
    explained = []
    for recommendation in ranked:
        if recommendation is best:
            reasons = winner_reasons(best, shown.alternatives, workload, hardware)
        else:
            reasons = loser_reasons(recommendation, best, workload)
        note = recommendation.candidate.option.note
        explained.append(
            replace(
                recommendation,
                reasons=reasons,
                warnings=(note,) if note else (),
                confidence=confidence(recommendation, hardware),
            )
        )
    return explained


def explain_model(result: RecommendationResult, model: str) -> tuple[Reason, ...]:
    """Why not this model? Its best ranked option against the pick, else the rejected option
    that came closest (a runtime that runs here, fewest limits broken, least memory), else why
    it was not read."""
    name = model.lower()

    def named(candidate: Candidate) -> bool:
        return name in (candidate.model_ref.lower(), candidate.model_id.lower())

    for recommendation in result.ranked:
        if named(recommendation.candidate):
            option = option_name(recommendation.candidate)
            if recommendation is result.best:
                return (Reason(GOOD, f"It is the pick: {option}"), *recommendation.reasons)
            lead = Reason(TRADEOFF, f"Its best option here: {option}")
            return (lead, *recommendation.reasons)
    rejected = [r for r in result.rejected if named(r.candidate)]
    if rejected:
        # A runtime that cannot run here stops at that one check, so it is not the closest.
        closest = min(
            rejected,
            key=lambda r: (
                r.failed[0].name == RUNTIME,
                len(r.failed),
                r.candidate.option.memory_required_gb,
            ),
        )
        lead = Reason(TRADEOFF, f"Its closest option: {option_name(closest.candidate)}")
        return (lead, *(Reason(BAD, check.text) for check in closest.failed))
    for label, why in result.unread:
        if label.lower() == name:
            return (Reason(BAD, f"Could not be read: {why}"),)
    return (Reason(BAD, "Not among the models considered"),)


# What the CLI and the TUI show (6F) ---------------------------------------------------------
# Both draw from these, so they always list the same rows with the same words.


@dataclass(frozen=True)
class CompareRow:
    """One format of one model: the pick, a ranked alternative, or a rejected one."""

    candidate: Candidate
    status: str  # PICK, RANKED or REJECTED
    why: str  # one line: why it is the pick, why not, or the limit it broke
    limit: str = ""  # REJECTED: the name of the first limit it broke (MEMORY, RUNTIME...)


PICK, RANKED, REJECTED = "pick", "ranked", "rejected"


def first_reason(reasons: tuple[Reason, ...]) -> str:
    for kind in (BAD, TRADEOFF):
        for reason in reasons:
            if reason.kind == kind:
                return reason.text
    return ""


def compare_rows(result: RecommendationResult, model_ref: str = "") -> list[CompareRow]:
    """Every format of one model (all models when empty), best first: each format's best
    ranked version, then the formats that cannot run here with the limit they broke."""

    def wanted(candidate: Candidate) -> bool:
        return not model_ref or candidate.model_ref == model_ref

    rows: list[CompareRow] = []
    seen: set[tuple[str, str]] = set()
    for recommendation in result.ranked:
        candidate = recommendation.candidate
        key = (candidate.model_ref, candidate.option.label)
        if not wanted(candidate) or key in seen:
            continue
        seen.add(key)
        if recommendation is result.best:
            rows.append(CompareRow(candidate, PICK, "Recommended"))
        else:
            rows.append(CompareRow(candidate, RANKED, first_reason(recommendation.reasons)))
    for rejection in result.rejected:
        candidate = rejection.candidate
        key = (candidate.model_ref, candidate.option.label)
        if not wanted(candidate) or key in seen or candidate.kv_cache != "f16":
            continue
        seen.add(key)
        first = rejection.failed[0]
        rows.append(CompareRow(candidate, REJECTED, first.text, first.name))
    return rows


def best_per_model(result: RecommendationResult) -> list[Recommendation]:
    """Each model's best ranked option, best model first (for a list of models to try)."""
    seen: set[str] = set()
    best = []
    for recommendation in result.ranked:
        if recommendation.candidate.model_ref not in seen:
            seen.add(recommendation.candidate.model_ref)
            best.append(recommendation)
    return best


LIMIT_WORDS = {
    MEMORY: "too big for the free memory",
    RUNTIME: "need a runtime this PC cannot run",
    CONTEXT: "hold less context than asked",
    ARCHITECTURE: "cannot be converted to GGUF yet",
    QUANTIZATION: "cannot be quantized on this PC",
    DISK: "need more disk than is free",
}


def rejected_summary(result: RecommendationResult) -> list[tuple[int, str]]:
    """(how many, why) for the options ruled out, by the first limit each broke."""
    counts: dict[str, int] = {}
    for rejection in result.rejected:
        name = rejection.failed[0].name
        counts[name] = counts.get(name, 0) + 1
    return sorted(((n, LIMIT_WORDS[name]) for name, n in counts.items()), reverse=True)
