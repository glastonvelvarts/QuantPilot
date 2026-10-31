"""Shared types for the OCQ pipeline."""

from __future__ import annotations

import re
from dataclasses import dataclass, field, replace
from enum import Enum
from pathlib import Path
from typing import Any

# Bits per element of every type a safetensors header can name (safetensors 0.6).
DTYPE_BITS = {
    "F64": 64,
    "I64": 64,
    "U64": 64,
    "F32": 32,
    "I32": 32,
    "U32": 32,
    "F16": 16,
    "BF16": 16,
    "I16": 16,
    "U16": 16,
    "F8_E4M3": 8,
    "F8_E5M2": 8,
    "F8_E8M0": 8,
    "I8": 8,
    "U8": 8,
    "BOOL": 8,
    "F6_E2M3": 6,
    "F6_E3M2": 6,
    "F4": 4,
}
INTEGER_TYPES = ("I64", "U64", "I32", "U32", "I16", "U16", "I8", "U8")

FLOAT_TYPES = ("F32", "BF16", "F16")

# Blackwell names, for when nvidia-smi cannot report the compute capability.
# "RTX 5000 Ada" is not Blackwell, the GeForce RTX 5050 to 5090 are.
BLACKWELL_NAMES = re.compile(r"blackwell|rtx 50[5-9]0|\bg?b(100|200|300)\b|\bgb10\b|\bthor\b")


class OcqError(RuntimeError):
    pass


class FactKind(str, Enum):
    EXACT = "exact"  # read from one of the model's files
    CROSS_CHECKED = "cross-checked"  # read from two sources that agree
    ESTIMATE = "estimate"  # derived, or the sources disagree
    UNKNOWN = "unknown"

    def __str__(self) -> str:
        return self.value


@dataclass(frozen=True)
class Fact:
    """One thing known about a model, where it was read from and how sure that is."""

    value: Any = None
    source: str = ""
    kind: FactKind = FactKind.UNKNOWN
    note: str = ""

    @property
    def known(self) -> bool:
        return self.kind != FactKind.UNKNOWN


# Runtime defaults the context memory depends on. They come from llama.cpp (build b11429), and
# its logs confirm them: Gemma 3 1B (window 512) at -c 32768 gets a sliding cache of
# "1024 cells" in llama-completion (1 conversation) and "2560 cells" in llama-server (4).
#   cells of a sliding layer = window x conversations + n_ubatch, padded to 256 cells
# Full layers share one cache of `context` cells between all conversations (kv_unified).
# vLLM and transformers keep no more than this.
SLIDING_WINDOW_UBATCH = 512  # llama.cpp -ub
KV_CELL_PADDING = 256
CONVERSATIONS = 4  # llama-server's default (n_parallel auto = 4), the largest of the runtimes
KV_BYTES_PER_VALUE = 2  # 16-bit keys and values (llama.cpp -ctk f16, vLLM kv-cache-dtype auto)

# The runtime that serves a row: GGUF in llama.cpp, AWQ/GPTQ/NVFP4 in vLLM, the original weights
# and bitsandbytes in transformers.
LLAMA_CPP = "llama.cpp"
VLLM = "vLLM"
TRANSFORMERS = "transformers"

# Cache types the user can ask for (--kv-cache), named as llama.cpp names them.
KV_CACHE_TYPES = ("f16", "q8_0", "q4_0")


@dataclass(frozen=True)
class KvLayer:
    """What one layer caches for every token of context."""

    bytes_per_token: int  # keys and values (16-bit), as llama.cpp and vLLM store them
    window: int = 0  # a sliding-window layer keeps only this many recent tokens; 0 = all
    # transformers expands MLA's compressed cache into full keys and values. 0 = same as above.
    transformers_bytes_per_token: int = 0
    # Mamba and convolution layers keep a fixed-size state per conversation instead of a cache
    # that grows (32-bit, llama.cpp's "llama_memory_recurrent").
    state_bytes: int = 0

    def cache_bytes(
        self,
        context: int,
        runtime: str = "",
        conversations: int = CONVERSATIONS,
        value_bytes: float = KV_BYTES_PER_VALUE,
    ) -> int:
        """value_bytes: what one cached value takes, 2 at 16-bit, 1.0625 for llama.cpp's q8_0."""
        per_token = self.bytes_per_token
        if runtime == TRANSFORMERS and self.transformers_bytes_per_token:
            per_token = self.transformers_bytes_per_token
        per_token = per_token * value_bytes / KV_BYTES_PER_VALUE
        tokens = context
        if self.window:
            wanted = self.window * conversations + SLIDING_WINDOW_UBATCH
            cells = -(-wanted // KV_CELL_PADDING) * KV_CELL_PADDING
            tokens = min(context, cells)
        return round(per_token * tokens) + self.state_bytes * conversations


@dataclass(frozen=True)
class ModelFacts:
    parameters: Fact = Fact()  # int: weights as the model counts them, packed ones unpacked
    precision: Fact = Fact()  # str: the type most weights are stored in
    stored_bytes: Fact = Fact()  # int: bytes of every tensor in the weight files
    architecture: Fact = Fact()  # str: model_type
    context_length: Fact = Fact()  # int: tokens
    kv_cache: Fact = Fact()  # int: bytes of keys and values (16-bit) per token of context
    active_parameters: Fact = Fact()  # int: weights read for one token (all of them when dense)
    quantization: Fact = Fact()  # str: "none", or a label such as "GPTQ 4-bit"


class OptimizationGoal(str, Enum):
    BALANCED = "balanced"
    SPEED = "speed"
    QUALITY = "quality"
    MEMORY = "memory"


class QuantAlgorithm(str, Enum):
    AWQ = "awq"
    GPTQ = "gptq"
    BNB = "bnb"
    GGUF = "gguf"
    NVFP4 = "nvfp4"


class RunMode(str, Enum):
    GPU = "gpu"
    GPU_FULL = "gpu_full"
    GPU_OFFLOAD = "gpu_offload"
    # Mixture of experts in llama.cpp: experts of some layers in RAM, everything else in VRAM.
    GPU_EXPERTS_RAM = "gpu_experts_ram"
    CPU = "cpu"
    CPU_ONLY = "cpu_only"
    UNFIT = "unfit"

    def __str__(self) -> str:
        return self.value


class FitLevel(str, Enum):
    PERFECT = "Perfect"  # Fits entirely in VRAM with headroom
    GOOD = "Good"  # Fits comfortably in VRAM
    MARGINAL = "Marginal"  # Requires heavy offloading or tight RAM
    TOO_TIGHT = "Too Tight"  # Out of memory / unusable

    def __str__(self) -> str:
        return self.value


@dataclass(frozen=True)
class GpuProfile:
    name: str
    vram_gb: float | None = None
    backend: str = "unknown"
    bandwidth_gbs: float = 0.0
    count: int = 1
    unified_memory: bool = False
    free_vram_gb: float | None = None
    measured_bandwidth_gbs: float = 0.0
    # CUDA compute capability, 12.0 for an RTX 5090. None when unknown.
    compute_capability: float | None = None

    @property
    def vram_bytes(self) -> int:
        return int(self.vram_gb * (1024**3)) if self.vram_gb is not None else 0

    @property
    def fp4_tensor_cores(self) -> bool:
        """Blackwell (compute capability 10 and up) computes in FP4 natively."""
        if "cuda" not in self.backend.lower():
            return False
        if self.compute_capability is not None:
            return self.compute_capability >= 10.0
        return bool(BLACKWELL_NAMES.search(self.name.lower()))

    @property
    def read_speed(self) -> float:
        """GiB the card reads per second: measured on this PC, else its specification."""
        return self.measured_bandwidth_gbs or self.bandwidth_gbs * 1e9 / 1024**3


# How a runtime spreads one model over several GPUs.
# llama.cpp and transformers put whole layers on each card: the free VRAM adds up, and the
# cards work one after another. vLLM gives every card an equal slice of every layer (tensor
# parallel): N cards hold N x the smallest card, and they read their slices at the same time.
ONE_GPU = "one GPU"
LAYER_SPLIT = "layers"
TENSOR_PARALLEL = "tensor parallel"


@dataclass(frozen=True)
class GpuPool:
    """The GPUs one runtime would use together for a model."""

    cards: tuple[GpuProfile, ...]
    free_gb: float
    read_speed: float  # GiB per second for reading all the weights once; 0 when unknown
    split: str

    @property
    def count(self) -> int:
        return len(self.cards)


@dataclass(frozen=True)
class HardwareProfile:
    """Hardware snapshot from llmfit — estimates only."""

    total_ram_gb: float
    available_ram_gb: float
    cpu_cores: int
    cpu_name: str
    has_gpu: bool
    backend: str
    unified_memory: bool
    ram_bandwidth_gbs: float = 0.0
    pcie_bandwidth_gbs: float = 0.0
    measured_ram_bandwidth_gbs: float = 0.0
    gpus: tuple[GpuProfile, ...] = ()
    os_name: str = ""
    source: str = "llmfit"  # llmfit | native
    raw: dict[str, Any] = field(default_factory=dict, repr=False)

    @property
    def has_cuda(self) -> bool:
        return self.has_gpu and "cuda" in self.backend.lower()

    @property
    def total_ram_bytes(self) -> int:
        return int(self.total_ram_gb * (1024**3))

    @property
    def available_ram_bytes(self) -> int:
        return int(self.available_ram_gb * (1024**3))

    @property
    def primary_vram_bytes(self) -> int | None:
        if not self.gpus:
            return None
        values = [g.vram_bytes for g in self.gpus if g.vram_gb is not None]
        return max(values) if values else None

    @property
    def primary_vram_gb(self) -> float | None:
        if not self.gpus:
            return None
        values = [g.vram_gb for g in self.gpus if g.vram_gb is not None]
        return max(values) if values else None

    @property
    def primary_gpu_bandwidth_gbs(self) -> float:
        if not self.gpus:
            return 0.0
        return max(g.bandwidth_gbs for g in self.gpus)

    @property
    def primary_gpu(self) -> GpuProfile | None:
        with_vram = [g for g in self.gpus if g.vram_gb is not None]
        return max(with_vram, key=lambda g: g.vram_gb) if with_vram else None

    @property
    def vllm_fp8_kv_cache(self) -> bool:
        """vLLM rejects --kv-cache-dtype fp8 below compute capability 8.9 (Ada, Hopper and
        Blackwell have it; Ampere and older do not). Unknown counts as no."""
        cards = [card for card in self.cards if "cuda" in card.backend.lower()]
        return (
            self.has_cuda
            and bool(cards)
            and all(
                card.compute_capability is not None and card.compute_capability >= 8.9
                for card in cards
            )
        )

    @property
    def has_fp4_tensor_cores(self) -> bool:
        gpu = self.primary_gpu
        return self.has_cuda and gpu is not None and gpu.fp4_tensor_cores

    @property
    def cards(self) -> tuple[GpuProfile, ...]:
        """Every GPU of the main backend, one entry per card.

        llmfit lists identical cards once with a count and one card's VRAM in vram_gb.
        """
        with_vram = [g for g in self.gpus if g.vram_gb is not None]
        main = [g for g in with_vram if g.backend.lower() == self.backend.lower()] or with_vram
        cards = []
        for gpu in main:
            each = replace(gpu, count=1, free_vram_gb=gpu.free_vram_gb if gpu.count == 1 else None)
            cards += [each] * gpu.count
        return tuple(cards)

    def card_free_gb(self, gpu: GpuProfile) -> float:
        """VRAM a model can use on this card right now. Unified memory shares the free RAM."""
        if self.unified_memory:
            return self.available_ram_gb
        return gpu.free_vram_gb if gpu.free_vram_gb is not None else gpu.vram_gb

    def gpu_pool(
        self, split: str = LAYER_SPLIT, attention_heads: int | None = None
    ) -> GpuPool | None:
        """The GPUs a runtime would use, with their free VRAM and read speed together.

        Tensor parallel needs the card count to divide the attention heads (vLLM checks it), so
        without the head count only one card is assumed.
        """
        cards = sorted(self.cards, key=self.card_free_gb, reverse=True)
        if not self.has_gpu or not cards:
            return None
        if self.unified_memory or len(cards) == 1 or split == ONE_GPU:
            best = cards[0]
            return GpuPool((best,), self.card_free_gb(best), best.read_speed, ONE_GPU)

        if split == LAYER_SPLIT:
            # llama.cpp splits by free memory: each card holds and reads its share in turn.
            free = [self.card_free_gb(card) for card in cards]
            total = sum(free)
            seconds_per_gib = 0.0
            for card, share in zip(cards, free, strict=True):
                if not card.read_speed:
                    seconds_per_gib = 0.0
                    break
                seconds_per_gib += share / total / card.read_speed
            speed = 1 / seconds_per_gib if seconds_per_gib else 0.0
            return GpuPool(tuple(cards), total, speed, LAYER_SPLIT)

        best = None
        for count in range(1, len(cards) + 1):
            if count > 1 and (not attention_heads or attention_heads % count):
                continue
            used = cards[:count]
            free = count * min(self.card_free_gb(card) for card in used)
            speed = count * min(card.read_speed for card in used)
            if best is None or free > best.free_gb:
                kind = TENSOR_PARALLEL if count > 1 else ONE_GPU
                best = GpuPool(tuple(used), free, speed, kind)
        return best

    @property
    def free_vram_gb(self) -> float:
        """VRAM a model can use right now, over every card when it is split by layers."""
        pool = self.gpu_pool(LAYER_SPLIT)
        return pool.free_gb if pool else 0.0


@dataclass(frozen=True)
class FeasibilityEstimate:
    """First-pass feasibility from llmfit for one quantization tier."""

    quant_label: str
    fit_level: FitLevel | str
    run_mode: RunMode | str
    score: float
    estimated_tps: float
    memory_required_gb: float
    memory_available_gb: float
    utilization_pct: float
    runtime: str
    estimate_confidence: str
    usable_context: int | None = None
    gpu_layers_offloaded: int = 0
    total_layers: int = 0
    raw: dict[str, Any] = field(default_factory=dict, repr=False)

    @property
    def memory_required_bytes(self) -> int:
        return int(self.memory_required_gb * (1024**3))

    @property
    def memory_available_bytes(self) -> int:
        return int(self.memory_available_gb * (1024**3))

    @property
    def runnable(self) -> bool:
        if isinstance(self.fit_level, FitLevel):
            return self.fit_level != FitLevel.TOO_TIGHT
        return str(self.fit_level).lower() not in ("too tight", "unknown")


@dataclass(frozen=True)
class QuantOption:
    """One row of the "what can this PC run" table."""

    label: str
    bits: float
    weights_gb: float
    context_gb: float
    exact: bool
    memory_required_gb: float
    memory_available_gb: float
    fit_level: FitLevel
    run_mode: RunMode
    estimated_tps: float
    speed_basis: str = ""
    # stored = checkpoint as saved; gguf = llama.cpp quant; awq/gptq/bnb = HF serving formats
    row_kind: str = "gguf"
    # False when config.json has no layer or head counts: context_gb is then 0, not measured,
    # and the fit counts the weights only.
    context_known: bool = True
    # GPUs the model runs on (none on the CPU), and how they share it: one GPU, layers,
    # tensor parallel.
    gpu_names: tuple[str, ...] = ()
    gpu_split: str = ""
    # GPU_EXPERTS_RAM: the experts of this many layers (the first ones) are kept in RAM.
    experts_in_ram: int = 0

    @property
    def runnable(self) -> bool:
        return self.fit_level != FitLevel.TOO_TIGHT

    @property
    def gpu_count(self) -> int:
        return len(self.gpu_names)

    @property
    def utilization_pct(self) -> float:
        if self.memory_available_gb <= 0:
            return 999.0
        return self.memory_required_gb / self.memory_available_gb * 100


@dataclass(frozen=True)
class ModelSnapshot:
    model_id: str
    provider: str = ""
    parameter_count: str | int = ""  # Store exact count or display string ("7B")
    params_b: float = 0.0
    context_length: int | None = 4096
    use_case: str = "General"
    is_moe: bool = False
    # Mixture of experts: routed experts per layer, and how many run for each token.
    experts: int = 0
    experts_per_token: int = 0
    llmfit_best_quant: str = "Q4_K_M"
    feasibility: FeasibilityEstimate = field(
        default_factory=lambda: FeasibilityEstimate(
            quant_label="unknown",
            fit_level=FitLevel.GOOD,
            run_mode=RunMode.GPU_FULL,
            score=0.0,
            estimated_tps=0.0,
            memory_required_gb=0.0,
            memory_available_gb=0.0,
            utilization_pct=0.0,
            runtime="unknown",
            estimate_confidence="unknown",
        )
    )

    # Structural specs for KV cache and execution calculations
    architecture: str = "unknown"
    num_layers: int = 32
    hidden_size: int = 4096
    num_attention_heads: int = 32
    num_kv_heads: int = 32
    vocab_size: int = 32000
    attention_head_dim: int = 0

    # Name and shape of every weight tensor, read from the model's files without downloading them.
    tensors: tuple[tuple[str, tuple[int, ...]], ...] = field(default=(), repr=False)

    catalog_source: str = "llmfit"  # llmfit | huggingface | local
    weights_dir: Path | None = None  # local HuggingFace-format folder; skip Hub download
    facts: ModelFacts = field(default_factory=ModelFacts)
    # One entry per layer, from config.json. Empty when config.json does not describe them.
    kv_layers: tuple[KvLayer, ...] = field(default=(), repr=False)
    raw: dict[str, Any] = field(default_factory=dict, repr=False)

    def __post_init__(self) -> None:
        if (
            self.params_b == 0.0
            and isinstance(self.parameter_count, (int, float))
            and self.parameter_count > 0
        ):
            object.__setattr__(self, "params_b", float(self.parameter_count) / 1_000_000_000.0)

    @property
    def architecture_known(self) -> bool:
        return self.architecture != "unknown"

    @property
    def stored_precision(self) -> str:
        """F32, BF16 or F16: the type most of the weights are stored in. Empty otherwise."""
        precision = self.facts.precision.value
        return precision if precision in FLOAT_TYPES else ""

    @property
    def original_gb(self) -> float | None:
        """Size of the weights as the repo stores them. None when it could not be read."""
        stored = self.facts.stored_bytes
        return stored.value / 1024**3 if stored.known else None

    @property
    def is_quantized(self) -> bool:
        quantization = self.facts.quantization
        return quantization.known and quantization.value != "none"

    def facts_summary(self) -> str:
        """How sure the facts behind the sizes are, in one line, for feasibility and plan."""
        facts = self.facts
        if not facts.parameters.known and self.catalog_source == "llmfit":
            return "from llmfit's catalog, not read from the model's files"
        parts = [f"parameters {facts.parameters.kind}", f"context memory {facts.kv_cache.kind}"]
        if self.is_moe:
            parts.append(f"used per token {facts.active_parameters.kind}")
        if self.is_quantized:
            parts.append(f"quantization {facts.quantization.kind}")
        return ", ".join(parts) + " (sources: ocq inspect)"

    @property
    def read_fraction(self) -> float:
        """Share of the weights read for each token: below 1 for mixture-of-experts models,
        and for models with a separate input embedding table, of which one row is read."""
        active, total = self.facts.active_parameters, self.facts.parameters
        if active.known and total.value:
            return min(active.value / total.value, 1.0)
        return 1.0

    def kv_cache_bytes(
        self,
        context: int,
        runtime: str = "",
        conversations: int = CONVERSATIONS,
        value_bytes: float = KV_BYTES_PER_VALUE,
    ) -> int | None:
        """Keys and values for `context` tokens. None when they could not be worked out.

        Sliding-window layers stop growing at their window (per conversation), and MLA caches a
        compressed vector (expanded by transformers). runtime is TRANSFORMERS or "" for
        llama.cpp and vLLM.
        """
        if self.kv_layers:
            return sum(
                layer.cache_bytes(context, runtime, conversations, value_bytes)
                for layer in self.kv_layers
            )
        scale = value_bytes / KV_BYTES_PER_VALUE
        if self.facts.kv_cache.known:
            return round(self.facts.kv_cache.value * context * scale)
        if self.architecture_known:
            per_token = 2 * self.num_layers * self.num_kv_heads * self.head_dim * KV_BYTES_PER_VALUE
            return round(per_token * context * scale)
        return None

    @property
    def head_dim(self) -> int:
        if self.attention_head_dim:
            return self.attention_head_dim
        if self.num_attention_heads > 0:
            return self.hidden_size // self.num_attention_heads
        return 0


@dataclass(frozen=True)
class QuantChoices:
    """What the user asked for on the command line. Anything left as None is decided for them."""

    algorithm: QuantAlgorithm | None = None
    bits: int | None = None
    gguf_type: str | None = None
    group_size: int | None = None
    calibration_samples: int | None = None
    calibration_seq_len: int | None = None
    context: int | None = None
    kv_cache: str | None = None
    double_quant: bool | None = None
    fp4_activations: bool | None = None


@dataclass(frozen=True)
class QuantConfig:
    algorithm: QuantAlgorithm
    bits: int
    group_size: int
    calibration_samples: int = 128
    calibration_seq_len: int = 512
    target_quant_label: str = "Q4_K_M"
    gguf_type: str = ""
    context: int = 4096
    kv_cache: str = "f16"  # f16, q8_0 or q4_0 (KV_CACHE_TYPES)
    # bitsandbytes 4-bit only: also quantize the per-block scales.
    double_quant: bool = False
    # NVFP4 only: activations in FP4 too (Blackwell), or weights only.
    fp4_activations: bool = False

    def summary(self) -> str:
        if self.algorithm == QuantAlgorithm.GGUF:
            return f"GGUF {self.gguf_type} ({self.bits}-bit)"
        if self.algorithm == QuantAlgorithm.BNB and self.bits == 4:
            quant = "double" if self.double_quant else "single"
            return f"BNB 4-bit NF4, {quant} quantization"
        if self.algorithm == QuantAlgorithm.NVFP4:
            what = "weights and activations" if self.fp4_activations else "weights only"
            return f"NVFP4 4-bit ({what})"
        gs = f", gs={self.group_size}" if self.group_size else ""
        return f"{self.algorithm.value.upper()} {self.bits}-bit{gs}"


@dataclass(frozen=True)
class BackendStatus:
    algorithm: QuantAlgorithm
    ready: bool
    detail: str


@dataclass
class OptimizationPlan:
    model_id: str
    goal: OptimizationGoal
    hardware: HardwareProfile
    model: ModelSnapshot
    quant_config: QuantConfig
    output_dir: Path
    rationale: list[str] = field(default_factory=list)
    llmfit_hypothesis_tps: float = 0.0
    warnings: list[str] = field(default_factory=list)
    predicted: QuantOption | None = None


@dataclass
class QuantizationResult:
    plan: OptimizationPlan
    artifact_path: Path | None
    backend: str
    success: bool
    message: str
    artifact_size_gb: float = 0.0


@dataclass
class ValidationResult:
    passed: bool
    checks: list[str]
    sample_output: str | None = None


@dataclass
class BenchmarkResult:
    """Ground-truth throughput after quantization — not llmfit's estimate."""

    tokens_per_second: float | None
    method: str
    notes: list[str] = field(default_factory=list)
    time_to_first_token_s: float | None = None
    device: str = ""
    sample_output: str = ""


@dataclass
class OptimizeResult:
    plan: OptimizationPlan
    quantization: QuantizationResult | None = None
    validation: ValidationResult | None = None
    benchmark: BenchmarkResult | None = None
    dry_run: bool = False
