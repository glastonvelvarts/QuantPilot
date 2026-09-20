"""Shared types for the OCQ pipeline."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any


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


class RunMode(str, Enum):
    GPU = "gpu"
    GPU_FULL = "gpu_full"
    GPU_OFFLOAD = "gpu_offload"
    CPU = "cpu"
    CPU_ONLY = "cpu_only"
    UNFIT = "unfit"

    def __str__(self) -> str:
        return self.value


class FitLevel(str, Enum):
    PERFECT = "Perfect"    # Fits entirely in VRAM with headroom
    GOOD = "Good"          # Fits comfortably in VRAM
    MARGINAL = "Marginal"  # Requires heavy offloading or tight RAM
    TOO_TIGHT = "Too Tight"# Out of memory / unusable

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

    @property
    def vram_bytes(self) -> int:
        return int(self.vram_gb * (1024 ** 3)) if self.vram_gb is not None else 0


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
    raw: dict[str, Any] = field(default_factory=dict, repr=False)

    @property
    def total_ram_bytes(self) -> int:
        return int(self.total_ram_gb * (1024 ** 3))

    @property
    def available_ram_bytes(self) -> int:
        return int(self.available_ram_gb * (1024 ** 3))

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
        return int(self.memory_required_gb * (1024 ** 3))

    @property
    def memory_available_bytes(self) -> int:
        return int(self.memory_available_gb * (1024 ** 3))

    @property
    def runnable(self) -> bool:
        if isinstance(self.fit_level, FitLevel):
            return self.fit_level != FitLevel.TOO_TIGHT
        return str(self.fit_level).lower() != "too tight"


@dataclass(frozen=True)
class ModelSnapshot:
    model_id: str
    provider: str = ""
    parameter_count: str | int = ""  # Store exact count or display string ("7B")
    params_b: float = 0.0
    context_length: int | None = 4096
    use_case: str = "General"
    is_moe: bool = False
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

    catalog_source: str = "llmfit"  # llmfit | huggingface
    raw: dict[str, Any] = field(default_factory=dict, repr=False)

    def __post_init__(self) -> None:
        if self.params_b == 0.0 and isinstance(self.parameter_count, (int, float)) and self.parameter_count > 0:
            object.__setattr__(self, "params_b", float(self.parameter_count) / 1_000_000_000.0)

    @property
    def head_dim(self) -> int:
        if self.num_attention_heads > 0:
            return self.hidden_size // self.num_attention_heads
        return 0


@dataclass(frozen=True)
class QuantConfig:
    algorithm: QuantAlgorithm
    bits: int
    group_size: int
    calibration_samples: int = 128
    calibration_seq_len: int = 512
    target_quant_label: str = "Q4_K_M"

    def summary(self) -> str:
        gs = f", gs={self.group_size}" if self.group_size else ""
        return f"{self.algorithm.value.upper()} {self.bits}-bit{gs}"


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


@dataclass
class QuantizationResult:
    plan: OptimizationPlan
    artifact_path: Path | None
    backend: str
    success: bool
    message: str


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


@dataclass
class OptimizeResult:
    plan: OptimizationPlan
    quantization: QuantizationResult | None = None
    validation: ValidationResult | None = None
    benchmark: BenchmarkResult | None = None
    dry_run: bool = False
