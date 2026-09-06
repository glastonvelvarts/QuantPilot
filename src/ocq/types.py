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


@dataclass(frozen=True)
class GpuProfile:
    name: str
    vram_gb: float | None
    backend: str
    count: int = 1
    unified_memory: bool = False


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
    gpus: tuple[GpuProfile, ...] = ()
    raw: dict[str, Any] = field(default_factory=dict, repr=False)

    @property
    def primary_vram_gb(self) -> float | None:
        if not self.gpus:
            return None
        values = [g.vram_gb for g in self.gpus if g.vram_gb is not None]
        return max(values) if values else None


@dataclass(frozen=True)
class FeasibilityEstimate:
    """First-pass feasibility from llmfit for one quantization tier."""

    quant_label: str
    fit_level: str
    run_mode: str
    score: float
    estimated_tps: float
    memory_required_gb: float
    memory_available_gb: float
    utilization_pct: float
    runtime: str
    estimate_confidence: str
    usable_context: int | None = None
    raw: dict[str, Any] = field(default_factory=dict, repr=False)

    @property
    def runnable(self) -> bool:
        return self.fit_level.lower() != "too tight"


@dataclass(frozen=True)
class ModelSnapshot:
    model_id: str
    provider: str
    parameter_count: str
    params_b: float
    context_length: int | None
    use_case: str
    is_moe: bool
    llmfit_best_quant: str
    feasibility: FeasibilityEstimate
    catalog_source: str = "llmfit"  # llmfit | huggingface
    raw: dict[str, Any] = field(default_factory=dict, repr=False)


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
