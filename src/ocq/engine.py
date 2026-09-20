"""Optimization engine — selects algorithm, bit width, and calibration params."""

from __future__ import annotations

from ocq.types import (
    HardwareProfile,
    ModelSnapshot,
    OptimizationGoal,
    QuantAlgorithm,
    QuantConfig,
)

# Map llmfit GGUF labels to approximate target bit widths for HF quant backends.
_GGUF_TO_BITS: dict[str, int] = {
    "Q8_0": 8,
    "Q6_K": 6,
    "Q5_K_M": 5,
    "Q5_K_S": 5,
    "Q4_K_M": 4,
    "Q4_0": 4,
    "Q3_K_M": 3,
    "Q2_K": 2,
    "AWQ-4bit": 4,
    "GPTQ-Int4": 4,
    "mlx-4bit": 4,
    "mlx-8bit": 8,
}


def decide_quant_config(
    hardware: HardwareProfile,
    model: ModelSnapshot,
    goal: OptimizationGoal,
    algorithm: QuantAlgorithm | None = None,
) -> QuantConfig:
    """Pick quantization algorithm and parameters from hardware + llmfit tier."""

    bits = _bits_for_goal(model, goal)
    if algorithm is None:
        algorithm = _pick_algorithm(hardware, model, bits)
    elif algorithm == QuantAlgorithm.AWQ:
        bits = 4
    group_size = _default_group_size(algorithm, bits)
    target_label = model.llmfit_best_quant

    return QuantConfig(
        algorithm=algorithm,
        bits=bits,
        group_size=group_size,
        calibration_samples=_calibration_samples(model.params_b),
        calibration_seq_len=512,
        target_quant_label=target_label,
    )


def build_rationale(
    hardware: HardwareProfile,
    model: ModelSnapshot,
    config: QuantConfig,
    goal: OptimizationGoal,
) -> list[str]:
    lines = [
        f"Goal: {goal.value}",
        f"llmfit fit: {model.feasibility.fit_level} ({model.feasibility.run_mode})",
        (
            f"llmfit hypothesis: {model.feasibility.estimated_tps:.1f} tok/s "
            f"@ {model.feasibility.quant_label} "
            f"({model.feasibility.estimate_confidence})"
        ),
        f"Memory: {model.feasibility.memory_required_gb:.1f} GB required / "
        f"{model.feasibility.memory_available_gb:.1f} GB available "
        f"({model.feasibility.utilization_pct:.0f}% util)",
        f"Selected backend: {config.summary()}",
    ]
    backend = hardware.backend.lower()
    if config.algorithm == QuantAlgorithm.GGUF:
        lines.append("GGUF selected - llama.cpp export for CPU/GPU cross-platform inference.")
    elif config.algorithm == QuantAlgorithm.AWQ:
        lines.append("AWQ selected - 4-bit activation-aware weight quantization for CUDA.")
    elif "cuda" in backend:
        lines.append("CUDA detected - AWQ/GPTQ preferred for deployment throughput.")
    elif "metal" in backend:
        lines.append("Apple Silicon - consider MLX export post-quant (v2); BNB for HF weights.")
    elif not hardware.has_gpu:
        lines.append("No GPU - BNB 4-bit load-in-8bit path for CPU/RAM constrained inference.")
    if not model.feasibility.runnable:
        lines.append(
            "Warning: llmfit marks this model Too Tight - quantization may still help "
            "but validate memory after quant."
        )
    return lines


def _bits_for_goal(model: ModelSnapshot, goal: OptimizationGoal) -> int:
    base = _gguf_bits(model.llmfit_best_quant)
    if goal == OptimizationGoal.QUALITY:
        return min(8, max(base, 4))
    if goal == OptimizationGoal.MEMORY:
        return min(base, 4)
    if goal == OptimizationGoal.SPEED:
        return min(base, 4)
    return base


def _gguf_bits(label: str) -> int:
    for key, bits in _GGUF_TO_BITS.items():
        if key.upper() in label.upper():
            return bits
    return 4


def _pick_algorithm(
    hardware: HardwareProfile,
    model: ModelSnapshot,
    bits: int,
) -> QuantAlgorithm:
    backend = hardware.backend.lower()
    runtime = model.feasibility.runtime.lower()

    if "awq" in model.llmfit_best_quant.lower():
        return QuantAlgorithm.AWQ
    if "gptq" in model.llmfit_best_quant.lower():
        return QuantAlgorithm.GPTQ

    if hardware.has_gpu and ("cuda" in backend or "rocm" in backend):
        # AWQ tends to win on Ampere+; GPTQ has broader tooling
        if bits <= 4 and model.params_b >= 13:
            return QuantAlgorithm.AWQ
        return QuantAlgorithm.GPTQ

    if "metal" in backend or "mlx" in runtime:
        return QuantAlgorithm.BNB

    if not hardware.has_gpu or (hardware.primary_vram_gb or 0) < (
        model.feasibility.memory_required_gb
    ):
        return QuantAlgorithm.BNB

    return QuantAlgorithm.GPTQ


def _default_group_size(algorithm: QuantAlgorithm, bits: int) -> int:
    if algorithm == QuantAlgorithm.BNB:
        return 0
    if bits <= 4:
        return 128
    return 64


def _calibration_samples(params_b: float) -> int:
    if params_b >= 70:
        return 256
    if params_b >= 13:
        return 128
    return 64
