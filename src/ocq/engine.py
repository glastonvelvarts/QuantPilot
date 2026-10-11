"""Optimization engine: picks the backend, bit width and calibration settings."""

from __future__ import annotations

from ocq.model_size import group_size_that_fits
from ocq.types import (
    HardwareProfile,
    ModelSnapshot,
    OcqError,
    OptimizationGoal,
    QuantAlgorithm,
    QuantChoices,
    QuantConfig,
)

SUPPORTED_BITS = {
    QuantAlgorithm.GGUF: (2, 3, 4, 5, 6, 8),
    QuantAlgorithm.AWQ: (4,),
    QuantAlgorithm.GPTQ: (4, 8),
    QuantAlgorithm.BNB: (4, 8),
}

GGUF_TYPES = {
    "Q2_K": 2,
    "Q3_K_S": 3,
    "Q3_K_M": 3,
    "Q3_K_L": 3,
    "Q4_0": 4,
    "Q4_K_S": 4,
    "Q4_K_M": 4,
    "Q5_0": 5,
    "Q5_K_S": 5,
    "Q5_K_M": 5,
    "Q6_K": 6,
    "Q8_0": 8,
}

DEFAULT_GGUF_TYPE = {2: "Q2_K", 3: "Q3_K_M", 4: "Q4_K_M", 5: "Q5_K_M", 6: "Q6_K", 8: "Q8_0"}

# Bit width implied by the quantization label llmfit recommends.
LABEL_BITS = {**GGUF_TYPES, "AWQ-4bit": 4, "GPTQ-Int4": 4, "mlx-4bit": 4, "mlx-8bit": 8}


def decide_quant_config(
    hardware: HardwareProfile,
    model: ModelSnapshot,
    goal: OptimizationGoal,
    choices: QuantChoices | None = None,
) -> QuantConfig:
    choices = choices or QuantChoices()
    algorithm = choices.algorithm
    asked_bits = choices.bits
    gguf_type = (choices.gguf_type or "").upper()

    if gguf_type:
        if gguf_type not in GGUF_TYPES:
            raise OcqError(f"Unknown GGUF type {gguf_type!r}. Use one of: {', '.join(GGUF_TYPES)}")
        if algorithm not in (None, QuantAlgorithm.GGUF):
            raise OcqError("--gguf-type only works with the gguf backend.")
        if asked_bits and asked_bits != GGUF_TYPES[gguf_type]:
            raise OcqError(f"{gguf_type} is {GGUF_TYPES[gguf_type]}-bit, not {asked_bits}-bit.")
        algorithm = QuantAlgorithm.GGUF
        asked_bits = GGUF_TYPES[gguf_type]

    bits = asked_bits or bits_for_goal(model, goal)
    if algorithm is None:
        algorithm = pick_algorithm(hardware, model, bits)

    supported = SUPPORTED_BITS[algorithm]
    if bits not in supported:
        if asked_bits:
            allowed = ", ".join(str(b) for b in supported)
            raise OcqError(
                f"{algorithm.value.upper()} supports {allowed}-bit, not {bits}-bit. "
                "GGUF covers 2 to 8 bits."
            )
        bits = min(supported, key=lambda candidate: abs(candidate - bits))

    if algorithm == QuantAlgorithm.GGUF and not gguf_type:
        gguf_type = DEFAULT_GGUF_TYPE[bits]

    group_size = choices.group_size or default_group_size(algorithm, bits)
    if group_size and model.tensors:
        group_size = group_size_that_fits(list(model.tensors), group_size)

    return QuantConfig(
        algorithm=algorithm,
        bits=bits,
        group_size=group_size,
        calibration_samples=choices.calibration_samples or calibration_samples(model.params_b),
        calibration_seq_len=choices.calibration_seq_len or 512,
        target_quant_label=model.llmfit_best_quant,
        gguf_type=gguf_type if algorithm == QuantAlgorithm.GGUF else "",
        context=choices.context or 4096,
    )


def build_rationale(
    hardware: HardwareProfile,
    model: ModelSnapshot,
    config: QuantConfig,
    goal: OptimizationGoal,
) -> list[str]:
    fit = model.feasibility
    lines = [
        f"Goal: {goal.value}",
        f"Best fit on this PC: {fit.quant_label}, {fit.fit_level} ({fit.run_mode}), "
        f"{fit.memory_required_gb:.2f} of {fit.memory_available_gb:.2f} GB free "
        f"({fit.utilization_pct:.0f}%)",
        f"Selected: {config.summary()}",
    ]
    if config.algorithm == QuantAlgorithm.GGUF:
        lines.append("GGUF runs on CPU, NVIDIA, AMD and Apple GPUs through llama.cpp or Ollama.")
    elif config.algorithm == QuantAlgorithm.AWQ:
        lines.append("AWQ keeps the weights that matter most accurate. Serve it with vLLM.")
    elif config.algorithm == QuantAlgorithm.GPTQ:
        lines.append("GPTQ corrects quantization error layer by layer. Serve it with vLLM.")
    elif config.algorithm == QuantAlgorithm.BNB:
        lines.append("bitsandbytes needs no calibration and loads straight into transformers.")
    if not fit.runnable:
        lines.append("This model is Too Tight for this PC even after quantization.")
    return lines


def bits_for_goal(model: ModelSnapshot, goal: OptimizationGoal) -> int:
    base = label_bits(model.llmfit_best_quant)
    if goal == OptimizationGoal.QUALITY:
        return min(8, max(base, 4))
    if goal in (OptimizationGoal.MEMORY, OptimizationGoal.SPEED):
        return min(base, 4)
    return base


def label_bits(label: str) -> int:
    for key, bits in LABEL_BITS.items():
        if key.upper() in label.upper():
            return bits
    return 4


def pick_algorithm(hardware: HardwareProfile, model: ModelSnapshot, bits: int) -> QuantAlgorithm:
    """AWQ/GPTQ for a Linux GPU box that can serve with vLLM, GGUF for everything else."""
    gpu_server = (
        hardware.has_cuda
        and hardware.os_name != "Windows"
        and str(model.feasibility.run_mode).lower() in ("gpu", "gpu_full")
    )
    if not gpu_server or not calibration_fits_in_ram(hardware, model):
        return QuantAlgorithm.GGUF

    if "gptq" in model.llmfit_best_quant.lower() or bits >= 8:
        return QuantAlgorithm.GPTQ
    return QuantAlgorithm.AWQ


def calibration_fits_in_ram(hardware: HardwareProfile, model: ModelSnapshot) -> bool:
    """AWQ and GPTQ load the whole original model into RAM before they start."""
    return model.original_gb <= hardware.available_ram_gb


def default_group_size(algorithm: QuantAlgorithm, bits: int) -> int:
    if algorithm in (QuantAlgorithm.BNB, QuantAlgorithm.GGUF):
        return 0
    return 128 if bits <= 4 else 64


def calibration_samples(params_b: float) -> int:
    if params_b >= 70:
        return 256
    if params_b >= 13:
        return 128
    return 64
