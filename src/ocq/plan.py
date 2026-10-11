"""Plan generator: turns engine decisions into an OptimizationPlan and flags what could go wrong."""

from __future__ import annotations

import shutil
from pathlib import Path

from ocq import measurements
from ocq.engine import (
    build_rationale,
    calibration_fits_in_ram,
    decide_quant_config,
)
from ocq.local_model import NO_TOKENIZER, has_tokenizer
from ocq.model_analyzer import predict_for_config
from ocq.quantizers import get_quantizer
from ocq.types import (
    HardwareProfile,
    ModelSnapshot,
    OptimizationGoal,
    OptimizationPlan,
    QuantAlgorithm,
    QuantChoices,
    QuantConfig,
    QuantOption,
)


def generate_plan(
    model_id: str,
    hardware: HardwareProfile,
    model: ModelSnapshot,
    goal: OptimizationGoal,
    output_dir: Path,
    choices: QuantChoices | None = None,
) -> OptimizationPlan:
    config = decide_quant_config(hardware, model, goal, choices)
    predicted = predict_for_config(model, hardware, config)
    rationale = build_rationale(hardware, model, config, goal)
    rationale.append(
        f"{predicted.label} needs {predicted.memory_required_gb:.2f} GB: "
        f"{predicted.weights_gb:.2f} GB weights + {predicted.context_gb:.2f} GB for a "
        f"{config.context} token context ({predicted.fit_level}, {predicted.run_mode})"
    )
    rationale.append(f"Speed: {describe_speed(predicted, config)}")
    warnings: list[str] = []

    if not predicted.exact:
        warnings.append(
            "This repo's tensor shapes or config could not be read, so sizes are estimated "
            "from the parameter count."
        )
    if not model.feasibility.runnable:
        warnings.append("Model is Too Tight on this PC. It may not load after quantization.")
    asked_group = choices.group_size if choices else None
    if asked_group and asked_group != config.group_size:
        warnings.append(
            f"Group size {asked_group} does not divide this model's layer widths, "
            f"using {config.group_size}."
        )
    if model.weights_dir is not None and not has_tokenizer(model.weights_dir):
        warnings.append(NO_TOKENIZER.format(folder=model.weights_dir))
    warnings += quantization_warnings(hardware, model, config, output_dir)

    return OptimizationPlan(
        model_id=model_id,
        goal=goal,
        hardware=hardware,
        model=model,
        quant_config=config,
        output_dir=output_dir,
        rationale=rationale,
        llmfit_hypothesis_tps=model.feasibility.estimated_tps,
        warnings=warnings,
        predicted=predicted,
    )


def describe_speed(option: QuantOption, config: QuantConfig) -> str:
    if not option.runnable:
        return "does not fit"
    if option.speed_basis == measurements.MEASURED:
        return f"about {option.estimated_tps:.0f} tok/s, {option.speed_basis}"
    limit = f"at most {option.estimated_tps:.0f} tok/s (memory bandwidth limit)."
    if config.algorithm != QuantAlgorithm.GGUF:
        return f"{limit} This PC's measurements are llama.cpp runs and do not apply here."
    return f"{limit} No expected speed yet: it is learned from benchmark runs on this PC."


def quantization_warnings(
    hardware: HardwareProfile,
    model: ModelSnapshot,
    config: QuantConfig,
    output_dir: Path,
) -> list[str]:
    """Running a model and quantizing it are different jobs. This checks the second one."""
    warnings = []
    original_gb = model.original_gb
    quantized_gb = predict_for_config(model, hardware, config).weights_gb

    ready, detail = get_quantizer(config.algorithm).check(hardware)
    if not ready:
        warnings.append(detail)

    disk_needed = original_gb + quantized_gb
    if config.algorithm == QuantAlgorithm.GGUF:
        disk_needed += original_gb
    disk_free = free_disk_gb(output_dir)
    if disk_needed > disk_free:
        warnings.append(
            f"Quantizing needs {disk_needed:.1f} GB of disk, only {disk_free:.1f} GB free."
        )

    if config.algorithm in (QuantAlgorithm.AWQ, QuantAlgorithm.GPTQ):
        if not calibration_fits_in_ram(hardware, model):
            warnings.append(
                f"{config.algorithm.value.upper()} loads the full original model first: "
                f"{original_gb:.1f} GB of RAM needed at least, "
                f"{hardware.available_ram_gb:.1f} GB free. GGUF converts without loading it."
            )
        if config.algorithm == QuantAlgorithm.GPTQ and not hardware.has_cuda:
            warnings.append("No NVIDIA GPU: calibration will run on the CPU and be slow.")

    if config.algorithm == QuantAlgorithm.BNB:
        vram = hardware.free_vram_gb
        if hardware.has_cuda and quantized_gb > vram:
            warnings.append(
                f"bitsandbytes quantizes on the GPU: {quantized_gb:.1f} GB of VRAM needed, "
                f"{vram:.1f} GB free."
            )

    return warnings


def free_disk_gb(path: Path) -> float:
    folder = path.resolve()
    while not folder.exists():
        folder = folder.parent
    return shutil.disk_usage(folder).free / 1024**3
