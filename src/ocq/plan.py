"""Plan generator: turns engine decisions into an OptimizationPlan and flags what could go wrong."""

from __future__ import annotations

import shutil
from dataclasses import replace
from pathlib import Path

from ocq import measurements
from ocq.engine import (
    build_rationale,
    calibration_fits_in_ram,
    decide_quant_config,
)
from ocq.local_model import NO_TOKENIZER, has_tokenizer
from ocq.model_analyzer import predict_for_config, runtime_of
from ocq.model_size import kv_value_bytes
from ocq.quantizers import get_quantizer
from ocq.types import (
    LAYER_SPLIT,
    LLAMA_CPP,
    TENSOR_PARALLEL,
    VLLM,
    HardwareProfile,
    ModelSnapshot,
    OcqError,
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
    if model.is_quantized:
        raise OcqError(already_quantized(model))
    config = decide_quant_config(hardware, model, goal, choices)
    predicted = predict_for_config(model, hardware, config)
    rationale = build_rationale(hardware, model, config, goal)
    if predicted.context_known:
        rationale.append(
            f"{predicted.label} needs {predicted.memory_required_gb:.2f} GB: "
            f"{predicted.weights_gb:.2f} GB weights + {predicted.context_gb:.2f} GB for a "
            f"{config.context} token context ({predicted.fit_level}, {predicted.run_mode})"
        )
    else:
        rationale.append(
            f"{predicted.label} needs {predicted.weights_gb:.2f} GB for the weights, plus an "
            f"unknown amount for a {config.context} token context "
            f"({predicted.fit_level}, {predicted.run_mode})"
        )
    gpus = gpu_setup(predicted, hardware)
    if gpus:
        rationale.append(gpus)
    if predicted.experts_in_ram:
        rationale.append(experts_note(model, predicted))
    if config.kv_cache != "f16":
        rationale.append(kv_cache_note(config, hardware))
    picked = not (choices and choices.algorithm)
    if hardware.os_name == "Windows" and hardware.has_cuda and picked:
        if config.algorithm == QuantAlgorithm.GGUF:
            rationale.append(
                "Windows: GGUF, because vLLM, which serves AWQ, GPTQ and NVFP4, does not run "
                "natively on Windows. In WSL2 QuantPilot sees Linux and picks those as there."
            )
    rationale.append(f"Speed: {describe_speed(predicted, config)}")
    if config.algorithm == QuantAlgorithm.BNB and config.bits == 4:
        rationale.append(double_quant_note(model, hardware, config, predicted))
    warnings: list[str] = []

    if not model.tensors:
        warnings.append(
            "This repo's tensor shapes could not be read, so sizes are estimated "
            "from the parameter count."
        )
    if not predicted.context_known:
        warnings.append(
            "config.json has no layer or head counts and the tensor names show no separate "
            "k_proj/v_proj, so the memory for the context is unknown and the fit counts the "
            "weights only."
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


def already_quantized(model: ModelSnapshot) -> str:
    """Quantizing quantized weights again stacks the rounding errors. Point to the original."""
    message = (
        f"{model.model_id} is already quantized ({model.facts.quantization.value}). "
        "Quantizing it again would stack the rounding errors"
    )
    base = (model.raw.get("cardData") or {}).get("base_model")
    if isinstance(base, list):
        base = base[0] if len(base) == 1 else None
    if base:
        return f"{message}; quantize the original instead: ocq optimize {base}"
    return f"{message}. Quantize the original model instead. (ocq feasibility shows if it fits.)"


def experts_note(model: ModelSnapshot, option: QuantOption) -> str:
    """Mixture of experts in llama.cpp with the experts of some layers kept in RAM."""
    layers = option.experts_in_ram
    return (
        f"Experts: the experts of the first {layers} of {model.num_layers} layers stay in RAM, "
        "attention, the context and everything else in VRAM. Run llama.cpp with "
        f"--n-cpu-moe {layers}. A token uses only {model.experts_per_token} of "
        f"{model.experts} experts, so far less is read from RAM than with whole layers there."
    )


def kv_cache_note(config: QuantConfig, hardware: HardwareProfile) -> str:
    """What --kv-cache means for the runtime that serves this backend, and the flags to pass."""
    cache = config.kv_cache
    runtime = runtime_of(
        "gguf" if config.algorithm == QuantAlgorithm.GGUF else config.algorithm.value
    )
    bits = kv_value_bytes(cache, runtime) * 8
    if runtime == LLAMA_CPP:
        loss = "almost no quality loss" if cache == "q8_0" else "some quality loss on long contexts"
        return (
            f"Context cache: {cache}, {bits:.1f} bits per value instead of 16 ({loss}). Run "
            f"llama.cpp with -ctk {cache} -ctv {cache}; it turns on flash attention for it."
        )
    if runtime == VLLM:
        if not hardware.vllm_fp8_kv_cache:
            gpu = hardware.primary_gpu
            capability = gpu.compute_capability if gpu else None
            this = f"compute capability {capability}" if capability else "no known capability"
            return (
                f"Context cache: stays at 16 bits. vLLM's fp8 cache needs compute capability "
                f"8.9 or newer (RTX 40xx, H100, RTX 50xx); this PC has {this}. "
                "GGUF (llama.cpp) can use q8_0 here."
            )
        extra = " (vLLM has no 4-bit cache)" if cache == "q4_0" else ""
        return f"Context cache: fp8, 8 bits per value{extra}. Serve with --kv-cache-dtype fp8."
    return (
        "Context cache: transformers keeps it at 16 bits; --kv-cache applies to llama.cpp and vLLM."
    )


def gpu_setup(option: QuantOption, hardware: HardwareProfile) -> str:
    """How the model uses a PC with several GPUs. Empty with one GPU or none."""
    cards = hardware.cards
    if len(cards) < 2 or hardware.unified_memory or not option.gpu_count:
        return ""
    names = " + ".join(option.gpu_names)
    if option.gpu_split == TENSOR_PARALLEL:
        return (
            f"GPUs: tensor parallel over {option.gpu_count} GPUs ({names}), each holding an "
            f"equal slice. Serve with vLLM --tensor-parallel-size {option.gpu_count}."
        )
    if option.gpu_split == LAYER_SPLIT:
        return (
            f"GPUs: layers split over {option.gpu_count} GPUs ({names}). llama.cpp and "
            "transformers spread a model over every GPU by themselves."
        )
    if option.row_kind in (QuantAlgorithm.AWQ.value, QuantAlgorithm.GPTQ.value, "nvfp4"):
        return (
            f"GPUs: runs on the {names} alone. Tensor parallel gives every GPU an equal slice, "
            "so with these GPUs or this model's head count more GPUs would not hold more."
        )
    return (
        f"GPUs: runs on the {names} alone. It fits there, and layers on the slower GPUs would "
        "lower the speed. llama.cpp spreads over every GPU unless you pick one, for example "
        "with CUDA_VISIBLE_DEVICES."
    )


def double_quant_note(
    model: ModelSnapshot,
    hardware: HardwareProfile,
    config: QuantConfig,
    predicted: QuantOption,
) -> str:
    """How much memory double quantization saves on this model, or would save."""
    other = predict_for_config(
        model, hardware, replace(config, double_quant=not config.double_quant)
    )
    saved = format_size(abs(other.weights_gb - predicted.weights_gb))
    if config.double_quant:
        return (
            "Double quantization: the 32-bit scale of every 64 weights is stored in 8 bits, "
            f"saving {saved} (about 0.37 bits per weight) for a slightly slower unpack. "
            "--single-quant turns it off."
        )
    return (
        "Single quantization: every 64 weights keep a 32-bit scale. "
        f"--double-quant would save {saved}."
    )


def format_size(gb: float) -> str:
    return f"{gb:.2f} GB" if gb >= 1 else f"{gb * 1024:.0f} MB"


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

    if original_gb is None:
        warnings.append(
            "The size of the original weight files could not be read, so disk space and RAM "
            "for quantizing were not checked."
        )
    else:
        disk_needed = original_gb + quantized_gb
        if config.algorithm == QuantAlgorithm.GGUF:
            disk_needed += original_gb
        disk_free = free_disk_gb(output_dir)
        if disk_needed > disk_free:
            warnings.append(
                f"Quantizing needs {disk_needed:.1f} GB of disk, only {disk_free:.1f} GB free."
            )

    if config.algorithm == QuantAlgorithm.NVFP4:
        warnings += nvfp4_warnings(hardware, config)

    vllm_formats = (QuantAlgorithm.AWQ, QuantAlgorithm.GPTQ, QuantAlgorithm.NVFP4)
    if hardware.os_name == "Windows" and config.algorithm in vllm_formats:
        warnings.append(
            f"{config.algorithm.value.upper()} is served by vLLM, which does not run natively on "
            "Windows: serve it in WSL2 or on Linux, or quantize to GGUF (-b gguf) to run it here."
        )

    if config.algorithm in (QuantAlgorithm.AWQ, QuantAlgorithm.GPTQ, QuantAlgorithm.NVFP4):
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


def nvfp4_warnings(hardware: HardwareProfile, config: QuantConfig) -> list[str]:
    """NVFP4 runs best on Blackwell. Say what happens on the GPU this PC has."""
    gpu = hardware.primary_gpu
    if not hardware.has_cuda or gpu is None:
        return [
            "No NVIDIA GPU on this PC. NVFP4 is served by vLLM on NVIDIA GPUs, so serve it on "
            "another machine. Quantizing and the benchmark run on the CPU here, slowly."
        ]
    if gpu.fp4_tensor_cores:
        return []
    if config.fp4_activations:
        return [
            f"{gpu.name} has no FP4 tensor cores (Blackwell has them). FP4 activations only "
            "run fast on a Blackwell GPU: quantize here, serve there."
        ]
    return [
        f"{gpu.name} has no FP4 tensor cores (Blackwell has them). vLLM runs NVFP4 weights "
        "through its Marlin kernel, about as fast as AWQ, which is better supported on this GPU."
    ]


def free_disk_gb(path: Path) -> float:
    folder = path.resolve()
    while not folder.exists():
        folder = folder.parent
    return shutil.disk_usage(folder).free / 1024**3
