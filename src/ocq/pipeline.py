"""End-to-end optimize pipeline."""

from __future__ import annotations

from pathlib import Path

from ocq import measurements
from ocq.hardware import detect_hardware
from ocq.model_analyzer import DEFAULT_CONTEXT, resolve_model
from ocq.plan import generate_plan
from ocq.quantizers import get_quantizer
from ocq.types import (
    BenchmarkResult,
    OptimizationGoal,
    OptimizationPlan,
    OptimizeResult,
    QuantAlgorithm,
    QuantChoices,
    QuantizationResult,
)
from ocq.validator import validate_artifact


def remember_speed(
    plan: OptimizationPlan, quantization: QuantizationResult, benchmark: BenchmarkResult
) -> None:
    """llama.cpp runs that used only the GPU or only the CPU feed later speed predictions."""
    if plan.quant_config.algorithm != QuantAlgorithm.GGUF or not benchmark.tokens_per_second:
        return
    if benchmark.device not in ("gpu", "cpu"):
        return
    read_gb = quantization.artifact_size_gb * plan.model.read_fraction
    measurements.record(benchmark.device, read_gb, benchmark.tokens_per_second)


def run_optimize(
    model_id: str,
    goal: OptimizationGoal,
    output_dir: Path,
    *,
    dry_run: bool = False,
    llmfit_bin: str | None = None,
    choices: QuantChoices | None = None,
) -> OptimizeResult:
    hardware = detect_hardware(llmfit_bin)
    context = (choices.context if choices else None) or DEFAULT_CONTEXT
    snapshot = resolve_model(model_id, hardware, llmfit_bin, context)
    plan = generate_plan(snapshot.model_id, hardware, snapshot, goal, output_dir, choices)

    if dry_run:
        return OptimizeResult(plan=plan, dry_run=True)

    quantization, benchmark = get_quantizer(plan.quant_config.algorithm).run(plan)
    validation = validate_artifact(plan, quantization, benchmark)
    remember_speed(plan, quantization, benchmark)

    return OptimizeResult(
        plan=plan,
        quantization=quantization,
        validation=validation,
        benchmark=benchmark,
        dry_run=False,
    )
