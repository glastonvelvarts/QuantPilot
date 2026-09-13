"""End-to-end optimize pipeline."""

from __future__ import annotations

from pathlib import Path

from ocq.llmfit_bridge import LlmfitBridge
from ocq.model_analyzer import resolve_model
from ocq.plan import generate_plan
from ocq.quantizers import get_quantizer
from ocq.types import OptimizationGoal, OptimizeResult, QuantAlgorithm
from ocq.validator import validate_artifact


def run_optimize(
    model_id: str,
    goal: OptimizationGoal,
    output_dir: Path,
    *,
    dry_run: bool = False,
    llmfit_bin: str | None = None,
    algorithm: QuantAlgorithm | None = None,
) -> OptimizeResult:
    bridge = LlmfitBridge(llmfit_bin)
    hardware = bridge.hardware_profile()
    snapshot = resolve_model(model_id, hardware, bridge)
    plan = generate_plan(model_id, hardware, snapshot, goal, output_dir, algorithm=algorithm)

    if dry_run:
        return OptimizeResult(plan=plan, dry_run=True)

    quantizer = get_quantizer(plan.quant_config.algorithm)
    quant_result, bench_result = quantizer.run(plan)
    validation = validate_artifact(
        plan, str(quant_result.artifact_path) if quant_result.artifact_path else None
    )

    return OptimizeResult(
        plan=plan,
        quantization=quant_result,
        validation=validation,
        benchmark=bench_result,
        dry_run=False,
    )
