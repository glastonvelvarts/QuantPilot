"""Plan generator — turns engine decisions into an executable OptimizationPlan."""

from __future__ import annotations

from pathlib import Path

from ocq.engine import build_rationale, decide_quant_config
from ocq.types import (
    HardwareProfile,
    ModelSnapshot,
    OptimizationGoal,
    OptimizationPlan,
)


def generate_plan(
    model_id: str,
    hardware: HardwareProfile,
    model: ModelSnapshot,
    goal: OptimizationGoal,
    output_dir: Path,
) -> OptimizationPlan:
    config = decide_quant_config(hardware, model, goal)
    rationale = build_rationale(hardware, model, config, goal)
    warnings: list[str] = []

    if model.catalog_source != "llmfit":
        warnings.append(
            "Model not in llmfit catalog - fit/speed numbers are rough HF estimates. "
            "Run: llmfit update  (or use a catalog model for better scores)"
        )
    if not model.feasibility.runnable:
        warnings.append("Model is Too Tight on current hardware per llmfit.")
    if model.feasibility.estimate_confidence == "estimated":
        warnings.append("llmfit speed estimate is formula-only - benchmark after quant.")

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
    )
