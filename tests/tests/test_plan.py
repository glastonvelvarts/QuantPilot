"""Tests for plan generation."""

from pathlib import Path

from ocq.plan import generate_plan
from ocq.types import (
    FeasibilityEstimate,
    HardwareProfile,
    ModelSnapshot,
    OptimizationGoal,
)


def test_generate_plan_has_rationale():
    hw = HardwareProfile(
        total_ram_gb=32,
        available_ram_gb=24,
        cpu_cores=8,
        cpu_name="CPU",
        has_gpu=True,
        backend="CUDA",
        unified_memory=False,
    )
    model = ModelSnapshot(
        model_id="meta-llama/Llama-3.1-8B-Instruct",
        provider="Meta",
        parameter_count="8B",
        params_b=8.0,
        context_length=8192,
        use_case="General",
        is_moe=False,
        llmfit_best_quant="Q4_K_M",
        feasibility=FeasibilityEstimate(
            quant_label="Q4_K_M",
            fit_level="Perfect",
            run_mode="gpu",
            score=90.0,
            estimated_tps=50.0,
            memory_required_gb=5.0,
            memory_available_gb=24.0,
            utilization_pct=21.0,
            runtime="llamacpp",
            estimate_confidence="estimated",
        ),
    )
    plan = generate_plan(
        "meta-llama/Llama-3.1-8B-Instruct",
        hw,
        model,
        OptimizationGoal.SPEED,
        Path("/tmp/out"),
    )
    assert plan.quant_config.bits == 4
    assert plan.llmfit_hypothesis_tps == 50.0
    assert any("tok/s" in line for line in plan.rationale)
