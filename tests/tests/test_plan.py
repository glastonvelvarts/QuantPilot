"""Tests for plan generation."""

from pathlib import Path

from ocq.plan import generate_plan
from ocq.types import (
    FeasibilityEstimate,
    HardwareProfile,
    ModelSnapshot,
    OptimizationGoal,
    QuantAlgorithm,
    QuantChoices,
)


def make_hardware() -> HardwareProfile:
    return HardwareProfile(
        total_ram_gb=32,
        available_ram_gb=24,
        cpu_cores=8,
        cpu_name="CPU",
        has_gpu=True,
        backend="CUDA",
        unified_memory=False,
    )


def make_model() -> ModelSnapshot:
    return ModelSnapshot(
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


def test_generate_plan_has_rationale():
    plan = generate_plan(
        "meta-llama/Llama-3.1-8B-Instruct",
        make_hardware(),
        make_model(),
        OptimizationGoal.SPEED,
        Path("/tmp/out"),
    )
    assert plan.quant_config.bits == 4
    assert plan.llmfit_hypothesis_tps == 50.0
    assert any("tok/s" in line for line in plan.rationale)


def test_bnb_plan_explains_double_quant_saving():
    def plan_for(double_quant):
        choices = QuantChoices(algorithm=QuantAlgorithm.BNB, bits=4, double_quant=double_quant)
        return generate_plan(
            "meta-llama/Llama-3.1-8B-Instruct",
            make_hardware(),
            make_model(),
            OptimizationGoal.BALANCED,
            Path("/tmp/out"),
            choices,
        )

    double, single = plan_for(True), plan_for(False)
    assert double.predicted.weights_gb < single.predicted.weights_gb
    # 8B parameters, 0.373 bits each
    assert any("saving 356 MB" in line for line in double.rationale)
    assert any("--double-quant would save 356 MB" in line for line in single.rationale)


def test_nvfp4_plan_warns_when_this_pc_cannot_run_it_fast():
    from dataclasses import replace

    from ocq.types import GpuProfile

    def warnings_on(hardware):
        choices = QuantChoices(algorithm=QuantAlgorithm.NVFP4)
        plan = generate_plan(
            "meta-llama/Llama-3.1-8B-Instruct",
            hardware,
            make_model(),
            OptimizationGoal.BALANCED,
            Path("/tmp/out"),
            choices,
        )
        return " ".join(plan.warnings)

    def with_gpu(name):
        return replace(make_hardware(), gpus=(GpuProfile(name=name, vram_gb=24, backend="CUDA"),))

    assert "No NVIDIA GPU" in warnings_on(replace(make_hardware(), has_gpu=False, backend="CPU"))
    assert "Marlin" in warnings_on(with_gpu("NVIDIA GeForce RTX 4090"))
    assert "FP4 tensor cores" not in warnings_on(with_gpu("NVIDIA GeForce RTX 5090"))
