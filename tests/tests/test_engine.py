"""Tests for optimization engine."""

from ocq.engine import decide_quant_config
from ocq.types import (
    FeasibilityEstimate,
    GpuProfile,
    HardwareProfile,
    ModelSnapshot,
    OptimizationGoal,
    QuantAlgorithm,
)


def _model(**overrides) -> ModelSnapshot:
    defaults = dict(
        model_id="Qwen/Qwen2.5-7B-Instruct",
        provider="Qwen",
        parameter_count="7B",
        params_b=7.0,
        context_length=32768,
        use_case="General",
        is_moe=False,
        llmfit_best_quant="Q4_K_M",
        feasibility=FeasibilityEstimate(
            quant_label="Q4_K_M",
            fit_level="Good",
            run_mode="gpu",
            score=85.0,
            estimated_tps=42.0,
            memory_required_gb=5.0,
            memory_available_gb=12.0,
            utilization_pct=42.0,
            runtime="llamacpp",
            estimate_confidence="estimated",
        ),
    )
    defaults.update(overrides)
    return ModelSnapshot(**defaults)


def _cuda_hw() -> HardwareProfile:
    return HardwareProfile(
        total_ram_gb=32,
        available_ram_gb=24,
        cpu_cores=8,
        cpu_name="Test CPU",
        has_gpu=True,
        backend="CUDA",
        unified_memory=False,
        gpus=(GpuProfile(name="RTX 4090", vram_gb=24, backend="CUDA"),),
    )


def test_engine_picks_gptq_on_cuda():
    config = decide_quant_config(_cuda_hw(), _model(), OptimizationGoal.BALANCED)
    assert config.algorithm in (QuantAlgorithm.GPTQ, QuantAlgorithm.AWQ)
    assert config.bits == 4


def test_engine_picks_bnb_without_gpu():
    hw = HardwareProfile(
        total_ram_gb=16,
        available_ram_gb=12,
        cpu_cores=8,
        cpu_name="Test CPU",
        has_gpu=False,
        backend="CPU (x86)",
        unified_memory=False,
    )
    config = decide_quant_config(hw, _model(), OptimizationGoal.MEMORY)
    assert config.algorithm == QuantAlgorithm.BNB
