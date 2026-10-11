"""Tests for optimization engine."""

from dataclasses import replace

import pytest

from ocq.engine import decide_quant_config
from ocq.types import (
    FeasibilityEstimate,
    GpuProfile,
    HardwareProfile,
    ModelSnapshot,
    OcqError,
    OptimizationGoal,
    QuantAlgorithm,
    QuantChoices,
)


def make_model(**overrides) -> ModelSnapshot:
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


def cuda_hardware(os_name: str = "Linux") -> HardwareProfile:
    return HardwareProfile(
        total_ram_gb=32,
        available_ram_gb=24,
        cpu_cores=8,
        cpu_name="Test CPU",
        has_gpu=True,
        backend="CUDA",
        unified_memory=False,
        gpus=(GpuProfile(name="RTX 4090", vram_gb=24, backend="CUDA"),),
        os_name=os_name,
    )


def cpu_hardware() -> HardwareProfile:
    return HardwareProfile(
        total_ram_gb=16,
        available_ram_gb=12,
        cpu_cores=8,
        cpu_name="Test CPU",
        has_gpu=False,
        backend="CPU (x86)",
        unified_memory=False,
    )


def test_engine_picks_awq_on_linux_cuda():
    config = decide_quant_config(cuda_hardware(), make_model(), OptimizationGoal.BALANCED)
    assert config.algorithm == QuantAlgorithm.AWQ
    assert config.bits == 4
    assert config.group_size == 128


def test_engine_picks_gguf_on_windows_cuda():
    config = decide_quant_config(cuda_hardware("Windows"), make_model(), OptimizationGoal.BALANCED)
    assert config.algorithm == QuantAlgorithm.GGUF
    assert config.gguf_type == "Q4_K_M"


def test_engine_avoids_calibration_that_does_not_fit_in_ram():
    hardware = replace(cuda_hardware(), available_ram_gb=8)
    config = decide_quant_config(hardware, make_model(), OptimizationGoal.BALANCED)
    assert config.algorithm == QuantAlgorithm.GGUF


def test_engine_picks_awq_when_best_fit_is_between_4_and_8_bits():
    model = make_model(llmfit_best_quant="Q6_K")
    config = decide_quant_config(cuda_hardware(), model, OptimizationGoal.BALANCED)
    assert config.algorithm == QuantAlgorithm.AWQ
    assert config.bits == 4


def test_engine_picks_gguf_without_gpu():
    config = decide_quant_config(cpu_hardware(), make_model(), OptimizationGoal.MEMORY)
    assert config.algorithm == QuantAlgorithm.GGUF
    assert config.bits == 4


def test_quality_goal_keeps_more_bits():
    model = make_model(llmfit_best_quant="Q8_0")
    assert decide_quant_config(cpu_hardware(), model, OptimizationGoal.QUALITY).bits == 8
    assert decide_quant_config(cpu_hardware(), model, OptimizationGoal.SPEED).bits == 4


def test_user_choices_win():
    choices = QuantChoices(
        algorithm=QuantAlgorithm.GPTQ,
        bits=8,
        group_size=32,
        calibration_samples=16,
        calibration_seq_len=256,
    )
    config = decide_quant_config(cpu_hardware(), make_model(), OptimizationGoal.SPEED, choices)
    assert config.algorithm == QuantAlgorithm.GPTQ
    assert config.bits == 8
    assert config.group_size == 32
    assert config.calibration_samples == 16
    assert config.calibration_seq_len == 256


def test_gguf_type_sets_backend_and_bits():
    choices = QuantChoices(gguf_type="q5_k_s")
    config = decide_quant_config(cuda_hardware(), make_model(), OptimizationGoal.BALANCED, choices)
    assert config.algorithm == QuantAlgorithm.GGUF
    assert config.gguf_type == "Q5_K_S"
    assert config.bits == 5


def test_chosen_backend_gets_a_bit_width_it_supports():
    model = make_model(llmfit_best_quant="Q6_K")
    choices = QuantChoices(algorithm=QuantAlgorithm.AWQ)
    config = decide_quant_config(cuda_hardware(), model, OptimizationGoal.BALANCED, choices)
    assert config.bits == 4


@pytest.mark.parametrize(
    "choices",
    [
        QuantChoices(algorithm=QuantAlgorithm.AWQ, bits=8),
        QuantChoices(algorithm=QuantAlgorithm.BNB, bits=3),
        QuantChoices(gguf_type="Q9_X"),
        QuantChoices(algorithm=QuantAlgorithm.AWQ, gguf_type="Q4_K_M"),
        QuantChoices(gguf_type="Q4_K_M", bits=8),
    ],
)
def test_impossible_choices_are_rejected(choices):
    with pytest.raises(OcqError):
        decide_quant_config(cuda_hardware(), make_model(), OptimizationGoal.BALANCED, choices)
