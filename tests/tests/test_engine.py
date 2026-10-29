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


def blackwell_hardware(name: str = "NVIDIA GeForce RTX 5090", capability=None) -> HardwareProfile:
    gpu = GpuProfile(name=name, vram_gb=32, backend="CUDA", compute_capability=capability)
    return replace(cuda_hardware(), gpus=(gpu,))


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
        QuantChoices(algorithm=QuantAlgorithm.AWQ, double_quant=True),
        QuantChoices(algorithm=QuantAlgorithm.GPTQ, double_quant=False),
        QuantChoices(gguf_type="Q4_K_M", double_quant=True),
        QuantChoices(algorithm=QuantAlgorithm.BNB, bits=8, double_quant=True),
        QuantChoices(algorithm=QuantAlgorithm.NVFP4, bits=8),
        QuantChoices(algorithm=QuantAlgorithm.NVFP4, group_size=32),
        QuantChoices(algorithm=QuantAlgorithm.NVFP4, double_quant=True),
        QuantChoices(algorithm=QuantAlgorithm.AWQ, fp4_activations=True),
        QuantChoices(gguf_type="Q4_K_M", fp4_activations=False),
    ],
)
def test_impossible_choices_are_rejected(choices):
    with pytest.raises(OcqError):
        decide_quant_config(cuda_hardware(), make_model(), OptimizationGoal.BALANCED, choices)


def test_bnb_4bit_double_quantizes_by_default():
    choices = QuantChoices(algorithm=QuantAlgorithm.BNB, bits=4)
    config = decide_quant_config(cuda_hardware(), make_model(), OptimizationGoal.BALANCED, choices)
    assert config.double_quant
    assert config.summary() == "BNB 4-bit NF4, double quantization"


def test_speed_goal_skips_double_quant():
    choices = QuantChoices(algorithm=QuantAlgorithm.BNB, bits=4)
    config = decide_quant_config(cuda_hardware(), make_model(), OptimizationGoal.SPEED, choices)
    assert not config.double_quant
    assert config.summary() == "BNB 4-bit NF4, single quantization"


def test_single_quant_can_be_asked_for():
    choices = QuantChoices(algorithm=QuantAlgorithm.BNB, bits=4, double_quant=False)
    config = decide_quant_config(cuda_hardware(), make_model(), OptimizationGoal.MEMORY, choices)
    assert not config.double_quant


@pytest.mark.parametrize("double_quant", [True, False])
def test_double_quant_flag_implies_bnb_4bit(double_quant):
    choices = QuantChoices(double_quant=double_quant)
    config = decide_quant_config(cpu_hardware(), make_model(), OptimizationGoal.QUALITY, choices)
    assert config.algorithm == QuantAlgorithm.BNB
    assert config.bits == 4
    assert config.double_quant == double_quant


def test_double_quant_is_off_where_it_does_not_apply():
    for choices in (
        QuantChoices(algorithm=QuantAlgorithm.BNB, bits=8),
        QuantChoices(algorithm=QuantAlgorithm.AWQ),
        QuantChoices(gguf_type="Q4_K_M"),
    ):
        config = decide_quant_config(
            cuda_hardware(), make_model(), OptimizationGoal.BALANCED, choices
        )
        assert not config.double_quant


@pytest.mark.parametrize("goal", [OptimizationGoal.BALANCED, OptimizationGoal.MEMORY])
def test_bnb_defaults_to_4bit(goal):
    model = make_model(llmfit_best_quant="Q8_0")
    choices = QuantChoices(algorithm=QuantAlgorithm.BNB)
    config = decide_quant_config(cuda_hardware(), model, goal, choices)
    assert config.bits == 4
    assert config.double_quant


def test_bnb_quality_goal_keeps_8bit():
    choices = QuantChoices(algorithm=QuantAlgorithm.BNB)
    config = decide_quant_config(cuda_hardware(), make_model(), OptimizationGoal.QUALITY, choices)
    assert config.bits == 8


def test_engine_picks_nvfp4_on_blackwell():
    config = decide_quant_config(blackwell_hardware(), make_model(), OptimizationGoal.BALANCED)
    assert config.algorithm == QuantAlgorithm.NVFP4
    assert config.fp4_activations
    assert config.group_size == 16
    assert config.summary() == "NVFP4 4-bit (weights and activations)"


def test_compute_capability_beats_the_gpu_name():
    blackwell = blackwell_hardware(name="Some Datacenter GPU", capability=10.0)
    hopper = blackwell_hardware(name="Some Datacenter GPU", capability=9.0)
    pick = lambda hw: decide_quant_config(hw, make_model(), OptimizationGoal.BALANCED).algorithm  # noqa: E731
    assert pick(blackwell) == QuantAlgorithm.NVFP4
    assert pick(hopper) == QuantAlgorithm.AWQ


def test_nvfp4_is_weights_only_without_fp4_tensor_cores():
    choices = QuantChoices(algorithm=QuantAlgorithm.NVFP4)
    config = decide_quant_config(cuda_hardware(), make_model(), OptimizationGoal.BALANCED, choices)
    assert not config.fp4_activations
    assert config.summary() == "NVFP4 4-bit (weights only)"


@pytest.mark.parametrize("fp4_activations", [True, False])
def test_fp4_flag_implies_nvfp4_and_wins(fp4_activations):
    choices = QuantChoices(fp4_activations=fp4_activations)
    config = decide_quant_config(cpu_hardware(), make_model(), OptimizationGoal.QUALITY, choices)
    assert config.algorithm == QuantAlgorithm.NVFP4
    assert config.bits == 4
    assert config.fp4_activations == fp4_activations
