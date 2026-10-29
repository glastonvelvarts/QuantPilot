"""Tests for model snapshots and fit."""

import pytest
from conftest import QWEN_CONFIG

from ocq import measurements
from ocq.model_analyzer import (
    best_option,
    build_snapshot,
    parse_params_from_name,
    quant_options,
    recommend_models,
)
from ocq.types import FitLevel, GpuProfile, HardwareProfile, RunMode


def make_hardware(
    ram_gb: float = 24, vram_gb: float | None = None, free_vram_gb: float | None = None
) -> HardwareProfile:
    gpus = ()
    if vram_gb:
        gpus = (
            GpuProfile(
                name="RTX 4060",
                vram_gb=vram_gb,
                backend="CUDA",
                free_vram_gb=free_vram_gb,
                measured_bandwidth_gbs=180.0,
            ),
        )
    return HardwareProfile(
        total_ram_gb=ram_gb,
        available_ram_gb=ram_gb,
        cpu_cores=8,
        cpu_name="CPU",
        has_gpu=bool(gpus),
        backend="CUDA" if gpus else "CPU",
        unified_memory=False,
        ram_bandwidth_gbs=89.6,
        gpus=gpus,
    )


def qwen_snapshot(tensors, hardware, context=4096):
    return build_snapshot("Qwen/Qwen2.5-1.5B-Instruct", {}, QWEN_CONFIG, tensors, hardware, context)


def options_by_label(model, hardware, context=4096) -> dict:
    return {option.label: option for option in quant_options(model, hardware, context)}


def test_exact_snapshot_from_tensor_shapes(qwen_tensors):
    snap = qwen_snapshot(qwen_tensors, make_hardware(vram_gb=8))
    assert snap.params_b == pytest.approx(1.5437, abs=0.0001)
    assert snap.num_layers == 28
    assert snap.num_kv_heads == 2
    assert snap.head_dim == 128
    assert snap.context_length == 32768


def test_memory_is_weights_plus_context(qwen_tensors):
    hardware = make_hardware(vram_gb=8)
    option = options_by_label(qwen_snapshot(qwen_tensors, hardware), hardware)["Q4_K_M"]
    assert option.exact
    assert option.weights_gb * 1024 == pytest.approx(934.69, abs=0.01)
    # 4096 tokens: 112 MiB of keys and values + 296.75 MiB of output scores
    assert option.context_gb * 1024 == pytest.approx(112 + 296.75, abs=0.01)
    assert option.memory_required_gb == option.weights_gb + option.context_gb


def test_bigger_context_needs_more_memory(qwen_tensors):
    hardware = make_hardware(vram_gb=8)
    model = qwen_snapshot(qwen_tensors, hardware)
    small = options_by_label(model, hardware, context=4096)["Q4_K_M"]
    large = options_by_label(model, hardware, context=32768)["Q4_K_M"]
    assert (large.context_gb - small.context_gb) * 1024 == pytest.approx(896 - 112, abs=0.01)


def test_fit_uses_free_vram_not_total(qwen_tensors):
    roomy = make_hardware(ram_gb=16, vram_gb=8, free_vram_gb=7.5)
    busy = make_hardware(ram_gb=16, vram_gb=8, free_vram_gb=1.0)
    assert options_by_label(qwen_snapshot(qwen_tensors, roomy), roomy)["Q4_K_M"].run_mode == (
        RunMode.GPU
    )
    spilled = options_by_label(qwen_snapshot(qwen_tensors, busy), busy)["Q4_K_M"]
    assert spilled.run_mode == RunMode.GPU_OFFLOAD
    assert spilled.fit_level == FitLevel.MARGINAL


def test_model_that_does_not_fit_is_unfit(qwen_tensors):
    tiny = make_hardware(ram_gb=0.5)
    options = quant_options(qwen_snapshot(qwen_tensors, tiny), tiny)
    assert not any(option.runnable for option in options)
    assert all(option.run_mode == RunMode.UNFIT for option in options)
    assert all(option.estimated_tps == 0 for option in options)


def test_best_option_is_highest_quality_that_fits_comfortably(qwen_tensors):
    hardware = make_hardware(vram_gb=8, free_vram_gb=2.0)
    best = best_option(quant_options(qwen_snapshot(qwen_tensors, hardware), hardware))
    assert best.label == "Q6_K"
    assert best.fit_level == FitLevel.GOOD


def test_without_tensor_shapes_sizes_are_marked_estimated():
    meta = {"safetensors": {"total": 7_000_000_000}, "pipeline_tag": "text-generation"}
    hardware = make_hardware()
    snap = build_snapshot("some-org/Some-Model-7B-Instruct", meta, {}, [], hardware)
    assert snap.catalog_source == "huggingface"
    assert snap.params_b == 7.0
    assert not any(option.exact for option in quant_options(snap, hardware))


def test_unknown_size_is_flagged():
    snap = build_snapshot("some-org/mystery-model", {}, {}, [], make_hardware())
    assert "unknown" in snap.parameter_count


def test_params_parsed_from_name():
    assert parse_params_from_name("Qwen/Qwen2.5-0.5B-Instruct") == 0.5
    assert parse_params_from_name("HuggingFaceTB/SmolLM2-135M-Instruct") == 0.135
    assert parse_params_from_name("org/no-size-here") is None


def test_speed_is_a_bandwidth_limit_until_this_pc_has_measurements(qwen_tensors):
    hardware = make_hardware(vram_gb=8)
    option = options_by_label(qwen_snapshot(qwen_tensors, hardware), hardware)["Q4_K_M"]
    assert option.speed_basis == measurements.UPPER_BOUND
    assert option.estimated_tps == pytest.approx(180.0 / option.weights_gb, abs=0.1)


def test_speed_comes_from_real_runs_once_recorded(qwen_tensors):
    measurements.record("gpu", 0.5, 150.0)
    measurements.record("gpu", 4.0, 30.0)
    hardware = make_hardware(vram_gb=8)
    option = options_by_label(qwen_snapshot(qwen_tensors, hardware), hardware)["Q4_K_M"]
    assert option.speed_basis == measurements.MEASURED
    assert 30.0 < option.estimated_tps < 150.0


def test_recommend_works_without_llmfit():
    models = recommend_models(make_hardware(ram_gb=8), limit=20, llmfit_bin="/no/such/llmfit")
    fits = {model.model_id: model.feasibility.runnable for model in models}
    assert fits["Qwen/Qwen2.5-0.5B-Instruct"] is True
    assert fits["Qwen/Qwen2.5-72B-Instruct"] is False


def test_feasibility_shows_bnb_single_and_double(qwen_tensors):
    hardware = make_hardware(vram_gb=8)
    options = options_by_label(qwen_snapshot(qwen_tensors, hardware), hardware)
    single, double = options["BNB 4-bit single"], options["BNB 4-bit double"]
    assert double.weights_gb < single.weights_gb
    # 0.373 bits per quantized weight, averaged with the 16-bit embeddings (15% of Qwen 1.5B)
    assert single.bits - double.bits == pytest.approx(0.316, abs=0.005)
    assert "BNB 8-bit" in options


def test_bnb_estimate_without_tensor_shapes_counts_scales():
    meta = {"safetensors": {"total": 7_000_000_000}}
    hardware = make_hardware()
    snap = build_snapshot("some-org/Some-Model-7B", meta, {}, [], hardware)
    options = options_by_label(snap, hardware)
    assert options["BNB 4-bit single"].bits == pytest.approx(4.5)
    assert options["BNB 4-bit double"].bits == pytest.approx(4.127, abs=0.001)


def test_nvfp4_row_only_with_an_nvidia_gpu(qwen_tensors):
    cpu_only = make_hardware()
    assert "NVFP4 4-bit" not in options_by_label(qwen_snapshot(qwen_tensors, cpu_only), cpu_only)

    nvidia = make_hardware(vram_gb=8)
    nvfp4 = options_by_label(qwen_snapshot(qwen_tensors, nvidia), nvidia)["NVFP4 4-bit"]
    assert nvfp4.row_kind == "nvfp4"
    assert nvfp4.exact
