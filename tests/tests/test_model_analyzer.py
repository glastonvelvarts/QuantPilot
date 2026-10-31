"""Tests for model snapshots and fit."""

import pytest
from conftest import (
    OCQ_NVFP4,
    QWEN,
    QWEN3_8B_NVFP4,
    QWEN_72B,
    QWEN_AWQ,
    QWEN_BNB,
    QWEN_CONFIG,
    QWEN_GPTQ,
    QWEN_W8A8,
    RTX_3060,
    RTX_3090,
    RTX_3090_BUSY,
    SMOLVLM,
    nvidia_pc,
    real_model,
)

from ocq import measurements
from ocq.model_analyzer import (
    best_option,
    build_snapshot,
    parse_params_from_name,
    quant_options,
    recommend_models,
)
from ocq.types import (
    ONE_GPU,
    TENSOR_PARALLEL,
    FactKind,
    FitLevel,
    GpuProfile,
    HardwareProfile,
    OcqError,
    RunMode,
)


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


def real_snapshot(repo, hardware=None, context=4096, **meta_changes):
    meta, config, tensors, dtypes = real_model(repo)
    return build_snapshot(repo, meta | meta_changes, config, tensors, hardware, context, dtypes)


# Bug 1: an already-quantized repo was read as a 409M model of 5.32 GB, marked exact.


def test_gptq_repo_is_read_as_quantized_with_its_real_count():
    snap = real_snapshot(QWEN_GPTQ)
    assert snap.is_quantized
    assert snap.facts.quantization.value == "GPTQ 4-bit"
    assert snap.facts.quantization.kind == FactKind.CROSS_CHECKED
    assert snap.facts.parameters.value == 1_543_714_304
    assert snap.facts.parameters.kind == FactKind.CROSS_CHECKED
    assert snap.params_b == pytest.approx(1.5437, abs=0.0001)
    assert snap.original_gb == pytest.approx(1.07, abs=0.005)


def test_quantized_repo_is_shown_as_stored_not_re_quantized():
    hardware = make_hardware(vram_gb=8)
    snap = real_snapshot(QWEN_GPTQ, hardware)
    (stored,) = quant_options(snap, hardware)
    assert stored.label == "GPTQ 4-bit (as stored)"
    assert stored.row_kind == "stored"
    assert stored.weights_gb == pytest.approx(snap.original_gb)
    assert stored.context_gb > 0
    assert snap.feasibility.quant_label == stored.label


def test_header_count_that_disagrees_with_huggingface_is_an_estimate():
    snap = real_snapshot(QWEN_GPTQ, safetensors={"total": 1_000_000_000})
    assert snap.facts.parameters.value == 1_543_714_304
    assert snap.facts.parameters.kind == FactKind.ESTIMATE
    assert "HF API says 1,000,000,000" in snap.facts.parameters.note


def test_awq_repo_counts_the_lm_head_it_stores():
    # The config ties lm_head to the embeddings, but this repo saves a copy anyway.
    snap = real_snapshot(QWEN_AWQ)
    assert snap.facts.quantization.value == "AWQ 4-bit"
    assert snap.facts.parameters.value == 1_777_088_000
    assert snap.facts.parameters.kind == FactKind.CROSS_CHECKED
    assert "lm_head stored" in snap.facts.architecture.note


def test_plain_repo_facts_match_the_hub():
    snap = real_snapshot(QWEN)
    facts = snap.facts
    assert not snap.is_quantized
    assert facts.quantization.value == "none"
    assert facts.parameters.value == 1_543_714_304
    assert facts.parameters.kind == FactKind.CROSS_CHECKED
    assert "338 tensors" in facts.parameters.note
    assert facts.precision.value == "BF16"
    assert facts.stored_bytes.value == 3_087_428_608
    assert facts.architecture.value == "qwen2"
    assert "tied embeddings" in facts.architecture.note
    assert facts.context_length.value == 32768


# Bug 2: layers and heads under text_config left the architecture unknown and the KV cache at 0.


def test_vision_model_reads_its_language_model_from_text_config():
    hardware = make_hardware(vram_gb=8)
    snap = real_snapshot(SMOLVLM, hardware)
    assert snap.architecture_known
    assert (snap.num_layers, snap.num_attention_heads, snap.num_kv_heads) == (30, 9, 3)
    assert snap.head_dim == 64
    assert snap.context_length == 8192  # only in text_config, None at the top level
    assert snap.facts.architecture.source.startswith("config.json text_config")
    assert options_by_label(snap, hardware)["Q4_K_M"].context_gb > 0


@pytest.mark.parametrize("key", ["text_config", "llm_config", "language_config"])
def test_nested_config_gives_the_same_memory_as_a_flat_one(qwen_tensors, key):
    hardware = make_hardware(vram_gb=8)
    flat = qwen_snapshot(qwen_tensors, hardware)
    nested_config = {"architectures": ["SomeVisionModel"], key: QWEN_CONFIG}
    nested = build_snapshot("org/vlm", {}, nested_config, qwen_tensors, hardware)
    flat_q4 = options_by_label(flat, hardware)["Q4_K_M"]
    nested_q4 = options_by_label(nested, hardware)["Q4_K_M"]
    assert nested_q4.context_gb == pytest.approx(flat_q4.context_gb)
    assert nested_q4.context_gb * 1024 == pytest.approx(112 + 296.75, abs=0.01)
    assert nested_q4.exact


# Bug 4: a missing count became 7B, and a size in the name looked like a real count.


def test_no_count_means_no_fit_verdict():
    hardware = make_hardware()
    snap = build_snapshot("some-org/Cool-70B-Chat", {}, {}, [], hardware)
    assert snap.params_b == 0
    assert snap.parameter_count == "unknown (name says 70.00B)"
    assert not snap.facts.parameters.known
    assert snap.facts.parameters.note == "name says 70.00B, not used"
    assert not snap.feasibility.runnable
    with pytest.raises(OcqError, match="parameter count .* is unknown"):
        quant_options(snap, hardware)


def test_without_hardware_only_the_facts_are_read():
    snap = real_snapshot(QWEN)
    assert snap.facts.parameters.value == 1_543_714_304
    assert snap.feasibility.quant_label == "unknown"


# Where the HF API count is wrong, the base model's count is the second source.


def test_bnb_repo_count_is_checked_against_its_base_model():
    facts = real_snapshot(QWEN_BNB).facts
    assert facts.quantization.value == "BNB 4-bit"
    assert facts.parameters.value == 1_543_714_304
    assert facts.parameters.kind == FactKind.CROSS_CHECKED
    assert facts.parameters.source == "safetensors headers + base model"
    assert "HF API says 1,584,858,476" in facts.parameters.note
    assert facts.precision.value == "4-bit in U8"


def test_nvidia_nvfp4_repo_is_unpacked_and_checked_against_its_base_model():
    snap = real_snapshot(QWEN3_8B_NVFP4)
    assert snap.facts.quantization.value == "ModelOpt NVFP4"
    assert snap.facts.parameters.value == 8_190_735_360
    assert snap.facts.parameters.kind == FactKind.CROSS_CHECKED
    assert "matches base model Qwen/Qwen3-8B" in snap.facts.parameters.note
    assert snap.original_gb == pytest.approx(5.96, abs=0.005)


def test_without_a_base_model_a_disagreeing_count_stays_an_estimate():
    facts = real_snapshot(QWEN_W8A8).facts
    assert facts.quantization.value == "compressed-tensors int-quantized 8-bit"
    assert facts.parameters.value == 1_777_088_000
    assert facts.parameters.kind == FactKind.ESTIMATE
    assert "HF API says 1,777,733,120" in facts.parameters.note


def test_a_stored_lm_head_copy_is_allowed_for_when_comparing_with_the_base_model():
    facts = real_snapshot(
        QWEN_W8A8, base_model_parameters=("Qwen/Qwen2.5-1.5B-Instruct", 1_543_714_304)
    ).facts
    assert facts.parameters.kind == FactKind.CROSS_CHECKED
    assert "plus a stored lm_head copy of 233,373,696" in facts.parameters.note


def test_layer_count_is_checked_against_the_tensor_names():
    # SmolVLM's vision encoder has 12 layers of its own; only the 30 language layers count.
    facts = real_snapshot(SMOLVLM).facts
    assert facts.architecture.kind == FactKind.CROSS_CHECKED
    assert facts.architecture.source == "config.json text_config + tensor names"


def test_layer_count_that_disagrees_with_the_tensor_names_is_an_estimate(qwen_tensors):
    missing_last_layer = [t for t in qwen_tensors if ".layers.27." not in t[0]]
    snap = build_snapshot("org/model", {}, QWEN_CONFIG, missing_last_layer, None)
    assert snap.facts.architecture.kind == FactKind.ESTIMATE
    assert "tensor names show 27 layers" in snap.facts.architecture.note


# Context memory: config.json and the K and V projection shapes are two sources.

ODD_CONFIG = {"model_type": "odd", "architectures": ["OddForCausalLM"]}


def fused_attention(tensors):
    """The same model with q, k and v in one tensor, as Phi-3 and GPT-NeoX store them."""
    fused = []
    for name, shape in tensors:
        if ".k_proj." in name or ".v_proj." in name:
            continue
        fused.append((name.replace(".q_proj.", ".qkv_proj."), shape))
    return fused


@pytest.mark.parametrize("repo", [QWEN, QWEN3_8B_NVFP4, QWEN_W8A8, SMOLVLM, OCQ_NVFP4])
def test_context_memory_from_config_matches_the_tensor_shapes(repo):
    meta, config, tensors, dtypes = real_model(repo)
    kv = build_snapshot(repo, meta, config, tensors, None, dtypes=dtypes).facts.kv_cache
    assert kv.kind == FactKind.CROSS_CHECKED
    # And without config.json, the shapes alone give the same number.
    alone = build_snapshot(repo, meta, {}, tensors, None, dtypes=dtypes).facts.kv_cache
    assert alone.value == kv.value
    assert alone.source == "tensor shapes"


def test_tensor_shapes_give_the_context_memory_when_config_has_no_counts(qwen_tensors):
    hardware = make_hardware(vram_gb=8)
    snap = build_snapshot("org/odd", {}, ODD_CONFIG, qwen_tensors, hardware)
    assert snap.facts.kv_cache.value == 28 * 2 * 256 * 2  # layers x (K + V widths) x 2 bytes
    assert snap.facts.kv_cache.kind == FactKind.ESTIMATE
    q4 = options_by_label(snap, hardware)["Q4_K_M"]
    # Same as with Qwen's config: 112 MiB of keys and values + 296.75 MiB of output scores.
    assert q4.context_known
    assert q4.context_gb * 1024 == pytest.approx(112 + 296.75, abs=0.01)
    assert q4.fit_level == FitLevel.PERFECT


def test_config_and_shapes_that_disagree_make_an_estimate(qwen_tensors):
    wrong_kv_heads = QWEN_CONFIG | {"num_key_value_heads": 4}  # the shapes show 2
    kv = build_snapshot("org/model", {}, wrong_kv_heads, qwen_tensors, None).facts.kv_cache
    assert kv.kind == FactKind.ESTIMATE
    assert "tensor shapes give 28,672 bytes per token" in kv.note


def test_unknown_attention_leaves_context_memory_unknown(qwen_tensors):
    hardware = make_hardware(vram_gb=8)
    snap = build_snapshot("org/odd", {}, ODD_CONFIG, fused_attention(qwen_tensors), hardware)
    options = quant_options(snap, hardware)
    assert not snap.facts.kv_cache.known
    assert not any(option.context_known for option in options)
    # Without the context, the headroom of Perfect or Good cannot be promised.
    assert {option.fit_level for option in options} == {FitLevel.MARGINAL}


def test_known_architecture_keeps_its_context_memory(qwen_tensors):
    hardware = make_hardware(vram_gb=8)
    option = options_by_label(qwen_snapshot(qwen_tensors, hardware), hardware)["Q4_K_M"]
    assert option.context_known
    assert option.fit_level == FitLevel.PERFECT


# Several GPUs: Qwen2.5-72B at 4 bits needs about 46 GB, one RTX 3090 has 24.


def test_72b_runs_fully_on_two_3090s_split_by_layers():
    one, two = nvidia_pc(RTX_3090), nvidia_pc(RTX_3090, RTX_3090_BUSY)
    on_one = options_by_label(real_snapshot(QWEN_72B, one), one)["Q4_K_M"]
    on_two = options_by_label(real_snapshot(QWEN_72B, two), two)["Q4_K_M"]
    assert on_one.run_mode == RunMode.GPU_OFFLOAD
    assert on_two.run_mode == RunMode.GPU  # was GPU_OFFLOAD: only one card counted
    assert on_two.gpu_count == 2
    assert on_two.memory_available_gb == pytest.approx(23.44 + 22.46)
    # The cards take turns, so the limit is one card's speed, not double.
    assert on_two.estimated_tps == pytest.approx(936e9 / 1024**3 / on_two.weights_gb, abs=0.1)


def test_awq_on_two_3090s_is_tensor_parallel_and_reads_twice_as_fast():
    two = nvidia_pc(RTX_3090, RTX_3090_BUSY)
    awq = options_by_label(real_snapshot(QWEN_72B, two), two)["AWQ 4-bit"]
    assert awq.gpu_split == TENSOR_PARALLEL
    assert awq.memory_available_gb == pytest.approx(2 * 22.46)  # 2 x the smaller free VRAM
    assert awq.estimated_tps == pytest.approx(2 * 936e9 / 1024**3 / awq.weights_gb, abs=0.1)


def test_a_model_that_fits_on_the_fast_card_stays_off_the_slow_one(qwen_tensors):
    mixed = nvidia_pc(RTX_3090, RTX_3060)
    q4 = options_by_label(qwen_snapshot(qwen_tensors, mixed), mixed)["Q4_K_M"]
    assert q4.gpu_names == ("NVIDIA GeForce RTX 3090",)
    assert q4.gpu_split == ONE_GPU
    # Identical cards are as fast split as alone, and split has more room.
    pair = nvidia_pc(RTX_3090, RTX_3090)
    assert options_by_label(qwen_snapshot(qwen_tensors, pair), pair)["Q4_K_M"].gpu_count == 2
