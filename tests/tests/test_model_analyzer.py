"""Tests for model snapshots and fit."""

import pytest
from conftest import (
    DEEPSEEK_V2_LITE,
    FALCON_H1,
    GEMMA2_2B,
    GEMMA3_1B,
    GPT_OSS,
    GRANITE_H,
    JAMBA,
    LFM2,
    LLAMA4_SCOUT,
    MISTRAL_V01,
    MIXTRAL,
    NEMOTRON_H,
    OCQ_NVFP4,
    QWEN,
    QWEN3_8B_NVFP4,
    QWEN3_MOE,
    QWEN3_NEXT,
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
from ocq.model_size import kv_value_bytes
from ocq.types import (
    LLAMA_CPP,
    ONE_GPU,
    TENSOR_PARALLEL,
    TRANSFORMERS,
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
    snap = real_snapshot(QWEN_72B, two)
    on_one = options_by_label(real_snapshot(QWEN_72B, one), one)["Q4_K_M"]
    on_two = options_by_label(snap, two)["Q4_K_M"]
    assert on_one.run_mode == RunMode.GPU_OFFLOAD
    assert on_two.run_mode == RunMode.GPU  # was GPU_OFFLOAD: only one card counted
    assert on_two.gpu_count == 2
    assert on_two.memory_available_gb == pytest.approx(23.44 + 22.46)
    # The cards take turns, so the limit is one card's speed, not double. Each token reads
    # the weights minus the input embedding table (one row of it).
    read_gb = on_two.weights_gb * snap.read_fraction
    assert on_two.estimated_tps == pytest.approx(936e9 / 1024**3 / read_gb, abs=0.1)


def test_awq_on_two_3090s_is_tensor_parallel_and_reads_twice_as_fast():
    two = nvidia_pc(RTX_3090, RTX_3090_BUSY)
    snap = real_snapshot(QWEN_72B, two)
    awq = options_by_label(snap, two)["AWQ 4-bit"]
    assert awq.gpu_split == TENSOR_PARALLEL
    assert awq.memory_available_gb == pytest.approx(2 * 22.46)  # 2 x the smaller free VRAM
    read_gb = awq.weights_gb * snap.read_fraction
    assert awq.estimated_tps == pytest.approx(2 * 936e9 / 1024**3 / read_gb, abs=0.1)


def test_a_model_that_fits_on_the_fast_card_stays_off_the_slow_one(qwen_tensors):
    mixed = nvidia_pc(RTX_3090, RTX_3060)
    q4 = options_by_label(qwen_snapshot(qwen_tensors, mixed), mixed)["Q4_K_M"]
    assert q4.gpu_names == ("NVIDIA GeForce RTX 3090",)
    assert q4.gpu_split == ONE_GPU
    # Identical cards are as fast split as alone, and split has more room.
    pair = nvidia_pc(RTX_3090, RTX_3090)
    assert options_by_label(qwen_snapshot(qwen_tensors, pair), pair)["Q4_K_M"].gpu_count == 2


# Mixture of experts: every expert is in memory, only a few run for each token.


@pytest.mark.parametrize("repo", [QWEN3_MOE, DEEPSEEK_V2_LITE, MIXTRAL, GPT_OSS])
def test_expert_count_is_cross_checked_and_marks_the_model_as_moe(repo):
    snap = real_snapshot(repo)
    assert snap.is_moe
    assert snap.facts.active_parameters.kind == FactKind.CROSS_CHECKED
    assert snap.facts.active_parameters.source == "config.json + tensor names"
    assert snap.read_fraction < 0.3


def test_moe_speed_limit_follows_the_weights_used_per_token():
    hardware = nvidia_pc(RTX_3090, RTX_3090_BUSY)
    snap = real_snapshot(QWEN3_MOE, hardware)
    q4 = options_by_label(snap, hardware)["Q4_K_M"]
    # All 30.5B weights must fit in memory ...
    assert q4.weights_gb > 16
    # ... but a token reads 3.0B of them (10%), so the limit is about 10x a dense model's.
    assert snap.read_fraction == pytest.approx(3_041_867_776 / 30_532_122_624)
    gpu = 936e9 / 1024**3
    assert q4.estimated_tps == pytest.approx(gpu / (q4.weights_gb * snap.read_fraction), abs=0.5)
    assert q4.estimated_tps > 9 * gpu / q4.weights_gb


def test_expert_counts_that_disagree_are_an_estimate():
    meta, config, tensors, dtypes = real_model(MIXTRAL)
    wrong = config | {"num_local_experts": 16}
    snap = build_snapshot(MIXTRAL, meta, wrong, tensors, None, dtypes=dtypes)
    assert snap.facts.active_parameters.kind == FactKind.ESTIMATE
    assert (
        "config.json says 16 experts, the tensor names show 8" in snap.facts.active_parameters.note
    )


def test_moe_is_read_from_the_files_not_the_name(qwen_tensors):
    # Before, a "moe" in the name or tags made any model a mixture of experts.
    snap = build_snapshot("org/moe-chat-1.5b", {"tags": ["moe"]}, QWEN_CONFIG, qwen_tensors, None)
    assert not snap.is_moe
    assert snap.read_fraction == 1.0


def test_without_experts_per_token_the_active_weights_are_unknown():
    meta, config, tensors, dtypes = real_model(MIXTRAL)
    no_top_k = {k: v for k, v in config.items() if k != "num_experts_per_tok"}
    snap = build_snapshot(MIXTRAL, meta, no_top_k, tensors, None, dtypes=dtypes)
    assert not snap.facts.active_parameters.known
    assert snap.read_fraction == 1.0


def test_benchmark_runs_are_stored_as_what_one_token_reads():
    from types import SimpleNamespace

    from ocq.pipeline import remember_speed
    from ocq.types import QuantAlgorithm

    snap = real_snapshot(QWEN3_MOE)
    plan = SimpleNamespace(model=snap, quant_config=SimpleNamespace(algorithm=QuantAlgorithm.GGUF))
    run = SimpleNamespace(artifact_size_gb=17.0)
    remember_speed(plan, run, SimpleNamespace(device="gpu", tokens_per_second=120.0))
    # A 17 GB MoE file that reads 1.7 GB per token must not look like a fast 17 GB dense model.
    assert measurements.load()[-1]["weights_gb"] == pytest.approx(17.0 * snap.read_fraction)


# Context memory for sliding-window attention and MLA. Expected sizes are from llama.cpp's own
# log (build b11429, -c N -v): "llama_kv_cache: size = ... MiB".

MIB = 2**20


@pytest.mark.parametrize(
    ("repo", "context", "conversations", "llama_cpp_mib"),
    [
        # llama-completion (1 conversation). Gemma 3 1B: 4 full layers x 16 or 128 MiB, and a
        # sliding cache of 1024 cells (512 window + 512 ubatch) for 22 layers: 22 MiB.
        (GEMMA3_1B, 4096, 1, 16 + 22),
        (GEMMA3_1B, 32768, 1, 128 + 22),
        # Gemma 2 2B: 13 full layers 416 MiB, 13 sliding layers of 4608 cells 234 MiB.
        (GEMMA2_2B, 8192, 1, 416 + 234),
        # llama-server (n_parallel auto = 4): the sliding cache holds 4 windows, 2560 cells ...
        (GEMMA3_1B, 32768, 4, 128 + 55),
        # ... and for Gemma 2 that is more than the 8192-token context, so all of it.
        (GEMMA2_2B, 8192, 4, 416 + 416),
    ],
)
def test_sliding_window_cache_matches_llama_cpp(repo, context, conversations, llama_cpp_mib):
    snap = real_snapshot(repo)
    assert snap.kv_cache_bytes(context, conversations=conversations) == llama_cpp_mib * MIB


def test_the_default_is_llama_servers_four_conversations():
    snap = real_snapshot(GEMMA3_1B)
    assert snap.kv_cache_bytes(32768) == snap.kv_cache_bytes(32768, conversations=4)


def test_gpt_oss_keeps_a_small_cache_on_its_sliding_layers():
    kv = real_snapshot(GPT_OSS).facts.kv_cache
    assert kv.kind == FactKind.CROSS_CHECKED
    assert "12 of 24 layers keep only the last 128 tokens" in kv.note
    # 12 full layers x 32,768 tokens + 12 sliding layers x 1024 cells (4 x 128 + 512), 2,048
    # bytes each
    expected = 12 * 2048 * 32768 + 12 * 2048 * 1024
    assert real_snapshot(GPT_OSS).kv_cache_bytes(32768) == expected


def test_a_window_on_every_layer_is_counted_in_full():
    # llama.cpp keeps Mistral 7B v0.1's whole cache: only models mixing layer kinds get less.
    snap = real_snapshot(MISTRAL_V01)
    assert snap.kv_cache_bytes(32768) == 32 * 4096 * 32768
    assert "on every layer, counted in full" in snap.facts.kv_cache.note


def test_a_disabled_window_is_ignored():
    # Qwen2.5 configs carry sliding_window 32768 with use_sliding_window false.
    assert real_snapshot(QWEN).kv_cache_bytes(32768) == 28 * 1024 * 32768


def test_mla_caches_a_compressed_vector_and_transformers_expands_it():
    snap = real_snapshot(DEEPSEEK_V2_LITE)
    kv = snap.facts.kv_cache
    # 27 layers x (kv_lora_rank 512 + rope 64) x 2 bytes, cross-checked with the width of
    # kv_a_proj_with_mqa, the projection that writes it. It used to be 221,184 (full heads).
    assert kv.value == 27 * 576 * 2
    assert kv.kind == FactKind.CROSS_CHECKED
    # transformers keeps 16 heads x (128 + 64) keys and 16 x 128 values per layer.
    assert snap.kv_cache_bytes(4096, TRANSFORMERS) == 27 * 16 * (192 + 128) * 2 * 4096


def test_fit_uses_the_cache_of_the_runtime_that_serves_the_row():
    hardware = nvidia_pc(RTX_3090)
    rows = options_by_label(real_snapshot(DEEPSEEK_V2_LITE, hardware), hardware)
    # GGUF (llama.cpp) keeps the compressed MLA cache, bitsandbytes (transformers) expands it.
    # The output buffer is the same for both, so the difference is the two caches.
    compressed, expanded = 27 * 576 * 2, 27 * 16 * (192 + 128) * 2
    difference = rows["BNB 4-bit double"].context_gb - rows["Q4_K_M"].context_gb
    assert difference * 1024**3 == pytest.approx((expanded - compressed) * 4096)


# Hybrids: only attention layers keep a cache that grows with the context. Reading the wrong
# layers would not match the tensor shapes, since only attention layers have k_proj and v_proj.


@pytest.mark.parametrize(
    ("repo", "attention_layers", "layers", "bytes_per_layer", "was"),
    [
        (LFM2, 6, 16, 2048, 32_768),  # layers 2, 5, 8, 10, 12, 14
        (JAMBA, 2, 28, 512, 14_336),  # layers 7 and 21
        (GRANITE_H, 4, 40, 2048, 81_920),  # layers 5, 15, 25, 35
        (NEMOTRON_H, 4, 56, 4096, 229_376),  # layers 14, 21, 30, 39
    ],
)
def test_only_attention_layers_of_a_hybrid_keep_a_cache(
    repo, attention_layers, layers, bytes_per_layer, was
):
    kv = real_snapshot(repo).facts.kv_cache
    assert kv.value == attention_layers * bytes_per_layer
    assert kv.value < was  # every layer used to be counted
    assert kv.kind == FactKind.CROSS_CHECKED
    stateful = layers - attention_layers
    assert f"{stateful} of {layers} layers keep a fixed-size state" in kv.note


def test_qwen3_next_has_attention_every_fourth_layer():
    kv = real_snapshot(QWEN3_NEXT).facts.kv_cache
    assert kv.value == 12 * 2 * 2 * 256 * 2  # 12 layers x keys and values x 2 heads x 256 x 2 B
    assert "as full_attention_interval says" in kv.note


def test_a_hybrid_with_attention_in_every_layer_keeps_it_all():
    # Falcon-H1 runs attention and Mamba side by side, so every layer has a cache.
    kv = real_snapshot(FALCON_H1).facts.kv_cache
    assert kv.value == 36 * 2 * 2 * 64 * 2
    assert "fixed-size state" not in kv.note


def test_llama4_chunked_layers_keep_one_chunk():
    snap = real_snapshot(LLAMA4_SCOUT)
    assert "36 of 48 layers attend within 8,192-token chunks" in snap.facts.kv_cache.note
    # 12 full layers keep every token, 36 chunked ones 8,192 + 512 cells for 1 conversation.
    per_layer = 2 * 8 * 128 * 2
    expected = 12 * per_layer * 65536 + 36 * per_layer * 8704
    assert snap.kv_cache_bytes(65536, conversations=1) == expected


def test_the_note_says_which_runtime_settings_are_assumed():
    assert "assumes 16-bit cache" in real_snapshot(QWEN).facts.kv_cache.note
    note = real_snapshot(GEMMA3_1B).facts.kv_cache.note
    assert "windows sized for 4 conversations" in note
    assert "183 MiB at 32,768 tokens (150 MiB for 1 conversation)" in note


# --kv-cache: llama.cpp's own log gives the sizes (-ctk/-ctv q8_0 or q4_0, -c N, 1 conversation).


@pytest.mark.parametrize(
    ("repo", "context", "cache", "llama_cpp_mib"),
    [
        (GEMMA3_1B, 32768, "q8_0", 68 + 11.6875),  # "68.00 MiB" + "11.69 MiB"
        (GEMMA3_1B, 32768, "q4_0", 36 + 6.1875),  # "36.00 MiB" + "6.19 MiB"
        # SmolLM2-135M, QuantPilot's own benchmark run with --kv-cache q8_0: "47.81 MiB"
        (OCQ_NVFP4, 4096, "q8_0", 47.8125),
    ],
)
def test_quantized_cache_matches_llama_cpp(repo, context, cache, llama_cpp_mib):
    value_bytes = kv_value_bytes(cache, LLAMA_CPP)
    size = real_snapshot(repo).kv_cache_bytes(context, LLAMA_CPP, 1, value_bytes)
    assert size == llama_cpp_mib * MIB


def test_each_row_uses_the_cache_of_its_runtime():
    hardware = nvidia_pc(RTX_3090)
    snap = real_snapshot(QWEN, hardware)
    f16 = options_by_label(snap, hardware, context=32768)
    q8 = {o.label: o for o in quant_options(snap, hardware, 32768, kv_cache="q8_0")}
    full = 28 * 1024 * 32768  # Qwen2.5-1.5B at 16-bit

    def saved(label):
        return (f16[label].context_gb - q8[label].context_gb) * 1024**3

    assert saved("Q4_K_M") == pytest.approx(full * (1 - 34 / 64))  # llama.cpp q8_0
    assert saved("AWQ 4-bit") == pytest.approx(full / 2)  # vLLM fp8
    assert saved("BNB 4-bit double") == 0  # transformers keeps 16 bits
