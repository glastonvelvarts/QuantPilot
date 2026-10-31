"""Sizes must match what llama.cpp really produces.

The expected numbers are not ours: they were printed by `llama-quantize --dry-run` (build b11429)
and by llama.cpp's load log for the real models.
"""

import pytest
from conftest import (
    OCQ_NVFP4,
    QWEN,
    QWEN3_8B_NVFP4,
    QWEN3_FP8,
    QWEN_AWQ,
    QWEN_BNB,
    QWEN_GPTQ,
    QWEN_W8A8,
    SMOLVLM,
    real_model,
)

from ocq.model_size import (
    bnb_scale_bits,
    bnb_weights_bytes,
    gguf_weights_bytes,
    kv_cache_bytes,
    nvfp4_weights_bytes,
    output_buffer_bytes,
    packed_weights_bytes,
    parameters_by_type,
    stored_bytes,
)

MIB = 1024**2

SMOLLM_135M = {
    "Q2_K": 82.41,
    "Q3_K_S": 82.41,
    "Q3_K_M": 87.48,
    "Q3_K_L": 91.31,
    "Q4_0": 85.77,
    "Q4_K_S": 95.61,
    "Q4_K_M": 98.87,
    "Q5_0": 98.43,
    "Q5_K_S": 103.18,
    "Q5_K_M": 105.21,
    "Q6_K": 130.27,
    "Q8_0": 136.40,
}

QWEN_1_5B = {
    "Q2_K": 639.30,
    "Q3_K_S": 720.02,
    "Q3_K_M": 780.32,
    "Q3_K_L": 833.71,
    "Q4_0": 885.97,
    "Q4_K_S": 891.08,
    "Q4_K_M": 934.69,
    "Q5_0": 1042.15,
    "Q5_K_S": 1042.15,
    "Q5_K_M": 1067.26,
    "Q6_K": 1208.10,
    "Q8_0": 1564.62,
}


@pytest.mark.parametrize("gguf_type, llama_mib", SMOLLM_135M.items())
def test_gguf_size_matches_llama_quantize_with_fallback_types(smollm_tensors, gguf_type, llama_mib):
    ours = gguf_weights_bytes(smollm_tensors, gguf_type, heads_per_kv_head=3) / MIB
    assert ours == pytest.approx(llama_mib, rel=0.0003)


@pytest.mark.parametrize("gguf_type, llama_mib", QWEN_1_5B.items())
def test_gguf_size_matches_llama_quantize(qwen_tensors, gguf_type, llama_mib):
    ours = gguf_weights_bytes(qwen_tensors, gguf_type, heads_per_kv_head=6) / MIB
    assert ours == pytest.approx(llama_mib, rel=0.0003)


def test_kv_cache_matches_llama_load_log():
    # "KV buffer size = 180.00 MiB" for SmolLM2-135M at 8192 tokens
    assert kv_cache_bytes(layers=30, kv_heads=3, head_dim=64, context=8192) == 180 * MIB
    # "KV buffer size = 384.00 MiB" for Qwen2.5-0.5B at 32768 tokens
    assert kv_cache_bytes(layers=24, kv_heads=2, head_dim=64, context=32768) == 384 * MIB
    # "KV buffer size = 896.00 MiB" for Qwen2.5-1.5B at 32768 tokens
    assert kv_cache_bytes(layers=28, kv_heads=2, head_dim=128, context=32768) == 896 * MIB


def test_output_buffer_is_most_of_llama_compute_buffer():
    # llama.cpp reported a 335.77 MiB compute buffer for Qwen2.5-0.5B (vocab 151936)
    assert 290 < output_buffer_bytes(151936) / MIB < 336


def test_packed_size_matches_real_awq_artifact(smollm_tensors):
    # The AWQ 4-bit, group size 64 artifact produced for SmolLM2-135M was 114,029,456 bytes.
    ours = packed_weights_bytes(smollm_tensors, bits=4, group_size=64)
    assert ours == pytest.approx(114_029_456, rel=0.01)


# Artifacts written by transformers 5.16.1 + bitsandbytes 0.50.2 for SmolLM2-135M-Instruct
# (save_pretrained, total bytes of the .safetensors files).
SMOLLM_BNB = {
    (4, False): 116_548_186,
    (4, True): 111_879_761,
    (8, False): 163_557_634,
}


@pytest.mark.parametrize("bits, double_quant", SMOLLM_BNB)
def test_bnb_size_matches_real_artifact(smollm_tensors, bits, double_quant):
    ours = bnb_weights_bytes(smollm_tensors, bits, double_quant)
    assert ours == pytest.approx(SMOLLM_BNB[bits, double_quant], rel=0.003)


def test_double_quant_saves_about_0_37_bits_per_weight(qwen_tensors):
    assert bnb_scale_bits(4, double_quant=False) == 0.5
    assert bnb_scale_bits(4, double_quant=True) == pytest.approx(0.127, abs=0.001)
    assert bnb_scale_bits(8, double_quant=True) == 0.0

    single = bnb_weights_bytes(qwen_tensors, 4, double_quant=False)
    double = bnb_weights_bytes(qwen_tensors, 4, double_quant=True)
    quantized = sum(
        shape[0] * shape[1]
        for name, shape in qwen_tensors
        if len(shape) == 2 and "embed" not in name
    )
    assert (single - double) * 8 / quantized == pytest.approx(0.373, abs=0.005)


# Artifacts written by llm-compressor 0.14.0 for SmolLM2-135M-Instruct (save_compressed=True).
SMOLLM_NVFP4 = {
    False: 116_492_240,  # NVFP4A16, weights only
    True: 116_515_496,  # NVFP4, weights and activations
}


@pytest.mark.parametrize("fp4_activations", SMOLLM_NVFP4)
def test_nvfp4_size_matches_real_artifact(smollm_tensors, fp4_activations):
    ours = nvfp4_weights_bytes(smollm_tensors, fp4_activations)
    assert ours == pytest.approx(SMOLLM_NVFP4[fp4_activations], rel=0.002)


@pytest.mark.parametrize(
    ("repo", "quantized_bits"),
    [(QWEN, None), (QWEN_GPTQ, 4), (QWEN_AWQ, 4), (QWEN3_FP8, 8), (SMOLVLM, None)],
)
def test_weight_count_matches_the_huggingface_api(repo, quantized_bits):
    # GPTQ and AWQ pack 8 weights per int32 and store scales and zeros next to them; FP8 stores
    # a scale per 128x128 block. Unpacked and without those, the count is HuggingFace's.
    meta, _, tensors, dtypes = real_model(repo)
    counts = parameters_by_type(tensors, dtypes, quantized_bits)
    assert sum(counts.values()) == meta["safetensors"]["total"]


def test_packed_weights_are_counted_apart_from_plain_ones():
    _, _, tensors, dtypes = real_model(QWEN_GPTQ)
    assert parameters_by_type(tensors, dtypes, 4) == {
        "4-bit in I32": 1_310_195_712,
        "F16": 233_518_592,
    }


def test_packed_weights_without_a_bit_width_or_type_cannot_be_counted():
    _, _, tensors, dtypes = real_model(QWEN_GPTQ)
    assert parameters_by_type(tensors, dtypes, 0) is None
    assert parameters_by_type(tensors, {}, 4) is None


@pytest.mark.parametrize(
    ("repo", "expected"),
    [
        (QWEN, 3_087_428_608),  # 1,543,714,304 weights x 2 bytes
        (QWEN_GPTQ, 1_149_760_512),  # 1.07 GiB, not the 5.32 GB the int32 counts suggested
        (QWEN3_FP8, 2_654_365_696),  # 1 byte per FP8 weight
    ],
)
def test_stored_bytes_follow_each_tensor_type(repo, expected):
    _, _, tensors, dtypes = real_model(repo)
    assert stored_bytes(tensors, dtypes) == expected


def test_stored_bytes_are_unknown_for_a_type_we_do_not_know():
    assert stored_bytes([("w", (4, 4))], {"w": "F3_NEW"}) is None
    assert stored_bytes([("w", (4, 4))], {}) is None
    assert stored_bytes([("w", (4, 4))], {"w": "F4"}) == 8


@pytest.mark.parametrize(
    ("repo", "quantized_bits", "expected"),
    [
        # 2 weights per byte; absmax, quant_map and quant_state are scales. The HF API counts
        # those too and says 1,584,858,476.
        (QWEN_BNB, 4, 1_543_714_304),
        # 2 FP4 weights per byte, FP8 scales per 16, F32 weight_scale_2 per tensor. The HF API
        # does not unpack and says 4,717,851,648. Qwen/Qwen3-8B has 8,190,735,360.
        (QWEN3_8B_NVFP4, 4, 8_190_735_360),
        # 1 weight per int8. 1.54B plus the lm_head copy this repo stores. The HF API counts
        # the 645,120 per-channel scales too.
        (QWEN_W8A8, 8, 1_777_088_000),
        # Our own NVFP4 output: weight_packed, 2 per byte. HuggingFaceTB/SmolLM2-135M-Instruct
        # has 134,515,008.
        (OCQ_NVFP4, 4, 134_515_008),
    ],
)
def test_weight_count_where_the_huggingface_api_is_wrong(repo, quantized_bits, expected):
    _, _, tensors, dtypes = real_model(repo)
    assert sum(parameters_by_type(tensors, dtypes, quantized_bits).values()) == expected


def test_quantized_weights_are_named_by_their_container():
    for repo, bits, label in [(QWEN_BNB, 4, "4-bit in U8"), (QWEN_W8A8, 8, "8-bit in I8")]:
        _, _, tensors, dtypes = real_model(repo)
        assert label in parameters_by_type(tensors, dtypes, bits)
