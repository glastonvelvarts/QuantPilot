"""Sizes must match what llama.cpp really produces.

The expected numbers are not ours: they were printed by `llama-quantize --dry-run` (build b11429)
and by llama.cpp's load log for the real models.
"""

import pytest

from ocq.model_size import (
    gguf_weights_bytes,
    kv_cache_bytes,
    output_buffer_bytes,
    packed_weights_bytes,
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
