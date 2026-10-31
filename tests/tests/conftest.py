"""Shared test setup."""

import json
import math
from pathlib import Path

import pytest

from ocq import measurements, model_analyzer
from ocq.types import DTYPE_BITS

# Real config.json, safetensors headers and HuggingFace API counts of a few repos, saved
# 2026-10-31. Headers only: names, types and shapes, no weights.
DATA = Path(__file__).parent / "data"


@pytest.fixture(autouse=True)
def private_caches(tmp_path, monkeypatch):
    """Tests must not read or write the speed measurements and model cache of the real user."""
    monkeypatch.setattr(measurements, "STORE", tmp_path / "measurements.json")
    monkeypatch.setattr(model_analyzer, "MODEL_CACHE", tmp_path / "models")


def llama_style_tensors(
    layers: int,
    hidden: int,
    intermediate: int,
    kv_width: int,
    vocab: int,
    attention_bias: bool = False,
) -> list[tuple[str, tuple[int, ...]]]:
    """Tensor names and shapes of a Llama/Qwen style model, as its safetensors files list them."""
    tensors = [("model.embed_tokens.weight", (vocab, hidden))]
    for layer in range(layers):
        prefix = f"model.layers.{layer}"
        tensors += [
            (f"{prefix}.self_attn.q_proj.weight", (hidden, hidden)),
            (f"{prefix}.self_attn.k_proj.weight", (kv_width, hidden)),
            (f"{prefix}.self_attn.v_proj.weight", (kv_width, hidden)),
            (f"{prefix}.self_attn.o_proj.weight", (hidden, hidden)),
            (f"{prefix}.mlp.gate_proj.weight", (intermediate, hidden)),
            (f"{prefix}.mlp.up_proj.weight", (intermediate, hidden)),
            (f"{prefix}.mlp.down_proj.weight", (hidden, intermediate)),
            (f"{prefix}.input_layernorm.weight", (hidden,)),
            (f"{prefix}.post_attention_layernorm.weight", (hidden,)),
        ]
        if attention_bias:
            tensors += [
                (f"{prefix}.self_attn.q_proj.bias", (hidden,)),
                (f"{prefix}.self_attn.k_proj.bias", (kv_width,)),
                (f"{prefix}.self_attn.v_proj.bias", (kv_width,)),
            ]
    tensors.append(("model.norm.weight", (hidden,)))
    return tensors


@pytest.fixture
def smollm_tensors():
    """HuggingFaceTB/SmolLM2-135M-Instruct. Its width (576) forces llama.cpp's fallback types."""
    return llama_style_tensors(layers=30, hidden=576, intermediate=1536, kv_width=192, vocab=49152)


@pytest.fixture
def qwen_tensors():
    """Qwen/Qwen2.5-1.5B-Instruct."""
    return llama_style_tensors(
        layers=28,
        hidden=1536,
        intermediate=8960,
        kv_width=256,
        vocab=151936,
        attention_bias=True,
    )


QWEN_CONFIG = {
    "architectures": ["Qwen2ForCausalLM"],
    "num_hidden_layers": 28,
    "hidden_size": 1536,
    "num_attention_heads": 12,
    "num_key_value_heads": 2,
    "vocab_size": 151936,
    "max_position_embeddings": 32768,
}


def write_safetensors(path, tensors, dtype: str | dict[str, str] = "BF16") -> None:
    """Writes only the header of a safetensors file: names, shapes and types, no weight data.

    dtype is one type for every tensor, or a type per tensor name.
    """
    header, offset = {}, 0
    for name, shape in tensors:
        tensor_dtype = dtype[name] if isinstance(dtype, dict) else dtype
        size = -(-math.prod(shape) * DTYPE_BITS[tensor_dtype] // 8)
        header[name] = {
            "dtype": tensor_dtype,
            "shape": list(shape),
            "data_offsets": [offset, offset + size],
        }
        offset += size
    encoded = json.dumps(header).encode()
    path.write_bytes(len(encoded).to_bytes(8, "little") + encoded)


def real_model(repo: str):
    """(meta, config, tensors, dtypes) of a saved repo, meta shaped like the HF API answer."""
    saved = json.loads((DATA / f"{repo.replace('/', '--')}.json").read_text())
    meta = {
        "safetensors": saved["hf_safetensors"],
        "cardData": {"base_model": saved["hf_base_model"]},
        "sha": saved["hf_sha"],
    }
    tensors = [(name, tuple(shape)) for name, (_, shape) in saved["tensors"].items()]
    dtypes = {name: dtype for name, (dtype, _) in saved["tensors"].items()}
    return meta, saved["config"], tensors, dtypes


QWEN = "Qwen/Qwen2.5-1.5B-Instruct"
QWEN_GPTQ = "Qwen/Qwen2.5-1.5B-Instruct-GPTQ-Int4"
QWEN_AWQ = "Qwen/Qwen2.5-1.5B-Instruct-AWQ"
QWEN3_FP8 = "Qwen/Qwen3-1.7B-FP8"
SMOLVLM = "HuggingFaceTB/SmolVLM-256M-Instruct"
