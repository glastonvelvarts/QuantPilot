"""Exact memory of a model, computed from its real tensor shapes.

GGUF sizes follow the same per-tensor rules llama-quantize applies (llama-quant.cpp, build b11429),
so the result matches the file it will write.
"""

from __future__ import annotations

import math
import re

Tensor = tuple[str, tuple[int, ...]]

# (bytes per block, weights per block) of every llama.cpp tensor type we can end up with.
BLOCKS = {
    "F32": (4, 1),
    "F16": (2, 1),
    "Q4_0": (18, 32),
    "Q5_0": (22, 32),
    "Q5_1": (24, 32),
    "Q8_0": (34, 32),
    "Q2_K": (84, 256),
    "Q3_K": (110, 256),
    "Q4_K": (144, 256),
    "Q5_K": (176, 256),
    "Q6_K": (210, 256),
}

BASE_TYPE = {
    "F32": "F32",
    "BF16": "F16",
    "F16": "F16",
    "FP16": "F16",
    "Q2_K": "Q2_K",
    "Q3_K_S": "Q3_K",
    "Q3_K_M": "Q3_K",
    "Q3_K_L": "Q3_K",
    "Q4_0": "Q4_0",
    "Q4_K_S": "Q4_K",
    "Q4_K_M": "Q4_K",
    "Q5_0": "Q5_0",
    "Q5_K_S": "Q5_K",
    "Q5_K_M": "Q5_K",
    "Q6_K": "Q6_K",
    "Q8_0": "Q8_0",
}

# Used when a tensor's width is not a multiple of the block size.
FALLBACK = {"Q2_K": "Q4_0", "Q3_K": "Q4_0", "Q4_K": "Q5_0", "Q5_K": "Q5_1", "Q6_K": "Q8_0"}

# llama.cpp evaluates this many tokens at once and keeps their output scores in memory.
EVAL_BATCH = 512


def gguf_weights_bytes(tensors: list[Tensor], gguf_type: str, heads_per_kv_head: int = 1) -> int:
    base = BASE_TYPE[gguf_type]
    tied_embeddings = not any(tensor_role(name) == "output" for name, _ in tensors)
    layers = 1 + max((layer_index(name) for name, _ in tensors), default=0)
    value_tensors = sorted(
        (layer_index(name), name) for name, _ in tensors if tensor_role(name) == "attn_v"
    )
    value_order = {name: position for position, (_, name) in enumerate(value_tensors)}

    total = 0
    for name, shape in tensors:
        count = math.prod(shape)
        if len(shape) < 2:
            tensor_type = "F32"
        elif base in ("F16", "F32"):
            tensor_type = base
        else:
            tensor_type = quantized_type(
                gguf_type,
                base,
                tensor_role(name),
                tied_embeddings,
                columns=shape[-1],
                layer=layer_index(name),
                layers=layers,
                value_index=value_order.get(name, 0),
                value_count=len(value_tensors),
                heads_per_kv_head=heads_per_kv_head,
            )
        block_bytes, block_weights = BLOCKS[tensor_type]
        size = count // block_weights * block_bytes
        total += -(-size // 32) * 32
    return total


def quantized_type(
    gguf_type: str,
    base: str,
    role: str,
    tied_embeddings: bool,
    columns: int,
    layer: int,
    layers: int,
    value_index: int,
    value_count: int,
    heads_per_kv_head: int,
) -> str:
    chosen = base

    if role == "output" or (tied_embeddings and role == "token_embd"):
        if columns % BLOCKS[base][1] != 0:
            chosen = "Q8_0"
        elif base != "Q8_0":
            chosen = "Q6_K"
    elif role == "attn_v":
        if gguf_type == "Q2_K":
            chosen = "Q4_K" if heads_per_kv_head >= 4 else "Q3_K"
        elif gguf_type == "Q3_K_M":
            chosen = "Q5_K" if value_index < 2 else "Q4_K"
        elif gguf_type == "Q3_K_L":
            chosen = "Q5_K"
        elif gguf_type in ("Q4_K_M", "Q5_K_M") and more_bits(value_index, value_count):
            chosen = "Q6_K"
        elif gguf_type == "Q4_K_S" and value_index < 4:
            chosen = "Q5_K"
    elif role == "ffn_down":
        if gguf_type == "Q2_K":
            chosen = "Q3_K"
        elif gguf_type == "Q3_K_M":
            chosen = "Q5_K" if layer < layers // 16 else "Q4_K"
        elif gguf_type == "Q3_K_L":
            chosen = "Q5_K"
        elif gguf_type in ("Q4_K_M", "Q5_K_M") and more_bits(layer, layers):
            chosen = "Q6_K"
        elif gguf_type == "Q4_K_S" and layer < layers // 8:
            chosen = "Q5_K"
    elif role == "attn_output":
        if gguf_type == "Q2_K":
            chosen = "Q3_K"
        elif gguf_type == "Q3_K_M":
            chosen = "Q4_K"
        elif gguf_type == "Q3_K_L":
            chosen = "Q5_K"

    if columns % BLOCKS[chosen][1] != 0:
        chosen = FALLBACK.get(chosen, chosen)
        if columns % BLOCKS[chosen][1] != 0:
            chosen = "F16"
    return chosen


def more_bits(index: int, count: int) -> bool:
    return index < count // 8 or index >= 7 * count // 8 or (index - count // 8) % 3 == 2


def tensor_role(name: str) -> str:
    if "lm_head" in name:
        return "output"
    if "embed_tokens" in name or name.endswith("wte.weight"):
        return "token_embd"
    if "v_proj" in name or "qkv_proj" in name:
        return "attn_v"
    if "o_proj" in name:
        return "attn_output"
    if "down_proj" in name:
        return "ffn_down"
    return "other"


def layer_index(name: str) -> int:
    match = re.search(r"\.(\d+)\.", name)
    return int(match.group(1)) if match else 0


def packed_weights_bytes(tensors: list[Tensor], bits: int, group_size: int) -> int:
    """AWQ, GPTQ and bitsandbytes pack the linear layers and leave the embeddings at 16-bit."""
    total = 0.0
    for name, shape in tensors:
        count = math.prod(shape)
        if len(shape) < 2 or tensor_role(name) in ("token_embd", "output"):
            total += count * 2
            continue
        total += count * bits / 8
        if group_size:
            total += count / group_size * 3
    return int(total)


# bitsandbytes 4-bit: every 64 weights share one scale (absmax).
BNB_BLOCK = 64
# Double quantization: those scales are stored in 8 bits, and every 256 of them share one
# 32-bit scale. Without it, each scale is a plain 32-bit float.
BNB_NESTED_BLOCK = 256
# Lookup tables saved once per tensor: the 16 NF4 values, plus 256 values for the 8-bit scales.
BNB_CODE_BYTES = 16 * 4
BNB_NESTED_CODE_BYTES = 256 * 4


def bnb_scale_bits(bits: int, double_quant: bool) -> float:
    """Extra bits per weight that bitsandbytes spends on scales. 0.5 single, about 0.127 double."""
    if bits != 4:
        return 0.0
    if double_quant:
        return 8 / BNB_BLOCK + 32 / (BNB_BLOCK * BNB_NESTED_BLOCK)
    return 32 / BNB_BLOCK


def bnb_weights_bytes(tensors: list[Tensor], bits: int, double_quant: bool = False) -> int:
    """bitsandbytes quantizes the linear layers and leaves embeddings, lm_head and norms at 16-bit.

    4-bit packs two weights per byte and adds the scales described above.
    8-bit stores one byte per weight plus one 32-bit scale per output row.
    """
    total = 0
    for name, shape in tensors:
        count = math.prod(shape)
        if len(shape) < 2 or tensor_role(name) in ("token_embd", "output"):
            total += count * 2
        elif bits == 8:
            total += count + shape[0] * 4 + 1
        else:
            blocks = -(-count // BNB_BLOCK)
            total += -(-count // 2) + BNB_CODE_BYTES
            if double_quant:
                total += blocks + -(-blocks // BNB_NESTED_BLOCK) * 4 + BNB_NESTED_CODE_BYTES
            else:
                total += blocks * 4
    return total


# NVFP4: every 16 weights share one 8-bit (FP8 E4M3) scale, and every tensor has one 32-bit
# scale on top of those. Two levels of scales, built in.
NVFP4_BLOCK = 16


def nvfp4_weights_bytes(tensors: list[Tensor], fp4_activations: bool = False) -> int:
    """llm-compressor's NVFP4 format. Embeddings, lm_head and norms stay at 16-bit.

    Each linear layer stores its 4-bit weights packed two per byte, one FP8 scale per 16 weights
    and a 32-bit global scale. FP4 activations add a 32-bit input scale per layer.
    """
    total = 0
    for name, shape in tensors:
        count = math.prod(shape)
        if len(shape) < 2 or tensor_role(name) in ("token_embd", "output"):
            total += count * 2
            continue
        total += -(-count // 2) + -(-count // NVFP4_BLOCK) + 4
        if fp4_activations:
            total += 4
    return total


def group_size_that_fits(tensors: list[Tensor], wanted: int) -> int:
    """AWQ and GPTQ need the group size to divide the width of every layer they quantize."""
    widths = {
        shape[-1]
        for name, shape in tensors
        if len(shape) >= 2 and tensor_role(name) not in ("token_embd", "output")
    }
    for size in (wanted, 128, 64, 32, 16, 8):
        if size <= wanted and all(width % size == 0 for width in widths):
            return size
    return wanted


def kv_cache_bytes(layers: int, kv_heads: int, head_dim: int, context: int) -> int:
    """Keys and values, 16-bit each, for every layer and every token of context."""
    return 2 * layers * context * kv_heads * head_dim * 2


def output_buffer_bytes(vocab_size: int) -> int:
    return vocab_size * EVAL_BATCH * 4
