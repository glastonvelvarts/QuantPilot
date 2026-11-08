"""Exact memory of a model, computed from its real tensor shapes.

GGUF sizes follow the same per-tensor rules llama-quantize applies (llama-quant.cpp, build b11429),
so the result matches the file it will write.
"""

from __future__ import annotations

import math
import re

from ocq.types import DTYPE_BITS, INTEGER_TYPES, KV_BYTES_PER_VALUE, TRANSFORMERS, VLLM

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

# The working memory counted for vLLM and transformers rows: the output scores of 512 tokens.
EVAL_BATCH = 512


def gguf_weights_bytes(tensors: list[Tensor], gguf_type: str, heads_per_kv_head: int = 1) -> int:
    return sum(gguf_tensor_bytes(tensors, gguf_type, heads_per_kv_head).values())


def gguf_tensor_bytes(
    tensors: list[Tensor], gguf_type: str, heads_per_kv_head: int = 1
) -> dict[str, int]:
    """Bytes of each tensor in the GGUF file, by its name in the original checkpoint."""
    base = BASE_TYPE[gguf_type]
    tied_embeddings = not any(tensor_role(name) == "output" for name, _ in tensors)
    layers = 1 + max((layer_index(name) for name, _ in tensors), default=0)
    value_tensors = sorted(
        (layer_index(name), name) for name, _ in tensors if tensor_role(name) == "attn_v"
    )
    value_order = {name: position for position, (_, name) in enumerate(value_tensors)}

    sizes = {}
    for name, shape in tensors:
        count = math.prod(shape)
        if len(shape) < 2 or tensor_role(name) == "router":
            # llama.cpp keeps norms and MoE routers (ffn_gate_inp) at 32 bits.
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
        sizes[name] = -(-size // 32) * 32
    return sizes


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
    # The MoE router: mlp.gate (Qwen, OLMoE, DeepSeek), block_sparse_moe.gate (Mixtral),
    # router (Granite, gpt-oss). Not mlp.gate_proj, the dense feed-forward.
    if name.endswith(".gate.weight") or ".router." in name:
        return "router"
    if "lm_head" in name:
        return "output"
    if "embed_tokens" in name or name.endswith("wte.weight"):
        return "token_embd"
    if "v_proj" in name or "qkv_proj" in name:
        return "attn_v"
    if "o_proj" in name:
        return "attn_output"
    # Granite's output_linear (experts and shared expert) becomes ffn_down_exps/_shexp in GGUF.
    if "down_proj" in name or "output_linear" in name:
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


def ffn_width(tensors: list[Tensor], experts: int = 0, experts_per_token: int = 0) -> int:
    """The widest feed-forward layer one token runs through: the input width of the down
    projection (down_proj, w2, Granite's output_linear), times experts_per_token for routed
    experts. 0 when no such tensor is found."""
    widest = 0
    for name, shape in tensors:
        if len(shape) < 2 or QUANT_METADATA.search(name) or name.endswith(("bias", "_bias")):
            continue
        if tensor_role(name) != "ffn_down" and ".w2." not in name:
            continue
        # gpt-oss's MXFP4 experts: [experts, rows, blocks, 16 bytes of 32 4-bit weights].
        width = shape[2] * shape[3] * 2 if len(shape) == 4 else shape[-1]
        if experts and experts_per_token and routed_expert(name, shape, experts):
            width *= experts_per_token
        widest = max(widest, width)
    return widest


# llama-server (b11429) needs two buffers besides the weights and the cache:
#   output   one row of logits (vocabulary x 4 bytes) per conversation;
#   compute  the attention masks of one 512-token batch (context x 512 x 2 bytes; sliding
#            window layers have a second one, as long as their cache) and the larger of two
#            peaks: the feed-forward one and the attention one.
# Feed-forward: on CUDA, for 9 models (dense, MoE, Mamba hybrids, gpt-oss), 3.7 to 9 times the
# widest feed-forward layer x 512 tokens x 4 bytes. 9 is used: an upper bound, at most 90 MiB
# over (Gemma 2 2B).
# Attention: when it needs a 16-bit copy of one layer's keys and values, that copy, the rows of
# activations around it and, on Metal, a scratch buffer for the partial results:
#   CUDA, quantized cache   its kernels read 16 bits (ggml_cuda_flash_attn_ext_get_alloc_size);
#   CUDA, layers in RAM     a layer whose cache is in RAM is brought to the GPU as stored;
#   Metal, any cache        it always reserves the copy ("extra_kv_f16", ggml-metal-ops.cpp).
# Measured besides the mask and the copy (MiB): CUDA q8_0 OLMoE 20.09, Qwen1.5-0.5B 10.05 (5.02
# rows of 512 x the attention width x 4 B); CUDA layers in RAM Qwen2.5-7B 42.00, OLMoE 24.00
# (6.0 rows); Metal Qwen1.5-0.5B 18.23 (4.13 scratch + 7.05 rows). 8 rows are used.
# A layer that runs attention beside a Mamba block (Falcon-H1) keeps the copy while the rest of
# the layer works, so the copy adds to the feed-forward peak: Falcon-H1 0.5B on CUDA, q8_0 over
# f16 at 4k / 16k / 64k, +4.03 / +9.03 / +33.03 MiB for a copy of 2 / 8 / 32 MiB.
UBATCH = 512
FFN_INTERMEDIATES = 9
ATTENTION_ACTIVATIONS = 8
METAL_WORKGROUPS = 32  # Metal's scratch: 32 rows x heads x 32 workgroups x (head + 2) x 4 B


def llama_server_buffers_bytes(
    vocab_size: int,
    context: int,
    ffn: int,
    sliding_tokens: int,
    conversations: int,
    kv_copy: int = 0,
    attention_width: int = 0,
    scratch: int = 0,
    beside_state: bool = False,
) -> int:
    """sliding_tokens: the cache length of the sliding-window layers (0: none).
    kv_copy: bytes of the 16-bit copy of one layer's keys and values, when one is made.
    attention_width: the wider of the hidden size and all heads' queries.
    scratch: Metal's buffer for partial results (metal_attention_scratch).
    beside_state: attention runs beside a Mamba block in the same layer (Falcon-H1)."""
    output = conversations * vocab_size * 4
    masks = (context + sliding_tokens) * UBATCH * 2
    intermediates = UBATCH * 4 * FFN_INTERMEDIATES * ffn
    if kv_copy:
        attention = kv_copy + UBATCH * 4 * ATTENTION_ACTIVATIONS * attention_width + scratch
        alongside = intermediates + kv_copy if beside_state else 0
        intermediates = max(intermediates, attention, alongside)
    return output + masks + intermediates


def metal_attention_scratch(heads: int, head_dim: int) -> int:
    return 4 * 32 * heads * METAL_WORKGROUPS * (head_dim + 2)


# Tensors a quantized checkpoint stores next to its weights: scales, zero points, group
# indexes, packed shapes. They are not weights of the model, so they are not counted as such.
# weight_scale_2 is NVIDIA ModelOpt's per-tensor NVFP4 scale; MXFP4 (gpt-oss) stores
# down_proj_blocks next to down_proj_scales; MLX stores its zero points as .biases (a layer's
# own bias is .bias), Ollama's MLX layout x.weight.scale and x.weight.bias.
QUANT_METADATA = re.compile(
    r"\.(qzeros|scales|biases|g_idx|SCB|absmax|quant_map|nested_absmax|nested_quant_map|weight_scale"
    r"|weight_scale_2|weight_scale_inv|weight_zero_point|weight_global_scale|weight_shape|input_scale"
    r"|input_global_scale|input_zero_point|k_scale|v_scale)$|_scales$|quant_state"
    r"|\.weight\.(scale|bias)$"
)


# The bit width of the layers quantized differently from the rest, by layer name, when a
# checkpoint mixes several (MLX lists them in config.json): {"model.embed_tokens": 8}.
LayerBits = dict[str, int]


def bits_of(name: str, quantized_bits: int | None, layer_bits: LayerBits | None) -> int | None:
    if quantized_bits is None or not layer_bits:
        return quantized_bits
    return layer_bits.get(name.rsplit(".", 1)[0], quantized_bits)


def parameters_by_type(
    tensors: list[Tensor],
    dtypes: dict[str, str],
    quantized_bits: int | None = None,
    layer_bits: LayerBits | None = None,
) -> dict[str, int] | None:
    """Weights per stored type, counted the way the original model counts them.

    quantized_bits is None for a model that is not quantized: every element is one weight.
    For a quantized model, scales and zero points are left out and integers holding several
    weights are unpacked: an int32 holds 8 weights at 4 bits (GPTQ, AWQ), a uint8 holds 2
    (bitsandbytes 4-bit, NVFP4), an int8 holds 1 (W8A8). Those are counted under
    "N-bit in <container>", for example "4-bit in I32".
    None when a packed tensor cannot be unpacked because its type or bit width is unknown.
    """
    counts: dict[str, int] = {}
    for name, shape in tensors:
        bits = bits_of(name, quantized_bits, layer_bits)
        held = tensor_weights(name, shape, dtypes.get(name, ""), bits)
        if held is None:
            return None
        label, weights = held
        if weights:
            counts[label] = counts.get(label, 0) + weights
    return counts


def tensor_weights(
    name: str, shape: tuple[int, ...], dtype: str, quantized_bits: int | None
) -> tuple[str, int] | None:
    """(type, weights) one tensor holds, as parameters_by_type counts them. A scale holds 0.

    None when a packed tensor cannot be unpacked.
    """
    elements = math.prod(shape)
    if quantized_bits is None:
        return dtype, elements
    if QUANT_METADATA.search(name):
        return dtype, 0
    if not dtype or dtype in INTEGER_TYPES:
        if not (dtype and quantized_bits):
            return None
        return f"{quantized_bits}-bit in {dtype}", elements * DTYPE_BITS[dtype] // quantized_bits
    return dtype, elements


# Mixture of experts: each layer has many experts, and a router picks a few for every token.
# One tensor per expert (Qwen3-MoE, DeepSeek, Mixtral: mlp.experts.7.up_proj.weight), or all
# experts in one tensor whose first dimension is the expert count (gpt-oss: mlp.experts.*).
EXPERT_INDEX = re.compile(r"\.experts\.(\d+)\.")
# Input embedding tables: Llama/Qwen, GPT-2, BLOOM, Nemotron-H (backbone.embeddings.weight).
EMBEDDING = (
    "embed_tokens.weight",
    "wte.weight",
    "word_embeddings.weight",
    "backbone.embeddings.weight",
)
OUTPUT = ("lm_head.weight", "output.weight")


def experts_in_names(tensors: list[Tensor]) -> int:
    """Routed experts per layer as the tensor names show them. 0 for a dense model.

    Shared experts (shared_experts, shared_expert) run for every token and are not counted.
    """
    indexed = {match.group(1) for name, _ in tensors if (match := EXPERT_INDEX.search(name))}
    if indexed:
        return len(indexed)
    fused = {shape[0] for name, shape in tensors if fused_experts(name, shape)}
    return fused.pop() if len(fused) == 1 else 0


def fused_experts(name: str, shape: tuple[int, ...]) -> bool:
    """All experts of a layer in one tensor, the expert count first: under mlp.experts
    (gpt-oss, Llama 4), or a 3-D tensor of an MoE block (Granite: block_sparse_moe.input_linear,
    [32, 1024, 1024]). The router ([32, 1024]) is 2-D and not an expert."""
    if ".experts." in name:
        return len(shape) >= 2
    return len(shape) == 3 and ("moe" in name or "expert" in name)


def routed_expert(name: str, shape: tuple[int, ...], experts: int) -> bool:
    """A tensor of the routed experts, of which only some run for each token."""
    if EXPERT_INDEX.search(name):
        return True
    return bool(experts) and fused_experts(name, shape) and shape[0] == experts


def weights_per_token(
    tensors: list[Tensor],
    dtypes: dict[str, str],
    quantized_bits: int | None,
    experts: int = 0,
    experts_per_token: int = 0,
    layer_bits: LayerBits | None = None,
) -> tuple[int, int] | None:
    """(weights read for one token, input embedding weights left out of that).

    A routed expert counts experts_per_token / experts of its weights: that share of the
    experts runs for each token. The input embedding table is a lookup, one row per token, so
    it is left out, unless no separate output layer is stored and it doubles as that.
    None when a packed tensor cannot be unpacked.
    """
    separate_output = any(name.endswith(OUTPUT) for name, _ in tensors)
    read, embedding = 0.0, 0
    for name, shape in tensors:
        bits = bits_of(name, quantized_bits, layer_bits)
        held = tensor_weights(name, shape, dtypes.get(name, ""), bits)
        if held is None:
            return None
        weights = held[1]
        if experts and experts_per_token and routed_expert(name, shape, experts):
            read += weights * experts_per_token / experts
        elif separate_output and name.endswith(EMBEDDING) and "vision" not in name:
            embedding += weights
        else:
            read += weights
    return round(read), embedding


def stored_bytes(tensors: list[Tensor], dtypes: dict[str, str]) -> int | None:
    """Bytes of every tensor in the weight files. None if a type is missing or not known."""
    total_bits = 0
    for name, shape in tensors:
        bits = DTYPE_BITS.get(dtypes.get(name, ""))
        if bits is None:
            return None
        total_bits += math.prod(shape) * bits
    return -(-total_bits // 8)


def kv_value_bytes(cache_type: str, runtime: str, vllm_fp8: bool = True) -> float:
    """Bytes one cached key or value takes with --kv-cache cache_type in this runtime.

    llama.cpp stores q8_0 and q4_0 in the same blocks as GGUF weights: 32 values in 34 or 18
    bytes (its log for Gemma 3 1B at 32k: f16 128 MiB, q8_0 68 MiB, q4_0 36 MiB). vLLM's
    quantized cache is fp8, 1 byte (--kv-cache-dtype fp8), and it has no 4-bit one; it needs a
    GPU of compute capability 8.9 or newer (vllm_fp8), else the cache stays at 16 bits.
    transformers keeps 16 bits.
    """
    if cache_type == "f16" or runtime == TRANSFORMERS:
        return KV_BYTES_PER_VALUE
    if runtime == VLLM:
        return 1.0 if vllm_fp8 else KV_BYTES_PER_VALUE
    block_bytes, values = BLOCKS[cache_type.upper()]
    return block_bytes / values
