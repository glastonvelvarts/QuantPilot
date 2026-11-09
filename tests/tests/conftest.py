"""Shared test setup."""

import gzip
import json
import math
from pathlib import Path

import pytest

from ocq import discovery, env_file, hardware, measurements, model_analyzer, quality, recommend
from ocq.quantizers import llama_cpp
from ocq.types import DTYPE_BITS, OcqError

# Real config.json, safetensors headers and HuggingFace API counts of a few repos, saved
# 2026-10-31. Headers only: names, types and shapes, no weights.
DATA = Path(__file__).parent / "data"


@pytest.fixture(autouse=True)
def private_caches(tmp_path, monkeypatch, request):
    """Tests must not read or write the speed measurements and model cache of the real user."""
    monkeypatch.setattr(measurements, "STORE", tmp_path / "measurements.json")
    monkeypatch.setattr(quality, "STORE", tmp_path / "quality.json")
    # Real quantizations (-m slow) keep the downloads they need: the WikiText text and
    # llama.cpp's release are fetched once, not on every run. Their results stay private.
    real_run = request.node.get_closest_marker("slow") is not None
    if not real_run:
        monkeypatch.setattr(quality, "DATASETS", tmp_path / "datasets")
    monkeypatch.setattr(model_analyzer, "MODEL_CACHE", tmp_path / "models")
    # Nor the recommendation engine's list of Hub models, nor the Hub itself.
    monkeypatch.setattr(recommend, "HUB_CATALOG_CACHE", tmp_path / "hub_catalog.json")

    def no_network(author, pipeline):
        raise OSError("no network in tests")

    monkeypatch.setattr(recommend, "fetch_hub_listing", no_network)
    # Nor the files ocq optimize made in the folder the tests run from.
    monkeypatch.setattr(discovery, "quantpilot_dirs", lambda: [])
    # Nor the llama.cpp this PC happens to have downloaded: a test that needs its converter
    # says which architectures it knows.
    monkeypatch.delenv("OCQ_LLAMA_CPP", raising=False)
    if not real_run:
        monkeypatch.setattr(llama_cpp, "CACHE", tmp_path / "llama.cpp")
    # Nor the user's Hugging Face token, from the shell or from QuantPilot's .env file.
    for name in env_file.HF_TOKEN_NAMES:
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setattr(env_file, "ENV_FILE", tmp_path / "settings" / ".env")
    monkeypatch.setattr(env_file, "from_file", set())
    # Nor the GPU driver of the PC they run on (real quantizations place models with it).
    if not real_run:
        monkeypatch.setattr(hardware, "cuda_memory_mib", lambda: None)


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
    path = DATA / f"{repo.replace('/', '--')}.json"
    if path.exists():
        saved = json.loads(path.read_text())
    else:  # big headers (thousands of expert tensors) are stored gzipped
        saved = json.loads(gzip.decompress(path.with_suffix(".json.gz").read_bytes()))
    meta = {
        "safetensors": saved["hf_safetensors"] or {},
        "cardData": {"base_model": saved["hf_base_model"]},
        "sha": saved["hf_sha"],
    }
    if saved.get("hf_base_model_parameters"):
        meta["base_model_parameters"] = tuple(saved["hf_base_model_parameters"])
    tensors = [(name, tuple(shape)) for name, (_, shape) in saved["tensors"].items()]
    dtypes = {name: dtype for name, (dtype, _) in saved["tensors"].items()}
    return meta, saved["config"], tensors, dtypes


QWEN = "Qwen/Qwen2.5-1.5B-Instruct"
QWEN_GPTQ = "Qwen/Qwen2.5-1.5B-Instruct-GPTQ-Int4"
QWEN_AWQ = "Qwen/Qwen2.5-1.5B-Instruct-AWQ"
QWEN3_FP8 = "Qwen/Qwen3-1.7B-FP8"
SMOLVLM = "HuggingFaceTB/SmolVLM-256M-Instruct"
QWEN_BNB = "unsloth/Qwen2.5-1.5B-Instruct-bnb-4bit"
QWEN_W8A8 = "RedHatAI/Qwen2.5-1.5B-Instruct-quantized.w8a8"
QWEN3_8B_NVFP4 = "nvidia/Qwen3-8B-NVFP4"
# Written by `ocq optimize HuggingFaceTB/SmolLM2-135M-Instruct -b nvfp4 --execute` (llm-compressor).
OCQ_NVFP4 = "ocq/SmolLM2-135M-Instruct-nvfp4a16"
QWEN_72B = "Qwen/Qwen2.5-72B-Instruct"
# MLX (Apple): 4-bit weights packed 8 to a U32, with .scales and .biases per 64 weights.
MLX_QWEN = "mlx-community/Qwen2.5-0.5B-Instruct-4bit"
# MLX MXFP4 experts with 122 layers at 8-bit, each listed in config.json.
MLX_GPT_OSS = "mlx-community/gpt-oss-20b-MXFP4-Q8"
# Mixtures of experts, each laying its experts out differently.
QWEN3_MOE = "Qwen/Qwen3-30B-A3B"  # 128 experts, 8 per token, one tensor per expert
DEEPSEEK_V2_LITE = "deepseek-ai/DeepSeek-V2-Lite-Chat"  # + 2 shared experts, layer 0 dense
MIXTRAL = "mistralai/Mixtral-8x7B-Instruct-v0.1"  # 8 experts, 2 per token
GPT_OSS = "openai/gpt-oss-20b"  # 32 experts fused in one tensor, MXFP4 quantized
OLMOE = "allenai/OLMoE-1B-7B-0924-Instruct"  # 64 experts, 8 per token
GRANITE_MOE = "ibm-granite/granite-3.1-1b-a400m-instruct"  # 32 experts fused, 3-D, no "experts"
# Sliding-window attention (Google's own Gemma repos are gated; unsloth's carry the same config).
GEMMA3_1B = "unsloth/gemma-3-1b-it"  # 22 of 26 layers keep the last 512 tokens
GEMMA2_2B = "unsloth/gemma-2-2b-it"  # every other layer keeps the last 4,096 tokens
MISTRAL_V01 = "mistralai/Mistral-7B-Instruct-v0.1"  # a 4,096 window on every layer
# Hybrids: only some layers have attention, the others keep a fixed-size state.
LFM2 = "LiquidAI/LFM2-1.2B"  # full_attn_idxs, convolutions elsewhere
JAMBA = "ai21labs/AI21-Jamba-Reasoning-3B"  # attn_layer_period/offset, Mamba elsewhere
GRANITE_H = "ibm-granite/granite-4.0-h-tiny"  # layer_types "mamba" / "attention"
NEMOTRON_H = "nvidia/NVIDIA-Nemotron-Nano-9B-v2"  # hybrid_override_pattern, mixer.k_proj
FALCON_H1 = "tiiuae/Falcon-H1-0.5B-Instruct"  # attention and Mamba side by side in every layer
LLAMA4_SCOUT = "unsloth/Llama-4-Scout-17B-16E-Instruct"  # 3 of 4 layers chunked attention
QWEN3_NEXT = "Qwen/Qwen3-Next-80B-A3B-Instruct"  # config.json only: every 4th layer attention
# Gated DeltaNet linear attention in 18 of 24 layers, a vision tower and an MTP block.
QWEN35 = "Qwen/Qwen3.5-0.8B"

# Lines as `nvidia-smi --query-gpu=name,memory.total,memory.free,compute_cap` prints them.
RTX_3090 = "NVIDIA GeForce RTX 3090, 24576, 24000, 8.6"
RTX_3090_BUSY = "NVIDIA GeForce RTX 3090, 24576, 23000, 8.6"
RTX_3060 = "NVIDIA GeForce RTX 3060, 12288, 12000, 8.6"
RTX_4090 = "NVIDIA GeForce RTX 4090, 24564, 24000, 8.9"
# Memory bandwidth from NVIDIA's specifications, GB/s.
SPEC_BANDWIDTH = {
    "NVIDIA GeForce RTX 3090": 936.0,
    "NVIDIA GeForce RTX 3060": 360.0,
    "NVIDIA GeForce RTX 4090": 1008.0,
    "NVIDIA GeForce RTX 4060 Laptop GPU": 272.0,
}


def nvidia_pc(
    *lines: str, ram_gb: float = 60, cuda_free_mib: float | None = None, windows: bool = False
):
    """A PC with these NVIDIA cards, as QuantPilot's own probe reads them: Linux, or Windows
    (WDDM), with CUDA's own free figure when given."""
    from dataclasses import replace

    from ocq.hardware import parse_nvidia_smi
    from ocq.types import HardwareProfile

    cards = parse_nvidia_smi("\n".join(lines))
    cuda_free = round(cuda_free_mib / 1024, 2) if cuda_free_mib is not None else None
    cards = tuple(
        replace(card, bandwidth_gbs=SPEC_BANDWIDTH[card.name], cuda_free_gb=cuda_free, wddm=windows)
        for card in cards
    )
    return HardwareProfile(
        total_ram_gb=ram_gb + 4,
        available_ram_gb=ram_gb,
        cpu_cores=16,
        cpu_name="CPU",
        has_gpu=True,
        backend="CUDA",
        unified_memory=False,
        ram_bandwidth_gbs=80.0,
        gpus=cards,
        os_name="Windows" if windows else "Linux",
    )


@pytest.fixture
def saved_hub(monkeypatch):
    """The Hub, answered from data/: saved repos only, anything else is not found."""

    def model(repo):
        try:
            return real_model(repo)
        except FileNotFoundError:
            raise OcqError(f"Model {repo!r} was not found on HuggingFace.") from None

    monkeypatch.setattr(model_analyzer, "fetch_hf_model", lambda repo: model(repo)[0])
    monkeypatch.setattr(model_analyzer, "fetch_model_files", lambda repo, sha: model(repo)[1:])

    def no_llmfit(*args):
        raise AssertionError("the engine must not ask llmfit")

    monkeypatch.setattr(model_analyzer, "catalog_entry", no_llmfit)
