"""Reading GGUF headers. Checked against real files too: for every GGUF downloaded while
building this (Gemma 2/3, OLMoE, Granite, Jamba, Falcon-H1, LFM2), the header plus the tensor
data added up to the exact file size."""

import struct

import pytest

from ocq.gguf_file import parse_header, quant_from_filename, read_local
from ocq.types import OcqError


def gguf_bytes(settings: dict, tensors: list, alignment: int = 32) -> bytes:
    """A GGUF file in the real format: header, then zero-filled tensor data."""

    def string(text):
        data = text.encode()
        return struct.pack("<Q", len(data)) + data

    def value(item):
        if isinstance(item, str):
            return struct.pack("<I", 8) + string(item)
        if isinstance(item, list):
            return struct.pack("<IIQ", 9, 4, len(item)) + b"".join(
                struct.pack("<I", x) for x in item
            )
        return struct.pack("<II", 4, item)

    out = b"GGUF" + struct.pack("<IQQ", 3, len(tensors), len(settings))
    for key, item in settings.items():
        out += string(key) + value(item)
    offset, sizes = 0, []
    for name, shape, code, size in tensors:
        out += string(name) + struct.pack("<I", len(shape))
        out += b"".join(struct.pack("<Q", d) for d in shape) + struct.pack("<IQ", code, offset)
        sizes.append(size)
        offset += -(-size // alignment) * alignment
    out += b"\0" * (-len(out) % alignment)
    return out + b"\0" * offset


def fetch_from(data: bytes):
    return lambda start, length: data[start : start + length]


def test_header_settings_and_tensor_table_are_read():
    data = gguf_bytes(
        {"general.architecture": "llama", "llama.block_count": 2, "llama.head_counts": [4, 0]},
        [
            ("token_embd.weight", (64, 10), 1, 64 * 10 * 2),
            ("blk.0.attn_k.weight", (64, 32), 8, 64 * 32 // 32 * 34),
        ],
    )
    header = parse_header(fetch_from(data))
    assert header.architecture == "llama"
    assert header.get("block_count") == 2
    assert header.get("head_counts") == [4, 0]
    assert [t.type_name for t in header.tensors] == ["F16", "Q8_0"]
    assert header.tensors[1].nbytes == 64 * 32 // 32 * 34
    # The tensor table adds up to the file: header, aligned data.
    assert header.data_start + header.data_bytes == len(data)


def test_reading_a_local_file(tmp_path):
    path = tmp_path / "model-Q8_0.gguf"
    path.write_bytes(gguf_bytes({"general.architecture": "qwen2", "general.file_type": 7}, []))
    header = read_local(path)
    assert header.file_type == "Q8_0"


def test_not_a_gguf_file_is_refused():
    with pytest.raises(OcqError, match="Not a GGUF file"):
        parse_header(fetch_from(b"PK\x03\x04 not gguf"))


@pytest.mark.parametrize(
    ("name", "quant"),
    [
        ("Qwen2.5-7B-Instruct-Q4_K_M.gguf", "Q4_K_M"),
        ("olmoe-1b-7b-0924-instruct-q2_k.gguf", "Q2_K"),
        ("granite-4.0-h-350m-UD-IQ3_XXS.gguf", "UD-IQ3_XXS"),
        ("gpt-oss-20b-MXFP4.gguf", "MXFP4"),
        ("model-F16.gguf", "F16"),
        ("Llama-3.3-70B-Q4_K_M-00001-of-00002.gguf", "Q4_K_M"),
        ("mmproj-model.gguf", ""),
    ],
)
def test_quant_type_is_read_from_the_file_name(name, quant):
    assert quant_from_filename(name) == quant


# Real GGUF headers (vocabularies left out), from files downloaded while building this.

import gzip  # noqa: E402
import json  # noqa: E402
from dataclasses import replace  # noqa: E402
from pathlib import Path  # noqa: E402

from ocq.gguf_file import (  # noqa: E402
    GgufHeader,
    GgufTensor,
    config_from_gguf,
    hf_style_tensors,
)
from ocq.model_analyzer import (  # noqa: E402
    build_snapshot,
    experts_to_ram,
    gguf_variants,
    model_header,
    pick_gguf_file,
)
from ocq.types import FactKind  # noqa: E402

GGUF_DATA = Path(__file__).parent / "data" / "gguf"


def real_header(stem: str) -> tuple[GgufHeader, int]:
    saved = json.loads(gzip.decompress((GGUF_DATA / f"{stem}.json.gz").read_bytes()))
    tensors = tuple(GgufTensor(n, tuple(s), t, o) for n, s, t, o in saved["tensors"])
    header = GgufHeader(saved["version"], saved["settings"], tensors, saved["data_start"])
    return header, saved["file_size"]


@pytest.mark.parametrize(
    "stem",
    [
        "gemma-3-1b-it-Q2_K",
        "gemma-2-2b-it-IQ3_M",
        "olmoe-1b-7b-0924-instruct-q2_k",
        "granite-3.1-1b-a400m-instruct-Q2_K",
        "granite-4.0-h-350m-Q2_K",
        "jamba-reasoning-3b-Q4_K_M",
        "Falcon-H1-0.5B-Instruct-IQ1_M",
        "LFM2-350M-Q4_0",
        "Qwen3.5-0.8B-Q8_0",
    ],
)
def test_tensor_table_adds_up_to_the_file_size(stem):
    header, file_size = real_header(stem)
    assert header.data_start + header.data_bytes == file_size


@pytest.mark.parametrize(
    ("stem", "context", "llama_cpp_mib"),
    [
        # llama.cpp's log for these files, 1 conversation: KV cache (+ recurrent state).
        ("gemma-3-1b-it-Q2_K", 4096, 38.00),  # Gemma 3: every 6th layer full, window 512
        ("gemma-3-1b-it-Q2_K", 32768, 150.00),
        ("gemma-2-2b-it-IQ3_M", 8192, 650.00),  # Gemma 2: every 2nd layer full
        ("LFM2-350M-Q4_0", 4096, 48.00 + 0.078125),  # head_count_kv per layer, 0 = convolution
        ("granite-4.0-h-350m-Q2_K", 4096, 16.00 + 21.57421875),  # Mamba-2 layers
        ("Falcon-H1-0.5B-Instruct-IQ1_M", 4096, 72.00 + 27.73828125),  # Mamba-2 in every layer
        ("jamba-reasoning-3b-Q4_K_M", 4096, 4.00 + 9.6484375),  # Mamba-1 layers
    ],
)
def test_context_memory_from_a_gguf_header_matches_llama_cpp(stem, context, llama_cpp_mib):
    header, _ = real_header(stem)
    snap = build_snapshot(stem, {}, config_from_gguf(header), hf_style_tensors(header), None)
    assert snap.facts.kv_cache.kind == FactKind.CROSS_CHECKED  # settings vs tensor shapes
    assert snap.kv_cache_bytes(context, conversations=1) == pytest.approx(
        llama_cpp_mib * 2**20, abs=1
    )


@pytest.mark.parametrize(
    ("stem", "used_per_token"),
    [
        # The same as from the models' safetensors files.
        ("olmoe-1b-7b-0924-instruct-q2_k", 1_178_994_688),
        ("granite-3.1-1b-a400m-instruct-Q2_K", 428_658_688),
    ],
)
def test_experts_are_read_from_a_gguf_file(stem, used_per_token):
    header, _ = real_header(stem)
    snap = build_snapshot(stem, {}, config_from_gguf(header), hf_style_tensors(header), None)
    assert snap.is_moe
    assert snap.facts.active_parameters.value == used_per_token


def test_gguf_variants_add_up_split_parts_and_skip_vision_projectors():
    files = [
        ("Model-Q4_K_M-00001-of-00002.gguf", 5_000_000),
        ("Model-Q4_K_M-00002-of-00002.gguf", 3_000_000),
        ("Model-Q8_0.gguf", 9_000_000),
        ("mmproj-Model-F16.gguf", 600_000),
    ]
    # Largest first; each loses the header bytes once per file.
    assert gguf_variants(files, header_bytes=1000) == [("Q8_0", 8_999_000), ("Q4_K_M", 7_998_000)]
    assert pick_gguf_file([name for name, _ in files]) == "Model-Q4_K_M-00001-of-00002.gguf"


def test_draft_models_next_to_a_model_are_left_out():
    # ggml-org/gpt-oss-20b-GGUF: the EAGLE3 draft files sort first, their header names them.
    names = [
        "eagle3-gpt-oss-20b-BF16.gguf",
        "eagle3-gpt-oss-20b-Q8_0.gguf",
        "gpt-oss-20b-MXFP4.gguf",
    ]
    archs = {"eagle3-gpt-oss-20b-BF16.gguf": "eagle3", "gpt-oss-20b-MXFP4.gguf": "gpt-oss"}
    read = []

    def header(name):
        read.append(name)
        return GgufHeader(3, {"general.architecture": archs[name]}, (), 0)

    chosen, found, kept = model_header(names, header)
    assert chosen == "gpt-oss-20b-MXFP4.gguf" and found.architecture == "gpt-oss"
    assert kept == ["gpt-oss-20b-MXFP4.gguf"]
    assert read == ["eagle3-gpt-oss-20b-BF16.gguf", "gpt-oss-20b-MXFP4.gguf"]


def test_expert_weights_and_biases_of_a_gguf_are_counted_once():
    # gpt-oss stores a bias next to each expert tensor (ffn_down_exps.weight and .bias). Both
    # were named mlp.experts.down_proj, so every layer's experts were counted twice and
    # --n-cpu-moe came out too small: 6 where the RTX 4060 (7.5 GB free) needed 10.
    header, _ = real_header("gpt-oss-20b-MXFP4")
    tensors = hf_style_tensors(header)
    assert len({name for name, _ in tensors}) == len(tensors)

    snap = build_snapshot("gpt-oss", {}, config_from_gguf(header), tensors, None)
    sizes = {name: t.nbytes for (name, _), t in zip(tensors, header.tensors, strict=True)}
    snap = replace(snap, gguf_tensor_bytes=sizes)
    layers, moved_gb = experts_to_ram(snap, "MXFP4", too_much_gb=2.3, ram_gb=100)
    assert layers == 6
    # llama.cpp b11429 on CUDA with --n-cpu-moe 6: "CUDA0 model buffer size = 8522.54 MiB",
    # everything but the input embeddings and the experts of the first 6 layers.
    embeddings = next(t.nbytes for t in header.tensors if t.name == "token_embd.weight")
    on_gpu = sum(sizes.values()) - embeddings - moved_gb * 2**30
    assert on_gpu / 2**20 == pytest.approx(8522.54, abs=0.03)


def test_feasibility_of_a_gguf_repo_gives_the_n_cpu_moe_flag(monkeypatch):
    # ocq plan refuses a GGUF repo ("already GGUF"), so feasibility has to give the N itself.
    from conftest import nvidia_pc
    from typer.testing import CliRunner

    from ocq import cli
    from ocq.model_analyzer import gguf_snapshot

    header, file_size = real_header("gpt-oss-20b-MXFP4")
    laptop = nvidia_pc("NVIDIA GeForce RTX 4060 Laptop GPU, 8188, 7610, 8.9")
    files = [("MXFP4", header.data_bytes)]
    snap = gguf_snapshot("ggml-org/gpt-oss-20b-GGUF", {}, header, files, file_size, True)
    monkeypatch.setattr(cli, "detect_hardware", lambda _bin: laptop)
    monkeypatch.setattr(cli, "resolve_model", lambda *a, **k: snap)
    result = CliRunner().invoke(cli.app, ["feasibility", snap.model_id], env={"COLUMNS": "200"})
    assert result.exit_code == 0, result.output
    assert "MXFP4: --n-cpu-moe" in result.output
    assert "see ocq plan" not in result.output


# llama-server b11429 on an RTX 4060 Laptop, 4 conversations: GPU compute buffer + output
# buffer, MiB, at 4096 and 16384 tokens, from llama-server's start-up logs.
LLAMA_SERVER_CUDA = {
    "Falcon-H1-0.5B-Instruct-IQ1_M": (31.64 + 0.50, 43.64 + 0.50),
    "gemma-2-2b-it-IQ3_M": (80.02 + 3.91, 104.02 + 3.91),
    "gemma-3-1b-it-Q2_K": (56.02 + 4.00, 68.02 + 4.00),
    "gpt-oss-20b-MXFP4": (206.99 + 3.07, 218.99 + 3.07),
    "granite-3.1-1b-a400m-instruct-Q2_K": (44.01 + 0.75, 56.01 + 0.75),
    "granite-4.0-h-350m-Q2_K": (29.69 + 1.53, 41.69 + 1.53),
    "jamba-reasoning-3b-Q4_K_M": (76.25 + 1.00, 88.25 + 1.00),
    "LFM2-350M-Q4_0": (39.01 + 1.00, 51.01 + 1.00),
    "olmoe-1b-7b-0924-instruct-q2_k": (84.01 + 0.77, 96.01 + 0.77),
}


@pytest.mark.parametrize("stem", sorted(LLAMA_SERVER_CUDA))
def test_llama_server_working_memory_is_an_upper_bound_of_the_measured(stem):
    from ocq.model_analyzer import working_bytes
    from ocq.types import LLAMA_CPP

    header, _ = real_header(stem)
    snap = build_snapshot(stem, {}, config_from_gguf(header), hf_style_tensors(header), None)
    for context, measured in zip((4096, 16384), LLAMA_SERVER_CUDA[stem], strict=True):
        predicted = working_bytes(snap, context, LLAMA_CPP) / 2**20
        assert measured <= predicted <= measured + 91, (context, predicted, measured)


# llama-server b11429, compute + output buffer in MiB. Attention copies one layer's keys and
# values at 16 bits on CUDA with a q8_0 cache, on CUDA when that layer's cache is in RAM, and
# on Metal always; it sets the peak when it is larger than the feed-forward one.
# CUDA: RTX 4060 Laptop, from llama-server's start-up logs.
# Metal: M1 16 GB. "deepseek-v2-lite-2-layers" is DeepSeek-V2-Lite's shapes in 2 layers with
# random weights (MLA, 64 experts; q_lora_rank 1536, which llama.cpp needs below 26 layers).
CUDA, CUDA_LAYERS_IN_RAM, METAL = "cuda", "cuda, layers in ram", "metal"
LLAMA_SERVER_ATTENTION = [
    # Attention beside a Mamba block in every layer: the copy adds to the peak.
    ("Falcon-H1-0.5B-Instruct-IQ1_M", 4096, "q8_0", CUDA, 35.67 + 0.50),
    ("Falcon-H1-0.5B-Instruct-IQ1_M", 16384, "q8_0", CUDA, 52.67 + 0.50),
    ("Falcon-H1-0.5B-Instruct-IQ1_M", 65536, "q8_0", CUDA, 124.67 + 0.50),
    ("gemma-2-2b-it-IQ3_M", 8192, "q8_0", CUDA, 88.55 + 3.91),
    ("gemma-3-1b-it-Q2_K", 16384, "q8_0", CUDA, 68.55 + 4.00),
    ("gemma-3-1b-it-Q2_K", 32768, "q8_0", CUDA, 84.55 + 4.00),
    ("gemma-3-1b-it-Q2_K", 32768, "f16", CUDA, 84.02 + 4.00),
    ("olmoe-1b-7b-0924-instruct-q2_k", 32768, "f16", CUDA, 112.01 + 0.77),
    ("olmoe-1b-7b-0924-instruct-q2_k", 16384, "q8_0", CUDA, 164.09 + 0.77),
    ("olmoe-1b-7b-0924-instruct-q2_k", 32768, "q8_0", CUDA, 308.09 + 0.77),
    ("olmoe-1b-7b-0924-instruct-q2_k", 65536, "q8_0", CUDA, 596.09 + 0.77),
    ("qwen1_5-0_5b-chat-q2_k", 8192, "f16", CUDA, 32.51 + 2.32),
    ("qwen1_5-0_5b-chat-q2_k", 32768, "f16", CUDA, 56.51 + 2.32),
    ("qwen1_5-0_5b-chat-q2_k", 16384, "q8_0", CUDA, 90.05 + 2.32),
    ("qwen1_5-0_5b-chat-q2_k", 32768, "q8_0", CUDA, 170.05 + 2.32),
    ("Qwen2.5-7B-Instruct-Q4_K_M", 32768, "q8_0", CUDA, 171.09 + 2.32),
    ("Qwen2.5-7B-Instruct-Q4_K_M", 65536, "f16", CUDA_LAYERS_IN_RAM, 271.00 + 2.32),
    ("Qwen2.5-7B-Instruct-Q4_K_M", 98304, "f16", CUDA_LAYERS_IN_RAM, 330.00 + 2.32),
    ("Qwen2.5-7B-Instruct-Q4_K_M", 131072, "f16", CUDA_LAYERS_IN_RAM, 426.00 + 2.32),
    ("qwen1_5-0_5b-chat-q2_k", 8192, "f16", METAL, 58.21 + 2.32),
    ("qwen1_5-0_5b-chat-q2_k", 8192, "q8_0", METAL, 58.49 + 2.32),
    ("qwen1_5-0_5b-chat-q2_k", 16384, "f16", METAL, 98.22 + 2.32),
    ("qwen1_5-0_5b-chat-q2_k", 32768, "f16", METAL, 178.23 + 2.32),
    ("deepseek-v2-lite-2-layers-q8_0", 4096, "f16", METAL, 115.84 + 1.56),
    ("deepseek-v2-lite-2-layers-q8_0", 16384, "f16", METAL, 141.35 + 1.56),
    ("deepseek-v2-lite-2-layers-q8_0", 65536, "f16", METAL, 243.40 + 1.56),
]


@pytest.mark.parametrize(("stem", "context", "cache", "where", "measured"), LLAMA_SERVER_ATTENTION)
def test_llama_server_working_memory_with_attentions_copy(stem, context, cache, where, measured):
    from ocq.model_analyzer import kv_copies, working_bytes
    from ocq.model_size import kv_value_bytes
    from ocq.types import LLAMA_CPP

    header, _ = real_header(stem)
    snap = build_snapshot(stem, {}, config_from_gguf(header), hf_style_tensors(header), None)
    metal = where == METAL
    copies = kv_copies(kv_value_bytes(cache, LLAMA_CPP), not metal, metal, where != CUDA)
    predicted = working_bytes(snap, context, LLAMA_CPP, copies, metal) / 2**20
    assert measured <= predicted, (predicted, measured)
    if stem.startswith(("olmoe", "qwen1_5", "Falcon-H1")) and copies:
        # The copy sets the peak: 8 rows of activations are counted, llama.cpp took 5-7.
        assert predicted <= measured + 12, (predicted, measured)


def test_olmoe_at_64k_with_a_q8_0_cache_keeps_3_layers_of_experts_in_ram():
    # The old plan, --n-cpu-moe 1, overfilled the RTX 4060 Laptop: 2,271 MiB of weights +
    # 4,352 cache + 596 compute with 7,106 free (llama.cpp's log). Each layer of
    # experts is 139 MiB: 2 leaves 26 MiB, 3 leaves 165 MiB.
    from conftest import nvidia_pc

    from ocq.model_analyzer import gguf_snapshot, quant_options
    from ocq.types import RunMode

    stem = "olmoe-1b-7b-0924-instruct-q2_k"
    header, file_size = real_header(stem)
    snap = gguf_snapshot(stem, {}, header, [("Q2_K", header.data_bytes)], file_size, True)
    laptop = nvidia_pc("NVIDIA GeForce RTX 4060 Laptop GPU, 8188, 7610, 8.9", ram_gb=16)
    (row,) = quant_options(snap, laptop, 65536, "q8_0")
    assert row.run_mode == RunMode.GPU_EXPERTS_RAM
    assert row.experts_in_ram == 3


def test_gpt_oss_on_the_rtx_4060_keeps_11_layers_of_experts_in_ram():
    # 7,610 MiB free, 7,106 once llama.cpp has started CUDA. GPU side: the weights but the
    # input embeddings (in RAM) and the experts of N layers, the cache, llama-server's buffers.
    # --n-cpu-moe 6 (the old plan) put 8,522 MiB of weights alone on the card.
    from conftest import nvidia_pc

    from ocq.model_analyzer import gguf_snapshot, quant_options
    from ocq.types import RunMode

    header, file_size = real_header("gpt-oss-20b-MXFP4")
    laptop = nvidia_pc("NVIDIA GeForce RTX 4060 Laptop GPU, 8188, 7610, 8.9")
    files = [("MXFP4", header.data_bytes)]
    snap = gguf_snapshot("ggml-org/gpt-oss-20b-GGUF", {}, header, files, file_size, True)
    (row,) = quant_options(snap, laptop)
    assert row.run_mode == RunMode.GPU_EXPERTS_RAM
    assert row.experts_in_ram == 11


def test_llama_cpp_fills_the_gpu_with_whole_layers_from_the_back():
    # Qwen2.5-7B Q4_K_M at 131k on the RTX 4060 Laptop, llama-completion b11429 with --fit
    # (data/gguf/llama-completion-qwen2.5-7b-131k-cuda.log): 15 layers would take
    # 6,324 MiB of its 6,082 MiB target, so "offloaded 14/29 layers to GPU", the output layer
    # and blocks 15-27: "CUDA0 model buffer size = 2172.17 MiB". QuantPilot shared the GPU out
    # evenly before and said 14.95 layers.
    from conftest import nvidia_pc

    from ocq.model_analyzer import gguf_snapshot, quant_options
    from ocq.types import RunMode

    header, file_size = real_header("Qwen2.5-7B-Instruct-Q4_K_M")
    assert header.data_start + header.data_bytes == file_size
    gpu = [t for t in header.tensors if not t.name.startswith(("blk.", "token_embd"))]
    last_13 = tuple(f"blk.{i}." for i in range(15, 28))
    gpu += [t for t in header.tensors if t.name.startswith(last_13)]
    assert sum(t.nbytes for t in gpu) / 2**20 == pytest.approx(2172.17, abs=0.01)

    laptop = nvidia_pc("NVIDIA GeForce RTX 4060 Laptop GPU, 8188, 7610, 8.9", ram_gb=16)
    files = [("Q4_K_M", header.data_bytes)]
    snap = gguf_snapshot("bartowski/Qwen2.5-7B-Instruct-GGUF", {}, header, files, file_size, True)
    (row,) = quant_options(snap, laptop, 131072)
    assert row.run_mode == RunMode.GPU_OFFLOAD
    assert row.gpu_layers == 14
    # 13.72 tok/s measured at 14 layers; the bound reads 14/29 of the weights from the GPU.
    assert 13.72 < row.estimated_tps < 30


def test_a_mixture_of_experts_too_big_even_without_experts_fills_dense_layers():
    # OLMoE Q2_K at 64k on the RTX 4060 Laptop: llama.cpp b11429 put 11/17 layers on the GPU,
    # the experts of all but the lowest of them in RAM
    # (data/gguf/llama-completion-olmoe-1b-7b-0924-instruct-q2_k-64k-cuda.log).
    from conftest import nvidia_pc

    from ocq.model_analyzer import gguf_snapshot, quant_options
    from ocq.types import RunMode

    stem = "olmoe-1b-7b-0924-instruct-q2_k"
    header, file_size = real_header(stem)
    snap = gguf_snapshot(stem, {}, header, [("Q2_K", header.data_bytes)], file_size, True)
    laptop = nvidia_pc("NVIDIA GeForce RTX 4060 Laptop GPU, 8188, 7610, 8.9", ram_gb=16)
    (row,) = quant_options(snap, laptop, 65536)
    assert row.run_mode == RunMode.GPU_OFFLOAD
    assert row.gpu_layers == 11


@pytest.mark.parametrize(
    ("stem", "per_token"),
    [
        # Converted before llama.cpp split attn_kv_b (no key_length_mla): llama.cpp runs it as
        # plain attention, 16 heads x (192 + 128) values of 2 bytes in each of 27 layers.
        ("DeepSeek-V2-Lite-Chat.Q2_K", 27 * 16 * (192 + 128) * 2),
        # attn_k_b / attn_v_b and key_length_mla: the compressed cache, 512 + 64 values.
        ("Moonlight-16B-A3B-Instruct-Q2_K", 27 * (512 + 64) * 2),
        # Converted with b11429 (2 layers): llama-server logged 144.00 MiB at 64k, 4 slots.
        ("deepseek-v2-lite-2-layers-q8_0", 2 * (512 + 64) * 2),
    ],
)
def test_an_mla_gguf_is_compressed_only_when_llama_cpp_runs_it_so(stem, per_token):
    from ocq.types import LLAMA_CPP

    header, _ = real_header(stem)
    snap = build_snapshot(stem, {}, config_from_gguf(header), hf_style_tensors(header), None)
    assert snap.kv_cache_bytes(1, LLAMA_CPP, 1) == per_token


def test_granite_moe_with_a_quantized_cache_names_llama_cpps_workaround():
    from conftest import nvidia_pc

    from ocq.model_analyzer import gguf_snapshot, quant_options

    stem = "granite-3.1-1b-a400m-instruct-Q2_K"
    header, file_size = real_header(stem)
    snap = gguf_snapshot(stem, {}, header, [("Q2_K", header.data_bytes)], file_size, True)
    laptop = nvidia_pc("NVIDIA GeForce RTX 4060 Laptop GPU, 8188, 7610, 8.9")
    (q8,) = quant_options(snap, laptop, 16384, "q8_0")
    assert "LLAMA_ATTN_ROT_DISABLE=1" in q8.note
    (f16,) = quant_options(snap, laptop, 16384, "f16")
    assert not f16.note


def test_an_expert_plan_on_windows_has_what_is_really_free():
    # gpt-oss-20b at 4k on the RTX 4060 Laptop under Windows: nvidia-smi 7,792 MiB
    # free, CUDA's fixed figure 7,106. llama-bench: --n-cpu-moe 9 (7,309 MiB of weights) was the
    # fastest, 8 (7,713) over nvidia-smi's free memory fell to a third of the prompt speed.
    from conftest import nvidia_pc

    from ocq.model_analyzer import gguf_snapshot, quant_options

    header, file_size = real_header("gpt-oss-20b-MXFP4")
    files = [("MXFP4", header.data_bytes)]
    snap = gguf_snapshot("ggml-org/gpt-oss-20b-GGUF", {}, header, files, file_size, True)
    card = "NVIDIA GeForce RTX 4060 Laptop GPU, 8188, 7792, 8.9"
    windows = nvidia_pc(card, cuda_free_mib=7106, windows=True)
    (row,) = quant_options(snap, windows)
    assert row.experts_in_ram == 9
    # On Linux CUDA's figure is what is left once the context exists: the plan keeps to it.
    linux = nvidia_pc(card, cuda_free_mib=7106)
    (row,) = quant_options(snap, linux)
    assert row.experts_in_ram == 11


def test_llama_cpps_fit_works_from_cudas_own_free_figure():
    # Qwen2.5-7B at 131k on the RTX 4060 Laptop: llama.cpp's --fit target was 7,106 - 1,024 =
    # 6,082 MiB whatever nvidia-smi said, and it put 14 of 29 layers on the GPU.
    from conftest import nvidia_pc

    from ocq.model_analyzer import gguf_snapshot, quant_options

    header, file_size = real_header("Qwen2.5-7B-Instruct-Q4_K_M")
    files = [("Q4_K_M", header.data_bytes)]
    snap = gguf_snapshot("bartowski/Qwen2.5-7B-Instruct-GGUF", {}, header, files, file_size, True)
    for smi_free in (7420, 7792):
        card = f"NVIDIA GeForce RTX 4060 Laptop GPU, 8188, {smi_free}, 8.9"
        laptop = nvidia_pc(card, ram_gb=16, cuda_free_mib=7106, windows=True)
        (row,) = quant_options(snap, laptop, 131072)
        assert row.gpu_layers == 14


def test_qwen35_layers_follow_its_recurrent_layer_list_without_the_prediction_block():
    header, _ = real_header("Qwen3.5-0.8B-Q8_0")
    assert (header.get("block_count"), header.get("nextn_predict_layers")) == (25, 1)
    config = config_from_gguf(header)
    assert config["num_hidden_layers"] == header.layers == 24  # llama.cpp's n_layer
    kinds = config["layer_types"]
    assert kinds.count("linear_attention") == 18 and kinds[3::4] == ["full_attention"] * 6
    # llama.cpp does not load block 24: 15 tensors, 20.74 MiB in its log.
    dropped = [t for t in header.tensors if t not in header.loaded().tensors]
    assert len(dropped) == 15 and {t.name.split(".")[1] for t in dropped} == {"24"}
    assert round(sum(t.nbytes for t in dropped) / 2**20, 2) == 20.74
