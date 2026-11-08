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
# buffer, MiB, at 4096 and 16384 tokens (results/windows-nvidia-.../server/summary.md).
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
