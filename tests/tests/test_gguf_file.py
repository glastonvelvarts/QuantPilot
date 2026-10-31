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
from pathlib import Path  # noqa: E402

from ocq.gguf_file import (  # noqa: E402
    GgufHeader,
    GgufTensor,
    config_from_gguf,
    hf_style_tensors,
)
from ocq.model_analyzer import build_snapshot, gguf_variants, pick_gguf_file  # noqa: E402
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
