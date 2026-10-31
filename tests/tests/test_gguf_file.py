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
