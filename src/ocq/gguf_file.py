"""Read a GGUF file's header: its settings and its tensor table, without reading the weights.

A GGUF file (llama.cpp's format, also what Ollama and LM Studio store) starts with:
  "GGUF", version (u32), tensor count (u64), setting count (u64)
  settings: key (string), value type (u32), value
  tensors:  name (string), dimension count (u32), dimensions (u64 each), type (u32), offset (u64)
then the weights, aligned to general.alignment (32 bytes unless it says otherwise). Strings are
a u64 length and UTF-8 bytes. Everything is little-endian.

The header can be read from a local file or, with HTTP range requests, from a file on the
Hugging Face Hub without downloading the weights.
"""

from __future__ import annotations

import math
import os
import re
import struct
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from ocq.types import OcqError

MAGIC = b"GGUF"

# Value types of the settings: struct format and size.
SCALARS = {
    0: ("<B", 1),  # uint8
    1: ("<b", 1),  # int8
    2: ("<H", 2),  # uint16
    3: ("<h", 2),  # int16
    4: ("<I", 4),  # uint32
    5: ("<i", 4),  # int32
    6: ("<f", 4),  # float32
    7: ("<?", 1),  # bool
    10: ("<Q", 8),  # uint64
    11: ("<q", 8),  # int64
    12: ("<d", 8),  # float64
}
STRING, ARRAY = 8, 9

# GGML tensor types: name, bytes per block, values per block (ggml's type_traits).
TENSOR_TYPES = {
    0: ("F32", 4, 1),
    1: ("F16", 2, 1),
    2: ("Q4_0", 18, 32),
    3: ("Q4_1", 20, 32),
    6: ("Q5_0", 22, 32),
    7: ("Q5_1", 24, 32),
    8: ("Q8_0", 34, 32),
    9: ("Q8_1", 36, 32),
    10: ("Q2_K", 84, 256),
    11: ("Q3_K", 110, 256),
    12: ("Q4_K", 144, 256),
    13: ("Q5_K", 176, 256),
    14: ("Q6_K", 210, 256),
    15: ("Q8_K", 292, 256),
    16: ("IQ2_XXS", 66, 256),
    17: ("IQ2_XS", 74, 256),
    18: ("IQ3_XXS", 98, 256),
    19: ("IQ1_S", 50, 256),
    20: ("IQ4_NL", 18, 32),
    21: ("IQ3_S", 110, 256),
    22: ("IQ2_S", 82, 256),
    23: ("IQ4_XS", 136, 256),
    24: ("I8", 1, 1),
    25: ("I16", 2, 1),
    26: ("I32", 4, 1),
    27: ("I64", 8, 1),
    28: ("F64", 8, 1),
    29: ("IQ1_M", 56, 256),
    30: ("BF16", 2, 1),
    34: ("TQ1_0", 54, 256),
    35: ("TQ2_0", 66, 256),
    39: ("MXFP4", 17, 32),
}

# general.file_type: how the file as a whole was quantized (llama.cpp's llama_ftype).
FILE_TYPES = {
    0: "F32",
    1: "F16",
    2: "Q4_0",
    3: "Q4_1",
    7: "Q8_0",
    8: "Q5_0",
    9: "Q5_1",
    10: "Q2_K",
    11: "Q3_K_S",
    12: "Q3_K_M",
    13: "Q3_K_L",
    14: "Q4_K_S",
    15: "Q4_K_M",
    16: "Q5_K_S",
    17: "Q5_K_M",
    18: "Q6_K",
    19: "IQ2_XXS",
    20: "IQ2_XS",
    21: "Q2_K_S",
    22: "IQ3_XS",
    23: "IQ3_XXS",
    24: "IQ1_S",
    25: "IQ4_NL",
    26: "IQ3_S",
    27: "IQ3_M",
    28: "IQ2_S",
    29: "IQ2_M",
    30: "IQ4_XS",
    31: "IQ1_M",
    32: "BF16",
    36: "TQ1_0",
    37: "TQ2_0",
    38: "MXFP4_MOE",
}


@dataclass(frozen=True)
class GgufTensor:
    name: str
    shape: tuple[int, ...]  # as ggml lists it: the fastest-changing dimension first
    type_name: str
    offset: int  # from the start of the tensor data

    @property
    def elements(self) -> int:
        return math.prod(self.shape)

    @property
    def nbytes(self) -> int | None:
        """Bytes of the tensor's data. None for a type this reader does not know."""
        for name, block_bytes, block_values in TENSOR_TYPES.values():
            if name == self.type_name:
                return self.elements // block_values * block_bytes
        return None


@dataclass(frozen=True)
class GgufHeader:
    version: int
    settings: dict[str, Any] = field(repr=False)
    tensors: tuple[GgufTensor, ...] = field(repr=False)
    data_start: int  # where the tensor data begins (after the header, aligned)

    @property
    def architecture(self) -> str:
        return str(self.settings.get("general.architecture", ""))

    def get(self, key: str, default: Any = None) -> Any:
        """An architecture setting, for example get("block_count") for "llama.block_count"."""
        return self.settings.get(f"{self.architecture}.{key}", default)

    @property
    def file_type(self) -> str:
        code = self.settings.get("general.file_type")
        return FILE_TYPES.get(code, f"type {code}") if code is not None else ""

    @property
    def data_bytes(self) -> int | None:
        """Bytes of all tensor data, alignment included: where the last tensor ends."""
        if not self.tensors:
            return 0
        last = max(self.tensors, key=lambda tensor: tensor.offset)
        return None if last.nbytes is None else last.offset + last.nbytes


class Reader:
    """Reads a byte stream forward, fetching more of it when needed."""

    def __init__(self, fetch, chunk: int = 4 * 1024 * 1024):
        self.fetch = fetch  # fetch(start, length) -> bytes
        self.chunk = chunk
        self.buffer = b""
        self.position = 0

    def take(self, length: int) -> bytes:
        while len(self.buffer) < self.position + length:
            more = self.fetch(len(self.buffer), max(self.chunk, length))
            if not more:
                raise OcqError("The GGUF header ends early: the file is cut off or not GGUF.")
            self.buffer += more
            self.chunk *= 2  # tokenizer vocabularies can take several MB
        data = self.buffer[self.position : self.position + length]
        self.position += length
        return data

    def scalar(self, kind: int) -> Any:
        fmt, size = SCALARS[kind]
        return struct.unpack(fmt, self.take(size))[0]

    def string(self) -> str:
        length = self.scalar(10)
        return self.take(length).decode("utf-8", errors="replace")

    def value(self, kind: int) -> Any:
        if kind == STRING:
            return self.string()
        if kind == ARRAY:
            item_kind, count = self.scalar(4), self.scalar(10)
            if item_kind in SCALARS:
                fmt, size = SCALARS[item_kind]
                raw = self.take(size * count)
                return list(struct.unpack(f"<{count}{fmt[1]}", raw))
            return [self.value(item_kind) for _ in range(count)]
        if kind not in SCALARS:
            raise OcqError(f"Unknown GGUF value type {kind}.")
        return self.scalar(kind)


def parse_header(fetch) -> GgufHeader:
    reader = Reader(fetch)
    if reader.take(4) != MAGIC:
        raise OcqError("Not a GGUF file (it does not start with 'GGUF').")
    version = reader.scalar(4)
    if version not in (2, 3):
        raise OcqError(f"GGUF version {version} is not supported (2 and 3 are).")
    tensor_count, setting_count = reader.scalar(10), reader.scalar(10)

    settings = {}
    for _ in range(setting_count):
        key = reader.string()
        settings[key] = reader.value(reader.scalar(4))

    tensors = []
    for _ in range(tensor_count):
        name = reader.string()
        dims = reader.scalar(4)
        shape = tuple(reader.scalar(10) for _ in range(dims))
        code = reader.scalar(4)
        type_name = TENSOR_TYPES.get(code, (f"type {code}",))[0]
        tensors.append(GgufTensor(name, shape, type_name, reader.scalar(10)))

    alignment = settings.get("general.alignment", 32)
    data_start = -(-reader.position // alignment) * alignment
    return GgufHeader(version, settings, tuple(tensors), data_start)


def read_local(path: Path) -> GgufHeader:
    try:
        with open(path, "rb") as handle:

            def fetch(start: int, length: int) -> bytes:
                handle.seek(start)
                return handle.read(length)

            return parse_header(fetch)
    except OSError as exc:
        raise OcqError(f"Could not read {path}") from exc


def read_remote(url: str) -> GgufHeader:
    """The header of a file on the web, read with HTTP range requests: never the weights."""
    token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")

    def fetch(start: int, length: int) -> bytes:
        headers = {"Range": f"bytes={start}-{start + length - 1}", "User-Agent": "ocq/0.1"}
        if token:
            headers["Authorization"] = f"Bearer {token}"
        request = urllib.request.Request(url, headers=headers)
        try:
            with urllib.request.urlopen(request, timeout=60) as response:
                return response.read()
        except urllib.error.HTTPError as exc:
            if exc.code == 416:  # asked past the end of the file
                return b""
            raise OcqError(f"Could not read {url}: HTTP {exc.code}") from exc
        except urllib.error.URLError as exc:
            raise OcqError(f"Could not read {url}: {exc.reason}") from exc

    return parse_header(fetch)


QUANT_IN_NAME = re.compile(
    r"(?<![A-Za-z0-9])((?:UD-)?(?:I?Q\d_[A-Z0-9_]+|Q\d_\d|F16|BF16|F32|MXFP4(?:_MOE)?|TQ\d_\d))"
    r"(?=[.-])",
    re.IGNORECASE,
)
SPLIT_PART = re.compile(r"-(\d{5})-of-(\d{5})\.gguf$")


def quant_from_filename(name: str) -> str:
    """The quantization in a GGUF file name, for example Q4_K_M in model-Q4_K_M.gguf."""
    match = QUANT_IN_NAME.search(Path(name).name)
    return match.group(1).upper() if match else ""
