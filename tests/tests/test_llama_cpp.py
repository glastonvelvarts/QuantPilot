"""Downloading the llama.cpp release."""

import io

import pytest

from ocq.quantizers import llama_cpp
from ocq.types import OcqError


class Response(io.BytesIO):
    def __init__(self, body: bytes, length: int):
        super().__init__(body)
        self.headers = {"Content-Length": str(length)}


def test_a_download_cut_off_early_is_an_error_not_a_broken_file(tmp_path, monkeypatch):
    monkeypatch.setattr(llama_cpp.urllib.request, "urlopen", lambda *a, **k: Response(b"PK12", 10))
    target = tmp_path / "llama.zip"
    with pytest.raises(OcqError, match="stopped after 4 of 10 bytes"):
        llama_cpp.download("https://example.com/llama.zip", target)
    assert list(tmp_path.iterdir()) == []


def test_a_whole_download_is_kept(tmp_path, monkeypatch):
    monkeypatch.setattr(llama_cpp.urllib.request, "urlopen", lambda *a, **k: Response(b"PK12", 4))
    target = llama_cpp.download("https://example.com/llama.zip", tmp_path / "llama.zip")
    assert target.read_bytes() == b"PK12"
