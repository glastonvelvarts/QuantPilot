"""Downloading the llama.cpp release: a cut-off download is resumed, and never kept broken."""

import io
import urllib.error

import pytest

from ocq.quantizers import llama_cpp
from ocq.types import OcqError

WHOLE = b"PK0123456789"  # 12 bytes
URL = "https://example.com/llama.zip"


class Response(io.BytesIO):
    def __init__(self, body: bytes, status: int = 200, headers: dict | None = None):
        super().__init__(body)
        self.status = status
        self.headers = headers or {}


def full(body: bytes = WHOLE, sent: bytes | None = None) -> Response:
    """The whole file announced; `sent` is what arrives before the connection closes."""
    return Response(body if sent is None else sent, headers={"Content-Length": str(len(body))})


def rest(start: int, body: bytes = WHOLE) -> Response:
    """What a server that resumes sends for "Range: bytes=start-"."""
    return Response(
        body[start:],
        status=206,
        headers={"Content-Range": f"bytes {start}-{len(body) - 1}/{len(body)}"},
    )


@pytest.fixture
def server(monkeypatch):
    """Answers each request with the next of `server.answers` (a response or an error) and
    keeps the Range header each request asked with."""

    class Server:
        answers: list = []
        ranges: list = []

    def urlopen(request, timeout):
        Server.ranges.append(request.get_header("Range"))
        answer = Server.answers.pop(0)
        if isinstance(answer, Exception):
            raise answer
        return answer

    monkeypatch.setattr(llama_cpp.urllib.request, "urlopen", urlopen)
    return Server


def test_a_whole_download_is_kept(tmp_path, server):
    server.answers = [full()]
    target = llama_cpp.download(URL, tmp_path / "llama.zip")
    assert target.read_bytes() == WHOLE and server.ranges == [None]


def test_a_cut_off_download_asks_for_the_rest(tmp_path, server, capsys):
    server.answers = [full(sent=WHOLE[:4]), rest(4)]
    target = llama_cpp.download(URL, tmp_path / "llama.zip")
    assert target.read_bytes() == WHOLE
    assert server.ranges == [None, "bytes=4-"]
    assert "Download cut off at 4 bytes of 12 bytes, resuming (try 2 of 3)" in (
        capsys.readouterr().out
    )
    assert sorted(p.name for p in tmp_path.iterdir()) == ["llama.zip"]  # no .part left


def test_a_server_that_cannot_resume_sends_it_all_again(tmp_path, server):
    server.answers = [full(sent=WHOLE[:4]), full()]  # answers the Range with the whole file
    target = llama_cpp.download(URL, tmp_path / "llama.zip")
    assert target.read_bytes() == WHOLE  # started over, not appended


def test_a_dropped_connection_is_tried_again(tmp_path, server):
    server.answers = [urllib.error.URLError("connection reset"), full()]
    assert llama_cpp.download(URL, tmp_path / "llama.zip").read_bytes() == WHOLE


def test_cut_off_every_time_is_an_error_not_a_broken_file(tmp_path, server):
    server.answers = [full(sent=WHOLE[:4]) for _ in range(llama_cpp.DOWNLOAD_TRIES)]
    with pytest.raises(OcqError, match=r"stopped after 4 of 12 bytes, 3 times\. Check the inter"):
        llama_cpp.download(URL, tmp_path / "llama.zip")
    assert list(tmp_path.iterdir()) == [] and len(server.ranges) == 3


def test_a_missing_file_is_not_asked_for_again(tmp_path, server):
    server.answers = [urllib.error.HTTPError(URL, 404, "Not Found", {}, None)]
    with pytest.raises(OcqError, match="404"):
        llama_cpp.download(URL, tmp_path / "llama.zip")
    assert len(server.ranges) == 1


def test_no_connection_at_all_says_so_after_every_try(tmp_path, server):
    server.answers = [urllib.error.URLError("no network")] * llama_cpp.DOWNLOAD_TRIES
    with pytest.raises(OcqError, match="in 3 tries: .*no network.*try again"):
        llama_cpp.download(URL, tmp_path / "llama.zip")
    assert list(tmp_path.iterdir()) == []
