"""Measured quality: perplexity on WikiText-2, as llama.cpp measures it, in either runtime."""

import hashlib
import io
import json
import math
import zipfile
from pathlib import Path

import pytest
from conftest import RTX_3090, nvidia_pc
from test_bench import fake_llama_bench
from test_discovery import empty_home, tiny_gguf  # noqa: F401 (empty_home: autouse)
from typer.testing import CliRunner

from ocq import cli, quality
from ocq.types import OcqError

DATA = Path(__file__).parent / "data/gguf"
# llama-perplexity b11429 -c 512 --chunks 40, Qwen3.5-0.8B Q4_K_M on an M1.
PERPLEXITY_LOG = DATA / "llama-perplexity-qwen3.5-0.8b-q4_k_m-40-chunks-metal.log"


def test_llama_perplexitys_final_estimate_is_read():
    result = quality.read_llama_output(PERPLEXITY_LOG.read_text(), "b11429")
    assert (result.value, result.error, result.chunks) == (17.2720, 0.51375, 40)
    assert (result.runtime, result.build) == ("llama.cpp", "b11429")
    assert result.tokens_scored == 40 * 255  # each chunk's second half, but its last token
    with pytest.raises(OcqError, match="no final estimate"):
        quality.read_llama_output("main: error: failed to load model")


def test_chunks_are_scored_on_their_second_half_as_llama_cpp_does():
    seen = []

    def chunk_logprobs(chunk):
        seen.append(chunk)
        # Every next token at a probability of 1/4, but the first half is not scored.
        return [-math.log(1000.0)] * 256 + [-math.log(4.0)] * 256

    tokens = list(range(5 * 512 + 100))
    result = quality.score_chunks(
        tokens, quality.Slice("test", 3), bos=1, chunk_logprobs=chunk_logprobs
    )
    assert result.chunks == 3 and len(seen) == 3
    assert result.value == pytest.approx(4.0) and result.error == pytest.approx(0.0)
    assert seen[1][0] == 1 and seen[1][1:] == tokens[513:1024]  # chunk start: the BOS token
    assert quality.score_chunks(tokens, quality.FULL, None, chunk_logprobs).chunks == 5
    with pytest.raises(OcqError, match="fewer than 512 tokens"):
        quality.score_chunks(tokens[:100], quality.FIXED, None, chunk_logprobs)


def wikitext_zip(text: bytes = b" = Robert Boulter = \n") -> bytes:
    data = io.BytesIO()
    with zipfile.ZipFile(data, "w") as bundle:
        bundle.writestr("wikitext-2-raw/wiki.test.raw", text)
        bundle.writestr("wikitext-2-raw/wiki.train.raw", b"not needed")
    return data.getvalue()


def test_the_text_is_downloaded_once_and_checked(monkeypatch):
    from ocq.quantizers import llama_cpp

    archive = wikitext_zip()
    downloads = []

    def download(url, target):
        downloads.append(url)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(archive)
        return target

    monkeypatch.setattr(llama_cpp, "download", download)
    monkeypatch.setattr(quality, "WIKITEXT_SHA256", "0" * 64)
    with pytest.raises(OcqError, match="checksum differs"):
        quality.wikitext()
    assert not (quality.DATASETS / "wikitext-2-raw-v1.zip").exists()

    monkeypatch.setattr(quality, "WIKITEXT_SHA256", hashlib.sha256(archive).hexdigest())
    text = quality.wikitext()
    assert text.read_bytes() == b" = Robert Boulter = \n"
    assert quality.wikitext() == text and downloads == [quality.WIKITEXT_URL] * 2
    assert sorted(p.name for p in quality.DATASETS.rglob("*")) == [
        "wiki.test.raw",
        "wikitext-2-raw",
    ]


def test_results_are_saved_and_found_by_repo_or_file():
    result = quality.Perplexity(17.27, 0.51, 40, "llama.cpp", "b11429")
    quality.record(result, quality.FIXED, model="Qwen/Qwen3.5-0.8B", label="Q4_K_M",
                   file="Qwen--Qwen3.5-0.8B-Q4_K_M.gguf")  # fmt: skip
    quality.record(
        quality.Perplexity(16.9, 0.5, 40, "transformers"), quality.FIXED,
        model="Qwen/Qwen3.5-0.8B", label="BF16",
    )  # fmt: skip
    (saved,) = quality.results_of("Qwen/Qwen3.5-0.8B", "q4_k_m")
    assert (saved["ppl"], saved["slice"], saved["dataset"]) == (17.27, "fixed", "WikiText-2 test")
    assert quality.results_of("local/Qwen--Qwen3.5-0.8B-Q4_K_M.gguf") == [saved]
    assert len(quality.results_of("Qwen/Qwen3.5-0.8B")) == 2
    assert quality.results_of("Qwen/Qwen3.5-0.8B", slice_name="full") == []
    quality.STORE.write_text('[{"ppl": "high"}, 3, {"model": "m", "ppl": 9.5}]')
    assert quality.load() == [{"model": "m", "ppl": 9.5}]


def test_ocq_bench_quality_measures_a_gguf_where_its_speed_ran(tmp_path, monkeypatch):
    from ocq.quantizers import gguf, llama_cpp

    text = quality.DATASETS / "wikitext-2-raw" / "wiki.test.raw"
    text.parent.mkdir(parents=True)
    text.write_text("text")
    model = tmp_path / "models" / "tiny-Q8_0.gguf"
    model.parent.mkdir()
    model.write_bytes(tiny_gguf())
    commands = []
    speed = fake_llama_bench(commands)

    def run_tool(command, log, env=None, progress=False):
        if command[0] == "llama-perplexity":
            commands.append(command)
            return "", PERPLEXITY_LOG.read_text()
        return speed(command, log, env, progress)

    monkeypatch.setattr(gguf, "run_tool", run_tool)
    monkeypatch.setattr(llama_cpp, "tool_path", lambda name, hardware: Path(name))
    monkeypatch.setattr(cli, "detect_hardware", lambda _bin: nvidia_pc(RTX_3090))
    result = CliRunner().invoke(cli.app, ["bench", str(model), "--quality"], env={"COLUMNS": "200"})
    assert result.exit_code == 0, result.output
    bench, perplexity = commands
    assert perplexity[perplexity.index("-f") + 1] == str(text)
    assert perplexity[perplexity.index("-c") + 1] == "512"
    assert perplexity[perplexity.index("--chunks") + 1] == "40"
    assert perplexity[perplexity.index("-ngl") + 1] == "999"  # placed as for its speed
    assert "perplexity 17.2720 ± 0.5138" in result.output
    (saved,) = quality.load()
    assert (saved["model"], saved["label"], saved["file"]) == (
        "local/tiny-Q8_0.gguf",
        "Q8_0",
        "tiny-Q8_0.gguf",
    )

    commands.clear()
    CliRunner().invoke(cli.app, ["bench", str(model), "--full"], env={"COLUMNS": "200"})
    assert "--chunks" not in commands[1]  # the whole text
    assert quality.load()[-1]["slice"] == "full"


def test_a_model_transformers_runs_needs_quality_asked_for(tmp_path, monkeypatch):
    from conftest import QWEN, real_model, write_safetensors

    folder = tmp_path / "my-model"
    folder.mkdir()
    _, config, tensors, dtypes = real_model(QWEN)
    (folder / "config.json").write_text(json.dumps(config))
    write_safetensors(folder / "model.safetensors", tensors, dtypes)
    (folder / "tokenizer.json").write_text("{}")
    monkeypatch.setattr(cli, "detect_hardware", lambda _bin: nvidia_pc(RTX_3090))
    result = CliRunner().invoke(cli.app, ["bench", str(folder)], env={"COLUMNS": "200"})
    assert result.exit_code == 1
    assert "add --quality to measure its quality" in result.output

    measured = []

    def transformers_perplexity(path, text, piece):
        measured.append(path)
        return quality.Perplexity(9.5, 0.2, 40, "transformers", "5.0")

    monkeypatch.setattr(quality, "transformers_perplexity", transformers_perplexity)
    monkeypatch.setattr(quality, "wikitext", lambda: tmp_path / "wiki.test.raw")
    result = CliRunner().invoke(
        cli.app, ["bench", str(folder), "--quality"], env={"COLUMNS": "200"}
    )
    assert result.exit_code == 0, result.output
    assert measured == [folder.resolve()]
    (saved,) = quality.load()
    assert (saved["model"], saved["label"], saved["runtime"]) == (
        "local/my-model",
        "BF16",
        "transformers",
    )


def test_a_hugging_face_model_with_only_its_config_cached_is_not_on_this_pc(tmp_path, monkeypatch):
    from test_discovery import hf_repo

    hf_repo(tmp_path / "hub", "org/model", {"config.json": b"{}"})
    monkeypatch.setenv("HF_HUB_CACHE", str(tmp_path / "hub"))
    result = CliRunner().invoke(cli.app, ["bench", "org/model", "--quality"])
    assert result.exit_code == 1 and "org/model is not on this PC" in result.output
