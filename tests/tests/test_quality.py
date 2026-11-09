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


# What a format keeps, against its reference (7E) --------------------------------------------


def save(
    label, ppl, model="Qwen/Qwen3.5-0.8B", device="gpu", piece=quality.FIXED, runtime="llama.cpp"
):
    quality.record(quality.Perplexity(ppl, 0.5, 40, runtime), piece, model=model, label=label,
                   device=device)  # fmt: skip


def test_quality_kept_is_the_perplexity_worth_in_parameters():
    # Q4_K_M of Qwen3.5-0.8B on the M1: 17.2720 against 16.6079 for the BF16 original.
    kept = quality.kept_share(17.2720, 16.6079)
    assert kept == pytest.approx(math.exp(-math.log(17.2720 / 16.6079) / quality.SIZE_SLOPE))
    assert 0.80 < kept < 0.81
    assert quality.kept_share(16.5, 16.6) == 1.0  # a hair better by chance: all of it, no more


def test_the_reference_is_the_original_else_q8_0_on_the_same_device():
    assert quality.reference_for("Qwen/Qwen3.5-0.8B", "fixed") is None
    save("Q8_0", 16.62)
    assert quality.reference_for("Qwen/Qwen3.5-0.8B", "fixed")["label"] == "Q8_0"
    save("BF16", 16.61, device="cpu", runtime="transformers")
    save("BF16", 16.6079)
    assert quality.reference_for("Qwen/Qwen3.5-0.8B", "fixed", "gpu")["ppl"] == 16.6079
    assert quality.reference_for("Qwen/Qwen3.5-0.8B", "fixed", "cpu")["ppl"] == 16.61
    assert quality.reference_for("Qwen/Qwen3.5-0.8B", "full") is None  # another slice


def test_a_format_is_compared_on_the_same_slice_and_never_with_itself():
    save("Q4_K_M", 17.2720)
    assert quality.measured_kept("Qwen/Qwen3.5-0.8B", "Q4_K_M") is None  # nothing to compare
    save("Q8_0", 16.62)
    via_q8 = quality.measured_kept("Qwen/Qwen3.5-0.8B", "Q4_K_M")
    assert via_q8.reference_label == "Q8_0"
    assert quality.measured_kept("Qwen/Qwen3.5-0.8B", "Q8_0") is None  # it is the reference
    save("BF16", 16.6079)
    kept = quality.measured_kept("Qwen/Qwen3.5-0.8B", "Q4_K_M")
    assert (kept.label, kept.reference_label, kept.slice, kept.same_device) == (
        "Q4_K_M", "BF16", "fixed", True,
    )  # fmt: skip
    assert quality.measured_kept("Qwen/Qwen3.5-0.8B", "BF16") is None
    save("Q4_K_M", 17.25, piece=quality.FULL)  # the whole text counts first, with its own ref
    assert quality.measured_kept("Qwen/Qwen3.5-0.8B", "Q4_K_M").slice == "fixed"
    save("BF16", 16.58, piece=quality.FULL)
    assert quality.measured_kept("Qwen/Qwen3.5-0.8B", "Q4_K_M").slice == "full"
    assert [m.label for m in quality.measured_formats("Qwen/Qwen3.5-0.8B")] == ["Q4_K_M", "Q8_0"]


def test_the_reference_is_made_from_the_original_in_the_hugging_face_cache(tmp_path, monkeypatch):
    from conftest import QWEN, real_model, write_safetensors
    from test_discovery import hf_repo

    from ocq import bench
    from ocq.quantizers import gguf, llama_cpp
    from ocq.types import RunMode

    _, config, tensors, dtypes = real_model(QWEN)
    write_safetensors(tmp_path / "weights", tensors, dtypes)
    files = {"config.json": json.dumps(config).encode(), "tokenizer.json": b"{}",
             "model.safetensors": (tmp_path / "weights").read_bytes()}  # fmt: skip
    hf_repo(tmp_path / "hub", QWEN, files)
    monkeypatch.setenv("HF_HUB_CACHE", str(tmp_path / "hub"))
    text = quality.DATASETS / "wikitext-2-raw" / "wiki.test.raw"
    text.parent.mkdir(parents=True)
    text.write_text("text")
    commands = []

    def run_tool(command, log, env=None, progress=False):
        commands.append(command)
        if "--outfile" in command:  # the converter
            Path(command[command.index("--outfile") + 1]).write_bytes(tiny_gguf(32))  # BF16
        elif "llama-quantize" in command[0]:
            Path(command[2]).write_bytes(tiny_gguf(7))  # Q8_0
        elif "llama-perplexity" in command[0]:
            return "", PERPLEXITY_LOG.read_text().replace("17.2720", "16.6079")
        return "", ""

    monkeypatch.setattr(gguf, "run_tool", run_tool)
    monkeypatch.setattr(llama_cpp, "tool_path", lambda name, hardware: Path(name))
    monkeypatch.setattr(llama_cpp, "converter_script", lambda: Path("convert_hf_to_gguf.py"))
    hardware = nvidia_pc(RTX_3090)
    saved = bench.make_reference(QWEN, hardware, quality.FIXED, tmp_path)
    assert (saved["model"], saved["label"], saved["ppl"]) == (QWEN, "BF16", 16.6079)
    convert, perplexity = commands
    assert convert[convert.index("--outtype") + 1] == "bf16"  # as stored
    assert perplexity[perplexity.index("--chunks") + 1] == "40"
    assert not (quality.DATASETS.parent / "reference").exists()  # its files are gone

    # Too big for this PC's memory in 16 bits: Q8_0 made from it.
    real_fitted = bench.fitted
    sizes = iter([RunMode.UNFIT, None])

    def fitted(model, hw):
        option, snapshot = real_fitted(model, hw)
        mode = next(sizes)
        return (replace(option, run_mode=mode) if mode else option), snapshot

    from dataclasses import replace

    monkeypatch.setattr(bench, "fitted", fitted)
    commands.clear()
    saved = bench.make_reference(QWEN, hardware, quality.FIXED, tmp_path)
    assert saved["label"] == "Q8_0"
    assert [c[0] for c in commands][1:] == ["llama-quantize", "llama-perplexity"]


def test_ocq_bench_quality_says_how_much_the_format_keeps(tmp_path, monkeypatch):
    from ocq.quantizers import gguf, llama_cpp

    text = quality.DATASETS / "wikitext-2-raw" / "wiki.test.raw"
    text.parent.mkdir(parents=True)
    text.write_text("text")
    made = tmp_path / "ocq-output" / "org--tiny" / "artifact" / "org--tiny-Q8_0.gguf"
    made.parent.mkdir(parents=True)
    made.write_bytes(tiny_gguf())
    speed = fake_llama_bench([])

    def run_tool(command, log, env=None, progress=False):
        if command[0] == "llama-perplexity":
            return "", PERPLEXITY_LOG.read_text()  # 17.2720
        return speed(command, log, env, progress)

    monkeypatch.setattr(gguf, "run_tool", run_tool)
    monkeypatch.setattr(llama_cpp, "tool_path", lambda name, hardware: Path(name))
    monkeypatch.setattr(cli, "detect_hardware", lambda _bin: nvidia_pc(RTX_3090))
    args = ["bench", str(made), "--quality"]

    # The original is not on this PC (not in the Hugging Face cache): nothing to compare with.
    result = CliRunner().invoke(cli.app, args, env={"COLUMNS": "200"})
    assert "the original weights of this model are not on this PC" in result.output

    save("BF16", 16.6079, model="org/tiny")
    result = CliRunner().invoke(cli.app, args, env={"COLUMNS": "200"})
    assert result.exit_code == 0, result.output
    assert (
        "Q8_0 keeps about 80% of the model's quality (perplexity 17.2720 vs 16.6079 for the "
        "original, BF16)"
        in " ".join(result.output.split())
    )
