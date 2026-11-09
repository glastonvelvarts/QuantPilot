"""The standard benchmark (ocq bench): llama-bench's tests, placed as QuantPilot plans, saved."""

import json
from pathlib import Path
from types import SimpleNamespace

import pytest
from conftest import RTX_3090, nvidia_pc
from test_discovery import tiny_gguf
from typer.testing import CliRunner

from ocq import bench, cli, measurements
from ocq.types import OcqError, RunMode

DATA = Path(__file__).parent / "data/gguf"
# llama-bench b11429 -p 512,4096 -n 128 -r 3, Qwen3.5-0.8B Q4_K_M on an M1 (ocq bench).
STANDARD_REPORT = DATA / "llama-bench-standard-qwen3.5-0.8b-q4_k_m-metal.json"


def test_the_report_gives_writing_reading_and_the_first_token():
    report = bench.read_report(STANDARD_REPORT.read_text())
    assert report.tokens_per_second == pytest.approx(73.070784)
    assert report.prompt_speeds == pytest.approx({512: 1216.101022, 4096: 1129.041272})
    assert report.build == "b11429"
    # Reading 4,096 tokens, then writing one.
    first = 4096 / 1129.041272 + 1 / 73.070784
    assert report.first_token_seconds(4096) == pytest.approx(first)
    assert report.first_token_seconds(2048) is None  # not measured


def test_a_report_without_a_writing_test_is_an_error():
    with pytest.raises(OcqError, match="no result for writing"):
        bench.read_report(json.dumps([{"n_prompt": 512, "n_gen": 0, "avg_ts": 900.0}]))
    with pytest.raises(OcqError):
        bench.read_report("llama-bench crashed")


def test_prompts_longer_than_the_model_reads_are_left_out():
    assert bench.prompts_for(bench.STANDARD, None) == (512, 4096)
    assert bench.prompts_for(bench.STANDARD, 32768) == (512, 4096)
    assert bench.prompts_for(bench.STANDARD, 2048) == (512,)  # 4,096 + 128 does not fit
    assert bench.prompts_for(bench.STANDARD, 256) == (512,)  # llama-bench says so itself


@pytest.mark.parametrize(
    ("mode", "fields", "flags", "device"),
    [
        (RunMode.GPU, {}, ["-ngl", "999"], "gpu"),
        (RunMode.CPU, {}, ["-ngl", "0", "-dev", "none"], "cpu"),
        (RunMode.GPU_OFFLOAD, {"gpu_layers": 17}, ["-ngl", "17"], "gpu+cpu (17 layers on gpu)"),
        (
            RunMode.GPU_EXPERTS_RAM,
            {"experts_in_ram": 9},
            ["-ngl", "999", "-ncmoe", "9"],
            "gpu (experts of 9 layers in ram)",
        ),
    ],
)
def test_the_model_is_placed_as_quantpilot_plans(mode, fields, flags, device):
    option = SimpleNamespace(**({"run_mode": mode, "gpu_layers": 0, "experts_in_ram": 0} | fields))
    assert bench.placement(option) == (flags, device)


def test_a_model_that_does_not_fit_is_not_run():
    with pytest.raises(OcqError, match="does not fit"):
        bench.placement(SimpleNamespace(run_mode=RunMode.UNFIT))


def test_a_file_quantpilot_made_counts_for_the_repo_it_came_from(tmp_path):
    made = tmp_path / "ocq-output" / "Qwen--Qwen3.5-0.8B" / "artifact" / "x-Q4_K_M.gguf"
    assert bench.made_from(made) == "Qwen/Qwen3.5-0.8B"
    assert bench.made_from(tmp_path / "models" / "x-Q4_K_M.gguf") is None

    # Its runs count for the repo (quantize it here) and for the file found on disk.
    measurements.record("gpu", 0.5, 70.0, model="Qwen/Qwen3.5-0.8B", label="Q4_K_M",
                        runtime="llama.cpp", file="x-Q4_K_M.gguf")  # fmt: skip
    assert len(measurements.runs_of("Qwen/Qwen3.5-0.8B", "Q4_K_M", "llama.cpp")) == 1
    assert len(measurements.runs_of("local/x-Q4_K_M.gguf", "Q4_K_M", "llama.cpp")) == 1
    assert measurements.runs_of("local/y-Q4_K_M.gguf", "Q4_K_M", "llama.cpp") == []


def fake_llama_bench(commands: list):
    """llama-bench answering with the real M1 report, for the prompt lengths asked."""
    tests = json.loads(STANDARD_REPORT.read_text())

    def run_tool(command, log, env=None, progress=False):
        commands.append(command)
        asked = {int(n) for n in command[command.index("-p") + 1].split(",")}
        return json.dumps([t for t in tests if t["n_gen"] or t["n_prompt"] in asked]), ""

    return run_tool


def test_ocq_bench_runs_the_standard_suite_and_saves_it(tmp_path, monkeypatch):
    from ocq.quantizers import gguf, llama_cpp

    made = tmp_path / "ocq-output" / "org--tiny" / "artifact" / "org--tiny-Q8_0.gguf"
    made.parent.mkdir(parents=True)
    made.write_bytes(tiny_gguf())  # a 2,048 token model: the 4,096 token prompt does not fit
    commands = []
    monkeypatch.setattr(gguf, "run_tool", fake_llama_bench(commands))
    monkeypatch.setattr(llama_cpp, "tool_path", lambda name, hardware: Path(name))
    monkeypatch.setattr(cli, "detect_hardware", lambda _bin: nvidia_pc(RTX_3090))

    result = CliRunner().invoke(cli.app, ["bench", str(made)], env={"COLUMNS": "200"})
    assert result.exit_code == 0, result.output
    (command,) = commands
    assert command[0] == "llama-bench"
    assert command[command.index("-p") + 1] == "512"
    assert command[command.index("-n") + 1] == "128"
    assert command[command.index("-ngl") + 1] == "999"  # all on the RTX 3090
    assert "73.1 tok/s" in result.output and "1,216 tok/s for 512 tokens" in result.output
    assert "0.43 s after 512 tokens" in result.output
    assert "expected up to" in result.output  # no runs yet: the memory speed limit

    saved = measurements.load()[-1]
    assert (saved["model"], saved["file"], saved["label"]) == (
        "org/tiny",
        "org--tiny-Q8_0.gguf",
        "Q8_0",
    )
    assert (saved["suite"], saved["device"], saved["runtime_version"]) == (
        "standard",
        "gpu",
        "b11429",
    )
    assert saved["prompt_speeds"] == {"512": pytest.approx(1216.101022)}
    assert saved["gpu"] == "NVIDIA GeForce RTX 3090, 24 GB"

    again = CliRunner().invoke(cli.app, ["bench", str(made)], env={"COLUMNS": "200"})
    assert "(2 runs of this setup)" in again.output


def test_ocq_bench_says_it_measures_models_on_this_pc():
    result = CliRunner().invoke(cli.app, ["bench", "Qwen/Qwen2.5-7B-Instruct"])
    assert result.exit_code == 1
    assert "measures a model that is on this PC" in result.output


def test_a_run_counts_for_the_same_setup_only_when_placed_the_same_way():
    # Seen on the M1: the CPU-only run of a model was counted with its GPU runs.
    for device in (
        "gpu",
        "cpu",
        "gpu+cpu (20/29 layers on gpu)",
        "gpu (experts of 9 layers in ram)",
    ):
        measurements.record(device, 0.5, 70.0, model="m", label="Q4_K_M", runtime="llama.cpp")
    for mode, placement in [
        (RunMode.GPU, "gpu"),
        (RunMode.CPU, "cpu"),
        (RunMode.GPU_OFFLOAD, "gpu+cpu"),
        (RunMode.GPU_EXPERTS_RAM, "gpu+experts"),
    ]:
        assert measurements.placed_as(mode) == placement
        (run,) = measurements.runs_of("m", "Q4_K_M", "llama.cpp", placement=placement)
        assert measurements.placed(run["device"]) == placement
    assert len(measurements.runs_of("m", "Q4_K_M", "llama.cpp")) == 4  # any placement
    # ocq bench's own name for a split matches the quantize benchmark's.
    assert measurements.placed("gpu+cpu (17 layers on gpu)") == "gpu+cpu"
