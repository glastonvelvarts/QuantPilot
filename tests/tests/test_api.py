"""The API and --json: plain data, versioned. The key sets below are the promise of schema
version 1 (docs/api.md): keys may be added (add them here), never renamed or removed."""

import json
from types import SimpleNamespace

import pytest
from conftest import QWEN
from test_discovery import empty_home  # noqa: F401 (autouse: no models of this PC)
from test_model_analyzer import make_hardware
from test_recommend import BIG_GPU
from typer.testing import CliRunner

from ocq import api, bench, cli, doctor, measurements, quality
from ocq.types import OcqError

ENVELOPE = {"schema_version", "quantpilot_version", "kind"}
HARDWARE = {
    "os", "cpu", "cpu_cores", "ram_gb", "ram_free_gb", "ram_bandwidth_gbs",
    "ram_bandwidth_measured_gbs", "backend", "unified_memory", "gpus",
}  # fmt: skip
GPU = {"name", "count", "vram_gb", "vram_free_gb", "backend", "bandwidth_gbs", "compute_capability"}
FORMAT = {
    "label", "kind", "runtime", "bits_per_weight", "weights_gb", "context_gb",
    "memory_required_gb", "memory_free_gb", "fit", "runs_on", "gpu_layers",
    "experts_in_ram_layers", "tokens_per_second", "tokens_per_second_basis",
    "prompt_tokens_per_second", "note",
}  # fmt: skip
RECOMMENDATION = {
    "model", "format", "context", "kv_cache", "needs_quantizing", "reasons", "warnings",
    "confidence", "lost_on", "command", "quality", "values",
}  # fmt: skip
FACTS = {
    "parameters", "active_parameters", "precision", "stored_bytes", "architecture",
    "context_length", "kv_cache", "quantization",
}  # fmt: skip


def keys(data: dict) -> set:
    return set(data)


def as_json(data: dict) -> dict:
    """What a script reads: the result through json and back, unchanged."""
    text = json.dumps(data)
    assert json.loads(text) == data
    return data


def test_profile(monkeypatch):
    import ocq.plan

    monkeypatch.setattr(ocq.plan, "free_disk_gb", lambda path: 50.0)
    data = as_json(api.profile(make_hardware(vram_gb=8)))
    assert keys(data) == ENVELOPE | {"hardware", "disk_free_gb"}
    assert (data["schema_version"], data["kind"]) == (1, "profile")
    assert keys(data["hardware"]) == HARDWARE and keys(data["hardware"]["gpus"][0]) == GPU


def test_inspect_and_feasibility(saved_hub, monkeypatch):
    from ocq import model_analyzer

    monkeypatch.setattr(model_analyzer, "catalog_entry", lambda *a: None)  # ocq inspect asks it
    data = as_json(api.inspect(QWEN))
    assert keys(data) == ENVELOPE | {"model", "facts"} and keys(data["facts"]) == FACTS
    assert data["facts"]["parameters"] == {
        "value": 1_543_714_304, "source": data["facts"]["parameters"]["source"],
        "kind": "cross-checked", "note": data["facts"]["parameters"]["note"],
    }  # fmt: skip

    data = as_json(api.feasibility(QWEN, 8192, "q8_0", hardware=BIG_GPU))
    assert keys(data) == ENVELOPE | {"model", "context", "kv_cache", "hardware", "formats"}
    assert all(keys(option) == FORMAT for option in data["formats"])
    q4 = next(option for option in data["formats"] if option["label"] == "Q4_K_M")
    assert (q4["runtime"], q4["fit"], q4["runs_on"]) == ("llama.cpp", "Perfect", "gpu")


def test_recommend(saved_hub):
    data = as_json(api.recommend([QWEN], "coding", hardware=BIG_GPU, limit=3))
    assert keys(data) == ENVELOPE | {
        "workload", "context", "hardware", "best", "alternatives", "considered", "ruled_out",
        "unread", "too_big",
    }  # fmt: skip
    best = data["best"]
    assert keys(best) == RECOMMENDATION and keys(best["format"]) == FORMAT
    assert keys(best["model"]) == {"ref", "id", "on_this_pc", "tags"}
    assert keys(best["quality"]) == {"kept", "measured"}
    assert keys(best["confidence"]) == {"level", "evidence"}
    assert best["values"]["tokens_per_second"] == best["format"]["tokens_per_second"]
    assert best["command"].startswith(f"ocq optimize {QWEN} -b ")
    assert {r["kind"] for r in best["reasons"]} <= {"good", "tradeoff", "bad"}
    assert len(data["alternatives"]) == 3 and all(a["lost_on"] for a in data["alternatives"])
    assert data["workload"]["priorities"] == ["quality", "context", "speed"]
    with pytest.raises(OcqError, match="Unknown workload 'gaming'"):
        api.recommend([QWEN], "gaming", hardware=BIG_GPU)


def test_low_latency_gives_the_first_token_in_seconds(saved_hub):
    measurements.record("gpu", 0.5, 300.0, prompt_tps=8000.0, active_params_b=0.5)
    best = api.recommend([QWEN], "low-latency", hardware=BIG_GPU)["best"]
    assert 0 < best["values"]["first_token_seconds"] < 1


def bench_outcome(piece=None):
    report = bench.BenchReport(78.2, 0.5, {512: 1228.0, 4096: 1140.0}, "b11429")
    option = SimpleNamespace(estimated_tps=80.0, speed_basis="from runs on this PC")
    outcome = bench.BenchOutcome("Qwen/Qwen3.5-0.8B", "Q4_K_M", option, "gpu", report, 3)
    if piece:
        outcome.piece = piece
        outcome.quality = quality.Perplexity(17.272, 0.51, 40, "llama.cpp", "b11429")
        outcome.kept = quality.MeasuredKept(
            "Q4_K_M", 0.8008, 17.272, 16.6079, "BF16", "fixed", True
        )
    return outcome


def test_bench(monkeypatch):
    calls = []

    def run_benchmark(model, hardware, *, piece=None, **kwargs):
        calls.append(piece.name if piece else "")
        return bench_outcome(piece)

    monkeypatch.setattr(bench, "run_benchmark", run_benchmark)
    data = as_json(api.bench("m.gguf", hardware=BIG_GPU))
    assert keys(data) == ENVELOPE | {"model", "label", "placement", "kv_cache", "speed", "quality"}
    assert data["quality"] is None
    assert keys(data["speed"]) == {
        "tokens_per_second", "tokens_per_second_spread", "prompt_tokens_per_second",
        "first_token_seconds", "expected_tokens_per_second", "expected_basis",
        "runtime_version", "runs_of_this_setup",
    }  # fmt: skip
    assert data["speed"]["first_token_seconds"]["512"] == round(512 / 1228 + 1 / 78.2, 4)

    data = as_json(api.bench("m.gguf", "fixed", hardware=BIG_GPU))
    assert keys(data["quality"]) == {
        "dataset", "slice", "chunks", "perplexity", "perplexity_error", "runtime",
        "is_the_original", "kept", "reference", "note",
    }  # fmt: skip
    assert data["quality"]["reference"] == {
        "label": "BF16",
        "perplexity": 16.6079,
        "same_device": True,
    }
    assert calls == ["", "fixed"]
    with pytest.raises(OcqError, match='"fixed" or "full"'):
        api.bench("m.gguf", "some", hardware=BIG_GPU)


def test_history_models_and_doctor(monkeypatch):
    measurements.record("gpu", 0.5, 80.0, model="org/m", label="Q4_K_M", runtime="llama.cpp")
    quality.record(quality.Perplexity(17.0, 0.5, 40, "llama.cpp"), quality.FIXED, model="org/m",
                   label="Q4_K_M")  # fmt: skip
    data = as_json(api.history("org/m"))
    assert keys(data) == ENVELOPE | {"speed_runs", "quality_results"}
    assert data["speed_runs"][0]["tps"] == 80.0 and data["quality_results"][0]["ppl"] == 17.0
    assert api.history("other/model")["speed_runs"] == []

    data = as_json(api.models(hardware=make_hardware()))
    assert keys(data) == ENVELOPE | {"context", "kv_cache", "models"} and data["models"] == []

    findings = [doctor.Finding("GGUF", doctor.INFO, "not installed", "uv sync --extra gguf")]
    monkeypatch.setattr(doctor, "run_checks", lambda hw, online=True: findings)
    data = as_json(api.doctor(hardware=make_hardware()))
    assert keys(data) == ENVELOPE | {"ready", "checks"} and data["ready"] is True
    assert data["checks"] == [
        {"name": "GGUF", "status": "info", "detail": "not installed", "fix": "uv sync --extra gguf"}
    ]


def test_json_on_the_command_line_is_only_json(monkeypatch):
    # A benchmark prints its steps as it goes: they go to stderr, stdout stays JSON.
    def run_benchmark(model, hardware, *, piece=None, **kwargs):
        print("Measuring the speed (3 runs of each test) ...")
        return bench_outcome(piece)

    monkeypatch.setattr(bench, "run_benchmark", run_benchmark)
    monkeypatch.setattr(cli, "detect_hardware", lambda _bin: BIG_GPU)
    monkeypatch.setattr(api, "read_hardware", lambda llmfit_bin=None: BIG_GPU)
    result = CliRunner().invoke(cli.app, ["bench", "m.gguf", "--json"])
    assert result.exit_code == 0
    assert json.loads(result.stdout)["speed"]["tokens_per_second"] == 78.2
    assert "Measuring the speed" in result.stderr

    # A failure: JSON with the message, and exit code 1.
    result = CliRunner().invoke(cli.app, ["recommend", "-w", "gaming", "--json"])
    assert result.exit_code == 1
    assert json.loads(result.stdout) == {
        "schema_version": 1,
        "error": result.stdout and json.loads(result.stdout)["error"],
    }
    assert json.loads(result.stdout)["error"].startswith("Unknown workload 'gaming'")

    # doctor: exit code 1 when something blocks, the JSON still printed.
    findings = [doctor.Finding("llama.cpp", doctor.FAIL, "does not start")]
    monkeypatch.setattr(doctor, "run_checks", lambda hw, online=True: findings)
    result = CliRunner().invoke(cli.app, ["doctor", "--json", "--offline"])
    assert result.exit_code == 1 and json.loads(result.stdout)["ready"] is False
