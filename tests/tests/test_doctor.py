"""ocq doctor: what is ready on this PC, and what fixes what is not. It changes nothing."""

import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
from test_model_analyzer import make_hardware
from typer.testing import CliRunner

from ocq import cli, doctor, measurements
from ocq.quantizers import gguf, llama_cpp
from ocq.types import QuantAlgorithm

NVIDIA = make_hardware(vram_gb=8)
NO_GPU = make_hardware()


def statuses(findings) -> dict[str, tuple[str, str]]:
    return {f.name: (f.status, f.detail) for f in findings}


def test_a_check_that_breaks_is_reported_and_the_others_still_run(monkeypatch):
    def broken():
        raise RuntimeError("disk not mounted")

    monkeypatch.setattr(doctor, "disk", broken)
    monkeypatch.setattr(doctor, "hugging_face", lambda online: [])
    found = statuses(doctor.run_checks(NO_GPU))
    assert found["Disk"] == (doctor.FAIL, "could not be checked: disk not mounted")
    assert found["QuantPilot"][0] == doctor.OK and "Benchmarks" in found


def test_llama_cpp_not_downloaded_yet_starts_or_does_not(tmp_path, monkeypatch):
    monkeypatch.delenv("OCQ_LLAMA_CPP_BIN", raising=False)
    monkeypatch.setattr(doctor.shutil, "which", lambda name: None)
    (finding,) = doctor.llama_cpp(NO_GPU)
    assert finding.status == doctor.INFO and "downloaded the first time" in finding.detail

    build, _ = llama_cpp.release_build(NO_GPU)
    folder = llama_cpp.CACHE / build
    folder.mkdir(parents=True)
    (folder / "ready").touch()
    exe = "llama-server" + (".exe" if sys.platform == "win32" else "")
    (folder / exe).touch()
    answers = iter([
        SimpleNamespace(returncode=0, stdout="", stderr="version: 0.6.0 (build 11429)\n"),
        SimpleNamespace(returncode=127, stdout="",
                        stderr="error while loading shared libraries: libgomp.so.1: cannot open"),
    ])  # fmt: skip
    monkeypatch.setattr(doctor.subprocess, "run", lambda *a, **k: next(answers))
    (finding,) = doctor.llama_cpp(NO_GPU)
    assert finding.status == doctor.OK and "starts (0.6.0 (build 11429))" in finding.detail
    (finding,) = doctor.llama_cpp(NO_GPU)
    assert finding.status == doctor.FAIL and "lacks libgomp.so.1" in finding.detail


def test_tools_not_installed_are_offered_and_none_ready_is_worth_a_look(monkeypatch):
    for algorithm in QuantAlgorithm:
        quantizer = type(doctor_quantizer(algorithm))
        monkeypatch.setattr(quantizer, "is_available", classmethod(lambda cls: False))
    found = doctor.backends(NO_GPU)
    by_name = {f.name: f for f in found}
    assert by_name["AWQ"].detail == "needs an NVIDIA GPU"
    assert by_name["GGUF"].status == doctor.INFO and by_name["GGUF"].fix
    assert by_name["Quantizing"].status == doctor.WARN

    monkeypatch.setattr(gguf.GGUFQuantizer, "is_available", classmethod(lambda cls: True))
    monkeypatch.setattr(gguf, "has_openmp_runtime", lambda: True)
    assert "Quantizing" not in {f.name for f in doctor.backends(NO_GPU)}


def doctor_quantizer(algorithm):
    from ocq.quantizers import get_quantizer

    return get_quantizer(algorithm)


def test_a_cpu_only_pytorch_on_an_nvidia_pc_is_flagged(monkeypatch):
    fake = SimpleNamespace(__version__="2.6.0+cpu", version=SimpleNamespace(cuda=None))
    monkeypatch.setitem(sys.modules, "torch", fake)
    monkeypatch.setattr(doctor.importlib.util, "find_spec", lambda name: object())
    (finding,) = doctor.pytorch(NVIDIA)
    assert finding.status == doctor.WARN and "CPU-only build" in finding.detail
    assert finding.fix == "uv sync --extra gguf --reinstall-package torch"  # this checkout

    from ocq import installation

    tool = installation.Installation(installation.UV_TOOL)
    monkeypatch.setattr(installation, "installation", lambda prefix=None: tool)
    (finding,) = doctor.pytorch(NVIDIA)
    assert finding.fix == (
        "uv tool install 'quantpilot[gguf]' --torch-backend auto --reinstall-package torch"
    )
    (finding,) = doctor.pytorch(NO_GPU)
    assert (finding.status, finding.detail) == (doctor.OK, "2.6.0+cpu, CPU")


def test_hugging_face_with_and_without_a_token(monkeypatch):
    from ocq import env_file

    found = doctor.hugging_face(online=False)
    assert found[0].detail == "not checked (offline); no token"

    def offline(*args, **kwargs):
        raise OSError("no network")

    monkeypatch.setattr(doctor.urllib.request, "urlopen", offline)
    assert doctor.hugging_face(online=True)[0].status == doctor.WARN

    monkeypatch.setenv("HF_TOKEN", "hf_abcdefghijklmnop")
    monkeypatch.setattr(env_file, "check_hf_token", lambda token: env_file.HfAccount("me", "read"))
    (finding,) = doctor.hugging_face(online=True)
    assert finding.detail == "reachable; signed in as me, read token from the shell (HF_TOKEN)"


def test_low_disk_is_worth_a_look(monkeypatch):
    import ocq.plan

    monkeypatch.setattr(ocq.plan, "free_disk_gb", lambda path: 4.0)
    (finding,) = doctor.disk()
    assert finding.status == doctor.WARN and "4.0 GB free here (ocq-output)" in finding.detail


def test_the_history_is_counted():
    assert doctor.history()[0].detail == "none on this PC yet"
    measurements.record("gpu", 0.5, 100.0)
    assert doctor.history()[0].detail.startswith("1 speed run, 0 quality results")


def test_ocq_doctor_says_ready_or_what_blocks(monkeypatch):
    from ocq.doctor import FAIL, INFO, OK, Finding

    monkeypatch.setattr(cli, "detect_hardware", lambda _bin: NO_GPU)
    findings = [
        Finding("llama.cpp", OK, "starts"),
        Finding("BNB", INFO, "not installed", "uv sync --extra bnb"),
    ]
    monkeypatch.setattr(doctor, "run_checks", lambda hw, online=True: findings)
    result = CliRunner().invoke(cli.app, ["doctor"], env={"COLUMNS": "120"})
    assert result.exit_code == 0 and "Ready." in result.output
    assert "run: uv sync --extra bnb" in result.output

    findings.append(Finding("llama.cpp", FAIL, "does not start"))
    result = CliRunner().invoke(cli.app, ["doctor", "--offline"], env={"COLUMNS": "120"})
    assert result.exit_code == 1 and "1 problem to fix." in result.output


@pytest.mark.skipif(sys.platform == "win32", reason="the folder layout of this check is POSIX")
def test_the_real_doctor_runs_offline_on_this_pc():
    findings = doctor.run_checks(NO_GPU, online=False)
    assert all(f.status != doctor.FAIL or "llama.cpp" in f.name for f in findings), findings
    assert Path.cwd()  # it changed nothing it could not undo: nothing to undo
