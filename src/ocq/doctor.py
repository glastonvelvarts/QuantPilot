"""ocq doctor: is QuantPilot ready to work on this PC, and if not, what fixes it.

Each check reads what is there and changes nothing: it does not download llama.cpp or install
a tool, it says what would and how. A check that fails on its own error is reported, never
stops the others.
"""

from __future__ import annotations

import importlib.util
import os
import shutil
import subprocess
import sys
import urllib.request
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

from ocq.types import HardwareProfile, OcqError, QuantAlgorithm

OK, INFO, WARN, FAIL = "ok", "info", "warn", "fail"
LOW_DISK_GB = 10.0  # less than this is not enough for a 7B model and its 16-bit copy


@dataclass(frozen=True)
class Finding:
    name: str
    status: str  # OK, INFO (nothing to do), WARN (works, but ...), FAIL (blocks real work)
    detail: str
    fix: str = ""  # what to type to fix it


def run_checks(hardware: HardwareProfile, online: bool = True) -> list[Finding]:
    checks: list[tuple[str, Callable[[], list[Finding]]]] = [
        ("QuantPilot", quantpilot_itself),
        ("This PC", lambda: this_pc(hardware)),
        ("llama.cpp", lambda: llama_cpp(hardware)),
        ("Tools", lambda: backends(hardware)),
        ("PyTorch", lambda: pytorch(hardware)),
        ("Hugging Face", lambda: hugging_face(online)),
        ("Disk", disk),
        ("Models", models),
        ("Benchmarks", history),
    ]
    findings: list[Finding] = []
    for name, check in checks:
        try:
            findings += check()
        except Exception as exc:  # a broken check says so; the others still run
            findings.append(Finding(name, FAIL, f"could not be checked: {exc}"))
    return findings


def quantpilot_itself() -> list[Finding]:
    from ocq import __version__
    from ocq.installation import installation

    found = installation()
    how = {"uv tool": "installed with uv tool", "checkout": "from a checkout of the repo"}.get(
        found.kind, "installed in a Python environment"
    )
    extras = f", with {', '.join(found.extras)}" if found.extras else ""
    python = f"Python {sys.version.split()[0]}"
    return [Finding("QuantPilot", OK, f"{__version__}, {how}{extras}, {python}")]


def this_pc(hardware: HardwareProfile) -> list[Finding]:
    from ocq.tui import words

    findings = [Finding("This PC", OK, words.hardware(hardware))]
    nvidia = shutil.which("nvidia-smi")
    if nvidia and not hardware.has_cuda:
        findings.append(
            Finding(
                "NVIDIA GPU",
                WARN,
                "nvidia-smi is here, but no CUDA GPU was found: the driver may need a restart "
                "or an update",
            )
        )
    elif hardware.has_cuda:
        driver = subprocess.run(
            [nvidia, "--query-gpu=driver_version", "--format=csv,noheader"],
            capture_output=True, text=True, timeout=20,
        ).stdout.strip().splitlines()  # fmt: skip
        findings.append(Finding("NVIDIA driver", OK, driver[0] if driver else "found"))
    return findings


def llama_cpp(hardware: HardwareProfile) -> list[Finding]:
    """The llama.cpp QuantPilot would use: own build, PATH, or its download (not fetched)."""
    from ocq.quantizers import llama_cpp as release

    exe = "llama-server" + (".exe" if os.name == "nt" else "")
    own = os.environ.get("OCQ_LLAMA_CPP_BIN")
    if own:
        tool, where = Path(own).expanduser() / exe, "OCQ_LLAMA_CPP_BIN"
    elif shutil.which("llama-server"):
        tool, where = Path(shutil.which("llama-server")), "on PATH"
    else:
        try:
            build, _ = release.release_build(hardware)
        except OcqError as exc:
            return [Finding("llama.cpp", FAIL, str(exc))]
        folder = release.CACHE / build
        if not (folder / "ready").exists():
            return [
                Finding(
                    "llama.cpp",
                    INFO,
                    f"release {release.TAG} ({build}) is downloaded the first time a GGUF is "
                    "made or measured",
                )
            ]
        tool, where = folder / exe, f"release {release.TAG} ({build})"
    if not tool.exists():
        return [Finding("llama.cpp", FAIL, f"{tool} does not exist ({where})")]
    started = subprocess.run(
        [str(tool), "--version"], capture_output=True, text=True, timeout=60,
        stdin=subprocess.DEVNULL,
    )  # fmt: skip
    output = started.stdout + started.stderr
    if started.returncode != 0:
        why = release.missing_library(output, started.returncode) or output.strip()[-300:]
        return [Finding("llama.cpp", FAIL, f"{where}: does not start: {why}")]
    version = next((line for line in output.splitlines() if line.startswith("version")), "")
    return [Finding("llama.cpp", OK, f"{where}, starts ({version.removeprefix('version: ')})")]


def backends(hardware: HardwareProfile) -> list[Finding]:
    from ocq.installation import install_hint
    from ocq.quantizers import get_quantizer

    findings = []
    for algorithm in QuantAlgorithm:
        quantizer = get_quantizer(algorithm)
        name = algorithm.value.upper()
        if quantizer.needs_cuda and not hardware.has_cuda:
            findings.append(Finding(name, INFO, "needs an NVIDIA GPU"))
            continue
        ready, detail = quantizer.check(hardware)
        if ready:
            findings.append(Finding(name, OK, "ready"))
        elif not quantizer.is_available():  # optional: each tool is installed when wanted
            hint = install_hint(algorithm.value, hardware)
            findings.append(Finding(name, INFO, "not installed", hint))
        else:
            findings.append(Finding(name, FAIL, detail))
    if not any(finding.status == OK for finding in findings):
        hint = install_hint(QuantAlgorithm.GGUF.value, hardware)
        findings.append(
            Finding(
                "Quantizing", WARN, "no tool is ready to quantize with: GGUF works on any PC", hint
            )
        )
    return findings


def pytorch(hardware: HardwareProfile) -> list[Finding]:
    """PyPI's PyTorch for Windows runs on the CPU only: on a PC with an NVIDIA GPU, AWQ, GPTQ,
    NVFP4 and bitsandbytes then cannot use it."""
    if importlib.util.find_spec("torch") is None:
        return [Finding("PyTorch", INFO, "not installed (comes with the tools that need it)")]
    import torch

    cuda = torch.version.cuda
    if hardware.has_cuda and not cuda:
        from ocq.installation import install_hint

        fix = install_hint("gguf", hardware)
        fix += " --reinstall-package torch" if fix.startswith("uv") else ""
        return [
            Finding(
                "PyTorch",
                WARN,
                f"{torch.__version__} is the CPU-only build, but this PC has an NVIDIA GPU",
                fix,
            )
        ]
    where = f"CUDA {cuda}" if cuda else ("Metal (MPS)" if hardware.apple_silicon else "CPU")
    return [Finding("PyTorch", OK, f"{torch.__version__}, {where}")]


def hugging_face(online: bool) -> list[Finding]:
    from ocq import env_file

    token, source = env_file.hf_token()
    where = {"shell": "from the shell (HF_TOKEN)", "file": "from QuantPilot's settings"}
    if not online:
        detail = f"token {where.get(source, '')} {env_file.mask(token)}" if token else "no token"
        return [Finding("Hugging Face", INFO, f"not checked (offline); {detail.strip()}")]
    if token:
        try:
            account = env_file.check_hf_token(token)
        except OcqError as exc:
            return [Finding("Hugging Face", FAIL, f"{exc} ({where.get(source, '')})")]
        except ConnectionError:
            return [Finding("Hugging Face", WARN, "not reachable: check the connection")]
        role = f", {account.role} token" if account.role else ""
        return [
            Finding(
                "Hugging Face",
                OK,
                f"reachable; signed in as {account.name}{role} {where.get(source, '')}",
            )
        ]
    try:
        request = urllib.request.Request(
            "https://huggingface.co/api/models?limit=1", headers={"User-Agent": "ocq/0.1"}
        )
        with urllib.request.urlopen(request, timeout=15):
            pass
    except OSError:
        return [Finding("Hugging Face", WARN, "not reachable: check the connection")]
    return [
        Finding(
            "Hugging Face",
            OK,
            "reachable; no token (gated models such as Llama and Gemma need one: Settings in "
            "the app)",
        )
    ]


def disk() -> list[Finding]:
    from ocq.discovery import hf_cache_dirs
    from ocq.plan import free_disk_gb

    places = {"here (ocq-output)": Path.cwd()}
    for cache in hf_cache_dirs()[:1]:
        places["Hugging Face cache"] = cache
    places["QuantPilot's cache"] = Path.home() / ".cache" / "ocq"
    seen, parts, low = set(), [], False
    for name, path in places.items():
        free = free_disk_gb(path)
        key = round(free, 1)
        if key in seen:  # the same disk
            continue
        seen.add(key)
        parts.append(f"{free:,.1f} GB free {name}")
        low = low or free < LOW_DISK_GB
    status = WARN if low else OK
    detail = "; ".join(parts) + (" (a 7B model needs about 15 GB while it is made)" if low else "")
    return [Finding("Disk", status, detail)]


def models() -> list[Finding]:
    from collections import Counter

    from ocq.discovery import find_models

    found = [model for model in find_models() if model.complete]
    if not found:
        return [Finding("Models", INFO, "none on this PC yet")]
    sources = Counter(model.source for model in found)
    by_source = ", ".join(f"{source} {count}" for source, count in sources.most_common())
    return [Finding("Models", OK, f"{len(found)} on this PC ({by_source})")]


def history() -> list[Finding]:
    from ocq import measurements, quality

    runs, results = len(measurements.load()), len(quality.load())
    if not runs and not results:
        return [Finding("Benchmarks", INFO, "none on this PC yet")]
    return [
        Finding(
            "Benchmarks",
            OK,
            f"{runs} speed run{'s' * (runs != 1)}, {results} quality result"
            f"{'s' * (results != 1)} ({measurements.STORE.parent})",
        )
    ]
