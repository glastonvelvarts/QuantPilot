"""The real tests of every format on this PC, in one report file to send back.

make test-gpu (or: uv run python scripts/gpu_report.py) runs every real quantization
(pytest -m slow): each format made from SmolLM2-135M, checked, its speed measured and its
quality compared with the original. The NVIDIA formats (AWQ, GPTQ, NVFP4, bitsandbytes) need an
NVIDIA GPU, AWQ and GPTQ also Linux or WSL2. The report starts with this PC and what is
installed, then every test's result and time; it is saved in results/ (not part of the repo).
"""

from __future__ import annotations

import datetime
import importlib.util
import platform
import re
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
LIBRARIES = ("torch", "transformers", "llmcompressor", "bitsandbytes", "accelerate")


def output_of(command: list[str]) -> str:
    try:
        done = subprocess.run(
            command, cwd=ROOT, capture_output=True, text=True, encoding="utf-8",
            errors="replace", timeout=120,
        )  # fmt: skip
    except (OSError, subprocess.TimeoutExpired) as exc:
        return f"(could not run {command[0]}: {exc})"
    return (done.stdout + done.stderr).strip()


def library_versions() -> list[str]:
    from importlib import metadata

    lines = []
    for name in LIBRARIES:
        try:
            lines.append(f"{name} {metadata.version(name)}")
        except metadata.PackageNotFoundError:
            lines.append(f"{name}: not installed")
    if importlib.util.find_spec("torch"):
        import torch

        cuda = torch.cuda.is_available()
        device = torch.cuda.get_device_name(0) if cuda else "none"
        lines.append(f"torch sees CUDA: {cuda} ({device}), CUDA {torch.version.cuda}")
    return lines


def header() -> list[str]:
    from ocq import __version__

    nvidia = shutil.which("nvidia-smi")
    gpu = output_of([nvidia, "--query-gpu=name,driver_version,memory.total,memory.free",
                     "--format=csv,noheader"]) if nvidia else "no nvidia-smi"  # fmt: skip
    return [
        f"QuantPilot GPU report, {datetime.datetime.now():%Y-%m-%d %H:%M}",
        f"QuantPilot {__version__}, commit {output_of(['git', 'rev-parse', '--short', 'HEAD'])}",
        f"{platform.platform()}, Python {platform.python_version()}",
        f"GPU: {gpu}",
        *library_versions(),
    ]


def main() -> int:
    stamp = datetime.datetime.now().strftime("%Y%m%d-%H%M")
    report = ROOT / "results" / f"gpu-report-{stamp}.txt"
    report.parent.mkdir(exist_ok=True)
    lines = header()
    print("\n".join(lines), flush=True)
    profile = output_of([sys.executable, "-m", "ocq.cli", "profile"])
    command = [sys.executable, "-m", "pytest", "-m", "slow", "-v", "-rs", "--durations=0"]
    print(f"\nRunning {' '.join(command[1:])} ...\n", flush=True)
    tests: list[str] = []
    with subprocess.Popen(
        command, cwd=ROOT, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
        encoding="utf-8", errors="replace",
    ) as process:  # fmt: skip
        for line in process.stdout:
            print(line, end="", flush=True)
            tests.append(line.rstrip("\n"))
    summary = next(
        (line for line in reversed(tests) if re.search(r"passed|failed|error", line)), ""
    )
    # A test skipped is a format not tested on this PC: said at the top, with the reason.
    skipped = [line.split(": ", 1)[-1] for line in tests if line.startswith("SKIPPED [")]
    verdict = "PASSED" if process.returncode == 0 else "SOMETHING FAILED"
    if process.returncode == 0 and skipped:
        verdict = f"PASSED, BUT {len(skipped)} NOT RUN HERE ({'; '.join(skipped)})"
    elif process.returncode == 0:
        verdict = "ALL PASSED"
    report.write_text(
        "\n".join([f"{verdict}: {summary.strip('= ')}", "", *lines, "", "ocq profile:", profile,
                   "", *tests]) + "\n",
        encoding="utf-8",
    )  # fmt: skip
    print(f"\n{verdict}. Report: {report}")
    return process.returncode


if __name__ == "__main__":
    sys.exit(main())
