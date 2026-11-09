"""Installing a quantization tool from the app: its Python libraries, and llama.cpp for GGUF.

Each tool is one of QuantPilot's extras (gguf, awq, gptq, bnb, nvfp4). The libraries go into
the Python that runs QuantPilot:
- a checkout of this repo run with uv: `uv sync --inexact --extra <tool>`, which keeps what is
  already installed and uses the project's own sources (CUDA PyTorch on Windows);
- anything else: the extra's requirements with `uv pip install` or `pip install`.
"""

from __future__ import annotations

import importlib
import re
import shutil
import subprocess
import sys
from collections.abc import Callable
from dataclasses import dataclass
from importlib import metadata
from pathlib import Path

from ocq import DISTRIBUTION
from ocq.quantizers import get_quantizer
from ocq.tui import words
from ocq.types import HardwareProfile, QuantAlgorithm

READY = "ready"
MISSING = "missing"  # can be installed from the app
UNSUPPORTED = "unsupported"  # this PC cannot run it
BLOCKED = "blocked"  # installed, but something outside Python is missing

NAMES = {
    QuantAlgorithm.GGUF: "GGUF",
    QuantAlgorithm.AWQ: "AWQ",
    QuantAlgorithm.GPTQ: "GPTQ",
    QuantAlgorithm.BNB: "BNB",
    QuantAlgorithm.NVFP4: "NVFP4",
}


@dataclass(frozen=True)
class ToolState:
    algorithm: QuantAlgorithm
    state: str
    message: str  # shown next to the tool

    @property
    def name(self) -> str:
        return NAMES[self.algorithm]


def tool_state(algorithm: QuantAlgorithm, hardware: HardwareProfile) -> ToolState:
    quantizer = get_quantizer(algorithm)
    if quantizer.needs_cuda and not hardware.has_cuda:
        return ToolState(algorithm, UNSUPPORTED, "Needs an NVIDIA GPU")
    ready, detail = quantizer.check(hardware)
    if ready:
        return ToolState(algorithm, READY, "Ready")
    if not quantizer.is_available():
        return ToolState(algorithm, MISSING, "Not installed")
    return ToolState(algorithm, BLOCKED, words.plain(detail))


def project_root() -> Path | None:
    """The repo folder when QuantPilot runs from a checkout (an editable install), else None."""
    root = Path(__file__).resolve().parents[3]
    pyproject = root / "pyproject.toml"
    try:
        named = rf'^name = "{DISTRIBUTION}"'
        if pyproject.is_file() and re.search(named, pyproject.read_text(), re.M):
            return root
    except OSError:
        pass
    return None


def requirements(extra: str) -> list[str]:
    """The packages an extra adds, read from QuantPilot's installed metadata."""
    found = []
    for requirement in metadata.requires(DISTRIBUTION) or []:
        spec, _, marker = requirement.partition(";")
        if re.search(rf"""extra\s*==\s*["']{re.escape(extra)}["']""", marker):
            found.append(spec.strip())
    return found


def install_command(extra: str) -> tuple[list[str], Path | None]:
    """(command, folder to run it in)."""
    uv = shutil.which("uv")
    root = project_root()
    if uv and root is not None:
        return [uv, "sync", "--inexact", "--extra", extra], root
    packages = requirements(extra)
    if uv:
        return [uv, "pip", "install", "--python", sys.executable, *packages], None
    return [sys.executable, "-m", "pip", "install", *packages], None


def install(
    algorithm: QuantAlgorithm, hardware: HardwareProfile, say: Callable[[str], None]
) -> ToolState:
    """Installs one tool and returns its state afterwards. `say` gets each progress line."""
    command, folder = install_command(algorithm.value)
    say(f"Installing the {NAMES[algorithm]} tools ...")
    process = subprocess.Popen(
        command,
        cwd=folder,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        stdin=subprocess.DEVNULL,
        text=True,
        encoding="utf-8",
        errors="replace",
    )
    assert process.stdout is not None
    for line in process.stdout:
        if line.strip():
            say(line.rstrip())
    if process.wait() != 0:
        raise RuntimeError(f"The installer stopped with exit code {process.returncode}.")
    importlib.invalidate_caches()  # let this process find the new packages
    if algorithm == QuantAlgorithm.GGUF:
        from ocq.quantizers.llama_cpp import tool_path

        say("Downloading llama.cpp ...")
        tool_path("llama-quantize", hardware)
    return tool_state(algorithm, hardware)
