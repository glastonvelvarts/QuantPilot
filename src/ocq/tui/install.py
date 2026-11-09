"""Installing a quantization tool from the app: its Python libraries, and llama.cpp for GGUF.

Each tool is one of QuantPilot's extras (gguf, awq, gptq, bnb, nvfp4), installed the way
QuantPilot itself was (ocq/installation.py): uv tool install, uv sync in a checkout, or uv pip
/ pip into its environment.
"""

from __future__ import annotations

import importlib
import os
import subprocess
from collections.abc import Callable
from dataclasses import dataclass

from ocq import processes
from ocq.installation import install_command, requirements  # noqa: F401 (requirements: tests)
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


def install(
    algorithm: QuantAlgorithm, hardware: HardwareProfile, say: Callable[[str], None]
) -> ToolState:
    """Installs one tool and returns its state afterwards. `say` gets each progress line."""
    # A uv tool on Windows: the running ocq.exe cannot be replaced, so into its environment.
    command, folder = install_command(algorithm.value, hardware, in_place=os.name == "nt")
    say(f"Installing the {NAMES[algorithm]} tools ...")
    process = processes.started(
        subprocess.Popen(
            command,
            cwd=folder,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            stdin=subprocess.DEVNULL,
            text=True,
            encoding="utf-8",
            errors="replace",
        )
    )
    assert process.stdout is not None
    try:
        for line in process.stdout:
            if line.strip():
                say(line.rstrip())
        code = process.wait()
    finally:
        processes.ended(process)
    if code != 0:
        raise RuntimeError(f"The installer stopped with exit code {process.returncode}.")
    importlib.invalidate_caches()  # let this process find the new packages
    if algorithm == QuantAlgorithm.GGUF:
        from ocq.quantizers.llama_cpp import tool_path

        say("Downloading llama.cpp ...")
        tool_path("llama-quantize", hardware)
    return tool_state(algorithm, hardware)
