"""Plain words for the TUI: how a fit, a speed or a place to run is shown to people.

The CLI prints the technical form (flags, bits, "<= 477"); the app shows these instead.
"""

from __future__ import annotations

import re

from rich.text import Text

from ocq import measurements
from ocq.types import FitLevel, RunMode

FIT_WORDS = {
    FitLevel.PERFECT: ("Fits easily", "green"),
    FitLevel.GOOD: ("Fits", "green"),
    FitLevel.MARGINAL: ("Tight", "yellow"),
    FitLevel.TOO_TIGHT: ("Too big", "red"),
}

PLACES = {
    RunMode.GPU: "GPU",
    RunMode.GPU_FULL: "GPU",
    RunMode.GPU_OFFLOAD: "GPU + RAM",
    RunMode.GPU_EXPERTS_RAM: "GPU + RAM",
    RunMode.CPU: "CPU",
    RunMode.CPU_ONLY: "CPU",
    RunMode.UNFIT: "-",
}


def fit(level: FitLevel) -> Text:
    word, color = FIT_WORDS.get(level, (str(level), ""))
    return Text(word, style=color)


def place(option) -> str:
    where = PLACES.get(option.run_mode, str(option.run_mode))
    if getattr(option, "gpu_count", 0) > 1:
        where = where.replace("GPU", f"{option.gpu_count} GPUs")
    return where


def speed(option) -> str:
    """'~40 tok/s' when learned from runs on this PC, 'up to 40 tok/s' for the memory limit."""
    if not option.runnable or option.estimated_tps <= 0:
        return "-"
    if getattr(option, "speed_basis", "") == measurements.MEASURED:
        return f"~{option.estimated_tps:.0f} tok/s"
    return f"up to {option.estimated_tps:.0f} tok/s"


def gb(value: float | None) -> str:
    if value is None:
        return "-"
    return f"{value:.1f} GB" if value >= 1 else f"{value * 1000:.0f} MB"


def context(tokens: int) -> str:
    return f"{tokens // 1024}K tokens" if tokens % 1024 == 0 else f"{tokens:,} tokens"


_COMMAND_HINT = re.compile(r"\s*(Run:|Use:)\s.*$|\s*\((?:-|ocq )[^)]*\)")
_NOT_INSTALLED = re.compile(r"^(\w+) backend not installed\.?$")


def warning(text: str) -> str:
    """A plan warning in plain words, without the command line some of them end with."""
    text = _COMMAND_HINT.sub("", text).strip()
    missing = _NOT_INSTALLED.match(text)
    if missing:
        return f"{missing.group(1)} tools are not installed on this PC."
    return text


def hardware(hw) -> str:
    """One line, for example: Apple M1 | 16 GB RAM | Apple M1 GPU, 16 GB."""
    parts = [hw.cpu_name or "CPU", f"{hw.total_ram_gb:.0f} GB RAM"]
    for gpu in hw.gpus:
        count = f"{gpu.count} x " if gpu.count > 1 else ""
        vram = f", {gpu.vram_gb:.0f} GB" if gpu.vram_gb else ""
        parts.append(f"{count}{gpu.name}{vram}")
    if not hw.gpus:
        parts.append("no GPU")
    return " | ".join(parts)
