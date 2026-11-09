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


def layers(option, num_layers: int) -> str:
    """'14 of 29 layers on GPU' when llama.cpp splits the model between the GPU and RAM
    (the output layer counts as one, as llama.cpp counts)."""
    if not getattr(option, "gpu_layers", 0) or not num_layers:
        return "-"
    return f"{option.gpu_layers} of {num_layers + 1} layers on GPU"


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


_SENTENCE = re.compile(r"(?<=[.!?])\s+")
_COMMANDS = re.compile(r"\bocq [a-z]|(?:^|\s)--?[a-z]|\buv (sync|pip)|\bpip install|\bsudo ")


def plain(text: str) -> str:
    """A message from QuantPilot's core without the commands and flags meant for the CLI:
    they are cut from the end of a sentence, and a sentence that is only advice to run a
    command is left out."""
    text = _COMMAND_HINT.sub("", text).strip()
    kept = [s for s in _SENTENCE.split(text) if s and not _COMMANDS.search(s)]
    return " ".join(kept) or text


def warning(text: str) -> str:
    """A plan warning in plain words."""
    text = plain(text)
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


CACHE_WORDS = {"f16": "full cache", "q8_0": "half-size cache", "q4_0": "quarter-size cache"}
CONFIDENCE_STYLES = {"Low": "red", "Medium": "yellow", "High": "green", "Very High": "green"}
REASON_MARKS = {"good": ("✓", "green"), "tradeoff": ("△", "yellow"), "bad": ("✗", "red")}


def cache(kv_cache: str) -> str:
    return CACHE_WORDS.get(kv_cache, f"{kv_cache} cache")


def confidence(level) -> Text:
    word = str(level)
    return Text(word, style=CONFIDENCE_STYLES.get(word, ""))


def reasons(items, sure=None) -> Text:
    """The ✓ △ ✗ lines of a recommendation, then what its confidence rests on."""
    text = Text()
    for reason in items:
        mark, style = REASON_MARKS[reason.kind.value]
        text.append(f"{mark} ", style)
        text.append(f"{plain(reason.text)}\n")
    if sure is not None:
        text.append("\nConfidence: ")
        text.append_text(confidence(sure.level))
        for evidence in sure.evidence:
            mark, style = ("✓", "green") if evidence.known else ("✗", "dim")
            text.append(f"\n  {mark} ", style)
            text.append(evidence.text, "dim")
    return text
