"""Speed prediction without invented constants.

Two sources only: the physical limit (measured memory bandwidth / bytes read per token) and what
earlier benchmark runs on this same PC actually measured.
"""

from __future__ import annotations

import json
from pathlib import Path

from ocq.types import HardwareProfile, RunMode

STORE = Path.home() / ".cache" / "ocq" / "measurements.json"
GB = 1024**3

UPPER_BOUND = "upper bound"
MEASURED = "from runs on this PC"


def record(device: str, weights_gb: float, tokens_per_second: float) -> None:
    runs = load()
    runs.append({"device": device, "weights_gb": weights_gb, "tps": tokens_per_second})
    STORE.parent.mkdir(parents=True, exist_ok=True)
    STORE.write_text(json.dumps(runs, indent=2))


def load() -> list[dict]:
    try:
        return json.loads(STORE.read_text())
    except (OSError, ValueError):
        return []


def predict_speed(
    weights_gb: float,
    run_mode: RunMode,
    vram_gb: float,
    hardware: HardwareProfile,
    use_measurements: bool = True,
) -> tuple[float, str]:
    """Returns tokens per second and where the number comes from.

    The stored runs are llama.cpp benchmarks, so they only apply to GGUF models.
    """
    if run_mode == RunMode.UNFIT or weights_gb <= 0:
        return 0.0, UPPER_BOUND

    limit = bandwidth_limit(weights_gb, run_mode, vram_gb, hardware)
    device = {RunMode.GPU: "gpu", RunMode.CPU: "cpu"}.get(run_mode)
    runs = [run for run in load() if run["device"] == device] if use_measurements else []
    fitted = fit_runs(runs)
    if fitted is None:
        return round(limit, 1), UPPER_BOUND

    fixed_seconds, seconds_per_gb = fitted
    measured = 1 / (fixed_seconds + seconds_per_gb * weights_gb)
    return round(min(measured, limit), 1), MEASURED


def bandwidth_limit(
    weights_gb: float, run_mode: RunMode, vram_gb: float, hardware: HardwareProfile
) -> float:
    """Each new token reads every weight once, so memory bandwidth caps the speed."""
    gpu = hardware.primary_gpu
    gpu_speed = 0.0
    if gpu is not None:
        gpu_speed = gpu.measured_bandwidth_gbs or gpu.bandwidth_gbs * 1e9 / GB
    cpu_speed = (hardware.ram_bandwidth_gbs or hardware.measured_ram_bandwidth_gbs) * 1e9 / GB

    if run_mode == RunMode.GPU and gpu_speed:
        return gpu_speed / weights_gb
    if run_mode == RunMode.GPU_OFFLOAD and gpu_speed and cpu_speed:
        on_gpu = min(vram_gb, weights_gb)
        return 1 / (on_gpu / gpu_speed + (weights_gb - on_gpu) / cpu_speed)
    if cpu_speed:
        return cpu_speed / weights_gb
    return 0.0


def fit_runs(runs: list[dict]) -> tuple[float, float] | None:
    """Least squares line through (model size, seconds per token). Needs two different sizes."""
    sizes = [run["weights_gb"] for run in runs]
    seconds = [1 / run["tps"] for run in runs if run["tps"] > 0]
    if len(seconds) != len(sizes) or len(sizes) < 2 or max(sizes) - min(sizes) < 0.05:
        return None

    mean_size = sum(sizes) / len(sizes)
    mean_seconds = sum(seconds) / len(seconds)
    spread = sum((size - mean_size) ** 2 for size in sizes)
    slope = (
        sum((s - mean_size) * (t - mean_seconds) for s, t in zip(sizes, seconds, strict=True))
        / spread
    )
    fixed = mean_seconds - slope * mean_size
    if slope <= 0:
        return None
    return max(fixed, 0.0), slope
