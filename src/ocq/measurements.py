"""The benchmark database, and speed prediction without invented constants.

Every benchmark run is saved with what ran (model, format, runtime and its version, context,
cache) and where (this machine, and the GPU when one took part). Speed predictions have two
sources only: the physical limit (measured memory bandwidth / bytes read per token) and what
earlier runs on this same PC measured.
"""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

from ocq.types import LLAMA_CPP, HardwareProfile, RunMode

STORE = Path.home() / ".cache" / "ocq" / "measurements.json"
GB = 1024**3

UPPER_BOUND = "upper bound"
MEASURED = "from runs on this PC"


def record(
    device: str,
    weights_gb: float,
    tokens_per_second: float,
    *,
    model: str = "",
    label: str = "",
    runtime: str = "",
    context: int = 0,
    kv_cache: str = "",
    hardware: HardwareProfile | None = None,
    runtime_version: str = "",
    prompt_tps: float = 0.0,
    prompt_speeds: dict[int, float] | None = None,
    suite: str = "",
    file: str = "",
) -> None:
    """weights_gb is what one token reads: the whole file for a dense model, less for a
    mixture of experts (see ModelSnapshot.read_fraction). The other fields say what ran, so
    a recommendation can say it rests on earlier runs of the same setup. prompt_tps: how fast
    the prompt was read (PROMPT_TOKENS of it), which sets the time to the first token;
    prompt_speeds: the same at every prompt length the suite (bench.Suite) timed. file: the
    model file's name, so a run also counts for that file found on disk (local/<file>)."""
    runs = load()
    run = {"device": device, "weights_gb": weights_gb, "tps": tokens_per_second}
    run |= {"model": model, "label": label, "runtime": runtime}
    run |= {"context": context, "kv_cache": kv_cache}
    if file:
        run |= {"file": file}
    run |= {"prompt_tps": prompt_tps, "runtime_version": runtime_version}
    if prompt_speeds:
        run |= {"prompt_speeds": {str(length): tps for length, tps in prompt_speeds.items()}}
    if suite:
        run |= {"suite": suite}
    if hardware is not None:
        run |= {"machine": machine(hardware)}
        run |= {"gpu": gpu_of(hardware) if device.startswith("gpu") else ""}
    run |= {"date": datetime.now().isoformat(timespec="seconds")}
    runs.append(run)
    STORE.parent.mkdir(parents=True, exist_ok=True)
    STORE.write_text(json.dumps(runs, indent=2))


def load() -> list[dict]:
    """Every saved run. Entries that are not runs (a hand-edited or damaged file) are left out."""
    try:
        runs = json.loads(STORE.read_text())
    except (OSError, ValueError):
        return []
    if not isinstance(runs, list):
        return []
    return [run for run in runs if isinstance(run, dict) and speed_of(run) > 0]


def speed_of(run: dict) -> float:
    try:
        return float(run.get("tps") or 0)
    except (TypeError, ValueError):
        return 0.0


def machine(hardware: HardwareProfile) -> str:
    """Which PC a run was on: its CPU and RAM. Not its GPU, so that a CPU-only run
    (OCQ_CPU_ONLY) on a PC with a GPU still counts as this PC's."""
    return f"{' '.join(hardware.cpu_name.split())}, {round(hardware.total_ram_gb)} GB RAM"


def gpu_of(hardware: HardwareProfile) -> str:
    gpu = hardware.primary_gpu if hardware.has_gpu else None
    return f"{gpu.name}, {round(gpu.vram_gb or 0)} GB" if gpu else ""


def on_this_pc(run: dict, hardware: HardwareProfile | None) -> bool:
    """Runs saved before they were tagged (no machine) were all made on this PC."""
    if hardware is None:
        return True
    if run.get("machine") and run["machine"] != machine(hardware):
        return False
    return not run.get("gpu") or run["gpu"] == gpu_of(hardware)


def placed(device: str) -> str:
    """How a run was placed, whatever the details: gpu, cpu, gpu+cpu (some layers on the GPU)
    or gpu+experts (the experts of some layers in RAM)."""
    if device.startswith("gpu+cpu"):
        return "gpu+cpu"
    if "experts" in device:
        return "gpu+experts"
    return device


def placed_as(run_mode: RunMode) -> str:
    """The same for a row QuantPilot plans (QuantOption.run_mode)."""
    return {
        RunMode.GPU: "gpu",
        RunMode.GPU_FULL: "gpu",
        RunMode.CPU: "cpu",
        RunMode.CPU_ONLY: "cpu",
        RunMode.GPU_OFFLOAD: "gpu+cpu",
        RunMode.GPU_EXPERTS_RAM: "gpu+experts",
    }.get(run_mode, "")


def runs_of(
    model: str,
    label: str,
    runtime: str,
    hardware: HardwareProfile | None = None,
    placement: str = "",
) -> list[dict]:
    """Earlier runs on this PC of this model in this format and runtime, placed this way when
    placement is given (see placed). Runs saved before these fields were recorded have none,
    and never match. model is a repo id, or local/<file> for a file found on disk."""
    return [
        run
        for run in load()
        if model.lower()
        in (str(run.get("model", "")).lower(), f"local/{run.get('file', '')}".lower())
        and str(run.get("label", "")).upper() == label.upper()
        and run.get("runtime") == runtime
        and on_this_pc(run, hardware)
        and (not placement or placed(str(run.get("device", ""))) == placement)
    ]


def predict_speed(
    weights_gb: float,
    run_mode: RunMode,
    vram_gb: float,
    hardware: HardwareProfile,
    use_measurements: bool = True,
    gpu_read_speed: float | None = None,
    read_fraction: float = 1.0,
    ram_read_gb: float = 0.0,
    gpu_share: float | None = None,
) -> tuple[float, str]:
    """Returns tokens per second and where the number comes from.

    read_fraction is the share of the weights each token reads (below 1 for a mixture of
    experts). The stored runs are llama.cpp benchmarks, so they only apply to GGUF models.
    """
    if run_mode == RunMode.UNFIT or weights_gb <= 0:
        return 0.0, UPPER_BOUND

    limit = bandwidth_limit(
        weights_gb,
        run_mode,
        vram_gb,
        hardware,
        gpu_read_speed,
        read_fraction,
        ram_read_gb,
        gpu_share,
    )
    device = {RunMode.GPU: "gpu", RunMode.CPU: "cpu"}.get(run_mode)
    runs = (
        [
            run
            for run in load()
            if run.get("device") == device
            and run.get("runtime", "") in ("", LLAMA_CPP)  # the runs before it was saved: llama.cpp
            and on_this_pc(run, hardware)
        ]
        if use_measurements
        else []
    )
    fitted = fit_runs(runs)
    if fitted is None:
        return round(limit, 1), UPPER_BOUND

    fixed_seconds, seconds_per_gb = fitted
    measured = 1 / (fixed_seconds + seconds_per_gb * weights_gb * read_fraction)
    return round(min(measured, limit), 1), MEASURED


def bandwidth_limit(
    weights_gb: float,
    run_mode: RunMode,
    vram_gb: float,
    hardware: HardwareProfile,
    gpu_read_speed: float | None = None,
    read_fraction: float = 1.0,
    ram_read_gb: float = 0.0,
    gpu_share: float | None = None,
) -> float:
    """Each new token reads every weight it uses once, so memory bandwidth caps the speed.

    gpu_read_speed is how fast the GPUs in use read the weights together (see GpuPool);
    without it, the largest GPU's speed. gpu_share: the share of the layers on the GPU when
    the model does not fit (GPU_OFFLOAD); without it, as much of the weights as fit in vram_gb.
    """
    gpu = hardware.primary_gpu
    gpu_speed = gpu_read_speed or 0.0
    if gpu_read_speed is None and gpu is not None:
        gpu_speed = gpu.read_speed
    cpu_speed = (hardware.ram_bandwidth_gbs or hardware.measured_ram_bandwidth_gbs) * 1e9 / GB

    read_gb = weights_gb * read_fraction
    if run_mode == RunMode.GPU_EXPERTS_RAM and gpu_speed and cpu_speed:
        # ram_read_gb: the experts a token uses from the layers kept in RAM. The rest is in VRAM.
        return 1 / ((read_gb - ram_read_gb) / gpu_speed + ram_read_gb / cpu_speed)
    if run_mode == RunMode.GPU and gpu_speed:
        return gpu_speed / read_gb
    if run_mode == RunMode.GPU_OFFLOAD and gpu_speed and cpu_speed:
        # The part that did not fit in VRAM is read from RAM.
        on_gpu = gpu_share if gpu_share is not None else min(vram_gb, weights_gb) / weights_gb
        return 1 / (read_gb * on_gpu / gpu_speed + read_gb * (1 - on_gpu) / cpu_speed)
    if cpu_speed:
        return cpu_speed / read_gb
    return 0.0


def fit_runs(runs: list[dict]) -> tuple[float, float] | None:
    """Least squares line through (model size, seconds per token). Needs two different sizes."""
    runs = [r for r in runs if isinstance(r.get("weights_gb"), (int, float)) and speed_of(r) > 0]
    sizes = [run["weights_gb"] for run in runs]
    seconds = [1 / speed_of(run) for run in runs]
    if len(sizes) < 2 or max(sizes) - min(sizes) < 0.05:
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
