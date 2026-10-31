"""Find out what this PC has: llmfit when it works, a plain OS probe when it does not."""

from __future__ import annotations

import json
import platform
import shutil
import subprocess
import time
from dataclasses import replace
from pathlib import Path

import psutil

from ocq.hardware_resolver import UniversalHardwareResolver, host_cpu_name
from ocq.llmfit_bridge import LlmfitBridge, LlmfitError
from ocq.types import GpuProfile, HardwareProfile

GB = 1024**3
GPU_PROBE_CACHE = Path.home() / ".cache" / "ocq" / "gpu_bandwidth.json"


def detect_hardware(llmfit_bin: str | None = None) -> HardwareProfile:
    try:
        profile = LlmfitBridge(llmfit_bin).hardware_profile()
    except LlmfitError:
        profile = UniversalHardwareResolver.resolve_profile(probe_hardware())
    return with_live_gpu_readings(profile)


def probe_hardware() -> HardwareProfile:
    memory = psutil.virtual_memory()
    gpus = nvidia_gpus() or apple_gpus(memory.total)
    return HardwareProfile(
        total_ram_gb=round(memory.total / GB, 2),
        available_ram_gb=round(memory.available / GB, 2),
        cpu_cores=psutil.cpu_count(logical=True) or 1,
        cpu_name=host_cpu_name(),
        has_gpu=bool(gpus),
        backend=gpus[0].backend if gpus else "CPU",
        unified_memory=any(gpu.unified_memory for gpu in gpus),
        gpus=tuple(gpus),
        os_name=platform.system(),
        source="native",
    )


def nvidia_gpus() -> list[GpuProfile]:
    if not shutil.which("nvidia-smi"):
        return []
    # Drivers older than 510 do not know compute_cap and reject the whole query.
    output = query_nvidia_smi("name,memory.total,memory.free,compute_cap")
    if output is None:
        output = query_nvidia_smi("name,memory.total,memory.free")
    return parse_nvidia_smi(output or "")


def query_nvidia_smi(fields: str) -> str | None:
    try:
        return subprocess.check_output(
            ["nvidia-smi", f"--query-gpu={fields}", "--format=csv,noheader,nounits"],
            text=True,
            stderr=subprocess.DEVNULL,
            timeout=15,
        )
    except (subprocess.SubprocessError, OSError):
        return None


def parse_nvidia_smi(output: str) -> list[GpuProfile]:
    """Rows of name, total MiB, free MiB and, from newer drivers, compute capability."""
    gpus = []
    for line in output.strip().splitlines():
        parts = [part.strip() for part in line.split(",")]
        capability = None
        if len(parts) >= 4:
            try:
                capability = float(parts[-1])
                parts = parts[:-1]
            except ValueError:
                pass
        try:
            total_mib, free_mib = float(parts[-2]), float(parts[-1])
        except (ValueError, IndexError):
            continue
        gpus.append(
            GpuProfile(
                name=", ".join(parts[:-2]),
                vram_gb=round(total_mib / 1024, 2),
                backend="CUDA",
                free_vram_gb=round(free_mib / 1024, 2),
                compute_capability=capability,
            )
        )
    return gpus


def apple_gpus(total_ram_bytes: int) -> list[GpuProfile]:
    if platform.system() != "Darwin" or platform.machine() != "arm64":
        return []
    return [
        GpuProfile(
            name=host_cpu_name(),
            vram_gb=round(total_ram_bytes / GB, 1),
            backend="Metal",
            unified_memory=True,
        )
    ]


def with_live_gpu_readings(profile: HardwareProfile) -> HardwareProfile:
    """Adds what only the running machine knows: free VRAM and measured VRAM bandwidth.

    nvidia-smi lists every card on its own line, so the NVIDIA cards are replaced by that list,
    one entry per card in card order. Matching by name would give two identical cards the
    readings of one of them, and llmfit lists identical cards once with a count.
    """
    if not profile.has_cuda:
        return profile

    live = nvidia_gpus()
    if not live:
        gpus = [
            replace(g, measured_bandwidth_gbs=measured_gpu_bandwidth(g.name)) for g in profile.gpus
        ]
        return replace(profile, gpus=tuple(gpus))

    spec = {gpu.name: gpu.bandwidth_gbs for gpu in profile.gpus}
    cards = [
        replace(
            card,
            bandwidth_gbs=spec.get(card.name, 0.0),
            measured_bandwidth_gbs=measured_gpu_bandwidth(card.name),
        )
        for card in live
    ]
    others = [gpu for gpu in profile.gpus if "cuda" not in gpu.backend.lower()]
    return replace(profile, gpus=tuple(cards + others))


def measured_gpu_bandwidth(gpu_name: str) -> float:
    """Measured once per GPU and remembered, because loading PyTorch takes a few seconds."""
    try:
        cache = json.loads(GPU_PROBE_CACHE.read_text())
    except (OSError, ValueError):
        cache = {}
    if gpu_name not in cache:
        measured = probe_gpu_bandwidth()
        if measured <= 0:
            return 0.0
        cache[gpu_name] = measured
        GPU_PROBE_CACHE.parent.mkdir(parents=True, exist_ok=True)
        GPU_PROBE_CACHE.write_text(json.dumps(cache, indent=2))
    return cache[gpu_name]


def probe_gpu_bandwidth(size_mb: int = 512, runs: int = 5) -> float:
    """Copies a buffer inside VRAM and times it, the same idea as the RAM probe."""
    try:
        import torch

        if not torch.cuda.is_available():
            return 0.0
        size = size_mb * 1024 * 1024
        source = torch.ones(size, dtype=torch.uint8, device="cuda")
        target = torch.empty_like(source)
        target.copy_(source)
        torch.cuda.synchronize()

        fastest = float("inf")
        for _ in range(runs):
            start = time.perf_counter()
            target.copy_(source)
            torch.cuda.synchronize()
            fastest = min(fastest, time.perf_counter() - start)

        del source, target
        torch.cuda.empty_cache()
        return round(2 * size / fastest / GB, 1)
    except (ImportError, RuntimeError):
        return 0.0
