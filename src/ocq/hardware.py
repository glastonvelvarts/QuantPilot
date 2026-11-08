"""Find out what this PC has: llmfit when it works, a plain OS probe when it does not."""

from __future__ import annotations

import ctypes
import json
import os
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
    if cpu_only():
        return replace(profile, has_gpu=False, backend="CPU", unified_memory=False, gpus=())
    return with_live_gpu_readings(profile)


def cpu_only() -> bool:
    """OCQ_CPU_ONLY=1: plan and run as on a PC without a GPU (llama.cpp gets --device none)."""
    return os.environ.get("OCQ_CPU_ONLY", "").strip().lower() not in ("", "0", "false", "no")


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
        # Without nvidia-smi only the CUDA driver knows what is free (without either, the total
        # stands in). It is then also the figure for expert plans: nothing reads what WDDM has
        # really left, so wddm stays off.
        cuda = cuda_memory_mib() or []
        nvidia = [g for g in profile.gpus if "cuda" in g.backend.lower()]
        free = {}
        if len(cuda) == len(nvidia):
            for card, (free_mib, total_mib) in zip(nvidia, cuda, strict=True):
                if card.vram_gb and abs(total_mib / 1024 - card.vram_gb) <= 0.02 * card.vram_gb:
                    free[id(card)] = round(free_mib / 1024, 2)
        gpus = [
            replace(
                g,
                measured_bandwidth_gbs=measured_gpu_bandwidth(g.name),
                free_vram_gb=free.get(id(g), g.free_vram_gb),
                cuda_free_gb=free.get(id(g), g.cuda_free_gb),
            )
            for g in profile.gpus
        ]
        return replace(profile, gpus=tuple(gpus))

    spec = {gpu.name: gpu.bandwidth_gbs for gpu in profile.gpus}
    cuda = cuda_memory_mib() or []
    wddm = platform.system() == "Windows"
    cards = []
    for index, card in enumerate(live):
        cuda_free = None
        # CUDA lists the cards in nvidia-smi's order (CUDA_DEVICE_ORDER=PCI_BUS_ID); the total
        # must agree too, or the reading belongs to another card.
        if len(cuda) == len(live) and card.vram_gb:
            free_mib, total_mib = cuda[index]
            if abs(total_mib / 1024 - card.vram_gb) <= 0.02 * card.vram_gb:
                cuda_free = round(free_mib / 1024, 2)
        cards.append(
            replace(
                card,
                bandwidth_gbs=spec.get(card.name, 0.0),
                measured_bandwidth_gbs=measured_gpu_bandwidth(card.name),
                cuda_free_gb=cuda_free,
                wddm=wddm,
            )
        )
    others = [gpu for gpu in profile.gpus if "cuda" not in gpu.backend.lower()]
    return replace(profile, gpus=tuple(cards + others))


def cuda_memory_mib() -> list[tuple[float, float]] | None:
    """(free, total) MiB of each card as CUDA reports it once a context exists, as a runtime
    sees it: through the NVIDIA driver (cuMemGetInfo), in about half a second. On Windows the
    free figure is fixed (7,106 MiB on an 8 GB RTX 4060 Laptop, whatever else runs). None when
    the driver library is missing or fails."""
    names = ["nvcuda.dll"] if platform.system() == "Windows" else ["libcuda.so.1", "libcuda.so"]
    driver = None
    for name in names:
        try:
            driver = ctypes.CDLL(name)
            break
        except OSError:
            continue
    if driver is None:
        return None
    os.environ.setdefault("CUDA_DEVICE_ORDER", "PCI_BUS_ID")
    count = ctypes.c_int()
    if driver.cuInit(0) != 0 or driver.cuDeviceGetCount(ctypes.byref(count)) != 0:
        return None
    readings = []
    for index in range(count.value):
        device, context = ctypes.c_int(), ctypes.c_void_p()
        free, total = ctypes.c_size_t(), ctypes.c_size_t()
        if driver.cuDeviceGet(ctypes.byref(device), index) != 0:
            return None
        if driver.cuDevicePrimaryCtxRetain(ctypes.byref(context), device) != 0:
            return None
        try:
            if driver.cuCtxPushCurrent_v2(context) != 0:
                return None
            ok = driver.cuMemGetInfo_v2(ctypes.byref(free), ctypes.byref(total)) == 0
            driver.cuCtxPopCurrent_v2(ctypes.byref(ctypes.c_void_p()))
        finally:
            driver.cuDevicePrimaryCtxRelease_v2(device)
        if not ok:
            return None
        readings.append((free.value / 2**20, total.value / 2**20))
    return readings


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
