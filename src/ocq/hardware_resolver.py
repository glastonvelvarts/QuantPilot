"""Universal Hardware Resolver — dynamic memory and interconnect bandwidth resolution.

Determines accurate RAM bandwidth, VRAM bandwidth, and PCIe transfer rates
across Intel, AMD, NVIDIA, and Apple Silicon (Metal) across macOS, Linux,
and Windows. Combines live memory micro-probing, driver bus telemetry
(bus width x clock frequency), and architectural calculations to eliminate
hallucinations in fitment and throughput estimates.
"""

from __future__ import annotations

import ctypes
import platform
import re
import shutil
import subprocess
import time
from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ocq.types import HardwareProfile


@dataclass(frozen=True)
class BandwidthProfile:
    ram_bandwidth_gbs: float
    vram_bandwidth_gbs: float
    pcie_bandwidth_gbs: float
    is_integrated_gpu: bool
    measured_ram_bandwidth_gbs: float = 0.0
    detection_method: str = "dynamic"


# Apple Silicon architectural peak bandwidths (GB/s)
# Derived from memory bus width and LPDDR clock specifications:
# Base: 128-bit bus | Pro: 192-256-bit bus | Max: 384-512-bit bus | Ultra: Dual-die UltraFusion
_APPLE_SILICON_SPECS: dict[str, float] = {
    # M1 Family (LPDDR4X-4266)
    "m1 ultra": 819.2,
    "m1 max": 409.6,
    "m1 pro": 204.8,
    "m1": 68.25,
    # M2 Family (LPDDR5-6400)
    "m2 ultra": 819.2,
    "m2 max": 409.6,
    "m2 pro": 204.8,
    "m2": 102.4,
    # M3 Family (LPDDR5-6400)
    "m3 max": 409.6,
    "m3 pro": 153.6,
    "m3": 102.4,
    # M4 Family (LPDDR5X-7500 / 8533)
    "m4 max": 546.0,
    "m4 pro": 273.0,
    "m4": 120.0,
}

# Known GPU architectural specifications (GB/s)
# Used when drivers restrict raw hardware query access (e.g. unprivileged containers)
_KNOWN_GPU_SPECS: dict[str, float] = {
    # NVIDIA Hopper & Blackwell
    "b200": 8000.0,
    "h200": 4800.0,
    "h100 sxm": 3350.0,
    "h100 nvl": 3900.0,
    "h100 pcie": 2000.0,
    "h100": 2000.0,
    # NVIDIA Ampere Data Center
    "a100-sxm4-80gb": 2039.0,
    "a100-pcie-80gb": 1935.0,
    "a100 80gb": 1935.0,
    "a100-sxm4-40gb": 1555.0,
    "a100-pcie-40gb": 1555.0,
    "a100": 1555.0,
    "a40": 696.0,
    "a30": 933.0,
    "a16": 200.0,
    "a10": 600.0,
    # NVIDIA Ada Lovelace Data Center / Workstation
    "l40s": 864.0,
    "l40": 864.0,
    "l4": 300.0,
    "rtx 6000 ada": 960.0,
    # NVIDIA Ada Lovelace (RTX 40-series)
    "rtx 4090": 1008.0,
    "rtx 4080 super": 736.0,
    "rtx 4080": 717.0,
    "rtx 4070 ti super": 672.0,
    "rtx 4070 ti": 504.0,
    "rtx 4070 super": 504.0,
    "rtx 4070": 504.0,
    "rtx 4060 ti": 288.0,
    "rtx 4060": 272.0,
    # NVIDIA Ampere (RTX 30-series)
    "rtx 3090 ti": 1008.0,
    "rtx 3090": 936.0,
    "rtx 3080 ti": 912.0,
    "rtx 3080": 760.0,
    "rtx 3070 ti": 608.0,
    "rtx 3070": 448.0,
    "rtx 3060 ti": 448.0,
    "rtx 3060": 360.0,
    "rtx 3050": 224.0,
    # NVIDIA Turing & Volta
    "rtx 2080 ti": 616.0,
    "rtx 2080 super": 496.0,
    "rtx 2080": 448.0,
    "rtx 2070 super": 448.0,
    "rtx 2070": 448.0,
    "rtx 2060 super": 448.0,
    "rtx 2060": 336.0,
    "titan rtx": 672.0,
    "titan v": 652.8,
    "v100": 900.0,
    "t4": 320.0,
    # AMD Instinct
    "mi300x": 5300.0,
    "mi300a": 5300.0,
    "mi250x": 3200.0,
    "mi250": 3200.0,
    "mi210": 1600.0,
    "mi100": 1228.8,
    # AMD Radeon RX
    "rx 7900 xtx": 960.0,
    "rx 7900 xt": 800.0,
    "rx 7900 gre": 576.0,
    "rx 7800 xt": 624.0,
    "rx 7700 xt": 432.0,
    "rx 6950 xt": 576.0,
    "rx 6900 xt": 512.0,
    "rx 6800 xt": 512.0,
    "rx 6800": 512.0,
    "rx 6700 xt": 384.0,
    # Intel Arc
    "arc a770": 560.0,
    "arc a750": 512.0,
    "arc a580": 512.0,
}


class UniversalHardwareResolver:
    """Accurately calculates memory bandwidth and interconnect rates across any platform."""

    @classmethod
    def probe_live_ram_bandwidth(cls, size_mb: int = 64, runs: int = 3) -> float:
        """Micro-benchmarks actual achievable RAM copy bandwidth in milliseconds.

        Allocates a buffer outside of CPU L1/L2/L3 cache sizes to test true DRAM access.
        Zero dependencies, zero sudo, works on any CPU architecture (x86, ARM64, RISC-V).
        """
        try:
            size_bytes = size_mb * 1024 * 1024
            buf1 = (ctypes.c_char * size_bytes)()
            buf2 = (ctypes.c_char * size_bytes)()

            # Warmup
            ctypes.memmove(buf2, buf1, size_bytes)

            start = time.perf_counter()
            for _ in range(runs):
                ctypes.memmove(buf2, buf1, size_bytes)
            elapsed = (time.perf_counter() - start) / runs

            if elapsed <= 0:
                return 0.0

            # 1 read + 1 write across DRAM = 2 * size_bytes
            bw_gbs = (2 * size_bytes) / (elapsed * (1024**3))
            return round(bw_gbs, 1)
        except Exception:
            return 0.0

    @classmethod
    def resolve_profile(cls, hardware: HardwareProfile) -> HardwareProfile:
        """Enriches a HardwareProfile with dynamically measured and calculated bandwidths."""
        from ocq.types import GpuProfile, HardwareProfile

        primary_gpu_name = hardware.gpus[0].name if hardware.gpus else ""
        profile = cls.resolve(
            gpu_name=primary_gpu_name,
            cpu_name=hardware.cpu_name,
            backend=hardware.backend,
            unified_memory=hardware.unified_memory,
        )

        updated_gpus = []
        for g in hardware.gpus:
            g_bw, _ = cls._detect_gpu_bandwidth(
                gpu_name=g.name,
                backend=g.backend or hardware.backend,
                unified_memory=g.unified_memory or hardware.unified_memory,
                system_ram_bw=profile.ram_bandwidth_gbs,
            )
            updated_gpus.append(
                GpuProfile(
                    name=g.name,
                    vram_gb=g.vram_gb,
                    backend=g.backend,
                    bandwidth_gbs=g_bw,
                    count=g.count,
                    unified_memory=g.unified_memory or hardware.unified_memory,
                )
            )

        return HardwareProfile(
            total_ram_gb=hardware.total_ram_gb,
            available_ram_gb=hardware.available_ram_gb,
            cpu_cores=hardware.cpu_cores,
            cpu_name=hardware.cpu_name,
            has_gpu=hardware.has_gpu,
            backend=hardware.backend,
            unified_memory=hardware.unified_memory or profile.is_integrated_gpu,
            ram_bandwidth_gbs=profile.ram_bandwidth_gbs,
            pcie_bandwidth_gbs=profile.pcie_bandwidth_gbs,
            measured_ram_bandwidth_gbs=profile.measured_ram_bandwidth_gbs,
            gpus=tuple(updated_gpus),
            raw=hardware.raw,
        )

    @classmethod
    def resolve(
        cls,
        gpu_name: str = "",
        cpu_name: str = "",
        backend: str = "",
        unified_memory: bool = False,
    ) -> BandwidthProfile:
        """Calculates exact RAM, VRAM, and PCIe bandwidth for the target hardware."""
        # 1. Measure live RAM bandwidth if running on local host
        measured_ram = 0.0
        if cls._is_local_host(cpu_name):
            measured_ram = cls.probe_live_ram_bandwidth()

        # 2. Determine peak RAM bandwidth from architecture / OS telemetry
        ram_bw = cls._detect_ram_bandwidth(
            cpu_name=cpu_name,
            backend=backend,
            measured_ram=measured_ram,
        )

        # 3. Determine GPU bandwidth
        vram_bw, is_integrated = cls._detect_gpu_bandwidth(
            gpu_name=gpu_name,
            backend=backend,
            unified_memory=unified_memory,
            system_ram_bw=ram_bw,
        )

        # 4. Determine PCIe bandwidth for CPU-to-GPU offloading
        pcie_bw = cls._detect_pcie_bandwidth(
            is_integrated=is_integrated or unified_memory,
            backend=backend,
        )

        return BandwidthProfile(
            ram_bandwidth_gbs=round(ram_bw, 1),
            vram_bandwidth_gbs=round(vram_bw, 1),
            pcie_bandwidth_gbs=round(pcie_bw, 1),
            is_integrated_gpu=is_integrated,
            measured_ram_bandwidth_gbs=measured_ram,
            detection_method="live_probe" if measured_ram > 0 else "hardware_telemetry",
        )

    @classmethod
    def _is_local_host(cls, cpu_name: str) -> bool:
        """Checks whether the requested CPU profile represents the current host machine."""
        if not cpu_name:
            return True
        host_cpu = ""
        try:
            if platform.system() == "Darwin":
                host_cpu = subprocess.check_output(
                    ["sysctl", "-n", "machdep.cpu.brand_string"], text=True
                ).strip()
            elif platform.system() == "Linux":
                with open("/proc/cpuinfo") as f:
                    for line in f:
                        if "model name" in line:
                            host_cpu = line.split(":", 1)[1].strip()
                            break
        except Exception:
            pass
        if not host_cpu:
            return True
        return cpu_name.lower() in host_cpu.lower() or host_cpu.lower() in cpu_name.lower()

    @classmethod
    def _detect_ram_bandwidth(
        cls,
        cpu_name: str = "",
        backend: str = "",
        measured_ram: float = 0.0,
    ) -> float:
        """Calculates peak RAM bandwidth using OS telemetry, Apple Silicon specs, or physics."""
        system = platform.system()
        cpu_lower = cpu_name.lower()

        is_explicit_x86 = any(
            tok in cpu_lower
            for tok in ["amd", "intel", "ryzen", "xeon", "epyc", "core i", "pentium", "celeron"]
        )

        # 1. Apple Silicon (macOS Metal)
        if not is_explicit_x86 and (
            system == "Darwin" or "metal" in backend.lower() or "apple" in cpu_lower
        ):
            bw = cls._detect_apple_silicon_bandwidth(cpu_name)
            if bw > 0:
                return bw

        # 2. Linux OS telemetry
        if system == "Linux" and not is_explicit_x86:
            bw = cls._detect_linux_memory_telemetry()
            if bw > 0:
                return bw

        # 3. Windows WMI telemetry
        if system == "Windows" and not is_explicit_x86:
            bw = cls._detect_windows_memory_telemetry()
            if bw > 0:
                return bw

        # 4. Generational architecture calculation from CPU memory controller
        arch_bw = cls._calculate_ram_bandwidth_from_cpu_arch(cpu_name)
        if arch_bw > 0:
            return arch_bw

        # 5. If CPU is completely generic (e.g. cloud VM) and we measured live RAM bandwidth:
        if measured_ram > 0:
            # Memory copy achieves ~80% of peak theoretical bus bandwidth
            return round(measured_ram * 1.25, 1)

        return 38.4  # Standard baseline: dual-channel DDR4-2400

    @classmethod
    def _detect_apple_silicon_bandwidth(cls, cpu_name: str = "") -> float:
        """Determines Apple Silicon memory bandwidth based on chip family architecture."""
        name_to_check = cpu_name
        if not name_to_check:
            try:
                out = subprocess.check_output(
                    ["sysctl", "-n", "machdep.cpu.brand_string"],
                    text=True,
                    stderr=subprocess.DEVNULL,
                ).strip()
                name_to_check = out
            except Exception:
                pass

        lower_name = name_to_check.lower()

        # Match longest tokens first (e.g. 'm1 ultra' before 'm1')
        for token, bw in sorted(_APPLE_SILICON_SPECS.items(), key=lambda x: -len(x[0])):
            if token in lower_name:
                return bw

        # If on arm64 macOS but chip name unlisted, assume modern Apple Silicon baseline
        if platform.machine().lower() in ("arm64", "aarch64"):
            return 100.0

        return 0.0

    @classmethod
    def _detect_linux_memory_telemetry(cls) -> float:
        """Queries Linux memory controller telemetry (speed and channel count)."""
        # 1. dmidecode if accessible
        if shutil.which("dmidecode"):
            try:
                out = subprocess.check_output(
                    ["dmidecode", "-t", "memory"],
                    text=True,
                    stderr=subprocess.DEVNULL,
                )
                speeds = [
                    int(m.group(1))
                    for m in re.finditer(r"Speed:\s+(\d+)\s+(?:MT/s|MHz)", out)
                ]
                active_speeds = [s for s in speeds if s > 0]
                if active_speeds:
                    channels = min(len(active_speeds), 8)
                    # Bandwidth = MT/s * 8 bytes * channels / 1000.0
                    return round((max(active_speeds) * 8 * channels) / 1000.0, 1)
            except Exception:
                pass

        # 2. lshw memory query
        if shutil.which("lshw"):
            try:
                out = subprocess.check_output(
                    ["lshw", "-C", "memory", "-short"],
                    text=True,
                    stderr=subprocess.DEVNULL,
                )
                speeds = [int(m) for m in re.findall(r"(\d+)\s*(?:MHz|MT/s)", out, re.IGNORECASE)]
                if speeds:
                    return round((max(speeds) * 8 * 2) / 1000.0, 1)  # Dual channel
            except Exception:
                pass

        return 0.0

    @classmethod
    def _detect_windows_memory_telemetry(cls) -> float:
        """Queries Windows WMI CimInstance for memory speed and channel count."""
        try:
            cmd = "Get-CimInstance Win32_PhysicalMemory | Select-Object -ExpandProperty Speed"
            out = subprocess.check_output(
                ["powershell", "-NoProfile", "-Command", cmd],
                text=True,
                stderr=subprocess.DEVNULL,
            )
            speeds = [int(s) for s in re.findall(r"\d+", out)]
            if speeds and max(speeds) > 0:
                channels = max(1, len(speeds))
                return round((max(speeds) * 8 * min(channels, 8)) / 1000.0, 1)
        except Exception:
            pass
        return 0.0

    @classmethod
    def _calculate_ram_bandwidth_from_cpu_arch(cls, cpu_name: str) -> float:
        """Calculates theoretical RAM bandwidth based on CPU memory controller architecture."""
        cpu = cpu_name.lower()

        # Multi-socket / Workstation / Server
        if any(tok in cpu for tok in ["epyc", "threadripper pro", "xeon platinum", "xeon gold"]):
            return 250.0  # 8-channel server
        if "threadripper" in cpu or "xeon" in cpu:
            return 120.0  # Quad-channel workstation

        # AMD Ryzen: match model generation (e.g. Ryzen 9 7950X -> 7xxx -> Zen 4 DDR5)
        ryzen_match = re.search(r"ryzen(?:\s+[3579])?\s+(\d{4})", cpu)
        if ryzen_match:
            series = int(ryzen_match.group(1))
            if series >= 7000:
                return 83.2  # Zen 4/5: Dual-channel DDR5-5200 (5200 * 16 / 1000)
            elif series >= 3000:
                return 51.2  # Zen 2/3: Dual-channel DDR4-3200 (3200 * 16 / 1000)
            else:
                return 38.4  # Zen 1/1+: Dual-channel DDR4-2400

        # Intel Core: match model generation or Core Ultra
        if any(tok in cpu for tok in ["core ultra", "ultra 7", "ultra 9", "ultra 5"]):
            return 89.6  # Dual-channel DDR5-5600

        intel_match = re.search(r"i[3579]-(\d{4,5})", cpu)
        if intel_match:
            gen_num = int(intel_match.group(1))
            if gen_num >= 12000:
                return 89.6  # 12th/13th/14th Gen: Dual-channel DDR5-5600
            elif gen_num >= 6000:
                return 45.0  # 6th-11th Gen: Dual-channel DDR4-2933/3200
            elif gen_num >= 2000:
                return 25.6  # 2nd-4th Gen: Dual-channel DDR3-1600

        # Legacy budget CPUs
        if any(tok in cpu for tok in ["pentium", "celeron", "core 2", "duo", "athlon"]):
            return 12.8

        return 38.4

    @classmethod
    def _detect_gpu_bandwidth(
        cls,
        gpu_name: str,
        backend: str,
        unified_memory: bool,
        system_ram_bw: float,
    ) -> tuple[float, bool]:
        """Calculates VRAM bandwidth dynamically using bus width and memory clocks."""
        gpu_lower = gpu_name.lower()
        backend_lower = backend.lower()

        # 1. Unified Memory / Apple Silicon (Metal)
        if unified_memory or "metal" in backend_lower or "apple" in gpu_lower:
            return system_ram_bw, True

        # 2. Integrated Graphics (Intel UHD/Iris/Xe, AMD Radeon APU)
        is_integrated = (
            not gpu_name
            or (
                "intel" in gpu_lower
                and any(t in gpu_lower for t in ["graphics", "uhd", "iris", "hd"])
            )
            or "radeon(tm) graphics" in gpu_lower
            or "radeon vega" in gpu_lower
        )
        if is_integrated:
            return system_ram_bw, True

        # 3. NVIDIA GPU: calculate dynamically from driver hardware bus telemetry
        if shutil.which("nvidia-smi"):
            bw = cls._query_nvidia_hardware_bus()
            if bw > 0:
                return bw, False

        # 4. Known GPU architectural database
        for model_name, bw in sorted(_KNOWN_GPU_SPECS.items(), key=lambda x: -len(x[0])):
            if model_name in gpu_lower:
                return bw, False

        # 5. Discrete GPU general modern baseline (e.g. RTX 3060 tier)
        return 360.0, False

    @classmethod
    def _query_nvidia_hardware_bus(cls) -> float:
        """Calculates NVIDIA bandwidth from bus width and memory clock directly:
        Bandwidth (GB/s) = (Bus Width in Bits * Effective Clock in MHz) / 8000
        """
        try:
            # Query max memory clock and bus width
            out = subprocess.check_output(
                [
                    "nvidia-smi",
                    "--query-gpu=memory.bus_width,clocks.max.memory",
                    "--format=csv,noheader,nounits",
                ],
                text=True,
                stderr=subprocess.DEVNULL,
            ).strip()

            if not out or "N/A" in out:
                # Fallback to current memory clock
                out = subprocess.check_output(
                    [
                        "nvidia-smi",
                        "--query-gpu=memory.bus_width,clocks.current.memory",
                        "--format=csv,noheader,nounits",
                    ],
                    text=True,
                    stderr=subprocess.DEVNULL,
                ).strip()

            if out:
                line = out.split("\n")[0]
                bus_width_bits, mem_clock_mhz = map(float, [x.strip() for x in line.split(",")])
                # Physical formula: (bus_width_bits / 8) * (effective_clock_mhz / 1000)
                # DDR transfer rate is 2x memory clock frequency
                vram_bw = (bus_width_bits * (mem_clock_mhz * 2)) / 8000.0
                return round(vram_bw, 1)
        except Exception:
            pass
        return 0.0

    @classmethod
    def _detect_pcie_bandwidth(cls, is_integrated: bool = False, backend: str = "") -> float:
        """Calculates PCIe transfer rate dynamically from link width and generation."""
        if is_integrated or "metal" in backend.lower():
            return 0.0

        if shutil.which("nvidia-smi"):
            try:
                out = subprocess.check_output(
                    [
                        "nvidia-smi",
                        "--query-gpu=pcie.link.gen.max,pcie.link.width.max",
                        "--format=csv,noheader,nounits",
                    ],
                    text=True,
                    stderr=subprocess.DEVNULL,
                ).strip()

                if not out or "N/A" in out:
                    out = subprocess.check_output(
                        [
                            "nvidia-smi",
                            "--query-gpu=pcie.link.gen.current,pcie.link.width.current",
                            "--format=csv,noheader,nounits",
                        ],
                        text=True,
                        stderr=subprocess.DEVNULL,
                    ).strip()

                if out:
                    gen, width = map(int, [x.strip() for x in out.split("\n")[0].split(",")])
                    # Throughput per PCIe lane (GB/s):
                    # Gen 1=0.25, Gen 2=0.5, Gen 3=0.985, Gen 4=1.969, Gen 5=3.938
                    per_lane_bw = {1: 0.25, 2: 0.5, 3: 0.985, 4: 1.969, 5: 3.938}
                    return round(per_lane_bw.get(gen, 1.969) * width, 1)
            except Exception:
                pass

        # Standard PCIe 4.0 x16 fallback (31.5 GB/s)
        return 31.5