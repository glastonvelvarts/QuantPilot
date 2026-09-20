"""Tests for UniversalHardwareResolver."""

from ocq.hardware_resolver import BandwidthProfile, UniversalHardwareResolver
from ocq.types import GpuProfile, HardwareProfile


def test_resolve_apple_silicon():
    profile = UniversalHardwareResolver.resolve(
        gpu_name="Apple M1",
        cpu_name="Apple M1",
        backend="Metal",
        unified_memory=True,
    )
    assert profile.ram_bandwidth_gbs == 68.2  # rounded to 1 decimal: 68.25 -> 68.2
    assert profile.vram_bandwidth_gbs == 68.2
    assert profile.is_integrated_gpu is True
    assert profile.pcie_bandwidth_gbs == 0.0


def test_resolve_apple_m3_max():
    profile = UniversalHardwareResolver.resolve(
        gpu_name="Apple M3 Max",
        cpu_name="Apple M3 Max",
        backend="Metal",
        unified_memory=True,
    )
    assert profile.ram_bandwidth_gbs == 409.6
    assert profile.vram_bandwidth_gbs == 409.6
    assert profile.is_integrated_gpu is True


def test_resolve_nvidia_known_gpu():
    profile = UniversalHardwareResolver.resolve(
        gpu_name="NVIDIA GeForce RTX 4090",
        cpu_name="AMD Ryzen 9 7950X",
        backend="CUDA",
        unified_memory=False,
    )
    assert profile.vram_bandwidth_gbs == 1008.0
    assert profile.ram_bandwidth_gbs == 83.2  # Zen 4 DDR5
    assert profile.is_integrated_gpu is False
    assert profile.pcie_bandwidth_gbs > 0.0


def test_resolve_intel_integrated_gpu():
    profile = UniversalHardwareResolver.resolve(
        gpu_name="Intel Iris Xe Graphics",
        cpu_name="11th Gen Intel Core i7-1165G7",
        backend="CPU",
        unified_memory=False,
    )
    assert profile.is_integrated_gpu is True
    # VRAM bandwidth routes directly to system RAM bandwidth for integrated GPUs
    assert profile.vram_bandwidth_gbs == profile.ram_bandwidth_gbs


def test_resolve_profile_enriches_hardware_profile():
    raw_hw = HardwareProfile(
        total_ram_gb=64.0,
        available_ram_gb=48.0,
        cpu_cores=16,
        cpu_name="AMD Ryzen 9 5950X",
        has_gpu=True,
        backend="CUDA",
        unified_memory=False,
        gpus=(
            GpuProfile(name="NVIDIA GeForce RTX 3090", vram_gb=24.0, backend="CUDA"),
        ),
    )
    enriched = UniversalHardwareResolver.resolve_profile(raw_hw)

    assert enriched.ram_bandwidth_gbs == 51.2  # Zen 3 DDR4-3200
    assert enriched.pcie_bandwidth_gbs > 0.0
    assert len(enriched.gpus) == 1
    assert enriched.gpus[0].bandwidth_gbs == 936.0  # RTX 3090 spec
    assert enriched.primary_gpu_bandwidth_gbs == 936.0


def test_live_ram_probe():
    bw = UniversalHardwareResolver.probe_live_ram_bandwidth(size_mb=16, runs=2)
    assert isinstance(bw, float)
    assert bw > 0.0


def test_resolve_generic_cloud_vm():
    profile = UniversalHardwareResolver.resolve(
        gpu_name="",
        cpu_name="Common KVM processor",
        backend="CPU",
        unified_memory=False,
    )
    # Generic cloud VM should resolve a non-zero valid bandwidth (via live probe or baseline)
    assert profile.ram_bandwidth_gbs > 0.0
    assert profile.is_integrated_gpu is True  # No GPU -> integrated

