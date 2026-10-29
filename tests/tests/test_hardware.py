"""Tests for hardware detection without llmfit."""

import pytest

from ocq.hardware import detect_hardware, parse_nvidia_smi, probe_hardware
from ocq.types import GpuProfile


def test_probe_reads_this_machine():
    hw = probe_hardware()
    assert hw.source == "native"
    assert hw.total_ram_gb > 0
    assert 0 < hw.available_ram_gb <= hw.total_ram_gb
    assert hw.cpu_cores >= 1
    assert hw.cpu_name
    assert hw.os_name in ("Windows", "Linux", "Darwin")
    assert hw.has_gpu == bool(hw.gpus)


def test_detect_falls_back_when_llmfit_is_missing():
    hw = detect_hardware(llmfit_bin="/no/such/llmfit")
    assert hw.source == "native"
    assert hw.total_ram_gb > 0
    assert hw.ram_bandwidth_gbs > 0


def test_nvidia_smi_rows_with_and_without_compute_capability():
    new_driver = parse_nvidia_smi("NVIDIA GeForce RTX 5090, 32607, 30000, 12.0\n")
    assert new_driver[0].name == "NVIDIA GeForce RTX 5090"
    assert new_driver[0].compute_capability == 12.0
    assert new_driver[0].fp4_tensor_cores

    old_driver = parse_nvidia_smi("NVIDIA H100 80GB HBM3, 81559, 80000\n[N/A]\n")
    assert len(old_driver) == 1
    assert old_driver[0].compute_capability is None
    assert old_driver[0].vram_gb == round(81559 / 1024, 2)
    assert not old_driver[0].fp4_tensor_cores


@pytest.mark.parametrize(
    "name, blackwell",
    [
        ("NVIDIA GeForce RTX 5090", True),
        ("NVIDIA GeForce RTX 5070 Laptop GPU", True),
        ("NVIDIA B200", True),
        ("NVIDIA GB10", True),
        ("NVIDIA RTX PRO 6000 Blackwell Workstation Edition", True),
        ("NVIDIA RTX 5000 Ada Generation", False),
        ("NVIDIA GeForce RTX 4090", False),
        ("NVIDIA H100 80GB HBM3", False),
    ],
)
def test_blackwell_is_recognized_by_name(name, blackwell):
    assert GpuProfile(name=name, backend="CUDA").fp4_tensor_cores == blackwell
