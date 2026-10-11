"""Tests for hardware detection without llmfit."""

from ocq.hardware import detect_hardware, probe_hardware


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
