"""Tests for hardware detection without llmfit."""

from unittest import mock

import pytest
from conftest import RTX_3060, RTX_3090, RTX_3090_BUSY, nvidia_pc

from ocq import hardware
from ocq.hardware import detect_hardware, parse_nvidia_smi, probe_hardware, with_live_gpu_readings
from ocq.types import LAYER_SPLIT, ONE_GPU, TENSOR_PARALLEL, GpuProfile, HardwareProfile


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


# Several GPUs


def live_readings(profile, *lines):
    with (
        mock.patch.object(hardware, "nvidia_gpus", return_value=parse_nvidia_smi("\n".join(lines))),
        mock.patch.object(hardware, "measured_gpu_bandwidth", return_value=0.0),
    ):
        return with_live_gpu_readings(profile)


def test_identical_cards_each_keep_their_own_live_reading():
    # Matched by name, both cards used to get the second card's free VRAM.
    stale = nvidia_pc(RTX_3090, RTX_3090)
    live = live_readings(stale, RTX_3090, RTX_3090_BUSY)
    assert [card.free_vram_gb for card in live.gpus] == [23.44, 22.46]


def test_llmfit_group_becomes_one_entry_per_card():
    # llmfit lists 2 identical cards once: vram_gb is one card's, count is 2.
    grouped = GpuProfile("NVIDIA GeForce RTX 3090", 24.0, "CUDA", 936.0, count=2)
    profile = HardwareProfile(64, 60, 16, "CPU", True, "CUDA", False, gpus=(grouped,))
    assert profile.free_vram_gb == 48.0  # was 24: only the largest card counted
    live = live_readings(profile, RTX_3090, RTX_3090_BUSY)
    assert [card.count for card in live.gpus] == [1, 1]
    assert [card.bandwidth_gbs for card in live.gpus] == [936.0, 936.0]
    assert live.free_vram_gb == pytest.approx(23.44 + 22.46)


def test_layer_split_adds_up_the_free_vram_of_every_card():
    pool = nvidia_pc(RTX_3090, RTX_3060).gpu_pool(LAYER_SPLIT)
    assert pool.split == LAYER_SPLIT
    assert pool.free_gb == pytest.approx(23.44 + 11.72)
    # Each card reads its share in turn: 2/3 at 936 GB/s, 1/3 at 360 GB/s.
    seconds = (23.44 / 35.16) / (936e9 / 1024**3) + (11.72 / 35.16) / (360e9 / 1024**3)
    assert pool.read_speed == pytest.approx(1 / seconds)


def test_tensor_parallel_needs_the_card_count_to_divide_the_heads():
    three = nvidia_pc(RTX_3090, RTX_3090, RTX_3090)
    assert three.gpu_pool(TENSOR_PARALLEL, attention_heads=96).count == 3
    assert three.gpu_pool(TENSOR_PARALLEL, attention_heads=64).count == 2
    assert three.gpu_pool(TENSOR_PARALLEL, attention_heads=None).count == 1


def test_tensor_parallel_reads_in_parallel_but_holds_only_the_smallest_slice():
    pair = nvidia_pc(RTX_3090, RTX_3090).gpu_pool(TENSOR_PARALLEL, attention_heads=64)
    assert pair.free_gb == pytest.approx(2 * 23.44)
    assert pair.read_speed == pytest.approx(2 * 936e9 / 1024**3)
    # 2 x the 3060's 11.7 GB is no more than the 3090 alone, so one card it is.
    mixed = nvidia_pc(RTX_3090, RTX_3060).gpu_pool(TENSOR_PARALLEL, attention_heads=64)
    assert mixed.split == ONE_GPU
    assert mixed.cards[0].name == "NVIDIA GeForce RTX 3090"
