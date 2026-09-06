"""Tests for HuggingFace model fallback."""

from ocq.model_analyzer import _snapshot_from_hf
from ocq.types import HardwareProfile


def _cuda_hw() -> HardwareProfile:
    return HardwareProfile(
        total_ram_gb=32,
        available_ram_gb=24,
        cpu_cores=8,
        cpu_name="CPU",
        has_gpu=True,
        backend="CUDA",
        unified_memory=False,
    )


def test_hf_fallback_any_repo():
    meta = {
        "id": "some-org/Some-Model-7B-Instruct",
        "safetensors": {"total": 7_000_000_000},
        "pipeline_tag": "text-generation",
        "tags": [],
    }
    snap = _snapshot_from_hf("some-org/Some-Model-7B-Instruct", meta, _cuda_hw())
    assert snap.catalog_source == "huggingface"
    assert snap.params_b == 7.0
    assert snap.feasibility.fit_level in ("Perfect", "Good", "Marginal", "Too Tight")
