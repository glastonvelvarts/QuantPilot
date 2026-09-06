"""Tests for llmfit JSON bridge."""

import json
from unittest.mock import MagicMock, patch

import pytest

from ocq.llmfit_bridge import LlmfitBridge, LlmfitError

SAMPLE_SYSTEM = {
    "system": {
        "total_ram_gb": 32.0,
        "available_ram_gb": 24.0,
        "cpu_cores": 8,
        "cpu_name": "Test CPU",
        "has_gpu": True,
        "backend": "CUDA",
        "unified_memory": False,
        "gpus": [{"name": "RTX 4090", "vram_gb": 24.0, "backend": "CUDA", "count": 1}],
    }
}

SAMPLE_INFO = {
    **SAMPLE_SYSTEM,
    "models": [
        {
            "name": "Qwen/Qwen2.5-7B-Instruct",
            "provider": "Qwen",
            "parameter_count": "7B",
            "params_b": 7.0,
            "context_length": 32768,
            "use_case": "General",
            "is_moe": False,
            "best_quant": "Q4_K_M",
            "fit_level": "Good",
            "run_mode": "gpu",
            "score": 88.0,
            "estimated_tps": 40.0,
            "memory_required_gb": 5.2,
            "memory_available_gb": 24.0,
            "utilization_pct": 22.0,
            "runtime": "llamacpp",
            "estimate_confidence": "estimated",
        }
    ],
}


def _mock_run(payload: dict, returncode: int = 0):
    completed = MagicMock()
    completed.returncode = returncode
    completed.stdout = json.dumps(payload)
    completed.stderr = ""
    return completed


@patch("ocq.llmfit_bridge.subprocess.run")
def test_hardware_profile(mock_run):
    mock_run.return_value = _mock_run(SAMPLE_SYSTEM)
    bridge = LlmfitBridge(llmfit_bin="/fake/llmfit")
    hw = bridge.hardware_profile()
    assert hw.cpu_cores == 8
    assert hw.has_gpu is True


@patch("ocq.llmfit_bridge.subprocess.run")
def test_model_fit_parses_catalog_entry(mock_run):
    mock_run.return_value = _mock_run(SAMPLE_INFO)
    bridge = LlmfitBridge(llmfit_bin="/fake/llmfit")
    _hw, model = bridge.model_fit("Qwen/Qwen2.5-7B-Instruct")
    assert model.catalog_source == "llmfit"
    assert model.feasibility.estimated_tps == 40.0


@patch("ocq.llmfit_bridge.subprocess.run")
def test_model_fit_not_found(mock_run):
    mock_run.return_value = _mock_run(
        {"error": {"kind": "not_found", "message": "model missing"}},
        returncode=1,
    )
    bridge = LlmfitBridge(llmfit_bin="/fake/llmfit")
    with pytest.raises(LlmfitError, match="not_found"):
        bridge.model_fit("missing")
