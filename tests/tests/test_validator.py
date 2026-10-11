"""Tests for post-quantization validation."""

from pathlib import Path

from ocq.plan import generate_plan
from ocq.types import (
    BenchmarkResult,
    HardwareProfile,
    ModelSnapshot,
    OptimizationGoal,
    QuantizationResult,
)
from ocq.validator import validate_artifact

GOOD_TEXT = "The sky looks blue because air scatters short blue wavelengths more than red ones."


def make_plan(tmp_path: Path):
    hw = HardwareProfile(
        total_ram_gb=16,
        available_ram_gb=12,
        cpu_cores=8,
        cpu_name="CPU",
        has_gpu=False,
        backend="CPU",
        unified_memory=False,
    )
    model = ModelSnapshot(model_id="org/model", params_b=1.0)
    return generate_plan("org/model", hw, model, OptimizationGoal.BALANCED, tmp_path)


def make_result(plan, size_gb: float, success: bool = True) -> QuantizationResult:
    return QuantizationResult(
        plan=plan,
        artifact_path=plan.output_dir / "model.gguf" if success else None,
        backend="gguf",
        success=success,
        message="",
        artifact_size_gb=size_gb,
    )


def test_good_artifact_passes(tmp_path):
    plan = make_plan(tmp_path)
    bench = BenchmarkResult(tokens_per_second=50.0, method="test", sample_output=GOOD_TEXT)
    assert validate_artifact(plan, make_result(plan, 0.6), bench).passed


def test_failed_quantization_fails(tmp_path):
    plan = make_plan(tmp_path)
    bench = BenchmarkResult(tokens_per_second=None, method="skipped")
    assert not validate_artifact(plan, make_result(plan, 0.0, success=False), bench).passed


def test_artifact_that_did_not_shrink_fails(tmp_path):
    plan = make_plan(tmp_path)
    bench = BenchmarkResult(tokens_per_second=50.0, method="test", sample_output=GOOD_TEXT)
    assert not validate_artifact(plan, make_result(plan, 2.5), bench).passed


def test_garbage_output_fails(tmp_path):
    plan = make_plan(tmp_path)
    bench = BenchmarkResult(tokens_per_second=50.0, method="test", sample_output="the the the the")
    assert not validate_artifact(plan, make_result(plan, 0.6), bench).passed
