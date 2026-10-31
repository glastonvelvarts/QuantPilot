"""Post-quantization validation: the artifact exists, got smaller, loads and writes real text."""

from __future__ import annotations

from ocq.types import BenchmarkResult, OptimizationPlan, QuantizationResult, ValidationResult


def validate_artifact(
    plan: OptimizationPlan,
    quantization: QuantizationResult,
    benchmark: BenchmarkResult,
) -> ValidationResult:
    if not quantization.success or quantization.artifact_path is None:
        return ValidationResult(passed=False, checks=["FAIL  no artifact was produced"])

    size_gb = quantization.artifact_size_gb
    original_gb = plan.model.original_gb
    words = benchmark.sample_output.split()

    if original_gb is None:
        smaller = (True, f"original size unknown, {size_gb:.2f} GB not compared")
    else:
        smaller = (
            size_gb < original_gb,
            f"smaller than the original: {size_gb:.2f} GB vs {original_gb:.2f} GB",
        )
    results = [
        (size_gb > 0, f"artifact on disk: {quantization.artifact_path} ({size_gb:.2f} GB)"),
        smaller,
        (benchmark.tokens_per_second is not None, "artifact loads and generates tokens"),
        (
            len(set(words)) >= 5,
            f"output is readable text ({len(words)} words, {len(set(words))} different)",
        ),
    ]

    return ValidationResult(
        passed=all(ok for ok, _ in results),
        checks=[f"{'ok  ' if ok else 'FAIL'}  {text}" for ok, text in results],
        sample_output=benchmark.sample_output,
    )
