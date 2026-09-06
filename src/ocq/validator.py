"""Post-quantization validation."""

from __future__ import annotations

from ocq.types import OptimizationPlan, ValidationResult


def validate_artifact(plan: OptimizationPlan, artifact_path: str | None) -> ValidationResult:
    """Sanity-check the quantized artifact loads and produces output."""

    checks: list[str] = []

    if artifact_path is None:
        return ValidationResult(
            passed=False,
            checks=["artifact missing"],
        )

    checks.append(f"artifact path exists: {artifact_path}")
    # TODO: load with transformers/llama.cpp and run a fixed prompt
    checks.append("load test: not implemented (v1 scaffold)")
    checks.append("generation sanity: not implemented (v1 scaffold)")

    return ValidationResult(
        passed=False,
        checks=checks,
        sample_output=None,
    )
