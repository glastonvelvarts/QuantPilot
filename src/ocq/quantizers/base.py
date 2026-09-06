"""Base stub quantizer — v1 scaffold until real backends are wired."""

from __future__ import annotations

from abc import ABC

from ocq.quantizers.protocol import Quantizer
from ocq.types import BenchmarkResult, OptimizationPlan, QuantizationResult


class StubQuantizer(Quantizer, ABC):
    """Placeholder that validates deps and documents the integration point."""

    package_extra: str = ""
    pip_hint: str = ""

    def validate(self, plan: OptimizationPlan) -> tuple[bool, str]:
        if not self.is_available():
            return False, f"{self.algorithm.value.upper()} backend not installed. Run: {self.pip_hint}"
        return True, "ok"

    def prepare(self, plan: OptimizationPlan) -> None:
        # TODO: huggingface_hub.snapshot_download(plan.model_id)
        pass

    def calibrate(self, plan: OptimizationPlan) -> None:
        # TODO: load calibration dataset (wikitext2 / pile subset)
        pass

    def quantize(self, plan: OptimizationPlan) -> None:
        # TODO: invoke auto_awq / auto_gptq / bnb quant path
        pass

    def save(self, plan: OptimizationPlan) -> str:
        out = plan.output_dir / f"{plan.model_id.replace('/', '--')}-{self.algorithm.value}"
        return str(out)

    def benchmark(self, plan: OptimizationPlan, artifact_path: str) -> BenchmarkResult:
        return BenchmarkResult(
            tokens_per_second=None,
            method=f"{self.algorithm.value}-bench-stub",
            notes=[
                "Post-quant benchmark not implemented in v1 scaffold.",
                f"Compare against llmfit hypothesis: {plan.llmfit_hypothesis_tps:.1f} tok/s",
                f"Artifact path: {artifact_path}",
            ],
        )

    def run(self, plan: OptimizationPlan) -> tuple[QuantizationResult, BenchmarkResult]:
        ok, msg = self.validate(plan)
        if not ok:
            return (
                QuantizationResult(
                    plan=plan,
                    artifact_path=None,
                    backend=self.algorithm.value,
                    success=False,
                    message=msg,
                ),
                BenchmarkResult(tokens_per_second=None, method="skipped", notes=[msg]),
            )
        return (
            QuantizationResult(
                plan=plan,
                artifact_path=None,
                backend=self.algorithm.value,
                success=False,
                message=(
                    f"{self.algorithm.value.upper()} deps found but quantize() is not implemented yet. "
                    f"See ocq/quantizers/{self.algorithm.value}.py"
                ),
            ),
            BenchmarkResult(
                tokens_per_second=None,
                method="not-implemented",
                notes=["Wire quantize/save/benchmark in the backend module."],
            ),
        )
