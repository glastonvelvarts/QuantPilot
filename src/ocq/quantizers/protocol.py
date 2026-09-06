"""Pluggable quantizer interface."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import ClassVar

from ocq.types import BenchmarkResult, OptimizationPlan, QuantAlgorithm, QuantizationResult


class Quantizer(ABC):
    """v1 quantizer contract — validate → prepare → calibrate → quantize → save → benchmark."""

    algorithm: ClassVar[QuantAlgorithm]

    @classmethod
    def is_available(cls) -> bool:
        return False

    @abstractmethod
    def validate(self, plan: OptimizationPlan) -> tuple[bool, str]:
        """Check deps, GPU memory, and model compatibility."""

    @abstractmethod
    def prepare(self, plan: OptimizationPlan) -> None:
        """Download weights / create workspace."""

    @abstractmethod
    def calibrate(self, plan: OptimizationPlan) -> None:
        """Run calibration forward passes."""

    @abstractmethod
    def quantize(self, plan: OptimizationPlan) -> None:
        """Execute quantization."""

    @abstractmethod
    def save(self, plan: OptimizationPlan) -> str:
        """Write artifact; return path."""

    @abstractmethod
    def benchmark(self, plan: OptimizationPlan, artifact_path: str) -> BenchmarkResult:
        """Measure real throughput of the produced artifact."""

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

        plan.output_dir.mkdir(parents=True, exist_ok=True)
        self.prepare(plan)
        self.calibrate(plan)
        self.quantize(plan)
        artifact = self.save(plan)
        bench = self.benchmark(plan, artifact)
        return (
            QuantizationResult(
                plan=plan,
                artifact_path=plan.output_dir / artifact if not artifact.startswith("/") else None,
                backend=self.algorithm.value,
                success=True,
                message=f"Quantized with {self.algorithm.value}",
            ),
            bench,
        )
