"""The steps every quantization backend goes through."""

from __future__ import annotations

import traceback
from abc import ABC, abstractmethod
from pathlib import Path
from typing import ClassVar

from ocq.types import (
    BenchmarkResult,
    HardwareProfile,
    OptimizationPlan,
    QuantAlgorithm,
    QuantizationResult,
)

BENCHMARK_PROMPT = "Explain in a few sentences why the sky is blue."
BENCHMARK_TOKENS = 128


class Quantizer(ABC):
    """check -> prepare -> calibrate -> quantize -> save -> benchmark."""

    algorithm: ClassVar[QuantAlgorithm]
    summary: ClassVar[str] = ""
    install_hint: ClassVar[str] = ""
    needs_cuda: ClassVar[bool] = False

    @classmethod
    def is_available(cls) -> bool:
        return False

    def check(self, hardware: HardwareProfile) -> tuple[bool, str]:
        name = self.algorithm.value.upper()
        if not self.is_available():
            return False, f"{name} backend not installed. Run: {self.install_hint}"
        if self.needs_cuda and not hardware.has_cuda:
            return False, f"{name} needs an NVIDIA GPU, none was found on this PC."
        return True, self.summary

    def validate(self, plan: OptimizationPlan) -> tuple[bool, str]:
        return self.check(plan.hardware)

    @abstractmethod
    def prepare(self, plan: OptimizationPlan) -> None:
        """Download the weights and anything else the backend needs."""

    @abstractmethod
    def calibrate(self, plan: OptimizationPlan) -> None:
        """Load calibration data, for backends that use it."""

    @abstractmethod
    def quantize(self, plan: OptimizationPlan) -> None:
        """Run the quantization and write the artifact."""

    @abstractmethod
    def save(self, plan: OptimizationPlan) -> str:
        """Return the path of the artifact."""

    @abstractmethod
    def benchmark(self, plan: OptimizationPlan, artifact_path: str) -> BenchmarkResult:
        """Load the artifact, generate text, and time it."""

    def run(self, plan: OptimizationPlan) -> tuple[QuantizationResult, BenchmarkResult]:
        ok, message = self.validate(plan)
        if not ok:
            return self.failed(plan, message)

        plan.output_dir.mkdir(parents=True, exist_ok=True)
        try:
            self.prepare(plan)
            self.calibrate(plan)
            self.quantize(plan)
            artifact = Path(self.save(plan))
        except Exception as exc:
            return self.failed(plan, self.describe_error(plan, exc))

        try:
            bench = self.benchmark(plan, str(artifact))
        except Exception as exc:
            bench = BenchmarkResult(
                tokens_per_second=None,
                method="failed",
                notes=[f"Benchmark failed: {self.describe_error(plan, exc)}"],
            )

        result = QuantizationResult(
            plan=plan,
            artifact_path=artifact,
            backend=self.algorithm.value,
            success=True,
            message=f"Quantized with {plan.quant_config.summary()}",
            artifact_size_gb=size_gb(artifact),
        )
        return result, bench

    def failed(
        self, plan: OptimizationPlan, message: str
    ) -> tuple[QuantizationResult, BenchmarkResult]:
        result = QuantizationResult(
            plan=plan,
            artifact_path=None,
            backend=self.algorithm.value,
            success=False,
            message=message,
        )
        return result, BenchmarkResult(tokens_per_second=None, method="skipped")

    def describe_error(self, plan: OptimizationPlan, exc: Exception) -> str:
        log = plan.output_dir / "error.log"
        log.write_text(traceback.format_exc(), encoding="utf-8")
        return f"{type(exc).__name__}: {exc}\n(full traceback: {log})"


def size_gb(path: Path) -> float:
    """Size of the weights. A model folder also holds tokenizer files, which are not counted."""
    files = [path] if path.is_file() else list(path.rglob("*.safetensors"))
    return sum(f.stat().st_size for f in files) / 1024**3
