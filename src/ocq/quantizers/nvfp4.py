"""NVFP4: 4-bit floats with two levels of scales. Blackwell GPUs compute in it natively."""

from __future__ import annotations

from contextlib import contextmanager

from ocq.quantizers.oneshot import OneShotQuantizer
from ocq.types import OptimizationPlan, QuantAlgorithm, QuantConfig


class NvFp4Quantizer(OneShotQuantizer):
    algorithm = QuantAlgorithm.NVFP4
    summary = "4-bit floats, two levels of scales. Native on Blackwell GPUs, served with vLLM."
    install_hint = "uv sync --extra nvfp4"

    def artifact_name(self, config: QuantConfig) -> str:
        return "nvfp4" if config.fp4_activations else "nvfp4a16"

    def calibrate(self, plan: OptimizationPlan) -> None:
        # Only FP4 activations need calibration data, to find their scale.
        if plan.quant_config.fp4_activations:
            super().calibrate(plan)

    def build_recipe(self, model, plan: OptimizationPlan) -> list:
        from llmcompressor.modifiers.quantization import QuantizationModifier

        scheme = "NVFP4" if plan.quant_config.fp4_activations else "NVFP4A16"
        return [QuantizationModifier(targets="Linear", scheme=scheme, ignore=["lm_head"])]

    def quantize(self, plan: OptimizationPlan) -> None:
        with without_apple_gpu():
            super().quantize(plan)


@contextmanager
def without_apple_gpu():
    """llm-compressor moves the work to an Apple GPU when there is one, and Apple GPUs cannot
    store the 8-bit float scales NVFP4 needs. Hide the Apple GPU so the CPU does it."""
    import torch

    if not torch.backends.mps.is_available():
        yield
        return
    mps_available, accelerator_available = torch.mps.is_available, torch.accelerator.is_available
    torch.mps.is_available = lambda: False
    torch.accelerator.is_available = lambda *args, **kwargs: False
    try:
        yield
    finally:
        torch.mps.is_available = mps_available
        torch.accelerator.is_available = accelerator_available
