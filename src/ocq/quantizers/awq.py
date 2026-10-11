"""AWQ: activation aware weight quantization."""

from __future__ import annotations

from ocq.quantizers.oneshot import OneShotQuantizer
from ocq.types import QuantAlgorithm


class AwqQuantizer(OneShotQuantizer):
    algorithm = QuantAlgorithm.AWQ
    summary = "4 bit, calibrated on an NVIDIA GPU. For GPU serving with vLLM or transformers."
    install_hint = "uv sync --extra awq"
    needs_cuda = True
    symmetric = False

    def recipe(self, scheme) -> list:
        from llmcompressor.modifiers.quantization import QuantizationModifier
        from llmcompressor.modifiers.transform.awq import AWQModifier

        return [
            AWQModifier(),
            QuantizationModifier(config_groups={"group_0": scheme}, ignore=["lm_head"]),
        ]
