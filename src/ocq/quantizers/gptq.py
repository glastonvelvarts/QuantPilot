"""GPTQ: layer by layer quantization that corrects the error it introduces."""

from __future__ import annotations

from ocq.quantizers.oneshot import OneShotQuantizer
from ocq.types import QuantAlgorithm


class GptqQuantizer(OneShotQuantizer):
    algorithm = QuantAlgorithm.GPTQ
    summary = "4 or 8 bit, calibrated (GPU or CPU). For GPU serving with vLLM or transformers."
    install_hint = "uv sync --extra gptq"
    symmetric = True

    def recipe(self, scheme) -> list:
        from llmcompressor.modifiers.gptq import GPTQModifier

        return [GPTQModifier(config_groups={"group_0": scheme}, ignore=["lm_head"])]
