"""Quantizer registry."""

from __future__ import annotations

from ocq.quantizers.protocol import Quantizer
from ocq.types import QuantAlgorithm


def get_quantizer(algorithm: QuantAlgorithm) -> Quantizer:
    if algorithm == QuantAlgorithm.AWQ:
        from ocq.quantizers.awq import AwqQuantizer

        return AwqQuantizer()
    if algorithm == QuantAlgorithm.GPTQ:
        from ocq.quantizers.gptq import GptqQuantizer

        return GptqQuantizer()
    if algorithm == QuantAlgorithm.BNB:
        from ocq.quantizers.bnb import BnbQuantizer

        return BnbQuantizer()
    msg = f"Unknown quantizer: {algorithm}"
    raise ValueError(msg)


def available_backends() -> list[QuantAlgorithm]:
    from ocq.quantizers.awq import AwqQuantizer
    from ocq.quantizers.bnb import BnbQuantizer
    from ocq.quantizers.gptq import GptqQuantizer

    out: list[QuantAlgorithm] = []
    for cls in (AwqQuantizer, GptqQuantizer, BnbQuantizer):
        if cls.is_available():
            out.append(cls.algorithm)
    return out


__all__ = ["Quantizer", "available_backends", "get_quantizer"]
