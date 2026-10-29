"""Quantizer registry."""

from __future__ import annotations

from ocq.quantizers.protocol import Quantizer
from ocq.types import BackendStatus, HardwareProfile, QuantAlgorithm


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
    if algorithm == QuantAlgorithm.GGUF:
        from ocq.quantizers.gguf import GGUFQuantizer

        return GGUFQuantizer()
    if algorithm == QuantAlgorithm.NVFP4:
        from ocq.quantizers.nvfp4 import NvFp4Quantizer

        return NvFp4Quantizer()
    raise ValueError(f"Unknown quantizer: {algorithm}")


def backend_status(hardware: HardwareProfile) -> list[BackendStatus]:
    statuses = []
    for algorithm in QuantAlgorithm:
        ready, detail = get_quantizer(algorithm).check(hardware)
        statuses.append(BackendStatus(algorithm=algorithm, ready=ready, detail=detail))
    return statuses


def available_backends() -> list[QuantAlgorithm]:
    return [algorithm for algorithm in QuantAlgorithm if get_quantizer(algorithm).is_available()]


__all__ = ["Quantizer", "available_backends", "backend_status", "get_quantizer"]
