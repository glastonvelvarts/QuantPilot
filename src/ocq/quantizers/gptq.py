"""AutoGPTQ backend — install with pip install ocq[gptq]."""

from __future__ import annotations

from ocq.quantizers.base import StubQuantizer
from ocq.types import QuantAlgorithm


class GptqQuantizer(StubQuantizer):
    algorithm = QuantAlgorithm.GPTQ
    package_extra = "gptq"
    pip_hint = "uv sync --extra gptq"

    @classmethod
    def is_available(cls) -> bool:
        try:
            import auto_gptq  # noqa: F401, PLC0415

            return True
        except ImportError:
            return False
