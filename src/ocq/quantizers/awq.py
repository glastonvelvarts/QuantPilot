"""AutoAWQ backend — install with pip install ocq[awq]."""

from __future__ import annotations

from ocq.quantizers.base import StubQuantizer
from ocq.types import QuantAlgorithm


class AwqQuantizer(StubQuantizer):
    algorithm = QuantAlgorithm.AWQ
    package_extra = "awq"
    pip_hint = "uv sync --extra awq"

    @classmethod
    def is_available(cls) -> bool:
        try:
            import awq  # noqa: F401, PLC0415

            return True
        except ImportError:
            return False
