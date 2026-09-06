"""bitsandbytes backend — install with pip install ocq[bnb]."""

from __future__ import annotations

from ocq.quantizers.base import StubQuantizer
from ocq.types import QuantAlgorithm


class BnbQuantizer(StubQuantizer):
    algorithm = QuantAlgorithm.BNB
    package_extra = "bnb"
    pip_hint = "uv sync --extra bnb"

    @classmethod
    def is_available(cls) -> bool:
        try:
            import bitsandbytes  # noqa: F401, PLC0415

            return True
        except ImportError:
            return False
