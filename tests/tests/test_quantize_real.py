"""Real quantization of a small model with every backend. Run with: uv run pytest -m slow"""

import pytest

from ocq.hardware import detect_hardware
from ocq.pipeline import run_optimize
from ocq.quantizers import get_quantizer
from ocq.types import OptimizationGoal, QuantAlgorithm, QuantChoices

MODEL = "HuggingFaceTB/SmolLM2-135M-Instruct"


@pytest.mark.slow
@pytest.mark.parametrize("algorithm", list(QuantAlgorithm), ids=lambda a: a.value)
def test_quantize_small_model(algorithm, tmp_path):
    ready, reason = get_quantizer(algorithm).check(detect_hardware())
    if not ready:
        pytest.skip(reason)

    choices = QuantChoices(
        algorithm=algorithm, bits=4, calibration_samples=16, calibration_seq_len=256
    )
    result = run_optimize(MODEL, OptimizationGoal.BALANCED, tmp_path, choices=choices)

    assert result.quantization.success, result.quantization.message
    assert result.validation.passed, result.validation.checks
    assert result.benchmark.tokens_per_second > 0
    assert result.benchmark.time_to_first_token_s > 0
