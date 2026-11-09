"""Real quantization of a small model with every backend, from the Hub and as a model of your
own (a plain folder), then what it gives: its speed, its quality against the original, and
(GGUF) llama.cpp's memory against QuantPilot's prediction.

Run with: uv run pytest -m slow  (GitHub runs the GGUF one every night; the NVIDIA ones run on
a PC with an NVIDIA GPU: make test-gpu)
"""

import pytest
from typer.testing import CliRunner

from ocq import bench, cli, quality
from ocq.hardware import detect_hardware
from ocq.pipeline import run_optimize
from ocq.quantizers import get_quantizer
from ocq.recommend import RecommendationRequest, kept_of, recommend
from ocq.types import OptimizationGoal, QuantAlgorithm, QuantChoices

MODEL = "HuggingFaceTB/SmolLM2-135M-Instruct"
# The label each format is saved and recommended under: they must match for the measured
# quality to reach the recommendations.
LABELS = {
    QuantAlgorithm.GGUF: "Q4_K_M",
    QuantAlgorithm.AWQ: "AWQ 4-bit",
    QuantAlgorithm.GPTQ: "GPTQ 4-bit",
    QuantAlgorithm.NVFP4: "NVFP4 4-bit",
    QuantAlgorithm.BNB: "BNB 4-bit",
}


def own_model(folder):
    """SmolLM2 as a model of your own: its files in a plain folder, as save_pretrained leaves
    them (no Hugging Face cache, no repo name)."""
    import shutil

    from huggingface_hub import snapshot_download

    snapshot = snapshot_download(MODEL, allow_patterns=["*.json", "*.safetensors", "*.txt"])
    shutil.copytree(snapshot, folder, symlinks=False)
    return folder


@pytest.mark.slow
@pytest.mark.parametrize("source", ["hub", "own"], ids=["hub", "own folder"])
@pytest.mark.parametrize("algorithm", list(QuantAlgorithm), ids=lambda a: a.value)
def test_quantize_measure_and_compare_with_the_original(algorithm, source, tmp_path):
    hardware = detect_hardware()
    ready, reason = get_quantizer(algorithm).check(hardware)
    if not ready:
        pytest.skip(reason)
    model = MODEL if source == "hub" else str(own_model(tmp_path / "my-smollm2"))
    expected = MODEL if source == "hub" else "local/my-smollm2"

    choices = QuantChoices(
        algorithm=algorithm, bits=4, calibration_samples=16, calibration_seq_len=256
    )
    if algorithm == QuantAlgorithm.GGUF:
        choices = QuantChoices(algorithm=algorithm, gguf_type="Q4_K_M")
    # Not ocq-output/<org>--<name>/: what QuantPilot made says itself where it came from.
    made = tmp_path / "somewhere"
    result = run_optimize(model, OptimizationGoal.BALANCED, made, choices=choices)

    assert result.quantization.success, result.quantization.message
    assert result.validation.passed, result.validation.checks
    assert result.benchmark.tokens_per_second > 0
    assert result.benchmark.time_to_first_token_s > 0
    artifact = result.quantization.artifact_path

    if algorithm == QuantAlgorithm.GGUF:
        # llama.cpp's own memory numbers against QuantPilot's prediction, on this PC.
        verified = CliRunner().invoke(cli.app, ["verify", str(artifact)], env={"COLUMNS": "200"})
        assert verified.exit_code == 0, verified.output

    # Its speed (GGUF), its quality, and the original's (made once from the Hugging Face cache).
    outcome = bench.run_benchmark(str(artifact), hardware, piece=quality.FIXED)
    assert outcome.model_id == expected  # the repo, or your own folder
    assert outcome.label.startswith(LABELS[algorithm])
    if algorithm == QuantAlgorithm.GGUF:
        assert outcome.speed.tokens_per_second > 0
        assert set(outcome.speed.prompt_speeds) == {512, 4096}
    assert 5 < outcome.quality.value < 60  # SmolLM2-135M: about 19 in 16 bits
    assert outcome.kept is not None, outcome.note
    assert outcome.kept.reference_label in ("BF16", "F16", "F32", "Q8_0")
    assert 0.2 < outcome.kept.kept <= 1.0

    # The recommendation for this model now uses what was measured.
    if algorithm == QuantAlgorithm.GGUF and source == "hub":
        result = recommend(RecommendationRequest(models=(MODEL,)), hardware)
        q4 = next(r.candidate for r in result.ranked if r.candidate.option.label == "Q4_K_M")
        kept, measured = kept_of(q4)
        assert measured is not None and kept == pytest.approx(outcome.kept.kept)
