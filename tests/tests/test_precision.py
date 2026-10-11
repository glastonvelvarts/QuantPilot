"""The GGUF conversion must start from the precision the model is really stored in."""

import pytest

from ocq.model_analyzer import build_snapshot, quant_options
from ocq.model_size import gguf_weights_bytes
from ocq.quantizers.gguf import intermediate_precision
from ocq.types import HardwareProfile


def make_hardware() -> HardwareProfile:
    return HardwareProfile(
        total_ram_gb=32,
        available_ram_gb=32,
        cpu_cores=8,
        cpu_name="CPU",
        has_gpu=False,
        backend="CPU",
        unified_memory=False,
    )


def snapshot_stored_as(parameters: dict, tensors=()):
    meta = {"safetensors": {"parameters": parameters, "total": sum(parameters.values())}}
    return build_snapshot("org/model", meta, {}, list(tensors), make_hardware())


@pytest.mark.parametrize("stored", ["F32", "BF16", "F16"])
def test_stored_precision_is_read_from_the_repo(stored):
    assert snapshot_stored_as({stored: 1_000_000}).stored_precision == stored


def test_stored_precision_is_the_type_most_weights_use():
    assert snapshot_stored_as({"F32": 10, "BF16": 1_000_000, "I64": 5}).stored_precision == "BF16"


def test_16_bit_models_convert_at_16_bit():
    assert intermediate_precision("BF16") == "bf16"
    assert intermediate_precision("F16") == "f16"


def test_32_bit_and_unknown_models_convert_at_32_bit():
    assert intermediate_precision("F32") == "f32"
    assert intermediate_precision("") == "f32"


def test_original_row_shows_the_real_precision_and_size(smollm_tensors):
    as_f32 = snapshot_stored_as({"F32": 134_515_008}, smollm_tensors)
    as_bf16 = snapshot_stored_as({"BF16": 134_515_008}, smollm_tensors)
    original_f32 = quant_options(as_f32, make_hardware())[0]
    original_bf16 = quant_options(as_bf16, make_hardware())[0]

    assert original_f32.label == "F32"
    assert original_bf16.label == "BF16"
    assert original_f32.weights_gb == pytest.approx(as_f32.original_gb, rel=0.001)
    assert original_bf16.weights_gb == pytest.approx(as_bf16.original_gb, rel=0.001)
    assert original_f32.weights_gb == pytest.approx(2 * original_bf16.weights_gb, rel=0.001)


def test_quantized_size_does_not_depend_on_the_stored_precision(smollm_tensors):
    # llama-quantize produced byte-identical Q4_K_M files from f16, bf16 and f32 copies of
    # SmolLM2-135M, so the predicted size must not change either.
    assert gguf_weights_bytes(smollm_tensors, "Q4_K_M", 3) == 103_668_480
