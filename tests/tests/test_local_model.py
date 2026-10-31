"""Local HuggingFace-format folders (a user's own fine-tune or pretrained model)."""

import json
from pathlib import Path

import pytest
from conftest import QWEN, QWEN_CONFIG, QWEN_GPTQ, real_model, write_safetensors

from ocq.local_model import fetch_local_model_files, parse_model_ref, validate_hf_folder
from ocq.model_analyzer import quant_options, resolve_model
from ocq.types import FactKind, HardwareProfile, OcqError


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


def make_model_folder(folder: Path, tensors, dtype: str = "BF16") -> Path:
    folder.mkdir(parents=True, exist_ok=True)
    (folder / "config.json").write_text(json.dumps(QWEN_CONFIG))
    write_safetensors(folder / "model.safetensors", tensors, dtype)
    return folder


def test_parse_hf_id():
    model_id, local = parse_model_ref("Qwen/Qwen2.5-0.5B-Instruct")
    assert model_id == "Qwen/Qwen2.5-0.5B-Instruct"
    assert local is None


def test_parse_local_folder(tmp_path, qwen_tensors):
    folder = make_model_folder(tmp_path / "my model", qwen_tensors[:3])
    model_id, local = parse_model_ref(str(folder))
    assert model_id == "local/my model"
    assert local == folder.resolve()


def test_missing_folder_raises():
    with pytest.raises(OcqError, match="not found"):
        parse_model_ref("./this-folder-does-not-exist-ocq")


def test_folder_without_config_is_rejected(tmp_path, qwen_tensors):
    write_safetensors(tmp_path / "model.safetensors", qwen_tensors[:2])
    assert "config.json" in validate_hf_folder(tmp_path)


def test_adapter_only_folder_says_to_merge_first(tmp_path):
    (tmp_path / "adapter_config.json").write_text("{}")
    write_safetensors(tmp_path / "adapter_model.safetensors", [("lora_A.weight", (8, 64))])
    assert "LoRA adapter" in validate_hf_folder(tmp_path)


def test_bin_only_folder_says_to_resave(tmp_path):
    (tmp_path / "config.json").write_text(json.dumps(QWEN_CONFIG))
    (tmp_path / "pytorch_model.bin").write_bytes(b"not safetensors")
    assert ".bin" in validate_hf_folder(tmp_path)


def test_local_snapshot_is_exact_and_matches_the_hub_numbers(tmp_path, qwen_tensors):
    folder = make_model_folder(tmp_path / "finetune", qwen_tensors)
    snap = resolve_model(str(folder), make_hardware())

    assert snap.catalog_source == "local"
    assert snap.weights_dir == folder.resolve()
    assert snap.stored_precision == "BF16"
    assert snap.params_b == pytest.approx(1.5437, abs=0.0001)

    by_label = {option.label: option for option in quant_options(snap, make_hardware())}
    assert by_label["BF16"].row_kind == "stored"
    assert by_label["Q4_K_M"].exact
    # Same value llama-quantize reports for Qwen2.5-1.5B from the Hub.
    assert by_label["Q4_K_M"].weights_gb * 1024 == pytest.approx(934.69, abs=0.01)
    assert by_label["AWQ 4-bit"].weights_gb < by_label["BF16"].weights_gb


def test_32_bit_fine_tune_is_seen_as_32_bit(tmp_path, qwen_tensors):
    folder = make_model_folder(tmp_path / "f32", qwen_tensors, dtype="F32")
    snap = resolve_model(str(folder), make_hardware())
    assert snap.stored_precision == "F32"
    assert quant_options(snap, make_hardware())[0].label == "F32"


def test_sharded_model_reads_every_shard(tmp_path, qwen_tensors):
    tmp_path.joinpath("config.json").write_text(json.dumps(QWEN_CONFIG))
    half = len(qwen_tensors) // 2
    write_safetensors(tmp_path / "model-00001-of-00002.safetensors", qwen_tensors[:half])
    write_safetensors(tmp_path / "model-00002-of-00002.safetensors", qwen_tensors[half:])
    weight_map = {name: "model-00001-of-00002.safetensors" for name, _ in qwen_tensors[:half]}
    weight_map |= {name: "model-00002-of-00002.safetensors" for name, _ in qwen_tensors[half:]}
    (tmp_path / "model.safetensors.index.json").write_text(json.dumps({"weight_map": weight_map}))

    _, tensors, _ = fetch_local_model_files(tmp_path)
    assert len(tensors) == len(qwen_tensors)


def test_leftover_adapter_file_is_not_counted_as_model_weights(tmp_path, qwen_tensors):
    folder = make_model_folder(tmp_path / "messy", qwen_tensors)
    write_safetensors(folder / "adapter_model.safetensors", [("lora_A.weight", (8, 1536))])
    _, tensors, _ = fetch_local_model_files(folder)
    assert len(tensors) == len(qwen_tensors)


def test_plan_warns_when_the_tokenizer_was_not_saved(tmp_path, qwen_tensors):
    from ocq.plan import generate_plan
    from ocq.types import OptimizationGoal

    folder = make_model_folder(tmp_path / "no-tokenizer", qwen_tensors)
    snap = resolve_model(str(folder), make_hardware())
    plan = generate_plan(snap.model_id, make_hardware(), snap, OptimizationGoal.BALANCED, tmp_path)
    assert any("no tokenizer files" in warning for warning in plan.warnings)

    (folder / "tokenizer.json").write_text("{}")
    plan = generate_plan(snap.model_id, make_hardware(), snap, OptimizationGoal.BALANCED, tmp_path)
    assert not any("no tokenizer files" in warning for warning in plan.warnings)


def test_group_size_is_settled_in_the_plan(smollm_tensors, tmp_path):
    from ocq.engine import decide_quant_config
    from ocq.types import OptimizationGoal, QuantAlgorithm, QuantChoices

    # SmolLM2 layers are 576 and 1536 wide: 128 does not divide 576, 64 does.
    folder = make_model_folder(tmp_path / "smol", smollm_tensors)
    snap = resolve_model(str(folder), make_hardware())
    config = decide_quant_config(
        make_hardware(), snap, OptimizationGoal.BALANCED, QuantChoices(algorithm=QuantAlgorithm.AWQ)
    )
    assert config.group_size == 64


def make_real_folder(folder: Path, repo: str) -> Path:
    """A local folder with the real config.json and safetensors headers of a Hub repo."""
    _, config, tensors, dtypes = real_model(repo)
    folder.mkdir(parents=True, exist_ok=True)
    (folder / "config.json").write_text(json.dumps(config))
    write_safetensors(folder / "model.safetensors", tensors, dtypes)
    return folder


def test_local_folder_facts_come_from_its_own_files(tmp_path):
    snap = resolve_model(str(make_real_folder(tmp_path / "qwen", QWEN)), None)
    # Only one source on disk, so exact, not cross-checked.
    assert snap.facts.parameters.value == 1_543_714_304
    assert snap.facts.parameters.kind == FactKind.EXACT
    assert snap.facts.parameters.source == "safetensors headers"
    assert snap.facts.stored_bytes.value == 3_087_428_608


def test_local_gptq_folder_is_seen_as_quantized(tmp_path):
    from ocq.plan import generate_plan
    from ocq.types import OptimizationGoal

    folder = make_real_folder(tmp_path / "gptq", QWEN_GPTQ)
    snap = resolve_model(str(folder), make_hardware())
    assert snap.facts.quantization.value == "GPTQ 4-bit"
    # Unpacked by us and nothing to check it against: an estimate.
    assert snap.facts.parameters.value == 1_543_714_304
    assert snap.facts.parameters.kind == FactKind.ESTIMATE
    assert snap.facts.stored_bytes.value == 1_149_760_512

    with pytest.raises(OcqError, match="already quantized"):
        generate_plan(snap.model_id, make_hardware(), snap, OptimizationGoal.BALANCED, tmp_path)


def test_inspect_prints_every_fact(tmp_path):
    from typer.testing import CliRunner

    from ocq.cli import app

    folder = make_real_folder(tmp_path / "gptq", QWEN_GPTQ)
    result = CliRunner().invoke(app, ["inspect", str(folder)], terminal_width=200)
    assert result.exit_code == 0, result.output
    assert "GPTQ 4-bit, 1.54B params (estimate), 1.07 GB of weights" in result.output
    facts = ("Parameters", "Precision", "Stored bytes", "Architecture", "Context length", "KV")
    for line in facts:
        assert line in result.output
    assert "1,543,714,304" in result.output
    assert "32,768" in result.output
