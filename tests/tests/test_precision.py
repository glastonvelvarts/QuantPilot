"""The GGUF conversion must start from the precision the model is really stored in."""

from pathlib import Path

import pytest
from conftest import QWEN3_FP8, QWEN_GPTQ, real_model

from ocq.model_analyzer import build_snapshot, quant_options
from ocq.model_size import gguf_weights_bytes
from ocq.quantizers.gguf import intermediate_precision
from ocq.types import FactKind, HardwareProfile


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


# Bug 3: FP8 was sized at 2 bytes per weight, and any unknown type was guessed as 2 bytes.


def test_fp8_model_is_sized_at_one_byte_per_weight():
    meta, config, tensors, dtypes = real_model(QWEN3_FP8)
    snap = build_snapshot(QWEN3_FP8, meta, config, tensors, None, dtypes=dtypes)
    assert snap.facts.precision.value == "F8_E4M3"
    assert snap.facts.precision.kind == FactKind.CROSS_CHECKED
    assert snap.facts.quantization.value == "FP8 e4m3"
    # 1.41B FP8 weights at 1 byte + 622M BF16 at 2 bytes. 2 bytes for all would be 3.78 GB.
    assert snap.original_gb == pytest.approx(2.47, abs=0.005)
    assert snap.stored_precision == ""  # not a float type llama.cpp converts from


def test_fp8_size_from_the_hub_counts_alone_is_an_estimate():
    # The HuggingFace API leaves out the FP8 scales (86,016 BF16 values), so this is a bit short.
    meta, config, _, _ = real_model(QWEN3_FP8)
    snap = build_snapshot(QWEN3_FP8, meta, config, [], None)
    assert snap.facts.stored_bytes.value == 622_453_760 * 2 + 1_409_286_144
    assert snap.facts.stored_bytes.kind == FactKind.ESTIMATE


def test_unknown_type_makes_the_size_unknown_not_guessed():
    snap = snapshot_stored_as({"BF16": 1_000_000, "F3_NEW": 1_000})
    assert snap.original_gb is None
    assert not snap.facts.stored_bytes.known
    assert "F3_NEW" in snap.facts.stored_bytes.note


def test_packed_counts_from_the_hub_are_not_turned_into_bytes():
    # The HF API counts GPTQ weights unpacked under I32: 4 bytes each would claim 5.32 GB.
    meta, config, _, _ = real_model(QWEN_GPTQ)
    snap = build_snapshot(QWEN_GPTQ, meta, config, [], None)
    assert snap.original_gb is None
    assert snap.facts.precision.value == "4-bit in I32"


def test_benchmark_runs_llama_cpp_with_the_planned_cache(tmp_path, monkeypatch):
    import json
    from types import SimpleNamespace

    from ocq.quantizers import gguf

    commands, envs = [], []
    load_log = (
        "load_tensors: offloaded 31/31 layers to GPU\n"
        "load_tensors:   CPU_Mapped model buffer size =   292.36 MiB\n"
        "load_tensors:        CUDA0 model buffer size =  1500.00 MiB\n"
    )

    def fake_run_tool(command, log, env=None):
        commands.append(command)
        envs.append(env)
        if "llama-bench" in command[0]:
            writing = {"n_prompt": 0, "n_gen": 128, "avg_ts": 100.0, "stddev_ts": 1.0}
            return json.dumps([writing]), ""
        return "The sky is blue.", load_log

    monkeypatch.setattr(gguf, "run_tool", fake_run_tool)
    monkeypatch.setattr(gguf, "tool_path", lambda name, hardware: Path(name))
    plan = SimpleNamespace(
        output_dir=tmp_path,
        hardware=SimpleNamespace(has_gpu=True),
        quant_config=SimpleNamespace(context=4096, kv_cache="q8_0"),
        predicted=None,
        model=SimpleNamespace(architecture="qwen2"),
    )
    gguf.GGUFQuantizer().benchmark(plan, "model.gguf")
    assert len(commands) == 2  # the sample answer (llama-completion), the speed (llama-bench)
    for command in commands:
        assert command[-4:] == ["-ctk", "q8_0", "-ctv", "q8_0"]
    assert envs == [{}, {}]

    # Granite 3.x MoE stops in llama.cpp b11429 with a quantized cache unless its rotation is
    # off (granite-3.1-1b-a400m-instruct Q2_K, 16k tokens, q8_0 cache, on CUDA).
    commands.clear(), envs.clear()
    plan.model = SimpleNamespace(architecture="granitemoe")
    gguf.GGUFQuantizer().benchmark(plan, "model.gguf")
    assert envs == [{"LLAMA_ATTN_ROT_DISABLE": "1"}] * 2
    plan.model = SimpleNamespace(architecture="qwen2")

    # A mixture of experts planned with the experts of 5 layers in RAM.
    commands.clear(), envs.clear()
    plan.predicted = SimpleNamespace(experts_in_ram=5)
    result = gguf.GGUFQuantizer().benchmark(plan, "model.gguf")
    completion, bench = commands
    assert completion[-2:] == ["--n-cpu-moe", "5"]
    assert bench[bench.index("-ncmoe") + 1] == "5"
    # Not a plain GPU run, so it is kept out of this PC's GPU speed history.
    assert result.device == "gpu (experts of 5 layers in ram)"
    plan.predicted = None


def test_a_benchmark_without_a_gpu_runs_on_the_cpu_and_is_recorded_so(tmp_path, monkeypatch):
    # llama.cpp's CPU build also logs "offloaded 17/17 layers to GPU" (ocq verify on Linux
    # without a GPU); only the buffers show where the weights went.
    import json
    from types import SimpleNamespace

    from ocq.quantizers import gguf

    commands, envs = [], []

    def fake_run_tool(command, log, env=None):
        commands.append(command)
        envs.append(env)
        if "llama-bench" in command[0]:
            return json.dumps([{"n_prompt": 0, "n_gen": 128, "avg_ts": 30.0, "stddev_ts": 1.0}]), ""
        return "The sky is blue.", (
            "load_tensors: offloaded 17/17 layers to GPU\n"
            "load_tensors:   CPU_Mapped model buffer size =   204.62 MiB\n"
            "load_tensors:   CPU_REPACK model buffer size =   154.12 MiB\n"
        )

    monkeypatch.setattr(gguf, "run_tool", fake_run_tool)
    monkeypatch.setattr(gguf, "tool_path", lambda name, hardware: Path(name))
    plan = SimpleNamespace(
        output_dir=tmp_path,
        hardware=SimpleNamespace(has_gpu=False),
        quant_config=SimpleNamespace(context=4096, kv_cache="f16"),
        predicted=None,
        model=SimpleNamespace(architecture="lfm2"),
    )
    result = gguf.GGUFQuantizer().benchmark(plan, "model.gguf")
    completion, bench = commands
    assert result.device == "cpu"
    assert bench[bench.index("-ngl") + 1] == "0" and bench[bench.index("-dev") + 1] == "none"
    # llama-completion is told by the environment, llama-bench by -dev (it reads no LLAMA_ARG_).
    assert envs == [{"LLAMA_ARG_DEVICE": "none"}, {}]


def test_the_benchmark_reads_prompt_speed_and_build_from_llama_bench(tmp_path, monkeypatch):
    # The real report of llama-bench b11429 -p 512 -n 128 -r 3 for Qwen3.5-0.8B Q4_K_M on an
    # M1: one entry for reading the prompt, one for writing tokens.
    from types import SimpleNamespace

    from ocq.quantizers import gguf

    report = Path(__file__).parent / "data/gguf/llama-bench-qwen3.5-0.8b-q4_k_m-metal.json"
    commands = []

    def fake_run_tool(command, log, env=None):
        commands.append(command)
        if "llama-bench" in command[0]:
            return report.read_text(), ""
        return "The sky is blue.", (
            "load_tensors: offloaded 25/25 layers to GPU\n"
            "load_tensors: MTL0_Mapped model buffer size =   500.00 MiB\n"
        )

    monkeypatch.setattr(gguf, "run_tool", fake_run_tool)
    monkeypatch.setattr(gguf, "tool_path", lambda name, hardware: Path(name))
    plan = SimpleNamespace(
        output_dir=tmp_path,
        hardware=SimpleNamespace(has_gpu=True),
        quant_config=SimpleNamespace(context=4096, kv_cache="f16"),
        predicted=None,
        model=SimpleNamespace(architecture="qwen35"),
    )
    result = gguf.GGUFQuantizer().benchmark(plan, "model.gguf")
    bench = commands[1]
    assert bench[bench.index("-p") + 1] == "512" and bench[bench.index("-n") + 1] == "128"
    assert result.tokens_per_second == pytest.approx(78.282661)
    assert result.prompt_tokens_per_second == pytest.approx(1226.169272)
    assert result.runtime_version == "b11429"
    assert "A 512 token prompt is read at 1226 tok/s." in result.notes
