"""Tests for plan generation."""

from pathlib import Path

from ocq.plan import generate_plan
from ocq.types import (
    Fact,
    FactKind,
    FeasibilityEstimate,
    HardwareProfile,
    ModelFacts,
    ModelSnapshot,
    OptimizationGoal,
    QuantAlgorithm,
    QuantChoices,
)


def make_hardware() -> HardwareProfile:
    return HardwareProfile(
        total_ram_gb=32,
        available_ram_gb=24,
        cpu_cores=8,
        cpu_name="CPU",
        has_gpu=True,
        backend="CUDA",
        unified_memory=False,
    )


def make_model() -> ModelSnapshot:
    return ModelSnapshot(
        model_id="meta-llama/Llama-3.1-8B-Instruct",
        provider="Meta",
        parameter_count="8B",
        params_b=8.0,
        context_length=8192,
        use_case="General",
        is_moe=False,
        llmfit_best_quant="Q4_K_M",
        feasibility=FeasibilityEstimate(
            quant_label="Q4_K_M",
            fit_level="Perfect",
            run_mode="gpu",
            score=90.0,
            estimated_tps=50.0,
            memory_required_gb=5.0,
            memory_available_gb=24.0,
            utilization_pct=21.0,
            runtime="llamacpp",
            estimate_confidence="estimated",
        ),
        # 8B weights stored at 16-bit
        facts=ModelFacts(stored_bytes=Fact(16 * 1024**3, "test", FactKind.EXACT)),
    )


def test_generate_plan_has_rationale():
    plan = generate_plan(
        "meta-llama/Llama-3.1-8B-Instruct",
        make_hardware(),
        make_model(),
        OptimizationGoal.SPEED,
        Path("/tmp/out"),
    )
    assert plan.quant_config.bits == 4
    assert plan.llmfit_hypothesis_tps == 50.0
    assert any("tok/s" in line for line in plan.rationale)


def test_bnb_plan_explains_double_quant_saving():
    def plan_for(double_quant):
        choices = QuantChoices(algorithm=QuantAlgorithm.BNB, bits=4, double_quant=double_quant)
        return generate_plan(
            "meta-llama/Llama-3.1-8B-Instruct",
            make_hardware(),
            make_model(),
            OptimizationGoal.BALANCED,
            Path("/tmp/out"),
            choices,
        )

    double, single = plan_for(True), plan_for(False)
    assert double.predicted.weights_gb < single.predicted.weights_gb
    # 8B parameters, 0.373 bits each
    assert any("saving 356 MB" in line for line in double.rationale)
    assert any("--double-quant would save 356 MB" in line for line in single.rationale)


def test_nvfp4_plan_warns_when_this_pc_cannot_run_it_fast():
    from dataclasses import replace

    from ocq.types import GpuProfile

    def warnings_on(hardware):
        choices = QuantChoices(algorithm=QuantAlgorithm.NVFP4)
        plan = generate_plan(
            "meta-llama/Llama-3.1-8B-Instruct",
            hardware,
            make_model(),
            OptimizationGoal.BALANCED,
            Path("/tmp/out"),
            choices,
        )
        return " ".join(plan.warnings)

    def with_gpu(name):
        return replace(make_hardware(), gpus=(GpuProfile(name=name, vram_gb=24, backend="CUDA"),))

    assert "No NVIDIA GPU" in warnings_on(replace(make_hardware(), has_gpu=False, backend="CPU"))
    assert "Marlin" in warnings_on(with_gpu("NVIDIA GeForce RTX 4090"))
    assert "FP4 tensor cores" not in warnings_on(with_gpu("NVIDIA GeForce RTX 5090"))


def test_plan_says_the_context_memory_is_unknown(tmp_path, qwen_tensors):
    from test_model_analyzer import ODD_CONFIG, fused_attention

    from ocq.model_analyzer import build_snapshot

    hardware = make_hardware()
    tensors = fused_attention(qwen_tensors)
    model = build_snapshot("org/odd", {}, ODD_CONFIG, tensors, hardware)
    plan = generate_plan(model.model_id, hardware, model, OptimizationGoal.BALANCED, tmp_path)
    assert any("unknown amount for a 4096 token context" in line for line in plan.rationale)
    assert any("memory for the context is unknown" in warning for warning in plan.warnings)
    assert not any("tensor shapes could not be read" in warning for warning in plan.warnings)


def test_plan_says_how_the_gpus_are_used():
    from conftest import QWEN_72B, RTX_3090, RTX_3090_BUSY, nvidia_pc, real_model

    from ocq.model_analyzer import build_snapshot

    two = nvidia_pc(RTX_3090, RTX_3090_BUSY)
    meta, config, tensors, dtypes = real_model(QWEN_72B)
    model = build_snapshot(QWEN_72B, meta, config, tensors, two, dtypes=dtypes)
    awq = QuantChoices(algorithm=QuantAlgorithm.AWQ)
    plan = generate_plan(model.model_id, two, model, OptimizationGoal.BALANCED, Path("/tmp"), awq)
    assert any("--tensor-parallel-size 2" in line for line in plan.rationale)
    gguf = QuantChoices(algorithm=QuantAlgorithm.GGUF)
    plan = generate_plan(model.model_id, two, model, OptimizationGoal.BALANCED, Path("/tmp"), gguf)
    assert any("layers split over 2 GPUs" in line for line in plan.rationale)


def test_plan_gives_the_cache_flags_of_the_runtime():
    model, hardware = make_model(), make_hardware()

    def cache_line(**choices):
        plan = generate_plan(
            model.model_id,
            hardware,
            model,
            OptimizationGoal.BALANCED,
            Path("/tmp"),
            QuantChoices(**choices),
        )
        return next(line for line in plan.rationale if line.startswith("Context cache"))

    assert "-ctk q8_0 -ctv q8_0" in cache_line(algorithm=QuantAlgorithm.GGUF, kv_cache="q8_0")
    # This test PC's GPU has no known compute capability, so vLLM's fp8 cache is not promised.
    awq = cache_line(algorithm=QuantAlgorithm.AWQ, kv_cache="q4_0")
    assert "stays at 16 bits" in awq
    assert "8.9 or newer" in awq


def test_windows_explains_gguf_and_warns_about_vllm_formats():
    from dataclasses import replace

    windows = replace(make_hardware(), os_name="Windows")
    model = make_model()

    def plan_on(hardware, **choices):
        return generate_plan(
            model.model_id,
            hardware,
            model,
            OptimizationGoal.BALANCED,
            Path("/tmp"),
            QuantChoices(**choices),
        )

    picked = plan_on(windows)
    assert picked.quant_config.algorithm == QuantAlgorithm.GGUF
    assert any(line.startswith("Windows: GGUF") for line in picked.rationale)
    forced = plan_on(windows, algorithm=QuantAlgorithm.AWQ)
    assert any("does not run natively on Windows" in warning for warning in forced.warnings)
    linux = plan_on(make_hardware(), algorithm=QuantAlgorithm.AWQ)
    assert not any("Windows" in warning for warning in linux.warnings)


def test_plan_promises_vllm_fp8_cache_only_on_a_capable_gpu():
    from conftest import RTX_4090, nvidia_pc

    model, ada = make_model(), nvidia_pc(RTX_4090)
    choices = QuantChoices(algorithm=QuantAlgorithm.AWQ, kv_cache="q4_0")
    plan = generate_plan(
        model.model_id, ada, model, OptimizationGoal.BALANCED, Path("/tmp"), choices
    )
    line = next(line for line in plan.rationale if line.startswith("Context cache"))
    assert "--kv-cache-dtype fp8" in line
    assert "no 4-bit cache" in line
