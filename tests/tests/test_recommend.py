"""The recommendation engine (ocq.recommend). Offline: the Ollama store and HF cache are built in
a temporary home (test_discovery's fixtures), and Hub repos are read from the saved headers in
data/."""

from dataclasses import replace

import pytest
from conftest import MLX_QWEN, QWEN, QWEN_AWQ, QWEN_BNB, real_model
from test_discovery import empty_home, hf_repo, smollm2_store  # noqa: F401 (empty_home: autouse)
from test_model_analyzer import make_hardware

from ocq import model_analyzer
from ocq.quantizers import llama_cpp
from ocq.recommend import (
    ARCHITECTURE,
    CATALOG,
    CODER,
    CONTEXT,
    DEFAULT_WORKLOAD,
    DISK,
    MEMORY,
    QUANTIZATION,
    RUNTIME,
    WORKLOADS,
    Objective,
    RecommendationRequest,
    RecommendationResult,
    apply_limits,
    disk_needed_gb,
    generate_candidates,
    model_sources,
)
from ocq.types import LLAMA_CPP, TRANSFORMERS, VLLM, OcqError


def test_workloads_rank_on_distinct_objectives():
    assert len(WORKLOADS) == 7
    assert DEFAULT_WORKLOAD in WORKLOADS
    for key, profile in WORKLOADS.items():
        assert profile.key == key
        assert profile.priorities, key
        assert len(set(profile.priorities)) == len(profile.priorities), key
        assert all(isinstance(objective, Objective) for objective in profile.priorities)
        assert profile.context >= 4096 and profile.conversations >= 1


def test_workloads_put_first_what_the_user_cares_about():
    assert WORKLOADS["rag"].priorities[0] == Objective.CONTEXT
    assert WORKLOADS["coding"].priorities[:2] == (Objective.QUALITY, Objective.CONTEXT)
    assert WORKLOADS["throughput"].priorities[0] == Objective.THROUGHPUT
    assert WORKLOADS["throughput"].conversations > 1


def test_request_uses_the_workload_context_unless_given():
    assert RecommendationRequest().context_tokens == WORKLOADS[DEFAULT_WORKLOAD].context
    coding = RecommendationRequest(workload=WORKLOADS["coding"])
    assert coding.context_tokens == 32768
    assert RecommendationRequest(workload=WORKLOADS["coding"], context=4096).context_tokens == 4096


def test_empty_result_has_no_best():
    result = RecommendationResult(RecommendationRequest(), ranked=(), rejected=())
    assert result.best is None
    assert result.alternatives == ()


# Candidates (6B) ----------------------------------------------------------------------------


@pytest.fixture
def ollama_pc(tmp_path, monkeypatch):
    root = tmp_path / "ollama"
    smollm2_store(root)
    monkeypatch.setenv("OLLAMA_MODELS", str(root))


@pytest.fixture
def saved_hub(monkeypatch):
    """The Hub, answered from data/: saved repos only, anything else is not found."""

    def model(repo):
        try:
            return real_model(repo)
        except FileNotFoundError:
            raise OcqError(f"Model {repo!r} was not found on HuggingFace.") from None

    monkeypatch.setattr(model_analyzer, "fetch_hf_model", lambda repo: model(repo)[0])
    monkeypatch.setattr(model_analyzer, "fetch_model_files", lambda repo, sha: model(repo)[1:])

    def no_llmfit(*args):
        raise AssertionError("the engine must not ask llmfit")

    monkeypatch.setattr(model_analyzer, "catalog_entry", no_llmfit)


def test_models_on_this_pc_come_first_then_the_catalog(ollama_pc):
    sources = model_sources(RecommendationRequest())
    assert (sources[0].ref, sources[0].label, sources[0].local) == (
        "ollama:smollm2:135m",
        "smollm2:135m",
        True,
    )
    assert [s.ref for s in sources[1:]] == [e.repo for e in CATALOG]
    assert not any(s.local for s in sources[1:])


def test_a_catalog_repo_already_downloaded_is_read_from_disk_once(tmp_path, monkeypatch):
    coder = next(e for e in CATALOG if CODER in e.tags)
    config = b'{"model_type": "qwen2"}'
    hf_repo(tmp_path / "hub", coder.repo, {"config.json": config, "model.safetensors": b"w"})
    monkeypatch.setenv("HF_HOME", str(tmp_path))

    sources = [s for s in model_sources(RecommendationRequest()) if s.label == coder.repo]
    assert len(sources) == 1
    assert sources[0].local and sources[0].tags == coder.tags
    assert sources[0].ref.endswith("snapshots/abc123")  # the folder on disk, not the Hub


def test_named_models_are_the_only_sources(ollama_pc):
    request = RecommendationRequest(models=("ollama:smollm2:135m", QWEN))
    sources = model_sources(request)
    assert [(s.ref, s.local) for s in sources] == [("ollama:smollm2:135m", True), (QWEN, False)]


def test_every_format_runtime_and_cache_type_becomes_a_candidate(ollama_pc, saved_hub):
    request = RecommendationRequest(
        workload=WORKLOADS["coding"],
        models=("ollama:smollm2:135m", QWEN, QWEN_AWQ, "nobody/missing"),
    )
    candidates, unread = generate_candidates(request, make_hardware(ram_gb=24))

    assert [label for label, _ in unread] == ["nobody/missing"]
    assert "not found" in unread[0][1]
    assert {c.context for c in candidates} == {32768}  # the workload's context

    ollama = [c for c in candidates if c.model_ref == "ollama:smollm2:135m"]
    assert {(c.option.label, c.kv_cache) for c in ollama} == {("Q8_0", "f16"), ("Q8_0", "q8_0")}
    assert all(c.local and not c.quantize and c.runtime == LLAMA_CPP for c in ollama)

    qwen = {(c.option.label, c.kv_cache): c for c in candidates if c.model_ref == QWEN}
    stored = qwen[("BF16", "f16")]
    assert (stored.runtime, stored.quantize, stored.local) == (TRANSFORMERS, False, False)
    assert ("BF16", "q8_0") not in qwen  # transformers keeps a 16-bit cache: same row
    assert qwen[("Q4_K_M", "f16")].quantize and qwen[("Q4_K_M", "q8_0")].quantize
    q4_f16, q4_q8 = qwen[("Q4_K_M", "f16")].option, qwen[("Q4_K_M", "q8_0")].option
    assert q4_q8.context_gb < q4_f16.context_gb
    assert qwen[("AWQ 4-bit", "f16")].runtime == VLLM
    assert ("AWQ 4-bit", "q8_0") not in qwen  # no fp8 cache without an NVIDIA GPU

    (awq,) = [c for c in candidates if c.model_ref == QWEN_AWQ]  # stored as it is, not again
    assert (awq.runtime, awq.quantize, awq.kv_cache) == (VLLM, False, "f16")


# Hard limits (6C) ---------------------------------------------------------------------------

LINUX_GPU = replace(make_hardware(ram_gb=24, vram_gb=12), os_name="Linux")


def limits(models, hardware, context=0, disk_free_gb=1000.0, gguf_architectures=None):
    """{(model, label, cache): failed check names} and {(model, label, cache): failed texts}."""
    request = RecommendationRequest(models=tuple(models), context=context)
    candidates, unread = generate_candidates(request, hardware)
    assert not unread
    allowed, rejected = apply_limits(candidates, hardware, disk_free_gb, gguf_architectures)
    key = lambda c: (c.model_ref, c.option.label, c.kv_cache)  # noqa: E731
    failed = {key(c): () for c, _ in allowed}
    failed |= {key(r.candidate): tuple(check.name for check in r.failed) for r in rejected}
    texts = {key(r.candidate): " ".join(check.text for check in r.failed) for r in rejected}
    return failed, texts


def test_runtimes_that_cannot_run_here_are_rejected_before_memory_is_looked_at(saved_hub):
    failed, texts = limits([QWEN, QWEN_BNB], make_hardware(ram_gb=24))  # no NVIDIA GPU
    assert failed[(QWEN, "AWQ 4-bit", "f16")] == (RUNTIME,)
    assert "vLLM needs an NVIDIA GPU" in texts[(QWEN, "AWQ 4-bit", "f16")]
    assert failed[(QWEN, "BNB 4-bit double", "f16")] == (RUNTIME,)
    assert "bitsandbytes needs an NVIDIA GPU" in texts[(QWEN, "BNB 4-bit double", "f16")]
    (stored_bnb,) = [k for k in failed if k[0] == QWEN_BNB]  # a repo stored in bitsandbytes
    assert failed[stored_bnb] == (RUNTIME,)
    assert failed[(QWEN, "Q4_K_M", "f16")] == ()  # llama.cpp and transformers run anywhere
    assert failed[(QWEN, "BF16", "f16")] == ()


def test_vllm_and_bitsandbytes_run_on_an_nvidia_gpu_under_linux_not_windows(saved_hub):
    failed, _ = limits([QWEN], LINUX_GPU)
    assert failed[(QWEN, "AWQ 4-bit", "f16")] == ()
    assert failed[(QWEN, "BNB 4-bit double", "f16")] == ()

    failed, texts = limits([QWEN], replace(LINUX_GPU, os_name="Windows"))
    assert failed[(QWEN, "AWQ 4-bit", "f16")] == (RUNTIME,)
    assert "only in WSL2 or on Linux" in texts[(QWEN, "AWQ 4-bit", "f16")]
    assert failed[(QWEN, "BNB 4-bit double", "f16")] == ()  # transformers runs on Windows


def test_mlx_models_run_only_on_apple_silicon(saved_hub):
    failed, texts = limits([MLX_QWEN], make_hardware(ram_gb=24))
    ((key, names),) = failed.items()
    assert names == (RUNTIME,) and "Apple Silicon" in texts[key]


def test_a_model_too_big_for_the_free_memory_is_rejected_with_the_numbers(saved_hub):
    failed, texts = limits([QWEN], make_hardware(ram_gb=2.5))
    assert failed[(QWEN, "BF16", "f16")] == (MEMORY,)
    assert texts[(QWEN, "BF16", "f16")].startswith("Needs ")
    assert "2.5 GB is free" in texts[(QWEN, "BF16", "f16")]
    assert failed[(QWEN, "Q4_K_M", "q8_0")] == ()


def test_more_context_than_the_model_holds_is_rejected(saved_hub):
    longest = real_model(QWEN)[1]["max_position_embeddings"]
    failed, texts = limits([QWEN], LINUX_GPU, context=longest * 2)
    assert CONTEXT in failed[(QWEN, "Q4_K_M", "q8_0")]
    assert f"up to {longest:,} tokens" in texts[(QWEN, "Q4_K_M", "q8_0")]
    failed, _ = limits([QWEN], LINUX_GPU, context=longest)
    assert CONTEXT not in failed[(QWEN, "Q4_K_M", "q8_0")]


SMOLLM2_CONTEXT = 2048  # what the small GGUF file of the Ollama fixture holds


def test_a_gguf_is_only_made_for_architectures_llama_cpp_converts(saved_hub, ollama_pc):
    models = [QWEN, "ollama:smollm2:135m"]
    failed, texts = limits(
        models, LINUX_GPU, SMOLLM2_CONTEXT, gguf_architectures=frozenset({"LlamaForCausalLM"})
    )
    assert failed[(QWEN, "Q4_K_M", "f16")] == (ARCHITECTURE,)
    assert "cannot convert Qwen2ForCausalLM" in texts[(QWEN, "Q4_K_M", "f16")]
    assert failed[(QWEN, "BF16", "f16")] == ()  # transformers runs it without converting
    assert failed[("ollama:smollm2:135m", "Q8_0", "f16")] == ()  # an existing GGUF file

    failed, _ = limits(
        models, LINUX_GPU, SMOLLM2_CONTEXT, gguf_architectures=frozenset({"Qwen2ForCausalLM"})
    )
    assert failed[(QWEN, "Q4_K_M", "f16")] == ()
    failed, _ = limits(
        models, LINUX_GPU, SMOLLM2_CONTEXT, gguf_architectures=None
    )  # converter not downloaded
    assert failed[(QWEN, "Q4_K_M", "f16")] == ()


def test_awq_is_made_on_an_nvidia_gpu_with_the_original_model_in_ram(saved_hub):
    failed, _ = limits([QWEN], LINUX_GPU)
    assert failed[(QWEN, "AWQ 4-bit", "f16")] == ()
    original_gb = real_model(QWEN)[0]["safetensors"]["total"] * 2 / 1024**3
    small_ram = replace(LINUX_GPU, available_ram_gb=original_gb / 2)
    failed, texts = limits([QWEN], small_ram)
    assert QUANTIZATION in failed[(QWEN, "GPTQ 4-bit", "f16")]
    assert "whole original model" in texts[(QWEN, "GPTQ 4-bit", "f16")]
    assert QUANTIZATION not in failed[(QWEN, "Q4_K_M", "f16")]  # GGUF converts without loading


def test_disk_for_downloading_and_quantizing(saved_hub, ollama_pc):
    models = (QWEN, "ollama:smollm2:135m")
    request = RecommendationRequest(models=models, context=SMOLLM2_CONTEXT)
    candidates, _ = generate_candidates(request, LINUX_GPU)
    by_key = {(c.model_ref, c.option.label, c.kv_cache): c for c in candidates}
    stored, q4 = by_key[(QWEN, "BF16", "f16")], by_key[(QWEN, "Q4_K_M", "f16")]
    original = stored.model.original_gb
    assert disk_needed_gb(stored) == original  # download only
    # download, the full-precision GGUF made first, then the Q4_K_M file
    assert disk_needed_gb(q4) == original + q4.option.weights_gb + original
    assert disk_needed_gb(by_key[("ollama:smollm2:135m", "Q8_0", "f16")]) == 0

    failed, texts = limits(models, LINUX_GPU, SMOLLM2_CONTEXT, disk_free_gb=original / 2)
    assert failed[(QWEN, "BF16", "f16")] == (DISK,)
    assert "to download it" in texts[(QWEN, "BF16", "f16")]
    assert "download and quantize it" in texts[(QWEN, "Q4_K_M", "f16")]
    assert failed[("ollama:smollm2:135m", "Q8_0", "f16")] == ()  # already on this PC


def test_converter_architectures_are_read_from_its_register_lines(tmp_path, monkeypatch):
    source = tmp_path / "llama.cpp"
    (source / "conversion").mkdir(parents=True)
    (source / "convert_hf_to_gguf.py").write_text('@ModelBase.register("LlamaForCausalLM")\n')
    (source / "conversion" / "qwen.py").write_text(
        '@ModelBase.register(\n    "Qwen2ForCausalLM",\n    "Qwen2Model",\n)\nclass Qwen2: ...\n'
    )
    monkeypatch.setenv("OCQ_LLAMA_CPP", str(source))
    assert llama_cpp.converter_architectures() == {
        "LlamaForCausalLM",
        "Qwen2ForCausalLM",
        "Qwen2Model",
    }

    monkeypatch.delenv("OCQ_LLAMA_CPP")
    monkeypatch.setattr(llama_cpp, "CACHE", tmp_path / "not-downloaded")
    assert llama_cpp.converter_architectures() is None  # and nothing is downloaded
