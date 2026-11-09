"""The recommendation engine (ocq.recommend). Offline: the Ollama store and HF cache are built in
a temporary home (test_discovery's fixtures), and Hub repos are read from the saved headers in
data/."""

import math
from dataclasses import replace

import pytest
from conftest import GPT_OSS, MLX_QWEN, QWEN, QWEN3_MOE, QWEN_AWQ, QWEN_BNB, real_model
from test_discovery import empty_home, hf_repo, smollm2_store  # noqa: F401 (empty_home: autouse)
from test_model_analyzer import make_hardware

from ocq import measurements, model_analyzer
from ocq.quantizers import llama_cpp
from ocq.recommend import (
    ARCHITECTURE,
    AS_RELEASED,
    CALIBRATED_4BIT,
    CATALOG,
    CODER,
    CONTEXT,
    DEFAULT_WORKLOAD,
    DISK,
    FULL,
    MEMORY,
    PLAIN_4BIT,
    QUANTIZATION,
    RUNTIME,
    SIZE_SLOPE,
    SLOW,
    SPECIALIST,
    TIGHT,
    WORKLOADS,
    Check,
    ConfidenceLevel,
    Objective,
    ReasonKind,
    RecommendationRequest,
    RecommendationResult,
    apply_limits,
    confidence,
    disk_needed_gb,
    drawbacks,
    effective_parameters,
    estimated_quality,
    explain_model,
    generate_candidates,
    model_sources,
    objective_values,
    quality_kept,
    rank,
    rated_as,
    recommend,
    refit,
)
from ocq.types import LLAMA_CPP, TRANSFORMERS, VLLM, Fact, FactKind, FitLevel, OcqError


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


# Ranking (6D) -------------------------------------------------------------------------------

BIG_GPU = replace(make_hardware(ram_gb=64, vram_gb=24), os_name="Linux")


def by_label(models, hardware, context=0, workload="chat"):
    request = RecommendationRequest(
        workload=WORKLOADS[workload], models=tuple(models), context=context
    )
    candidates, _ = generate_candidates(request, hardware)
    return {(c.model_ref, c.option.label, c.kv_cache): c for c in candidates}


def test_quality_kept_follows_llama_cpp_measurements():
    ladder = ["Q2_K", "Q3_K_S", "Q3_K_M", "Q4_0", "Q4_K_S", "Q4_K_M", "Q5_K_M", "Q6_K", "Q8_0"]
    kept = [quality_kept(t) for t in ladder]
    assert kept == sorted(kept) and kept[-1] < quality_kept(FULL) == 1.0
    assert round(quality_kept("Q4_K_M"), 3) == 0.854
    assert quality_kept("Q2_K") < 0.1 and quality_kept("Q8_0") > 0.99
    # The exchange rate comes from llama.cpp's LLaMA table, and agrees with its other rows:
    # 13B at Q4_0 (5.3860) is still worth more than 7B at F16 (5.9066).
    assert round(SIZE_SLOPE, 4) == 0.1765
    assert 13.0 * math.exp(-math.log(5.3860 / 5.2543) / SIZE_SLOPE) > 6.7


def test_each_format_is_rated_as_the_gguf_type_closest_to_it(saved_hub):
    rows = by_label([QWEN, QWEN_AWQ, MLX_QWEN, GPT_OSS], BIG_GPU)
    rating = {(ref, label): rated_as(c) for (ref, label, kv), c in rows.items() if kv == "f16"}
    assert rating[(QWEN, "BF16")] == FULL
    assert rating[(QWEN, "Q4_K_M")] == "Q4_K_M"
    assert rating[(QWEN, "AWQ 4-bit")] == rating[(QWEN, "GPTQ 4-bit")] == CALIBRATED_4BIT
    assert rating[(QWEN, "BNB 4-bit double")] == PLAIN_4BIT
    assert rating[(QWEN, "GPTQ 8-bit")] == rating[(QWEN, "BNB 8-bit")] == "Q8_0"
    assert rating[(QWEN_AWQ, "AWQ 4-bit (as stored)")] == CALIBRATED_4BIT
    (mlx,) = [r for (ref, _), r in rating.items() if ref == MLX_QWEN]
    assert mlx == PLAIN_4BIT
    (gpt_oss,) = [r for (ref, _), r in rating.items() if ref == GPT_OSS]
    assert gpt_oss == AS_RELEASED


def test_a_mixture_of_experts_counts_between_its_active_and_total_size(saved_hub):
    (moe, *_) = by_label([QWEN3_MOE], BIG_GPU).values()
    total, active = moe.model.facts.parameters.value, moe.model.facts.active_parameters.value
    assert active < effective_parameters(moe.model) < total
    assert effective_parameters(moe.model) == math.sqrt(total * active)


def test_near_equal_quality_goes_to_the_faster_one(saved_hub):
    # A big GPU runs every format; Q8_0 and Q6_K keep about the same quality, Q6_K is faster.
    best = rank(list(by_label([QWEN], BIG_GPU).values()), WORKLOADS["chat"], BIG_GPU)[0]
    assert (best.candidate.option.label, best.candidate.kv_cache) == ("Q6_K", "f16")
    assert best.scores[Objective.QUALITY] > 0.95 and best.scores[Objective.SPEED] < 1


def test_a_slow_or_tight_candidate_ranks_after_a_comfortable_one(saved_hub):
    rows = by_label([QWEN], BIG_GPU)
    good, smaller = rows[(QWEN, "Q8_0", "f16")], rows[(QWEN, "Q4_K_M", "f16")]
    slow = replace(good, option=replace(good.option, estimated_tps=3.0))
    tight = replace(good, option=replace(good.option, fit_level=FitLevel.MARGINAL))
    assert drawbacks(slow, WORKLOADS["chat"]) == [SLOW]
    assert drawbacks(tight, WORKLOADS["chat"]) == [TIGHT]
    for worse in (slow, tight):
        ranked = rank([worse, smaller], WORKLOADS["chat"], BIG_GPU)
        assert ranked[0].candidate is smaller  # lower quality, but nothing against it
    # Serving many conversations has no reading-speed floor.
    assert SLOW not in drawbacks(slow, WORKLOADS["throughput"])


def test_a_coding_model_is_kept_for_coding(saved_hub):
    rows = by_label([QWEN], BIG_GPU)
    general = rows[(QWEN, "Q4_K_M", "f16")]
    coder = replace(rows[(QWEN, "Q8_0", "f16")], tags=frozenset({CODER}))
    assert drawbacks(coder, WORKLOADS["chat"]) == [SPECIALIST]
    assert rank([coder, general], WORKLOADS["chat"], BIG_GPU)[0].candidate is general
    assert rank([coder, general], WORKLOADS["coding"], BIG_GPU)[0].candidate is coder


def test_ties_go_to_the_default_cache_and_to_files_already_here(saved_hub):
    rows = by_label([QWEN], BIG_GPU)
    f16, q8 = rows[(QWEN, "Q6_K", "f16")], rows[(QWEN, "Q6_K", "q8_0")]
    assert rank([q8, f16], WORKLOADS["chat"], BIG_GPU)[0].candidate is f16
    here = replace(f16, local=True)
    assert rank([f16, here], WORKLOADS["chat"], BIG_GPU)[0].candidate is here


def test_context_room_is_counted_up_to_four_times_the_context_asked(saved_hub):
    longest = real_model(QWEN)[1]["max_position_embeddings"]
    rows = by_label([QWEN], BIG_GPU, context=4096, workload="rag")
    values = objective_values(rows[(QWEN, "Q4_K_M", "f16")], WORKLOADS["rag"], BIG_GPU)
    assert values[Objective.CONTEXT] == min(4 * 4096, longest)


def test_throughput_counts_the_conversations_that_fit(saved_hub):
    workload = WORKLOADS["throughput"]
    rows = by_label([QWEN], BIG_GPU, workload="throughput")
    q4 = rows[(QWEN, "Q4_K_M", "f16")]
    values = objective_values(q4, workload, BIG_GPU)
    assert values[Objective.THROUGHPUT] == workload.conversations * q4.option.estimated_tps
    small = make_hardware(ram_gb=2.2)  # room for the weights and only some conversations
    q4_small = by_label([QWEN], small, workload="throughput")[(QWEN, "Q4_K_M", "f16")]
    fewer = objective_values(q4_small, workload, small)[Objective.THROUGHPUT]
    conversations = fewer / q4_small.option.estimated_tps
    assert conversations == round(conversations) and 1 < conversations < workload.conversations
    fit = refit(q4_small, small, int(conversations) * q4_small.context)
    assert fit.runnable  # exactly as many as fit: one more does not
    assert not refit(q4_small, small, int(conversations + 1) * q4_small.context).runnable


def test_recommend_runs_the_whole_engine(saved_hub):
    request = RecommendationRequest(models=(QWEN, QWEN_AWQ, "nobody/missing"))
    result = recommend(request, make_hardware(ram_gb=24), disk_free_gb=1000)
    assert result.best is not None and result.best.candidate.runtime == "llama.cpp"
    assert result.unread[0][0] == "nobody/missing"
    assert {r.failed[0].name for r in result.rejected} == {RUNTIME}  # vLLM, bitsandbytes
    for recommendation in result.ranked:
        assert all(0 <= score <= 1 for score in recommendation.scores.values())
    qualities = [estimated_quality(r.candidate) for r in result.ranked]
    assert qualities[0] >= max(qualities) * 0.95  # quality first, within its tolerance


# Explanations and confidence (6E) -----------------------------------------------------------


def texts(reasons, kind=None):
    return [r.text for r in reasons if kind is None or r.kind == kind]


def big_gpu_chat(**request):
    return recommend(RecommendationRequest(models=(QWEN,), **request), BIG_GPU, disk_free_gb=1000)


def test_the_pick_says_what_it_does_here_and_what_it_takes(saved_hub):
    best = big_gpu_chat().best
    good = texts(best.reasons, ReasonKind.GOOD)
    assert good[0].startswith("Fits entirely in GPU memory: ")
    assert good[1].startswith("Up to ") and "tok/s" in good[1]
    assert good[2] == "Holds the 8K tokens of context asked for"
    assert good[3] == "Q6_K keeps about 98% of the model's quality (llama.cpp's measurements)"
    steps = texts(best.reasons, ReasonKind.TRADEOFF)
    assert any(t.startswith("Not on this PC yet: download it first (about ") for t in steps)
    assert "QuantPilot makes the Q6_K file first" in steps
    # The closest alternatives: what they would give, and what they would cost.
    assert any(t.startswith("Q5_K_M: ") and ", but keeps less" in t for t in steps)


def test_alternatives_are_distinct_choices_and_say_why_they_lost(saved_hub):
    result = big_gpu_chat()
    keys = [(a.candidate.model_ref, a.candidate.option.label) for a in result.alternatives]
    assert len(keys) == len(set(keys))
    assert (QWEN, result.best.candidate.option.label) not in keys
    q8 = next(a for a in result.alternatives if a.candidate.option.label == "Q8_0")
    assert texts(q8.reasons, ReasonKind.BAD)[-1].endswith(
        "slower (" + texts(q8.reasons)[-1].split("(")[1]
    )
    # Q5_K_M is within 5% of Q6_K's quality but not of the best quality there is (BF16), so
    # it dropped out on quality: the explanation says so instead of calling it a tie.
    q5 = next(a for a in result.alternatives if a.candidate.option.label == "Q5_K_M")
    assert texts(q5.reasons, ReasonKind.BAD) == ["Keeps less of the model's quality (95% vs 98%)"]
    # The same file with the other cache ties on everything; the default cache wins the tie.
    same = next(
        r
        for r in result.ranked
        if r.candidate.option.label == "Q6_K" and r.candidate.kv_cache == "q8_0"
    )
    assert texts(same.reasons) == [
        "As good as the pick here; the pick uses the default (f16) cache"
    ]


def test_confidence_rises_with_runs_on_this_pc(saved_hub):
    best = big_gpu_chat().best
    assert best.confidence.level == ConfidenceLevel.MEDIUM
    evidence = {e.text: e.known for e in best.confidence.evidence}
    assert (
        evidence["Model size read from its files"]
        and evidence["Context memory read from its files"]
    )
    assert evidence["Memory speed measured on this PC"]
    assert evidence["Speed is an upper bound"] is False
    assert evidence["Quality estimated from size and format (not measured yet)"] is False

    # Runs of other sizes on this GPU: speed predictions come from them.
    measurements.record("gpu", 0.5, 300.0)
    measurements.record("gpu", 4.0, 60.0)
    assert big_gpu_chat().best.confidence.level == ConfidenceLevel.HIGH

    label = big_gpu_chat().best.candidate.option.label
    for tps in (101.0, 104.0):
        measurements.record("gpu", 1.2, tps, model=QWEN, label=label, runtime="llama.cpp")
    best = big_gpu_chat().best
    assert best.confidence.level == ConfidenceLevel.VERY_HIGH
    assert "2 runs of this setup on this PC" in [e.text for e in best.confidence.evidence]
    assert "Measured 101–104 tok/s in 2 runs on this PC" in texts(best.reasons)


def test_confidence_is_low_when_a_fact_is_estimated_or_a_check_was_not_made(saved_hub):
    best = big_gpu_chat().best
    facts = best.candidate.model.facts
    guessed = Fact(
        1, "config.json", FactKind.ESTIMATE, "2 of 28 layers keep a state of unknown size"
    )
    model = replace(best.candidate.model, facts=replace(facts, kv_cache=guessed))
    unsure = replace(best, candidate=replace(best.candidate, model=model))
    low = confidence(unsure, BIG_GPU)
    assert low.level == ConfidenceLevel.LOW
    assert "Context memory estimated: some layers keep a state of unknown size" in [
        e.text for e in low.evidence
    ]
    unchecked = replace(best, checks=(Check(ARCHITECTURE, True, "not checked: no converter"),))
    assert confidence(unchecked, BIG_GPU).level == ConfidenceLevel.LOW


def test_why_not_a_model(saved_hub):
    request = RecommendationRequest(models=(QWEN, QWEN_AWQ, "nobody/missing"))
    result = recommend(request, make_hardware(ram_gb=24), disk_free_gb=1000)
    assert texts(explain_model(result, QWEN))[0].startswith("It is the pick: ")
    awq = explain_model(result, QWEN_AWQ)  # stored for vLLM, and this PC has no NVIDIA GPU
    assert awq[0].text == "Its closest option: AWQ 4-bit (as stored) in vLLM, f16 cache"
    assert "vLLM needs an NVIDIA GPU" in awq[1].text
    assert texts(explain_model(result, "nobody/missing"))[0].startswith("Could not be read: ")
    assert texts(explain_model(result, "other/model")) == ["Not among the models considered"]


def test_why_not_names_the_limit_that_matters_not_a_runtime_that_cannot_run(saved_hub):
    request = RecommendationRequest(models=(QWEN,))
    result = recommend(request, make_hardware(ram_gb=0.5), disk_free_gb=1000)
    assert result.best is None
    reasons = explain_model(result, QWEN)
    assert reasons[0].text == "Its closest option: Q2_K in llama.cpp, q8_0 cache"
    assert reasons[1].text.startswith("Needs ") and "0.5 GB is free" in reasons[1].text
