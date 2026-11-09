"""The recommendation engine (ocq.recommend). Offline: the Ollama store and HF cache are built in
a temporary home (test_discovery's fixtures), and Hub repos are read from the saved headers in
data/."""

import pytest
from conftest import QWEN, QWEN_AWQ, real_model
from test_discovery import empty_home, hf_repo, smollm2_store  # noqa: F401 (empty_home: autouse)
from test_model_analyzer import make_hardware

from ocq import model_analyzer
from ocq.recommend import (
    CATALOG,
    CODER,
    DEFAULT_WORKLOAD,
    WORKLOADS,
    Objective,
    RecommendationRequest,
    RecommendationResult,
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
