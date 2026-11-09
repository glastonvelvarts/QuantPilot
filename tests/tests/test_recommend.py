"""The recommendation engine (ocq.recommend)."""

from ocq.recommend import (
    DEFAULT_WORKLOAD,
    WORKLOADS,
    Objective,
    RecommendationRequest,
    RecommendationResult,
)


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
