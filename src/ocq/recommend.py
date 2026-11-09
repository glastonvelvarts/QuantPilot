"""Recommendation engine: which model, quantization and runtime to use on this PC, and why.

The CLI (`ocq recommend`) and the TUI (Check page) both call this module, so they always agree.
It does not size anything itself: memory, layer split and speed come from fit_at, the same code
`ocq feasibility` uses. This module adds what to do with those rows:

1. candidates: every model x quantization x runtime x cache type worth trying
2. hard limits: a candidate that cannot run here is rejected, with the reason
3. ranking: the rest are ordered by what the workload cares about most
4. explanation: why the winner won, why the others lost, and how sure we are
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum

from ocq.types import QuantOption


class Objective(str, Enum):
    """What a workload can care about. Each one is measured on its own, never added up."""

    QUALITY = "quality"  # how good the answers are (estimated until phase 7 measures it)
    SPEED = "speed"  # tokens per second for one conversation
    CONTEXT = "context"  # how much text fits in one conversation
    MEMORY = "memory"  # how much room is left for other programs
    THROUGHPUT = "throughput"  # conversations served at the same time

    def __str__(self) -> str:
        return self.value


@dataclass(frozen=True)
class WorkloadProfile:
    """What the user wants to do, in their words, and what that means for the ranking."""

    key: str
    label: str
    # Most important first. Candidates are compared on the first objective, and the next one
    # only breaks near-ties (see the ranking step).
    priorities: tuple[Objective, ...]
    # Tokens of context the workload needs when the user does not say.
    context: int
    # Conversations served at the same time.
    conversations: int = 1


WORKLOADS: dict[str, WorkloadProfile] = {
    profile.key: profile
    for profile in (
        WorkloadProfile(
            "chat",
            "General Chat",
            (Objective.QUALITY, Objective.SPEED, Objective.MEMORY),
            context=8192,
        ),
        WorkloadProfile(
            "coding",
            "Coding",
            (Objective.QUALITY, Objective.CONTEXT, Objective.SPEED),
            context=32768,
        ),
        WorkloadProfile(
            "rag",
            "RAG (documents)",
            (Objective.CONTEXT, Objective.QUALITY, Objective.SPEED),
            context=32768,
        ),
        WorkloadProfile(
            "agentic",
            "Agentic",
            (Objective.QUALITY, Objective.SPEED, Objective.CONTEXT),
            context=32768,
        ),
        WorkloadProfile(
            "reasoning",
            "Reasoning",
            (Objective.QUALITY, Objective.SPEED, Objective.CONTEXT),
            context=32768,
        ),
        WorkloadProfile(
            "long-context",
            "Long Context",
            (Objective.CONTEXT, Objective.QUALITY, Objective.MEMORY),
            context=131072,
        ),
        WorkloadProfile(
            "throughput",
            "High Throughput",
            (Objective.THROUGHPUT, Objective.MEMORY, Objective.QUALITY),
            context=8192,
            conversations=8,
        ),
    )
}
DEFAULT_WORKLOAD = "chat"


@dataclass(frozen=True)
class RecommendationRequest:
    """What the user asked for. Empty fields mean: work it out."""

    workload: WorkloadProfile = WORKLOADS[DEFAULT_WORKLOAD]
    # Models to consider (HF repo ids, local paths, ollama:name:tag). Empty: the models on this
    # PC plus the curated list of popular repos.
    models: tuple[str, ...] = ()
    # Tokens of context. 0: the workload's own.
    context: int = 0
    # Cache types to try. Empty: f16 and q8_0.
    kv_caches: tuple[str, ...] = ()

    @property
    def context_tokens(self) -> int:
        return self.context or self.workload.context


@dataclass(frozen=True)
class Candidate:
    """One way to run one model: a row of fit_at at one context and cache type."""

    model_ref: str  # what the user (or discovery) named it
    model_id: str  # the model it resolved to
    option: QuantOption  # memory, run mode, layer split and speed, from fit_at
    context: int
    kv_cache: str
    local: bool  # already on this PC; False means it has to be downloaded first

    @property
    def runtime(self) -> str:
        return self.option.runtime

    @property
    def name(self) -> str:
        return f"{self.model_id} {self.option.label} · {self.runtime}"


@dataclass(frozen=True)
class Check:
    """The outcome of one hard limit for one candidate."""

    name: str  # memory, runtime, architecture, context, quantization
    passed: bool
    text: str  # plain words, shown when the check fails


@dataclass(frozen=True)
class Rejection:
    """A candidate that cannot run here, and every limit it broke."""

    candidate: Candidate
    failed: tuple[Check, ...]


class ReasonKind(str, Enum):
    GOOD = "good"  # ✓ a point in its favour
    TRADEOFF = "tradeoff"  # △ something given up for it
    BAD = "bad"  # ✗ why it lost or cannot run


@dataclass(frozen=True)
class Reason:
    kind: ReasonKind
    text: str


class ConfidenceLevel(str, Enum):
    LOW = "Low"
    MEDIUM = "Medium"
    HIGH = "High"
    VERY_HIGH = "Very High"

    def __str__(self) -> str:
        return self.value


@dataclass(frozen=True)
class Evidence:
    """One thing the recommendation rests on, and whether it was measured or guessed."""

    text: str  # "hardware measured", "no runs of this model on this PC yet"
    known: bool


@dataclass(frozen=True)
class Confidence:
    level: ConfidenceLevel
    evidence: tuple[Evidence, ...] = ()


@dataclass(frozen=True)
class Recommendation:
    """A candidate that can run here, with what it scores and why."""

    candidate: Candidate
    # Each objective on its own scale (0 worst, 1 best among the candidates). Used to order
    # them, never shown as a number.
    scores: dict[Objective, float] = field(default_factory=dict)
    reasons: tuple[Reason, ...] = ()
    warnings: tuple[str, ...] = ()
    confidence: Confidence = Confidence(ConfidenceLevel.LOW)


@dataclass(frozen=True)
class RecommendationResult:
    request: RecommendationRequest
    ranked: tuple[Recommendation, ...]  # best first
    rejected: tuple[Rejection, ...]

    @property
    def best(self) -> Recommendation | None:
        return self.ranked[0] if self.ranked else None

    @property
    def alternatives(self) -> tuple[Recommendation, ...]:
        return self.ranked[1:]
