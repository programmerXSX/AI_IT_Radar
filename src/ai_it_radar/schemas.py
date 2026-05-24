"""Project-wide Pydantic models. All inter-agent state flows through these."""
from __future__ import annotations

from datetime import datetime
from enum import Enum
from typing import Any, Literal

from pydantic import BaseModel, Field, HttpUrl


class SourceKind(str, Enum):
    ARXIV = "arxiv"
    GITHUB = "github"
    HUGGINGFACE = "huggingface"
    OTHER = "other"


class CandidateKind(str, Enum):
    PAPER = "paper"
    REPO = "repo"
    MODEL = "model"
    DATASET = "dataset"
    BLOG = "blog"
    OTHER = "other"


class Candidate(BaseModel):
    """Normalized output from a Source adapter — ScoutAgent's product."""

    uid: str = Field(..., description="Stable, source-prefixed unique id, e.g. arxiv:2501.12345")
    source: SourceKind
    kind: CandidateKind
    title: str
    url: str
    summary: str = ""
    content: str = Field("", description="Full README / abstract / extracted text")
    authors: list[str] = Field(default_factory=list)
    published_at: datetime | None = None
    fetched_at: datetime = Field(default_factory=datetime.utcnow)
    metadata: dict[str, Any] = Field(default_factory=dict)


class TriageDecision(str, Enum):
    DUPLICATE = "duplicate"
    OUT_OF_SCOPE = "out_of_scope"
    KEEP = "keep"
    EXPLORE = "explore"  # bypassed profile filter via exploration budget


class TriageResult(BaseModel):
    candidate_uid: str
    decision: TriageDecision
    profile_match_score: float = Field(0.0, ge=0.0, le=1.0)
    duplicate_of: str | None = None
    rationale: str = ""


class Analysis(BaseModel):
    """AnalystAgent's structured extraction output."""

    candidate_uid: str
    key_capabilities: list[str] = Field(default_factory=list)
    method_summary: str = ""
    dependency_stack: list[str] = Field(default_factory=list)
    data_requirements: str = ""
    license: str | None = None
    known_limitations: list[str] = Field(default_factory=list)
    related_uids: list[str] = Field(
        default_factory=list, description="Analogous KB items found via RAG"
    )


class DimensionScore(BaseModel):
    dimension_id: str
    score: int = Field(..., ge=0, le=5)
    confidence: float = Field(0.0, ge=0.0, le=1.0)
    rationale: str = ""
    quote: str = ""
    extras: dict[str, Any] = Field(default_factory=dict)


class CriticVerdict(BaseModel):
    dimension_id: str
    agree: bool
    suggested_score: int | None = None
    disagreement_reason: str = ""


class Score(BaseModel):
    """Aggregated EvaluatorAgent output for one Candidate."""

    candidate_uid: str
    dimensions: list[DimensionScore] = Field(default_factory=list)
    critic_verdicts: list[CriticVerdict] = Field(default_factory=list)
    aggregate: float = 0.0
    band: Literal["strong_recommend", "watch", "monitor"] = "monitor"
    needs_human: bool = False
    evaluated_at: datetime = Field(default_factory=datetime.utcnow)


class FeedbackTag(str, Enum):
    ADOPT = "adopt"
    IGNORE = "ignore"
    WATCH = "watch"


class Feedback(BaseModel):
    candidate_uid: str
    tag: FeedbackTag
    note: str = ""
    user: str = "anonymous"
    created_at: datetime = Field(default_factory=datetime.utcnow)


class ReportItem(BaseModel):
    candidate: Candidate
    analysis: Analysis | None = None
    score: Score | None = None


class Report(BaseModel):
    period_start: datetime
    period_end: datetime
    strong_recommend: list[ReportItem] = Field(default_factory=list)
    watch: list[ReportItem] = Field(default_factory=list)
    monitor: list[ReportItem] = Field(default_factory=list)
    generated_at: datetime = Field(default_factory=datetime.utcnow)


# ---- LangGraph node state (one cycle of the radar) ----------------------------------------

class GraphState(BaseModel):
    """The mutable state that flows through Scout -> Triage -> Analyst -> Evaluator -> Reporter."""

    cycle_id: str
    candidates: list[Candidate] = Field(default_factory=list)
    triage: list[TriageResult] = Field(default_factory=list)
    analyses: list[Analysis] = Field(default_factory=list)
    scores: list[Score] = Field(default_factory=list)
    report: Report | None = None
    errors: list[str] = Field(default_factory=list)

    model_config = {"arbitrary_types_allowed": True}
