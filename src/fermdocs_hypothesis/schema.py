"""HypothesisStage contracts — input, output, and all view shapes.

Plan ref: plans/2026-05-03-hypothesis-debate-v0.md §3, §4, §9, §10.

Design rules (mirror diagnose for consistency):
  - Closed-vocabulary enums over free strings.
  - Confidence cap at 0.85 matches diagnose convention for LLM output.
  - Citation discipline: hypothesis must cite ≥1 finding OR narrative.
  - Views are Pydantic models so the projector spec is type-checked.
  - All IDs are short stable shapes the ranker and tests can pattern-match.

ID shapes:
  - topic_id  T-NNNN
  - facet_id  FCT-NNNN
  - hyp_id    H-NNNN
  - qid       Q-NNNN  (hypothesis-stage open questions, distinct from D-Q-NNNN)
"""

from __future__ import annotations

import re
from datetime import datetime
from enum import Enum
from typing import Any, Literal
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from fermdocs_characterize.schema import Severity
from fermdocs_diagnose.schema import (
    ConfidenceBasis,
    LLM_CONFIDENCE_CAP,
    TrajectoryRef,
)

TOPIC_ID_RE = re.compile(r"^T-\d{4,}$")
FACET_ID_RE = re.compile(r"^FCT-\d{4,}$")
HYP_ID_RE = re.compile(r"^H-\d{4,}$")
QID_RE = re.compile(r"^Q-\d{4,}$")

SpecialistRole = Literal["kinetics", "mass_transfer", "metabolic"]
ExitReason = Literal[
    "budget_exhausted",
    "max_turns_reached",
    "consensus_reached",
    "no_topics_left",
]


# ---------- Topic + ranking ----------


class TopicSourceType(str, Enum):
    FAILURE = "failure"
    ANALYSIS = "analysis"
    TREND = "trend"
    OPEN_QUESTION = "open_question"


class SeedTopic(BaseModel):
    """A candidate debate topic derived from the upstream DiagnosisOutput."""

    model_config = ConfigDict(frozen=True)

    topic_id: str
    summary: str = Field(min_length=1)
    source_type: TopicSourceType
    source_id: str
    cited_finding_ids: list[str] = Field(default_factory=list)
    cited_narrative_ids: list[str] = Field(default_factory=list)
    cited_trajectories: list[TrajectoryRef] = Field(default_factory=list)
    affected_variables: list[str] = Field(default_factory=list)
    severity: Severity
    priority: float = Field(ge=0.0, le=1.0)

    @field_validator("topic_id")
    @classmethod
    def _topic_id_shape(cls, v: str) -> str:
        if not TOPIC_ID_RE.match(v):
            raise ValueError(f"topic_id must match T-NNNN, got {v!r}")
        return v


class RankedTopic(BaseModel):
    """Topic + ranker score, surfaced to the orchestrator."""

    model_config = ConfigDict(frozen=True)

    topic_id: str
    summary: str
    score: float
    is_synthetic: bool = Field(
        default=False,
        description="True when the topic was synthesized from an open question"
        " rather than a SeedTopic.",
    )


class TopicSpec(BaseModel):
    """The current topic as handed to specialists/synthesizer."""

    model_config = ConfigDict(frozen=True)

    topic_id: str
    summary: str
    source_type: TopicSourceType
    cited_finding_ids: list[str] = Field(default_factory=list)
    cited_narrative_ids: list[str] = Field(default_factory=list)
    cited_trajectories: list[TrajectoryRef] = Field(default_factory=list)
    affected_variables: list[str] = Field(default_factory=list)


# ---------- Facets + hypotheses ----------


class FacetSummary(BaseModel):
    """Compact view of a facet for cross-specialist context (vote-prep / view)."""

    model_config = ConfigDict(frozen=True)

    facet_id: str
    specialist: SpecialistRole
    summary: str
    confidence: float


class FacetFull(BaseModel):
    """Full facet contribution; what synthesizer consumes."""

    model_config = ConfigDict(frozen=True)

    facet_id: str
    specialist: SpecialistRole
    summary: str = Field(min_length=1)
    cited_finding_ids: list[str] = Field(default_factory=list)
    cited_narrative_ids: list[str] = Field(default_factory=list)
    cited_trajectories: list[TrajectoryRef] = Field(default_factory=list)
    affected_variables: list[str] = Field(default_factory=list)
    confidence: float = Field(ge=0.0, le=LLM_CONFIDENCE_CAP)
    confidence_basis: ConfidenceBasis

    @field_validator("facet_id")
    @classmethod
    def _facet_id_shape(cls, v: str) -> str:
        if not FACET_ID_RE.match(v):
            raise ValueError(f"facet_id must match FCT-NNNN, got {v!r}")
        return v

    @model_validator(mode="after")
    def _has_citation(self) -> FacetFull:
        if (
            not self.cited_finding_ids
            and not self.cited_narrative_ids
            and not self.cited_trajectories
        ):
            raise ValueError(
                f"{self.facet_id}: facet must cite ≥1 finding, narrative,"
                " or trajectory"
            )
        return self


class HypothesisFull(BaseModel):
    """Synthesized hypothesis. Citation discipline mirrors FailureClaim."""

    model_config = ConfigDict(frozen=True)

    hyp_id: str
    summary: str = Field(min_length=1)
    facet_ids: list[str] = Field(min_length=1)
    cited_finding_ids: list[str] = Field(default_factory=list)
    cited_narrative_ids: list[str] = Field(default_factory=list)
    cited_trajectories: list[TrajectoryRef] = Field(default_factory=list)
    affected_variables: list[str] = Field(default_factory=list)
    confidence: float = Field(ge=0.0, le=LLM_CONFIDENCE_CAP)
    confidence_basis: ConfidenceBasis
    provenance_downgraded: bool = False

    @field_validator("hyp_id")
    @classmethod
    def _hyp_id_shape(cls, v: str) -> str:
        if not HYP_ID_RE.match(v):
            raise ValueError(f"hyp_id must match H-NNNN, got {v!r}")
        return v

    @model_validator(mode="after")
    def _has_citation(self) -> HypothesisFull:
        if (
            not self.cited_finding_ids
            and not self.cited_narrative_ids
            and not self.cited_trajectories
        ):
            raise ValueError(
                f"{self.hyp_id}: hypothesis must cite ≥1 finding, narrative,"
                " or trajectory"
            )
        return self


class CritiqueFull(BaseModel):
    """Critic's structured output."""

    model_config = ConfigDict(frozen=True)

    hyp_id: str
    flag: Literal["red", "green"]
    reasons: list[str] = Field(default_factory=list)
    tool_calls_used: int = Field(default=0, ge=0)

    @model_validator(mode="after")
    def _red_flag_needs_reasons(self) -> CritiqueFull:
        if self.flag == "red" and not self.reasons:
            raise ValueError(
                f"{self.hyp_id}: red-flag critique must include ≥1 reason"
            )
        return self


# ---------- Open questions + ledger ----------


class OpenQuestionRef(BaseModel):
    """Hypothesis-stage open question. `qid` is Q-NNNN; distinct from
    DiagnosisOutput's D-Q-NNNN to keep namespaces separable.
    """

    model_config = ConfigDict(frozen=True)

    qid: str
    question: str = Field(min_length=1)
    raised_by: str = Field(
        description="Specialist role string or 'orchestrator'."
    )
    tags: list[str] = Field(default_factory=list)
    resolved: bool = False
    resolution: str | None = None

    @field_validator("qid")
    @classmethod
    def _qid_shape(cls, v: str) -> str:
        if not QID_RE.match(v):
            raise ValueError(f"qid must match Q-NNNN, got {v!r}")
        return v

    @model_validator(mode="after")
    def _resolved_needs_resolution(self) -> OpenQuestionRef:
        if self.resolved and not self.resolution:
            raise ValueError(
                f"{self.qid}: resolved=True requires non-empty resolution"
            )
        return self


# ---------- Budget + instrumentation ----------


class BudgetSnapshot(BaseModel):
    """Live budget state. Mutated only by runner via decrement helpers."""

    max_turns: int = 5
    max_critic_cycles_per_topic: int = 2
    max_tool_calls_total: int = 80
    max_tokens_per_agent_call: int = 4000
    max_open_questions: int = 15
    max_total_input_tokens: int = 200_000

    turns_used: int = 0
    tool_calls_used: int = 0
    total_input_tokens: int = 0
    total_output_tokens: int = 0

    def is_exhausted(self) -> tuple[bool, str | None]:
        if self.tool_calls_used >= self.max_tool_calls_total:
            return True, "max_tool_calls_total"
        if self.total_input_tokens >= self.max_total_input_tokens:
            return True, "max_total_input_tokens"
        if self.turns_used >= self.max_turns:
            return True, "max_turns"
        return False, None


class TokenReport(BaseModel):
    """Per-agent input/output totals at end of stage."""

    per_agent_input: dict[str, int] = Field(default_factory=dict)
    per_agent_output: dict[str, int] = Field(default_factory=dict)
    total_input: int = 0
    total_output: int = 0


# ---------- Outcome refs ----------


class HypothesisRef(BaseModel):
    """Compact reference for OrchestratorView."""

    model_config = ConfigDict(frozen=True)

    hyp_id: str
    summary: str


class TurnOutcome(BaseModel):
    """One-line summary of last turn for OrchestratorView."""

    model_config = ConfigDict(frozen=True)

    turn: int
    topic_id: str
    outcome: Literal["accepted", "rejected", "no_hypothesis"]
    hyp_id: str | None = None


class CitationCatalog(BaseModel):
    """Union of all cited IDs across facets, surfaced to synthesizer."""

    model_config = ConfigDict(frozen=True)

    finding_ids: list[str] = Field(default_factory=list)
    narrative_ids: list[str] = Field(default_factory=list)
    trajectories: list[TrajectoryRef] = Field(default_factory=list)


# ---------- Lightweight refs for views ----------


class FindingRef(BaseModel):
    model_config = ConfigDict(frozen=True)

    finding_id: str
    summary: str
    variables_involved: list[str] = Field(default_factory=list)


class NarrativeRef(BaseModel):
    model_config = ConfigDict(frozen=True)

    narrative_id: str
    tag: str
    summary: str
    run_id: str | None = None


class TrajectoryViewRef(BaseModel):
    model_config = ConfigDict(frozen=True)

    run_id: str
    variable: str
    note: str = ""


class ResolvedPriorRef(BaseModel):
    """Mirror of fermdocs.domain.process_priors.ResolvedPrior, but as a
    BaseModel so views serialize cleanly. Loaded lazily from the priors layer."""

    model_config = ConfigDict(frozen=True)

    organism: str
    process_family: str
    variable: str
    range_low: float | None = None
    range_high: float | None = None
    typical: float | None = None
    source: str = ""


# ---------- Role-shaped views ----------


class OrchestratorView(BaseModel):
    """What the orchestrator-LLM sees each turn. Per plan §4."""

    current_turn: int
    budget_remaining: BudgetSnapshot
    top_topics: list[RankedTopic]
    open_questions: list[OpenQuestionRef]
    last_turn_outcome: TurnOutcome | None = None
    accepted_hypotheses_so_far: list[HypothesisRef] = Field(default_factory=list)


class SpecialistView(BaseModel):
    specialist_role: SpecialistRole
    current_topic: TopicSpec
    relevant_findings: list[FindingRef] = Field(default_factory=list)
    relevant_narratives: list[NarrativeRef] = Field(default_factory=list)
    relevant_trajectories: list[TrajectoryViewRef] = Field(default_factory=list)
    relevant_priors: list[ResolvedPriorRef] = Field(default_factory=list)
    open_questions_in_domain: list[OpenQuestionRef] = Field(default_factory=list)
    prior_facets_this_topic: list[FacetSummary] = Field(default_factory=list)


class SynthesizerView(BaseModel):
    current_topic: TopicSpec
    facets: list[FacetFull]
    citation_universe: CitationCatalog


class CriticView(BaseModel):
    hypothesis: HypothesisFull
    citation_lookups: dict[str, Any] = Field(default_factory=dict)
    relevant_priors: list[ResolvedPriorRef] = Field(default_factory=list)
    debate_summary_one_line: str = ""


class JudgeView(BaseModel):
    hypothesis: HypothesisFull
    critique: CritiqueFull
    citation_lookups: dict[str, Any] = Field(default_factory=dict)
    # Explicitly NO debate history — see plan §14 (judge collusion mitigation).


# ---------- Output (final + rejected) ----------


class FinalHypothesis(BaseModel):
    """Green-flagged or terminal hypothesis surfaced to humans."""

    model_config = ConfigDict(frozen=True)

    hyp_id: str
    summary: str
    facet_ids: list[str]
    cited_finding_ids: list[str] = Field(default_factory=list)
    cited_narrative_ids: list[str] = Field(default_factory=list)
    cited_trajectories: list[TrajectoryRef] = Field(default_factory=list)
    affected_variables: list[str] = Field(default_factory=list)
    confidence: float = Field(ge=0.0, le=LLM_CONFIDENCE_CAP)
    confidence_basis: ConfidenceBasis
    provenance_downgraded: bool = False
    supporting_specialists: list[SpecialistRole] = Field(default_factory=list)
    critic_flag: Literal["red", "green"]
    judge_ruled_criticism_valid: bool


class RejectedHypothesis(BaseModel):
    """Hypothesis rejected by judge-approved criticism."""

    model_config = ConfigDict(frozen=True)

    hyp_id: str
    summary: str
    rejection_reason: str
    critic_reasons: list[str] = Field(default_factory=list)
    judge_rationale: str = ""


# ---------- Meta + top-level output ----------


class HypothesisMeta(BaseModel):
    model_config = ConfigDict(frozen=True)

    schema_version: str = "1.0"
    hypothesis_version: str
    hypothesis_id: UUID
    supersedes_diagnosis_id: UUID
    generation_timestamp: datetime
    model: str
    provider: Literal["anthropic", "gemini", "stub"]
    budget_used: BudgetSnapshot
    error: str | None = None


class HypothesisInput(BaseModel):
    """What the hypothesis stage receives. Stage 2 wires from real bundles;
    Stage 1 fixtures synthesize fakes.
    """

    diagnosis: Any  # DiagnosisOutput, kept loose for stub fixtures
    characterization: Any  # CharacterizationOutput, ditto
    bundle_path: str | None = None
    seed_topics: list[SeedTopic]
    organism: str | None = None
    process_family: str | None = None


class HypothesisOutput(BaseModel):
    meta: HypothesisMeta
    final_hypotheses: list[FinalHypothesis] = Field(default_factory=list)
    rejected_hypotheses: list[RejectedHypothesis] = Field(default_factory=list)
    open_questions: list[OpenQuestionRef] = Field(default_factory=list)
    debate_summary: str = ""
    global_md_path: str | None = None
    token_report: TokenReport = Field(default_factory=TokenReport)

    @model_validator(mode="after")
    def _no_duplicate_hyp_ids(self) -> HypothesisOutput:
        seen: set[str] = set()
        for h in (*self.final_hypotheses, *self.rejected_hypotheses):
            if h.hyp_id in seen:
                raise ValueError(f"duplicate hyp_id {h.hyp_id!r}")
            seen.add(h.hyp_id)
        return self

    @model_validator(mode="after")
    def _error_implies_empty(self) -> HypothesisOutput:
        if self.meta.error is not None and (
            self.final_hypotheses or self.rejected_hypotheses
        ):
            raise ValueError(
                "meta.error is set but hypothesis lists are non-empty"
            )
        return self
