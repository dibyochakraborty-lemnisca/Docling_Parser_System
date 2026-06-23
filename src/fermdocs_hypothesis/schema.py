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

from fermdocs.domain.user_question import UserQuestion
from typing import Literal

from fermdocs_characterize.schema import Severity
from fermdocs_diagnose.schema import (
    ConfidenceBasis,
    LLM_CONFIDENCE_CAP,
    TrajectoryRef,
)


# ---------- Chart specs (LLM intent → deterministic Plotly render) ----------

CHART_KIND = Literal[
    "time_series_overlay",
    "scatter_correlation",
    "faceted_time_series",
]


class ChartAnnotation(BaseModel):
    """LLM-authored callout on a chart.

    For time-series charts: time_h pins the x position; run_id optionally
    pins to a specific trace. For scatter charts: time_h is the x-axis
    value, run_id pins to the labelled point.
    """

    model_config = ConfigDict(frozen=True)

    text: str = Field(min_length=1, max_length=200)
    time_h: float | None = None
    run_id: str | None = None


class ChartSpec(BaseModel):
    """Synthesizer-emitted intent for one chart.

    The LLM picks WHAT to plot (kind, runs, variables, story). The
    deterministic builder (chart_builder.py) slices the bundle data and
    constructs the Plotly JSON. This split lets the LLM be creative
    without being able to fabricate values.

    Per chart kind:
      - time_series_overlay: x = time, y = variables[0], one trace per run.
        Highlight runs get bolder strokes / distinct colors.
      - scatter_correlation: x = variables[0], y = variables[1] across
        the listed runs (one point per run). Builder adds regression
        line + bootstrap CI band when n ≥ 4.
      - faceted_time_series: same as overlay but one subplot per run
        instead of overlaying. Useful when y-scales differ per run.
    """

    model_config = ConfigDict(frozen=True)

    kind: CHART_KIND
    title: str = Field(min_length=1, max_length=200)
    rationale: str = Field(
        min_length=1,
        max_length=400,
        description=(
            "One sentence: why this chart strengthens the hypothesis. Read by"
            " reviewers, not the LLM — keep it concrete."
        ),
    )
    runs: list[str] = Field(
        default_factory=list,
        description="Subset of cohort run_ids to plot. Empty = all runs.",
    )
    variables: list[str] = Field(
        min_length=1,
        max_length=2,
        description=(
            "For time_series_overlay/faceted: 1 variable. For"
            " scatter_correlation: 2 variables (x, y)."
        ),
    )
    highlight_runs: list[str] = Field(
        default_factory=list,
        description=(
            "Subset of `runs` to emphasise. Renderer uses thicker line /"
            " different color. Optional."
        ),
    )
    annotations: list[ChartAnnotation] = Field(
        default_factory=list,
        max_length=8,
        description="LLM-authored callouts on the chart.",
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
    # PR-A2 drive posture: synthetic topics manufactured from the user's
    # follow-up question itself, not from the diagnose output. Specialists
    # see them as "the user proposes / asks about X — test it."
    USER_MECHANISM = "user_mechanism"
    USER_COMPARISON = "user_comparison"
    USER_SCOPE = "user_scope"  # placeholder for empty-scope path (D3)


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
    effect_size: float = Field(
        default=0.0, ge=0.0, le=1.0,
        description=(
            "Normalized cross-run association strength (|delta|/outcome spread) "
            "for opportunity-debate lever topics. 0.0 for every other topic "
            "(diagnosis failures/analyses), so the ranker's effect term is inert "
            "outside the optimize debate."
        ),
    )

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
    cited_association_ids: list[str] = Field(
        default_factory=list,
        description=(
            "Within-run association ids (WRA-*) this facet is grounded in — the"
            " cross-run lever->objective evidence a design factor cites when it"
            " has no measured trajectory. Counts as grounding, so an"
            " association-grounded facet is NOT flagged schema_only."
        ),
    )
    affected_variables: list[str] = Field(default_factory=list)
    confidence: float = Field(ge=0.0, le=LLM_CONFIDENCE_CAP)
    confidence_basis: ConfidenceBasis

    @field_validator("facet_id")
    @classmethod
    def _facet_id_shape(cls, v: str) -> str:
        if not FACET_ID_RE.match(v):
            raise ValueError(f"facet_id must match FCT-NNNN, got {v!r}")
        return v

    @model_validator(mode="before")
    @classmethod
    def _flag_if_uncited(cls, data):
        """A facet that cites no finding/narrative/trajectory/association is NOT
        rejected — it's kept but flagged un-grounded (schema_only basis,
        confidence capped), so the critic can judge it on merit and red-flag weak
        provenance instead of the whole run crashing. (flag-don't-fake: we never
        fabricate a citation just to pass validation.)"""
        if isinstance(data, dict):
            grounded = bool(
                data.get("cited_finding_ids")
                or data.get("cited_narrative_ids")
                or data.get("cited_trajectories")
                or data.get("cited_association_ids")
            )
            if not grounded:
                data = {
                    **data,
                    "confidence_basis": ConfidenceBasis.SCHEMA_ONLY,
                    "confidence": min(float(data.get("confidence") or 0.0), 0.4),
                }
        return data


class HypothesisFull(BaseModel):
    """Synthesized hypothesis. Citation discipline mirrors FailureClaim."""

    model_config = ConfigDict(frozen=True)

    hyp_id: str
    summary: str = Field(min_length=1)
    facet_ids: list[str] = Field(min_length=1)
    cited_finding_ids: list[str] = Field(default_factory=list)
    cited_narrative_ids: list[str] = Field(default_factory=list)
    cited_trajectories: list[TrajectoryRef] = Field(default_factory=list)
    cited_association_ids: list[str] = Field(
        default_factory=list,
        description="Within-run association ids (WRA-*) carried up from the facets.",
    )
    affected_variables: list[str] = Field(default_factory=list)
    confidence: float = Field(ge=0.0, le=LLM_CONFIDENCE_CAP)
    confidence_basis: ConfidenceBasis
    provenance_downgraded: bool = False
    question_answered: Literal["yes", "partial", "insufficient_data"] | None = Field(
        default=None,
        description=(
            "Synthesizer populates when view.user_question is non-null."
            " Carries forward to FinalHypothesis untouched. None on legacy"
            " runs."
        ),
    )
    question_response_summary: str | None = Field(
        default=None,
        max_length=800,
        description=(
            "One-paragraph response to the user's question. Synthesizer"
            " populates when user_question is non-null. Carries forward."
        ),
    )
    actionable_recommendation: str | None = Field(
        default=None,
        max_length=600,
        description=(
            "Concrete next-batch parameter change implied by the"
            " hypothesis (e.g. 'design Batch 7 with DO setpoint ≥40%"
            " throughout fed-batch'). On green-flagged hypotheses the"
            " judge enforces presence; the explicit string"
            " 'insufficient evidence to recommend: <reason>' is also"
            " valid. None on red-flagged or legacy runs."
        ),
    )
    chart_specs: list[ChartSpec] = Field(
        default_factory=list,
        max_length=3,
        description=(
            "LLM-emitted chart intents for this hypothesis. Empty ="
            " descriptive-only claim where a chart wouldn't add signal."
            " Builder turns each spec into Plotly JSON downstream."
        ),
    )
    parent_hypothesis_ids: list[str] = Field(
        default_factory=list,
        description=(
            "Prior hypotheses this follow-up answer refines or contradicts."
            " Empty on original and legacy runs."
        ),
    )

    @field_validator("hyp_id")
    @classmethod
    def _hyp_id_shape(cls, v: str) -> str:
        if not HYP_ID_RE.match(v):
            raise ValueError(f"hyp_id must match H-NNNN, got {v!r}")
        return v

    @model_validator(mode="before")
    @classmethod
    def _flag_if_uncited(cls, data):
        """An uncited hypothesis is kept but flagged (provenance_downgraded,
        schema_only basis, confidence capped) rather than rejected — the critic
        then judges its weak grounding and can red-flag it. We never fabricate a
        citation to pass validation (flag-don't-fake)."""
        if isinstance(data, dict):
            grounded = bool(
                data.get("cited_finding_ids")
                or data.get("cited_narrative_ids")
                or data.get("cited_trajectories")
                or data.get("cited_association_ids")
            )
            if not grounded:
                data = {
                    **data,
                    "provenance_downgraded": True,
                    "confidence_basis": ConfidenceBasis.SCHEMA_ONLY,
                    "confidence": min(float(data.get("confidence") or 0.0), 0.4),
                }
        return data


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


# ---------- Feedback ledger (debate history visible to retrying agents) ----------


class HumanInputRecord(BaseModel):
    """Slot for HITL v1. Empty in v0; populated when humans inject guidance
    on an attempt (accept-with-note, reject-with-note, free-form steer).

    Schema kept minimal so the projector + agent prompts can reference the
    field today; richer fields (qid linkage, structured payload) land with
    HITL wiring.
    """

    model_config = ConfigDict(frozen=True)

    note: str = Field(min_length=1)


class AttemptRecord(BaseModel):
    """One synthesizer→critic→judge cycle on a single topic.

    Built by `state.topic_history` from event log. Surfaced into agent
    views so retries can address prior critic reasons explicitly instead
    of re-emitting the same overreach.
    """

    model_config = ConfigDict(frozen=True)

    hyp_id: str
    hypothesis_summary: str
    critic_flag: Literal["red", "green"] | None = None
    critic_reasons: list[str] = Field(default_factory=list)
    judge_ruling: Literal["valid", "invalid"] | None = None
    judge_rationale: str | None = None
    human_input: HumanInputRecord | None = None


class TopicHistoryEntry(BaseModel):
    """All attempts on one topic + terminal status.

    `status` reflects the topic's outcome at the time of projection:
      - in_progress: at least one attempt, none accepted, retry budget left
      - rejected_terminal: retry budget exhausted on this topic
      - accepted: hypothesis on this topic was accepted by judge
      - deferred: topic selected but no synthesis happened (rare)
    """

    model_config = ConfigDict(frozen=True)

    topic_id: str
    summary: str
    attempts: list[AttemptRecord] = Field(default_factory=list)
    status: Literal["in_progress", "rejected_terminal", "accepted", "deferred"]


class Lesson(BaseModel):
    """One distilled lesson with a stable id (D2, memory-layer plan).

    `lesson_id` is minted by the lessons_summarizer at emission time and
    survives across runs — when this lesson is persisted to the memory
    layer (Tier 1), the id round-trips as `MemoryRecord.memory_id` and
    serves as Synap's `document_id` idempotency key.
    """

    model_config = ConfigDict(frozen=True)

    lesson_id: str = Field(min_length=1, description="L-<run_short>-<NNNN> form.")
    text: str = Field(min_length=1, max_length=400)
    tags: list[str] = Field(default_factory=list)


class LessonsDigest(BaseModel):
    """LLM-summarized cross-topic critic patterns.

    Computed by LessonsSummarizerAgent on retry. `source_reason_count` and
    `computed_at_event_idx` are cache keys: the runner skips recompute when
    the live critic-reason count hasn't grown past `source_reason_count`.

    Two equivalent representations of the lessons live on this object:
      - `digest`: legacy single-string form for prompt rendering and for
        existing global.md replay back-compat. Always populated.
      - `lessons`: structured list with stable lesson_ids (D2). Populated
        on new runs; empty on legacy global.md replay where the event
        only carried the digest blob.

    When persisting to memory, the runner reads `lessons` (writes one
    MemoryRecord per Lesson). When `lessons` is empty (legacy event),
    no memory write happens — graceful degradation.
    """

    model_config = ConfigDict(frozen=True)

    digest: str = Field(min_length=1)
    source_reason_count: int = Field(ge=0)
    computed_at_event_idx: int = Field(ge=0)
    lessons: list[Lesson] = Field(
        default_factory=list,
        description=(
            "Structured form. Empty on legacy events; populated on every"
            " new run after memory-layer Phase 1 lands."
        ),
    )


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
    """Live budget state. Mutated only by runner via decrement helpers.

    Defaults reflect Stage 3 production budget. Tests pass tighter caps.
    """

    max_turns: int = 20
    max_critic_cycles_per_topic: int = 6
    max_tool_calls_total: int = 160
    max_tokens_per_agent_call: int = 8000
    max_open_questions: int = 15
    max_total_input_tokens: int = 400_000
    skip_critic: bool = False

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
    metric_id: str | None = Field(
        default=None,
        description=(
            "Stable identifier from fermdocs_characterize.agents.metric_catalog "
            "when this finding came from a verified toolkit function. Lets the "
            "hypothesis projector route findings to specialists by domain "
            "(kinetics/mass_transfer/metabolic) without depending on tag "
            "string overlap."
        ),
    )


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


class AnalysisRef(BaseModel):
    """Diagnose-layer analysis claim surfaced to specialists as a caveat.

    When the diagnose stage emitted an analysis explaining-away a finding
    (e.g. spec_alignment: 'these aren't process anomalies, the schema is
    misconfigured'), specialists need to see it so they don't re-derive
    a hypothesis that ignores the caveat.
    """

    model_config = ConfigDict(frozen=True)

    claim_id: str
    summary: str
    kind: str
    cited_finding_ids: list[str] = Field(default_factory=list)
    affected_variables: list[str] = Field(default_factory=list)


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


# ---------- Follow-up context ----------


class PriorHypothesisRef(BaseModel):
    """Compact prior conclusion shown to follow-up runs."""

    model_config = ConfigDict(frozen=True)

    hyp_id: str
    summary: str
    question_answered: Literal["yes", "partial", "insufficient_data"] | None = None
    question_response_summary: str | None = None
    affected_variables: list[str] = Field(default_factory=list)
    confidence: float = Field(ge=0.0, le=LLM_CONFIDENCE_CAP)
    actionable_recommendation: str | None = None
    parent_hypothesis_ids: list[str] = Field(default_factory=list)


class PriorFollowupRef(BaseModel):
    """Compact record of a completed follow-up on the same run."""

    model_config = ConfigDict(frozen=True)

    followup_index: int = Field(ge=1)
    user_question_text: str
    final_hypotheses: list[PriorHypothesisRef] = Field(default_factory=list)


class FollowupContext(BaseModel):
    """Conversation memory for drive-posture follow-up runs.

    The frozen evidence bundle remains the source of truth. This context only
    tells agents what they previously concluded so follow-up questions can
    refine, compare, or contradict prior answers.
    """

    model_config = ConfigDict(frozen=True)

    original_final_hypotheses: list[PriorHypothesisRef] = Field(default_factory=list)
    previous_followups: list[PriorFollowupRef] = Field(default_factory=list)


# ---------- Role-shaped views ----------


class OrchestratorView(BaseModel):
    """What the orchestrator-LLM sees each turn. Per plan §4."""

    current_turn: int
    budget_remaining: BudgetSnapshot
    top_topics: list[RankedTopic]
    open_questions: list[OpenQuestionRef]
    last_turn_outcome: TurnOutcome | None = None
    accepted_hypotheses_so_far: list[HypothesisRef] = Field(default_factory=list)
    user_question: UserQuestion | None = None
    followup_context: FollowupContext | None = None


class WithinRunAssociationRef(BaseModel):
    """A cross-run association computed WITHIN this experiment: how a controllable
    design factor (lever) tracked the objective across the runs. Distinct from
    `cross_run_lessons` (memory from PRIOR experiments) — this is this-experiment
    evidence the opportunity-debate specialists reason from, so a design factor
    with no measured trajectory still has grounded numbers to argue. Observational
    (association, not proven causation)."""

    model_config = ConfigDict(frozen=True)

    assoc_id: str
    lever: str
    summary: str = Field(min_length=1)
    delta: float
    direction: str
    n: int
    norm_effect: float = Field(ge=0.0, le=1.0)
    objective: str
    best_setting: str | float | None = None


class SpecialistView(BaseModel):
    specialist_role: SpecialistRole
    current_topic: TopicSpec
    relevant_findings: list[FindingRef] = Field(default_factory=list)
    relevant_narratives: list[NarrativeRef] = Field(default_factory=list)
    relevant_trajectories: list[TrajectoryViewRef] = Field(default_factory=list)
    relevant_priors: list[ResolvedPriorRef] = Field(default_factory=list)
    relevant_analyses: list[AnalysisRef] = Field(
        default_factory=list,
        description=(
            "Diagnose-layer analyses overlapping the topic's cited findings."
            " Specialists must read these as caveats: if an analysis already"
            " explains a finding away (e.g. spec_alignment), the hypothesis"
            " should reflect that, not re-derive a process anomaly."
        ),
    )
    open_questions_in_domain: list[OpenQuestionRef] = Field(default_factory=list)
    prior_facets_this_topic: list[FacetSummary] = Field(default_factory=list)
    relevant_within_run: list[WithinRunAssociationRef] = Field(
        default_factory=list,
        description=(
            "Within-experiment lever->objective associations (opportunity debate)."
            " Top design factors by effect size, so the specialist argues from"
            " what actually moved the objective across runs. Empty in the"
            " diagnosis-driven hypothesis stage (no levers)."
        ),
    )
    user_question: UserQuestion | None = None
    followup_context: FollowupContext | None = None
    cross_run_lessons: LessonsDigest | None = Field(
        default=None,
        description=(
            "Lessons from PRIOR RUNS on the same process_family, fetched"
            " from the memory layer at view-build time. Distinct from"
            " cross_topic_lessons (this-run only). None on cold-start"
            " process_families and on memory-disabled (NoopBackend) runs."
        ),
    )


class SynthesizerView(BaseModel):
    current_topic: TopicSpec
    facets: list[FacetFull]
    citation_universe: CitationCatalog
    previous_attempts: list[AttemptRecord] = Field(
        default_factory=list,
        description=(
            "Prior rejected attempts on `current_topic`, oldest-first."
            " Empty on first attempt. Synthesizer must address each"
            " critic_reasons entry rather than re-emit the same claim."
        ),
    )
    cross_topic_lessons: LessonsDigest | None = Field(
        default=None,
        description=(
            "LLM-summarized recurring critic complaints across the run."
            " Populated only on retries when enough critic reasons have"
            " accumulated. None means: nothing to add beyond per-topic"
            " context."
        ),
    )
    cross_run_lessons: LessonsDigest | None = Field(
        default=None,
        description=(
            "Lessons from PRIOR RUNS on the same process_family, fetched"
            " from the memory layer. Distinct from cross_topic_lessons"
            " which only spans the current run."
        ),
    )
    user_question: UserQuestion | None = None
    followup_context: FollowupContext | None = None


class CriticView(BaseModel):
    hypothesis: HypothesisFull
    citation_lookups: dict[str, Any] = Field(default_factory=dict)
    relevant_priors: list[ResolvedPriorRef] = Field(default_factory=list)
    debate_summary_one_line: str = ""
    previous_attempts: list[AttemptRecord] = Field(
        default_factory=list,
        description=(
            "Prior attempts on this topic so the critic doesn't repeat"
            " itself or contradict an earlier ruling."
        ),
    )
    cross_topic_lessons: LessonsDigest | None = None
    cross_run_lessons: LessonsDigest | None = None
    user_question: UserQuestion | None = None
    followup_context: FollowupContext | None = None


class JudgeView(BaseModel):
    hypothesis: HypothesisFull
    critique: CritiqueFull
    citation_lookups: dict[str, Any] = Field(default_factory=dict)
    # Explicitly NO debate history of *other* topics — see plan §14 (judge
    # collusion mitigation). previous_attempts is scoped to *this* topic
    # only, which is consistent with judge already seeing this hypothesis
    # + critique pair.
    previous_attempts: list[AttemptRecord] = Field(default_factory=list)
    cross_topic_lessons: LessonsDigest | None = None
    user_question: UserQuestion | None = None
    followup_context: FollowupContext | None = None


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
    cited_association_ids: list[str] = Field(
        default_factory=list,
        description="Within-run association ids (WRA-*) grounding this hypothesis.",
    )
    affected_variables: list[str] = Field(default_factory=list)
    confidence: float = Field(ge=0.0, le=LLM_CONFIDENCE_CAP)
    confidence_basis: ConfidenceBasis
    provenance_downgraded: bool = False
    supporting_specialists: list[SpecialistRole] = Field(default_factory=list)
    critic_flag: Literal["red", "green"]
    judge_ruled_criticism_valid: bool
    question_answered: Literal["yes", "partial", "insufficient_data"] | None = Field(
        default=None,
        description=(
            "When the run had a UserQuestion, synthesizer populates this:"
            " 'yes' = evidence directly addresses the question,"
            " 'partial' = evidence addresses part of it,"
            " 'insufficient_data' = bundle doesn't support an answer."
            " None on runs without a UserQuestion (back-compat)."
        ),
    )
    question_response_summary: str | None = Field(
        default=None,
        max_length=800,
        description=(
            "One paragraph stating what we learned about the user's"
            " question. None on runs without a UserQuestion."
        ),
    )
    actionable_recommendation: str | None = Field(
        default=None,
        max_length=600,
        description=(
            "Concrete next-batch parameter change implied by this"
            " hypothesis. Judge enforces presence on green-flagged"
            " hypotheses; the literal prefix 'insufficient evidence'"
            " satisfies the gate when the bundle truly doesn't support"
            " a recommendation. None on red-flagged or legacy runs."
        ),
    )
    chart_specs: list[ChartSpec] = Field(
        default_factory=list,
        max_length=3,
        description=(
            "Synthesizer-emitted chart intents. Each entry is rendered"
            " by the deterministic chart_builder against the bundle's"
            " trajectory data; the frontend renders the resulting"
            " Plotly JSON inline. Empty on legacy runs and on hypotheses"
            " where the LLM judged a chart wouldn't add signal."
        ),
    )
    plotly_charts: list[dict] = Field(
        default_factory=list,
        max_length=3,
        description=(
            "Rendered Plotly figure JSON, one per chart_spec. Populated"
            " by the runner's projector after build_charts(). Each entry"
            " is a {'data': [...], 'layout': {...}, 'spec': {...}} dict"
            " the frontend feeds straight into react-plotly.js. Empty"
            " mirrors empty chart_specs."
        ),
    )
    parent_hypothesis_ids: list[str] = Field(
        default_factory=list,
        description=(
            "Hypotheses (by hyp_id) from a prior run on the same bundle that"
            " this hypothesis refines or contradicts. Populated only on"
            " follow-up runs (PR-A2, drive posture). Empty on the original"
            " run and on legacy runs. Frontend uses this to render an"
            " '↑ refines H-NNNN' link between chained cards."
        ),
    )
    # A3 — the claim's type, load-bearing for the deterministic gates: a 'causal'
    # or 'recommendation' claim must survive the confound/objective/materiality
    # gates; an 'observational' one is not required to condition. Default
    # 'observational' keeps legacy/uncategorized hypotheses ungated (back-compat).
    claim_type: Literal["observational", "causal", "recommendation"] = "observational"
    # Names of deterministic gates this claim FAILED (attached by the gate bridge,
    # A3). A non-empty hard-fail list means the claim must not enter the verdict.
    gate_failures: list[str] = Field(default_factory=list)


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
    user_question: UserQuestion | None = Field(
        default=None,
        description=(
            "Set when the human typed a question at run start (PR-A, bias"
            " posture) or as a follow-up (PR-A2, drive posture). None on"
            " legacy runs — every downstream agent treats None as 'no"
            " question, run as today'."
        ),
    )
    followup_context: FollowupContext | None = Field(
        default=None,
        description=(
            "Compact memory of prior final hypotheses and completed"
            " follow-ups on the same run. Set only for follow-up runs."
        ),
    )


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
