"""Event types for the hypothesis stage event log.

Plan ref: plans/2026-05-03-hypothesis-debate-v0.md §3, §16.

The event log is the canonical state. Every other artifact (views, ledgers,
tallies, output) is derived from these events. Single writer (Observer);
no agent reads raw events — they read role-shaped views from the projector.

Reserved (no-op in v0, populated in HITL-enabled v1):
  - StagePausedEvent
  - HumanInputReceivedEvent

These are wired through the discriminated union now so that v0 code paths
already serialize/deserialize them correctly when v1 starts emitting them.
"""

from __future__ import annotations

from datetime import datetime
from typing import Annotated, Any, Literal, Union

from pydantic import BaseModel, ConfigDict, Field

from fermdocs_diagnose.schema import ConfidenceBasis, TrajectoryRef
from fermdocs_hypothesis.schema import (
    BudgetSnapshot,
    ExitReason,
    SpecialistRole,
)


class _EventBase(BaseModel):
    model_config = ConfigDict(frozen=True)

    ts: datetime
    turn: int = Field(ge=0, description="0 for stage_started/exited; ≥1 for in-loop events.")


class StageStartedEvent(_EventBase):
    type: Literal["stage_started"] = "stage_started"
    input_diagnosis_id: str
    budget: BudgetSnapshot


class TopicSelectedEvent(_EventBase):
    type: Literal["topic_selected"] = "topic_selected"
    topic_id: str
    summary: str
    rationale: str


class FacetContributedEvent(_EventBase):
    type: Literal["facet_contributed"] = "facet_contributed"
    facet_id: str
    topic_id: str
    specialist: SpecialistRole
    summary: str
    cited_finding_ids: list[str] = Field(default_factory=list)
    cited_narrative_ids: list[str] = Field(default_factory=list)
    cited_trajectories: list[TrajectoryRef] = Field(default_factory=list)
    affected_variables: list[str] = Field(default_factory=list)
    confidence: float
    confidence_basis: ConfidenceBasis


class HypothesisSynthesizedEvent(_EventBase):
    type: Literal["hypothesis_synthesized"] = "hypothesis_synthesized"
    hyp_id: str
    topic_id: str
    summary: str
    facet_ids: list[str]
    cited_finding_ids: list[str] = Field(default_factory=list)
    cited_narrative_ids: list[str] = Field(default_factory=list)
    cited_trajectories: list[TrajectoryRef] = Field(default_factory=list)
    affected_variables: list[str] = Field(default_factory=list)
    confidence: float
    confidence_basis: ConfidenceBasis


class CritiqueFiledEvent(_EventBase):
    type: Literal["critique_filed"] = "critique_filed"
    hyp_id: str
    flag: Literal["red", "green"]
    reasons: list[str] = Field(default_factory=list)
    tool_calls_used: int = 0


class JudgeRulingEvent(_EventBase):
    type: Literal["judge_ruling"] = "judge_ruling"
    hyp_id: str
    criticism_valid: bool
    rationale: str


class HypothesisAcceptedEvent(_EventBase):
    type: Literal["hypothesis_accepted"] = "hypothesis_accepted"
    hyp_id: str
    topic_id: str


class HypothesisRejectedEvent(_EventBase):
    type: Literal["hypothesis_rejected"] = "hypothesis_rejected"
    hyp_id: str
    topic_id: str
    reason: str


class QuestionAddedEvent(_EventBase):
    type: Literal["question_added"] = "question_added"
    qid: str
    question: str
    raised_by: str
    tags: list[str] = Field(default_factory=list)


class QuestionResolvedEvent(_EventBase):
    type: Literal["question_resolved"] = "question_resolved"
    qid: str
    resolution: str


class TokensUsedEvent(_EventBase):
    type: Literal["tokens_used"] = "tokens_used"
    agent: str
    input: int = Field(ge=0)
    output: int = Field(ge=0)


class StagePausedEvent(_EventBase):
    """Reserved for HITL v1. Runner emits when waiting on human input."""

    type: Literal["stage_paused"] = "stage_paused"
    reason: Literal["awaiting_human"] = "awaiting_human"
    context: dict[str, Any] = Field(default_factory=dict)


class HumanInputReceivedEvent(_EventBase):
    """Reserved for HITL v1. Runner emits when human input is consumed."""

    type: Literal["human_input_received"] = "human_input_received"
    input_type: Literal["answer", "critique", "accept"]
    payload: dict[str, Any] = Field(default_factory=dict)


class StageExitedEvent(_EventBase):
    type: Literal["stage_exited"] = "stage_exited"
    reason: ExitReason
    final_hyp_ids: list[str] = Field(default_factory=list)


Event = Annotated[
    Union[
        StageStartedEvent,
        TopicSelectedEvent,
        FacetContributedEvent,
        HypothesisSynthesizedEvent,
        CritiqueFiledEvent,
        JudgeRulingEvent,
        HypothesisAcceptedEvent,
        HypothesisRejectedEvent,
        QuestionAddedEvent,
        QuestionResolvedEvent,
        TokensUsedEvent,
        StagePausedEvent,
        HumanInputReceivedEvent,
        StageExitedEvent,
    ],
    Field(discriminator="type"),
]


class EventEnvelope(BaseModel):
    """Container so we can parse a single event without knowing its type."""

    event: Event
