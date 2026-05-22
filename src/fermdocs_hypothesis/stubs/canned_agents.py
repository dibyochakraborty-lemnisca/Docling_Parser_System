"""Stub agents — deterministic, scriptable, no LLM.

Plan ref: plans/2026-05-03-hypothesis-debate-v0.md §11 Stage 1.

Each stub returns the structured object the real LLM agent will return in
Stage 2, plus a fake (input_tokens, output_tokens) pair so the meter
exercises end-to-end. The runner doesn't know stubs from real agents.

Behavior is scripted via StubScript — a per-topic plan of what facets each
specialist contributes, what flag the critic raises, how the judge rules.
This lets one e2e test exercise accept paths, reject-then-retry, and exit
conditions without code branches in the runner.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

from fermdocs_diagnose.schema import ConfidenceBasis, TrajectoryRef
from fermdocs_hypothesis.schema import (
    CritiqueFull,
    FacetFull,
    HypothesisFull,
    SpecialistRole,
    SpecialistView,
    SynthesizerView,
    TopicSpec,
)


@dataclass
class StubFacetPlan:
    summary: str
    cited_finding_ids: list[str] = field(default_factory=list)
    cited_narrative_ids: list[str] = field(default_factory=list)
    cited_trajectories: list[TrajectoryRef] = field(default_factory=list)
    affected_variables: list[str] = field(default_factory=list)
    confidence: float = 0.7
    confidence_basis: ConfidenceBasis = ConfidenceBasis.SCHEMA_ONLY
    input_tokens: int = 800
    output_tokens: int = 200


@dataclass
class StubTopicPlan:
    """How the canned run should play out for one topic_id."""

    facets: dict[SpecialistRole, StubFacetPlan]
    synthesis_summary: str
    synthesis_confidence: float = 0.7
    synthesis_basis: ConfidenceBasis = ConfidenceBasis.SCHEMA_ONLY
    critic_flag: Literal["red", "green"] = "green"
    critic_reasons: list[str] = field(default_factory=list)
    critic_tool_calls: int = 1
    judge_criticism_valid: bool = False
    judge_rationale: str = "stub: critic green-flagged, nothing to judge"
    synthesizer_input_tokens: int = 1200
    synthesizer_output_tokens: int = 300
    critic_input_tokens: int = 1500
    critic_output_tokens: int = 250
    judge_input_tokens: int = 600
    judge_output_tokens: int = 100


@dataclass
class StubScript:
    """Top-level script: per-topic plans + orchestrator topic-pick order."""

    topic_plans: dict[str, StubTopicPlan]
    topic_order: list[str]
    orchestrator_input_tokens: int = 500
    orchestrator_output_tokens: int = 80
    open_questions_to_add: list[tuple[str, str, list[str]]] = field(
        default_factory=list,
        metadata={"doc": "(qid, question, tags) tuples — added on first orchestrator turn."},
    )


# ---------- stub agent callables ----------


def stub_orchestrator_pick_topic(
    script: StubScript,
    *,
    topics_already_used_in_order: list[str],
    available_topic_ids: set[str],
) -> str | None:
    """Pick the next topic from script.topic_order that hasn't been used yet
    and is in available_topic_ids. Returns None when exhausted.
    """
    for tid in script.topic_order:
        if tid in topics_already_used_in_order:
            continue
        if tid in available_topic_ids:
            return tid
    return None


def stub_specialist_contribute(
    view: SpecialistView,
    plan: StubFacetPlan,
    facet_id: str,
) -> FacetFull:
    return FacetFull(
        facet_id=facet_id,
        specialist=view.specialist_role,
        summary=plan.summary,
        cited_finding_ids=list(plan.cited_finding_ids),
        cited_narrative_ids=list(plan.cited_narrative_ids),
        cited_trajectories=list(plan.cited_trajectories),
        affected_variables=list(plan.affected_variables) or list(view.current_topic.affected_variables),
        confidence=plan.confidence,
        confidence_basis=plan.confidence_basis,
    )


def stub_synthesizer_emit(
    view: SynthesizerView,
    plan: StubTopicPlan,
    hyp_id: str,
) -> HypothesisFull:
    return HypothesisFull(
        hyp_id=hyp_id,
        summary=plan.synthesis_summary,
        facet_ids=[f.facet_id for f in view.facets],
        cited_finding_ids=list(view.citation_universe.finding_ids),
        cited_narrative_ids=list(view.citation_universe.narrative_ids),
        cited_trajectories=list(view.citation_universe.trajectories),
        affected_variables=list(view.current_topic.affected_variables),
        confidence=plan.synthesis_confidence,
        confidence_basis=plan.synthesis_basis,
    )


def stub_critic_file(plan: StubTopicPlan, hyp_id: str) -> CritiqueFull:
    return CritiqueFull(
        hyp_id=hyp_id,
        flag=plan.critic_flag,
        reasons=list(plan.critic_reasons),
        tool_calls_used=plan.critic_tool_calls,
    )


def stub_judge_rule(plan: StubTopicPlan, hyp_id: str) -> tuple[bool, str]:
    return plan.judge_criticism_valid, plan.judge_rationale


# ---------- helpers for the runner ----------


def topic_spec_from_seed(seed) -> TopicSpec:
    """Convert a SeedTopic to the TopicSpec view shape."""
    return TopicSpec(
        topic_id=seed.topic_id,
        summary=seed.summary,
        source_type=seed.source_type,
        cited_finding_ids=list(seed.cited_finding_ids),
        cited_narrative_ids=list(seed.cited_narrative_ids),
        cited_trajectories=list(seed.cited_trajectories),
        affected_variables=list(seed.affected_variables),
    )
