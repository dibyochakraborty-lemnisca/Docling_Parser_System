"""Shared fixtures for hypothesis-stage Stage 1 tests.

Synthesizes minimal SeedTopics, FacetFulls, HypothesisFulls and a stub
HypothesisInput so tests don't need a real DiagnosisOutput / Characterization
bundle (those land in Stage 2).
"""

from __future__ import annotations

from datetime import datetime, timezone
from uuid import UUID

from fermdocs_characterize.schema import Severity
from fermdocs_diagnose.schema import ConfidenceBasis, TrajectoryRef
from fermdocs_hypothesis.schema import (
    BudgetSnapshot,
    FacetFull,
    HypothesisFull,
    HypothesisInput,
    OpenQuestionRef,
    SeedTopic,
    TopicSourceType,
)
from fermdocs_hypothesis.stubs.canned_agents import (
    StubFacetPlan,
    StubScript,
    StubTopicPlan,
)

CHAR_ID = "00000000-0000-0000-0000-000000000042"
DIAG_ID = UUID(int=99)
NOW = datetime(2026, 5, 3, 12, 0, 0, tzinfo=timezone.utc)


def now_factory_const():
    return NOW


def make_seed_topic(
    topic_id: str = "T-0001",
    summary: str = "biomass plateau 40-60h",
    finding_ids: list[str] | None = None,
    narrative_ids: list[str] | None = None,
    affected_variables: list[str] | None = None,
    severity: Severity = Severity.MAJOR,
    priority: float = 0.8,
    source_type: TopicSourceType = TopicSourceType.FAILURE,
) -> SeedTopic:
    return SeedTopic(
        topic_id=topic_id,
        summary=summary,
        source_type=source_type,
        source_id="D-F-0001",
        cited_finding_ids=finding_ids or [f"{CHAR_ID}:F-0001"],
        cited_narrative_ids=narrative_ids or [],
        cited_trajectories=[],
        affected_variables=affected_variables or ["biomass_g_l"],
        severity=severity,
        priority=priority,
    )


def make_facet(
    facet_id: str = "FCT-0001",
    role: str = "kinetics",
    summary: str = "biomass curve plateaus despite available glucose",
    finding_ids: list[str] | None = None,
) -> FacetFull:
    return FacetFull(
        facet_id=facet_id,
        specialist=role,  # type: ignore[arg-type]
        summary=summary,
        cited_finding_ids=finding_ids or [f"{CHAR_ID}:F-0001"],
        cited_narrative_ids=[],
        cited_trajectories=[],
        affected_variables=["biomass_g_l"],
        confidence=0.7,
        confidence_basis=ConfidenceBasis.SCHEMA_ONLY,
    )


def make_hypothesis(
    hyp_id: str = "H-0001",
    summary: str = "growth-limited plateau likely substrate-related",
) -> HypothesisFull:
    return HypothesisFull(
        hyp_id=hyp_id,
        summary=summary,
        facet_ids=["FCT-0001"],
        cited_finding_ids=[f"{CHAR_ID}:F-0001"],
        cited_narrative_ids=[],
        cited_trajectories=[],
        affected_variables=["biomass_g_l"],
        confidence=0.7,
        confidence_basis=ConfidenceBasis.SCHEMA_ONLY,
    )


def make_simple_script(
    *,
    topic_ids: list[str] | None = None,
    critic_flag: str = "green",
    judge_valid: bool = False,
    facet_input_tokens: int = 800,
) -> StubScript:
    """One topic plan per topic_id, all green-flagged by default."""
    topic_ids = topic_ids or ["T-0001"]
    plans: dict[str, StubTopicPlan] = {}
    for tid in topic_ids:
        plans[tid] = StubTopicPlan(
            facets={
                "kinetics": StubFacetPlan(
                    summary=f"{tid}: kinetic angle",
                    cited_finding_ids=[f"{CHAR_ID}:F-0001"],
                    affected_variables=["biomass_g_l"],
                    input_tokens=facet_input_tokens,
                ),
                "mass_transfer": StubFacetPlan(
                    summary=f"{tid}: transport angle",
                    cited_finding_ids=[f"{CHAR_ID}:F-0001"],
                    affected_variables=["DO"],
                    input_tokens=facet_input_tokens,
                ),
                "metabolic": StubFacetPlan(
                    summary=f"{tid}: metabolic angle",
                    cited_finding_ids=[f"{CHAR_ID}:F-0001"],
                    affected_variables=["acetate"],
                    input_tokens=facet_input_tokens,
                ),
            },
            synthesis_summary=f"{tid}: synthesized hypothesis combining all 3 angles",
            critic_flag=critic_flag,  # type: ignore[arg-type]
            critic_reasons=[] if critic_flag == "green" else ["citation weak"],
            judge_criticism_valid=judge_valid,
            judge_rationale="stub rationale",
        )
    return StubScript(topic_plans=plans, topic_order=topic_ids)


def make_input(seed_topics: list[SeedTopic] | None = None) -> HypothesisInput:
    return HypothesisInput(
        diagnosis=None,
        characterization=None,
        bundle_path=None,
        seed_topics=seed_topics or [make_seed_topic()],
    )


def make_budget(**kwargs) -> BudgetSnapshot:
    return BudgetSnapshot(**kwargs)
