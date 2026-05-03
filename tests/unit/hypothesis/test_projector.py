"""Projector — view shapes, role-domain filtering, hard caps."""

from __future__ import annotations

from datetime import datetime, timezone

from fermdocs_hypothesis.events import TopicSelectedEvent
from fermdocs_hypothesis.projector import (
    VIEW_CAPS,
    project_critic,
    project_judge,
    project_orchestrator,
    project_specialist,
    project_synthesizer,
)
from fermdocs_hypothesis.schema import (
    BudgetSnapshot,
    CritiqueFull,
    FindingRef,
    NarrativeRef,
    OpenQuestionRef,
    ResolvedPriorRef,
    TopicSpec,
    TrajectoryViewRef,
)
from fermdocs_hypothesis.stubs.canned_agents import topic_spec_from_seed
from tests.unit.hypothesis.fixtures import (
    make_facet,
    make_hypothesis,
    make_seed_topic,
)

NOW = datetime(2026, 5, 3, 12, 0, 0, tzinfo=timezone.utc)


def test_orchestrator_view_includes_top_topics():
    seeds = [make_seed_topic(topic_id=f"T-000{i}", summary=f"t{i}") for i in range(1, 5)]
    view = project_orchestrator(
        events=[],
        seed_topics=seeds,
        budget=BudgetSnapshot(),
        current_turn=0,
    )
    assert len(view.top_topics) <= VIEW_CAPS["top_topics_k"]


def test_specialist_view_filters_findings_to_domain():
    seed = make_seed_topic(affected_variables=["DO"])
    topic = topic_spec_from_seed(seed)
    findings = [
        FindingRef(finding_id="F-1", summary="DO crash", variables_involved=["DO"]),
        FindingRef(finding_id="F-2", summary="acetate spike", variables_involved=["acetate"]),
    ]
    view = project_specialist(
        events=[],
        role="mass_transfer",
        current_topic=topic,
        available_findings=findings,
        available_narratives=[],
        available_trajectories=[],
        available_priors=[],
    )
    ids = {f.finding_id for f in view.relevant_findings}
    # mass_transfer's domain includes DO; should keep F-1
    assert "F-1" in ids


def test_specialist_view_caps_findings():
    seed = make_seed_topic(affected_variables=["DO"])
    topic = topic_spec_from_seed(seed)
    many = [FindingRef(finding_id=f"F-{i}", summary="DO", variables_involved=["DO"]) for i in range(50)]
    view = project_specialist(
        events=[],
        role="mass_transfer",
        current_topic=topic,
        available_findings=many,
        available_narratives=[],
        available_trajectories=[],
        available_priors=[],
    )
    assert len(view.relevant_findings) <= VIEW_CAPS["findings_per_specialist"]


def test_specialist_view_excludes_own_facet_from_prior_facets():
    seed = make_seed_topic(topic_id="T-0001")
    topic = topic_spec_from_seed(seed)
    from fermdocs_diagnose.schema import ConfidenceBasis
    from fermdocs_hypothesis.events import FacetContributedEvent

    events = [
        TopicSelectedEvent(ts=NOW, turn=1, topic_id="T-0001", summary="x", rationale="r"),
        FacetContributedEvent(
            ts=NOW, turn=1, facet_id="FCT-0001", topic_id="T-0001", specialist="kinetics",
            summary="kin angle", confidence=0.6, confidence_basis=ConfidenceBasis.SCHEMA_ONLY,
        ),
        FacetContributedEvent(
            ts=NOW, turn=1, facet_id="FCT-0002", topic_id="T-0001", specialist="metabolic",
            summary="met angle", confidence=0.6, confidence_basis=ConfidenceBasis.SCHEMA_ONLY,
        ),
    ]
    view = project_specialist(
        events=events,
        role="kinetics",
        current_topic=topic,
        available_findings=[],
        available_narratives=[],
        available_trajectories=[],
        available_priors=[],
    )
    # Kinetics should see the metabolic facet but not its own.
    roles = {f.specialist for f in view.prior_facets_this_topic}
    assert "kinetics" not in roles
    assert "metabolic" in roles


def test_synthesizer_view_unions_citations():
    seed = make_seed_topic()
    topic = topic_spec_from_seed(seed)
    f1 = make_facet(facet_id="FCT-0001", finding_ids=["F-A"])
    f2 = make_facet(facet_id="FCT-0002", finding_ids=["F-B"])
    view = project_synthesizer(current_topic=topic, facets=[f1, f2])
    assert set(view.citation_universe.finding_ids) == {"F-A", "F-B"}


def test_synthesizer_view_dedups_citations_across_facets():
    seed = make_seed_topic()
    topic = topic_spec_from_seed(seed)
    f1 = make_facet(facet_id="FCT-0001", finding_ids=["F-A"])
    f2 = make_facet(facet_id="FCT-0002", finding_ids=["F-A", "F-B"])
    view = project_synthesizer(current_topic=topic, facets=[f1, f2])
    assert view.citation_universe.finding_ids == ["F-A", "F-B"]


def test_critic_view_carries_hypothesis_and_lookups():
    hyp = make_hypothesis()
    view = project_critic(hypothesis=hyp, citation_lookups={"F-A": {"text": "..."}})
    assert view.hypothesis.hyp_id == "H-0001"
    assert "F-A" in view.citation_lookups


def test_judge_view_has_no_debate_history_field():
    hyp = make_hypothesis()
    crit = CritiqueFull(hyp_id="H-0001", flag="green", reasons=[])
    view = project_judge(hypothesis=hyp, critique=crit)
    # Sanity: judge view structure has only the three approved fields.
    field_names = set(view.model_dump().keys())
    assert field_names == {"hypothesis", "critique", "citation_lookups"}
