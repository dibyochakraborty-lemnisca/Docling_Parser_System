"""All 5 view types carry user_question through projection.

PR-A on caisc-hitl, commit 3. Plan ref:
plans/2026-05-04-user-question-and-hitl.md.

Back-compat invariant: when user_question kwarg is omitted, projection
output is unchanged from today (every existing test still passes,
covered by full-suite assertion in commit message).
"""

from __future__ import annotations

from datetime import datetime, timezone

from fermdocs.domain.user_question import UserQuestion
from fermdocs_hypothesis.projector import (
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
    FollowupContext,
    HypothesisFull,
    PriorHypothesisRef,
)
from fermdocs_hypothesis.stubs.canned_agents import topic_spec_from_seed
from tests.unit.hypothesis.fixtures import (
    make_facet,
    make_hypothesis,
    make_seed_topic,
)

NOW = datetime(2026, 5, 5, 12, 0, 0, tzinfo=timezone.utc)


def _q() -> UserQuestion:
    return UserQuestion(
        text="Why did RUN-0002 plateau early?",
        shape="scoping",
        affected_runs=["RUN-0002"],
        affected_variables=["biomass_g_l"],
    )


def _followup_context() -> FollowupContext:
    return FollowupContext(
        original_final_hypotheses=[
            PriorHypothesisRef(
                hyp_id="H-0001",
                summary="Prior answer: oxygen limitation was most likely.",
                question_answered="partial",
                question_response_summary="The prior run found partial evidence.",
                affected_variables=["dissolved_oxygen_pct"],
                confidence=0.72,
            )
        ]
    )


# ---------- orchestrator ----------


def test_orchestrator_view_carries_user_question_when_provided() -> None:
    seeds = [make_seed_topic(topic_id="T-0001")]
    view = project_orchestrator(
        events=[],
        seed_topics=seeds,
        budget=BudgetSnapshot(),
        current_turn=0,
        user_question=_q(),
    )
    assert view.user_question is not None
    assert view.user_question.text == "Why did RUN-0002 plateau early?"
    assert view.user_question.shape == "scoping"


def test_orchestrator_view_user_question_defaults_none() -> None:
    seeds = [make_seed_topic(topic_id="T-0001")]
    view = project_orchestrator(
        events=[],
        seed_topics=seeds,
        budget=BudgetSnapshot(),
        current_turn=0,
    )
    assert view.user_question is None


def test_orchestrator_view_carries_followup_context_when_provided() -> None:
    seeds = [make_seed_topic(topic_id="T-0001")]
    view = project_orchestrator(
        events=[],
        seed_topics=seeds,
        budget=BudgetSnapshot(),
        current_turn=0,
        followup_context=_followup_context(),
    )
    assert view.followup_context is not None
    assert view.followup_context.original_final_hypotheses[0].hyp_id == "H-0001"


# ---------- specialist ----------


def test_specialist_view_carries_user_question() -> None:
    seed = make_seed_topic(affected_variables=["biomass_g_l"])
    topic = topic_spec_from_seed(seed)
    view = project_specialist(
        events=[],
        role="kinetics",
        current_topic=topic,
        available_findings=[
            FindingRef(finding_id="F-1", summary="?", variables_involved=["biomass_g_l"])
        ],
        available_narratives=[],
        available_trajectories=[],
        available_priors=[],
        user_question=_q(),
    )
    assert view.user_question is not None
    assert view.user_question.shape == "scoping"


def test_specialist_view_user_question_defaults_none() -> None:
    seed = make_seed_topic(affected_variables=["biomass_g_l"])
    topic = topic_spec_from_seed(seed)
    view = project_specialist(
        events=[],
        role="kinetics",
        current_topic=topic,
        available_findings=[],
        available_narratives=[],
        available_trajectories=[],
        available_priors=[],
    )
    assert view.user_question is None


def test_specialist_view_carries_followup_context() -> None:
    seed = make_seed_topic(affected_variables=["biomass_g_l"])
    topic = topic_spec_from_seed(seed)
    view = project_specialist(
        events=[],
        role="kinetics",
        current_topic=topic,
        available_findings=[],
        available_narratives=[],
        available_trajectories=[],
        available_priors=[],
        followup_context=_followup_context(),
    )
    assert view.followup_context is not None


# ---------- synthesizer ----------


def test_synthesizer_view_carries_user_question() -> None:
    seed = make_seed_topic()
    topic = topic_spec_from_seed(seed)
    facets = [make_facet(facet_id="FCT-0001", role="kinetics")]
    view = project_synthesizer(
        current_topic=topic,
        facets=facets,
        events=[],
        user_question=_q(),
    )
    assert view.user_question is not None
    assert view.user_question.affected_runs == ["RUN-0002"]


def test_synthesizer_view_user_question_defaults_none() -> None:
    seed = make_seed_topic()
    topic = topic_spec_from_seed(seed)
    facets = [make_facet(facet_id="FCT-0001", role="kinetics")]
    view = project_synthesizer(current_topic=topic, facets=facets)
    assert view.user_question is None


def test_synthesizer_view_carries_followup_context() -> None:
    seed = make_seed_topic()
    topic = topic_spec_from_seed(seed)
    facets = [make_facet(facet_id="FCT-0001", role="kinetics")]
    view = project_synthesizer(
        current_topic=topic,
        facets=facets,
        followup_context=_followup_context(),
    )
    assert view.followup_context is not None
    assert "oxygen limitation" in view.followup_context.original_final_hypotheses[0].summary


# ---------- critic ----------


def test_critic_view_carries_user_question() -> None:
    hyp = make_hypothesis(hyp_id="H-0001")
    view = project_critic(
        hypothesis=hyp,
        user_question=_q(),
    )
    assert view.user_question is not None


def test_critic_view_user_question_defaults_none() -> None:
    hyp = make_hypothesis(hyp_id="H-0001")
    view = project_critic(hypothesis=hyp)
    assert view.user_question is None


def test_critic_view_carries_followup_context() -> None:
    hyp = make_hypothesis(hyp_id="H-0001")
    view = project_critic(
        hypothesis=hyp,
        followup_context=_followup_context(),
    )
    assert view.followup_context is not None


# ---------- judge ----------


def test_judge_view_carries_user_question() -> None:
    hyp = make_hypothesis(hyp_id="H-0001")
    critique = CritiqueFull(
        critique_id="CRIT-0001",
        hyp_id=hyp.hyp_id,
        flag="green",
        reasons=[],
        cited_finding_ids=[],
        cited_narrative_ids=[],
        cited_trajectories=[],
        cited_priors=[],
    )
    view = project_judge(
        hypothesis=hyp,
        critique=critique,
        user_question=_q(),
    )
    assert view.user_question is not None
    assert view.user_question.shape == "scoping"


def test_judge_view_user_question_defaults_none() -> None:
    hyp = make_hypothesis(hyp_id="H-0001")
    critique = CritiqueFull(
        critique_id="CRIT-0001",
        hyp_id=hyp.hyp_id,
        flag="green",
        reasons=[],
        cited_finding_ids=[],
        cited_narrative_ids=[],
        cited_trajectories=[],
        cited_priors=[],
    )
    view = project_judge(hypothesis=hyp, critique=critique)
    assert view.user_question is None


def test_judge_view_carries_followup_context() -> None:
    hyp = make_hypothesis(hyp_id="H-0001")
    critique = CritiqueFull(
        critique_id="CRIT-0001",
        hyp_id=hyp.hyp_id,
        flag="green",
        reasons=[],
        cited_finding_ids=[],
        cited_narrative_ids=[],
        cited_trajectories=[],
        cited_priors=[],
    )
    view = project_judge(
        hypothesis=hyp,
        critique=critique,
        followup_context=_followup_context(),
    )
    assert view.followup_context is not None
