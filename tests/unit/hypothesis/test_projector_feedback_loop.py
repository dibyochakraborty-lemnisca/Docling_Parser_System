"""Projector tests for feedback-loop fields on synthesizer/critic/judge views.

Asserts:
  - previous_attempts is empty when events=None (back-compat for old call sites)
  - previous_attempts populated when events are passed and prior critiqued
    attempts exist on the same topic
  - cross_topic_lessons populated from latest LessonsSummarizedEvent
  - critic/judge view scoping (current hypothesis filtered out of previous_attempts)
"""

from __future__ import annotations

from datetime import datetime, timezone

from fermdocs_diagnose.schema import ConfidenceBasis
from fermdocs_hypothesis.events import (
    CritiqueFiledEvent,
    HypothesisRejectedEvent,
    HypothesisSynthesizedEvent,
    JudgeRulingEvent,
    LessonsSummarizedEvent,
    TopicSelectedEvent,
)
from fermdocs_hypothesis.projector import (
    project_critic,
    project_judge,
    project_synthesizer,
)
from fermdocs_hypothesis.schema import (
    CritiqueFull,
    FacetFull,
    HypothesisFull,
    TopicSourceType,
    TopicSpec,
)

NOW = datetime(2026, 5, 3, 12, 0, 0, tzinfo=timezone.utc)


def _topic_spec(topic_id="T-0001"):
    return TopicSpec(
        topic_id=topic_id,
        summary="biomass plateau",
        source_type=TopicSourceType.FAILURE,
        cited_finding_ids=["F-0001"],
        affected_variables=["biomass_g_l"],
    )


def _facet():
    return FacetFull(
        facet_id="FCT-0001",
        specialist="kinetics",
        summary="growth phase analysis",
        cited_finding_ids=["F-0001"],
        confidence=0.6,
        confidence_basis=ConfidenceBasis.SCHEMA_ONLY,
    )


def _hyp(hyp_id="H-0099"):
    return HypothesisFull(
        hyp_id=hyp_id,
        summary="hypothesis text",
        facet_ids=["FCT-0001"],
        cited_finding_ids=["F-0001"],
        confidence=0.6,
        confidence_basis=ConfidenceBasis.SCHEMA_ONLY,
    )


def _critique(hyp_id, flag="red"):
    return CritiqueFull(hyp_id=hyp_id, flag=flag, reasons=["r"])


def _events_with_one_rejected_attempt(topic_id="T-0001"):
    """Events: topic selected, one synth, critic, judge, reject."""
    return [
        TopicSelectedEvent(
            ts=NOW, turn=1, topic_id=topic_id, summary="biomass plateau", rationale="r"
        ),
        HypothesisSynthesizedEvent(
            ts=NOW,
            turn=1,
            hyp_id="H-0001",
            topic_id=topic_id,
            summary="prior attempt",
            facet_ids=["FCT-0001"],
            cited_finding_ids=["F-0001"],
            confidence=0.6,
            confidence_basis=ConfidenceBasis.SCHEMA_ONLY,
        ),
        CritiqueFiledEvent(
            ts=NOW,
            turn=1,
            hyp_id="H-0001",
            flag="red",
            reasons=["documented absence ≠ ruled out"],
        ),
        JudgeRulingEvent(
            ts=NOW, turn=1, hyp_id="H-0001", criticism_valid=True, rationale="upheld"
        ),
        HypothesisRejectedEvent(
            ts=NOW, turn=1, hyp_id="H-0001", topic_id=topic_id, reason="upheld"
        ),
    ]


# ---------- synthesizer ----------


def test_project_synthesizer_no_events_back_compat():
    """Old call sites (events=None) keep working with empty history."""
    view = project_synthesizer(current_topic=_topic_spec(), facets=[_facet()])
    assert view.previous_attempts == []
    assert view.cross_topic_lessons is None


def test_project_synthesizer_populates_previous_attempts_from_events():
    events = _events_with_one_rejected_attempt()
    view = project_synthesizer(
        current_topic=_topic_spec(),
        facets=[_facet()],
        events=events,
    )
    assert len(view.previous_attempts) == 1
    a = view.previous_attempts[0]
    assert a.hyp_id == "H-0001"
    assert a.critic_reasons == ["documented absence ≠ ruled out"]
    assert a.critic_flag == "red"


def test_project_synthesizer_filters_uncritiqued_attempts():
    """Mid-cycle synth without critic shouldn't surface as a 'prior attempt'."""
    events = [
        TopicSelectedEvent(
            ts=NOW, turn=1, topic_id="T-0001", summary="s", rationale="r"
        ),
        HypothesisSynthesizedEvent(
            ts=NOW,
            turn=1,
            hyp_id="H-0001",
            topic_id="T-0001",
            summary="mid cycle",
            facet_ids=["FCT-0001"],
            cited_finding_ids=["F-0001"],
            confidence=0.6,
            confidence_basis=ConfidenceBasis.SCHEMA_ONLY,
        ),
        # no critique yet
    ]
    view = project_synthesizer(
        current_topic=_topic_spec(), facets=[_facet()], events=events
    )
    assert view.previous_attempts == []


def test_project_synthesizer_cross_topic_lessons_picked_up():
    events = _events_with_one_rejected_attempt() + [
        LessonsSummarizedEvent(
            ts=NOW, turn=2, digest="lesson body", source_reason_count=1
        )
    ]
    view = project_synthesizer(
        current_topic=_topic_spec(), facets=[_facet()], events=events
    )
    assert view.cross_topic_lessons is not None
    assert view.cross_topic_lessons.digest == "lesson body"
    assert view.cross_topic_lessons.source_reason_count == 1


# ---------- critic ----------


def test_project_critic_no_events_back_compat():
    view = project_critic(hypothesis=_hyp())
    assert view.previous_attempts == []
    assert view.cross_topic_lessons is None


def test_project_critic_excludes_current_hypothesis_from_history():
    """The hypothesis the critic is about to review must not appear in
    previous_attempts (or the critic would see itself)."""
    events = _events_with_one_rejected_attempt()
    # Pretend the current hypothesis IS H-0001 (re-reviewing it)
    view = project_critic(
        hypothesis=_hyp("H-0001"), events=events, topic_id="T-0001"
    )
    # H-0001 filtered out → empty
    assert view.previous_attempts == []


def test_project_critic_includes_other_attempts_on_same_topic():
    events = _events_with_one_rejected_attempt()
    view = project_critic(
        hypothesis=_hyp("H-0099"),  # different hyp_id from prior attempts
        events=events,
        topic_id="T-0001",
    )
    assert [a.hyp_id for a in view.previous_attempts] == ["H-0001"]


# ---------- judge ----------


def test_project_judge_back_compat_no_events():
    view = project_judge(hypothesis=_hyp(), critique=_critique("H-0099"))
    assert view.previous_attempts == []
    assert view.cross_topic_lessons is None


def test_project_judge_scoped_to_topic():
    events = _events_with_one_rejected_attempt()
    view = project_judge(
        hypothesis=_hyp("H-0099"),
        critique=_critique("H-0099"),
        events=events,
        topic_id="T-0001",
    )
    assert [a.hyp_id for a in view.previous_attempts] == ["H-0001"]
