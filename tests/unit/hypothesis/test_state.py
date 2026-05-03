"""Derived state from event lists — purity, ledger, tallies."""

from __future__ import annotations

from datetime import datetime, timezone

from fermdocs_diagnose.schema import ConfidenceBasis
from fermdocs_hypothesis.events import (
    FacetContributedEvent,
    HypothesisAcceptedEvent,
    HypothesisRejectedEvent,
    HypothesisSynthesizedEvent,
    QuestionAddedEvent,
    QuestionResolvedEvent,
    TopicSelectedEvent,
)
from fermdocs_hypothesis.state import (
    accepted_hypotheses,
    critic_cycles_for_current_topic,
    facets_for_current_topic,
    last_turn_outcome,
    open_questions,
    specialist_domain_tags,
    topic_attempt_counts,
    topic_rejection_counts,
)

NOW = datetime(2026, 5, 3, 12, 0, 0, tzinfo=timezone.utc)


def _q_added(qid, q="why?", tags=None):
    return QuestionAddedEvent(ts=NOW, turn=1, qid=qid, question=q, raised_by="kinetics", tags=tags or [])


def _q_resolved(qid, res="found it"):
    return QuestionResolvedEvent(ts=NOW, turn=2, qid=qid, resolution=res)


def test_open_questions_unresolved_by_default():
    qs = open_questions([_q_added("Q-0001"), _q_added("Q-0002")])
    assert all(not q.resolved for q in qs)
    assert [q.qid for q in qs] == ["Q-0001", "Q-0002"]


def test_open_questions_resolution_marks_resolved():
    qs = open_questions([_q_added("Q-0001"), _q_resolved("Q-0001", "answered")])
    assert qs[0].resolved is True
    assert qs[0].resolution == "answered"


def test_open_questions_resolve_unknown_qid_is_noop():
    qs = open_questions([_q_resolved("Q-0099")])
    assert qs == []


def test_topic_attempt_counts_dedups_per_topic():
    events = [
        TopicSelectedEvent(ts=NOW, turn=1, topic_id="T-0001", summary="a", rationale="r"),
        TopicSelectedEvent(ts=NOW, turn=2, topic_id="T-0002", summary="b", rationale="r"),
        TopicSelectedEvent(ts=NOW, turn=3, topic_id="T-0001", summary="a", rationale="r"),
    ]
    c = topic_attempt_counts(events)
    assert c["T-0001"] == 2
    assert c["T-0002"] == 1


def test_topic_rejection_counts():
    events = [
        HypothesisRejectedEvent(ts=NOW, turn=1, hyp_id="H-0001", topic_id="T-0001", reason="bad"),
        HypothesisRejectedEvent(ts=NOW, turn=2, hyp_id="H-0002", topic_id="T-0001", reason="bad"),
        HypothesisRejectedEvent(ts=NOW, turn=3, hyp_id="H-0003", topic_id="T-0002", reason="bad"),
    ]
    c = topic_rejection_counts(events)
    assert c["T-0001"] == 2
    assert c["T-0002"] == 1


def test_critic_cycles_counts_only_after_last_select():
    events = [
        TopicSelectedEvent(ts=NOW, turn=1, topic_id="T-0001", summary="a", rationale="r"),
        HypothesisRejectedEvent(ts=NOW, turn=1, hyp_id="H-0001", topic_id="T-0001", reason="r"),
        TopicSelectedEvent(ts=NOW, turn=2, topic_id="T-0001", summary="a", rationale="r"),
        HypothesisRejectedEvent(ts=NOW, turn=2, hyp_id="H-0002", topic_id="T-0001", reason="r"),
    ]
    # Only the rejection after the most-recent select counts
    assert critic_cycles_for_current_topic(events, "T-0001") == 1


def test_critic_cycles_zero_when_topic_never_selected():
    assert critic_cycles_for_current_topic([], "T-0001") == 0


def test_facets_for_current_topic_filters_to_latest_select():
    events = [
        TopicSelectedEvent(ts=NOW, turn=1, topic_id="T-0001", summary="a", rationale="r"),
        FacetContributedEvent(
            ts=NOW, turn=1, facet_id="FCT-0001", topic_id="T-0001", specialist="kinetics",
            summary="x", confidence=0.6, confidence_basis=ConfidenceBasis.SCHEMA_ONLY,
        ),
        TopicSelectedEvent(ts=NOW, turn=2, topic_id="T-0002", summary="b", rationale="r"),
        FacetContributedEvent(
            ts=NOW, turn=2, facet_id="FCT-0002", topic_id="T-0002", specialist="kinetics",
            summary="y", confidence=0.6, confidence_basis=ConfidenceBasis.SCHEMA_ONLY,
        ),
    ]
    facets = facets_for_current_topic(events, "T-0002")
    assert [f.facet_id for f in facets] == ["FCT-0002"]


def test_accepted_hypotheses_returns_refs_in_order():
    events = [
        HypothesisSynthesizedEvent(
            ts=NOW, turn=1, hyp_id="H-0001", topic_id="T-0001", summary="alpha",
            facet_ids=["FCT-0001"], confidence=0.7, confidence_basis=ConfidenceBasis.SCHEMA_ONLY,
        ),
        HypothesisAcceptedEvent(ts=NOW, turn=1, hyp_id="H-0001", topic_id="T-0001"),
        HypothesisSynthesizedEvent(
            ts=NOW, turn=2, hyp_id="H-0002", topic_id="T-0002", summary="beta",
            facet_ids=["FCT-0002"], confidence=0.7, confidence_basis=ConfidenceBasis.SCHEMA_ONLY,
        ),
        HypothesisAcceptedEvent(ts=NOW, turn=2, hyp_id="H-0002", topic_id="T-0002"),
    ]
    refs = accepted_hypotheses(events)
    assert [r.hyp_id for r in refs] == ["H-0001", "H-0002"]
    assert refs[0].summary == "alpha"


def test_last_turn_outcome_picks_most_recent():
    events = [
        HypothesisAcceptedEvent(ts=NOW, turn=1, hyp_id="H-0001", topic_id="T-0001"),
        HypothesisRejectedEvent(ts=NOW, turn=2, hyp_id="H-0002", topic_id="T-0002", reason="r"),
    ]
    outcome = last_turn_outcome(events)
    assert outcome.outcome == "rejected"
    assert outcome.turn == 2


def test_specialist_domain_tags_are_disjoint_enough():
    k = specialist_domain_tags("kinetics")
    m = specialist_domain_tags("metabolic")
    # Kinetics and metabolic share substrate-y concepts but should not both
    # claim "DO" — that's mass_transfer's territory.
    mt = specialist_domain_tags("mass_transfer")
    assert "DO" in mt and "DO" not in k and "DO" not in m


def test_state_functions_pure():
    """Calling open_questions twice on same input gives identical results."""
    events = [_q_added("Q-0001"), _q_added("Q-0002"), _q_resolved("Q-0001")]
    assert open_questions(events) == open_questions(events)
