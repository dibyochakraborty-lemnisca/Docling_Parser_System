"""Topic ranker — determinism, penalties, tie detection, synthetic topics."""

from __future__ import annotations

from datetime import datetime, timezone

from fermdocs_characterize.schema import Severity
from fermdocs_hypothesis.events import (
    HypothesisRejectedEvent,
    QuestionAddedEvent,
    TopicSelectedEvent,
)
from fermdocs_hypothesis.ranker import is_tie, rank_topics
from tests.unit.hypothesis.fixtures import make_seed_topic

NOW = datetime(2026, 5, 3, 12, 0, 0, tzinfo=timezone.utc)


def test_rank_returns_at_most_k():
    seeds = [make_seed_topic(topic_id=f"T-000{i}", summary=f"t{i}") for i in range(1, 6)]
    ranked = rank_topics(seeds, [], k=3)
    assert len(ranked) == 3


def test_rank_deterministic_same_input_same_output():
    seeds = [make_seed_topic(topic_id=f"T-000{i}", summary=f"t{i}") for i in range(1, 5)]
    a = rank_topics(seeds, [], k=3)
    b = rank_topics(seeds, [], k=3)
    assert a == b


def test_higher_severity_outranks_lower_at_same_priority():
    major = make_seed_topic(topic_id="T-0001", severity=Severity.MAJOR, priority=0.5)
    minor = make_seed_topic(topic_id="T-0002", severity=Severity.MINOR, priority=0.5)
    ranked = rank_topics([minor, major], [])
    assert ranked[0].topic_id == "T-0001"


def test_attempt_penalty_demotes_repeated_topics():
    """A topic selected twice should drop below an unattempted equal-priority topic."""
    a = make_seed_topic(topic_id="T-0001", priority=0.8)
    b = make_seed_topic(topic_id="T-0002", priority=0.8)
    events = [
        TopicSelectedEvent(ts=NOW, turn=1, topic_id="T-0001", summary="a", rationale="r"),
        TopicSelectedEvent(ts=NOW, turn=2, topic_id="T-0001", summary="a", rationale="r"),
    ]
    ranked = rank_topics([a, b], events)
    assert ranked[0].topic_id == "T-0002"


def test_rejection_penalty_demotes_judged_bad_topics():
    a = make_seed_topic(topic_id="T-0001", priority=0.9)
    b = make_seed_topic(topic_id="T-0002", priority=0.5)
    events = [
        HypothesisRejectedEvent(ts=NOW, turn=1, hyp_id="H-0001", topic_id="T-0001", reason="r"),
    ]
    # T-0001 starts higher but a -1.0 rejection penalty should flip it.
    ranked = rank_topics([a, b], events)
    assert ranked[0].topic_id == "T-0002"


def test_synthetic_open_question_topic_appears_in_ranking():
    seed = make_seed_topic(topic_id="T-0001", priority=0.1, severity=Severity.INFO)
    events = [
        QuestionAddedEvent(ts=NOW, turn=1, qid="Q-0001", question="x?", raised_by="kinetics", tags=["DO"]),
    ]
    ranked = rank_topics([seed], events, k=5)
    synthetic_ids = [r.topic_id for r in ranked if r.is_synthetic]
    assert synthetic_ids  # at least one synthetic topic is ranked


def test_is_tie_true_for_close_scores():
    seed_a = make_seed_topic(topic_id="T-0001", priority=0.5)
    seed_b = make_seed_topic(topic_id="T-0002", priority=0.5)
    ranked = rank_topics([seed_a, seed_b], [])
    assert is_tie(ranked)


def test_is_tie_false_for_distant_scores():
    seed_a = make_seed_topic(topic_id="T-0001", priority=0.9, severity=Severity.MAJOR)
    seed_b = make_seed_topic(topic_id="T-0002", priority=0.1, severity=Severity.INFO)
    ranked = rank_topics([seed_a, seed_b], [])
    assert not is_tie(ranked)


def test_topics_sort_tiebreak_by_topic_id_asc():
    """When scores are exactly equal, lower topic_id wins."""
    seed_a = make_seed_topic(topic_id="T-0001", priority=0.5)
    seed_b = make_seed_topic(topic_id="T-0002", priority=0.5)
    ranked = rank_topics([seed_b, seed_a], [])
    # Both tied → T-0001 first
    assert ranked[0].topic_id == "T-0001"
