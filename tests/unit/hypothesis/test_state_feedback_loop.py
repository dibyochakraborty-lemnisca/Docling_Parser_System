"""Tests for the feedback-loop state projections: topic_history,
all_critic_reasons, latest_lessons_digest.

These projections are the substrate for previous_attempts and
cross_topic_lessons in agent views. They must be deterministic given an
event list (same events → same projection) so the runner's caching
behavior is reproducible across test runs.
"""

from __future__ import annotations

from datetime import datetime, timezone

from fermdocs_diagnose.schema import ConfidenceBasis
from fermdocs_hypothesis.events import (
    CritiqueFiledEvent,
    HypothesisAcceptedEvent,
    HypothesisRejectedEvent,
    HypothesisSynthesizedEvent,
    JudgeRulingEvent,
    LessonsSummarizedEvent,
    TopicSelectedEvent,
)
from fermdocs_hypothesis.state import (
    all_critic_reasons,
    latest_lessons_digest,
    topic_history,
)

NOW = datetime(2026, 5, 3, 12, 0, 0, tzinfo=timezone.utc)


def _topic(topic_id="T-0001", summary="biomass plateau"):
    return TopicSelectedEvent(
        ts=NOW, turn=1, topic_id=topic_id, summary=summary, rationale="r"
    )


def _synth(hyp_id, topic_id="T-0001", summary="hyp summary"):
    return HypothesisSynthesizedEvent(
        ts=NOW,
        turn=1,
        hyp_id=hyp_id,
        topic_id=topic_id,
        summary=summary,
        facet_ids=["FCT-0001"],
        cited_finding_ids=["F-0001"],
        confidence=0.6,
        confidence_basis=ConfidenceBasis.SCHEMA_ONLY,
    )


def _critique(hyp_id, flag="red", reasons=None):
    return CritiqueFiledEvent(
        ts=NOW,
        turn=1,
        hyp_id=hyp_id,
        flag=flag,
        reasons=reasons or ["overreach: documented absence treated as ruled out"],
    )


def _judge(hyp_id, valid=True, rationale="critique stands"):
    return JudgeRulingEvent(
        ts=NOW, turn=1, hyp_id=hyp_id, criticism_valid=valid, rationale=rationale
    )


def _reject(hyp_id, topic_id="T-0001"):
    return HypothesisRejectedEvent(
        ts=NOW, turn=1, hyp_id=hyp_id, topic_id=topic_id, reason="judge upheld"
    )


# ---------- topic_history ----------


def test_topic_history_empty_events_returns_deferred():
    h = topic_history([], "T-0001")
    assert h.topic_id == "T-0001"
    assert h.attempts == []
    assert h.status == "deferred"


def test_topic_history_topic_selected_no_attempts_is_deferred():
    h = topic_history([_topic("T-0001", "biomass plateau")], "T-0001")
    assert h.summary == "biomass plateau"
    assert h.attempts == []
    assert h.status == "deferred"


def test_topic_history_one_rejected_attempt_in_progress():
    events = [
        _topic("T-0001"),
        _synth("H-0001"),
        _critique("H-0001", flag="red", reasons=["r1"]),
        _judge("H-0001", valid=True),
        _reject("H-0001"),
    ]
    h = topic_history(events, "T-0001")
    assert len(h.attempts) == 1
    a = h.attempts[0]
    assert a.hyp_id == "H-0001"
    assert a.critic_flag == "red"
    assert a.critic_reasons == ["r1"]
    assert a.judge_ruling == "valid"
    assert a.judge_rationale == "critique stands"
    assert h.status == "in_progress"


def test_topic_history_multiple_rejected_attempts_oldest_first():
    events = [
        _topic("T-0001"),
        _synth("H-0001"),
        _critique("H-0001", reasons=["r1"]),
        _judge("H-0001"),
        _reject("H-0001"),
        _synth("H-0002"),
        _critique("H-0002", reasons=["r2"]),
        _judge("H-0002"),
        _reject("H-0002"),
        _synth("H-0003"),
        _critique("H-0003", reasons=["r3"]),
        _judge("H-0003"),
        _reject("H-0003"),
    ]
    h = topic_history(events, "T-0001")
    assert [a.hyp_id for a in h.attempts] == ["H-0001", "H-0002", "H-0003"]
    assert [a.critic_reasons[0] for a in h.attempts] == ["r1", "r2", "r3"]
    assert h.status == "in_progress"


def test_topic_history_accepted_status():
    events = [
        _topic("T-0001"),
        _synth("H-0001"),
        _critique("H-0001", flag="green", reasons=[]),
        _judge("H-0001", valid=False, rationale="green flag, nothing to validate"),
        HypothesisAcceptedEvent(ts=NOW, turn=1, hyp_id="H-0001", topic_id="T-0001"),
    ]
    h = topic_history(events, "T-0001")
    assert h.status == "accepted"
    assert len(h.attempts) == 1
    assert h.attempts[0].critic_flag == "green"


def test_topic_history_filters_by_topic_id():
    """Attempts on T-0002 must not appear in T-0001's history."""
    events = [
        _topic("T-0001"),
        _synth("H-0001", topic_id="T-0001"),
        _critique("H-0001"),
        _judge("H-0001"),
        _reject("H-0001", topic_id="T-0001"),
        _topic("T-0002"),
        _synth("H-0002", topic_id="T-0002"),
        _critique("H-0002"),
        _judge("H-0002"),
        _reject("H-0002", topic_id="T-0002"),
    ]
    h1 = topic_history(events, "T-0001")
    h2 = topic_history(events, "T-0002")
    assert [a.hyp_id for a in h1.attempts] == ["H-0001"]
    assert [a.hyp_id for a in h2.attempts] == ["H-0002"]


def test_topic_history_synthesized_only_no_critique_yet():
    """Mid-cycle: synth happened but critic hasn't filed. Attempt exists
    with critic_flag=None so the projector can filter it out."""
    events = [_topic("T-0001"), _synth("H-0001")]
    h = topic_history(events, "T-0001")
    assert len(h.attempts) == 1
    assert h.attempts[0].critic_flag is None
    assert h.attempts[0].critic_reasons == []


# ---------- all_critic_reasons ----------


def test_all_critic_reasons_empty():
    assert all_critic_reasons([]) == []


def test_all_critic_reasons_collects_across_topics_in_order():
    events = [
        _critique("H-0001", reasons=["a", "b"]),
        _critique("H-0002", reasons=["c"]),
        _critique("H-0003", reasons=["d", "e"]),
    ]
    assert all_critic_reasons(events) == ["a", "b", "c", "d", "e"]


def test_all_critic_reasons_caps_at_limit_keeps_newest():
    events = [_critique(f"H-{i:04d}", reasons=[f"r{i}"]) for i in range(1, 26)]
    out = all_critic_reasons(events, limit=10)
    assert len(out) == 10
    # Newest 10 → r16..r25
    assert out == [f"r{i}" for i in range(16, 26)]


# ---------- latest_lessons_digest ----------


def test_latest_lessons_digest_none_when_no_event():
    assert latest_lessons_digest([]) is None
    assert latest_lessons_digest([_critique("H-0001")]) is None


def test_latest_lessons_digest_returns_most_recent():
    events = [
        LessonsSummarizedEvent(
            ts=NOW, turn=2, digest="lesson v1", source_reason_count=3
        ),
        _critique("H-0002"),
        LessonsSummarizedEvent(
            ts=NOW, turn=4, digest="lesson v2 newer", source_reason_count=8
        ),
    ]
    d = latest_lessons_digest(events)
    assert d is not None
    assert d.digest == "lesson v2 newer"
    assert d.source_reason_count == 8
    # event index of the newer lesson event in the list above
    assert d.computed_at_event_idx == 2
