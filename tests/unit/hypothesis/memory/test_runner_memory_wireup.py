"""Runner integrates with MemoryBackend on normal exit (D6 revised).

Plan ref: plans/2026-05-10-memory-layer.md commit 3.

REGRESSION-CRITICAL: with NoopBackend (default), behavior is unchanged.
Tests assert:

  1. Any recognised ExitReason with a process_family persists lessons
     via memory.write (the lessons summarizer is the quality gate).
  2. exit_reason=None does NOT write (defensive guard).
  3. Runs without process_family do NOT write (we have no scope key).
  4. NoopBackend default keeps the runner contract identical.
  5. Memory write exceptions are absorbed (memory is opt-in).
"""

from __future__ import annotations

from typing import Any

from fermdocs_hypothesis.runner import (
    _collect_lessons_from_events,
    _persist_lessons_to_memory,
)
from fermdocs_hypothesis.events import LessonsSummarizedEvent
from fermdocs_hypothesis.schema import (
    HypothesisInput,
    Lesson,
    LessonsDigest,
)


class _SpyBackend:
    """Records writes; no-op on fetch/supersede."""

    def __init__(self):
        self.writes: list = []

    def write(self, record):
        self.writes.append(record)

    def fetch(self, query):
        return []

    def supersede(self, memory_id, by):
        pass


class _BoomBackend:
    """Raises on write to exercise the absorption path."""

    def write(self, record):
        raise RuntimeError("simulated outage")

    def fetch(self, query):
        return []

    def supersede(self, memory_id, by):
        pass


def _make_event_with_lessons(lessons):
    """Build a LessonsSummarizedEvent carrying the structured lessons list.

    The event's `digest` field stays a plain string (back-compat); the
    new `lessons` field carries the structured Lesson objects the
    runner reads.
    """
    from datetime import datetime, timezone
    return LessonsSummarizedEvent(
        ts=datetime.now(timezone.utc),
        turn=1,
        digest="placeholder digest text",
        source_reason_count=1,
        lessons=lessons,
    )


def _make_hyp_input(process_family: str | None = "yeast_intracellular_product_fedbatch"):
    # Build a minimal HypothesisInput. Several fields are required; we
    # use defaults / None where the runner doesn't read them in this path.
    return HypothesisInput.model_construct(
        process_family=process_family,
        organism="S. cerevisiae",
        seed_topics=[],
        diagnosis=None,
        characterization=None,
        bundle_path=None,
    )


# ---- _collect_lessons_from_events ----


def test_collect_lessons_walks_all_events():
    e1 = _make_event_with_lessons([
        Lesson(lesson_id="L-1", text="lesson 1"),
        Lesson(lesson_id="L-2", text="lesson 2"),
    ])
    e2 = _make_event_with_lessons([
        Lesson(lesson_id="L-3", text="lesson 3"),
    ])
    out = _collect_lessons_from_events([e1, e2])
    assert [l.lesson_id for l in out] == ["L-1", "L-2", "L-3"]


def test_collect_lessons_dedupes_by_lesson_id():
    """If the same lesson_id appears across turns (re-emission), surface once."""
    e1 = _make_event_with_lessons([Lesson(lesson_id="L-1", text="first")])
    e2 = _make_event_with_lessons([Lesson(lesson_id="L-1", text="dup")])
    out = _collect_lessons_from_events([e1, e2])
    assert len(out) == 1
    assert out[0].lesson_id == "L-1"


def test_collect_lessons_handles_legacy_events_with_empty_lessons():
    """Pre-Phase-1 LessonsSummarizedEvent had no structured `lessons`.
    Skip those silently rather than crashing."""
    e_legacy = _make_event_with_lessons([])
    out = _collect_lessons_from_events([e_legacy])
    assert out == []


# ---- _persist_lessons_to_memory ----


def test_persist_writes_on_clean_exit_consensus_reached():
    spy = _SpyBackend()
    events = [_make_event_with_lessons([
        Lesson(lesson_id="L-1", text="lesson 1"),
        Lesson(lesson_id="L-2", text="lesson 2"),
    ])]
    n = _persist_lessons_to_memory(
        memory=spy,
        events=events,
        hyp_input=_make_hyp_input(),
        exit_reason="consensus_reached",
        run_id="run-abc",
    )
    assert n == 2
    assert len(spy.writes) == 2
    rec = spy.writes[0]
    assert rec.kind == "lesson"
    assert rec.process_family == "yeast_intracellular_product_fedbatch"
    assert rec.organism == "S. cerevisiae"
    assert rec.provenance["run_id"] == "run-abc"


def test_persist_writes_on_clean_exit_no_topics_left():
    spy = _SpyBackend()
    events = [_make_event_with_lessons([Lesson(lesson_id="L-1", text="x")])]
    n = _persist_lessons_to_memory(
        memory=spy,
        events=events,
        hyp_input=_make_hyp_input(),
        exit_reason="no_topics_left",
        run_id="run-1",
    )
    assert n == 1
    assert len(spy.writes) == 1


def test_persist_writes_on_budget_exhausted():
    """D6 revised: budget-exhausted runs still produced valid debate —
    the lessons summarizer is the quality gate, not the exit reason."""
    spy = _SpyBackend()
    events = [_make_event_with_lessons([Lesson(lesson_id="L-1", text="x")])]
    n = _persist_lessons_to_memory(
        memory=spy,
        events=events,
        hyp_input=_make_hyp_input(),
        exit_reason="budget_exhausted",
        run_id="run-1",
    )
    assert n == 1
    assert len(spy.writes) == 1


def test_persist_writes_on_max_turns_reached():
    """D6 revised: max-turns runs completed their full allocation."""
    spy = _SpyBackend()
    events = [_make_event_with_lessons([Lesson(lesson_id="L-1", text="x")])]
    n = _persist_lessons_to_memory(
        memory=spy,
        events=events,
        hyp_input=_make_hyp_input(),
        exit_reason="max_turns_reached",
        run_id="run-1",
    )
    assert n == 1


def test_persist_skips_on_none_exit_reason():
    """Defensive guard: if exit_reason is somehow None, skip persist."""
    spy = _SpyBackend()
    events = [_make_event_with_lessons([Lesson(lesson_id="L-1", text="x")])]
    n = _persist_lessons_to_memory(
        memory=spy,
        events=events,
        hyp_input=_make_hyp_input(),
        exit_reason=None,
        run_id="run-1",
    )
    assert n == 0
    assert spy.writes == []


def test_persist_skips_when_process_family_is_none():
    """No scope key → no write. Avoids polluting the store with
    untraceable lessons."""
    spy = _SpyBackend()
    events = [_make_event_with_lessons([Lesson(lesson_id="L-1", text="x")])]
    n = _persist_lessons_to_memory(
        memory=spy,
        events=events,
        hyp_input=_make_hyp_input(process_family=None),
        exit_reason="consensus_reached",
        run_id="run-1",
    )
    assert n == 0
    assert spy.writes == []


def test_persist_skips_when_no_lessons():
    spy = _SpyBackend()
    n = _persist_lessons_to_memory(
        memory=spy,
        events=[],
        hyp_input=_make_hyp_input(),
        exit_reason="consensus_reached",
        run_id="run-1",
    )
    assert n == 0


def test_persist_absorbs_backend_exceptions():
    """Memory write failures must not break a successful run."""
    boom = _BoomBackend()
    events = [_make_event_with_lessons([
        Lesson(lesson_id="L-1", text="x"),
        Lesson(lesson_id="L-2", text="y"),
    ])]
    # Should not raise:
    n = _persist_lessons_to_memory(
        memory=boom,
        events=events,
        hyp_input=_make_hyp_input(),
        exit_reason="consensus_reached",
        run_id="run-1",
    )
    # Both attempts failed → 0 written
    assert n == 0


# ---- regression: NoopBackend default keeps everything quiet ----


def test_noop_default_writes_nothing():
    from fermdocs_memory import NoopBackend
    backend = NoopBackend()
    events = [_make_event_with_lessons([Lesson(lesson_id="L-1", text="x")])]
    n = _persist_lessons_to_memory(
        memory=backend,
        events=events,
        hyp_input=_make_hyp_input(),
        exit_reason="consensus_reached",
        run_id="run-1",
    )
    # NoopBackend.write returns None; helper increments only on success
    # since the call didn't raise. Counter reflects "tried-and-succeeded";
    # the noop's silence is still a "successful no-op write."
    assert n == 1
