"""Lessons summarizer emits Lesson objects with stable lesson_ids (D2).

Plan ref: plans/2026-05-10-memory-layer.md commit 3.

The LessonsSummarizerAgent's previous contract was: emit a free-form
digest string. Memory-layer Phase 1 adds a structured `lessons:
list[Lesson]` field on LessonsDigest, with each Lesson carrying a
stable `lesson_id`. This is the round-trip key into the memory layer.

REGRESSION: existing callers that only read `digest.digest` keep
working; the new structured form is additive.
"""

from __future__ import annotations

from fermdocs_hypothesis.agents.lessons_summarizer import (
    LessonsSummarizerAgent,
    LessonsView,
)
from fermdocs_hypothesis.schema import Lesson, LessonsDigest


def test_stub_mode_emits_structured_lessons_on_non_empty_input():
    agent = LessonsSummarizerAgent(client=None, run_id="abc12345-run")
    view = LessonsView(recent_critic_reasons=["reason A", "reason B"])
    result = agent.summarize(view, source_reason_count=2)
    assert result.digest.lessons  # non-empty
    for lesson in result.digest.lessons:
        assert isinstance(lesson, Lesson)
        assert lesson.lesson_id.startswith("L-")
        assert lesson.text


def test_stub_mode_emits_empty_lessons_on_empty_input():
    """Empty input must still produce a valid LessonsDigest (back-compat)
    with an empty structured lessons list."""
    agent = LessonsSummarizerAgent(client=None)
    view = LessonsView(recent_critic_reasons=[])
    result = agent.summarize(view, source_reason_count=0)
    assert result.digest.lessons == []
    # Legacy digest string still present.
    assert result.digest.digest


def test_lesson_ids_are_unique_within_one_agent():
    agent = LessonsSummarizerAgent(client=None, run_id="abc12345-run")
    view = LessonsView(recent_critic_reasons=["a", "b", "c"])
    result = agent.summarize(view, source_reason_count=3)
    ids = [l.lesson_id for l in result.digest.lessons]
    assert len(ids) == len(set(ids))


def test_lesson_id_format_includes_run_short():
    """lesson_id form is L-<run_short>-<NNNN>. run_short is the first
    segment of the run_id; encodes the source run for traceability."""
    agent = LessonsSummarizerAgent(client=None, run_id="abc12345-run")
    view = LessonsView(recent_critic_reasons=["a"])
    result = agent.summarize(view, source_reason_count=1)
    lesson = result.digest.lessons[0]
    # "abc12345" first segment, padded counter
    assert lesson.lesson_id == "L-abc12345-0001"


def test_lessons_digest_back_compat_with_legacy_constructor():
    """LessonsDigest can still be built without the new `lessons` field,
    matching how global.md replay deserializes legacy events."""
    d = LessonsDigest(
        digest="some legacy digest text",
        source_reason_count=3,
        computed_at_event_idx=7,
    )
    assert d.lessons == []
    assert d.digest == "some legacy digest text"
