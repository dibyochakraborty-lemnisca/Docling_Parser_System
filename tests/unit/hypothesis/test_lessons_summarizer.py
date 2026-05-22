"""Tests for LessonsSummarizerAgent.

Two scopes:
  - Stub mode (client=None): deterministic output, used by runner-gating
    tests so summarizer behavior is reproducible without an LLM.
  - Real-prompt mode with a mocked Gemini client: asserts the prompt
    structure and schema parsing path. Doesn't exercise actual model
    quality (that belongs in evals).
"""

from __future__ import annotations

from fermdocs_hypothesis.agents.lessons_summarizer import (
    LESSONS_INVARIANTS,
    LESSONS_SYSTEM,
    LessonsSummarizerAgent,
    LessonsView,
)


# ---------- stub mode ----------


def test_stub_summarize_empty_view_safe():
    agent = LessonsSummarizerAgent(client=None)
    res = agent.summarize(LessonsView(recent_critic_reasons=[]), source_reason_count=0)
    assert res.digest.digest == "DETERMINISTIC[0]: (empty)"
    assert res.digest.source_reason_count == 0
    assert res.input_tokens == 0
    assert res.output_tokens == 0


def test_stub_summarize_encodes_inputs_deterministically():
    agent = LessonsSummarizerAgent(client=None)
    view = LessonsView(recent_critic_reasons=["a", "b", "c"])
    res = agent.summarize(view, source_reason_count=3)
    assert res.digest.digest == "DETERMINISTIC[3]: a | b | c"
    assert res.digest.source_reason_count == 3


def test_stub_summarize_caps_at_5_for_readability():
    agent = LessonsSummarizerAgent(client=None)
    reasons = [f"r{i}" for i in range(10)]
    res = agent.summarize(
        LessonsView(recent_critic_reasons=reasons), source_reason_count=10
    )
    # Stub joins first 5 only (digest readability), but encodes total count
    assert "DETERMINISTIC[10]:" in res.digest.digest
    assert res.digest.digest.count("|") == 4  # 5 items → 4 separators


# ---------- real-prompt mode (mocked Gemini client) ----------


class _MockClient:
    """Stand-in for GeminiHypothesisClient.call. Records args, returns
    canned parsed output."""

    def __init__(self, lessons: list[str]):
        self._lessons = lessons
        self.last_system: str = ""
        self.last_user_text: str = ""

    def call(self, *, system, user_text, response_schema, temperature=0.0):
        self.last_system = system
        self.last_user_text = user_text
        return {"lessons": list(self._lessons)}, 100, 50


def test_real_mode_calls_client_with_lessons_system_prompt():
    mock = _MockClient(lessons=["lesson 1", "lesson 2"])
    agent = LessonsSummarizerAgent(client=mock)  # type: ignore[arg-type]
    view = LessonsView(recent_critic_reasons=["overreach a", "overreach b"])

    res = agent.summarize(view, source_reason_count=2)

    # System prompt landed and contains the identity line
    assert "Lessons Summarizer" in mock.last_system
    # Invariants survive into prompt assembly
    assert any(inv in mock.last_system for inv in LESSONS_INVARIANTS)
    # Input reasons land in the user_text view section
    assert "overreach a" in mock.last_user_text
    # Token counts pass through
    assert res.input_tokens == 100
    assert res.output_tokens == 50
    # Lessons formatted into a digest
    assert "lesson 1" in res.digest.digest
    assert "lesson 2" in res.digest.digest
    assert res.digest.source_reason_count == 2


def test_real_mode_empty_lessons_returns_no_pattern_message():
    mock = _MockClient(lessons=[])
    agent = LessonsSummarizerAgent(client=mock)  # type: ignore[arg-type]
    res = agent.summarize(
        LessonsView(recent_critic_reasons=["a", "b"]), source_reason_count=2
    )
    assert "no recurring patterns surfaced" in res.digest.digest
    # Still records source_reason_count so the runner doesn't keep retrying
    assert res.digest.source_reason_count == 2


def test_real_mode_truncates_long_lessons_to_200_chars():
    long = "x" * 400
    mock = _MockClient(lessons=[long])
    agent = LessonsSummarizerAgent(client=mock)  # type: ignore[arg-type]
    res = agent.summarize(
        LessonsView(recent_critic_reasons=["a"]), source_reason_count=1
    )
    # 200 chars + the "  - " prefix from _format_digest
    assert "x" * 200 in res.digest.digest
    assert "x" * 201 not in res.digest.digest


# ---------- prompt-content invariants ----------


def test_lessons_system_prompt_disallows_one_off_complaints():
    """Single-occurrence reasons would noise the digest. Prompt must say so."""
    assert "≥2" in LESSONS_SYSTEM or "appear" in LESSONS_SYSTEM.lower()


def test_lessons_invariants_cap_lesson_length():
    flat = " ".join(LESSONS_INVARIANTS)
    assert "200" in flat  # the ≤200 char rule
