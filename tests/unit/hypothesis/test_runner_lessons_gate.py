"""Runner-level: lessons summarizer is invoked on retry phases when the
cumulative critic-reason count grew past the cached digest's
source_reason_count, and skipped otherwise.

This is the cache-key behavior that bounds summarizer cost — without it
the runner would call the LLM on every retry regardless of whether the
input changed.
"""

from __future__ import annotations

from typing import Any

from fermdocs_hypothesis.events import (
    LessonsSummarizedEvent,
    TokensUsedEvent,
    TopicSelectedEvent,
)
from fermdocs_hypothesis.runner import StubHooks, run_stage
from fermdocs_hypothesis.schema import BudgetSnapshot
from tests.unit.hypothesis.fixtures import (
    DIAG_ID,
    make_input,
    make_seed_topic,
    make_simple_script,
    now_factory_const,
)


class _RecordingHooks(StubHooks):
    """Wraps StubHooks but records each summarize_lessons call so the test
    can assert call count + arg progression."""

    def __init__(self, script):
        super().__init__(script)
        self.calls: list[dict[str, Any]] = []

    def summarize_lessons(self, state, recent_reasons, source_reason_count):
        self.calls.append(
            {
                "reason_count": source_reason_count,
                "reasons": list(recent_reasons),
            }
        )
        return super().summarize_lessons(state, recent_reasons, source_reason_count)


def _make_recording_hooks(script):
    return _RecordingHooks(script)


def test_summarizer_invoked_once_per_retry_when_reasons_grow(tmp_path):
    """T-0001 rejected 3x at cycle cap. Reason count grows 0→1→2→3 across
    rejections, so the summarizer fires on retries 1 and 2 (live_count=1
    then 2). Initial attempt has no retry phase. Final rejection at
    cap=3 doesn't trigger a retry phase, so no third call."""
    seeds = [make_seed_topic(topic_id="T-0001")]
    script = make_simple_script(
        topic_ids=["T-0001"], critic_flag="red", judge_valid=True
    )
    hooks = _make_recording_hooks(script)
    budget = BudgetSnapshot(max_turns=10, max_critic_cycles_per_topic=3)

    result = run_stage(
        hyp_input=make_input(seeds),
        hooks=hooks,
        global_md_path=tmp_path / "global.md",
        diagnosis_id=DIAG_ID,
        budget=budget,
        now_factory=now_factory_const,
    )

    # 3 attempts → 2 retries → 2 summarizer invocations
    assert len(hooks.calls) == 2, (
        f"expected 2 summarizer calls (one per retry), got {len(hooks.calls)}"
    )
    # Reason count strictly grows
    counts = [c["reason_count"] for c in hooks.calls]
    assert counts == sorted(counts) and len(set(counts)) == len(counts), (
        f"reason_count should grow monotonically, got {counts}"
    )
    # Each call should see ALL accumulated reasons up to that point
    assert hooks.calls[0]["reason_count"] == 1
    assert hooks.calls[1]["reason_count"] == 2

    # Each call emits a LessonsSummarizedEvent
    lessons_events = [
        e for e in result.events if isinstance(e, LessonsSummarizedEvent)
    ]
    assert len(lessons_events) == 2
    # source_reason_count on event matches what hook saw
    assert [e.source_reason_count for e in lessons_events] == [1, 2]


def test_summarizer_not_invoked_when_no_retry_happens(tmp_path):
    """Single accepted hypothesis → no retry → no summarizer call."""
    seeds = [make_seed_topic(topic_id="T-0001")]
    script = make_simple_script(topic_ids=["T-0001"], critic_flag="green")
    hooks = _make_recording_hooks(script)
    budget = BudgetSnapshot(max_turns=10, max_critic_cycles_per_topic=3)

    result = run_stage(
        hyp_input=make_input(seeds),
        hooks=hooks,
        global_md_path=tmp_path / "global.md",
        diagnosis_id=DIAG_ID,
        budget=budget,
        now_factory=now_factory_const,
    )
    assert hooks.calls == []
    assert [e for e in result.events if isinstance(e, LessonsSummarizedEvent)] == []


def test_summarizer_failure_does_not_block_retry(tmp_path):
    """When summarize_lessons raises, the runner falls back silently —
    retry still happens, no LessonsSummarizedEvent is emitted, debate
    continues. Lessons are advisory; never load-bearing."""

    class _FailingHooks(StubHooks):
        def __init__(self, script):
            super().__init__(script)
            self.calls = 0

        def summarize_lessons(self, state, recent_reasons, source_reason_count):
            self.calls += 1
            raise RuntimeError("simulated gemini timeout")

    seeds = [make_seed_topic(topic_id="T-0001")]
    script = make_simple_script(
        topic_ids=["T-0001"], critic_flag="red", judge_valid=True
    )
    hooks = _FailingHooks(script)
    budget = BudgetSnapshot(max_turns=10, max_critic_cycles_per_topic=3)

    result = run_stage(
        hyp_input=make_input(seeds),
        hooks=hooks,
        global_md_path=tmp_path / "global.md",
        diagnosis_id=DIAG_ID,
        budget=budget,
        now_factory=now_factory_const,
    )

    # Summarizer was called (and raised) on each retry
    assert hooks.calls == 2
    # No lessons events emitted (failures are silent)
    assert [e for e in result.events if isinstance(e, LessonsSummarizedEvent)] == []
    # Retry still happened — 3 topic selections, 3 rejections
    selects = [e for e in result.events if isinstance(e, TopicSelectedEvent)]
    assert len(selects) == 3


def test_summarizer_tokens_recorded_in_token_report(tmp_path):
    """When summarizer returns non-zero tokens, they land in the token
    report under agent='lessons_summarizer'."""

    class _TokenReportingHooks(StubHooks):
        def summarize_lessons(self, state, recent_reasons, source_reason_count):
            digest, _, _ = super().summarize_lessons(
                state, recent_reasons, source_reason_count
            )
            return digest, 200, 80

    seeds = [make_seed_topic(topic_id="T-0001")]
    script = make_simple_script(
        topic_ids=["T-0001"], critic_flag="red", judge_valid=True
    )
    hooks = _TokenReportingHooks(script)
    budget = BudgetSnapshot(max_turns=10, max_critic_cycles_per_topic=3)

    result = run_stage(
        hyp_input=make_input(seeds),
        hooks=hooks,
        global_md_path=tmp_path / "global.md",
        diagnosis_id=DIAG_ID,
        budget=budget,
        now_factory=now_factory_const,
    )

    report = result.output.token_report
    assert report.per_agent_input.get("lessons_summarizer") == 400  # 2 calls × 200
    assert report.per_agent_output.get("lessons_summarizer") == 160  # 2 calls × 80

    # Per-call TokensUsedEvents present
    token_evs = [
        e
        for e in result.events
        if isinstance(e, TokensUsedEvent) and e.agent == "lessons_summarizer"
    ]
    assert len(token_evs) == 2
