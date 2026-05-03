"""HITL resume — runner.resume_stage carries forward state and answers.

v0.1 ships CLI HITL: after a run exits with unresolved open questions,
the user can answer them and trigger one more debate round. The runner
side of that loop is `resume_stage`. These tests verify it without
hitting any LLM (uses StubHooks).
"""

from __future__ import annotations

from fermdocs_hypothesis.events import (
    HumanInputReceivedEvent,
    QuestionResolvedEvent,
    StageStartedEvent,
)
from fermdocs_hypothesis.runner import StubHooks, resume_stage, run_stage
from fermdocs_hypothesis.schema import BudgetSnapshot
from fermdocs_hypothesis.state import open_questions
from tests.unit.hypothesis.fixtures import (
    DIAG_ID,
    make_input,
    make_seed_topic,
    make_simple_script,
    now_factory_const,
)


def test_resume_emits_human_input_and_resolved_events(tmp_path):
    seeds = [make_seed_topic(topic_id="T-0001")]
    script = make_simple_script(topic_ids=["T-0001"], critic_flag="green")
    global_md = tmp_path / "global.md"

    # First run — pre-seed an open question via runner pending_question_seeds
    result = run_stage(
        hyp_input=make_input(seeds),
        hooks=StubHooks(script),
        global_md_path=global_md,
        diagnosis_id=DIAG_ID,
        pending_question_seeds=[("Was DO crash at 30h?", ["DO"])],
        now_factory=now_factory_const,
    )
    unresolved = [q for q in result.output.open_questions if not q.resolved]
    assert len(unresolved) == 1
    qid = unresolved[0].qid

    # Resume with one answer
    resume_result = resume_stage(
        hyp_input=make_input(seeds),
        hooks=StubHooks(script),
        global_md_path=global_md,
        diagnosis_id=DIAG_ID,
        answers=[(qid, "Yes — DO crashed because of mixer trip at 30h.")],
        now_factory=now_factory_const,
    )

    # The new event log contains human_input_received + question_resolved
    has_human_input = any(
        isinstance(ev, HumanInputReceivedEvent) and ev.payload.get("qid") == qid
        for ev in resume_result.events
    )
    has_resolved = any(
        isinstance(ev, QuestionResolvedEvent) and ev.qid == qid
        for ev in resume_result.events
    )
    assert has_human_input
    assert has_resolved


def test_resume_question_marked_resolved_in_output(tmp_path):
    seeds = [make_seed_topic(topic_id="T-0001")]
    script = make_simple_script(topic_ids=["T-0001"], critic_flag="green")
    global_md = tmp_path / "global.md"

    run_stage(
        hyp_input=make_input(seeds),
        hooks=StubHooks(script),
        global_md_path=global_md,
        diagnosis_id=DIAG_ID,
        pending_question_seeds=[("test?", ["DO"])],
        now_factory=now_factory_const,
    )
    resume_result = resume_stage(
        hyp_input=make_input(seeds),
        hooks=StubHooks(script),
        global_md_path=global_md,
        diagnosis_id=DIAG_ID,
        answers=[("Q-0001", "answered.")],
        now_factory=now_factory_const,
    )
    qs = open_questions(resume_result.events)
    assert any(q.resolved and q.qid == "Q-0001" for q in qs)


def test_resume_carries_forward_prior_finals(tmp_path):
    """Resume should not lose the hypotheses accepted in the prior round."""
    seeds = [
        make_seed_topic(topic_id="T-0001", priority=0.9),
        make_seed_topic(topic_id="T-0002", priority=0.8),
    ]
    script = make_simple_script(topic_ids=["T-0001", "T-0002"], critic_flag="green")
    global_md = tmp_path / "global.md"

    first = run_stage(
        hyp_input=make_input(seeds),
        hooks=StubHooks(script),
        global_md_path=global_md,
        diagnosis_id=DIAG_ID,
        pending_question_seeds=[("test?", ["DO"])],
        now_factory=now_factory_const,
    )
    prior_final_ids = {h.hyp_id for h in first.output.final_hypotheses}
    assert prior_final_ids  # sanity

    resumed = resume_stage(
        hyp_input=make_input(seeds),
        hooks=StubHooks(script),
        global_md_path=global_md,
        diagnosis_id=DIAG_ID,
        answers=[("Q-0001", "answered.")],
        now_factory=now_factory_const,
    )
    new_final_ids = {h.hyp_id for h in resumed.output.final_hypotheses}
    # prior finals must still be in the resumed output
    assert prior_final_ids.issubset(new_final_ids)


def test_resume_emits_fresh_stage_started_event(tmp_path):
    seeds = [make_seed_topic(topic_id="T-0001")]
    script = make_simple_script(topic_ids=["T-0001"], critic_flag="green")
    global_md = tmp_path / "global.md"
    run_stage(
        hyp_input=make_input(seeds),
        hooks=StubHooks(script),
        global_md_path=global_md,
        diagnosis_id=DIAG_ID,
        pending_question_seeds=[("test?", ["DO"])],
        now_factory=now_factory_const,
    )
    resumed = resume_stage(
        hyp_input=make_input(seeds),
        hooks=StubHooks(script),
        global_md_path=global_md,
        diagnosis_id=DIAG_ID,
        answers=[("Q-0001", "answered.")],
        now_factory=now_factory_const,
    )
    started_events = [e for e in resumed.events if isinstance(e, StageStartedEvent)]
    # One stage_started per round (initial + resume = 2)
    assert len(started_events) == 2


def test_resume_with_empty_answers_runs_a_round(tmp_path):
    """Even with no answers (just a fresh debate round), resume_stage
    should run cleanly. Useful for re-running with a different budget."""
    seeds = [make_seed_topic(topic_id="T-0001")]
    script = make_simple_script(topic_ids=["T-0001"], critic_flag="green")
    global_md = tmp_path / "global.md"
    run_stage(
        hyp_input=make_input(seeds),
        hooks=StubHooks(script),
        global_md_path=global_md,
        diagnosis_id=DIAG_ID,
        now_factory=now_factory_const,
    )
    resumed = resume_stage(
        hyp_input=make_input(seeds),
        hooks=StubHooks(script),
        global_md_path=global_md,
        diagnosis_id=DIAG_ID,
        answers=[],
        now_factory=now_factory_const,
    )
    # Resume completed and produced a valid output
    assert resumed.output is not None
    assert resumed.state.phase == "done"


def test_resume_preserves_topic_attempt_counts(tmp_path):
    """If T-0001 was attempted in the prior round, the resume should treat
    it as already used (not pick it again unless retry logic kicks in)."""
    seeds = [
        make_seed_topic(topic_id="T-0001", priority=0.9),
        make_seed_topic(topic_id="T-0002", priority=0.8),
    ]
    script = make_simple_script(topic_ids=["T-0001", "T-0002"], critic_flag="green")
    global_md = tmp_path / "global.md"
    first = run_stage(
        hyp_input=make_input(seeds),
        hooks=StubHooks(script),
        global_md_path=global_md,
        diagnosis_id=DIAG_ID,
        budget=BudgetSnapshot(max_turns=10),
        now_factory=now_factory_const,
    )
    used_in_first = set(first.state.used_topic_ids)
    resumed = resume_stage(
        hyp_input=make_input(seeds),
        hooks=StubHooks(script),
        global_md_path=global_md,
        diagnosis_id=DIAG_ID,
        answers=[],
        budget=BudgetSnapshot(max_turns=10),
        now_factory=now_factory_const,
    )
    used_in_resume = set(resumed.state.used_topic_ids)
    # Used set in resume must include everything used in first
    assert used_in_first.issubset(used_in_resume)
