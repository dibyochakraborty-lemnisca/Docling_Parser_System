"""Forward-compat audit: feedback loop + HITL coexist cleanly.

HITL v1 (next feature after this one) ships mid-debate human steers —
operator says "actually, this hypothesis is wrong because X" or "narrow
to batches 1-3 only". Those steers will arrive as events on the same
log the feedback loop already projects from.

These tests prove:
  1. HumanInputRecord slot on AttemptRecord is real and reachable —
     when HITL fills it via projector extension, AttemptRecord
     serializes correctly today.
  2. resume_stage (the existing v0.1 HITL surface) survives the new
     view fields with empty defaults; paused→answered→resumed runs
     still work.
  3. AttemptRecord can be hand-constructed with a populated
     human_input field today — no schema change needed when HITL
     starts emitting events that feed it.
  4. The view caps work uniformly: previous_attempts is unbounded,
     but the runner only emits up to max_critic_cycles_per_topic
     attempts per topic, so the practical bound is enforced upstream.
"""

from __future__ import annotations

from datetime import datetime, timezone

from fermdocs_hypothesis.events import (
    HumanInputReceivedEvent,
    QuestionResolvedEvent,
)
from fermdocs_hypothesis.runner import StubHooks, resume_stage, run_stage
from fermdocs_hypothesis.schema import (
    AttemptRecord,
    BudgetSnapshot,
    HumanInputRecord,
    LessonsDigest,
    SynthesizerView,
    TopicHistoryEntry,
)
from tests.unit.hypothesis.fixtures import (
    DIAG_ID,
    make_input,
    make_seed_topic,
    make_simple_script,
    now_factory_const,
)

NOW = datetime(2026, 5, 3, 12, 0, 0, tzinfo=timezone.utc)


# ---------- (1) HumanInputRecord slot is real today ----------


def test_attempt_record_accepts_human_input_payload():
    """When HITL v1 emits an event mapping a human note to a hyp_id, the
    projector will populate AttemptRecord.human_input. This test proves
    the schema accepts it today — no migration needed."""
    note = HumanInputRecord(note="Operator: narrow to batches 1-3 only")
    a = AttemptRecord(
        hyp_id="H-0001",
        hypothesis_summary="x",
        critic_flag="red",
        critic_reasons=["scope drift"],
        judge_ruling="valid",
        judge_rationale="upheld",
        human_input=note,
    )
    assert a.human_input is not None
    assert a.human_input.note.startswith("Operator")


def test_human_input_record_requires_non_empty_note():
    """min_length=1 guards against accidental empty-string population
    once HITL lands. An empty note is meaningless; require explicit
    omission instead (None)."""
    import pytest
    from pydantic import ValidationError

    with pytest.raises(ValidationError):
        HumanInputRecord(note="")


def test_topic_history_entry_serializes_with_human_input():
    """Serialization roundtrip: HumanInputRecord nested in AttemptRecord
    inside TopicHistoryEntry must JSON-roundtrip. The event log persists
    via Pydantic, so this confirms the wire format is HITL-ready."""
    note = HumanInputRecord(note="redirect: focus on temperature transients")
    history = TopicHistoryEntry(
        topic_id="T-0001",
        summary="biomass plateau",
        attempts=[
            AttemptRecord(
                hyp_id="H-0001",
                hypothesis_summary="initial overreach",
                critic_flag="red",
                critic_reasons=["doc absence ≠ ruled out"],
                judge_ruling="valid",
                human_input=note,
            ),
        ],
        status="in_progress",
    )
    payload = history.model_dump_json()
    restored = TopicHistoryEntry.model_validate_json(payload)
    assert restored.attempts[0].human_input is not None
    assert restored.attempts[0].human_input.note == note.note


# ---------- (2) resume_stage survives feedback-loop view changes ----------


def test_resume_stage_survives_with_feedback_loop_fields(tmp_path):
    """The HITL resume path drove fine before the feedback loop landed.
    With new fields (previous_attempts, cross_topic_lessons), the resume
    path must still pass — defaults are empty, so no behavior changes
    on resume of a green-flag topic."""
    seeds = [make_seed_topic(topic_id="T-0001")]
    script = make_simple_script(topic_ids=["T-0001"], critic_flag="green")
    global_md = tmp_path / "global.md"

    run_stage(
        hyp_input=make_input(seeds),
        hooks=StubHooks(script),
        global_md_path=global_md,
        diagnosis_id=DIAG_ID,
        pending_question_seeds=[("Was DO crash at 30h?", ["DO"])],
        now_factory=now_factory_const,
    )

    resume_result = resume_stage(
        hyp_input=make_input(seeds),
        hooks=StubHooks(script),
        global_md_path=global_md,
        diagnosis_id=DIAG_ID,
        answers=[("Q-0001", "Yes — DO crashed at 30h.")],
        now_factory=now_factory_const,
    )

    # Resume produced human-input + resolved events, same as before.
    assert any(isinstance(e, HumanInputReceivedEvent) for e in resume_result.events)
    assert any(isinstance(e, QuestionResolvedEvent) for e in resume_result.events)
    # The hypothesis output completed without error.
    assert resume_result.output.meta.error is None


def test_resume_after_rejection_loop_carries_prior_attempts(tmp_path):
    """Run 1: T-0001 rejected at cycle cap (3 attempts).
    Resume: prior rejected hypotheses are surfaced in the resumed run's
    state, so a HITL operator answering open questions still sees the
    full prior debate context."""
    seeds = [
        make_seed_topic(topic_id="T-0001", priority=0.9),
        make_seed_topic(topic_id="T-0002", priority=0.5),
    ]
    # T-0001 always red, T-0002 always green
    from fermdocs_hypothesis.stubs.canned_agents import (
        StubFacetPlan,
        StubScript,
        StubTopicPlan,
    )

    F1 = "00000000-0000-0000-0000-000000000042:F-0001"

    plans = {
        "T-0001": StubTopicPlan(
            facets={
                role: StubFacetPlan(
                    summary=f"T-0001 {role}",
                    cited_finding_ids=[F1],
                    affected_variables=["biomass_g_l"],
                )
                for role in ("kinetics", "mass_transfer", "metabolic")
            },
            synthesis_summary="T-0001 hyp",
            critic_flag="red",
            critic_reasons=["weak"],
            judge_criticism_valid=True,
            judge_rationale="upheld",
        ),
        "T-0002": StubTopicPlan(
            facets={
                role: StubFacetPlan(
                    summary=f"T-0002 {role}",
                    cited_finding_ids=[F1],
                    affected_variables=["biomass_g_l"],
                )
                for role in ("kinetics", "mass_transfer", "metabolic")
            },
            synthesis_summary="T-0002 hyp",
            critic_flag="green",
        ),
    }
    script = StubScript(topic_plans=plans, topic_order=["T-0001", "T-0002"])
    global_md = tmp_path / "global.md"
    budget = BudgetSnapshot(max_turns=10, max_critic_cycles_per_topic=3)

    first = run_stage(
        hyp_input=make_input(seeds),
        hooks=StubHooks(script),
        global_md_path=global_md,
        diagnosis_id=DIAG_ID,
        pending_question_seeds=[("Manual check pls?", ["biomass_g_l"])],
        budget=budget,
        now_factory=now_factory_const,
    )
    # 3 rejections on T-0001 + 1 acceptance on T-0002
    assert len(first.output.rejected_hypotheses) == 3
    assert len(first.output.final_hypotheses) == 1

    # Operator answers, run resumes
    resumed = resume_stage(
        hyp_input=make_input(seeds),
        hooks=StubHooks(script),
        global_md_path=global_md,
        diagnosis_id=DIAG_ID,
        answers=[("Q-0001", "Yes, the operator confirms")],
        budget=budget,
        now_factory=now_factory_const,
    )
    # Prior rejections survive into resumed output
    assert len(resumed.output.rejected_hypotheses) >= 3
    # Prior acceptance survives too
    assert len(resumed.output.final_hypotheses) >= 1


# ---------- (3) view shape stable for HITL extension ----------


def test_synthesizer_view_accepts_lessons_digest_with_optional_fields():
    """LessonsDigest is reused unchanged when HITL adds operator-injected
    lessons (e.g. 'operator note: future hypotheses must cite trajectory
    data'). The digest field is a plain string — extension is by content,
    not by schema change."""
    digest = LessonsDigest(
        digest="OPERATOR: prefer trajectory-grounded claims",
        source_reason_count=0,  # 0 because the source isn't critic_reasons
        computed_at_event_idx=42,
    )
    # The digest field is untyped beyond min_length=1, so HITL-sourced
    # digests fit without schema migration.
    assert "OPERATOR" in digest.digest


def test_attempt_record_field_set_is_minimal_and_extension_safe():
    """Schema discipline: AttemptRecord exposes ONLY what views need.
    If HITL adds a new dimension (e.g. operator_action), it lands as a
    new optional field — the existing fields stay frozen.

    This test pins the current shape so a future PR adding fields is
    explicit about what's new vs accidental."""
    expected = {
        "hyp_id",
        "hypothesis_summary",
        "critic_flag",
        "critic_reasons",
        "judge_ruling",
        "judge_rationale",
        "human_input",
    }
    actual = set(AttemptRecord.model_fields.keys())
    assert actual == expected, (
        f"AttemptRecord shape changed. Added: {actual - expected}, "
        f"Removed: {expected - actual}. If intentional, update this test."
    )


# ---------- (4) view caps still bound context ----------


def test_synthesizer_view_previous_attempts_bounded_by_runner_cap():
    """previous_attempts has no schema-level cap, but the runner enforces
    max_critic_cycles_per_topic which bounds the practical depth. A
    topic at cycle cap=3 produces at most 2 prior attempts visible to
    a 3rd-attempt synthesis (the current attempt isn't yet in history)."""
    # Just constructing a SynthesizerView with a long previous_attempts
    # list to confirm the schema doesn't reject it — the bound is policy,
    # not schema. Documents that intent.
    long_history = [
        AttemptRecord(
            hyp_id=f"H-{i:04d}",
            hypothesis_summary=f"attempt {i}",
            critic_flag="red",
            critic_reasons=[f"reason {i}"],
        )
        for i in range(1, 11)
    ]
    from fermdocs_hypothesis.schema import (
        CitationCatalog,
        TopicSourceType,
        TopicSpec,
    )

    view = SynthesizerView(
        current_topic=TopicSpec(
            topic_id="T-0001",
            summary="x",
            source_type=TopicSourceType.FAILURE,
        ),
        facets=[],
        citation_universe=CitationCatalog(),
        previous_attempts=long_history,
    )
    assert len(view.previous_attempts) == 10  # accepted; runner bounds it
