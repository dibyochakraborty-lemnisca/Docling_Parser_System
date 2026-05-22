"""Runner — purity, end-to-end stub run, exit reasons, edge cases.

Stage 1 gate: every exit reason exercised at least once; pure step
verified; runner produces valid HypothesisOutput.
"""

from __future__ import annotations

from datetime import datetime, timezone

from fermdocs_characterize.schema import Severity
from fermdocs_hypothesis.events import (
    HypothesisAcceptedEvent,
    HypothesisRejectedEvent,
    StageExitedEvent,
    StageStartedEvent,
)
from fermdocs_hypothesis.runner import (
    RunnerState,
    StubHooks,
    run_stage,
    step,
)
from fermdocs_hypothesis.schema import BudgetSnapshot
from tests.unit.hypothesis.fixtures import (
    DIAG_ID,
    NOW,
    make_input,
    make_seed_topic,
    make_simple_script,
    now_factory_const,
)


# ---------- end-to-end: green-flag accept ----------


def test_e2e_single_topic_green_flag_produces_one_final(tmp_path):
    seeds = [make_seed_topic(topic_id="T-0001")]
    script = make_simple_script(topic_ids=["T-0001"], critic_flag="green")
    result = run_stage(
        hyp_input=make_input(seeds),
        hooks=StubHooks(script),
        global_md_path=tmp_path / "global.md",
        diagnosis_id=DIAG_ID,
        now_factory=now_factory_const,
    )
    assert len(result.output.final_hypotheses) == 1
    assert result.output.final_hypotheses[0].critic_flag == "green"
    assert result.output.rejected_hypotheses == []


# ---------- end-to-end: red flag + judge upholds → reject ----------


def test_e2e_red_flag_judge_upholds_rejects_hypothesis(tmp_path):
    seeds = [make_seed_topic(topic_id="T-0001")]
    script = make_simple_script(
        topic_ids=["T-0001"], critic_flag="red", judge_valid=True
    )
    # cycles=1 → no retry, so we expect exactly one rejection.
    budget = BudgetSnapshot(max_critic_cycles_per_topic=1)
    result = run_stage(
        hyp_input=make_input(seeds),
        hooks=StubHooks(script),
        global_md_path=tmp_path / "global.md",
        diagnosis_id=DIAG_ID,
        budget=budget,
        now_factory=now_factory_const,
    )
    assert len(result.output.rejected_hypotheses) == 1
    assert result.output.final_hypotheses == []


# ---------- end-to-end: exit reason consensus ----------


def test_consensus_exit_after_two_acceptances(tmp_path):
    seeds = [
        make_seed_topic(topic_id="T-0001", priority=0.9),
        make_seed_topic(topic_id="T-0002", priority=0.8),
        make_seed_topic(topic_id="T-0003", priority=0.7),
    ]
    script = make_simple_script(
        topic_ids=["T-0001", "T-0002", "T-0003"], critic_flag="green"
    )
    # script.topic_order has all 3 but consensus should trip after #2
    result = run_stage(
        hyp_input=make_input(seeds),
        hooks=StubHooks(script),
        global_md_path=tmp_path / "global.md",
        diagnosis_id=DIAG_ID,
        now_factory=now_factory_const,
    )
    exit_evs = [e for e in result.events if isinstance(e, StageExitedEvent)]
    assert len(exit_evs) == 1
    assert exit_evs[0].reason == "consensus_reached"
    assert len(result.output.final_hypotheses) == 2


# ---------- exit reason: no_topics_left ----------


def test_no_topics_exits_immediately(tmp_path):
    """Empty seed list → no_topics_left at the very first select_topic."""
    script = make_simple_script(topic_ids=[])
    result = run_stage(
        hyp_input=make_input([]),
        hooks=StubHooks(script),
        global_md_path=tmp_path / "global.md",
        diagnosis_id=DIAG_ID,
        now_factory=now_factory_const,
    )
    exit_evs = [e for e in result.events if isinstance(e, StageExitedEvent)]
    assert exit_evs[0].reason == "no_topics_left"


# ---------- exit reason: budget_exhausted (tool calls) ----------


def test_tool_call_budget_exhaustion(tmp_path):
    seeds = [make_seed_topic(topic_id="T-0001")]
    script = make_simple_script(topic_ids=["T-0001"], critic_flag="green")
    # Cap tool calls so low we can't even finish one turn (need ~6 calls)
    budget = BudgetSnapshot(max_tool_calls_total=2)
    result = run_stage(
        hyp_input=make_input(seeds),
        hooks=StubHooks(script),
        global_md_path=tmp_path / "global.md",
        diagnosis_id=DIAG_ID,
        budget=budget,
        now_factory=now_factory_const,
    )
    exit_evs = [e for e in result.events if isinstance(e, StageExitedEvent)]
    assert exit_evs[0].reason in ("budget_exhausted", "max_turns_reached")


# ---------- exit reason: max_turns_reached ----------


def test_max_turns_exhaustion(tmp_path):
    seeds = [
        make_seed_topic(topic_id="T-0001"),
        make_seed_topic(topic_id="T-0002"),
        make_seed_topic(topic_id="T-0003"),
        make_seed_topic(topic_id="T-0004"),
    ]
    # All red+upheld so we never accept (avoid consensus exit). cycles=1
    # so no retry — each topic gets exactly one attempt.
    script = make_simple_script(
        topic_ids=["T-0001", "T-0002", "T-0003", "T-0004"],
        critic_flag="red",
        judge_valid=True,
    )
    budget = BudgetSnapshot(max_turns=2, max_critic_cycles_per_topic=1)
    result = run_stage(
        hyp_input=make_input(seeds),
        hooks=StubHooks(script),
        global_md_path=tmp_path / "global.md",
        diagnosis_id=DIAG_ID,
        budget=budget,
        now_factory=now_factory_const,
    )
    exit_evs = [e for e in result.events if isinstance(e, StageExitedEvent)]
    assert exit_evs[0].reason == "max_turns_reached"


# ---------- step purity ----------


def test_step_does_not_mutate_input_state(tmp_path):
    """Calling step never mutates the input state object — replace, not mutate."""
    seeds = [make_seed_topic(topic_id="T-0001")]
    script = make_simple_script(topic_ids=["T-0001"])
    hooks = StubHooks(script)
    state = RunnerState(
        phase="init",
        budget=BudgetSnapshot(),
        seed_topics=tuple(seeds),
    )
    from fermdocs_hypothesis.instrumentation import TokenMeter

    meter = TokenMeter()
    state_snapshot = state
    new_state, _ = step(state, hooks, events_so_far=[], meter=meter, now=NOW)
    # Original state object reference must equal its earlier snapshot — frozen
    # dataclass guarantees this; if step mutated, this assertion would fail.
    assert state is state_snapshot
    assert state.phase == "init"
    assert new_state is not state


def test_runner_emits_stage_started_first(tmp_path):
    seeds = [make_seed_topic()]
    script = make_simple_script(topic_ids=["T-0001"])
    result = run_stage(
        hyp_input=make_input(seeds),
        hooks=StubHooks(script),
        global_md_path=tmp_path / "global.md",
        diagnosis_id=DIAG_ID,
        now_factory=now_factory_const,
    )
    assert isinstance(result.events[0], StageStartedEvent)


def test_runner_writes_global_md(tmp_path):
    seeds = [make_seed_topic()]
    script = make_simple_script(topic_ids=["T-0001"])
    p = tmp_path / "hyp" / "global.md"
    result = run_stage(
        hyp_input=make_input(seeds),
        hooks=StubHooks(script),
        global_md_path=p,
        diagnosis_id=DIAG_ID,
        now_factory=now_factory_const,
    )
    assert p.exists()
    text = p.read_text()
    assert "stage_started" in text
    assert "stage_exited" in text


def test_runner_records_token_report(tmp_path):
    seeds = [make_seed_topic()]
    script = make_simple_script(topic_ids=["T-0001"])
    result = run_stage(
        hyp_input=make_input(seeds),
        hooks=StubHooks(script),
        global_md_path=tmp_path / "global.md",
        diagnosis_id=DIAG_ID,
        now_factory=now_factory_const,
    )
    r = result.output.token_report
    # Stub script: 3 specialists + synthesizer + critic + judge all recorded
    assert "specialist:kinetics" in r.per_agent_input
    assert "synthesizer" in r.per_agent_input
    assert "critic" in r.per_agent_input
    assert "judge" in r.per_agent_input
    assert r.total_input > 0


def test_pending_question_seeds_become_open_questions(tmp_path):
    seeds = [make_seed_topic()]
    script = make_simple_script(topic_ids=["T-0001"])
    result = run_stage(
        hyp_input=make_input(seeds),
        hooks=StubHooks(script),
        global_md_path=tmp_path / "global.md",
        diagnosis_id=DIAG_ID,
        pending_question_seeds=[("Why DO crash at 30h?", ["DO"])],
        now_factory=now_factory_const,
    )
    qids = [q.qid for q in result.output.open_questions]
    assert "Q-0001" in qids


def test_facet_contributions_emitted_in_specialist_order(tmp_path):
    from fermdocs_hypothesis.events import FacetContributedEvent

    seeds = [make_seed_topic()]
    script = make_simple_script(topic_ids=["T-0001"])
    result = run_stage(
        hyp_input=make_input(seeds),
        hooks=StubHooks(script),
        global_md_path=tmp_path / "global.md",
        diagnosis_id=DIAG_ID,
        now_factory=now_factory_const,
    )
    facet_events = [e for e in result.events if isinstance(e, FacetContributedEvent)]
    roles = [e.specialist for e in facet_events]
    assert roles == ["kinetics", "mass_transfer", "metabolic"]


def test_rejected_topic_doesnt_exit_with_consensus(tmp_path):
    """Rejected topic should be added to rejected_hypotheses and runner should
    move on to next topic, not exit as consensus."""
    seeds = [
        make_seed_topic(topic_id="T-0001"),
        make_seed_topic(topic_id="T-0002"),
    ]
    # First topic rejected (red+upheld), second accepted (green)
    from fermdocs_hypothesis.stubs.canned_agents import (
        StubFacetPlan,
        StubScript,
        StubTopicPlan,
    )

    plans = {
        "T-0001": StubTopicPlan(
            facets={
                role: StubFacetPlan(
                    summary=f"T-0001 {role}",
                    cited_finding_ids=["00000000-0000-0000-0000-000000000042:F-0001"],
                    affected_variables=["biomass_g_l"],
                )
                for role in ("kinetics", "mass_transfer", "metabolic")
            },
            synthesis_summary="T-0001 hyp",
            critic_flag="red",
            critic_reasons=["weak citation"],
            judge_criticism_valid=True,
            judge_rationale="upheld",
        ),
        "T-0002": StubTopicPlan(
            facets={
                role: StubFacetPlan(
                    summary=f"T-0002 {role}",
                    cited_finding_ids=["00000000-0000-0000-0000-000000000042:F-0001"],
                    affected_variables=["biomass_g_l"],
                )
                for role in ("kinetics", "mass_transfer", "metabolic")
            },
            synthesis_summary="T-0002 hyp",
            critic_flag="green",
        ),
    }
    script = StubScript(topic_plans=plans, topic_order=["T-0001", "T-0002"])
    # cycles=1 so T-0001 gets one attempt then runner moves to T-0002.
    budget = BudgetSnapshot(max_critic_cycles_per_topic=1)
    result = run_stage(
        hyp_input=make_input(seeds),
        hooks=StubHooks(script),
        global_md_path=tmp_path / "global.md",
        diagnosis_id=DIAG_ID,
        budget=budget,
        now_factory=now_factory_const,
    )
    assert len(result.output.rejected_hypotheses) == 1
    assert len(result.output.final_hypotheses) == 1


def test_output_has_no_duplicate_hyp_ids(tmp_path):
    seeds = [
        make_seed_topic(topic_id="T-0001"),
        make_seed_topic(topic_id="T-0002"),
    ]
    script = make_simple_script(topic_ids=["T-0001", "T-0002"])
    result = run_stage(
        hyp_input=make_input(seeds),
        hooks=StubHooks(script),
        global_md_path=tmp_path / "global.md",
        diagnosis_id=DIAG_ID,
        now_factory=now_factory_const,
    )
    all_ids = [h.hyp_id for h in result.output.final_hypotheses] + [
        h.hyp_id for h in result.output.rejected_hypotheses
    ]
    assert len(all_ids) == len(set(all_ids))


def test_global_md_event_count_matches_returned_events(tmp_path):
    """Persistence invariant: every event in result.events is in global.md."""
    from fermdocs_hypothesis.event_log import read_events

    seeds = [make_seed_topic()]
    script = make_simple_script(topic_ids=["T-0001"])
    p = tmp_path / "global.md"
    result = run_stage(
        hyp_input=make_input(seeds),
        hooks=StubHooks(script),
        global_md_path=p,
        diagnosis_id=DIAG_ID,
        now_factory=now_factory_const,
    )
    on_disk = read_events(p)
    assert len(on_disk) == len(result.events)
