"""Topic retry-on-rejection — runner exercises max_critic_cycles_per_topic."""

from __future__ import annotations

from fermdocs_hypothesis.events import (
    HypothesisRejectedEvent,
    StageExitedEvent,
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


def test_topic_retried_on_rejection_under_cycle_cap(tmp_path):
    """When critic+judge reject a hypothesis and we're under the cycle cap,
    the runner should retry the SAME topic (re-emitting topic_selected)
    instead of moving to the next."""
    seeds = [make_seed_topic(topic_id="T-0001")]
    # All topic plans red+judge-upheld → every attempt rejected
    script = make_simple_script(topic_ids=["T-0001"], critic_flag="red", judge_valid=True)
    budget = BudgetSnapshot(max_turns=10, max_critic_cycles_per_topic=3)
    result = run_stage(
        hyp_input=make_input(seeds),
        hooks=StubHooks(script),
        global_md_path=tmp_path / "global.md",
        diagnosis_id=DIAG_ID,
        budget=budget,
        now_factory=now_factory_const,
    )
    # Should see 3 topic_selected events for T-0001 (initial + 2 retries = 3 attempts at cap)
    selects = [e for e in result.events if isinstance(e, TopicSelectedEvent) and e.topic_id == "T-0001"]
    rejects = [e for e in result.events if isinstance(e, HypothesisRejectedEvent) and e.topic_id == "T-0001"]
    assert len(selects) == 3, f"expected 3 attempts at T-0001, got {len(selects)}"
    assert len(rejects) == 3
    # After cap exhausted, runner moves on (no other topics) → no_topics_left
    exit_evs = [e for e in result.events if isinstance(e, StageExitedEvent)]
    assert exit_evs[0].reason == "no_topics_left"


def test_no_retry_when_cycle_cap_is_one(tmp_path):
    """max_critic_cycles_per_topic=1 means one attempt only — no retry."""
    seeds = [make_seed_topic(topic_id="T-0001")]
    script = make_simple_script(topic_ids=["T-0001"], critic_flag="red", judge_valid=True)
    budget = BudgetSnapshot(max_turns=10, max_critic_cycles_per_topic=1)
    result = run_stage(
        hyp_input=make_input(seeds),
        hooks=StubHooks(script),
        global_md_path=tmp_path / "global.md",
        diagnosis_id=DIAG_ID,
        budget=budget,
        now_factory=now_factory_const,
    )
    selects = [e for e in result.events if isinstance(e, TopicSelectedEvent) and e.topic_id == "T-0001"]
    assert len(selects) == 1


def test_accept_path_not_affected_by_retry_logic(tmp_path):
    """Sanity: green-flagged hypothesis still produces accept; no retry."""
    seeds = [make_seed_topic(topic_id="T-0001")]
    script = make_simple_script(topic_ids=["T-0001"], critic_flag="green")
    budget = BudgetSnapshot(max_turns=10, max_critic_cycles_per_topic=3)
    result = run_stage(
        hyp_input=make_input(seeds),
        hooks=StubHooks(script),
        global_md_path=tmp_path / "global.md",
        diagnosis_id=DIAG_ID,
        budget=budget,
        now_factory=now_factory_const,
    )
    assert len(result.output.final_hypotheses) == 1
    assert result.output.rejected_hypotheses == []


def test_retry_then_move_on_when_other_topics_exist(tmp_path):
    """T-0001 rejected 3x (cap), then T-0002 accepted — both paths exercised."""
    seeds = [
        make_seed_topic(topic_id="T-0001", priority=0.9),
        make_seed_topic(topic_id="T-0002", priority=0.8),
    ]
    from fermdocs_hypothesis.stubs.canned_agents import (
        StubFacetPlan, StubScript, StubTopicPlan,
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
            critic_reasons=["weak"],
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
    budget = BudgetSnapshot(max_turns=10, max_critic_cycles_per_topic=3)
    result = run_stage(
        hyp_input=make_input(seeds),
        hooks=StubHooks(script),
        global_md_path=tmp_path / "global.md",
        diagnosis_id=DIAG_ID,
        budget=budget,
        now_factory=now_factory_const,
    )
    # T-0001 rejected 3x then T-0002 accepted
    rejects_t1 = [e for e in result.events if isinstance(e, HypothesisRejectedEvent) and e.topic_id == "T-0001"]
    selects_t1 = [e for e in result.events if isinstance(e, TopicSelectedEvent) and e.topic_id == "T-0001"]
    selects_t2 = [e for e in result.events if isinstance(e, TopicSelectedEvent) and e.topic_id == "T-0002"]
    assert len(selects_t1) == 3
    assert len(rejects_t1) == 3
    assert len(selects_t2) == 1
    assert len(result.output.final_hypotheses) == 1
    assert result.output.final_hypotheses[0].hyp_id  # accepted
