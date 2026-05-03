"""Stage runner — pure step-function orchestration loop.

Plan ref: plans/2026-05-03-hypothesis-debate-v0.md §11 Stage 1, §16 (LangGraph
forward-compat).

The runner is intentionally a single explicit state machine. Each call to
`step(state) -> (state, events)` advances exactly one micro-step:
  - select_topic
  - contribute_facets (one per specialist, sequentially)
  - synthesize
  - critique
  - judge
  - finalize_turn (accept or reject)

A turn is N micro-steps (3 facets + synth + crit + judge + finalize = 7).
Budget is checked between micro-steps, not mid-step. This keeps the runner
trivially mappable to LangGraph nodes later.

State is a single immutable dataclass; runner mutates only by replacing it.
The Observer is the only side-effect surface (writes to global.md).

For Stage 1 the agent layer is canned stubs. Stage 2 will swap in real LLM
agents behind the same RunnerHooks protocol.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Literal, Protocol
from uuid import UUID, uuid4

from fermdocs_diagnose.schema import ConfidenceBasis
from fermdocs_hypothesis.budget import (
    add_tool_calls,
    increment_turn,
)
from fermdocs_hypothesis.event_log import Observer
from fermdocs_hypothesis.events import (
    CritiqueFiledEvent,
    Event,
    FacetContributedEvent,
    HypothesisAcceptedEvent,
    HypothesisRejectedEvent,
    HypothesisSynthesizedEvent,
    JudgeRulingEvent,
    QuestionAddedEvent,
    StageExitedEvent,
    StageStartedEvent,
    TokensUsedEvent,
    TopicSelectedEvent,
)
from fermdocs_hypothesis.instrumentation import TokenMeter
from fermdocs_hypothesis.memory import NullPastInsightStore, PastInsightStore
from fermdocs_hypothesis.projector import (
    project_specialist,
    project_synthesizer,
)
from fermdocs_hypothesis.schema import (
    BudgetSnapshot,
    CritiqueFull,
    ExitReason,
    FacetFull,
    FinalHypothesis,
    HypothesisFull,
    HypothesisInput,
    HypothesisMeta,
    HypothesisOutput,
    OpenQuestionRef,
    RejectedHypothesis,
    SeedTopic,
    SpecialistRole,
    TopicSpec,
)
from fermdocs_hypothesis.state import (
    critic_cycles_for_current_topic,
    open_questions,
)
from fermdocs_hypothesis.stubs.canned_agents import (
    StubScript,
    stub_critic_file,
    stub_judge_rule,
    stub_orchestrator_pick_topic,
    stub_specialist_contribute,
    stub_synthesizer_emit,
    topic_spec_from_seed,
)

SPECIALIST_ORDER: tuple[SpecialistRole, ...] = (
    "kinetics",
    "mass_transfer",
    "metabolic",
)

Phase = Literal[
    "init",
    "select_topic",
    "contribute_facet",
    "synthesize",
    "critique",
    "judge",
    "finalize_turn",
    "exit",
    "done",
]


@dataclass(frozen=True)
class RunnerState:
    """Single immutable state object. Replaced (not mutated) each step.

    LangGraph forward-compat: this is exactly the shape a LangGraph
    StateGraph would carry. `phase` field encodes the node-graph position.
    """

    phase: Phase
    budget: BudgetSnapshot
    seed_topics: tuple[SeedTopic, ...]
    current_turn: int = 0
    current_topic: TopicSpec | None = None
    current_facets: tuple[FacetFull, ...] = ()
    next_specialist_idx: int = 0
    current_hypothesis: HypothesisFull | None = None
    current_critique: CritiqueFull | None = None
    current_judge_valid: bool | None = None
    used_topic_ids: tuple[str, ...] = ()
    facet_counter: int = 0
    hyp_counter: int = 0
    qid_counter: int = 0
    finalized_finals: tuple[FinalHypothesis, ...] = ()
    finalized_rejected: tuple[RejectedHypothesis, ...] = ()
    exit_reason: ExitReason | None = None
    pending_question_seeds: tuple[tuple[str, list[str]], ...] = ()


class RunnerHooks(Protocol):
    """Protocol so Stage 2 can swap stubs for real LLM agents without
    changing runner code."""

    def pick_topic(self, state: RunnerState) -> str | None: ...
    def contribute_facet(
        self,
        state: RunnerState,
        role: SpecialistRole,
        facet_id: str,
    ) -> tuple[FacetFull, int, int]: ...
    def synthesize(
        self, state: RunnerState, hyp_id: str
    ) -> tuple[HypothesisFull, int, int]: ...
    def critique(
        self, state: RunnerState
    ) -> tuple[CritiqueFull, int, int]: ...
    def judge(self, state: RunnerState) -> tuple[bool, str, int, int]: ...


class StubHooks:
    """Stage 1 hooks driven by a StubScript. Pure: same script + same
    state → same outputs.
    """

    def __init__(self, script: StubScript):
        self.script = script

    def pick_topic(self, state: RunnerState) -> str | None:
        available = {t.topic_id for t in state.seed_topics} - set(state.used_topic_ids)
        return stub_orchestrator_pick_topic(
            self.script,
            topics_already_used_in_order=list(state.used_topic_ids),
            available_topic_ids=available,
        )

    def contribute_facet(
        self, state: RunnerState, role: SpecialistRole, facet_id: str
    ) -> tuple[FacetFull, int, int]:
        assert state.current_topic is not None
        plan = self.script.topic_plans[state.current_topic.topic_id].facets[role]
        # Build a SpecialistView shell — projector requires upstream pools but
        # stub specialists don't read them; pass empties.
        view = project_specialist(
            events=(),
            role=role,
            current_topic=state.current_topic,
            available_findings=[],
            available_narratives=[],
            available_trajectories=[],
            available_priors=[],
        )
        facet = stub_specialist_contribute(view, plan, facet_id)
        return facet, plan.input_tokens, plan.output_tokens

    def synthesize(
        self, state: RunnerState, hyp_id: str
    ) -> tuple[HypothesisFull, int, int]:
        assert state.current_topic is not None
        plan = self.script.topic_plans[state.current_topic.topic_id]
        view = project_synthesizer(
            current_topic=state.current_topic,
            facets=list(state.current_facets),
        )
        hyp = stub_synthesizer_emit(view, plan, hyp_id)
        return hyp, plan.synthesizer_input_tokens, plan.synthesizer_output_tokens

    def critique(
        self, state: RunnerState
    ) -> tuple[CritiqueFull, int, int]:
        assert state.current_topic is not None
        assert state.current_hypothesis is not None
        plan = self.script.topic_plans[state.current_topic.topic_id]
        crit = stub_critic_file(plan, state.current_hypothesis.hyp_id)
        return crit, plan.critic_input_tokens, plan.critic_output_tokens

    def judge(self, state: RunnerState) -> tuple[bool, str, int, int]:
        assert state.current_topic is not None
        assert state.current_hypothesis is not None
        plan = self.script.topic_plans[state.current_topic.topic_id]
        valid, rationale = stub_judge_rule(plan, state.current_hypothesis.hyp_id)
        return valid, rationale, plan.judge_input_tokens, plan.judge_output_tokens


# ---------- pure step function ----------


def step(
    state: RunnerState,
    hooks: RunnerHooks,
    *,
    events_so_far: list[Event],
    meter: TokenMeter,
    now: datetime,
) -> tuple[RunnerState, list[Event]]:
    """Advance the state machine by one micro-step.

    Returns (new_state, emitted_events). Caller (the run loop) is
    responsible for persisting events to the Observer and re-deriving any
    needed views before calling step again.

    Pure-ish: the only "impurity" is the TokenMeter (which is itself
    state-bearing by design — the per-agent ledger).
    """
    if state.phase == "init":
        # On first turn-entry, drain any pending_question_seeds into events
        # so derived state sees them before topic ranking.
        new_events: list[Event] = []
        new_qid_counter = state.qid_counter
        for question, tags in state.pending_question_seeds:
            new_qid_counter += 1
            qid = f"Q-{new_qid_counter:04d}"
            new_events.append(
                QuestionAddedEvent(
                    ts=now,
                    turn=0,
                    qid=qid,
                    question=question,
                    raised_by="orchestrator",
                    tags=list(tags),
                )
            )
        return (
            replace(
                state,
                phase="select_topic",
                pending_question_seeds=(),
                qid_counter=new_qid_counter,
            ),
            new_events,
        )

    if state.phase == "select_topic":
        # Budget check first
        exhausted, reason = state.budget.is_exhausted()
        if exhausted:
            return _to_exit(state, _budget_exit_reason(reason))

        topic_id = hooks.pick_topic(state)
        if topic_id is None:
            return _to_exit(state, "no_topics_left")

        seed = next((t for t in state.seed_topics if t.topic_id == topic_id), None)
        if seed is None:
            return _to_exit(state, "no_topics_left")

        topic_spec = topic_spec_from_seed(seed)
        new_turn = state.current_turn + 1
        new_budget = increment_turn(state.budget)
        # Did increment push us over max_turns?
        if new_budget.turns_used > new_budget.max_turns:
            return _to_exit(
                replace(state, budget=new_budget),
                "max_turns_reached",
            )

        return (
            replace(
                state,
                phase="contribute_facet",
                budget=new_budget,
                current_topic=topic_spec,
                current_facets=(),
                next_specialist_idx=0,
                current_hypothesis=None,
                current_critique=None,
                current_judge_valid=None,
                current_turn=new_turn,
                used_topic_ids=state.used_topic_ids + (topic_id,),
            ),
            [
                TopicSelectedEvent(
                    ts=now,
                    turn=new_turn,
                    topic_id=topic_id,
                    summary=seed.summary,
                    rationale=f"ranker-top: {seed.summary}",
                )
            ],
        )

    if state.phase == "contribute_facet":
        if state.next_specialist_idx >= len(SPECIALIST_ORDER):
            return (replace(state, phase="synthesize"), [])
        role = SPECIALIST_ORDER[state.next_specialist_idx]
        new_facet_counter = state.facet_counter + 1
        facet_id = f"FCT-{new_facet_counter:04d}"
        facet, in_tok, out_tok = hooks.contribute_facet(state, role, facet_id)
        new_budget = meter.record(
            state.budget,
            agent=f"specialist:{role}",
            input_tokens=in_tok,
            output_tokens=out_tok,
        )
        new_budget = add_tool_calls(new_budget, 1)
        ev = FacetContributedEvent(
            ts=now,
            turn=state.current_turn,
            facet_id=facet_id,
            topic_id=state.current_topic.topic_id,  # type: ignore[union-attr]
            specialist=role,
            summary=facet.summary,
            cited_finding_ids=list(facet.cited_finding_ids),
            cited_narrative_ids=list(facet.cited_narrative_ids),
            cited_trajectories=list(facet.cited_trajectories),
            affected_variables=list(facet.affected_variables),
            confidence=facet.confidence,
            confidence_basis=facet.confidence_basis,
        )
        token_ev = TokensUsedEvent(
            ts=now,
            turn=state.current_turn,
            agent=f"specialist:{role}",
            input=in_tok,
            output=out_tok,
        )
        return (
            replace(
                state,
                budget=new_budget,
                current_facets=state.current_facets + (facet,),
                next_specialist_idx=state.next_specialist_idx + 1,
                facet_counter=new_facet_counter,
            ),
            [ev, token_ev],
        )

    if state.phase == "synthesize":
        if not state.current_facets:
            # No facets contributed; treat as no_hypothesis turn — go pick next topic.
            return (
                replace(state, phase="select_topic", current_topic=None),
                [],
            )
        new_hyp_counter = state.hyp_counter + 1
        hyp_id = f"H-{new_hyp_counter:04d}"
        hyp, in_tok, out_tok = hooks.synthesize(state, hyp_id)
        new_budget = meter.record(
            state.budget,
            agent="synthesizer",
            input_tokens=in_tok,
            output_tokens=out_tok,
        )
        new_budget = add_tool_calls(new_budget, 1)
        return (
            replace(
                state,
                phase="critique",
                budget=new_budget,
                current_hypothesis=hyp,
                hyp_counter=new_hyp_counter,
            ),
            [
                HypothesisSynthesizedEvent(
                    ts=now,
                    turn=state.current_turn,
                    hyp_id=hyp.hyp_id,
                    topic_id=state.current_topic.topic_id,  # type: ignore[union-attr]
                    summary=hyp.summary,
                    facet_ids=list(hyp.facet_ids),
                    cited_finding_ids=list(hyp.cited_finding_ids),
                    cited_narrative_ids=list(hyp.cited_narrative_ids),
                    cited_trajectories=list(hyp.cited_trajectories),
                    affected_variables=list(hyp.affected_variables),
                    confidence=hyp.confidence,
                    confidence_basis=hyp.confidence_basis,
                ),
                TokensUsedEvent(
                    ts=now,
                    turn=state.current_turn,
                    agent="synthesizer",
                    input=in_tok,
                    output=out_tok,
                ),
            ],
        )

    if state.phase == "critique":
        crit, in_tok, out_tok = hooks.critique(state)
        new_budget = meter.record(
            state.budget,
            agent="critic",
            input_tokens=in_tok,
            output_tokens=out_tok,
        )
        new_budget = add_tool_calls(new_budget, max(1, crit.tool_calls_used))
        return (
            replace(
                state,
                phase="judge",
                budget=new_budget,
                current_critique=crit,
            ),
            [
                CritiqueFiledEvent(
                    ts=now,
                    turn=state.current_turn,
                    hyp_id=crit.hyp_id,
                    flag=crit.flag,
                    reasons=list(crit.reasons),
                    tool_calls_used=crit.tool_calls_used,
                ),
                TokensUsedEvent(
                    ts=now,
                    turn=state.current_turn,
                    agent="critic",
                    input=in_tok,
                    output=out_tok,
                ),
            ],
        )

    if state.phase == "judge":
        valid, rationale, in_tok, out_tok = hooks.judge(state)
        new_budget = meter.record(
            state.budget,
            agent="judge",
            input_tokens=in_tok,
            output_tokens=out_tok,
        )
        new_budget = add_tool_calls(new_budget, 1)
        return (
            replace(
                state,
                phase="finalize_turn",
                budget=new_budget,
                current_judge_valid=valid,
            ),
            [
                JudgeRulingEvent(
                    ts=now,
                    turn=state.current_turn,
                    hyp_id=state.current_hypothesis.hyp_id,  # type: ignore[union-attr]
                    criticism_valid=valid,
                    rationale=rationale,
                ),
                TokensUsedEvent(
                    ts=now,
                    turn=state.current_turn,
                    agent="judge",
                    input=in_tok,
                    output=out_tok,
                ),
            ],
        )

    if state.phase == "finalize_turn":
        assert state.current_topic is not None
        assert state.current_hypothesis is not None
        assert state.current_critique is not None
        assert state.current_judge_valid is not None

        criticism_upheld = state.current_judge_valid is True and state.current_critique.flag == "red"

        if criticism_upheld:
            # Reject. Possibly reuse same topic for another cycle, up to cap.
            cycles = critic_cycles_for_current_topic(
                events_so_far, state.current_topic.topic_id
            )
            # cycles counts past rejections; this rejection makes (cycles + 1)
            new_rejected = state.finalized_rejected + (
                RejectedHypothesis(
                    hyp_id=state.current_hypothesis.hyp_id,
                    summary=state.current_hypothesis.summary,
                    rejection_reason=", ".join(state.current_critique.reasons),
                    critic_reasons=list(state.current_critique.reasons),
                    judge_rationale=_judge_rationale(events_so_far, state.current_hypothesis.hyp_id),
                ),
            )
            ev = HypothesisRejectedEvent(
                ts=now,
                turn=state.current_turn,
                hyp_id=state.current_hypothesis.hyp_id,
                topic_id=state.current_topic.topic_id,
                reason=", ".join(state.current_critique.reasons) or "judge upheld critique",
            )
            # If we've hit max_critic_cycles_per_topic, drop the topic by
            # re-marking it as used (it already is) — runner naturally moves on.
            # If under cap, still move on — Stage 1 stub doesn't retry same
            # topic; retry-same-topic behavior lands in Stage 2 with a real
            # orchestrator that decides between retry vs new topic.
            return (
                replace(
                    state,
                    phase="select_topic",
                    finalized_rejected=new_rejected,
                    current_topic=None,
                    current_hypothesis=None,
                    current_critique=None,
                    current_judge_valid=None,
                    current_facets=(),
                    next_specialist_idx=0,
                ),
                [ev],
            )

        # Accept path
        new_finals = state.finalized_finals + (
            _build_final(state.current_hypothesis, state.current_critique, state.current_judge_valid),
        )
        ev = HypothesisAcceptedEvent(
            ts=now,
            turn=state.current_turn,
            hyp_id=state.current_hypothesis.hyp_id,
            topic_id=state.current_topic.topic_id,
        )
        # Consensus check
        consensus = len(new_finals) >= 2
        next_state = replace(
            state,
            phase="exit" if consensus else "select_topic",
            finalized_finals=new_finals,
            current_topic=None,
            current_hypothesis=None,
            current_critique=None,
            current_judge_valid=None,
            current_facets=(),
            next_specialist_idx=0,
            exit_reason="consensus_reached" if consensus else None,
        )
        return next_state, [ev]

    if state.phase == "exit":
        ev = StageExitedEvent(
            ts=now,
            turn=state.current_turn,
            reason=state.exit_reason or "no_topics_left",
            final_hyp_ids=[h.hyp_id for h in state.finalized_finals],
        )
        return (replace(state, phase="done"), [ev])

    # done
    return (state, [])


def _to_exit(state: RunnerState, reason: ExitReason) -> tuple[RunnerState, list[Event]]:
    return (replace(state, phase="exit", exit_reason=reason), [])


def _budget_exit_reason(reason: str | None) -> ExitReason:
    if reason == "max_turns":
        return "max_turns_reached"
    return "budget_exhausted"


def _judge_rationale(events: list[Event], hyp_id: str) -> str:
    for ev in reversed(events):
        if isinstance(ev, JudgeRulingEvent) and ev.hyp_id == hyp_id:
            return ev.rationale
    return ""


def _build_final(
    hyp: HypothesisFull, crit: CritiqueFull, judge_valid: bool
) -> FinalHypothesis:
    return FinalHypothesis(
        hyp_id=hyp.hyp_id,
        summary=hyp.summary,
        facet_ids=list(hyp.facet_ids),
        cited_finding_ids=list(hyp.cited_finding_ids),
        cited_narrative_ids=list(hyp.cited_narrative_ids),
        cited_trajectories=list(hyp.cited_trajectories),
        affected_variables=list(hyp.affected_variables),
        confidence=hyp.confidence,
        confidence_basis=hyp.confidence_basis,
        provenance_downgraded=hyp.provenance_downgraded,
        supporting_specialists=list(SPECIALIST_ORDER),
        critic_flag=crit.flag,
        judge_ruled_criticism_valid=judge_valid,
    )


# ---------- top-level run ----------


@dataclass
class RunResult:
    output: HypothesisOutput
    state: RunnerState
    events: list[Event]


def run_stage(
    *,
    hyp_input: HypothesisInput,
    hooks: RunnerHooks,
    global_md_path: Path,
    diagnosis_id: UUID,
    hypothesis_version: str = "v0.1.0",
    model_name: str = "stub",
    provider: Literal["anthropic", "gemini", "stub"] = "stub",
    budget: BudgetSnapshot | None = None,
    past_insights: PastInsightStore | None = None,
    pending_question_seeds: list[tuple[str, list[str]]] | None = None,
    now_factory=lambda: datetime.now(timezone.utc),
) -> RunResult:
    """Drive the step-function loop until phase == 'done'.

    The Observer is the one side-effect surface: it persists events. Everything
    else is pure: state in, state out.

    `pending_question_seeds`: optional [(question, tags), ...] added on the
    first init step. Lets tests and Stage 2 pre-seed the open-questions
    ledger.
    """
    past_insights = past_insights or NullPastInsightStore()
    seed_topics_tuple = tuple(hyp_input.seed_topics)
    initial_budget = budget or BudgetSnapshot()
    state = RunnerState(
        phase="init",
        budget=initial_budget,
        seed_topics=seed_topics_tuple,
        pending_question_seeds=tuple(pending_question_seeds or ()),
    )

    observer = Observer(global_md_path)
    meter = TokenMeter()
    events: list[Event] = []

    # Emit stage_started
    started = StageStartedEvent(
        ts=now_factory(),
        turn=0,
        input_diagnosis_id=str(diagnosis_id),
        budget=initial_budget,
    )
    observer.write(started)
    events.append(started)

    # Loop
    safety_counter = 0
    SAFETY_MAX = 1000
    while state.phase != "done":
        safety_counter += 1
        if safety_counter > SAFETY_MAX:
            # Hard safety net so a logic bug never wedges the test suite.
            state = replace(state, phase="exit", exit_reason="budget_exhausted")
            continue
        new_state, new_events = step(
            state,
            hooks,
            events_so_far=events,
            meter=meter,
            now=now_factory(),
        )
        if new_events:
            observer.write_many(new_events)
            events.extend(new_events)
        state = new_state

    # Build output
    meta = HypothesisMeta(
        schema_version="1.0",
        hypothesis_version=hypothesis_version,
        hypothesis_id=uuid4(),
        supersedes_diagnosis_id=diagnosis_id,
        generation_timestamp=now_factory(),
        model=model_name,
        provider=provider,
        budget_used=state.budget,
    )
    output = HypothesisOutput(
        meta=meta,
        final_hypotheses=list(state.finalized_finals),
        rejected_hypotheses=list(state.finalized_rejected),
        open_questions=open_questions(events),
        debate_summary=_render_debate_summary(events, state),
        global_md_path=str(global_md_path),
        token_report=meter.report,
    )
    return RunResult(output=output, state=state, events=events)


def _render_debate_summary(events: list[Event], state: RunnerState) -> str:
    """One-paragraph render of what happened. Stage 2 may upgrade to LLM."""
    n_topics = sum(1 for ev in events if isinstance(ev, TopicSelectedEvent))
    n_facets = sum(1 for ev in events if isinstance(ev, FacetContributedEvent))
    n_accepted = len(state.finalized_finals)
    n_rejected = len(state.finalized_rejected)
    return (
        f"Stage exited via {state.exit_reason}. "
        f"{n_topics} topics debated, {n_facets} facets contributed, "
        f"{n_accepted} hypotheses accepted, {n_rejected} rejected."
    )
