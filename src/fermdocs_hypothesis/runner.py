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
    LessonsSummarizedEvent,
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
    all_critic_reasons,
    critic_cycles_for_current_topic,
    latest_lessons_digest,
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
    "retry_topic",
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
    # Set by finalize_turn rejection branch when the topic should be re-attempted
    # (under max_critic_cycles_per_topic). Consumed by retry_topic phase, which
    # re-emits topic_selected so the cycle counter ticks.
    retry_topic_id: str | None = None


class RunnerHooks(Protocol):
    """Protocol so Stage 2 can swap stubs for real LLM agents without
    changing runner code.

    `events` is the event log up to this point. The runner passes it into
    synthesize/critique/judge so projectors can populate previous_attempts
    and cross_topic_lessons (the feedback loop). Stub implementations may
    ignore it; LiveHooks threads it into project_synthesizer/critic/judge.
    """

    def pick_topic(self, state: RunnerState) -> str | None: ...
    def contribute_facet(
        self,
        state: RunnerState,
        role: SpecialistRole,
        facet_id: str,
    ) -> tuple[FacetFull, int, int]: ...
    def synthesize(
        self,
        state: RunnerState,
        hyp_id: str,
        *,
        events: list[Event],
    ) -> tuple[HypothesisFull, int, int]: ...
    def critique(
        self,
        state: RunnerState,
        *,
        events: list[Event],
    ) -> tuple[CritiqueFull, int, int]: ...
    def judge(
        self,
        state: RunnerState,
        *,
        events: list[Event],
    ) -> tuple[bool, str, int, int]: ...

    def summarize_lessons(
        self, state: RunnerState, recent_reasons: list[str], source_reason_count: int
    ) -> tuple[str, int, int]:
        """Compress recurring critic complaints into a digest. Called by
        runner on retry_topic phases when reason count grew past the cached
        digest's source_reason_count.

        Returns (digest_text, input_tokens, output_tokens). On error the
        runner falls back silently — implementations may return ("", 0, 0)
        to mean "no digest available, skip this round".
        """
        ...


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
        self, state: RunnerState, hyp_id: str, *, events: list[Event]
    ) -> tuple[HypothesisFull, int, int]:
        assert state.current_topic is not None
        plan = self.script.topic_plans[state.current_topic.topic_id]
        # Stubs ignore `events` — the canned synthesizer doesn't read
        # previous_attempts. Production LiveHooks passes events into the
        # projector so the feedback loop reaches real LLM prompts.
        view = project_synthesizer(
            current_topic=state.current_topic,
            facets=list(state.current_facets),
        )
        hyp = stub_synthesizer_emit(view, plan, hyp_id)
        return hyp, plan.synthesizer_input_tokens, plan.synthesizer_output_tokens

    def critique(
        self, state: RunnerState, *, events: list[Event]
    ) -> tuple[CritiqueFull, int, int]:
        assert state.current_topic is not None
        assert state.current_hypothesis is not None
        plan = self.script.topic_plans[state.current_topic.topic_id]
        crit = stub_critic_file(plan, state.current_hypothesis.hyp_id)
        return crit, plan.critic_input_tokens, plan.critic_output_tokens

    def judge(
        self, state: RunnerState, *, events: list[Event]
    ) -> tuple[bool, str, int, int]:
        assert state.current_topic is not None
        assert state.current_hypothesis is not None
        plan = self.script.topic_plans[state.current_topic.topic_id]
        valid, rationale = stub_judge_rule(plan, state.current_hypothesis.hyp_id)
        return valid, rationale, plan.judge_input_tokens, plan.judge_output_tokens

    def summarize_lessons(
        self, state: RunnerState, recent_reasons: list[str], source_reason_count: int
    ) -> tuple[str, int, int]:
        # Deterministic stub: encodes inputs so the runner's caching test
        # can assert exact strings without an LLM round-trip.
        joined = " | ".join(recent_reasons[:5])
        text = f"DETERMINISTIC[{len(recent_reasons)}]: {joined}" if recent_reasons else ""
        return text, 0, 0


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

    if state.phase == "retry_topic":
        # Re-attempt the same topic. Budget check first so a retry can also
        # exhaust max_turns.
        exhausted, reason = state.budget.is_exhausted()
        if exhausted:
            return _to_exit(state, _budget_exit_reason(reason))
        topic_id = state.retry_topic_id
        if topic_id is None or state.current_topic is None or state.current_topic.topic_id != topic_id:
            # Defensive: shouldn't happen; fall back to picking new topic.
            return (replace(state, phase="select_topic", retry_topic_id=None), [])
        seed = next((t for t in state.seed_topics if t.topic_id == topic_id), None)
        if seed is None:
            return (replace(state, phase="select_topic", retry_topic_id=None), [])

        new_turn = state.current_turn + 1
        new_budget = increment_turn(state.budget)
        if new_budget.turns_used > new_budget.max_turns:
            return _to_exit(
                replace(state, budget=new_budget),
                "max_turns_reached",
            )

        # ---- Lessons summarizer gate ----
        # On retry, if cumulative critic-reason count grew past the cached
        # digest's source_reason_count, refresh the digest so the next
        # synthesizer/critic/judge see updated cross-topic lessons.
        # Cap input at last 20 reasons to bound token cost regardless of
        # debate length (state.all_critic_reasons enforces the cap).
        emitted_lessons: list[Event] = []
        recent_reasons = all_critic_reasons(events_so_far, limit=20)
        cached = latest_lessons_digest(events_so_far)
        cached_count = cached.source_reason_count if cached else 0
        live_count = sum(
            len(ev.reasons)
            for ev in events_so_far
            if isinstance(ev, CritiqueFiledEvent)
        )
        if recent_reasons and live_count > cached_count:
            try:
                digest_text, in_tok, out_tok = hooks.summarize_lessons(
                    state, recent_reasons, live_count
                )
            except Exception:
                # Silent fallback: lessons are advisory; never block retry
                # on summarizer failure. Synthesizer still has previous_attempts.
                digest_text, in_tok, out_tok = "", 0, 0
            if digest_text:
                new_budget = meter.record(
                    new_budget,
                    agent="lessons_summarizer",
                    input_tokens=in_tok,
                    output_tokens=out_tok,
                )
                emitted_lessons.append(
                    LessonsSummarizedEvent(
                        ts=now,
                        turn=new_turn,
                        digest=digest_text,
                        source_reason_count=live_count,
                    )
                )
                if in_tok or out_tok:
                    emitted_lessons.append(
                        TokensUsedEvent(
                            ts=now,
                            turn=new_turn,
                            agent="lessons_summarizer",
                            input=in_tok,
                            output=out_tok,
                        )
                    )

        # Re-emit topic_selected so cycle counter (which scans from latest
        # topic_selected) and topic_attempt_counts both tick up correctly.
        # Don't re-add to used_topic_ids — already there from first attempt.
        return (
            replace(
                state,
                phase="contribute_facet",
                budget=new_budget,
                current_facets=(),
                next_specialist_idx=0,
                current_hypothesis=None,
                current_critique=None,
                current_judge_valid=None,
                current_turn=new_turn,
                retry_topic_id=None,
            ),
            emitted_lessons + [
                TopicSelectedEvent(
                    ts=now,
                    turn=new_turn,
                    topic_id=topic_id,
                    summary=seed.summary,
                    rationale=f"retry attempt for topic {topic_id} after rejection",
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
        hyp, in_tok, out_tok = hooks.synthesize(state, hyp_id, events=events_so_far)
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
        crit, in_tok, out_tok = hooks.critique(state, events=events_so_far)
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
        valid, rationale, in_tok, out_tok = hooks.judge(state, events=events_so_far)
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
            # Reject. Decide retry-same-topic vs move-on based on TOTAL
            # rejections for this topic (across all attempts), not just
            # since the last topic_selected — otherwise retries reset
            # the counter and the loop never terminates.
            from fermdocs_hypothesis.state import topic_rejection_counts
            past_rejections = topic_rejection_counts(events_so_far).get(
                state.current_topic.topic_id, 0
            )
            cycles = past_rejections  # this rejection makes (past + 1)
            new_rejected = state.finalized_rejected + (
                RejectedHypothesis(
                    hyp_id=state.current_hypothesis.hyp_id,
                    summary=state.current_hypothesis.summary,
                    rejection_reason=", ".join(state.current_critique.reasons),
                    critic_reasons=list(state.current_critique.reasons),
                    judge_rationale=_judge_rationale(events_so_far, state.current_hypothesis.hyp_id),
                ),
            )
            reject_ev = HypothesisRejectedEvent(
                ts=now,
                turn=state.current_turn,
                hyp_id=state.current_hypothesis.hyp_id,
                topic_id=state.current_topic.topic_id,
                reason=", ".join(state.current_critique.reasons) or "judge upheld critique",
            )
            # cycles counts past rejections for this topic since its last
            # topic_selected; this rejection makes the count (cycles + 1).
            # If still under cap, retry the same topic; else move on.
            attempts_so_far = cycles + 1
            should_retry = attempts_so_far < state.budget.max_critic_cycles_per_topic
            if should_retry:
                return (
                    replace(
                        state,
                        phase="retry_topic",
                        finalized_rejected=new_rejected,
                        retry_topic_id=state.current_topic.topic_id,
                        current_hypothesis=None,
                        current_critique=None,
                        current_judge_valid=None,
                        current_facets=(),
                        next_specialist_idx=0,
                    ),
                    [reject_ev],
                )
            # Move on to next topic
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
                [reject_ev],
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
    validate: bool = False,
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

    if validate and hyp_input.characterization is not None:
        from fermdocs_hypothesis.validators import validate_hypothesis_output
        try:
            from fermdocs.domain.process_priors import cached_priors
            priors = cached_priors()
        except Exception:
            priors = None
        output = validate_hypothesis_output(
            output,
            upstream=hyp_input.characterization,
            drop_unknown_citations=True,
            priors=priors,
            organism=hyp_input.organism,
        )

    return RunResult(output=output, state=state, events=events)


def resume_stage(
    *,
    hyp_input: HypothesisInput,
    hooks: RunnerHooks,
    global_md_path: Path,
    diagnosis_id: UUID,
    answers: list[tuple[str, str]],
    hypothesis_version: str = "v0.1.0",
    model_name: str = "stub",
    provider: Literal["anthropic", "gemini", "stub"] = "stub",
    budget: BudgetSnapshot | None = None,
    past_insights: PastInsightStore | None = None,
    now_factory=lambda: datetime.now(timezone.utc),
    validate: bool = False,
) -> RunResult:
    """Resume a paused hypothesis stage with human answers to open questions.

    Reads existing events from `global_md_path`, emits HumanInputReceivedEvent
    + QuestionResolvedEvent for each (qid, resolution) in `answers`, then
    drives one more round of debate. The new round inherits prior finalized
    finals + rejected, prior used_topic_ids, prior counters; budget is fresh
    (caller can pass a tighter cap to bound the resume).

    HITL contract: this is the v0.1 minimal-mode resume. Each call runs ONE
    additional debate round (the hard turn cap is respected within budget).
    Multi-round HITL (re-prompt for more questions) is left to the CLI loop.
    """
    from fermdocs_hypothesis.events import (
        HumanInputReceivedEvent,
        HypothesisAcceptedEvent,
        HypothesisRejectedEvent,
        QuestionResolvedEvent,
        StageStartedEvent,
    )
    from fermdocs_hypothesis.event_log import read_events

    past_insights = past_insights or NullPastInsightStore()
    seed_topics_tuple = tuple(hyp_input.seed_topics)
    initial_budget = budget or BudgetSnapshot()

    # Read prior events
    prior_events: list[Event] = read_events(global_md_path) if global_md_path.exists() else []

    # Reconstruct counters + finalized lists from prior events
    prior_finals: list[FinalHypothesis] = []
    prior_rejected: list[RejectedHypothesis] = []
    used_topic_ids: list[str] = []
    facet_counter = 0
    hyp_counter = 0
    qid_counter = 0
    accepted_ids: set[str] = set()
    rejected_id_with_topic: list[tuple[str, str]] = []

    # Build hyp_id -> HypothesisSynthesizedEvent map for reconstructing finals
    hyp_synth: dict[str, HypothesisSynthesizedEvent] = {}
    crit_for_hyp: dict[str, CritiqueFiledEvent] = {}
    judge_for_hyp: dict[str, JudgeRulingEvent] = {}
    rejection_event_for_hyp: dict[str, HypothesisRejectedEvent] = {}

    for ev in prior_events:
        if isinstance(ev, FacetContributedEvent):
            n = int(ev.facet_id.removeprefix("FCT-"))
            facet_counter = max(facet_counter, n)
        elif isinstance(ev, HypothesisSynthesizedEvent):
            n = int(ev.hyp_id.removeprefix("H-"))
            hyp_counter = max(hyp_counter, n)
            hyp_synth[ev.hyp_id] = ev
        elif isinstance(ev, CritiqueFiledEvent):
            crit_for_hyp[ev.hyp_id] = ev
        elif isinstance(ev, JudgeRulingEvent):
            judge_for_hyp[ev.hyp_id] = ev
        elif isinstance(ev, HypothesisAcceptedEvent):
            accepted_ids.add(ev.hyp_id)
        elif isinstance(ev, HypothesisRejectedEvent):
            rejection_event_for_hyp[ev.hyp_id] = ev
        elif isinstance(ev, TopicSelectedEvent):
            if ev.topic_id not in used_topic_ids:
                used_topic_ids.append(ev.topic_id)
        elif isinstance(ev, QuestionAddedEvent):
            n = int(ev.qid.removeprefix("Q-"))
            qid_counter = max(qid_counter, n)

    # Rebuild prior_finals
    for hyp_id in accepted_ids:
        synth = hyp_synth.get(hyp_id)
        crit = crit_for_hyp.get(hyp_id)
        judge = judge_for_hyp.get(hyp_id)
        if synth is None or crit is None:
            continue
        synth_hyp = HypothesisFull(
            hyp_id=synth.hyp_id,
            summary=synth.summary,
            facet_ids=list(synth.facet_ids),
            cited_finding_ids=list(synth.cited_finding_ids),
            cited_narrative_ids=list(synth.cited_narrative_ids),
            cited_trajectories=list(synth.cited_trajectories),
            affected_variables=list(synth.affected_variables),
            confidence=synth.confidence,
            confidence_basis=synth.confidence_basis,
        )
        crit_full = CritiqueFull(
            hyp_id=crit.hyp_id,
            flag=crit.flag,
            reasons=list(crit.reasons),
            tool_calls_used=crit.tool_calls_used,
        )
        judge_valid = judge.criticism_valid if judge else False
        prior_finals.append(_build_final(synth_hyp, crit_full, judge_valid))

    # Rebuild prior_rejected
    for hyp_id, rej in rejection_event_for_hyp.items():
        synth = hyp_synth.get(hyp_id)
        crit = crit_for_hyp.get(hyp_id)
        if synth is None:
            continue
        prior_rejected.append(
            RejectedHypothesis(
                hyp_id=hyp_id,
                summary=synth.summary,
                rejection_reason=rej.reason,
                critic_reasons=list(crit.reasons) if crit else [],
                judge_rationale=judge_for_hyp[hyp_id].rationale if hyp_id in judge_for_hyp else "",
            )
        )

    # Build state for resume
    state = RunnerState(
        phase="init",
        budget=initial_budget,
        seed_topics=seed_topics_tuple,
        finalized_finals=tuple(prior_finals),
        finalized_rejected=tuple(prior_rejected),
        used_topic_ids=tuple(used_topic_ids),
        facet_counter=facet_counter,
        hyp_counter=hyp_counter,
        qid_counter=qid_counter,
    )

    observer = Observer(global_md_path)
    meter = TokenMeter()
    events: list[Event] = list(prior_events)

    # Emit a fresh stage_started so resume rounds are visually distinct in
    # the event log.
    started = StageStartedEvent(
        ts=now_factory(),
        turn=0,
        input_diagnosis_id=str(diagnosis_id),
        budget=initial_budget,
    )
    new_events: list[Event] = [started]

    # Emit human_input_received + question_resolved for each answer
    for qid, resolution in answers:
        if not resolution.strip():
            continue
        new_events.append(
            HumanInputReceivedEvent(
                ts=now_factory(),
                turn=0,
                input_type="answer",
                payload={"qid": qid, "resolution": resolution},
            )
        )
        new_events.append(
            QuestionResolvedEvent(
                ts=now_factory(),
                turn=0,
                qid=qid,
                resolution=resolution,
            )
        )

    observer.write_many(new_events)
    events.extend(new_events)

    # Loop
    safety_counter = 0
    SAFETY_MAX = 1000
    while state.phase != "done":
        safety_counter += 1
        if safety_counter > SAFETY_MAX:
            state = replace(state, phase="exit", exit_reason="budget_exhausted")
            continue
        next_state, more_events = step(
            state,
            hooks,
            events_so_far=events,
            meter=meter,
            now=now_factory(),
        )
        if more_events:
            observer.write_many(more_events)
            events.extend(more_events)
        state = next_state

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

    if validate and hyp_input.characterization is not None:
        from fermdocs_hypothesis.validators import validate_hypothesis_output
        try:
            from fermdocs.domain.process_priors import cached_priors
            priors = cached_priors()
        except Exception:
            priors = None
        output = validate_hypothesis_output(
            output,
            upstream=hyp_input.characterization,
            drop_unknown_citations=True,
            priors=priors,
            organism=hyp_input.organism,
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
