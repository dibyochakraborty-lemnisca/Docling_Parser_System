"""Derived state from the event log.

Plan ref: plans/2026-05-03-hypothesis-debate-v0.md §3, §8.

All functions in this module are pure: same event list → same derived state.
This is the lever that lets us treat global.md as the single source of truth
without per-agent persistent yamls.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Iterable

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
    QuestionResolvedEvent,
    TopicSelectedEvent,
)
from fermdocs_hypothesis.schema import (
    AttemptRecord,
    FacetSummary,
    HypothesisRef,
    LessonsDigest,
    OpenQuestionRef,
    SpecialistRole,
    TopicHistoryEntry,
    TurnOutcome,
)


def open_questions(events: Iterable[Event]) -> list[OpenQuestionRef]:
    """Reconstruct the open-questions ledger from events.

    Resolved questions are returned with resolved=True and their resolution
    string filled in. Order matches insertion order.
    """
    questions: dict[str, OpenQuestionRef] = {}
    for ev in events:
        if isinstance(ev, QuestionAddedEvent):
            questions[ev.qid] = OpenQuestionRef(
                qid=ev.qid,
                question=ev.question,
                raised_by=ev.raised_by,
                tags=list(ev.tags),
                resolved=False,
            )
        elif isinstance(ev, QuestionResolvedEvent):
            existing = questions.get(ev.qid)
            if existing is None:
                continue
            questions[ev.qid] = existing.model_copy(
                update={"resolved": True, "resolution": ev.resolution}
            )
    return list(questions.values())


def topic_attempt_counts(events: Iterable[Event]) -> Counter[str]:
    """How many times each topic_id has been selected. Used by ranker for
    times_attempted penalty.
    """
    return Counter(ev.topic_id for ev in events if isinstance(ev, TopicSelectedEvent))


def topic_rejection_counts(events: Iterable[Event]) -> Counter[str]:
    """How many times each topic_id had a hypothesis rejected. Used by
    ranker for times_rejected penalty.

    Maps via HypothesisRejectedEvent.topic_id which the runner sets when
    emitting the event.
    """
    return Counter(
        ev.topic_id
        for ev in events
        if isinstance(ev, HypothesisRejectedEvent)
    )


def critic_cycles_for_current_topic(
    events: Iterable[Event], topic_id: str
) -> int:
    """Count critic-rejection cycles for the currently-active topic since the
    last time this topic_id was selected.

    Scans backward to the most recent TopicSelectedEvent for topic_id, then
    counts HypothesisRejectedEvent matching that topic_id after.
    """
    events_list = list(events)
    last_selected_idx = -1
    for i in range(len(events_list) - 1, -1, -1):
        ev = events_list[i]
        if isinstance(ev, TopicSelectedEvent) and ev.topic_id == topic_id:
            last_selected_idx = i
            break
    if last_selected_idx == -1:
        return 0
    return sum(
        1
        for ev in events_list[last_selected_idx + 1 :]
        if isinstance(ev, HypothesisRejectedEvent) and ev.topic_id == topic_id
    )


def accepted_hypotheses(events: Iterable[Event]) -> list[HypothesisRef]:
    """List of accepted-hypothesis refs in event order."""
    summaries = synthesized_summaries(events)
    out: list[HypothesisRef] = []
    for ev in events:
        if isinstance(ev, HypothesisAcceptedEvent):
            out.append(
                HypothesisRef(
                    hyp_id=ev.hyp_id,
                    summary=summaries.get(ev.hyp_id, ""),
                )
            )
    return out


def synthesized_summaries(events: Iterable[Event]) -> dict[str, str]:
    """hyp_id -> short summary, for use building HypothesisRefs."""
    return {
        ev.hyp_id: ev.summary
        for ev in events
        if isinstance(ev, HypothesisSynthesizedEvent)
    }


def facets_for_current_topic(
    events: Iterable[Event], topic_id: str
) -> list[FacetSummary]:
    """All facets contributed against topic_id since its most recent selection.

    Returns FacetSummary (the compact form for SpecialistView's
    prior_facets_this_topic).
    """
    events_list = list(events)
    last_selected_idx = -1
    for i in range(len(events_list) - 1, -1, -1):
        ev = events_list[i]
        if isinstance(ev, TopicSelectedEvent) and ev.topic_id == topic_id:
            last_selected_idx = i
            break
    if last_selected_idx == -1:
        return []
    return [
        FacetSummary(
            facet_id=ev.facet_id,
            specialist=ev.specialist,
            summary=ev.summary,
            confidence=ev.confidence,
        )
        for ev in events_list[last_selected_idx + 1 :]
        if isinstance(ev, FacetContributedEvent) and ev.topic_id == topic_id
    ]


def last_turn_outcome(events: Iterable[Event]) -> TurnOutcome | None:
    """Outcome of the most recently-completed turn, for OrchestratorView.

    A turn is "complete" when it ended in HypothesisAcceptedEvent,
    HypothesisRejectedEvent, or its TopicSelectedEvent had no synthesis
    follow-up by stage exit.
    """
    events_list = list(events)
    for ev in reversed(events_list):
        if isinstance(ev, HypothesisAcceptedEvent):
            return TurnOutcome(
                turn=ev.turn,
                topic_id=ev.topic_id,
                outcome="accepted",
                hyp_id=ev.hyp_id,
            )
        if isinstance(ev, HypothesisRejectedEvent):
            return TurnOutcome(
                turn=ev.turn,
                topic_id=ev.topic_id,
                outcome="rejected",
                hyp_id=ev.hyp_id,
            )
    return None


def specialist_domain_tags(role: SpecialistRole) -> set[str]:
    """Coarse tag-set per specialist for SpecialistView filtering of
    open_questions and findings. Hand-curated; keep in sync with persona
    specs in Stage 2.
    """
    if role == "kinetics":
        return {
            "biomass",
            "growth",
            "mu",
            "yield",
            "substrate",
            "glucose",
            "kinetics",
            "phase",
        }
    if role == "mass_transfer":
        return {
            "DO",
            "do",
            "kLa",
            "kla",
            "OUR",
            "CER",
            "mixing",
            "foam",
            "pH",
            "temperature",
            "T",
            "agitation",
        }
    if role == "metabolic":
        return {
            "acetate",
            "ethanol",
            "byproduct",
            "induction",
            "stress",
            "metabolic",
            "PAA",
            "carbon",
            "nitrogen",
            "lysis",
        }
    return set()


def latest_judge_ruling_for(events: Iterable[Event], hyp_id: str) -> JudgeRulingEvent | None:
    """Most recent JudgeRulingEvent for hyp_id, if any."""
    last: JudgeRulingEvent | None = None
    for ev in events:
        if isinstance(ev, JudgeRulingEvent) and ev.hyp_id == hyp_id:
            last = ev
    return last


def topic_history(events: Iterable[Event], topic_id: str) -> TopicHistoryEntry:
    """All synthesizer→critic→judge cycles on `topic_id`, oldest-first.

    Walks the event log once, grouping events by hyp_id under topic_id.
    A cycle is anchored on HypothesisSynthesizedEvent; critic/judge events
    are attached if present. Status reflects the topic's terminal state at
    projection time.

    Used by the projector to populate `previous_attempts` on retrying
    agents — the synthesizer must address each prior critic_reasons entry
    rather than re-emit the same overreach.
    """
    events_list = list(events)

    topic_summary = ""
    for ev in events_list:
        if isinstance(ev, TopicSelectedEvent) and ev.topic_id == topic_id:
            topic_summary = ev.summary

    # hyp_id → AttemptRecord-in-progress (mutable scratch dict)
    attempts_by_hyp: dict[str, dict] = {}
    hyp_order: list[str] = []  # preserve synthesis order

    accepted = False
    for ev in events_list:
        if isinstance(ev, HypothesisSynthesizedEvent) and ev.topic_id == topic_id:
            attempts_by_hyp[ev.hyp_id] = {
                "hyp_id": ev.hyp_id,
                "hypothesis_summary": ev.summary,
                "critic_flag": None,
                "critic_reasons": [],
                "judge_ruling": None,
                "judge_rationale": None,
                "human_input": None,
            }
            hyp_order.append(ev.hyp_id)
        elif isinstance(ev, CritiqueFiledEvent) and ev.hyp_id in attempts_by_hyp:
            attempts_by_hyp[ev.hyp_id]["critic_flag"] = ev.flag
            attempts_by_hyp[ev.hyp_id]["critic_reasons"] = list(ev.reasons)
        elif isinstance(ev, JudgeRulingEvent) and ev.hyp_id in attempts_by_hyp:
            attempts_by_hyp[ev.hyp_id]["judge_ruling"] = (
                "valid" if ev.criticism_valid else "invalid"
            )
            attempts_by_hyp[ev.hyp_id]["judge_rationale"] = ev.rationale
        elif isinstance(ev, HypothesisAcceptedEvent) and ev.topic_id == topic_id:
            accepted = True
        elif isinstance(ev, HypothesisRejectedEvent) and ev.topic_id == topic_id:
            pass  # already captured via critic/judge events

    attempts = [AttemptRecord(**attempts_by_hyp[hid]) for hid in hyp_order]

    if accepted:
        status = "accepted"
    elif not attempts:
        status = "deferred"
    else:
        # Runner enforces max_critic_cycles_per_topic; we expose the raw
        # state and let downstream consumers decide. "in_progress" is
        # accurate from the projection's pure-function perspective.
        status = "in_progress"

    return TopicHistoryEntry(
        topic_id=topic_id,
        summary=topic_summary,
        attempts=attempts,
        status=status,
    )


def all_critic_reasons(
    events: Iterable[Event], *, limit: int = 20
) -> list[str]:
    """Last `limit` critic reasons across ALL topics, newest-last.

    Input to LessonsSummarizerAgent. Capping protects token budget — older
    reasons matter less than recent recurring patterns.
    """
    reasons: list[str] = []
    for ev in events:
        if isinstance(ev, CritiqueFiledEvent):
            reasons.extend(ev.reasons)
    if len(reasons) > limit:
        reasons = reasons[-limit:]
    return reasons


def latest_lessons_digest(events: Iterable[Event]) -> LessonsDigest | None:
    """Most recent LessonsSummarizedEvent projected as a LessonsDigest, or
    None if the summarizer hasn't run yet.

    `computed_at_event_idx` is the event-list index at which the summary
    landed — used by the runner as a cache key alongside source_reason_count.
    """
    events_list = list(events)
    last_idx = -1
    last_ev: LessonsSummarizedEvent | None = None
    for i, ev in enumerate(events_list):
        if isinstance(ev, LessonsSummarizedEvent):
            last_idx = i
            last_ev = ev
    if last_ev is None:
        return None
    return LessonsDigest(
        digest=last_ev.digest,
        source_reason_count=last_ev.source_reason_count,
        computed_at_event_idx=last_idx,
    )
