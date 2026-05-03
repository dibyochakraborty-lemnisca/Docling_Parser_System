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
    Event,
    FacetContributedEvent,
    HypothesisAcceptedEvent,
    HypothesisRejectedEvent,
    HypothesisSynthesizedEvent,
    JudgeRulingEvent,
    QuestionAddedEvent,
    QuestionResolvedEvent,
    TopicSelectedEvent,
)
from fermdocs_hypothesis.schema import (
    FacetSummary,
    HypothesisRef,
    OpenQuestionRef,
    SpecialistRole,
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
