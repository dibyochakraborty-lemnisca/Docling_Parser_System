"""Deterministic topic ranker.

Plan ref: plans/2026-05-03-hypothesis-debate-v0.md §8.

Score formula:
    score(topic) =
        severity_weight(topic.source_type|severity) * topic.priority
      + 0.3 * citation_density(topic)
      + 0.2 * unresolved_question_overlap(topic)
      - 0.5 * times_attempted(topic)
      - 1.0 * times_rejected_by_judge(topic)

Returns top-K (K=3 by default) sorted by score descending. Ties broken by
topic_id ascending so the order is fully deterministic — no LLM jitter
upstream of agent calls.

Open questions enter the ranker as synthetic topics (priority derived from
their tag overlap with seed topics).
"""

from __future__ import annotations

from collections.abc import Iterable

from fermdocs_characterize.schema import Severity
from fermdocs_hypothesis.events import Event
from fermdocs_hypothesis.schema import (
    OpenQuestionRef,
    RankedTopic,
    SeedTopic,
    TopicSourceType,
)
from fermdocs_hypothesis.state import (
    open_questions,
    topic_attempt_counts,
    topic_rejection_counts,
)

DEFAULT_K = 3
TIE_EPSILON = 0.05

_SEVERITY_WEIGHT = {
    Severity.MAJOR: 1.0,
    Severity.MINOR: 0.6,
    Severity.INFO: 0.3,
}


def _citation_density(topic: SeedTopic) -> float:
    """Normalized count of distinct citations attached to the topic.

    Caps at 1.0 (5+ citations). Encourages topics with stronger evidence.
    """
    n = (
        len(topic.cited_finding_ids)
        + len(topic.cited_narrative_ids)
        + len(topic.cited_trajectories)
    )
    return min(n / 5.0, 1.0)


def _unresolved_question_overlap(
    topic: SeedTopic, unresolved: list[OpenQuestionRef]
) -> float:
    """Fraction of unresolved open-question tags that overlap with the
    topic's affected_variables. 0.0 when there are no unresolved questions.
    """
    if not unresolved:
        return 0.0
    topic_vars = {v.lower() for v in topic.affected_variables}
    if not topic_vars:
        return 0.0
    matches = sum(
        1
        for q in unresolved
        if any(t.lower() in topic_vars for t in q.tags)
    )
    return matches / len(unresolved)


def _score_seed(
    topic: SeedTopic,
    *,
    unresolved: list[OpenQuestionRef],
    attempts: int,
    rejections: int,
) -> float:
    sev = _SEVERITY_WEIGHT.get(topic.severity, 0.3)
    return (
        sev * topic.priority
        + 0.3 * _citation_density(topic)
        + 0.2 * _unresolved_question_overlap(topic, unresolved)
        - 0.5 * attempts
        - 1.0 * rejections
    )


def _synthetic_topic_from_question(q: OpenQuestionRef) -> SeedTopic:
    """Project an unresolved question into the topic shape so the ranker
    can score it against seed topics on the same scale.
    """
    return SeedTopic(
        topic_id=_qid_to_topic_id(q.qid),
        summary=q.question,
        source_type=TopicSourceType.OPEN_QUESTION,
        source_id=q.qid,
        affected_variables=list(q.tags),
        severity=Severity.MINOR,
        priority=0.4,
    )


def _qid_to_topic_id(qid: str) -> str:
    """Q-NNNN -> T-9NNN so synthetic topics never collide with seed T-NNNN.

    Fixed offset 9000; if seed_topics ever exceed 8999 entries we have a
    bigger problem than namespace collision.
    """
    n = int(qid.removeprefix("Q-"))
    return f"T-{9000 + n:04d}"


def rank_topics(
    seed_topics: Iterable[SeedTopic],
    events: Iterable[Event],
    *,
    k: int = DEFAULT_K,
) -> list[RankedTopic]:
    """Score seed topics + synthetic-question topics, return top-K.

    Deterministic: same inputs → same output (sorted score desc, then
    topic_id asc).
    """
    events_list = list(events)
    unresolved = [q for q in open_questions(events_list) if not q.resolved]
    attempts = topic_attempt_counts(events_list)
    rejections = topic_rejection_counts(events_list)

    candidates: list[tuple[SeedTopic, bool]] = [
        (t, False) for t in seed_topics
    ]
    for q in unresolved:
        candidates.append((_synthetic_topic_from_question(q), True))

    scored: list[RankedTopic] = []
    for topic, synthetic in candidates:
        score = _score_seed(
            topic,
            unresolved=unresolved,
            attempts=attempts.get(topic.topic_id, 0),
            rejections=rejections.get(topic.topic_id, 0),
        )
        scored.append(
            RankedTopic(
                topic_id=topic.topic_id,
                summary=topic.summary,
                score=round(score, 6),
                is_synthetic=synthetic,
            )
        )

    scored.sort(key=lambda r: (-r.score, r.topic_id))
    return scored[:k]


def is_tie(top: list[RankedTopic], epsilon: float = TIE_EPSILON) -> bool:
    """True when top-2 scores are within epsilon — runner uses this to
    decide whether to ask the orchestrator-LLM to break the tie or
    auto-pick #1.
    """
    if len(top) < 2:
        return False
    return abs(top[0].score - top[1].score) < epsilon
