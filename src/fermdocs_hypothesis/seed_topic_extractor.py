"""DiagnosisOutput → list[SeedTopic].

Plan ref: plans/2026-05-03-hypothesis-debate-v0.md §9.

Each failure / analysis / trend / open_question becomes a SeedTopic. The
ranker (§8) further scores them; this module just produces the
candidate set with a reasonable initial priority.

Priority heuristic (0.0-1.0):
  - failures: 0.5 + 0.4 * severity_weight + 0.1 * citation_count_norm
  - analyses: 0.4 + 0.1 * citation_count_norm
  - trends:   0.5 + 0.1 * citation_count_norm
  - open_qs:  0.4 (constant; ranker boosts via overlap)

Severity weight: critical=1.0, major=0.7, minor=0.4, info=0.2.
"""

from __future__ import annotations

from fermdocs_characterize.schema import Severity
from fermdocs_diagnose.schema import (
    AnalysisClaim,
    DiagnosisOutput,
    FailureClaim,
    OpenQuestion,
    TrendClaim,
)
from fermdocs_hypothesis.schema import (
    SeedTopic,
    TopicSourceType,
)

_SEV_WEIGHT = {
    Severity.CRITICAL: 1.0,
    Severity.MAJOR: 0.7,
    Severity.MINOR: 0.4,
    Severity.INFO: 0.2,
}


_SUPPRESSED_ANALYSIS_KINDS: frozenset[str] = frozenset(
    {"data_quality_caveat", "spec_alignment"}
)
"""Analysis kinds that describe meta-properties of the data / system
configuration rather than hypothesizable claims about the experiment.

  - data_quality_caveat: "the data is sparse / has gaps / is malformed"
  - spec_alignment: "process specs are missing / nominal != measured"

Both bias specialists toward arguing about data plumbing instead of
biology. Observed regression: when only data_quality_caveat was
suppressed, the diagnose agent learned to dodge by emitting the same
meta-claim under spec_alignment instead. The fix is to suppress the
whole "meta" kind family.

`cross_run_observation` and `phase_characterization` are still
hypothesizable — cross-run variation and phase boundaries are real
engineering questions — so they pass through.
"""


def extract_seed_topics(diag: DiagnosisOutput) -> list[SeedTopic]:
    """Project every claim/question into a SeedTopic. Topic IDs are
    assigned in deterministic order: failures, then analyses, then trends,
    then open questions.

    Filters out analysis claims whose kind is in _SUPPRESSED_ANALYSIS_KINDS
    (data_quality_caveat) — these are meta-observations about the data
    itself, not candidates for hypothesis-debate. Letting them seed topics
    derails specialists into framing "the registry doesn't match" or "the
    data is sparse" as the central question, when the actual document
    is asking biological questions.
    """
    topics: list[SeedTopic] = []
    counter = 0

    for f in diag.failures:
        counter += 1
        topics.append(_from_failure(f, counter))
    for a in diag.analysis:
        if a.kind in _SUPPRESSED_ANALYSIS_KINDS:
            continue
        counter += 1
        topics.append(_from_analysis(a, counter))
    for t in diag.trends:
        counter += 1
        topics.append(_from_trend(t, counter))
    for q in diag.open_questions:
        counter += 1
        topics.append(_from_open_question(q, counter))

    return topics


def _topic_id(n: int) -> str:
    return f"T-{n:04d}"


def _norm_citations(*lists) -> float:
    n = sum(len(lst or []) for lst in lists)
    return min(n / 5.0, 1.0)


def _from_failure(f: FailureClaim, idx: int) -> SeedTopic:
    sev = _normalize_severity(f.severity)
    priority = 0.5 + 0.4 * _SEV_WEIGHT.get(sev, 0.4) + 0.1 * _norm_citations(
        f.cited_finding_ids, f.cited_narrative_ids, f.cited_trajectories
    )
    return SeedTopic(
        topic_id=_topic_id(idx),
        summary=f.summary[:200],
        source_type=TopicSourceType.FAILURE,
        source_id=f.claim_id,
        cited_finding_ids=list(f.cited_finding_ids),
        cited_narrative_ids=list(f.cited_narrative_ids),
        cited_trajectories=list(f.cited_trajectories),
        affected_variables=list(f.affected_variables),
        severity=sev,
        priority=min(priority, 1.0),
    )


def _from_analysis(a: AnalysisClaim, idx: int) -> SeedTopic:
    priority = 0.4 + 0.1 * _norm_citations(a.cited_finding_ids, a.cited_narrative_ids)
    # Analyses don't carry severity; map the kind heuristically.
    sev = _severity_from_analysis_kind(a.kind)
    return SeedTopic(
        topic_id=_topic_id(idx),
        summary=a.summary[:200],
        source_type=TopicSourceType.ANALYSIS,
        source_id=a.claim_id,
        cited_finding_ids=list(a.cited_finding_ids),
        cited_narrative_ids=list(a.cited_narrative_ids),
        cited_trajectories=[],
        affected_variables=list(a.affected_variables),
        severity=sev,
        priority=min(priority, 1.0),
    )


def _from_trend(t: TrendClaim, idx: int) -> SeedTopic:
    priority = 0.5 + 0.1 * _norm_citations(
        t.cited_finding_ids, t.cited_narrative_ids, t.cited_trajectories
    )
    return SeedTopic(
        topic_id=_topic_id(idx),
        summary=t.summary[:200],
        source_type=TopicSourceType.TREND,
        source_id=t.claim_id,
        cited_finding_ids=list(t.cited_finding_ids),
        cited_narrative_ids=list(t.cited_narrative_ids),
        cited_trajectories=list(t.cited_trajectories),
        affected_variables=list(t.affected_variables),
        severity=Severity.MINOR,
        priority=min(priority, 1.0),
    )


def _from_open_question(q: OpenQuestion, idx: int) -> SeedTopic:
    return SeedTopic(
        topic_id=_topic_id(idx),
        summary=q.question[:200],
        source_type=TopicSourceType.OPEN_QUESTION,
        source_id=q.question_id,
        cited_finding_ids=list(q.cited_finding_ids),
        cited_narrative_ids=list(q.cited_narrative_ids),
        cited_trajectories=[],
        affected_variables=[],
        severity=Severity.MINOR,
        priority=0.4,
    )


def _normalize_severity(sev) -> Severity:
    """Diagnose schema permits CRITICAL but the hypothesis ranker only
    knows MAJOR/MINOR/INFO. Map CRITICAL → MAJOR for ranker scoring (the
    weight table here keeps CRITICAL distinct internally for priority
    tilt; ranker will see whatever Severity is on SeedTopic).
    """
    return sev


def _severity_from_analysis_kind(kind: str) -> Severity:
    if kind == "data_quality_caveat":
        return Severity.MINOR
    if kind == "cross_run_observation":
        return Severity.MAJOR
    return Severity.MINOR
