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

import re

from fermdocs_characterize.schema import Severity
from fermdocs_diagnose.schema import (
    AnalysisClaim,
    ConfidenceBasis,
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


_SPEC_LANGUAGE_RE = re.compile(
    r"\b("
    r"sigma|"
    r"nominal|"
    r"specification|"
    r"spec|"
    r"setpoint|"
    r"set\s*point|"
    r"schema"
    r")\b",
    re.IGNORECASE,
)
"""Vocabulary fingerprint for 'this isn't a biological anomaly, it's a
schema/spec interpretation issue'. Used by `_is_spec_only_failure` to
detect when a FailureClaim's evidence is purely a measured-vs-nominal
delta.

Conservative on purpose: matches whole words only so 'biospecification'
or 'schemata' don't false-positive. The IndPenSim regression's three
failures all use 'nominal specification' or 'sigma' verbatim.
"""


def _is_spec_only_failure(f: FailureClaim) -> bool:
    """True when a FailureClaim is grounded ONLY in spec-mismatch logic,
    not in real biological/operational evidence.

    The IndPenSim case (May 2026): when `unknown_process` is set and
    recipe-specific priors are unavailable, the diagnose agent computes
    `(measured - nominal) / std_dev` against generic schema specs and
    reports the result as a FailureClaim. For unknown_process bundles
    these specs are usually wrong (e.g. biomass nominal = inoculum
    density, not steady-state) — so the 'failure' is a schema artifact,
    not a real anomaly. The synthesizer + critic correctly reject
    debating it; the seed topic just wastes turns.

    Predicate (all three must hold):
      1. confidence_basis == 'schema_only' (not grounded in process priors
         or cross-run data)
      2. No narrative or trajectory citations (no operator-witnessed event,
         no time-series anomaly to anchor on)
      3. Summary contains spec-vocabulary words (nominal/spec/sigma/etc.)

    A failure that has narrative or trajectory citations is kept even
    when its summary mentions specs — the real evidence is what makes it
    debatable. We only filter when spec-talk is the sole grounding.
    """
    if f.confidence_basis != ConfidenceBasis.SCHEMA_ONLY:
        return False
    if f.cited_narrative_ids or f.cited_trajectories:
        return False
    if not _SPEC_LANGUAGE_RE.search(f.summary):
        return False
    return True


def _is_spec_only_open_question(q: OpenQuestion) -> bool:
    """Open questions framed as 'what is the correct spec for X?' suffer
    the same problem — they seed topics that route specialists into
    arguing about data plumbing instead of biology.

    Detection mirrors `_is_spec_only_failure` but on the question text:
    spec-vocabulary present and only finding citations (no narratives).
    """
    if q.cited_narrative_ids:
        return False
    text = f"{q.question} {q.why_it_matters or ''}"
    return bool(_SPEC_LANGUAGE_RE.search(text))


def extract_seed_topics(diag: DiagnosisOutput) -> list[SeedTopic]:
    """Project every claim/question into a SeedTopic. Topic IDs are
    assigned in deterministic order: failures, then analyses, then trends,
    then open questions.

    Two filters applied:

      1. Analysis claims whose kind is in _SUPPRESSED_ANALYSIS_KINDS
         (data_quality_caveat / spec_alignment) — meta-observations
         about the data itself, not candidates for hypothesis-debate.

      2. Failures and open questions that are SPEC-ONLY: their evidence
         is purely 'measured value differs from nominal spec' with no
         narrative or trajectory anchoring. These derail debate when
         specs are misaligned (the common case for unknown_process
         bundles like IndPenSim) — the synthesizer correctly rejects
         debating spec-vs-measurement deltas, so the seed topic just
         wastes turns. See `_is_spec_only_failure`.

    Failures that mix spec language WITH narrative/trajectory citations
    are kept — the real evidence makes them debatable.
    """
    topics: list[SeedTopic] = []
    counter = 0

    for f in diag.failures:
        if _is_spec_only_failure(f):
            continue
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
        if _is_spec_only_open_question(q):
            continue
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
