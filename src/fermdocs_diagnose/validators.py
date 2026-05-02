"""Cross-output validators for DiagnosisOutput.

Plan ref: plans/2026-05-02-diagnosis-agent.md §3.

Hard rejection vs soft enforcement:
  - Hard (raises): cited_finding_ids reference a real Finding; cited_trajectories
    reference a real (run_id, variable) pair in the upstream summary.
  - Soft (logs warning, mutates output): provenance downgrade under UNKNOWN_*
    flags; forbidden causal phrases in summaries.

Why split? A reject-and-fail loop on a multi-claim output throws away an entire
LLM batch when one claim drifts. Hard rejection is reserved for issues that
break the data contract downstream agents depend on (unknown citations).
Soft enforcement covers what the prompt should already prevent and the
validator catches as a drift backstop.
"""

from __future__ import annotations

import logging
import re
from collections.abc import Iterable

from fermdocs_characterize.flags import ProcessFlag
from fermdocs_characterize.schema import CharacterizationOutput
from fermdocs_diagnose.schema import (
    AnalysisClaim,
    BaseClaim,
    ConfidenceBasis,
    DiagnosisOutput,
    FailureClaim,
    OpenQuestion,
    TrendClaim,
)

_log = logging.getLogger(__name__)

FORBIDDEN_CAUSAL_PHRASES = (
    "because",
    "due to",
    "caused by",
    "resulted in",
    "leading to",
)

UNKNOWN_FLAGS = frozenset(
    {ProcessFlag.UNKNOWN_PROCESS, ProcessFlag.UNKNOWN_ORGANISM}
)


class CitationIntegrityError(ValueError):
    """Raised when a claim cites an ID that does not exist upstream."""


def validate_diagnosis(
    output: DiagnosisOutput,
    *,
    upstream: CharacterizationOutput,
    flags: Iterable[ProcessFlag] = (),
    drop_unknown_citations: bool = True,
) -> DiagnosisOutput:
    """Apply cross-output validation and soft enforcement.

    Args:
        output: The DiagnosisOutput as constructed (already passed schema-level
            validation).
        upstream: The CharacterizationOutput the diagnosis was built from.
            Used to verify cited finding/trajectory IDs.
        flags: Process flags from the AgentContext that fed this run.
        drop_unknown_citations: When True (the default), claims with at least
            one unknown cited_finding_id are dropped (logged) rather than
            raising. When False, an unknown citation raises
            CitationIntegrityError. Drop-mode is the production default; raise-
            mode is used by tests asserting strict integrity.

    Returns:
        A new DiagnosisOutput with soft enforcements applied (provenance
        downgrades, forbidden-phrase warnings logged, optionally dropped
        claims).
    """
    finding_ids = {f.finding_id for f in upstream.findings}
    trajectory_keys = {(t.run_id, t.variable) for t in upstream.trajectories}
    flags = frozenset(flags)

    failures = list(_filter_claims(
        output.failures,
        finding_ids=finding_ids,
        kind_label="failure",
        drop=drop_unknown_citations,
    ))
    analysis = list(_filter_claims(
        output.analysis,
        finding_ids=finding_ids,
        kind_label="analysis",
        drop=drop_unknown_citations,
    ))
    trends = list(_filter_trends(
        output.trends,
        finding_ids=finding_ids,
        trajectory_keys=trajectory_keys,
        drop=drop_unknown_citations,
    ))
    open_questions = list(_filter_questions(
        output.open_questions,
        finding_ids=finding_ids,
        drop=drop_unknown_citations,
    ))

    failures = [_apply_soft_enforcement(c, flags) for c in failures]
    trends = [_apply_soft_enforcement(c, flags) for c in trends]
    analysis = [_apply_soft_enforcement(c, flags) for c in analysis]

    return output.model_copy(
        update={
            "failures": failures,
            "trends": trends,
            "analysis": analysis,
            "open_questions": open_questions,
        }
    )


def _filter_claims(
    claims: Iterable[FailureClaim | AnalysisClaim],
    *,
    finding_ids: set[str],
    kind_label: str,
    drop: bool,
):
    for claim in claims:
        unknown = [fid for fid in claim.cited_finding_ids if fid not in finding_ids]
        if unknown:
            msg = (
                f"{kind_label} claim {claim.claim_id} cites unknown finding_ids:"
                f" {unknown}"
            )
            if drop:
                _log.warning("%s — dropping claim", msg)
                continue
            raise CitationIntegrityError(msg)
        yield claim


def _filter_trends(
    claims: Iterable[TrendClaim],
    *,
    finding_ids: set[str],
    trajectory_keys: set[tuple[str, str]],
    drop: bool,
):
    for claim in claims:
        bad_findings = [fid for fid in claim.cited_finding_ids if fid not in finding_ids]
        bad_trajectories = [
            (ref.run_id, ref.variable)
            for ref in claim.cited_trajectories
            if (ref.run_id, ref.variable) not in trajectory_keys
        ]
        if bad_findings or bad_trajectories:
            msg = (
                f"trend claim {claim.claim_id} cites unknown refs"
                f" findings={bad_findings} trajectories={bad_trajectories}"
            )
            if drop:
                _log.warning("%s — dropping claim", msg)
                continue
            raise CitationIntegrityError(msg)
        yield claim


def _filter_questions(
    questions: Iterable[OpenQuestion],
    *,
    finding_ids: set[str],
    drop: bool,
):
    for q in questions:
        unknown = [fid for fid in q.cited_finding_ids if fid not in finding_ids]
        if unknown:
            msg = (
                f"open_question {q.question_id} cites unknown finding_ids:"
                f" {unknown}"
            )
            if drop:
                _log.warning("%s — dropping question", msg)
                continue
            raise CitationIntegrityError(msg)
        yield q


def _apply_soft_enforcement(claim, flags: frozenset[ProcessFlag]):
    """Apply soft rules to a single claim. Returns a copy with mutations applied.

    Today: provenance downgrade + forbidden-phrase warn.
    """
    updates: dict = {}

    if (
        claim.confidence_basis == ConfidenceBasis.PROCESS_PRIORS
        and flags & UNKNOWN_FLAGS
    ):
        _log.warning(
            "claim %s used process_priors under UNKNOWN flag(s) %s — downgrading"
            " to schema_only",
            claim.claim_id,
            sorted(f.value for f in (flags & UNKNOWN_FLAGS)),
        )
        updates["confidence_basis"] = ConfidenceBasis.SCHEMA_ONLY
        updates["provenance_downgraded"] = True

    forbidden = _scan_forbidden_phrases(claim.summary)
    if forbidden:
        _log.warning(
            "claim %s summary contains causal phrasing %s — diagnosis must stay"
            " observational",
            claim.claim_id,
            forbidden,
        )

    if not updates:
        return claim
    return claim.model_copy(update=updates)


_FORBIDDEN_RE = re.compile(
    r"\b(" + "|".join(re.escape(p) for p in FORBIDDEN_CAUSAL_PHRASES) + r")\b",
    re.IGNORECASE,
)


def _scan_forbidden_phrases(summary: str) -> list[str]:
    return [m.group(1).lower() for m in _FORBIDDEN_RE.finditer(summary)]
