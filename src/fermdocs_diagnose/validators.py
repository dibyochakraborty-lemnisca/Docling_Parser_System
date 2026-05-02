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

from fermdocs.domain.process_priors import ProcessPriors, resolve_priors
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
    priors: ProcessPriors | None = None,
    organism: str | None = None,
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
        priors: The ProcessPriors set the agent had access to during this run.
            When set, claims marked confidence_basis='process_priors' are
            checked: if no matching prior exists for any of the claim's
            affected_variables, the claim is downgraded to schema_only with
            provenance_downgraded=True (Plan A Stage 3 enforcement).
        organism: Organism string (from dossier / get_meta) used to filter
            priors. Required when `priors` is given for the basis check to
            actually find anything.

    Returns:
        A new DiagnosisOutput with soft enforcements applied (provenance
        downgrades, forbidden-phrase warnings logged, optionally dropped
        claims).
    """
    finding_ids = {f.finding_id for f in upstream.findings}
    trajectory_keys = {(t.run_id, t.variable) for t in upstream.trajectories}
    narrative_ids = {n.narrative_id for n in upstream.narrative_observations}
    flags = frozenset(flags)

    # Build the set of variables that have priors loaded for this organism.
    # The validator uses this to decide whether a process_priors claim is
    # actually grounded. When `priors` arg is omitted entirely, leave this
    # as None so the soft-enforcement check skips — backward-compatible
    # with callers that don't know about the priors layer yet.
    priors_variables: set[str] | None = None
    if priors is not None:
        priors_variables = (
            {r.variable for r in resolve_priors(priors, organism=organism)}
            if organism
            else set()
        )

    failures = list(_filter_claims(
        output.failures,
        finding_ids=finding_ids,
        narrative_ids=narrative_ids,
        kind_label="failure",
        drop=drop_unknown_citations,
    ))
    analysis = list(_filter_claims(
        output.analysis,
        finding_ids=finding_ids,
        narrative_ids=narrative_ids,
        kind_label="analysis",
        drop=drop_unknown_citations,
    ))
    trends = list(_filter_trends(
        output.trends,
        finding_ids=finding_ids,
        trajectory_keys=trajectory_keys,
        narrative_ids=narrative_ids,
        drop=drop_unknown_citations,
    ))
    open_questions = list(_filter_questions(
        output.open_questions,
        finding_ids=finding_ids,
        narrative_ids=narrative_ids,
        drop=drop_unknown_citations,
    ))

    failures = [_apply_soft_enforcement(c, flags, priors_variables) for c in failures]
    trends = [_apply_soft_enforcement(c, flags, priors_variables) for c in trends]
    analysis = [_apply_soft_enforcement(c, flags, priors_variables) for c in analysis]

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
    narrative_ids: set[str],
    kind_label: str,
    drop: bool,
):
    for claim in claims:
        unknown_findings = [fid for fid in claim.cited_finding_ids if fid not in finding_ids]
        unknown_narratives = [
            nid for nid in claim.cited_narrative_ids if nid not in narrative_ids
        ]
        if unknown_findings or unknown_narratives:
            msg = (
                f"{kind_label} claim {claim.claim_id} cites unknown refs"
                f" findings={unknown_findings} narratives={unknown_narratives}"
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
    narrative_ids: set[str],
    drop: bool,
):
    for claim in claims:
        bad_findings = [fid for fid in claim.cited_finding_ids if fid not in finding_ids]
        bad_trajectories = [
            (ref.run_id, ref.variable)
            for ref in claim.cited_trajectories
            if (ref.run_id, ref.variable) not in trajectory_keys
        ]
        bad_narratives = [
            nid for nid in claim.cited_narrative_ids if nid not in narrative_ids
        ]
        if bad_findings or bad_trajectories or bad_narratives:
            msg = (
                f"trend claim {claim.claim_id} cites unknown refs"
                f" findings={bad_findings} trajectories={bad_trajectories}"
                f" narratives={bad_narratives}"
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
    narrative_ids: set[str],
    drop: bool,
):
    for q in questions:
        unknown_findings = [fid for fid in q.cited_finding_ids if fid not in finding_ids]
        unknown_narratives = [
            nid for nid in q.cited_narrative_ids if nid not in narrative_ids
        ]
        if unknown_findings or unknown_narratives:
            msg = (
                f"open_question {q.question_id} cites unknown refs"
                f" findings={unknown_findings} narratives={unknown_narratives}"
            )
            if drop:
                _log.warning("%s — dropping question", msg)
                continue
            raise CitationIntegrityError(msg)
        yield q


def _apply_soft_enforcement(
    claim,
    flags: frozenset[ProcessFlag],
    priors_variables: set[str] | None = None,
):
    """Apply soft rules to a single claim. Returns a copy with mutations applied.

    Today:
      - provenance downgrade under UNKNOWN_PROCESS / UNKNOWN_ORGANISM flags
      - provenance downgrade when claim cites process_priors but no matching
        prior is loaded for any of its affected_variables (Plan A Stage 3)
      - forbidden-phrase warn
    """
    updates: dict = {}
    downgrade_reason: str | None = None

    if (
        claim.confidence_basis == ConfidenceBasis.PROCESS_PRIORS
        and flags & UNKNOWN_FLAGS
    ):
        downgrade_reason = (
            f"unknown flags {sorted(f.value for f in (flags & UNKNOWN_FLAGS))}"
        )

    elif (
        claim.confidence_basis == ConfidenceBasis.PROCESS_PRIORS
        and priors_variables is not None
        and not _claim_has_matching_prior(claim, priors_variables)
    ):
        downgrade_reason = (
            f"no matching prior loaded for variables {claim.affected_variables}"
        )

    if downgrade_reason is not None:
        _log.warning(
            "claim %s used process_priors but %s — downgrading to schema_only",
            claim.claim_id,
            downgrade_reason,
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


def _claim_has_matching_prior(claim, priors_variables: set[str]) -> bool:
    """A claim is prior-grounded if at least one affected_variable has a
    loaded prior. Generous on purpose — single-variable misses are common
    and we'd rather warn-via-downgrade than block a multi-variable claim
    where most variables are covered."""
    if not priors_variables:
        return False
    for v in claim.affected_variables:
        if v in priors_variables:
            return True
    return False


_FORBIDDEN_RE = re.compile(
    r"\b(" + "|".join(re.escape(p) for p in FORBIDDEN_CAUSAL_PHRASES) + r")\b",
    re.IGNORECASE,
)


def _scan_forbidden_phrases(summary: str) -> list[str]:
    return [m.group(1).lower() for m in _FORBIDDEN_RE.finditer(summary)]
