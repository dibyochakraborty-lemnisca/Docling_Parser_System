"""Cross-output validators for HypothesisOutput.

Plan ref: plans/2026-05-03-hypothesis-debate-v0.md §10, §13 (defer
provenance downgrade to Stage 3 here).

Hard rejection vs soft enforcement (mirrors diagnose validators.py):

  Hard (raises by default in strict mode; drops in default mode):
    - cited_finding_ids reference a real Finding in the upstream
      CharacterizationOutput
    - cited_narrative_ids reference a real NarrativeObservation
    - cited_trajectories reference a real (run_id, variable) pair

  Soft (mutates output, logs warning):
    - provenance downgrade: hypotheses claiming process_priors basis
      when no matching prior is loaded for any affected_variable get
      downgraded to schema_only with provenance_downgraded=True.
      (Same Plan A Stage 3 enforcement as diagnose.)

Why split? A reject-and-fail loop on a multi-hypothesis output throws
away an entire stage when one hypothesis drifts. Hard rejection is
reserved for issues that break the contract downstream consumers
depend on (unknown citations). Soft enforcement covers what the prompt
should already prevent and the validator catches as a drift backstop.
"""

from __future__ import annotations

import logging
from collections.abc import Iterable

from fermdocs.domain.process_priors import ProcessPriors, resolve_priors
from fermdocs_characterize.schema import CharacterizationOutput
from fermdocs_diagnose.schema import ConfidenceBasis
from fermdocs_hypothesis.schema import (
    FinalHypothesis,
    HypothesisOutput,
    RejectedHypothesis,
)

_log = logging.getLogger(__name__)


class CitationIntegrityError(ValueError):
    """Raised when a hypothesis cites an ID that does not exist upstream."""


def validate_hypothesis_output(
    output: HypothesisOutput,
    *,
    upstream: CharacterizationOutput,
    drop_unknown_citations: bool = True,
    priors: ProcessPriors | None = None,
    organism: str | None = None,
) -> HypothesisOutput:
    """Apply cross-output validation and soft enforcement.

    Args:
        output: HypothesisOutput as constructed by the runner.
        upstream: CharacterizationOutput the diagnosis (and thus this
            hypothesis stage) was built on.
        drop_unknown_citations: When True (default), hypotheses with
            unknown citations are dropped (logged). When False, they
            raise CitationIntegrityError.
        priors: ProcessPriors set the stage had access to. When set,
            FinalHypothesis claims marked confidence_basis='process_priors'
            are checked: if no matching prior exists for any affected
            variable, the hypothesis is downgraded to schema_only with
            provenance_downgraded=True.
        organism: Used with priors. Required for the basis check to
            actually find anything.

    Returns:
        New HypothesisOutput with soft enforcements applied.
    """
    finding_ids = {f.finding_id for f in upstream.findings}
    trajectory_keys = {(t.run_id, t.variable) for t in upstream.trajectories}
    narrative_ids = {n.narrative_id for n in upstream.narrative_observations}

    priors_variables: set[str] | None = None
    if priors is not None:
        priors_variables = (
            {r.variable for r in resolve_priors(priors, organism=organism)}
            if organism
            else set()
        )

    cleaned_finals = list(_filter_finals(
        output.final_hypotheses,
        finding_ids=finding_ids,
        narrative_ids=narrative_ids,
        trajectory_keys=trajectory_keys,
        drop=drop_unknown_citations,
    ))
    cleaned_finals = [
        _apply_soft_enforcement(h, priors_variables) for h in cleaned_finals
    ]

    # Rejected hypotheses are kept untouched; they're audit records, not
    # outputs anyone consumes for reasoning.
    return output.model_copy(
        update={"final_hypotheses": cleaned_finals},
    )


def _filter_finals(
    hyps: Iterable[FinalHypothesis],
    *,
    finding_ids: set[str],
    narrative_ids: set[str],
    trajectory_keys: set[tuple[str, str]],
    drop: bool,
):
    for h in hyps:
        unknown_findings = [fid for fid in h.cited_finding_ids if fid not in finding_ids]
        unknown_narratives = [
            nid for nid in h.cited_narrative_ids if nid not in narrative_ids
        ]
        unknown_trajs = [
            (ref.run_id, ref.variable)
            for ref in h.cited_trajectories
            if (ref.run_id, ref.variable) not in trajectory_keys
        ]
        if unknown_findings or unknown_narratives or unknown_trajs:
            msg = (
                f"hypothesis {h.hyp_id} cites unknown refs"
                f" findings={unknown_findings} narratives={unknown_narratives}"
                f" trajectories={unknown_trajs}"
            )
            if drop:
                _log.warning("%s — dropping hypothesis", msg)
                continue
            raise CitationIntegrityError(msg)
        yield h


def _apply_soft_enforcement(
    h: FinalHypothesis,
    priors_variables: set[str] | None,
) -> FinalHypothesis:
    """Provenance downgrade — see diagnose validators._apply_soft_enforcement
    for the canonical version of this rule."""
    if h.confidence_basis != ConfidenceBasis.PROCESS_PRIORS:
        return h
    if priors_variables is None:
        # Caller didn't supply priors; can't validate, leave alone.
        return h
    if _claim_has_matching_prior(h, priors_variables):
        return h
    _log.warning(
        "hypothesis %s used process_priors but no matching prior loaded for"
        " variables %s — downgrading to schema_only",
        h.hyp_id,
        h.affected_variables,
    )
    return h.model_copy(
        update={
            "confidence_basis": ConfidenceBasis.SCHEMA_ONLY,
            "provenance_downgraded": True,
        }
    )


def _claim_has_matching_prior(h: FinalHypothesis, priors_variables: set[str]) -> bool:
    if not priors_variables:
        return False
    for v in h.affected_variables:
        if v in priors_variables:
            return True
    return False


__all__ = [
    "CitationIntegrityError",
    "validate_hypothesis_output",
]
