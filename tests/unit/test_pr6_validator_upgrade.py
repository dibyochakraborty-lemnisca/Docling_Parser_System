"""Validator auto-upgrades schema_only → statistical_toolkit when warranted.

Production bug from runs a64b38c9 (PDF) and 2305f6af (IndPenSim): both
runs had diagnose claims citing exclusively STATISTICAL findings with
metric_ids (B16 carbon balance, A8 mu_max, A9 doubling time, etc), yet
the model emitted confidence_basis="schema_only" — the most cautious
basis. The PR-4 prompt rule asked for statistical_toolkit but Gemini
defaulted to schema_only.

Fix: validator detects this case post-hoc and upgrades. Honest because
the cited findings ARE catalog-grounded; the basis was just labelled
wrong.

The downgrade-when-dishonest path from PR 4 is preserved; this is the
mirror upgrade-when-too-cautious path.
"""

from __future__ import annotations

from datetime import datetime, timezone
from uuid import UUID

from fermdocs_characterize.schema import (
    CharacterizationOutput,
    EvidenceStrength,
    ExtractedVia,
    Finding,
    FindingType,
    Meta,
    Severity,
    Tier,
)
from fermdocs_diagnose.schema import (
    ConfidenceBasis,
    DiagnosisMeta,
    DiagnosisOutput,
    FailureClaim,
    TrendClaim,
)
from fermdocs_diagnose.validators import validate_diagnosis

CHAR_ID = UUID("66666666-6666-6666-6666-666666666666")
DIAG_ID = UUID("77777777-7777-7777-7777-777777777777")


def _meta() -> Meta:
    return Meta(
        schema_version="1.0",
        characterization_version="0.1.0",
        characterization_id=CHAR_ID,
        generation_timestamp=datetime(2026, 5, 5, tzinfo=timezone.utc),
        source_dossier_ids=["dossier-test"],
    )


def _toolkit_finding(idx: int = 1, metric_id: str = "A8") -> Finding:
    return Finding(
        finding_id=f"{CHAR_ID}:F-{idx:04d}",
        type=FindingType.TRAJECTORY_PATTERN,
        severity=Severity.MINOR,
        tier=Tier.A,
        summary=f"compute_mu RUN-0001 mu_max=0.42 ({metric_id})",
        confidence=0.95,
        extracted_via=ExtractedVia.STATISTICAL,
        evidence_strength=EvidenceStrength(n_observations=10, n_independent_runs=1),
        evidence_observation_ids=["OBS-0001"],
        variables_involved=["biomass_g_l"],
        run_ids=["RUN-0001"],
        statistics={"metric_id": metric_id, "tier": "A"},
    )


def _legacy_finding(idx: int = 5) -> Finding:
    return Finding(
        finding_id=f"{CHAR_ID}:F-{idx:04d}",
        type=FindingType.RANGE_VIOLATION,
        severity=Severity.MAJOR,
        tier=Tier.A,
        summary="biomass_g_l exceeded nominal spec",
        confidence=0.9,
        extracted_via=ExtractedVia.DETERMINISTIC,
        evidence_strength=EvidenceStrength(n_observations=10, n_independent_runs=1),
        evidence_observation_ids=["OBS-0050"],
        variables_involved=["biomass_g_l"],
        run_ids=["RUN-0001"],
    )


def _upstream(findings: list[Finding]) -> CharacterizationOutput:
    return CharacterizationOutput(
        meta=_meta(), findings=findings, narrative_observations=[], trajectories=[]
    )


def _diag_with(failures=(), trends=()) -> DiagnosisOutput:
    return DiagnosisOutput(
        meta=DiagnosisMeta(
            schema_version="1.0",
            diagnosis_version="0.1.0",
            diagnosis_id=DIAG_ID,
            supersedes_characterization_id=CHAR_ID,
            generation_timestamp=datetime(2026, 5, 5, tzinfo=timezone.utc),
            model="test",
            provider="gemini",
        ),
        failures=list(failures),
        analysis=[],
        trends=list(trends),
        open_questions=[],
    )


# ---------- upgrade path ----------


def test_schema_only_upgraded_when_all_cited_findings_are_toolkit() -> None:
    """The bug we hit on run 2305f6af: model emitted schema_only on a
    trend citing only STATISTICAL/metric_id findings; validator should
    upgrade to statistical_toolkit."""
    upstream = _upstream(
        [_toolkit_finding(idx=1, metric_id="A8"), _toolkit_finding(idx=2, metric_id="A9")]
    )
    trend = TrendClaim(
        claim_id="D-T-0001",
        summary="mu_max peaked at 0.033 1/h, doubling time 20.7h",
        cited_finding_ids=[f"{CHAR_ID}:F-0001", f"{CHAR_ID}:F-0002"],
        confidence=0.85,
        confidence_basis=ConfidenceBasis.SCHEMA_ONLY,
        affected_variables=["biomass_g_l"],
        direction="increasing",
    )
    out = validate_diagnosis(_diag_with(trends=[trend]), upstream=upstream)
    [upgraded] = out.trends
    assert upgraded.confidence_basis == ConfidenceBasis.STATISTICAL_TOOLKIT
    # Upgrade is not a downgrade — provenance_downgraded stays False.
    assert upgraded.provenance_downgraded is False


def test_schema_only_NOT_upgraded_when_any_citation_is_legacy() -> None:
    """Mixed citations (some toolkit, some range_violation) stay schema_only.
    All-or-nothing keeps the upgrade honest."""
    upstream = _upstream(
        [_toolkit_finding(idx=1, metric_id="A8"), _legacy_finding(idx=5)]
    )
    failure = FailureClaim(
        claim_id="D-F-0001",
        summary="something blended",
        cited_finding_ids=[f"{CHAR_ID}:F-0001", f"{CHAR_ID}:F-0005"],
        confidence=0.85,
        confidence_basis=ConfidenceBasis.SCHEMA_ONLY,
        severity=Severity.MAJOR,
        affected_variables=["biomass_g_l"],
    )
    out = validate_diagnosis(_diag_with(failures=[failure]), upstream=upstream)
    [kept] = out.failures
    assert kept.confidence_basis == ConfidenceBasis.SCHEMA_ONLY


def test_schema_only_NOT_upgraded_when_no_citations() -> None:
    """Zero-citation claim is schema_only by definition; nothing to verify."""
    upstream = _upstream([_toolkit_finding(idx=1, metric_id="A8")])
    failure = FailureClaim(
        claim_id="D-F-0001",
        summary="ungrounded",
        cited_finding_ids=[],
        cited_narrative_ids=["foo:N-0001"],
        confidence=0.6,
        confidence_basis=ConfidenceBasis.SCHEMA_ONLY,
        severity=Severity.MINOR,
        affected_variables=[],
    )
    upstream_with_narrative = upstream.model_copy(
        update={
            "narrative_observations": [],  # citation will fail integrity below
        }
    )
    # We don't care about narrative integrity for this path; just verify
    # zero finding citations doesn't trigger an upgrade.
    out = validate_diagnosis(
        _diag_with(failures=[failure]),
        upstream=upstream,
        drop_unknown_citations=True,
    )
    if out.failures:
        assert out.failures[0].confidence_basis == ConfidenceBasis.SCHEMA_ONLY


def test_existing_statistical_toolkit_kept_when_grounded() -> None:
    """Pre-existing STATISTICAL_TOOLKIT claim stays put when grounded.
    (Regression check: the upgrade path mustn't accidentally re-flag it.)"""
    upstream = _upstream([_toolkit_finding(idx=1, metric_id="A8")])
    trend = TrendClaim(
        claim_id="D-T-0001",
        summary="A8 cited",
        cited_finding_ids=[f"{CHAR_ID}:F-0001"],
        confidence=0.85,
        confidence_basis=ConfidenceBasis.STATISTICAL_TOOLKIT,
        affected_variables=["biomass_g_l"],
        direction="increasing",
    )
    out = validate_diagnosis(_diag_with(trends=[trend]), upstream=upstream)
    [kept] = out.trends
    assert kept.confidence_basis == ConfidenceBasis.STATISTICAL_TOOLKIT
    assert kept.provenance_downgraded is False


def test_dishonest_statistical_toolkit_still_downgraded() -> None:
    """The PR-4 downgrade path stays intact — claim says toolkit but
    cites only legacy findings → forced down to schema_only."""
    upstream = _upstream([_legacy_finding(idx=5)])
    trend = TrendClaim(
        claim_id="D-T-0001",
        summary="dishonest claim",
        cited_finding_ids=[f"{CHAR_ID}:F-0005"],
        confidence=0.85,
        confidence_basis=ConfidenceBasis.STATISTICAL_TOOLKIT,
        affected_variables=["biomass_g_l"],
        direction="increasing",
    )
    out = validate_diagnosis(_diag_with(trends=[trend]), upstream=upstream)
    [downgraded] = out.trends
    assert downgraded.confidence_basis == ConfidenceBasis.SCHEMA_ONLY
    assert downgraded.provenance_downgraded is True
