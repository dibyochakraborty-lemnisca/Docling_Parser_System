"""Regression tests for the PR-4 hotfixes.

PR 4 adds:
  - ConfidenceBasis.STATISTICAL_TOOLKIT
  - validator downgrade when STATISTICAL_TOOLKIT is claimed without
    citing any catalog-grounded finding
  - trajectory_analyzer per-bundle metric checklist injected into the
    user prompt
  - MAX_TOOL_CALLS lifted from 8 to 20
"""

from __future__ import annotations

from datetime import datetime, timezone
from uuid import UUID, uuid4

import pytest

from fermdocs_characterize.agents.trajectory_analyzer import (
    MAX_TOOL_CALLS,
    TrajectoryAnalyzerAgent,
)
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
    AnalysisClaim,
    ConfidenceBasis,
    DiagnosisMeta,
    DiagnosisOutput,
    FailureClaim,
)
from fermdocs_diagnose.validators import validate_diagnosis


CHAR_ID = UUID("33333333-3333-3333-3333-333333333333")
DIAG_ID = UUID("44444444-4444-4444-4444-444444444444")


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
        summary=f"Using compute_mu, RUN-0001 mu_max=0.42",
        confidence=0.95,
        extracted_via=ExtractedVia.STATISTICAL,
        evidence_strength=EvidenceStrength(n_observations=10, n_independent_runs=1),
        evidence_observation_ids=["OBS-0001"],
        variables_involved=["biomass_g_l"],
        run_ids=["RUN-0001"],
        statistics={"metric_id": metric_id, "tier": "A", "mu_max": 0.42},
    )


def _legacy_finding(idx: int = 5) -> Finding:
    """Range_violation finding without metric_id — the old shape."""
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


def _diagnosis_with_failure(failure: FailureClaim) -> DiagnosisOutput:
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
        failures=[failure],
        analysis=[],
        trends=[],
        open_questions=[],
    )


# ---------- ConfidenceBasis enum sanity ----------


def test_statistical_toolkit_is_a_confidence_basis_value() -> None:
    assert ConfidenceBasis.STATISTICAL_TOOLKIT.value == "statistical_toolkit"


# ---------- validator: STATISTICAL_TOOLKIT honesty check ----------


def test_statistical_toolkit_kept_when_citing_metric_id_finding() -> None:
    upstream = _upstream([_toolkit_finding(idx=1, metric_id="A8")])
    failure = FailureClaim(
        claim_id="D-F-0001",
        summary="A8 mu_max collapse to 0.05 across both runs",
        cited_finding_ids=[f"{CHAR_ID}:F-0001"],
        confidence=0.85,
        confidence_basis=ConfidenceBasis.STATISTICAL_TOOLKIT,
        severity=Severity.MAJOR,
        affected_variables=["biomass_g_l"],
    )
    out = validate_diagnosis(_diagnosis_with_failure(failure), upstream=upstream)
    [kept] = out.failures
    assert kept.confidence_basis == ConfidenceBasis.STATISTICAL_TOOLKIT
    assert kept.provenance_downgraded is False


def test_statistical_toolkit_downgraded_when_no_metric_id_citations() -> None:
    upstream = _upstream([_legacy_finding(idx=5)])  # no metric_id anywhere
    failure = FailureClaim(
        claim_id="D-F-0001",
        summary="claims toolkit grounding it doesn't have",
        cited_finding_ids=[f"{CHAR_ID}:F-0005"],
        confidence=0.85,
        confidence_basis=ConfidenceBasis.STATISTICAL_TOOLKIT,
        severity=Severity.MAJOR,
        affected_variables=["biomass_g_l"],
    )
    out = validate_diagnosis(_diagnosis_with_failure(failure), upstream=upstream)
    [downgraded] = out.failures
    assert downgraded.confidence_basis == ConfidenceBasis.SCHEMA_ONLY
    assert downgraded.provenance_downgraded is True


def test_schema_only_unchanged_when_basis_already_schema_only() -> None:
    upstream = _upstream([_legacy_finding(idx=5)])
    failure = FailureClaim(
        claim_id="D-F-0001",
        summary="legacy claim",
        cited_finding_ids=[f"{CHAR_ID}:F-0005"],
        confidence=0.6,
        confidence_basis=ConfidenceBasis.SCHEMA_ONLY,
        severity=Severity.MAJOR,
        affected_variables=["biomass_g_l"],
    )
    out = validate_diagnosis(_diagnosis_with_failure(failure), upstream=upstream)
    [kept] = out.failures
    assert kept.confidence_basis == ConfidenceBasis.SCHEMA_ONLY
    assert kept.provenance_downgraded is False


# ---------- analyzer: tool budget + checklist ----------


def test_max_tool_calls_lifted_to_20() -> None:
    assert MAX_TOOL_CALLS >= 20


def test_metric_checklist_marks_applicable_for_biomass_only_bundle() -> None:
    out = TrajectoryAnalyzerAgent._build_metric_checklist(
        variables={"biomass_g_l", "time_h"}, n_runs=1
    )
    # A8/A9/A10/A11 need biomass — APPLICABLE
    for mid in ("A8", "A9", "A10", "A11"):
        assert f"[APPLICABLE] {mid}" in out, f"{mid} should be applicable"
    # B10 needs OUR + CER — DATA_GAP
    assert "[DATA_GAP] B10" in out
    # A20 needs >= 3 runs — DATA_GAP at n_runs=1
    assert "[DATA_GAP] A20" in out


def test_metric_checklist_marks_applicable_for_full_indpensim_shape() -> None:
    out = TrajectoryAnalyzerAgent._build_metric_checklist(
        variables={
            "biomass_g_l", "our_mmol_per_l_per_h", "cer_mmol_per_l_per_h",
            "dissolved_o2_mg_l", "agitation_rpm", "substrate_g_l",
            "temperature_k",
        },
        n_runs=2,
    )
    # B10 RQ should now be applicable (OUR + CER both present)
    assert "[APPLICABLE] B10" in out
    # A14 DO margin applicable
    assert "[APPLICABLE] A14" in out
    # A19 KPI table applicable at n_runs=2
    assert "[APPLICABLE] A19" in out
    # A21 still data_gap (needs >= 5 runs)
    assert "[DATA_GAP] A21" in out
