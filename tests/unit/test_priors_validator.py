"""Plan A Stage 3: validator basis enforcement for process_priors.

Covers:
  - Claim with confidence_basis='process_priors' AND a matching prior →
    keeps process_priors basis, no downgrade.
  - Claim with confidence_basis='process_priors' BUT no matching prior →
    downgraded to schema_only with provenance_downgraded=True.
  - Multi-variable claim where ANY one variable has a matching prior →
    no downgrade (generous on purpose; warn-via-downgrade is for the
    no-match case).
  - UNKNOWN_PROCESS still downgrades regardless of priors set
    (legacy behavior preserved).
  - End-to-end through DiagnosisAgent: agent emits process_priors basis
    on a real organism + variable; validator keeps it. Same flow on a
    bogus variable; validator downgrades.
"""

from __future__ import annotations

import json
import uuid
from datetime import datetime
from pathlib import Path

import pytest

from fermdocs.bundle import BundleReader, BundleWriter
from fermdocs.domain.process_priors import cached_priors
from fermdocs_characterize.flags import ProcessFlag
from fermdocs_characterize.schema import (
    CharacterizationOutput,
    DataQuality,
    EvidenceStrength,
    ExtractedVia,
    Finding,
    FindingType,
    Meta,
    Severity,
    Tier,
    Trajectory,
)
from fermdocs_diagnose.agent import DiagnosisAgent, _extract_organism
from fermdocs_diagnose.schema import (
    AnalysisClaim,
    ConfidenceBasis,
    DiagnosisMeta,
    DiagnosisOutput,
    FailureClaim,
)
from fermdocs_diagnose.validators import validate_diagnosis


CHAR_ID = uuid.UUID(int=2026)


def _build_upstream() -> CharacterizationOutput:
    return CharacterizationOutput(
        meta=Meta(
            schema_version="2.0",
            characterization_version="v1.0.0",
            characterization_id=CHAR_ID,
            generation_timestamp=datetime(2026, 5, 3),
            source_dossier_ids=["EXP-PA-S3"],
        ),
        findings=[
            Finding(
                finding_id=f"{CHAR_ID}:F-0001",
                type=FindingType.RANGE_VIOLATION,
                severity=Severity.MAJOR,
                tier=Tier.A,
                summary="biomass below typical",
                confidence=0.8,
                extracted_via=ExtractedVia.DETERMINISTIC,
                evidence_strength=EvidenceStrength(n_observations=4, n_independent_runs=1),
                evidence_observation_ids=["O-1"],
                variables_involved=["biomass_g_l"],
                run_ids=["RUN-Z"],
            ),
        ],
        trajectories=[
            Trajectory(
                trajectory_id="T-0001",
                run_id="RUN-Z",
                variable="biomass_g_l",
                time_grid=[0.0, 8.0, 16.0],
                values=[1.0, 30.0, 60.0],
                imputation_flags=[False, False, False],
                source_observation_ids=["O-1"],
                unit="g/L",
                quality=1.0,
                data_quality=DataQuality(pct_missing=0.0, pct_imputed=0.0, pct_real=1.0),
            ),
        ],
    )


def _wrap(claim) -> DiagnosisOutput:
    return DiagnosisOutput(
        meta=DiagnosisMeta(
            schema_version="1.0",
            diagnosis_version="v1.0.0",
            diagnosis_id=uuid.UUID(int=42),
            supersedes_characterization_id=CHAR_ID,
            generation_timestamp=datetime(2026, 5, 3),
            model="claude-opus-4-7",
            provider="anthropic",
        ),
        failures=[claim],
    )


# ---------------------------------------------------------------------------
# Direct validator tests
# ---------------------------------------------------------------------------


def test_priors_basis_kept_when_variable_has_matching_prior() -> None:
    upstream = _build_upstream()
    claim = FailureClaim(
        claim_id="D-F-0001",
        summary="biomass below typical fed-batch yeast endpoint",
        cited_finding_ids=[f"{CHAR_ID}:F-0001"],
        affected_variables=["biomass_endpoint_g_l"],  # exists in yeast priors
        confidence=0.8,
        confidence_basis=ConfidenceBasis.PROCESS_PRIORS,
        domain_tags=["growth"],
        severity=Severity.MAJOR,
    )
    result = validate_diagnosis(
        _wrap(claim),
        upstream=upstream,
        priors=cached_priors(),
        organism="Saccharomyces cerevisiae",
    )
    f = result.failures[0]
    assert f.confidence_basis == ConfidenceBasis.PROCESS_PRIORS
    assert f.provenance_downgraded is False


def test_priors_basis_downgraded_when_no_matching_prior() -> None:
    upstream = _build_upstream()
    claim = FailureClaim(
        claim_id="D-F-0001",
        summary="some unrelated variable claim",
        cited_finding_ids=[f"{CHAR_ID}:F-0001"],
        affected_variables=["unobtainium_mg_l"],  # not in any prior
        confidence=0.8,
        confidence_basis=ConfidenceBasis.PROCESS_PRIORS,
        domain_tags=["other"],
        severity=Severity.MINOR,
    )
    result = validate_diagnosis(
        _wrap(claim),
        upstream=upstream,
        priors=cached_priors(),
        organism="Saccharomyces cerevisiae",
    )
    f = result.failures[0]
    assert f.confidence_basis == ConfidenceBasis.SCHEMA_ONLY
    assert f.provenance_downgraded is True


def test_priors_basis_downgraded_when_organism_unknown() -> None:
    """No matching organism → no priors-variables → all process_priors claims
    get downgraded."""
    upstream = _build_upstream()
    claim = FailureClaim(
        claim_id="D-F-0001",
        summary="biomass dropped",
        cited_finding_ids=[f"{CHAR_ID}:F-0001"],
        affected_variables=["biomass_endpoint_g_l"],
        confidence=0.8,
        confidence_basis=ConfidenceBasis.PROCESS_PRIORS,
        domain_tags=["growth"],
        severity=Severity.MINOR,
    )
    result = validate_diagnosis(
        _wrap(claim),
        upstream=upstream,
        priors=cached_priors(),
        organism="Bacillus subtilis",  # not in priors
    )
    f = result.failures[0]
    assert f.confidence_basis == ConfidenceBasis.SCHEMA_ONLY
    assert f.provenance_downgraded is True


def test_priors_basis_kept_when_priors_arg_omitted() -> None:
    """Without priors arg, the validator skips the new check.
    Backward-compatible."""
    upstream = _build_upstream()
    claim = FailureClaim(
        claim_id="D-F-0001",
        summary="claim",
        cited_finding_ids=[f"{CHAR_ID}:F-0001"],
        affected_variables=["unobtainium_mg_l"],
        confidence=0.8,
        confidence_basis=ConfidenceBasis.PROCESS_PRIORS,
        domain_tags=["other"],
        severity=Severity.MINOR,
    )
    result = validate_diagnosis(_wrap(claim), upstream=upstream)
    assert result.failures[0].confidence_basis == ConfidenceBasis.PROCESS_PRIORS
    assert result.failures[0].provenance_downgraded is False


def test_multi_variable_claim_partial_match_keeps_basis() -> None:
    """ANY matching variable suffices — generous on purpose."""
    upstream = _build_upstream()
    claim = FailureClaim(
        claim_id="D-F-0001",
        summary="multivariable",
        cited_finding_ids=[f"{CHAR_ID}:F-0001"],
        affected_variables=["biomass_endpoint_g_l", "unobtainium_mg_l"],
        confidence=0.8,
        confidence_basis=ConfidenceBasis.PROCESS_PRIORS,
        domain_tags=["growth"],
        severity=Severity.MAJOR,
    )
    result = validate_diagnosis(
        _wrap(claim),
        upstream=upstream,
        priors=cached_priors(),
        organism="S. cerevisiae",
    )
    assert result.failures[0].confidence_basis == ConfidenceBasis.PROCESS_PRIORS


def test_unknown_process_flag_still_downgrades() -> None:
    """Legacy UNKNOWN_PROCESS flag downgrade is preserved alongside the new
    priors-not-matched check."""
    upstream = _build_upstream()
    claim = FailureClaim(
        claim_id="D-F-0001",
        summary="claim with prior match",
        cited_finding_ids=[f"{CHAR_ID}:F-0001"],
        affected_variables=["biomass_endpoint_g_l"],
        confidence=0.8,
        confidence_basis=ConfidenceBasis.PROCESS_PRIORS,
        domain_tags=["growth"],
        severity=Severity.MAJOR,
    )
    result = validate_diagnosis(
        _wrap(claim),
        upstream=upstream,
        flags=[ProcessFlag.UNKNOWN_PROCESS],
        priors=cached_priors(),
        organism="Saccharomyces cerevisiae",
    )
    # UNKNOWN flag wins; priors match doesn't rescue
    f = result.failures[0]
    assert f.confidence_basis == ConfidenceBasis.SCHEMA_ONLY
    assert f.provenance_downgraded is True


def test_schema_only_claims_unaffected_by_priors_check() -> None:
    upstream = _build_upstream()
    claim = FailureClaim(
        claim_id="D-F-0001",
        summary="schema_only claim",
        cited_finding_ids=[f"{CHAR_ID}:F-0001"],
        affected_variables=["unobtainium_mg_l"],
        confidence=0.5,
        confidence_basis=ConfidenceBasis.SCHEMA_ONLY,
        domain_tags=["other"],
        severity=Severity.MINOR,
    )
    result = validate_diagnosis(
        _wrap(claim),
        upstream=upstream,
        priors=cached_priors(),
        organism="S. cerevisiae",
    )
    f = result.failures[0]
    assert f.confidence_basis == ConfidenceBasis.SCHEMA_ONLY
    assert f.provenance_downgraded is False


# ---------------------------------------------------------------------------
# _extract_organism helper
# ---------------------------------------------------------------------------


def test_extract_organism_from_observed() -> None:
    dossier = {
        "experiment": {"process": {"observed": {"organism": "Saccharomyces cerevisiae"}}}
    }
    assert _extract_organism(dossier) == "Saccharomyces cerevisiae"


def test_extract_organism_falls_back_through_layers() -> None:
    dossier = {"experiment": {"organism": "E. coli BL21"}}
    assert _extract_organism(dossier) == "E. coli BL21"


def test_extract_organism_returns_none_when_absent() -> None:
    assert _extract_organism({}) is None
    assert _extract_organism({"experiment": {}}) is None


# ---------------------------------------------------------------------------
# End-to-end through DiagnosisAgent
# ---------------------------------------------------------------------------


class _ScriptedClient:
    def __init__(self, responses: list[dict]) -> None:
        self._responses = list(responses)

    def call(self, system: str, messages: list[dict[str, str]]) -> dict:
        return self._responses.pop(0)


def _build_bundle(tmp_path: Path, upstream: CharacterizationOutput) -> Path:
    writer = BundleWriter.create(
        tmp_path,
        run_ids=["RUN-Z"],
        golden_schema_version="2.0",
        pipeline_version="0.1.0",
    )
    writer.write_dossier(
        {
            "experiment": {
                "experiment_id": "EXP-PA-S3",
                "process": {"observed": {"organism": "Saccharomyces cerevisiae"}},
            }
        }
    )
    writer.write_characterization(upstream.model_dump_json())
    return writer.finalize()


def test_agent_keeps_priors_basis_for_yeast_biomass(tmp_path: Path) -> None:
    upstream = _build_upstream()
    bundle_path = _build_bundle(tmp_path, upstream)
    reader = BundleReader(bundle_path)
    fid = f"{CHAR_ID}:F-0001"
    client = _ScriptedClient(
        [
            {"action": "tool_call", "tool": "get_meta", "args": {}},
            {
                "action": "emit",
                "failures": [
                    {
                        "summary": "biomass below typical yeast fed-batch endpoint",
                        "cited_finding_ids": [fid],
                        "affected_variables": ["biomass_endpoint_g_l"],
                        "confidence": 0.8,
                        "confidence_basis": "process_priors",
                        "domain_tags": ["growth"],
                        "severity": "major",
                    }
                ],
                "trends": [],
                "analysis": [],
                "open_questions": [],
            },
        ]
    )
    agent = DiagnosisAgent(client=client, max_steps=4)
    result = agent.diagnose(
        {
            "experiment": {
                "experiment_id": "EXP-PA-S3",
                "process": {"observed": {"organism": "Saccharomyces cerevisiae"}},
            }
        },
        upstream,
        bundle=reader,
        diagnosis_id=uuid.UUID(int=11),
        generation_timestamp=datetime(2026, 5, 3),
    )
    assert result.meta.error is None
    f = result.failures[0]
    assert f.confidence_basis == ConfidenceBasis.PROCESS_PRIORS
    assert f.provenance_downgraded is False


def test_agent_downgrades_priors_basis_for_unmatched_variable(tmp_path: Path) -> None:
    upstream = _build_upstream()
    bundle_path = _build_bundle(tmp_path, upstream)
    reader = BundleReader(bundle_path)
    fid = f"{CHAR_ID}:F-0001"
    client = _ScriptedClient(
        [
            {"action": "tool_call", "tool": "get_meta", "args": {}},
            {
                "action": "emit",
                "failures": [
                    {
                        "summary": "claim about a variable with no prior",
                        "cited_finding_ids": [fid],
                        "affected_variables": ["unobtainium_mg_l"],
                        "confidence": 0.8,
                        "confidence_basis": "process_priors",
                        "domain_tags": ["other"],
                        "severity": "minor",
                    }
                ],
                "trends": [],
                "analysis": [],
                "open_questions": [],
            },
        ]
    )
    agent = DiagnosisAgent(client=client, max_steps=4)
    result = agent.diagnose(
        {
            "experiment": {
                "experiment_id": "EXP-PA-S3",
                "process": {"observed": {"organism": "Saccharomyces cerevisiae"}},
            }
        },
        upstream,
        bundle=reader,
        diagnosis_id=uuid.UUID(int=12),
        generation_timestamp=datetime(2026, 5, 3),
    )
    assert result.meta.error is None
    f = result.failures[0]
    assert f.confidence_basis == ConfidenceBasis.SCHEMA_ONLY
    assert f.provenance_downgraded is True
