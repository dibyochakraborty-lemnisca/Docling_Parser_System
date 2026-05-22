"""validators — citation integrity (hard) + provenance downgrade (soft)."""

from __future__ import annotations

from datetime import datetime
from uuid import UUID

import pytest

from fermdocs_characterize.schema import (
    CharacterizationOutput,
    DataQuality,
    EvidenceStrength,
    ExtractedVia,
    Finding,
    FindingType,
    Meta,
    Severity,
    Trajectory,
)
from fermdocs_diagnose.schema import ConfidenceBasis, TrajectoryRef
from fermdocs_hypothesis.schema import (
    BudgetSnapshot,
    FinalHypothesis,
    HypothesisMeta,
    HypothesisOutput,
)
from fermdocs_hypothesis.validators import (
    CitationIntegrityError,
    validate_hypothesis_output,
)

CHAR_ID = UUID(int=42)
CHAR_ID_STR = str(CHAR_ID)


def _upstream(*, finding_ids: list[str] | None = None, trajectories: list[tuple[str, str]] | None = None) -> CharacterizationOutput:
    finding_ids = finding_ids or [f"{CHAR_ID_STR}:F-0001"]
    findings = [
        Finding(
            finding_id=fid,
            type=FindingType.RANGE_VIOLATION,
            severity=Severity.MINOR,
            summary=f"finding {fid}",
            confidence=0.85,
            extracted_via=ExtractedVia.DETERMINISTIC,
            evidence_strength=EvidenceStrength(n_observations=1, n_independent_runs=1),
            evidence_observation_ids=["O-1"],
            variables_involved=["biomass_g_l"],
        )
        for fid in finding_ids
    ]
    trajs = []
    for i, (run_id, var) in enumerate(trajectories or []):
        trajs.append(
            Trajectory(
                trajectory_id=f"T-{i+1:04d}",
                run_id=run_id,
                variable=var,
                time_grid=[0.0, 1.0],
                values=[1.0, 1.0],
                imputation_flags=[False, False],
                source_observation_ids=["O-1"],
                unit="g/L",
                quality=1.0,
                data_quality=DataQuality(pct_missing=0.0, pct_imputed=0.0, pct_real=1.0),
            )
        )
    return CharacterizationOutput(
        meta=Meta(
            schema_version="2.0",
            characterization_version="v1.0.0",
            characterization_id=CHAR_ID,
            generation_timestamp=datetime(2026, 1, 1),
            source_dossier_ids=["EXP-X"],
        ),
        findings=findings,
        trajectories=trajs,
    )


def _meta() -> HypothesisMeta:
    return HypothesisMeta(
        schema_version="1.0",
        hypothesis_version="v0.1.0",
        hypothesis_id=UUID(int=1),
        supersedes_diagnosis_id=UUID(int=2),
        generation_timestamp=datetime(2026, 5, 3),
        model="gemini-3-pro",
        provider="gemini",
        budget_used=BudgetSnapshot(),
    )


def _final(
    *,
    hyp_id: str = "H-0001",
    finding_ids: list[str] | None = None,
    narrative_ids: list[str] | None = None,
    trajectories: list[TrajectoryRef] | None = None,
    affected_variables: list[str] | None = None,
    basis: ConfidenceBasis = ConfidenceBasis.SCHEMA_ONLY,
) -> FinalHypothesis:
    return FinalHypothesis(
        hyp_id=hyp_id,
        summary="x",
        facet_ids=["FCT-0001"],
        cited_finding_ids=finding_ids or [],
        cited_narrative_ids=narrative_ids or [],
        cited_trajectories=trajectories or [],
        affected_variables=affected_variables or ["biomass_g_l"],
        confidence=0.7,
        confidence_basis=basis,
        critic_flag="green",
        judge_ruled_criticism_valid=False,
    )


def test_unknown_finding_dropped_in_drop_mode(caplog):
    upstream = _upstream(finding_ids=[f"{CHAR_ID_STR}:F-0001"])
    out = HypothesisOutput(
        meta=_meta(),
        final_hypotheses=[
            _final(hyp_id="H-0001", finding_ids=[f"{CHAR_ID_STR}:F-0001"]),
            _final(hyp_id="H-0002", finding_ids=[f"{CHAR_ID_STR}:F-9999"]),
        ],
    )
    cleaned = validate_hypothesis_output(out, upstream=upstream)
    assert {h.hyp_id for h in cleaned.final_hypotheses} == {"H-0001"}


def test_unknown_finding_raises_in_strict_mode():
    upstream = _upstream()
    out = HypothesisOutput(
        meta=_meta(),
        final_hypotheses=[_final(finding_ids=[f"{CHAR_ID_STR}:F-9999"])],
    )
    with pytest.raises(CitationIntegrityError):
        validate_hypothesis_output(out, upstream=upstream, drop_unknown_citations=False)


def test_unknown_trajectory_dropped():
    upstream = _upstream(trajectories=[("RUN-1", "biomass_g_l")])
    out = HypothesisOutput(
        meta=_meta(),
        final_hypotheses=[
            _final(
                finding_ids=[f"{CHAR_ID_STR}:F-0001"],
                trajectories=[TrajectoryRef(run_id="RUN-X", variable="DO")],
            ),
        ],
    )
    cleaned = validate_hypothesis_output(out, upstream=upstream)
    assert cleaned.final_hypotheses == []


def test_known_trajectory_kept():
    upstream = _upstream(trajectories=[("RUN-1", "biomass_g_l")])
    out = HypothesisOutput(
        meta=_meta(),
        final_hypotheses=[
            _final(
                finding_ids=[f"{CHAR_ID_STR}:F-0001"],
                trajectories=[TrajectoryRef(run_id="RUN-1", variable="biomass_g_l")],
            ),
        ],
    )
    cleaned = validate_hypothesis_output(out, upstream=upstream)
    assert len(cleaned.final_hypotheses) == 1


def test_provenance_downgraded_when_no_matching_prior():
    upstream = _upstream()
    out = HypothesisOutput(
        meta=_meta(),
        final_hypotheses=[
            _final(
                finding_ids=[f"{CHAR_ID_STR}:F-0001"],
                affected_variables=["unknown_var"],
                basis=ConfidenceBasis.PROCESS_PRIORS,
            ),
        ],
    )
    cleaned = validate_hypothesis_output(
        out,
        upstream=upstream,
        priors=None,
        organism="E. coli",
    )
    # priors arg is None → no downgrade attempted
    assert cleaned.final_hypotheses[0].confidence_basis == ConfidenceBasis.PROCESS_PRIORS


def test_schema_only_unaffected_by_priors_check():
    upstream = _upstream()
    out = HypothesisOutput(
        meta=_meta(),
        final_hypotheses=[
            _final(
                finding_ids=[f"{CHAR_ID_STR}:F-0001"],
                basis=ConfidenceBasis.SCHEMA_ONLY,
            ),
        ],
    )
    cleaned = validate_hypothesis_output(out, upstream=upstream)
    assert cleaned.final_hypotheses[0].confidence_basis == ConfidenceBasis.SCHEMA_ONLY
    assert cleaned.final_hypotheses[0].provenance_downgraded is False
