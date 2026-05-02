"""Cross-output validator tests.

Hard rejection: unknown citations.
Soft enforcement: provenance downgrade under UNKNOWN flags, forbidden phrases.
"""

from __future__ import annotations

import logging
import uuid
from datetime import datetime

import pytest

from fermdocs_characterize.flags import ProcessFlag
from fermdocs_characterize.schema import (
    CharacterizationOutput,
    EvidenceStrength,
    ExtractedVia,
    Finding,
    FindingType,
    Meta,
    Severity,
    Trajectory,
    DataQuality,
)
from fermdocs_diagnose.schema import (
    AnalysisClaim,
    ConfidenceBasis,
    DiagnosisMeta,
    DiagnosisOutput,
    FailureClaim,
    OpenQuestion,
    TrajectoryRef,
    TrendClaim,
)
from fermdocs_diagnose.validators import (
    CitationIntegrityError,
    validate_diagnosis,
)


CHAR_ID = uuid.UUID(int=42)


def _upstream(
    *,
    finding_ids: list[str] | None = None,
    trajectories: list[tuple[str, str]] | None = None,
) -> CharacterizationOutput:
    finding_ids = finding_ids or [f"{CHAR_ID}:F-0001"]
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


def _meta() -> DiagnosisMeta:
    return DiagnosisMeta(
        schema_version="1.0",
        diagnosis_version="v1.0.0",
        diagnosis_id=uuid.UUID(int=1),
        supersedes_characterization_id=CHAR_ID,
        generation_timestamp=datetime(2026, 5, 2),
        model="claude-opus-4-7",
        provider="anthropic",
    )


def _failure(
    claim_id: str = "D-F-0001",
    cited: list[str] | None = None,
    basis: ConfidenceBasis = ConfidenceBasis.SCHEMA_ONLY,
    summary: str = "biomass plateau in 40h-60h window",
) -> FailureClaim:
    return FailureClaim(
        claim_id=claim_id,
        summary=summary,
        cited_finding_ids=cited or [f"{CHAR_ID}:F-0001"],
        confidence=0.7,
        confidence_basis=basis,
        severity=Severity.MAJOR,
    )


# ---------- citation integrity (hard) ----------


def test_unknown_citation_dropped_in_drop_mode(caplog):
    upstream = _upstream(finding_ids=[f"{CHAR_ID}:F-0001"])
    out = DiagnosisOutput(
        meta=_meta(),
        failures=[
            _failure(claim_id="D-F-0001", cited=[f"{CHAR_ID}:F-0001"]),
            _failure(claim_id="D-F-0002", cited=[f"{CHAR_ID}:F-9999"]),
        ],
    )
    with caplog.at_level(logging.WARNING):
        cleaned = validate_diagnosis(out, upstream=upstream)
    assert len(cleaned.failures) == 1
    assert cleaned.failures[0].claim_id == "D-F-0001"
    assert any(
        "unknown refs" in r.message and "F-9999" in r.message
        for r in caplog.records
    )


def test_unknown_citation_raises_in_strict_mode():
    upstream = _upstream(finding_ids=[f"{CHAR_ID}:F-0001"])
    out = DiagnosisOutput(
        meta=_meta(),
        failures=[_failure(cited=[f"{CHAR_ID}:F-9999"])],
    )
    with pytest.raises(CitationIntegrityError):
        validate_diagnosis(out, upstream=upstream, drop_unknown_citations=False)


def test_trend_unknown_trajectory_dropped(caplog):
    upstream = _upstream(
        finding_ids=[f"{CHAR_ID}:F-0001"],
        trajectories=[("RUN-1", "biomass_g_l")],
    )
    bad_trend = TrendClaim(
        claim_id="D-T-0001",
        summary="DO declines",
        cited_trajectories=[TrajectoryRef(run_id="RUN-X", variable="DO")],
        confidence=0.6,
        confidence_basis=ConfidenceBasis.SCHEMA_ONLY,
        direction="decreasing",
    )
    out = DiagnosisOutput(meta=_meta(), trends=[bad_trend])
    with caplog.at_level(logging.WARNING):
        cleaned = validate_diagnosis(out, upstream=upstream)
    assert cleaned.trends == []


def test_trend_with_known_trajectory_kept():
    upstream = _upstream(
        finding_ids=[f"{CHAR_ID}:F-0001"],
        trajectories=[("RUN-1", "biomass_g_l")],
    )
    good = TrendClaim(
        claim_id="D-T-0001",
        summary="biomass plateau",
        cited_trajectories=[TrajectoryRef(run_id="RUN-1", variable="biomass_g_l")],
        confidence=0.6,
        confidence_basis=ConfidenceBasis.SCHEMA_ONLY,
        direction="plateau",
    )
    out = DiagnosisOutput(meta=_meta(), trends=[good])
    cleaned = validate_diagnosis(out, upstream=upstream)
    assert len(cleaned.trends) == 1


def test_open_question_unknown_citation_dropped(caplog):
    upstream = _upstream()
    bad_q = OpenQuestion(
        question_id="D-Q-0001",
        question="x?",
        why_it_matters="y",
        cited_finding_ids=[f"{CHAR_ID}:F-9999"],
        answer_format_hint="yes_no",
    )
    out = DiagnosisOutput(meta=_meta(), open_questions=[bad_q])
    with caplog.at_level(logging.WARNING):
        cleaned = validate_diagnosis(out, upstream=upstream)
    assert cleaned.open_questions == []


# ---------- soft enforcement: provenance downgrade ----------


def test_process_priors_downgraded_under_unknown_process(caplog):
    upstream = _upstream()
    claim = _failure(basis=ConfidenceBasis.PROCESS_PRIORS)
    out = DiagnosisOutput(meta=_meta(), failures=[claim])
    with caplog.at_level(logging.WARNING):
        cleaned = validate_diagnosis(
            out, upstream=upstream, flags=[ProcessFlag.UNKNOWN_PROCESS]
        )
    assert cleaned.failures[0].confidence_basis == ConfidenceBasis.SCHEMA_ONLY
    assert cleaned.failures[0].provenance_downgraded is True
    assert any("downgrading" in r.message for r in caplog.records)


def test_process_priors_downgraded_under_unknown_organism():
    upstream = _upstream()
    claim = _failure(basis=ConfidenceBasis.PROCESS_PRIORS)
    out = DiagnosisOutput(meta=_meta(), failures=[claim])
    cleaned = validate_diagnosis(
        out, upstream=upstream, flags=[ProcessFlag.UNKNOWN_ORGANISM]
    )
    assert cleaned.failures[0].confidence_basis == ConfidenceBasis.SCHEMA_ONLY
    assert cleaned.failures[0].provenance_downgraded is True


def test_process_priors_kept_when_no_unknown_flag():
    upstream = _upstream()
    claim = _failure(basis=ConfidenceBasis.PROCESS_PRIORS)
    out = DiagnosisOutput(meta=_meta(), failures=[claim])
    cleaned = validate_diagnosis(
        out, upstream=upstream, flags=[ProcessFlag.SPARSE_DATA]
    )
    assert cleaned.failures[0].confidence_basis == ConfidenceBasis.PROCESS_PRIORS
    assert cleaned.failures[0].provenance_downgraded is False


def test_schema_only_unaffected_under_unknown_flag():
    upstream = _upstream()
    claim = _failure(basis=ConfidenceBasis.SCHEMA_ONLY)
    out = DiagnosisOutput(meta=_meta(), failures=[claim])
    cleaned = validate_diagnosis(
        out, upstream=upstream, flags=[ProcessFlag.UNKNOWN_PROCESS]
    )
    assert cleaned.failures[0].confidence_basis == ConfidenceBasis.SCHEMA_ONLY
    assert cleaned.failures[0].provenance_downgraded is False


# ---------- soft enforcement: forbidden phrases ----------


def test_forbidden_causal_phrase_warns_does_not_drop(caplog):
    upstream = _upstream()
    claim = _failure(summary="biomass plateau because nitrogen was limiting")
    out = DiagnosisOutput(meta=_meta(), failures=[claim])
    with caplog.at_level(logging.WARNING):
        cleaned = validate_diagnosis(out, upstream=upstream)
    assert len(cleaned.failures) == 1
    assert any("causal phrasing" in r.message for r in caplog.records)


def test_clean_summary_no_warning(caplog):
    upstream = _upstream()
    claim = _failure(summary="biomass plateau between 40h and 60h")
    out = DiagnosisOutput(meta=_meta(), failures=[claim])
    with caplog.at_level(logging.WARNING):
        validate_diagnosis(out, upstream=upstream)
    assert not any("causal phrasing" in r.message for r in caplog.records)


def test_forbidden_phrase_match_is_word_boundary():
    """'because' should match; 'beecause' substring inside an unrelated word
    should not. Guards against over-matching on common English bigrams.
    """
    from fermdocs_diagnose.validators import _scan_forbidden_phrases

    assert "because" in _scan_forbidden_phrases("plateau because limiting")
    assert _scan_forbidden_phrases("the cause is unknown") == []


# ---------- composite ----------


def test_mixed_good_and_bad_claims(caplog):
    """A typical batch: one citation-clean claim + one with unknown citation +
    one with process_priors under UNKNOWN_PROCESS. Result: clean kept,
    unknown-citation dropped, prior-driven claim downgraded.
    """
    upstream = _upstream(finding_ids=[f"{CHAR_ID}:F-0001", f"{CHAR_ID}:F-0002"])
    failures = [
        _failure(claim_id="D-F-0001", cited=[f"{CHAR_ID}:F-0001"]),
        _failure(claim_id="D-F-0002", cited=[f"{CHAR_ID}:F-9999"]),
        _failure(
            claim_id="D-F-0003",
            cited=[f"{CHAR_ID}:F-0002"],
            basis=ConfidenceBasis.PROCESS_PRIORS,
        ),
    ]
    out = DiagnosisOutput(meta=_meta(), failures=failures)
    with caplog.at_level(logging.WARNING):
        cleaned = validate_diagnosis(
            out, upstream=upstream, flags=[ProcessFlag.UNKNOWN_PROCESS]
        )
    assert {c.claim_id for c in cleaned.failures} == {"D-F-0001", "D-F-0003"}
    downgraded = next(c for c in cleaned.failures if c.claim_id == "D-F-0003")
    assert downgraded.provenance_downgraded is True
