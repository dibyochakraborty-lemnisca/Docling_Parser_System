"""Schema-level tests for DiagnosisOutput.

These tests target what the schema enforces on its own (id shapes, confidence
caps, intra-output uniqueness, citation-must-be-non-empty). Cross-output
citation integrity lives in test_diagnose_validators.py.
"""

from __future__ import annotations

import uuid
from datetime import datetime

import pytest

from fermdocs_characterize.schema import Severity, TimeWindow
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


def _meta(error: str | None = None) -> DiagnosisMeta:
    return DiagnosisMeta(
        schema_version="1.0",
        diagnosis_version="v1.0.0",
        diagnosis_id=uuid.UUID(int=1),
        supersedes_characterization_id=uuid.UUID(int=42),
        generation_timestamp=datetime(2026, 5, 2),
        model="claude-opus-4-7",
        provider="anthropic",
        error=error,
    )


def _failure(claim_id: str = "D-F-0001") -> FailureClaim:
    return FailureClaim(
        claim_id=claim_id,
        summary="biomass plateau between 40h and 60h",
        cited_finding_ids=["F-0001"],
        affected_variables=["biomass_g_l"],
        confidence=0.8,
        confidence_basis=ConfidenceBasis.SCHEMA_ONLY,
        domain_tags=["growth"],
        severity=Severity.MAJOR,
        time_window=TimeWindow(start=40, end=60),
    )


def _trend(claim_id: str = "D-T-0001") -> TrendClaim:
    return TrendClaim(
        claim_id=claim_id,
        summary="DO declines monotonically after 30h",
        cited_finding_ids=[],
        cited_trajectories=[TrajectoryRef(run_id="RUN-1", variable="DO")],
        affected_variables=["DO"],
        confidence=0.75,
        confidence_basis=ConfidenceBasis.SCHEMA_ONLY,
        direction="decreasing",
        domain_tags=["environmental"],
    )


def _analysis(claim_id: str = "D-A-0001") -> AnalysisClaim:
    return AnalysisClaim(
        claim_id=claim_id,
        summary="biomass and substrate trajectories align with stationary phase",
        cited_finding_ids=["F-0001"],
        affected_variables=["biomass_g_l", "substrate"],
        confidence=0.7,
        confidence_basis=ConfidenceBasis.SCHEMA_ONLY,
        kind="phase_characterization",
        domain_tags=["growth"],
    )


def _question(qid: str = "D-Q-0001") -> OpenQuestion:
    return OpenQuestion(
        question_id=qid,
        question="Was the sampling cadence different in the 40h-60h window?",
        why_it_matters="changes whether the plateau is real or an artifact",
        cited_finding_ids=["F-0001"],
        answer_format_hint="yes_no",
        domain_tags=["data_quality"],
    )


# ---------- claim_id shape ----------


def test_failure_claim_id_must_match_pattern():
    with pytest.raises(ValueError, match="claim_id"):
        FailureClaim(
            claim_id="F-0001",  # missing D- prefix
            summary="x",
            cited_finding_ids=["F-0001"],
            confidence=0.5,
            confidence_basis=ConfidenceBasis.SCHEMA_ONLY,
            severity=Severity.MINOR,
        )


def test_failure_claim_id_rejects_wrong_kind_letter():
    with pytest.raises(ValueError, match="D-F-"):
        FailureClaim(
            claim_id="D-T-0001",  # T means trend, not failure
            summary="x",
            cited_finding_ids=["F-0001"],
            confidence=0.5,
            confidence_basis=ConfidenceBasis.SCHEMA_ONLY,
            severity=Severity.MINOR,
        )


def test_trend_claim_id_must_start_d_t():
    with pytest.raises(ValueError, match="D-T-"):
        TrendClaim(
            claim_id="D-A-0001",
            summary="x",
            cited_trajectories=[TrajectoryRef(run_id="R", variable="V")],
            confidence=0.5,
            confidence_basis=ConfidenceBasis.SCHEMA_ONLY,
            direction="increasing",
        )


def test_analysis_claim_id_must_start_d_a():
    with pytest.raises(ValueError, match="D-A-"):
        AnalysisClaim(
            claim_id="D-F-0001",
            summary="x",
            cited_finding_ids=["F-0001"],
            confidence=0.5,
            confidence_basis=ConfidenceBasis.SCHEMA_ONLY,
            kind="phase_characterization",
        )


def test_question_id_pattern():
    with pytest.raises(ValueError, match="question_id"):
        OpenQuestion(
            question_id="Q-0001",
            question="x",
            why_it_matters="y",
            cited_finding_ids=["F-0001"],
            answer_format_hint="yes_no",
        )


# ---------- confidence cap ----------


def test_confidence_cap_at_0_85():
    """LLM-authored claims cap at 0.85 to match identity_extractor convention."""
    with pytest.raises(ValueError):
        FailureClaim(
            claim_id="D-F-0001",
            summary="x",
            cited_finding_ids=["F-0001"],
            confidence=0.95,
            confidence_basis=ConfidenceBasis.SCHEMA_ONLY,
            severity=Severity.MINOR,
        )


def test_confidence_at_cap_is_allowed():
    c = _failure()
    assert c.confidence <= 0.85


# ---------- citation must be non-empty ----------


def test_failure_must_cite_finding():
    with pytest.raises(ValueError, match="must cite"):
        FailureClaim(
            claim_id="D-F-0001",
            summary="x",
            cited_finding_ids=[],
            confidence=0.5,
            confidence_basis=ConfidenceBasis.SCHEMA_ONLY,
            severity=Severity.MINOR,
        )


def test_analysis_must_cite_finding():
    with pytest.raises(ValueError, match="must cite"):
        AnalysisClaim(
            claim_id="D-A-0001",
            summary="x",
            cited_finding_ids=[],
            confidence=0.5,
            confidence_basis=ConfidenceBasis.SCHEMA_ONLY,
            kind="phase_characterization",
        )


def test_trend_must_cite_finding_or_trajectory():
    with pytest.raises(ValueError, match="must cite"):
        TrendClaim(
            claim_id="D-T-0001",
            summary="x",
            cited_finding_ids=[],
            cited_trajectories=[],
            confidence=0.5,
            confidence_basis=ConfidenceBasis.SCHEMA_ONLY,
            direction="increasing",
        )


def test_trend_with_only_trajectory_is_valid():
    t = _trend()
    assert not t.cited_finding_ids
    assert t.cited_trajectories


def test_question_requires_at_least_one_finding():
    with pytest.raises(ValueError):
        OpenQuestion(
            question_id="D-Q-0001",
            question="x",
            why_it_matters="y",
            cited_finding_ids=[],
            answer_format_hint="yes_no",
        )


# ---------- intra-output uniqueness ----------


def test_duplicate_claim_ids_across_lists_rejected():
    """Claim IDs are namespaced by prefix; we still reject duplicates across
    lists because downstream agents look up by raw id.
    """
    bad_failure = _failure("D-F-0001")
    bad_failure_clone = _failure("D-F-0001")
    with pytest.raises(ValueError, match="duplicate id"):
        DiagnosisOutput(
            meta=_meta(),
            failures=[bad_failure, bad_failure_clone],
        )


def test_duplicate_question_ids_rejected():
    with pytest.raises(ValueError, match="duplicate id"):
        DiagnosisOutput(
            meta=_meta(),
            open_questions=[_question("D-Q-0001"), _question("D-Q-0001")],
        )


# ---------- happy path ----------


def test_happy_path_all_claim_kinds():
    out = DiagnosisOutput(
        meta=_meta(),
        failures=[_failure()],
        trends=[_trend()],
        analysis=[_analysis()],
        open_questions=[_question()],
    )
    assert len(out.failures) == 1
    assert len(out.trends) == 1
    assert len(out.analysis) == 1
    assert len(out.open_questions) == 1


# ---------- meta.error contract ----------


def test_meta_error_with_empty_lists_ok():
    out = DiagnosisOutput(meta=_meta(error="llm_output_unparseable"))
    assert out.meta.error == "llm_output_unparseable"
    assert out.failures == []


def test_meta_error_with_non_empty_lists_rejected():
    with pytest.raises(ValueError, match="error is set"):
        DiagnosisOutput(
            meta=_meta(error="llm_output_unparseable"),
            failures=[_failure()],
        )


# ---------- TrajectoryRef shape ----------


def test_trajectory_ref_serializes_as_object_not_tuple():
    """Regression guard: previous design used tuple[str, str] which JSON-
    serializes positionally. TrajectoryRef must serialize as a named object.
    """
    ref = TrajectoryRef(run_id="RUN-1", variable="biomass_g_l")
    dumped = ref.model_dump()
    assert dumped == {"run_id": "RUN-1", "variable": "biomass_g_l"}


# ---------- provenance_downgraded default ----------


def test_provenance_downgraded_default_false():
    f = _failure()
    assert f.provenance_downgraded is False
