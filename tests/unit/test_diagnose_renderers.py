"""Renderer tests. Snapshot-style — assert structure and presence of key bits,
not exact byte-for-byte output, so cosmetic tweaks don't break the suite.
"""

from __future__ import annotations

import uuid
from datetime import datetime

import pytest

from fermdocs_characterize.schema import Severity, TimeWindow
from fermdocs_diagnose.renderers import (
    render_analysis_md,
    render_diagnosis_md,
    render_failures_md,
    render_questions_md,
    render_trends_md,
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


def _meta() -> DiagnosisMeta:
    return DiagnosisMeta(
        schema_version="1.0",
        diagnosis_version="v1.0.0",
        diagnosis_id=uuid.UUID(int=1),
        supersedes_characterization_id=uuid.UUID(int=42),
        generation_timestamp=datetime(2026, 5, 2),
        model="claude-opus-4-7",
        provider="anthropic",
    )


def _failure() -> FailureClaim:
    return FailureClaim(
        claim_id="D-F-0001",
        summary="biomass plateau between 40h and 60h",
        cited_finding_ids=["F-0001", "F-0002"],
        affected_variables=["biomass_g_l"],
        confidence=0.78,
        confidence_basis=ConfidenceBasis.SCHEMA_ONLY,
        domain_tags=["growth"],
        severity=Severity.MAJOR,
        time_window=TimeWindow(start=40, end=60),
    )


def _trend() -> TrendClaim:
    return TrendClaim(
        claim_id="D-T-0001",
        summary="DO declines after 30h",
        cited_trajectories=[TrajectoryRef(run_id="RUN-1", variable="DO")],
        confidence=0.7,
        confidence_basis=ConfidenceBasis.SCHEMA_ONLY,
        direction="decreasing",
    )


def _analysis() -> AnalysisClaim:
    return AnalysisClaim(
        claim_id="D-A-0001",
        summary="early plateau aligns with stationary phase",
        cited_finding_ids=["F-0001"],
        confidence=0.6,
        confidence_basis=ConfidenceBasis.SCHEMA_ONLY,
        kind="phase_characterization",
    )


def _question() -> OpenQuestion:
    return OpenQuestion(
        question_id="D-Q-0001",
        question="Was sampling cadence reduced in the 40h-60h window?",
        why_it_matters="determines whether the plateau is real or an artifact",
        cited_finding_ids=["F-0001"],
        answer_format_hint="yes_no",
        domain_tags=["data_quality"],
    )


# ---------- empty-state rendering ----------


def test_failures_md_empty():
    out = DiagnosisOutput(meta=_meta())
    rendered = render_failures_md(out)
    assert "No failures" in rendered


def test_trends_md_empty():
    out = DiagnosisOutput(meta=_meta())
    assert "No trends" in render_trends_md(out)


def test_analysis_md_empty():
    out = DiagnosisOutput(meta=_meta())
    assert "No analysis" in render_analysis_md(out)


def test_questions_md_empty():
    out = DiagnosisOutput(meta=_meta())
    assert "No open questions" in render_questions_md(out)


# ---------- populated rendering ----------


def test_failures_md_includes_id_severity_summary():
    out = DiagnosisOutput(meta=_meta(), failures=[_failure()])
    rendered = render_failures_md(out)
    assert "D-F-0001" in rendered
    assert "major" in rendered
    assert "biomass plateau" in rendered
    assert "F-0001" in rendered
    assert "F-0002" in rendered


def test_failures_md_renders_time_window():
    out = DiagnosisOutput(meta=_meta(), failures=[_failure()])
    rendered = render_failures_md(out)
    assert "40" in rendered and "60" in rendered


def test_trends_md_includes_direction_and_trajectory():
    out = DiagnosisOutput(meta=_meta(), trends=[_trend()])
    rendered = render_trends_md(out)
    assert "decreasing" in rendered
    assert "RUN-1" in rendered
    assert "DO" in rendered


def test_analysis_md_includes_kind():
    out = DiagnosisOutput(meta=_meta(), analysis=[_analysis()])
    rendered = render_analysis_md(out)
    assert "phase_characterization" in rendered


def test_questions_md_includes_why_it_matters():
    out = DiagnosisOutput(meta=_meta(), open_questions=[_question()])
    rendered = render_questions_md(out)
    assert "why_it_matters" in rendered
    assert "artifact" in rendered


# ---------- combined report ----------


def test_diagnosis_md_combines_all_sections():
    out = DiagnosisOutput(
        meta=_meta(),
        failures=[_failure()],
        trends=[_trend()],
        analysis=[_analysis()],
        open_questions=[_question()],
        narrative="Run shows an early biomass plateau.",
    )
    rendered = render_diagnosis_md(out)
    assert "Diagnosis Report" in rendered
    assert "diagnosis_id" in rendered
    assert "Narrative" in rendered
    assert "early biomass plateau" in rendered
    assert "D-F-0001" in rendered
    assert "D-T-0001" in rendered
    assert "D-A-0001" in rendered
    assert "D-Q-0001" in rendered


def test_diagnosis_md_surfaces_meta_error():
    """When the agent failed, the combined report must say so loudly."""
    out = DiagnosisOutput(meta=_meta().model_copy(update={"error": "llm_call_failed:RuntimeError"}))
    rendered = render_diagnosis_md(out)
    assert "error" in rendered.lower()
    assert "llm_call_failed" in rendered


# ---------- citation truncation ----------


def test_long_citation_list_truncated_with_more_indicator():
    claim = FailureClaim(
        claim_id="D-F-0001",
        summary="x",
        cited_finding_ids=[f"F-{i:04d}" for i in range(10)],
        confidence=0.5,
        confidence_basis=ConfidenceBasis.SCHEMA_ONLY,
        severity=Severity.MINOR,
    )
    out = DiagnosisOutput(meta=_meta(), failures=[claim])
    rendered = render_failures_md(out)
    assert "+5 more" in rendered  # 10 cited, top-5 shown


# ---------- provenance downgrade surfaces ----------


def test_provenance_downgraded_visible_in_render():
    claim = _failure().model_copy(update={"provenance_downgraded": True})
    out = DiagnosisOutput(meta=_meta(), failures=[claim])
    rendered = render_failures_md(out)
    assert "provenance_downgraded" in rendered
