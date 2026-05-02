"""Plan B Stage 1: schema additions for narrative observations.

Covers:
  - NarrativeObservation Pydantic shape + ID validator
  - NarrativeTag closed enum
  - CharacterizationOutput namespacing + duplicate-id checks include narrative
  - cited_narrative_ids on every claim type
  - Citation validators on FailureClaim / TrendClaim / AnalysisClaim / OpenQuestion
    accept narrative-only citations
  - Bundle persistence: narrative_observations.json round-trip + missing-file
    backward compat (returns "[]")
  - Existing fixtures still pass with the new optional field defaulting empty
"""

from __future__ import annotations

import json
import uuid
from datetime import datetime
from pathlib import Path

import pytest

from fermdocs.bundle import BundleReader, BundleWriter
from fermdocs_characterize.schema import (
    CharacterizationOutput,
    DataQuality,
    EvidenceStrength,
    ExtractedVia,
    Finding,
    FindingType,
    Meta,
    NarrativeObservation,
    NarrativeSourceLocator,
    NarrativeTag,
    Severity,
    Tier,
    Trajectory,
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


CHAR_ID = uuid.UUID(int=11)


# ---------------------------------------------------------------------------
# NarrativeObservation
# ---------------------------------------------------------------------------


def test_narrative_observation_minimal_shape() -> None:
    n = NarrativeObservation(
        narrative_id=f"{CHAR_ID}:N-0001",
        tag=NarrativeTag.CLOSURE_EVENT,
        text="terminated at 82h, white cells observed",
        confidence=0.8,
        extraction_model="gemini-3.1-pro-preview",
    )
    assert n.tag == NarrativeTag.CLOSURE_EVENT
    assert n.run_id is None
    assert n.affected_variables == []


def test_narrative_observation_id_must_contain_marker() -> None:
    with pytest.raises(ValueError, match=":N-"):
        NarrativeObservation(
            narrative_id="bad-id",
            tag=NarrativeTag.OBSERVATION,
            text="x",
            confidence=0.5,
            extraction_model="m",
        )


def test_narrative_confidence_capped_at_0_85() -> None:
    with pytest.raises(ValueError):
        NarrativeObservation(
            narrative_id=f"{CHAR_ID}:N-0001",
            tag=NarrativeTag.OBSERVATION,
            text="x",
            confidence=0.99,
            extraction_model="m",
        )


def test_narrative_text_required_nonempty() -> None:
    with pytest.raises(ValueError):
        NarrativeObservation(
            narrative_id=f"{CHAR_ID}:N-0001",
            tag=NarrativeTag.OBSERVATION,
            text="",
            confidence=0.5,
            extraction_model="m",
        )


def test_narrative_source_locator_optional_fields() -> None:
    loc = NarrativeSourceLocator(page=3, section="Results", paragraph_index=4)
    assert loc.page == 3
    assert loc.section == "Results"
    # All-optional construction:
    NarrativeSourceLocator()


# ---------------------------------------------------------------------------
# CharacterizationOutput integration
# ---------------------------------------------------------------------------


def _meta() -> Meta:
    return Meta(
        schema_version="2.0",
        characterization_version="v1.0.0",
        characterization_id=CHAR_ID,
        generation_timestamp=datetime(2026, 5, 3),
        source_dossier_ids=["EXP-PB-S1"],
    )


def test_characterization_output_defaults_narrative_to_empty() -> None:
    out = CharacterizationOutput(meta=_meta())
    assert out.narrative_observations == []


def test_characterization_output_namespaces_narrative_ids() -> None:
    n = NarrativeObservation(
        narrative_id=f"{CHAR_ID}:N-0001",
        tag=NarrativeTag.CLOSURE_EVENT,
        text="x",
        confidence=0.7,
        extraction_model="m",
    )
    out = CharacterizationOutput(meta=_meta(), narrative_observations=[n])
    assert out.narrative_observations[0].narrative_id == f"{CHAR_ID}:N-0001"


def test_characterization_output_rejects_unnamespaced_narrative_id() -> None:
    other_uuid = uuid.UUID(int=99)
    n = NarrativeObservation(
        narrative_id=f"{other_uuid}:N-0001",  # wrong namespace
        tag=NarrativeTag.OBSERVATION,
        text="x",
        confidence=0.5,
        extraction_model="m",
    )
    with pytest.raises(ValueError, match="not namespaced"):
        CharacterizationOutput(meta=_meta(), narrative_observations=[n])


def test_characterization_output_dedup_includes_narrative() -> None:
    n1 = NarrativeObservation(
        narrative_id=f"{CHAR_ID}:N-0001",
        tag=NarrativeTag.OBSERVATION,
        text="a",
        confidence=0.5,
        extraction_model="m",
    )
    n2 = NarrativeObservation(
        narrative_id=f"{CHAR_ID}:N-0001",  # duplicate
        tag=NarrativeTag.DEVIATION,
        text="b",
        confidence=0.5,
        extraction_model="m",
    )
    with pytest.raises(ValueError, match="duplicate id"):
        CharacterizationOutput(meta=_meta(), narrative_observations=[n1, n2])


# ---------------------------------------------------------------------------
# Claim citation: narrative-only is sufficient
# ---------------------------------------------------------------------------


def test_failure_claim_accepts_narrative_only_citation() -> None:
    # No findings, no trajectories — just a narrative reference. Should pass.
    f = FailureClaim(
        claim_id="D-F-0001",
        summary="white cells observed at end of run",
        cited_narrative_ids=[f"{CHAR_ID}:N-0001"],
        affected_variables=["biomass_g_l"],
        confidence=0.8,
        confidence_basis=ConfidenceBasis.SCHEMA_ONLY,
        domain_tags=["data_quality"],
        severity=Severity.MAJOR,
    )
    assert f.cited_narrative_ids == [f"{CHAR_ID}:N-0001"]


def test_failure_claim_still_rejects_zero_citations() -> None:
    with pytest.raises(ValueError, match="must cite"):
        FailureClaim(
            claim_id="D-F-0001",
            summary="x",
            confidence=0.5,
            confidence_basis=ConfidenceBasis.SCHEMA_ONLY,
            severity=Severity.MINOR,
        )


def test_trend_claim_accepts_narrative_only_citation() -> None:
    t = TrendClaim(
        claim_id="D-T-0001",
        summary="report mentions late-phase pigment loss",
        cited_narrative_ids=[f"{CHAR_ID}:N-0001"],
        affected_variables=["viability_pct"],
        confidence=0.7,
        confidence_basis=ConfidenceBasis.SCHEMA_ONLY,
        direction="decreasing",
    )
    assert t.cited_narrative_ids


def test_analysis_claim_accepts_narrative_only_citation() -> None:
    a = AnalysisClaim(
        claim_id="D-A-0001",
        summary="report flags carotenoid yield concerns",
        cited_narrative_ids=[f"{CHAR_ID}:N-0001"],
        confidence=0.7,
        confidence_basis=ConfidenceBasis.SCHEMA_ONLY,
        kind="data_quality_caveat",
    )
    assert a.cited_narrative_ids


def test_open_question_accepts_narrative_only_citation() -> None:
    q = OpenQuestion(
        question_id="D-Q-0001",
        question="Was carotenoid concentration measured at termination?",
        why_it_matters="Prose mentions pigment loss; need yield data to confirm.",
        cited_narrative_ids=[f"{CHAR_ID}:N-0001"],
        answer_format_hint="yes_no",
    )
    assert q.cited_narrative_ids


def test_open_question_rejects_zero_citations() -> None:
    with pytest.raises(ValueError, match="must cite"):
        OpenQuestion(
            question_id="D-Q-0001",
            question="?",
            why_it_matters="?",
            answer_format_hint="yes_no",
        )


# ---------------------------------------------------------------------------
# Bundle persistence
# ---------------------------------------------------------------------------


def test_bundle_writes_and_reads_narrative_observations(tmp_path: Path) -> None:
    writer = BundleWriter.create(
        tmp_path,
        run_ids=["RUN-N"],
        golden_schema_version="2.0",
        pipeline_version="0.1.0",
    )
    writer.write_dossier({"experiment": {"experiment_id": "EXP-PB-S1"}})
    writer.write_characterization("{}")
    payload = json.dumps(
        [
            {
                "narrative_id": f"{CHAR_ID}:N-0001",
                "tag": "closure_event",
                "text": "white cells",
                "source_locator": {"page": 4, "section": "Results"},
                "run_id": "RUN-N",
                "time_h": 82.0,
                "affected_variables": ["viability_pct"],
                "confidence": 0.8,
                "extraction_model": "gemini-3.1-pro-preview",
            }
        ],
        indent=2,
    )
    writer.write_narrative_observations(payload)
    bundle_path = writer.finalize()

    reader = BundleReader(bundle_path)
    assert reader.has_narrative_observations()
    raw = reader.get_narrative_observations_json()
    parsed = json.loads(raw)
    assert len(parsed) == 1
    assert parsed[0]["tag"] == "closure_event"


def test_bundle_reader_returns_empty_when_no_narrative_file(tmp_path: Path) -> None:
    """Backward-compat: bundles produced before Plan B Stage 1 (no
    narrative_observations.json) read as narrative-empty rather than
    raising."""
    writer = BundleWriter.create(
        tmp_path,
        run_ids=["RUN-N"],
        golden_schema_version="2.0",
        pipeline_version="0.1.0",
    )
    writer.write_dossier({"experiment": {"experiment_id": "EXP"}})
    writer.write_characterization("{}")
    bundle_path = writer.finalize()

    reader = BundleReader(bundle_path)
    assert reader.has_narrative_observations() is False
    assert reader.get_narrative_observations_json() == "[]"


# ---------------------------------------------------------------------------
# Round-trip: model → JSON → model
# ---------------------------------------------------------------------------


def test_narrative_observation_roundtrip_through_json() -> None:
    n = NarrativeObservation(
        narrative_id=f"{CHAR_ID}:N-0001",
        tag=NarrativeTag.INTERVENTION,
        text="200 mL IPM added at 24h",
        source_locator=NarrativeSourceLocator(page=11, section="Procedure"),
        run_id="BATCH-05",
        time_h=24.0,
        affected_variables=["wcw_g_l"],
        confidence=0.75,
        extraction_model="gemini-3.1-pro-preview",
    )
    payload = n.model_dump_json()
    restored = NarrativeObservation.model_validate_json(payload)
    assert restored == n
