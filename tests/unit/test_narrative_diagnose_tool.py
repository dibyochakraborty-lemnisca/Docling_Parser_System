"""Plan B Stage 3: get_narrative_observations tool + dispatcher routing
+ narrative-citation integrity in the validator.

Hermetic — scripted client, no live API.
"""

from __future__ import annotations

import json
import uuid
from datetime import datetime
from pathlib import Path

import pytest

from fermdocs.bundle import BundleReader, BundleWriter
from fermdocs_characterize.flags import ProcessFlag
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
from fermdocs_characterize.specs import DictSpecsProvider
from fermdocs_diagnose.agent import DiagnosisAgent
from fermdocs_diagnose.schema import (
    AnalysisClaim,
    ConfidenceBasis,
    DiagnosisMeta,
    DiagnosisOutput,
    FailureClaim,
    OpenQuestion,
)
from fermdocs_diagnose.tools_bundle import DiagnosisToolBundle, make_diagnosis_tools
from fermdocs_diagnose.validators import (
    CitationIntegrityError,
    validate_diagnosis,
)


CHAR_ID = uuid.UUID(int=4242)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _narrative(
    narrative_id: str,
    *,
    tag: NarrativeTag = NarrativeTag.CLOSURE_EVENT,
    text: str = "white cells observed",
    run_id: str | None = "BATCH-01",
    time_h: float | None = 82.0,
    affected: list[str] | None = None,
) -> NarrativeObservation:
    return NarrativeObservation(
        narrative_id=narrative_id,
        tag=tag,
        text=text,
        source_locator=NarrativeSourceLocator(page=3, section="Results"),
        run_id=run_id,
        time_h=time_h,
        affected_variables=affected or ["viability_pct"],
        confidence=0.8,
        extraction_model="gemini-3.1-pro-preview",
    )


def _build_upstream_with_narratives(narratives: list[NarrativeObservation]) -> CharacterizationOutput:
    return CharacterizationOutput(
        meta=Meta(
            schema_version="2.0",
            characterization_version="v1.0.0",
            characterization_id=CHAR_ID,
            generation_timestamp=datetime(2026, 5, 3),
            source_dossier_ids=["EXP-PB-S3"],
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
                run_ids=["BATCH-01"],
            ),
        ],
        trajectories=[
            Trajectory(
                trajectory_id="T-0001",
                run_id="BATCH-01",
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
        narrative_observations=narratives,
    )


@pytest.fixture
def reader_with_narratives(tmp_path: Path) -> BundleReader:
    upstream = _build_upstream_with_narratives(
        [
            _narrative(f"{CHAR_ID}:N-0001", tag=NarrativeTag.CLOSURE_EVENT,
                       text="terminated at 82h, white cells observed", run_id="BATCH-01"),
            _narrative(f"{CHAR_ID}:N-0002", tag=NarrativeTag.INTERVENTION,
                       text="200 mL IPM added at 24h", run_id="BATCH-05", time_h=24.0),
            _narrative(f"{CHAR_ID}:N-0003", tag=NarrativeTag.OBSERVATION,
                       text="exponential growth observed", run_id="BATCH-01", time_h=None),
        ]
    )
    writer = BundleWriter.create(
        tmp_path,
        run_ids=["BATCH-01", "BATCH-05"],
        golden_schema_version="2.0",
        pipeline_version="0.1.0",
    )
    writer.write_dossier({"experiment": {"experiment_id": "EXP-PB-S3"}})
    writer.write_characterization(upstream.model_dump_json())
    bundle_path = writer.finalize()
    return BundleReader(bundle_path)


@pytest.fixture
def tool_bundle(reader_with_narratives: BundleReader) -> DiagnosisToolBundle:
    upstream = _build_upstream_with_narratives(
        [
            _narrative(f"{CHAR_ID}:N-0001", tag=NarrativeTag.CLOSURE_EVENT,
                       text="terminated at 82h, white cells observed", run_id="BATCH-01"),
            _narrative(f"{CHAR_ID}:N-0002", tag=NarrativeTag.INTERVENTION,
                       text="200 mL IPM added at 24h", run_id="BATCH-05", time_h=24.0),
            _narrative(f"{CHAR_ID}:N-0003", tag=NarrativeTag.OBSERVATION,
                       text="exponential growth observed", run_id="BATCH-01", time_h=None),
        ]
    )
    return make_diagnosis_tools(reader_with_narratives, upstream)


# ---------------------------------------------------------------------------
# get_narrative_observations
# ---------------------------------------------------------------------------


def test_get_narrative_observations_returns_all(tool_bundle: DiagnosisToolBundle) -> None:
    out = tool_bundle.get_narrative_observations()
    assert out["total"] == 3
    assert out["truncated"] is False
    assert sorted(out["tags_present"]) == ["closure_event", "intervention", "observation"]
    assert len(out["observations"]) == 3


def test_get_narrative_observations_filter_by_tag(tool_bundle: DiagnosisToolBundle) -> None:
    out = tool_bundle.get_narrative_observations(tag="closure_event")
    assert out["total"] == 1
    assert out["observations"][0]["tag"] == "closure_event"


def test_get_narrative_observations_filter_by_run_id(tool_bundle: DiagnosisToolBundle) -> None:
    out = tool_bundle.get_narrative_observations(run_id="BATCH-01")
    assert out["total"] == 2
    assert all(o["run_id"] == "BATCH-01" for o in out["observations"])


def test_get_narrative_observations_filter_by_variable(tool_bundle: DiagnosisToolBundle) -> None:
    out = tool_bundle.get_narrative_observations(variable="viability_pct")
    # All three test narratives default to viability_pct in affected_variables
    assert out["total"] == 3


def test_get_narrative_observations_filter_no_match(tool_bundle: DiagnosisToolBundle) -> None:
    out = tool_bundle.get_narrative_observations(tag="protocol_note")
    assert out["total"] == 0
    assert out["observations"] == []
    # tags_present still reports all tags in upstream so the agent can pivot
    assert "closure_event" in out["tags_present"]


def test_get_narrative_observations_state_machine_after_submit(
    tool_bundle: DiagnosisToolBundle,
) -> None:
    tool_bundle.submit_diagnosis({"failures": []})
    out = tool_bundle.get_narrative_observations()
    assert out == {"error": "already_finalized", "tool": "get_narrative_observations"}


def test_get_meta_surfaces_narrative_count_and_tags(tool_bundle: DiagnosisToolBundle) -> None:
    meta = tool_bundle.get_meta()
    assert meta["narrative_observations_count"] == 3
    assert "closure_event" in meta["narrative_observation_tags"]
    assert "intervention" in meta["narrative_observation_tags"]


def test_get_meta_zero_when_no_narratives(tmp_path: Path) -> None:
    upstream = _build_upstream_with_narratives([])
    writer = BundleWriter.create(
        tmp_path,
        run_ids=["BATCH-01"],
        golden_schema_version="2.0",
        pipeline_version="0.1.0",
    )
    writer.write_dossier({"experiment": {"experiment_id": "EXP"}})
    writer.write_characterization(upstream.model_dump_json())
    reader = BundleReader(writer.finalize())
    bundle = make_diagnosis_tools(reader, upstream)
    meta = bundle.get_meta()
    assert meta["narrative_observations_count"] == 0
    assert meta["narrative_observation_tags"] == []


# ---------------------------------------------------------------------------
# Validator: narrative citation integrity
# ---------------------------------------------------------------------------


def _diagnosis_meta() -> DiagnosisMeta:
    return DiagnosisMeta(
        schema_version="1.0",
        diagnosis_version="v1.0.0",
        diagnosis_id=uuid.UUID(int=99),
        supersedes_characterization_id=CHAR_ID,
        generation_timestamp=datetime(2026, 5, 3),
        model="claude-opus-4-7",
        provider="anthropic",
    )


def test_validator_keeps_failure_with_known_narrative_citation() -> None:
    upstream = _build_upstream_with_narratives(
        [_narrative(f"{CHAR_ID}:N-0001")]
    )
    f = FailureClaim(
        claim_id="D-F-0001",
        summary="white cells reported at termination",
        cited_narrative_ids=[f"{CHAR_ID}:N-0001"],
        affected_variables=["viability_pct"],
        confidence=0.8,
        confidence_basis=ConfidenceBasis.SCHEMA_ONLY,
        domain_tags=["data_quality"],
        severity=Severity.MAJOR,
    )
    out = DiagnosisOutput(meta=_diagnosis_meta(), failures=[f])
    cleaned = validate_diagnosis(out, upstream=upstream)
    assert len(cleaned.failures) == 1


def test_validator_drops_failure_with_unknown_narrative() -> None:
    upstream = _build_upstream_with_narratives(
        [_narrative(f"{CHAR_ID}:N-0001")]
    )
    f = FailureClaim(
        claim_id="D-F-0001",
        summary="x",
        cited_narrative_ids=[f"{CHAR_ID}:N-9999"],  # not in upstream
        affected_variables=["viability_pct"],
        confidence=0.8,
        confidence_basis=ConfidenceBasis.SCHEMA_ONLY,
        domain_tags=["x"],
        severity=Severity.MINOR,
    )
    out = DiagnosisOutput(meta=_diagnosis_meta(), failures=[f])
    cleaned = validate_diagnosis(out, upstream=upstream)
    assert cleaned.failures == []


def test_validator_strict_mode_raises_on_unknown_narrative() -> None:
    upstream = _build_upstream_with_narratives(
        [_narrative(f"{CHAR_ID}:N-0001")]
    )
    f = FailureClaim(
        claim_id="D-F-0001",
        summary="x",
        cited_narrative_ids=[f"{CHAR_ID}:N-9999"],
        affected_variables=["viability_pct"],
        confidence=0.8,
        confidence_basis=ConfidenceBasis.SCHEMA_ONLY,
        domain_tags=["x"],
        severity=Severity.MINOR,
    )
    out = DiagnosisOutput(meta=_diagnosis_meta(), failures=[f])
    with pytest.raises(CitationIntegrityError):
        validate_diagnosis(
            out, upstream=upstream, drop_unknown_citations=False
        )


def test_validator_open_question_with_unknown_narrative_dropped() -> None:
    upstream = _build_upstream_with_narratives(
        [_narrative(f"{CHAR_ID}:N-0001")]
    )
    q = OpenQuestion(
        question_id="D-Q-0001",
        question="?",
        why_it_matters="?",
        cited_narrative_ids=[f"{CHAR_ID}:N-9999"],
        answer_format_hint="yes_no",
    )
    out = DiagnosisOutput(meta=_diagnosis_meta(), open_questions=[q])
    cleaned = validate_diagnosis(out, upstream=upstream)
    assert cleaned.open_questions == []


# ---------------------------------------------------------------------------
# End-to-end: agent dispatch + emit with narrative citation
# ---------------------------------------------------------------------------


class _ScriptedClient:
    def __init__(self, responses: list[dict]) -> None:
        self._responses = list(responses)
        self.message_logs: list[list[dict]] = []

    def call(self, system: str, messages: list[dict[str, str]]) -> dict:
        self.message_logs.append(list(messages))
        return self._responses.pop(0)


def test_agent_dispatches_get_narrative_observations(reader_with_narratives: BundleReader) -> None:
    upstream = _build_upstream_with_narratives(
        [
            _narrative(f"{CHAR_ID}:N-0001", tag=NarrativeTag.CLOSURE_EVENT,
                       text="terminated at 82h, white cells observed", run_id="BATCH-01"),
        ]
    )
    nid = f"{CHAR_ID}:N-0001"
    fid = f"{CHAR_ID}:F-0001"
    client = _ScriptedClient(
        [
            {"action": "tool_call", "tool": "get_narrative_observations", "args": {"tag": "closure_event"}},
            {
                "action": "emit",
                "failures": [
                    {
                        "summary": "Pigment loss / white cells reported at termination of BATCH-01",
                        "cited_finding_ids": [fid],
                        "cited_narrative_ids": [nid],
                        "affected_variables": ["viability_pct"],
                        "confidence": 0.8,
                        "confidence_basis": "schema_only",
                        "domain_tags": ["data_quality"],
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
        {"experiment": {"experiment_id": "EXP-PB-S3", "process": {"observed": {"organism": "Saccharomyces cerevisiae"}}}},
        upstream,
        bundle=reader_with_narratives,
        diagnosis_id=uuid.UUID(int=11),
        generation_timestamp=datetime(2026, 5, 3),
    )
    assert result.meta.error is None
    assert len(result.failures) == 1
    assert result.failures[0].cited_narrative_ids == [nid]
    # Tool result was fed back as a user message containing the narrative
    last_messages = client.message_logs[-1]
    serialized = "\n".join(m.get("content", "") for m in last_messages)
    assert "white cells" in serialized
