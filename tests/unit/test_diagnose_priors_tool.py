"""Plan A Stage 2: get_priors tool wiring.

Covers:
  - get_priors filters by organism / process_family / variable through the
    tool surface
  - get_meta surfaces process_priors_version + available organisms
  - Wiring through DiagnosisAgent's _dispatch_tool_bundle returns the right
    payload shape
  - The no-priors path returns an empty list with a helpful note
  - State machine still gates after submit_diagnosis (priors tool
    respects the SUBMITTED state)
"""

from __future__ import annotations

import json
import uuid
from datetime import datetime
from pathlib import Path

import pytest

from fermdocs.bundle import BundleReader, BundleWriter
from fermdocs.domain.process_priors import (
    OrganismPriors,
    PriorBound,
    ProcessFamily,
    ProcessPriors,
)
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
from fermdocs_characterize.specs import DictSpecsProvider, Spec
from fermdocs_diagnose.agent import DiagnosisAgent
from fermdocs_diagnose.tools_bundle import (
    DiagnosisToolBundle,
    make_diagnosis_tools,
)


CHAR_ID = uuid.UUID(int=1729)


def _build_upstream() -> CharacterizationOutput:
    return CharacterizationOutput(
        meta=Meta(
            schema_version="2.0",
            characterization_version="v1.0.0",
            characterization_id=CHAR_ID,
            generation_timestamp=datetime(2026, 5, 3),
            source_dossier_ids=["EXP-PA-S2"],
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
                run_ids=["RUN-Y"],
            ),
        ],
        trajectories=[
            Trajectory(
                trajectory_id="T-0001",
                run_id="RUN-Y",
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


def _build_bundle(tmp_path: Path, upstream: CharacterizationOutput) -> Path:
    writer = BundleWriter.create(
        tmp_path,
        run_ids=["RUN-Y"],
        golden_schema_version="2.0",
        pipeline_version="0.1.0",
    )
    writer.write_dossier({"experiment": {"experiment_id": "EXP-PA-S2"}})
    writer.write_characterization(upstream.model_dump_json())
    return writer.finalize()


@pytest.fixture
def reader(tmp_path: Path) -> BundleReader:
    return BundleReader(_build_bundle(tmp_path, _build_upstream()))


@pytest.fixture
def tool_bundle(reader: BundleReader) -> DiagnosisToolBundle:
    upstream = _build_upstream()
    return make_diagnosis_tools(reader, upstream)


@pytest.fixture
def stub_priors() -> ProcessPriors:
    """Minimal hand-built priors for tests that don't want shipped YAML coupling."""
    return ProcessPriors(
        version="1.0",
        organisms=[
            OrganismPriors(
                name="Test organism",
                aliases=["TO", "test_org"],
                process_families=[
                    ProcessFamily(
                        name="test_family",
                        description="stub",
                        priors={
                            "biomass_g_l": PriorBound(
                                range=(50.0, 100.0),
                                typical=80.0,
                                source="Stub 2026",
                                note="Test note",
                            ),
                        },
                    ),
                ],
            ),
        ],
    )


# ---------------------------------------------------------------------------
# Default priors loaded by make_diagnosis_tools
# ---------------------------------------------------------------------------


def test_default_priors_loaded(tool_bundle: DiagnosisToolBundle) -> None:
    """make_diagnosis_tools loads the shipped process_priors.yaml by default."""
    assert tool_bundle.priors is not None
    assert tool_bundle.priors.version == "1.0"
    names = {o.name for o in tool_bundle.priors.organisms}
    assert "Saccharomyces cerevisiae" in names


def test_priors_can_be_disabled(reader: BundleReader) -> None:
    bundle = make_diagnosis_tools(
        reader, _build_upstream(), load_default_priors=False
    )
    assert bundle.priors is None


# ---------------------------------------------------------------------------
# get_priors tool — basic shape and filters
# ---------------------------------------------------------------------------


def test_get_priors_returns_all_when_no_filter(tool_bundle: DiagnosisToolBundle) -> None:
    out = tool_bundle.get_priors()
    assert out["n"] > 5
    assert len(out["available_organisms"]) >= 3
    assert isinstance(out["priors"], list)


def test_get_priors_filters_by_organism(tool_bundle: DiagnosisToolBundle) -> None:
    out = tool_bundle.get_priors(organism="S. cerevisiae")
    assert out["matched_organism"] == "Saccharomyces cerevisiae"
    assert out["n"] > 0
    assert all(p["organism"] == "Saccharomyces cerevisiae" for p in out["priors"])


def test_get_priors_filters_by_organism_and_variable(tool_bundle: DiagnosisToolBundle) -> None:
    out = tool_bundle.get_priors(organism="E. coli", variable="acetate_g_l")
    assert out["n"] == 1
    p = out["priors"][0]
    assert p["organism"] == "Escherichia coli"
    assert p["variable"] == "acetate_g_l"
    assert p["range"] == [0.0, 2.0]
    assert "Eiteman" in p["source"]


def test_get_priors_unknown_organism_returns_empty_with_available(
    tool_bundle: DiagnosisToolBundle,
) -> None:
    out = tool_bundle.get_priors(organism="Bacillus subtilis")
    assert out["n"] == 0
    assert out["matched_organism"] is None
    assert "Saccharomyces cerevisiae" in out["available_organisms"]


def test_get_priors_no_priors_loaded_returns_helpful_payload(
    reader: BundleReader,
) -> None:
    bundle = make_diagnosis_tools(
        reader, _build_upstream(), load_default_priors=False
    )
    out = bundle.get_priors(organism="anything")
    assert out["n"] == 0
    assert out["available_organisms"] == []
    assert "note" in out


def test_get_priors_with_explicit_priors_arg(
    reader: BundleReader, stub_priors: ProcessPriors
) -> None:
    bundle = make_diagnosis_tools(reader, _build_upstream(), priors=stub_priors)
    out = bundle.get_priors(organism="TO", variable="biomass_g_l")
    assert out["n"] == 1
    assert out["priors"][0]["typical"] == 80.0
    assert out["priors"][0]["note"] == "Test note"


# ---------------------------------------------------------------------------
# get_meta surfaces priors version + organisms
# ---------------------------------------------------------------------------


def test_get_meta_surfaces_priors_version(tool_bundle: DiagnosisToolBundle) -> None:
    meta = tool_bundle.get_meta()
    assert meta["process_priors_version"] == "1.0"
    assert "Escherichia coli" in meta["process_priors_organisms"]


def test_get_meta_when_priors_disabled(reader: BundleReader) -> None:
    bundle = make_diagnosis_tools(
        reader, _build_upstream(), load_default_priors=False
    )
    meta = bundle.get_meta()
    assert meta["process_priors_version"] is None
    assert meta["process_priors_organisms"] == []


# ---------------------------------------------------------------------------
# State machine: get_priors gates after submit
# ---------------------------------------------------------------------------


def test_get_priors_gated_after_submit(tool_bundle: DiagnosisToolBundle) -> None:
    tool_bundle.submit_diagnosis({"failures": []})
    out = tool_bundle.get_priors(organism="E. coli")
    assert out == {"error": "already_finalized", "tool": "get_priors"}


# ---------------------------------------------------------------------------
# End-to-end through the agent dispatcher
# ---------------------------------------------------------------------------


class _ScriptedClient:
    def __init__(self, responses: list[dict]) -> None:
        self._responses = list(responses)
        self.message_logs: list[list[dict]] = []

    def call(self, system: str, messages: list[dict[str, str]]) -> dict:
        self.message_logs.append(list(messages))
        return self._responses.pop(0)


def test_agent_dispatcher_routes_get_priors(reader: BundleReader) -> None:
    upstream = _build_upstream()
    fid = f"{CHAR_ID}:F-0001"
    client = _ScriptedClient(
        [
            {
                "action": "tool_call",
                "tool": "get_priors",
                "args": {"organism": "S. cerevisiae", "variable": "ethanol_g_l"},
            },
            {
                "action": "emit",
                "failures": [
                    {
                        "summary": "biomass below typical fed-batch yeast endpoint",
                        "cited_finding_ids": [fid],
                        "affected_variables": ["biomass_g_l"],
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
        {"experiment": {"experiment_id": "EXP-PA-S2"}},
        upstream,
        bundle=reader,
        diagnosis_id=uuid.UUID(int=1),
        generation_timestamp=datetime(2026, 5, 3),
    )
    assert result.meta.error is None
    # The tool result was fed back as a user message containing the prior
    last_messages = client.message_logs[-1]
    serialized = "\n".join(m.get("content", "") for m in last_messages)
    assert "ethanol_g_l" in serialized
    assert "Saccharomyces cerevisiae" in serialized


def test_agent_trace_records_get_priors_call(reader: BundleReader) -> None:
    upstream = _build_upstream()
    fid = f"{CHAR_ID}:F-0001"
    client = _ScriptedClient(
        [
            {
                "action": "tool_call",
                "tool": "get_priors",
                "args": {"organism": "Penicillium"},
            },
            {
                "action": "emit",
                "failures": [
                    {
                        "summary": "noted",
                        "cited_finding_ids": [fid],
                        "affected_variables": ["biomass_g_l"],
                        "confidence": 0.7,
                        "confidence_basis": "schema_only",
                        "domain_tags": ["growth"],
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
    agent.diagnose(
        {"experiment": {"experiment_id": "EXP-PA-S2"}},
        upstream,
        bundle=reader,
        diagnosis_id=uuid.UUID(int=2),
        generation_timestamp=datetime(2026, 5, 3),
    )
    trace_path = reader.dir / "audit" / "diagnosis_trace.jsonl"
    records = [
        json.loads(line) for line in trace_path.read_text().splitlines() if line.strip()
    ]
    tool_results = [r for r in records if r["kind"] == "tool_result"]
    assert any(r["tool"] == "get_priors" for r in tool_results)
