"""Stage 2: bundle-aware diagnosis tool surface.

Covers fetch tools, the state machine on submit_diagnosis, and the agent's
dispatch path when --bundle is in play.
"""

from __future__ import annotations

import json
import uuid
from datetime import datetime, timezone
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


CHAR_ID = uuid.UUID(int=7)


def _build_upstream() -> CharacterizationOutput:
    return CharacterizationOutput(
        meta=Meta(
            schema_version="2.0",
            characterization_version="v1.0.0",
            characterization_id=CHAR_ID,
            generation_timestamp=datetime(2026, 1, 1),
            source_dossier_ids=["EXP-S2"],
        ),
        findings=[
            Finding(
                finding_id=f"{CHAR_ID}:F-0001",
                type=FindingType.RANGE_VIOLATION,
                severity=Severity.MAJOR,
                tier=Tier.A,
                summary="glucose excursion",
                confidence=0.8,
                extracted_via=ExtractedVia.DETERMINISTIC,
                evidence_strength=EvidenceStrength(n_observations=4, n_independent_runs=1),
                evidence_observation_ids=["O-1"],
                variables_involved=["glucose_g_l"],
                run_ids=["RUN-A"],
                statistics={"sigma": 3.1},
            ),
            Finding(
                finding_id=f"{CHAR_ID}:F-0002",
                type=FindingType.RANGE_VIOLATION,
                severity=Severity.MINOR,
                tier=Tier.A,
                summary="biomass dip",
                confidence=0.7,
                extracted_via=ExtractedVia.DETERMINISTIC,
                evidence_strength=EvidenceStrength(n_observations=2, n_independent_runs=1),
                evidence_observation_ids=["O-2"],
                variables_involved=["biomass_g_l"],
                run_ids=["RUN-B"],
                statistics={"sigma": 2.1},
            ),
        ],
        trajectories=[
            Trajectory(
                trajectory_id="T-0001",
                run_id="RUN-A",
                variable="glucose_g_l",
                time_grid=[0.0, 1.0, 2.0, 3.0],
                values=[10.0, 12.0, 15.0, 8.0],
                imputation_flags=[False, False, False, False],
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
        run_ids=["RUN-A", "RUN-B"],
        golden_schema_version="2.0",
        pipeline_version="0.1.0",
    )
    writer.write_dossier({"experiment": {"experiment_id": "EXP-S2"}})
    writer.write_characterization(upstream.model_dump_json())
    return writer.finalize()


@pytest.fixture
def bundle_reader(tmp_path: Path) -> BundleReader:
    upstream = _build_upstream()
    bundle_path = _build_bundle(tmp_path, upstream)
    return BundleReader(bundle_path)


@pytest.fixture
def tool_bundle(tmp_path: Path, bundle_reader: BundleReader) -> DiagnosisToolBundle:
    upstream = _build_upstream()
    specs = DictSpecsProvider(
        {"glucose_g_l": Spec(nominal=10.0, std_dev=1.0, unit="g/L", provenance="dossier")}
    )
    return make_diagnosis_tools(bundle_reader, upstream, specs=specs)


# ---------------------------------------------------------------------------
# Fetch tools
# ---------------------------------------------------------------------------


def test_list_runs(tool_bundle: DiagnosisToolBundle) -> None:
    assert tool_bundle.list_runs() == {"run_ids": ["RUN-A", "RUN-B"]}


def test_get_meta_carries_versions(tool_bundle: DiagnosisToolBundle) -> None:
    meta = tool_bundle.get_meta()
    assert meta["bundle_schema_version"] == "1.0"
    assert meta["golden_schema_version"] == "2.0"
    assert "RUN-A" in meta["run_ids"]
    # observations_csv_path may be None when the bundle was built without one
    # (this fixture's bundle has no observations.csv); the column hint is
    # always present so the agent knows the schema.
    assert "observations_csv_columns" in meta
    assert meta["observations_csv_columns"][0] == "run_id"


def test_get_meta_exposes_observations_csv_when_present(tmp_path: Path) -> None:
    upstream = _build_upstream()
    writer = BundleWriter.create(
        tmp_path,
        run_ids=["RUN-A"],
        golden_schema_version="2.0",
        pipeline_version="0.1.0",
    )
    writer.write_dossier({"experiment": {"experiment_id": "EXP-S2"}})
    writer.write_characterization(upstream.model_dump_json())
    writer.write_observations_csv(
        [{"run_id": "RUN-A", "variable": "glucose_g_l", "time_h": 0.0, "value": 10.0, "imputed": False, "unit": "g/L"}]
    )
    bundle_path = writer.finalize()
    reader = BundleReader(bundle_path)
    bundle_with_csv = make_diagnosis_tools(reader, upstream)
    meta = bundle_with_csv.get_meta()
    assert meta["observations_csv_path"] is not None
    assert meta["observations_csv_path"].endswith("observations.csv")


def test_get_findings_returns_all_when_unfiltered(tool_bundle: DiagnosisToolBundle) -> None:
    out = tool_bundle.get_findings()
    assert out["total"] == 2
    assert out["truncated"] is False
    assert len(out["findings"]) == 2


def test_get_findings_filters_by_run_id(tool_bundle: DiagnosisToolBundle) -> None:
    out = tool_bundle.get_findings(run_id="RUN-A")
    assert out["total"] == 1
    assert out["findings"][0]["finding_id"].endswith(":F-0001")


def test_get_findings_filters_by_severity_and_tier(tool_bundle: DiagnosisToolBundle) -> None:
    out = tool_bundle.get_findings(severity="major", tier="A")
    assert out["total"] == 1
    assert out["findings"][0]["severity"] == "major"
    assert out["findings"][0]["tier"] == "A"


def test_get_findings_filters_by_finding_id(tool_bundle: DiagnosisToolBundle) -> None:
    out = tool_bundle.get_findings(finding_id=f"{CHAR_ID}:F-0002")
    assert out["total"] == 1
    assert out["findings"][0]["summary"] == "biomass dip"


def test_get_specs_returns_spec_payload(tool_bundle: DiagnosisToolBundle) -> None:
    out = tool_bundle.get_specs("glucose_g_l")
    assert out["spec"]["nominal"] == 10.0
    assert out["spec"]["std_dev"] == 1.0


def test_get_specs_unknown_variable(tool_bundle: DiagnosisToolBundle) -> None:
    out = tool_bundle.get_specs("unobtainium")
    assert out == {"variable": "unobtainium", "spec": None}


def test_get_timecourse_returns_full_grid(tool_bundle: DiagnosisToolBundle) -> None:
    out = tool_bundle.get_timecourse("RUN-A", "glucose_g_l")
    assert out["unit"] == "g/L"
    assert out["n_points"] == 4
    assert out["values"] == [10.0, 12.0, 15.0, 8.0]
    assert out["truncated"] is False


def test_get_timecourse_filters_time_range(tool_bundle: DiagnosisToolBundle) -> None:
    out = tool_bundle.get_timecourse("RUN-A", "glucose_g_l", time_range_h=[1.0, 2.5])
    assert out["time_grid"] == [1.0, 2.0]
    assert out["values"] == [12.0, 15.0]


def test_get_timecourse_truncates_at_max_points(tool_bundle: DiagnosisToolBundle) -> None:
    out = tool_bundle.get_timecourse("RUN-A", "glucose_g_l", max_points=2)
    assert out["truncated"] is True
    assert out["n_points"] == 2


def test_get_timecourse_unknown_run(tool_bundle: DiagnosisToolBundle) -> None:
    out = tool_bundle.get_timecourse("RUN-Z", "glucose_g_l")
    assert out["error"] == "trajectory_not_found"
    assert {"run_id": "RUN-A", "variable": "glucose_g_l"} in out["available"]


# ---------------------------------------------------------------------------
# State machine
# ---------------------------------------------------------------------------


def test_submit_diagnosis_first_call_succeeds(tool_bundle: DiagnosisToolBundle) -> None:
    out = tool_bundle.submit_diagnosis({"failures": [], "trends": []})
    assert out == {"ok": True}
    assert tool_bundle.state.submitted is True


def test_submit_diagnosis_idempotent_on_same_payload(tool_bundle: DiagnosisToolBundle) -> None:
    payload = {"failures": []}
    tool_bundle.submit_diagnosis(payload)
    out = tool_bundle.submit_diagnosis(payload)
    assert out == {"ok": True, "idempotent": True}


def test_submit_diagnosis_rejects_different_payload(tool_bundle: DiagnosisToolBundle) -> None:
    tool_bundle.submit_diagnosis({"failures": []})
    out = tool_bundle.submit_diagnosis({"failures": [{"x": 1}]})
    assert out == {"error": "diagnosis_already_submitted"}


def test_tool_call_after_submit_is_gated(tool_bundle: DiagnosisToolBundle) -> None:
    tool_bundle.submit_diagnosis({"failures": []})
    out = tool_bundle.list_runs()
    assert out == {"error": "already_finalized", "tool": "list_runs"}


def test_execute_python_after_submit_is_gated(tool_bundle: DiagnosisToolBundle) -> None:
    tool_bundle.submit_diagnosis({"failures": []})
    out = tool_bundle.execute_python("print(1)")
    assert out == {"error": "already_finalized", "tool": "execute_python"}


# ---------------------------------------------------------------------------
# Agent integration: tool dispatch routes through the bundle when bundle is set
# ---------------------------------------------------------------------------


class _ScriptedClient:
    def __init__(self, responses: list[dict]) -> None:
        self._responses = list(responses)
        self.calls: list[list[dict]] = []

    def call(self, system: str, messages: list[dict[str, str]]) -> dict:
        self.calls.append(list(messages))
        return self._responses.pop(0)


def test_agent_with_bundle_dispatches_to_bundle_tools(
    tmp_path: Path, bundle_reader: BundleReader
) -> None:
    upstream = _build_upstream()
    finding_id = f"{CHAR_ID}:F-0001"

    client = _ScriptedClient(
        [
            {"action": "tool_call", "tool": "list_runs", "args": {}},
            {
                "action": "emit",
                "failures": [
                    {
                        "summary": "glucose dip observed",
                        "cited_finding_ids": [finding_id],
                        "affected_variables": ["glucose_g_l"],
                        "confidence": 0.8,
                        "confidence_basis": "schema_only",
                        "domain_tags": ["metabolism"],
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
        {"experiment": {"experiment_id": "EXP-S2"}},
        upstream,
        bundle=bundle_reader,
        diagnosis_id=uuid.UUID(int=11),
        generation_timestamp=datetime(2026, 1, 2),
    )
    # Tool result should have been fed back as a user message containing run_ids.
    assert any(
        '"run_ids"' in m.get("content", "") for m in client.calls[-1]
    )
    assert result.meta.error is None
    assert len(result.failures) == 1


def test_agent_writes_python_trace_when_bundle_set(
    tmp_path: Path, bundle_reader: BundleReader
) -> None:
    upstream = _build_upstream()
    client = _ScriptedClient(
        [
            {
                "action": "tool_call",
                "tool": "execute_python",
                "args": {"code": "print(2 + 2)"},
            },
            {
                "action": "emit",
                "failures": [],
                "trends": [],
                "analysis": [],
                "open_questions": [],
            },
        ]
    )
    agent = DiagnosisAgent(client=client, max_steps=4)
    agent.diagnose(
        {"experiment": {"experiment_id": "EXP-S2"}},
        upstream,
        bundle=bundle_reader,
        diagnosis_id=uuid.UUID(int=12),
        generation_timestamp=datetime(2026, 1, 3),
    )
    trace_path = bundle_reader.dir / "audit" / "diagnosis_trace.jsonl"
    assert trace_path.exists()
    records = [
        json.loads(line)
        for line in trace_path.read_text().splitlines()
        if line.strip()
    ]
    python_calls = [r for r in records if r["kind"] == "python_call"]
    assert len(python_calls) == 1
    assert "2 + 2" in python_calls[0]["code"]


def test_agent_without_bundle_uses_legacy_tools(tmp_path: Path) -> None:
    """Wave 1 path is preserved: no bundle → no bundle-aware dispatch."""
    upstream = _build_upstream()
    finding_id = f"{CHAR_ID}:F-0001"
    client = _ScriptedClient(
        [
            {
                "action": "tool_call",
                "tool": "get_finding",
                "args": {"finding_id": finding_id},
            },
            {
                "action": "emit",
                "failures": [
                    {
                        "summary": "noted",
                        "cited_finding_ids": [finding_id],
                        "affected_variables": ["glucose_g_l"],
                        "confidence": 0.7,
                        "confidence_basis": "schema_only",
                        "domain_tags": ["metabolism"],
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
        {"experiment": {"experiment_id": "EXP-S2"}},
        upstream,
        diagnosis_id=uuid.UUID(int=13),
        generation_timestamp=datetime(2026, 1, 4),
    )
    assert result.meta.error is None
    # The legacy tool result for get_finding should mention the cited id
    last_messages = client.calls[-1]
    assert any(finding_id in m.get("content", "") for m in last_messages)
