"""Stage 3: spine flip + hard tool-use enforcement + ReAct trajectory persistence.

Covers:
  - Bundle-mode prompt is used when bundle is set
  - Step budget ramps to BUNDLE_MAX_STEPS (20) by default with --bundle
  - Hard enforcement: zero-tool first emit triggers a single retry
  - Enforcement failure (still no tools after retry) returns error DiagnosisOutput
  - Every LLM call + tool result + budget event is captured in
    audit/diagnosis_trace.jsonl
  - submit_diagnosis tool path is honored as a terminator
  - Budget exhaustion flips meta.flags.budget_exhausted
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
    Severity,
    Tier,
    Trajectory,
)
from fermdocs_characterize.specs import DictSpecsProvider, Spec
from fermdocs_diagnose.agent import (
    BUNDLE_MAX_STEPS,
    DEFAULT_MAX_STEPS,
    DiagnosisAgent,
    _BUNDLE_SYSTEM_PROMPT,
    _SYSTEM_PROMPT,
)


CHAR_ID = uuid.UUID(int=314)


def _build_upstream() -> CharacterizationOutput:
    return CharacterizationOutput(
        meta=Meta(
            schema_version="2.0",
            characterization_version="v1.0.0",
            characterization_id=CHAR_ID,
            generation_timestamp=datetime(2026, 5, 1),
            source_dossier_ids=["EXP-S3"],
        ),
        findings=[
            Finding(
                finding_id=f"{CHAR_ID}:F-0001",
                type=FindingType.RANGE_VIOLATION,
                severity=Severity.MAJOR,
                tier=Tier.A,
                summary="biomass excursion",
                confidence=0.8,
                extracted_via=ExtractedVia.DETERMINISTIC,
                evidence_strength=EvidenceStrength(n_observations=5, n_independent_runs=1),
                evidence_observation_ids=["O-1"],
                variables_involved=["biomass_g_l"],
                run_ids=["RUN-X"],
            ),
        ],
        trajectories=[
            Trajectory(
                trajectory_id="T-0001",
                run_id="RUN-X",
                variable="biomass_g_l",
                time_grid=[0.0, 1.0],
                values=[1.0, 2.0],
                imputation_flags=[False, False],
                source_observation_ids=["O-1"],
                unit="g/L",
                quality=1.0,
                data_quality=DataQuality(pct_missing=0.0, pct_imputed=0.0, pct_real=1.0),
            )
        ],
    )


def _build_bundle(tmp_path: Path, upstream: CharacterizationOutput) -> Path:
    writer = BundleWriter.create(
        tmp_path,
        run_ids=["RUN-X"],
        golden_schema_version="2.0",
        pipeline_version="0.1.0",
    )
    writer.write_dossier({"experiment": {"experiment_id": "EXP-S3"}})
    writer.write_characterization(upstream.model_dump_json())
    return writer.finalize()


@pytest.fixture
def reader(tmp_path: Path) -> BundleReader:
    upstream = _build_upstream()
    return BundleReader(_build_bundle(tmp_path, upstream))


class _CapturingClient:
    """Scripted LLM client that records the system prompt it received."""

    def __init__(self, responses: list[dict]) -> None:
        self._responses = list(responses)
        self.systems: list[str] = []
        self.message_logs: list[list[dict]] = []

    def call(self, system: str, messages: list[dict[str, str]]) -> dict:
        self.systems.append(system)
        self.message_logs.append(list(messages))
        if not self._responses:
            raise AssertionError("scripted client out of responses")
        return self._responses.pop(0)


# ---------------------------------------------------------------------------
# Spine flip
# ---------------------------------------------------------------------------


def _good_emit(finding_id: str) -> dict:
    return {
        "action": "emit",
        "failures": [
            {
                "summary": "biomass excursion observed",
                "cited_finding_ids": [finding_id],
                "affected_variables": ["biomass_g_l"],
                "confidence": 0.8,
                "confidence_basis": "schema_only",
                "domain_tags": ["growth"],
                "severity": "major",
            }
        ],
        "trends": [],
        "analysis": [],
        "open_questions": [],
    }


def test_bundle_mode_uses_execute_python_prompt(reader: BundleReader) -> None:
    upstream = _build_upstream()
    fid = f"{CHAR_ID}:F-0001"
    client = _CapturingClient(
        [
            {"action": "tool_call", "tool": "list_runs", "args": {}},
            _good_emit(fid),
        ]
    )
    DiagnosisAgent(client=client).diagnose(
        {"experiment": {"experiment_id": "EXP-S3"}},
        upstream,
        bundle=reader,
        diagnosis_id=uuid.UUID(int=1),
        generation_timestamp=datetime(2026, 5, 2),
    )
    assert client.systems[0] is _BUNDLE_SYSTEM_PROMPT
    assert "execute_python-default" in _BUNDLE_SYSTEM_PROMPT


def test_no_bundle_keeps_legacy_prompt() -> None:
    upstream = _build_upstream()
    fid = f"{CHAR_ID}:F-0001"
    client = _CapturingClient([_good_emit(fid)])
    DiagnosisAgent(client=client).diagnose(
        {"experiment": {"experiment_id": "EXP-S3"}},
        upstream,
        diagnosis_id=uuid.UUID(int=2),
        generation_timestamp=datetime(2026, 5, 2),
    )
    assert client.systems[0] is _SYSTEM_PROMPT


def test_bundle_mode_ramps_step_budget(reader: BundleReader) -> None:
    """Default max_steps becomes BUNDLE_MAX_STEPS when bundle is given."""
    upstream = _build_upstream()
    # Script enough tool calls to exceed legacy DEFAULT_MAX_STEPS=6 but stay
    # under BUNDLE_MAX_STEPS=20. The agent should not bail.
    responses = [
        {"action": "tool_call", "tool": "list_runs", "args": {}}
        for _ in range(8)
    ]
    responses.append(_good_emit(f"{CHAR_ID}:F-0001"))
    client = _CapturingClient(responses)
    result = DiagnosisAgent(client=client).diagnose(
        {"experiment": {"experiment_id": "EXP-S3"}},
        upstream,
        bundle=reader,
        diagnosis_id=uuid.UUID(int=3),
        generation_timestamp=datetime(2026, 5, 2),
    )
    assert result.meta.error is None
    assert len(client.systems) >= DEFAULT_MAX_STEPS + 1


# ---------------------------------------------------------------------------
# Hard tool-use enforcement
# ---------------------------------------------------------------------------


def test_enforcement_retries_when_first_response_has_zero_tools(
    reader: BundleReader,
) -> None:
    upstream = _build_upstream()
    fid = f"{CHAR_ID}:F-0001"
    # First: bare emit (no tools) → triggers retry. Second: tool. Third: emit.
    client = _CapturingClient(
        [
            _good_emit(fid),
            {"action": "tool_call", "tool": "list_runs", "args": {}},
            _good_emit(fid),
        ]
    )
    result = DiagnosisAgent(client=client).diagnose(
        {"experiment": {"experiment_id": "EXP-S3"}},
        upstream,
        bundle=reader,
        diagnosis_id=uuid.UUID(int=4),
        generation_timestamp=datetime(2026, 5, 2),
    )
    assert result.meta.error is None
    # Last user message before the second LLM call should be the retry prompt
    second_call_messages = client.message_logs[1]
    assert any(
        "MUST fetch evidence via tools" in m.get("content", "")
        for m in second_call_messages
    )


def test_enforcement_failure_returns_error_diagnosis(reader: BundleReader) -> None:
    """Two zero-tool responses in a row → error output, never fabricated."""
    upstream = _build_upstream()
    fid = f"{CHAR_ID}:F-0001"
    client = _CapturingClient([_good_emit(fid), _good_emit(fid)])
    result = DiagnosisAgent(client=client).diagnose(
        {"experiment": {"experiment_id": "EXP-S3"}},
        upstream,
        bundle=reader,
        diagnosis_id=uuid.UUID(int=5),
        generation_timestamp=datetime(2026, 5, 2),
    )
    assert result.meta.error == "enforcement_failed:zero_tool_calls"
    assert result.failures == []


def test_enforcement_off_when_no_bundle() -> None:
    """Wave 1 path: zero-tool first emit is a happy path, not enforced."""
    upstream = _build_upstream()
    fid = f"{CHAR_ID}:F-0001"
    client = _CapturingClient([_good_emit(fid)])
    result = DiagnosisAgent(client=client).diagnose(
        {"experiment": {"experiment_id": "EXP-S3"}},
        upstream,
        diagnosis_id=uuid.UUID(int=6),
        generation_timestamp=datetime(2026, 5, 2),
    )
    assert result.meta.error is None


# ---------------------------------------------------------------------------
# ReAct trajectory persistence
# ---------------------------------------------------------------------------


def test_trace_captures_every_llm_call_and_tool_result(reader: BundleReader) -> None:
    upstream = _build_upstream()
    fid = f"{CHAR_ID}:F-0001"
    client = _CapturingClient(
        [
            {"action": "tool_call", "tool": "list_runs", "args": {}},
            _good_emit(fid),
        ]
    )
    DiagnosisAgent(client=client).diagnose(
        {"experiment": {"experiment_id": "EXP-S3"}},
        upstream,
        bundle=reader,
        diagnosis_id=uuid.UUID(int=7),
        generation_timestamp=datetime(2026, 5, 2),
    )
    trace_path = reader.dir / "audit" / "diagnosis_trace.jsonl"
    assert trace_path.exists()
    records = [json.loads(line) for line in trace_path.read_text().splitlines() if line.strip()]
    kinds = [r["kind"] for r in records]
    assert kinds.count("llm_response") == 2
    assert kinds.count("tool_result") == 1
    # tool_result for list_runs should carry the result payload
    tr = next(r for r in records if r["kind"] == "tool_result")
    assert tr["tool"] == "list_runs"
    assert tr["result"]["result"]["run_ids"] == ["RUN-X"]


def test_trace_records_enforcement_event(reader: BundleReader) -> None:
    upstream = _build_upstream()
    fid = f"{CHAR_ID}:F-0001"
    client = _CapturingClient(
        [
            _good_emit(fid),  # zero-tool emit → enforcement retry
            {"action": "tool_call", "tool": "list_runs", "args": {}},
            _good_emit(fid),
        ]
    )
    DiagnosisAgent(client=client).diagnose(
        {"experiment": {"experiment_id": "EXP-S3"}},
        upstream,
        bundle=reader,
        diagnosis_id=uuid.UUID(int=8),
        generation_timestamp=datetime(2026, 5, 2),
    )
    trace_path = reader.dir / "audit" / "diagnosis_trace.jsonl"
    records = [json.loads(line) for line in trace_path.read_text().splitlines() if line.strip()]
    assert any(r["kind"] == "enforcement_retry" for r in records)


# ---------------------------------------------------------------------------
# submit_diagnosis terminator
# ---------------------------------------------------------------------------


def test_submit_diagnosis_terminates_loop(reader: BundleReader) -> None:
    upstream = _build_upstream()
    fid = f"{CHAR_ID}:F-0001"
    payload = {
        "failures": [
            {
                "summary": "biomass dip seen",
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
    }
    client = _CapturingClient(
        [
            {"action": "tool_call", "tool": "list_runs", "args": {}},
            {
                "action": "tool_call",
                "tool": "submit_diagnosis",
                "args": {"payload": payload},
            },
            # Should never be reached:
            {"action": "tool_call", "tool": "list_runs", "args": {}},
        ]
    )
    result = DiagnosisAgent(client=client).diagnose(
        {"experiment": {"experiment_id": "EXP-S3"}},
        upstream,
        bundle=reader,
        diagnosis_id=uuid.UUID(int=9),
        generation_timestamp=datetime(2026, 5, 2),
    )
    assert result.meta.error is None
    assert len(result.failures) == 1
    assert result.failures[0].summary == "biomass dip seen"
    # The third response was never consumed
    assert len(client.message_logs) == 2


# ---------------------------------------------------------------------------
# Budget exhaustion
# ---------------------------------------------------------------------------


def test_budget_exhaustion_flips_meta_flag(tmp_path: Path) -> None:
    """Running until step budget hits should set flags.budget_exhausted=True."""
    upstream = _build_upstream()
    bundle_path = _build_bundle(tmp_path, upstream)
    reader_local = BundleReader(bundle_path)
    # Force a small budget so we exhaust quickly: pass max_steps=2 directly.
    # That short-circuits the bundle-mode ramp (the code only ramps when
    # max_steps == DEFAULT_MAX_STEPS).
    client = _CapturingClient(
        [
            {"action": "tool_call", "tool": "list_runs", "args": {}},
            {"action": "tool_call", "tool": "list_runs", "args": {}},
            {"action": "tool_call", "tool": "list_runs", "args": {}},
        ]
    )
    agent = DiagnosisAgent(client=client, max_steps=2)
    result = agent.diagnose(
        {"experiment": {"experiment_id": "EXP-S3"}},
        upstream,
        bundle=reader_local,
        diagnosis_id=uuid.UUID(int=10),
        generation_timestamp=datetime(2026, 5, 2),
    )
    assert result.meta.error == "step_budget_exhausted"
    meta = json.loads((bundle_path / "meta.json").read_text())
    assert meta["flags"].get("budget_exhausted") is True
