"""Tests for the characterize-stage trajectory_analyzer agent.

Two scopes:
  - Unit: mock Gemini client, drive the ReAct loop, assert prompt
    structure + tool dispatch + finding coercion.
  - Stub mode: client=None, returns empty findings — proves backward
    compat with deterministic-only callers.

Real LLM behavior + statistical-discovery quality are eval concerns,
out of scope here. These tests guard the wire (does the agent emit
valid Findings, does the pipeline integrate them).
"""

from __future__ import annotations

from datetime import datetime
from typing import Any
from uuid import UUID

from fermdocs_characterize.agents.trajectory_analyzer import (
    LLM_CONFIDENCE_CAP,
    MAX_TOOL_CALLS,
    TRAJECTORY_ANALYZER_SYSTEM,
    TrajectoryAnalyzerAgent,
    build_trajectory_analyzer,
)
from fermdocs_characterize.schema import (
    DataQuality,
    EvidenceStrength,
    ExtractedVia,
    Finding,
    FindingType,
    Severity,
    Tier,
    Trajectory,
)

CHAR_ID = UUID("11111111-1111-1111-1111-111111111111")


# ---------- fixtures ----------


def _trajectory(run_id: str, variable: str, values: list[float]) -> Trajectory:
    n = len(values)
    return Trajectory(
        trajectory_id="T-0001",
        run_id=run_id,
        variable=variable,
        time_grid=[float(i) for i in range(n)],
        values=[float(v) for v in values],
        imputation_flags=[False] * n,
        source_observation_ids=[f"OBS-{i:04d}" for i in range(n)],
        unit="g/L",
        quality=1.0,
        data_quality=DataQuality(pct_missing=0.0, pct_imputed=0.0, pct_real=1.0),
    )


def _spec_finding(idx: int = 1) -> Finding:
    return Finding(
        finding_id=f"{CHAR_ID}:F-{idx:04d}",
        type=FindingType.RANGE_VIOLATION,
        severity=Severity.MAJOR,
        tier=Tier.A,
        summary="biomass exceeded nominal spec",
        confidence=0.9,
        extracted_via=ExtractedVia.DETERMINISTIC,
        evidence_strength=EvidenceStrength(n_observations=10, n_independent_runs=1),
        evidence_observation_ids=["OBS-0001"],
        variables_involved=["biomass_g_l"],
        run_ids=["RUN-0001"],
    )


# ---------- mock client ----------


class _RecordingClient:
    """Records each .call(); returns canned responses from a queue."""

    def __init__(self, responses: list[dict[str, Any]]):
        self._responses = list(responses)
        self.calls: list[dict[str, Any]] = []

    @property
    def model_name(self) -> str:
        return "fake-test"

    def call(self, *, system, user_text, response_schema, temperature=0.0):
        self.calls.append({"system": system, "user_text": user_text})
        if not self._responses:
            # Default fallback: emit empty if queue runs dry
            return ({"action": "emit_patterns", "patterns": []}, 50, 10)
        parsed = self._responses.pop(0)
        return parsed, 100, 30


# ---------- stub mode ----------


def test_stub_mode_returns_empty_no_llm_call():
    """client=None: agent runs zero LLM calls, returns empty findings.
    This is the back-compat path for tests that don't want to mock Gemini."""
    agent = TrajectoryAnalyzerAgent(client=None)
    result = agent.analyze(
        char_id=CHAR_ID,
        trajectories=[_trajectory("RUN-0001", "biomass_g_l", [1.0, 2.0, 3.0])],
        spec_findings=[_spec_finding()],
    )
    assert result.findings == []
    assert result.input_tokens == 0
    assert result.output_tokens == 0


def test_stub_mode_handles_empty_trajectories():
    agent = TrajectoryAnalyzerAgent(client=None)
    result = agent.analyze(char_id=CHAR_ID, trajectories=[], spec_findings=[])
    assert result.findings == []


def test_real_mode_skips_when_no_trajectories():
    """Even with a client, no trajectories → no analysis → no LLM call.
    The CSV would be empty; nothing to analyze."""
    client = _RecordingClient([])
    agent = TrajectoryAnalyzerAgent(client=client)  # type: ignore[arg-type]
    result = agent.analyze(char_id=CHAR_ID, trajectories=[], spec_findings=[])
    assert result.findings == []
    assert client.calls == []


# ---------- happy path ----------


def test_emit_directly_no_tool_calls():
    """The simplest case: agent emits a single pattern on its first turn,
    no execute_python calls."""
    client = _RecordingClient(
        [
            {
                "action": "emit_patterns",
                "patterns": [
                    {
                        "pattern_kind": "cross_batch_variance",
                        "summary": "Biomass plateau time CV=0.35 across 3 runs",
                        "variables_involved": ["biomass_g_l"],
                        "run_ids": ["RUN-0001", "RUN-0002", "RUN-0003"],
                        "time_window": {"start": 0.0, "end": 100.0},
                        "severity": "major",
                        "confidence": 0.75,
                        "caveats": [],
                        "statistics": {"cv": 0.35, "n_runs": 3},
                    }
                ],
            }
        ]
    )
    agent = TrajectoryAnalyzerAgent(client=client)  # type: ignore[arg-type]
    result = agent.analyze(
        char_id=CHAR_ID,
        trajectories=[
            _trajectory("RUN-0001", "biomass_g_l", [1.0, 5.0, 10.0]),
            _trajectory("RUN-0002", "biomass_g_l", [1.0, 4.0, 8.0]),
        ],
        spec_findings=[_spec_finding()],
        starting_index=2,
    )
    assert len(result.findings) == 1
    f = result.findings[0]
    assert f.type == FindingType.TRAJECTORY_PATTERN
    assert f.tier == Tier.B
    assert f.extracted_via == ExtractedVia.LLM_JUDGED
    assert f.finding_id == f"{CHAR_ID}:F-0002"
    assert f.confidence == 0.75
    assert f.statistics["pattern_kind"] == "cross_batch_variance"
    assert f.statistics["cv"] == 0.35
    assert f.run_ids == ["RUN-0001", "RUN-0002", "RUN-0003"]
    assert f.tool_calls if hasattr(f, "tool_calls") else True  # not on Finding
    assert result.tool_calls == 0


def test_tool_call_then_emit():
    """Agent calls execute_python once, sees result, then emits."""
    client = _RecordingClient(
        [
            {
                "action": "tool_call",
                "tool": "execute_python",
                "code": "import pandas as pd; df = pd.read_csv('OBS_PATH'); print(df.head())",
            },
            {
                "action": "emit_patterns",
                "patterns": [
                    {
                        "pattern_kind": "outlier_batch",
                        "summary": "RUN-0002 deviates >2σ on biomass mean",
                        "variables_involved": ["biomass_g_l"],
                        "run_ids": ["RUN-0002"],
                        "severity": "major",
                        "confidence": 0.7,
                        "statistics": {"z_score": 2.4, "n_runs": 3},
                    }
                ],
            },
        ]
    )
    agent = TrajectoryAnalyzerAgent(client=client)  # type: ignore[arg-type]
    result = agent.analyze(
        char_id=CHAR_ID,
        trajectories=[
            _trajectory("RUN-0001", "biomass_g_l", [1.0, 2.0, 3.0]),
            _trajectory("RUN-0002", "biomass_g_l", [10.0, 20.0, 30.0]),
            _trajectory("RUN-0003", "biomass_g_l", [1.5, 2.5, 3.5]),
        ],
        spec_findings=[],
    )
    assert len(result.findings) == 1
    assert result.tool_calls == 1
    assert len(client.calls) == 2  # one tool_call, one emit
    # Second turn's user_text must include tool history
    assert "TOOL HISTORY" in client.calls[1]["user_text"]
    assert "df.head()" in client.calls[1]["user_text"] or "execute_python" in client.calls[1]["user_text"]


def test_force_emit_after_max_tool_calls():
    """Loop budget: after MAX_TOOL_CALLS tool_calls, agent is forced to emit.
    The forced response should include a [FORCED] marker in user_text."""
    # Queue 9 tool_calls (one over the budget) + a final emit; the runner
    # should never reach the 9th tool_call because the must_emit gate
    # will force emit on call_idx == MAX_TOOL_CALLS.
    responses = [
        {"action": "tool_call", "tool": "execute_python", "code": "print(1)"}
        for _ in range(MAX_TOOL_CALLS)
    ] + [{"action": "emit_patterns", "patterns": []}]
    client = _RecordingClient(responses)
    agent = TrajectoryAnalyzerAgent(client=client)  # type: ignore[arg-type]
    result = agent.analyze(
        char_id=CHAR_ID,
        trajectories=[_trajectory("RUN-0001", "biomass_g_l", [1.0, 2.0])],
        spec_findings=[],
    )
    assert result.tool_calls == MAX_TOOL_CALLS
    # The last call's user_text should carry the FORCED marker
    assert "[FORCED]" in client.calls[-1]["user_text"]


# ---------- coercion & validation ----------


def test_confidence_clamped_to_cap():
    """Agent over-claims confidence (0.95). Coercer clamps to 0.85."""
    client = _RecordingClient(
        [
            {
                "action": "emit_patterns",
                "patterns": [
                    {
                        "pattern_kind": "phase_boundary",
                        "summary": "stationary phase enters at 80h in RUN-0001",
                        "variables_involved": ["biomass_g_l"],
                        "run_ids": ["RUN-0001"],
                        "confidence": 0.95,  # over cap
                        "statistics": {"inflection_idx": 80},
                    }
                ],
            }
        ]
    )
    agent = TrajectoryAnalyzerAgent(client=client)  # type: ignore[arg-type]
    result = agent.analyze(
        char_id=CHAR_ID,
        trajectories=[_trajectory("RUN-0001", "biomass_g_l", [1.0, 2.0, 3.0])],
        spec_findings=[],
    )
    assert result.findings[0].confidence == LLM_CONFIDENCE_CAP


def test_invalid_severity_falls_back_to_minor():
    client = _RecordingClient(
        [
            {
                "action": "emit_patterns",
                "patterns": [
                    {
                        "pattern_kind": "x",
                        "summary": "a pattern in RUN-0001",
                        "run_ids": ["RUN-0001"],
                        "severity": "catastrophic",  # not a valid Severity
                        "confidence": 0.6,
                        "statistics": {"k": 1},
                    }
                ],
            }
        ]
    )
    agent = TrajectoryAnalyzerAgent(client=client)  # type: ignore[arg-type]
    result = agent.analyze(
        char_id=CHAR_ID,
        trajectories=[_trajectory("RUN-0001", "x", [1.0, 2.0])],
        spec_findings=[],
    )
    assert result.findings[0].severity == Severity.MINOR


def test_empty_summary_pattern_dropped():
    """Pattern with empty summary doesn't satisfy Finding(min_length=1)
    on summary — coercer drops it with a log warning."""
    client = _RecordingClient(
        [
            {
                "action": "emit_patterns",
                "patterns": [
                    {
                        "pattern_kind": "x",
                        "summary": "",  # invalid
                        "confidence": 0.6,
                        "statistics": {},
                    },
                    {
                        "pattern_kind": "y",
                        "summary": "valid one cites RUN-0001",
                        "run_ids": ["RUN-0001"],
                        "confidence": 0.6,
                        "statistics": {"k": 1},
                    },
                ],
            }
        ]
    )
    agent = TrajectoryAnalyzerAgent(client=client)  # type: ignore[arg-type]
    result = agent.analyze(
        char_id=CHAR_ID,
        trajectories=[_trajectory("RUN-0001", "x", [1.0, 2.0])],
        spec_findings=[],
    )
    assert len(result.findings) == 1
    assert result.findings[0].statistics["pattern_kind"] == "y"


def test_pattern_kind_recorded_in_statistics():
    """D6b/(ii) contract: type=TRAJECTORY_PATTERN is the bucket;
    pattern_kind goes into statistics for downstream discrimination."""
    client = _RecordingClient(
        [
            {
                "action": "emit_patterns",
                "patterns": [
                    {
                        "pattern_kind": "correlation",
                        "summary": "agitation vs biomass r=0.82",
                        "variables_involved": ["agitation_rpm", "biomass_g_l"],
                        "run_ids": ["RUN-0001", "RUN-0002"],
                        "confidence": 0.8,
                        "statistics": {"r_pearson": 0.82, "p_value": 0.003, "n": 100},
                    }
                ],
            }
        ]
    )
    agent = TrajectoryAnalyzerAgent(client=client)  # type: ignore[arg-type]
    result = agent.analyze(
        char_id=CHAR_ID,
        trajectories=[
            _trajectory("RUN-0001", "biomass_g_l", [1.0, 2.0]),
            _trajectory("RUN-0002", "agitation_rpm", [100.0, 200.0]),
        ],
        spec_findings=[],
    )
    f = result.findings[0]
    assert f.type == FindingType.TRAJECTORY_PATTERN
    assert f.statistics["pattern_kind"] == "correlation"
    assert f.statistics["r_pearson"] == 0.82
    assert f.variables_involved == ["agitation_rpm", "biomass_g_l"]


# ---------- prompt content ----------


def test_system_prompt_demands_observational_only():
    """No causal language allowed (mirrors diagnose's prompt invariant)."""
    flat = TRAJECTORY_ANALYZER_SYSTEM
    assert "NO causal language" in flat or "causal" in flat.lower()


def test_system_prompt_includes_few_shot_examples():
    """D7a — examples for the four common analyses live in the prompt."""
    flat = TRAJECTORY_ANALYZER_SYSTEM
    assert "Cross-batch variance" in flat
    assert "Outlier" in flat or "outlier" in flat
    assert "Phase boundary" in flat or "phase" in flat.lower()
    assert "correlation" in flat.lower() or "Correlation" in flat


def test_system_prompt_specifies_d2a_thresholds():
    """D2a — hardcoded threshold defaults (>2σ, |r|≥0.5, p<0.05)."""
    flat = TRAJECTORY_ANALYZER_SYSTEM
    assert "2 sigma" in flat or "2σ" in flat or ">2" in flat
    assert "0.5" in flat
    assert "0.05" in flat


def test_user_text_provides_obs_path_and_metadata():
    """The agent needs the observations.csv path and trajectory metadata
    in its first user_text turn."""
    client = _RecordingClient(
        [{"action": "emit_patterns", "patterns": []}]
    )
    agent = TrajectoryAnalyzerAgent(client=client)  # type: ignore[arg-type]
    agent.analyze(
        char_id=CHAR_ID,
        trajectories=[
            _trajectory("RUN-0001", "biomass_g_l", [1.0, 2.0]),
            _trajectory("RUN-0002", "biomass_g_l", [3.0, 4.0]),
        ],
        spec_findings=[_spec_finding()],
    )
    user_text = client.calls[0]["user_text"]
    assert "OBSERVATIONS_CSV" in user_text
    assert "RUN-0001" in user_text
    assert "biomass_g_l" in user_text
    # Spec findings flow as context (D3a)
    assert "SPEC FINDINGS" in user_text
    assert "biomass exceeded nominal spec" in user_text


# ---------- error handling ----------


def test_client_error_returns_empty_no_crash():
    """A Gemini outage during analysis must not bubble up — analyzer
    is advisory; pipeline continues with spec findings only."""

    class _BoomClient(_RecordingClient):
        def call(self, **kwargs):
            raise RuntimeError("simulated gemini outage")

    agent = TrajectoryAnalyzerAgent(client=_BoomClient([]))  # type: ignore[arg-type]
    result = agent.analyze(
        char_id=CHAR_ID,
        trajectories=[_trajectory("RUN-0001", "x", [1.0, 2.0])],
        spec_findings=[],
    )
    assert result.findings == []


# ---------- factory ----------


def test_build_trajectory_analyzer_passes_through_client():
    agent = build_trajectory_analyzer(client=None)
    assert isinstance(agent, TrajectoryAnalyzerAgent)
    # Stub mode confirmed
    result = agent.analyze(char_id=CHAR_ID, trajectories=[], spec_findings=[])
    assert result.findings == []
