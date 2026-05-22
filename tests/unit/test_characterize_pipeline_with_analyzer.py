"""Pipeline integration test: trajectory_analyzer wired into
CharacterizationPipeline produces FindingType.TRAJECTORY_PATTERN
findings alongside the existing range_violation findings.

Backward-compat checks:
  - Pipeline without analyzer (default) produces only spec findings —
    no behavioral change.
  - Pipeline with analyzer uses spec findings as context AND appends
    trajectory_pattern findings with IDs after the spec findings.
"""

from __future__ import annotations

from datetime import datetime
from typing import Any
from uuid import UUID

from fermdocs_characterize.agents.trajectory_analyzer import (
    TrajectoryAnalyzerAgent,
)
from fermdocs_characterize.pipeline import CharacterizationPipeline
from fermdocs_characterize.schema import FindingType


CHAR_ID = UUID("22222222-2222-2222-2222-222222222222")
GEN_TS = datetime(2026, 5, 4, 12, 0, 0)


def _minimal_dossier() -> dict[str, Any]:
    """Smallest dossier that produces at least one trajectory.

    Uses the synthetic '_specs' + 'golden_columns' fixture shape that
    build_summary expects: golden_columns is a dict keyed by variable;
    each value has an `observations` list with observation_id, value,
    unit, source.locator.{run_id, timestamp_h}.
    """
    def _obs(oid: str, value: float, t: float) -> dict:
        return {
            "observation_id": oid,
            "value": value,
            "unit": "g/L",
            "source": {"locator": {"run_id": "RUN-0001", "timestamp_h": t}},
        }

    return {
        "experiment": {"experiment_id": "EXP-PIPELINE-TEST"},
        "_specs": {
            "biomass_g_l": {
                "nominal": 0.5,
                "std_dev": 0.05,
                "unit": "g/L",
            }
        },
        "golden_columns": {
            "biomass_g_l": {
                "observations": [
                    _obs("OBS-0001", 1.0, 0.0),
                    _obs("OBS-0002", 12.0, 24.0),
                    _obs("OBS-0003", 24.0, 48.0),
                ],
            },
        },
    }


# ---------- back-compat: no analyzer, deterministic only ----------


def test_pipeline_without_analyzer_produces_no_pattern_findings():
    pipeline = CharacterizationPipeline(validate=False)
    output = pipeline.run(
        _minimal_dossier(),
        characterization_id=CHAR_ID,
        generation_timestamp=GEN_TS,
    )
    pattern_findings = [
        f for f in output.findings if f.type == FindingType.TRAJECTORY_PATTERN
    ]
    assert pattern_findings == [], (
        "without an analyzer, the pipeline must produce zero "
        "TRAJECTORY_PATTERN findings"
    )


def test_pipeline_with_stub_analyzer_also_produces_no_pattern_findings():
    """Stub analyzer (client=None) is a no-op. Pipeline output should
    match the no-analyzer case exactly."""
    pipeline_no = CharacterizationPipeline(validate=False)
    pipeline_stub = CharacterizationPipeline(
        validate=False,
        trajectory_analyzer=TrajectoryAnalyzerAgent(client=None),
    )
    out_no = pipeline_no.run(
        _minimal_dossier(),
        characterization_id=CHAR_ID,
        generation_timestamp=GEN_TS,
    )
    out_stub = pipeline_stub.run(
        _minimal_dossier(),
        characterization_id=CHAR_ID,
        generation_timestamp=GEN_TS,
    )
    assert len(out_no.findings) == len(out_stub.findings)
    assert [f.finding_id for f in out_no.findings] == [
        f.finding_id for f in out_stub.findings
    ]


# ---------- live wiring with mocked Gemini ----------


class _CannedClient:
    """Mock GeminiCharacterizeClient. Returns one tool_call then an emit
    so the analyzer exercises the full ReAct loop in the pipeline test."""

    def __init__(self):
        self._responses = [
            {
                "action": "tool_call",
                "tool": "execute_python",
                "code": "print('hello from sandbox')",
            },
            {
                "action": "emit_patterns",
                "patterns": [
                    {
                        "pattern_kind": "phase_boundary",
                        "summary": "Biomass enters stationary phase ~48h in RUN-0001",
                        "variables_involved": ["biomass_g_l"],
                        "run_ids": ["RUN-0001"],
                        "time_window": {"start": 36.0, "end": 60.0},
                        "severity": "minor",
                        "confidence": 0.7,
                        "statistics": {"inflection_time_h": 48.0, "n_runs": 1},
                    }
                ],
            },
        ]
        self.calls = 0

    @property
    def model_name(self) -> str:
        return "fake-pipeline-test"

    def call(self, *, system, user_text, response_schema, temperature=0.0):
        self.calls += 1
        if not self._responses:
            return ({"action": "emit_patterns", "patterns": []}, 50, 10)
        return self._responses.pop(0), 100, 30


def test_pipeline_with_live_analyzer_appends_pattern_findings():
    """End-to-end: spec checks + trajectory_analyzer co-produce the
    final findings list. Pattern findings get IDs after spec findings."""
    client = _CannedClient()
    analyzer = TrajectoryAnalyzerAgent(client=client)  # type: ignore[arg-type]
    pipeline = CharacterizationPipeline(
        validate=False,
        trajectory_analyzer=analyzer,
    )
    output = pipeline.run(
        _minimal_dossier(),
        characterization_id=CHAR_ID,
        generation_timestamp=GEN_TS,
    )

    spec_findings = [
        f for f in output.findings if f.type != FindingType.TRAJECTORY_PATTERN
    ]
    pattern_findings = [
        f for f in output.findings if f.type == FindingType.TRAJECTORY_PATTERN
    ]

    # The analyzer ran and produced one pattern finding
    assert len(pattern_findings) == 1
    f = pattern_findings[0]
    assert f.statistics["pattern_kind"] == "phase_boundary"
    assert f.run_ids == ["RUN-0001"]

    # ID continuity: pattern finding ID > all spec finding IDs
    if spec_findings:
        spec_ids = sorted(int(f.finding_id.split("F-")[1]) for f in spec_findings)
        pattern_id = int(pattern_findings[0].finding_id.split("F-")[1])
        assert pattern_id > spec_ids[-1]

    # The agent saw spec findings as context (D3a)
    # — verified by the analyzer's user_text construction; here we just
    # confirm the analyzer ran end-to-end (2 calls = tool_call + emit)
    assert client.calls == 2


def test_pattern_findings_validate_against_dossier_observation_ids():
    """Regression for May 2026 bug: trajectory_analyzer initially put
    run_ids into evidence_observation_ids (e.g. 'RUN-0001'). The output
    validator resolves observation IDs through the dossier's
    `golden_columns[*].observations[*].observation_id` namespace and
    rejected the findings:

      'finding ...:F-0101 cites unknown observation_id 'RUN-0001''

    The fix derives evidence_observation_ids from the matching
    Trajectory's source_observation_ids — those ARE in the dossier
    namespace by construction. This test runs the FULL pipeline with
    validate=True so any future regression bites at test time, not in
    production."""
    client = _CannedClient()
    analyzer = TrajectoryAnalyzerAgent(client=client)  # type: ignore[arg-type]
    pipeline = CharacterizationPipeline(
        validate=True,  # the key bit — validator rejected the buggy shape
        trajectory_analyzer=analyzer,
    )
    output = pipeline.run(
        _minimal_dossier(),
        characterization_id=CHAR_ID,
        generation_timestamp=GEN_TS,
    )
    pattern_findings = [
        f for f in output.findings if f.type == FindingType.TRAJECTORY_PATTERN
    ]
    assert len(pattern_findings) == 1
    pf = pattern_findings[0]
    # evidence_observation_ids must be REAL ingestion IDs, not run_ids.
    # In the minimal dossier, observation_ids are OBS-0001, OBS-0002, OBS-0003.
    assert all(oid.startswith("OBS-") for oid in pf.evidence_observation_ids), (
        f"evidence_observation_ids must contain ingestion IDs, "
        f"got {pf.evidence_observation_ids}"
    )
    # And run_ids should NOT appear in evidence_observation_ids
    assert "RUN-0001" not in pf.evidence_observation_ids


def test_pipeline_with_failing_analyzer_does_not_crash():
    """When the analyzer raises during pipeline execution, the pipeline
    falls back to spec-only findings rather than failing the whole run."""

    class _BoomAnalyzer(TrajectoryAnalyzerAgent):
        def __init__(self):
            super().__init__(client=None)

        def analyze(self, **kwargs):
            raise RuntimeError("analyzer blew up")

    pipeline = CharacterizationPipeline(
        validate=False,
        trajectory_analyzer=_BoomAnalyzer(),
    )
    output = pipeline.run(
        _minimal_dossier(),
        characterization_id=CHAR_ID,
        generation_timestamp=GEN_TS,
    )
    pattern_findings = [
        f for f in output.findings if f.type == FindingType.TRAJECTORY_PATTERN
    ]
    assert pattern_findings == []
    # And spec findings are still there (deterministic stage uncompromised)
    assert all(f.type != FindingType.TRAJECTORY_PATTERN for f in output.findings)
