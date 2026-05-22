"""Regression tests for the unanchored-pattern crash on PDF-only bundles.

Production bug: the carotenoid PDF run hit
  Validation failed: finding ...:F-0011 cites unknown observation_id
  'trajectory_pattern_unanchored'

Root cause: when an LLM-emitted pattern cited (run_ids, variables) that
didn't match any Trajectory in the bundle, the coercer fell back to a
sentinel string 'trajectory_pattern_unanchored' as the
evidence_observation_id. The diagnose validator rejected the unknown
ID and crashed the run.

Fix: degrade gracefully — try anchoring against the cited runs only
(drop variable filter), then any trajectory at all, then drop the
finding via ValueError (which _build_findings already handles).
"""

from __future__ import annotations

from uuid import UUID

import pytest

from fermdocs_characterize.agents.trajectory_analyzer import TrajectoryAnalyzerAgent
from fermdocs_characterize.schema import DataQuality, Trajectory

CHAR_ID = UUID("55555555-5555-5555-5555-555555555555")


def _traj(run_id: str, variable: str, obs_ids: list[str]) -> Trajectory:
    n = len(obs_ids)
    return Trajectory(
        trajectory_id="T-0001",
        run_id=run_id,
        variable=variable,
        time_grid=[float(i) for i in range(n)],
        values=[1.0] * n,
        imputation_flags=[False] * n,
        unit="g/L",
        quality=1.0,
        data_quality=DataQuality(pct_missing=0.0, pct_imputed=0.0, pct_real=1.0),
        source_observation_ids=obs_ids,
    )


def _coerce(pattern: dict, trajectories: list[Trajectory]):
    agent = TrajectoryAnalyzerAgent(client=None)
    return agent._build_findings(
        [pattern], char_id=CHAR_ID, starting_index=1, trajectories=trajectories
    )


def test_unanchored_pattern_dropped_silently_no_sentinel() -> None:
    """Pattern cites a run we don't have AND no trajectories exist at all
    in the bundle. Should be dropped, not emit the legacy
    'trajectory_pattern_unanchored' sentinel."""
    pattern = {
        "pattern_kind": "phantom",
        "summary": "RUN-9999 reports something we cannot anchor",
        "run_ids": ["RUN-9999"],
        "variables_involved": ["nonexistent_var"],
        "confidence": 0.7,
        "statistics": {},
    }
    findings = _coerce(pattern, trajectories=[])
    assert findings == []


def test_unanchored_pattern_anchors_to_cited_run_when_variable_missing() -> None:
    """Pattern cites a run we DO have but a variable we don't — should
    fall back to anchoring against any observation of that run."""
    traj = _traj("RUN-0001", "biomass_g_l", ["OBS-0001", "OBS-0002"])
    pattern = {
        "pattern_kind": "var_mismatch",
        "summary": "RUN-0001 metabolic_x event",
        "run_ids": ["RUN-0001"],
        "variables_involved": ["metabolic_x_we_dont_have"],
        "confidence": 0.7,
        "statistics": {},
    }
    [finding] = _coerce(pattern, [traj])
    assert "trajectory_pattern_unanchored" not in finding.evidence_observation_ids
    assert "OBS-0001" in finding.evidence_observation_ids


def test_unanchored_pattern_falls_back_to_any_trajectory() -> None:
    """Pattern cites a run we don't have, but trajectories exist. Should
    anchor against any available trajectory rather than emitting the
    sentinel."""
    traj = _traj("RUN-0001", "biomass_g_l", ["OBS-0001"])
    pattern = {
        "pattern_kind": "phantom_run",
        "summary": "RUN-9999 something",
        "run_ids": ["RUN-9999"],
        "variables_involved": ["biomass_g_l"],
        "confidence": 0.7,
        "statistics": {},
    }
    [finding] = _coerce(pattern, [traj])
    assert "trajectory_pattern_unanchored" not in finding.evidence_observation_ids
    assert finding.evidence_observation_ids == ["OBS-0001"]


def test_anchored_pattern_uses_exact_match_first() -> None:
    """When (run_id, variable) match exactly, only those obs_ids are used
    — fallback paths must not pollute the citation."""
    traj_a = _traj("RUN-0001", "biomass_g_l", ["OBS-0001", "OBS-0002"])
    traj_b = _traj("RUN-0002", "ethanol_g_l", ["OBS-0099"])
    pattern = {
        "pattern_kind": "anchored",
        "summary": "RUN-0001 biomass observation",
        "run_ids": ["RUN-0001"],
        "variables_involved": ["biomass_g_l"],
        "confidence": 0.7,
        "statistics": {},
    }
    [finding] = _coerce(pattern, [traj_a, traj_b])
    assert set(finding.evidence_observation_ids) == {"OBS-0001", "OBS-0002"}
    assert "OBS-0099" not in finding.evidence_observation_ids


def test_no_trajectories_at_all_drops_pattern() -> None:
    """When the bundle has zero trajectories (PDF-only edge case), patterns
    can't be anchored to anything and must be dropped, not crash."""
    pattern = {
        "pattern_kind": "data_gap",
        "metric_id": "B10",
        "summary": "RQ unavailable: bundle has no OUR/CER trajectories",
        "run_ids": [],
        "variables_involved": [],
        "confidence": 0.5,
        "statistics": {},
    }
    findings = _coerce(pattern, trajectories=[])
    # Drop is the contract: dropping is preferable to a sentinel that
    # crashes the diagnose validator.
    assert findings == []
