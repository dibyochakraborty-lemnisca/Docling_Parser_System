"""Tests for catalog-grounded pattern coercion.

When the trajectory analyzer emits a pattern with a metric_id matching
a ready catalog entry, the coercer should:
  - record metric_id + tier in statistics
  - flip extracted_via to STATISTICAL
  - allow confidence up to STATISTICAL_CONFIDENCE_CAP (0.95) instead of 0.85
  - inherit tier from the catalog entry

When metric_id is absent, unknown, or pending, behavior matches the
existing LLM_JUDGED path (cap 0.85, default Tier.B).
"""

from __future__ import annotations

from uuid import UUID

from fermdocs_characterize.agents.trajectory_analyzer import (
    LLM_CONFIDENCE_CAP,
    STATISTICAL_CONFIDENCE_CAP,
    TrajectoryAnalyzerAgent,
)
from fermdocs_characterize.schema import DataQuality, ExtractedVia, Tier, Trajectory

CHAR_ID = UUID("22222222-2222-2222-2222-222222222222")


def _traj(run_id: str, variable: str) -> Trajectory:
    return Trajectory(
        trajectory_id="T-0001",
        run_id=run_id,
        variable=variable,
        time_grid=[0.0, 1.0, 2.0],
        values=[0.1, 0.2, 0.3],
        imputation_flags=[False, False, False],
        unit="g/L",
        quality=1.0,
        data_quality=DataQuality(pct_missing=0.0, pct_imputed=0.0, pct_real=1.0),
        source_observation_ids=["OBS-0001", "OBS-0002", "OBS-0003"],
    )


def _coerce(pattern: dict, traj: Trajectory):
    """Helper: run _build_findings end-to-end with a one-trajectory bundle."""
    agent = TrajectoryAnalyzerAgent(client=None)
    findings = agent._build_findings(
        [pattern], char_id=CHAR_ID, starting_index=1, trajectories=[traj]
    )
    return findings


def test_metric_id_a8_is_statistical_with_tier_a() -> None:
    traj = _traj("RUN-0001", "biomass_g_l")
    pattern = {
        "pattern_kind": "specific_growth_rate",
        "metric_id": "A8",
        "summary": "RUN-0001 mu_max=0.42 1/h at t=12h",
        "run_ids": ["RUN-0001"],
        "variables_involved": ["biomass_g_l"],
        "confidence": 0.92,  # above LLM cap; should survive at STATISTICAL cap
        "statistics": {"mu_max": 0.42, "t_mu_max_h": 12.0},
    }
    [finding] = _coerce(pattern, traj)
    assert finding.extracted_via == ExtractedVia.STATISTICAL
    assert finding.tier == Tier.A
    assert finding.confidence == 0.92
    assert finding.statistics["metric_id"] == "A8"
    assert finding.statistics["tier"] == "A"


def test_no_metric_id_stays_llm_judged() -> None:
    traj = _traj("RUN-0001", "biomass_g_l")
    pattern = {
        "pattern_kind": "ad_hoc_observation",
        "summary": "RUN-0001 plateaus around t=15h",
        "run_ids": ["RUN-0001"],
        "variables_involved": ["biomass_g_l"],
        "confidence": 0.92,  # above LLM cap; gets clamped
        "statistics": {"n_runs": 1},
    }
    [finding] = _coerce(pattern, traj)
    assert finding.extracted_via == ExtractedVia.LLM_JUDGED
    assert finding.confidence == LLM_CONFIDENCE_CAP
    assert "metric_id" not in finding.statistics


def test_unknown_metric_id_falls_back_to_llm_judged() -> None:
    traj = _traj("RUN-0001", "biomass_g_l")
    pattern = {
        "pattern_kind": "made_up",
        "metric_id": "Z99",  # not in catalog
        "summary": "RUN-0001 something",
        "run_ids": ["RUN-0001"],
        "variables_involved": ["biomass_g_l"],
        "confidence": 0.9,
        "statistics": {},
    }
    [finding] = _coerce(pattern, traj)
    assert finding.extracted_via == ExtractedVia.LLM_JUDGED
    assert finding.confidence == LLM_CONFIDENCE_CAP
    # metric_id is still recorded for audit even when unknown,
    # so we can grep for "agent claimed metric_id X but it's not real"
    assert finding.statistics["metric_id"] == "Z99"


def test_pending_metric_id_falls_back_to_llm_judged() -> None:
    traj = _traj("RUN-0001", "biomass_g_l")
    pattern = {
        "pattern_kind": "kla_pending",
        "metric_id": "B11",  # in catalog, status="pending" until PR 3
        "summary": "RUN-0001 kLa estimate pending",
        "run_ids": ["RUN-0001"],
        "variables_involved": ["biomass_g_l"],
        "confidence": 0.9,
        "statistics": {},
    }
    [finding] = _coerce(pattern, traj)
    assert finding.extracted_via == ExtractedVia.LLM_JUDGED
    assert finding.confidence == LLM_CONFIDENCE_CAP
    assert finding.statistics["metric_id"] == "B11"
    # tier still picked up from catalog even for pending entries — it's
    # an audit-friendly default.
    assert finding.statistics["tier"] == "B"


def test_statistical_cap_clamps_when_overshoot() -> None:
    traj = _traj("RUN-0001", "biomass_g_l")
    pattern = {
        "pattern_kind": "specific_growth_rate",
        "metric_id": "A8",
        "summary": "RUN-0001 mu_max=0.42",
        "run_ids": ["RUN-0001"],
        "variables_involved": ["biomass_g_l"],
        "confidence": 0.999,  # above STATISTICAL cap
        "statistics": {"mu_max": 0.42},
    }
    [finding] = _coerce(pattern, traj)
    assert finding.confidence == STATISTICAL_CONFIDENCE_CAP


def test_catalog_tier_a8_maps_to_tier_a_a10_to_tier_a() -> None:
    traj = _traj("RUN-0001", "biomass_g_l")
    for metric_id in ("A8", "A10", "A11"):
        pattern = {
            "pattern_kind": "test",
            "metric_id": metric_id,
            "summary": f"RUN-0001 via {metric_id}",
            "run_ids": ["RUN-0001"],
            "variables_involved": ["biomass_g_l"],
            "confidence": 0.7,
            "statistics": {},
        }
        [finding] = _coerce(pattern, traj)
        assert finding.tier == Tier.A, f"expected Tier.A for {metric_id}"
