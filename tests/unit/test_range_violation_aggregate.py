"""Tests for the range_violation aggregation circuit breaker.

When a single (variable, run_id) pair produces more than AGGREGATE_THRESHOLD
per-row range_violations, the candidates collapse into one rollup finding.
This protects the diagnosis agent's prompt budget from runaway findings when
schema specs have setpoint semantics but the variable is a trajectory.
"""

from __future__ import annotations

from fermdocs_characterize.candidates.range_violation import (
    AGGREGATE_THRESHOLD,
    find_range_violations,
)
from fermdocs_characterize.schema import Severity
from fermdocs_characterize.views.summary import (
    Summary,
    SummaryRow,
)


def _row(i: int, value: float, *, run_id: str = "RUN-1", variable: str = "biomass_g_l") -> SummaryRow:
    return SummaryRow(
        observation_id=f"O-{i:04d}",
        run_id=run_id,
        time=float(i),
        variable=variable,
        value=value,
        unit="g/L",
        expected=0.5,
        expected_std_dev=0.05,
    )


def _summary(rows: list[SummaryRow]) -> Summary:
    return Summary(
        rows=rows,
        dropped=[],
        run_ids=sorted({r.run_id for r in rows}),
        variables=sorted({r.variable for r in rows}),
    )


# ---------- Below threshold: no aggregation ----------


def test_under_threshold_keeps_per_row_findings():
    """A small group keeps individual findings — agents need detail when there
    are only a handful of violations.
    """
    rows = [_row(i, value=10.0) for i in range(5)]  # 5 violations, all critical
    candidates = find_range_violations(_summary(rows), trajectories=[])
    assert len(candidates) == 5
    assert all(not c.statistics.get("aggregated") for c in candidates)


def test_at_threshold_no_aggregation():
    """Exactly AGGREGATE_THRESHOLD findings still pass through unaggregated."""
    rows = [_row(i, value=10.0) for i in range(AGGREGATE_THRESHOLD)]
    candidates = find_range_violations(_summary(rows), trajectories=[])
    assert len(candidates) == AGGREGATE_THRESHOLD
    assert all(not c.statistics.get("aggregated") for c in candidates)


# ---------- Above threshold: aggregation kicks in ----------


def test_runaway_collapses_to_single_rollup():
    """101 findings on the same (variable, run) collapse to 1."""
    rows = [_row(i, value=10.0 + i * 0.1) for i in range(AGGREGATE_THRESHOLD + 1)]
    candidates = find_range_violations(_summary(rows), trajectories=[])
    assert len(candidates) == 1
    rollup = candidates[0]
    assert rollup.statistics["aggregated"] is True
    assert rollup.statistics["n_violations"] == AGGREGATE_THRESHOLD + 1
    assert rollup.severity == Severity.CRITICAL


def test_rollup_carries_max_abs_sigmas():
    """Largest deviation surfaces in the rollup, not an averaged value."""
    rows = [_row(i, value=10.0) for i in range(50)]  # ~190σ
    rows += [_row(i + 50, value=20.0) for i in range(60)]  # ~390σ — larger
    candidates = find_range_violations(_summary(rows), trajectories=[])
    assert len(candidates) == 1
    sig = candidates[0].statistics["max_abs_sigmas"]
    assert sig > 380  # the 20.0 group dominates


def test_rollup_carries_observed_range():
    """Min and max observed values surface so the agent sees the spread."""
    rows = [_row(i, value=2.0) for i in range(60)]
    rows += [_row(i + 60, value=10.0) for i in range(60)]
    candidates = find_range_violations(_summary(rows), trajectories=[])
    assert len(candidates) == 1
    s = candidates[0].statistics
    assert s["observed_min"] == 2.0
    assert s["observed_max"] == 10.0


def test_rollup_carries_all_observation_ids():
    """The agent must be able to drill down via get_finding to any contributing
    observation. We bound by group size, not arbitrary truncation.
    """
    n = AGGREGATE_THRESHOLD + 50
    rows = [_row(i, value=10.0) for i in range(n)]
    candidates = find_range_violations(_summary(rows), trajectories=[])
    assert len(candidates) == 1
    assert len(candidates[0].evidence_observation_ids) == n


def test_rollup_caveat_explains_aggregation():
    """The diagnosis agent reads caveats; this one signals the aggregation
    happened so the agent doesn't double-count.
    """
    rows = [_row(i, value=10.0) for i in range(AGGREGATE_THRESHOLD + 1)]
    candidates = find_range_violations(_summary(rows), trajectories=[])
    assert any("aggregated" in c for c in candidates[0].caveats)
    assert any("setpoint" in c for c in candidates[0].caveats)


def test_rollup_summary_mentions_count_and_sigma():
    """Human-readable summary should name the violation count and max sigma."""
    rows = [_row(i, value=10.0) for i in range(AGGREGATE_THRESHOLD + 1)]
    candidates = find_range_violations(_summary(rows), trajectories=[])
    summary = candidates[0].summary
    assert str(AGGREGATE_THRESHOLD + 1) in summary
    assert "σ" in summary
    assert "biomass_g_l" in summary
    assert "RUN-1" in summary


# ---------- Mixed groups: only the runaway aggregates ----------


def test_mixed_runs_aggregate_independently():
    """Two runs each above threshold collapse to one rollup each, not one
    combined.
    """
    rows = [_row(i, value=10.0, run_id="RUN-A") for i in range(150)]
    rows += [_row(i + 200, value=10.0, run_id="RUN-B") for i in range(150)]
    candidates = find_range_violations(_summary(rows), trajectories=[])
    assert len(candidates) == 2
    runs = {c.run_ids[0] for c in candidates}
    assert runs == {"RUN-A", "RUN-B"}


def test_mixed_variables_aggregate_independently():
    rows = [_row(i, value=10.0, variable="biomass_g_l") for i in range(150)]
    rows += [_row(i + 200, value=10.0, variable="substrate_g_l") for i in range(150)]
    candidates = find_range_violations(_summary(rows), trajectories=[])
    assert len(candidates) == 2
    vars_ = {c.variables_involved[0] for c in candidates}
    assert vars_ == {"biomass_g_l", "substrate_g_l"}


def test_runaway_variable_aggregates_normal_passes_through():
    """One variable produces 200 violations (rolls up), another produces 5
    (stays per-row). Result: 1 rollup + 5 per-row = 6 candidates.
    """
    rows = [_row(i, value=10.0, variable="biomass_g_l") for i in range(200)]
    rows += [
        _row(i + 300, value=10.0, variable="substrate_g_l") for i in range(5)
    ]
    candidates = find_range_violations(_summary(rows), trajectories=[])
    assert len(candidates) == 6
    rollups = [c for c in candidates if c.statistics.get("aggregated")]
    per_row = [c for c in candidates if not c.statistics.get("aggregated")]
    assert len(rollups) == 1
    assert rollups[0].variables_involved == ["biomass_g_l"]
    assert len(per_row) == 5
    assert all(c.variables_involved == ["substrate_g_l"] for c in per_row)
