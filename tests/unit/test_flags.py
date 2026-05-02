"""Tests for ProcessFlag emission rules.

One test per flag plus one boundary check per numeric threshold. Flags
must be deterministic (same inputs -> same output) and order-stable.
"""

from __future__ import annotations

from typing import Any

import pytest

from fermdocs_characterize.flags import (
    LOW_QUALITY_THRESHOLD,
    SPARSE_COVERAGE_PCT_THRESHOLD,
    SPARSE_ROWS_THRESHOLD,
    ProcessFlag,
    compute_flags,
)
from fermdocs_characterize.schema import DataQuality, Trajectory
from fermdocs_characterize.views.summary import (
    DroppedObservation,
    Summary,
    SummaryRow,
)


def _summary(
    rows: list[SummaryRow] | None = None,
    run_ids: list[str] | None = None,
    variables: list[str] | None = None,
) -> Summary:
    rows = rows or []
    return Summary(
        rows=rows,
        dropped=[],
        run_ids=run_ids if run_ids is not None else sorted({r.run_id for r in rows}),
        variables=variables if variables is not None else sorted({r.variable for r in rows}),
    )


def _row(
    *,
    variable: str = "biomass_g_l",
    run_id: str = "RUN-1",
    expected: float | None = 0.5,
    expected_std_dev: float | None = 0.05,
    obs_id: str = "O-1",
    time: float = 0.0,
) -> SummaryRow:
    return SummaryRow(
        observation_id=obs_id,
        run_id=run_id,
        time=time,
        variable=variable,
        value=0.5,
        unit="g/L",
        expected=expected,
        expected_std_dev=expected_std_dev,
    )


def _dossier(
    *,
    organism: str | None = "Penicillium chrysogenum",
    process_id: str | None = "penicillin_indpensim",
    registered_provenance: str = "llm_whitelisted",
    coverage_pct: int = 80,
    stale_versions: list[str] | None = None,
) -> dict[str, Any]:
    return {
        "ingestion_summary": {
            "stale_schema_versions": stale_versions or [],
            "golden_coverage_percent": coverage_pct,
        },
        "experiment": {
            "process": {
                "observed": {"organism": organism},
                "registered": {
                    "process_id": process_id,
                    "provenance": registered_provenance,
                },
            }
        },
    }


def _trajectory(quality: float, run_id: str = "RUN-1", variable: str = "X") -> Trajectory:
    n = 10
    n_imputed = max(0, int((1 - quality) * n))
    n_real = n - n_imputed
    return Trajectory(
        trajectory_id="T-0001",
        run_id=run_id,
        variable=variable,
        time_grid=[float(i) for i in range(n)],
        values=[1.0] * n,
        imputation_flags=[i < n_imputed for i in range(n)],
        imputation_method="linear" if n_imputed else None,
        source_observation_ids=["O-1"],
        unit="g/L",
        quality=quality,
        data_quality=DataQuality(
            pct_missing=0.0,
            pct_imputed=n_imputed / n,
            pct_real=n_real / n,
        ),
    )


# ---------- Result is sorted, deterministic ----------


def test_compute_flags_returns_sorted_list():
    summary = _summary([_row() for _ in range(SPARSE_ROWS_THRESHOLD + 5)])
    dossier = _dossier(coverage_pct=90)
    flags = compute_flags(dossier, summary, [])
    assert flags == sorted(flags, key=lambda f: f.value)


def test_compute_flags_is_deterministic():
    summary = _summary([_row(obs_id=f"O-{i}") for i in range(5)])
    dossier = _dossier(coverage_pct=90)
    a = compute_flags(dossier, summary, [])
    b = compute_flags(dossier, summary, [])
    assert a == b


# ---------- STALE_SCHEMA_OBSERVATIONS ----------


def test_stale_schema_fires_when_versions_present():
    summary = _summary([_row()])
    dossier = _dossier(stale_versions=["1.0", "1.5"])
    assert ProcessFlag.STALE_SCHEMA_OBSERVATIONS in compute_flags(dossier, summary, [])


def test_stale_schema_silent_when_versions_empty():
    summary = _summary([_row()])
    dossier = _dossier(stale_versions=[])
    assert ProcessFlag.STALE_SCHEMA_OBSERVATIONS not in compute_flags(dossier, summary, [])


# ---------- LOW_QUALITY_TRAJECTORY ----------


def test_low_quality_trajectory_fires_below_threshold():
    summary = _summary([_row()])
    dossier = _dossier()
    traj = _trajectory(quality=LOW_QUALITY_THRESHOLD - 0.01)
    assert ProcessFlag.LOW_QUALITY_TRAJECTORY in compute_flags(dossier, summary, [traj])


def test_low_quality_trajectory_silent_at_threshold():
    """Boundary: quality == 0.8 does NOT fire (rule is strictly less than)."""
    summary = _summary([_row()])
    dossier = _dossier()
    traj = _trajectory(quality=LOW_QUALITY_THRESHOLD)
    assert ProcessFlag.LOW_QUALITY_TRAJECTORY not in compute_flags(dossier, summary, [traj])


def test_low_quality_fires_when_any_trajectory_below():
    summary = _summary([_row()])
    dossier = _dossier()
    good = _trajectory(quality=0.99)
    bad = _trajectory(quality=0.5, variable="other")
    assert ProcessFlag.LOW_QUALITY_TRAJECTORY in compute_flags(dossier, summary, [good, bad])


# ---------- SPARSE_DATA ----------


def test_sparse_data_fires_when_too_few_rows():
    summary = _summary([_row(obs_id=f"O-{i}") for i in range(SPARSE_ROWS_THRESHOLD - 1)])
    dossier = _dossier(coverage_pct=90)
    assert ProcessFlag.SPARSE_DATA in compute_flags(dossier, summary, [])


def test_sparse_data_silent_at_row_threshold():
    """Boundary: rows == 20 does NOT fire (rule is strictly less than)."""
    summary = _summary([_row(obs_id=f"O-{i}") for i in range(SPARSE_ROWS_THRESHOLD)])
    dossier = _dossier(coverage_pct=90)
    assert ProcessFlag.SPARSE_DATA not in compute_flags(dossier, summary, [])


def test_sparse_data_fires_on_low_coverage_even_with_many_rows():
    summary = _summary([_row(obs_id=f"O-{i}") for i in range(SPARSE_ROWS_THRESHOLD + 10)])
    dossier = _dossier(coverage_pct=SPARSE_COVERAGE_PCT_THRESHOLD - 1)
    assert ProcessFlag.SPARSE_DATA in compute_flags(dossier, summary, [])


def test_sparse_data_silent_at_coverage_threshold():
    """Boundary: coverage == 50 does NOT fire."""
    summary = _summary([_row(obs_id=f"O-{i}") for i in range(SPARSE_ROWS_THRESHOLD + 10)])
    dossier = _dossier(coverage_pct=SPARSE_COVERAGE_PCT_THRESHOLD)
    assert ProcessFlag.SPARSE_DATA not in compute_flags(dossier, summary, [])


# ---------- MIXED_RUNS ----------


def test_mixed_runs_fires_when_more_than_one_run():
    summary = _summary([_row(run_id="A"), _row(run_id="B", obs_id="O-2")])
    dossier = _dossier()
    assert ProcessFlag.MIXED_RUNS in compute_flags(dossier, summary, [])


def test_mixed_runs_silent_with_one_run():
    summary = _summary([_row(run_id="A"), _row(run_id="A", obs_id="O-2")])
    dossier = _dossier()
    assert ProcessFlag.MIXED_RUNS not in compute_flags(dossier, summary, [])


# ---------- UNKNOWN_PROCESS ----------


def test_unknown_process_fires_when_registered_provenance_unknown():
    summary = _summary([_row()])
    dossier = _dossier(registered_provenance="unknown")
    assert ProcessFlag.UNKNOWN_PROCESS in compute_flags(dossier, summary, [])


def test_unknown_process_silent_when_registered_llm_whitelisted():
    summary = _summary([_row()])
    dossier = _dossier(registered_provenance="llm_whitelisted")
    assert ProcessFlag.UNKNOWN_PROCESS not in compute_flags(dossier, summary, [])


def test_unknown_process_silent_when_registered_manifest():
    summary = _summary([_row()])
    dossier = _dossier(registered_provenance="manifest")
    assert ProcessFlag.UNKNOWN_PROCESS not in compute_flags(dossier, summary, [])


# ---------- UNKNOWN_ORGANISM ----------


def test_unknown_organism_fires_when_organism_null():
    summary = _summary([_row()])
    dossier = _dossier(organism=None)
    assert ProcessFlag.UNKNOWN_ORGANISM in compute_flags(dossier, summary, [])


def test_unknown_organism_silent_when_organism_present():
    summary = _summary([_row()])
    dossier = _dossier(organism="S. cerevisiae")
    assert ProcessFlag.UNKNOWN_ORGANISM not in compute_flags(dossier, summary, [])


def test_unknown_organism_independent_of_unknown_process():
    """Yeast case: organism present, process not in registry. UNKNOWN_PROCESS
    fires alone -- UNKNOWN_ORGANISM does not.
    """
    summary = _summary([_row(obs_id=f"O-{i}") for i in range(SPARSE_ROWS_THRESHOLD)])
    dossier = _dossier(
        organism="S. cerevisiae",
        process_id=None,
        registered_provenance="unknown",
        coverage_pct=80,
    )
    flags = compute_flags(dossier, summary, [])
    assert ProcessFlag.UNKNOWN_PROCESS in flags
    assert ProcessFlag.UNKNOWN_ORGANISM not in flags


# ---------- SPECS_MOSTLY_MISSING ----------


def test_specs_mostly_missing_fires_when_more_than_half_lack_specs():
    rows = [
        _row(variable="A", expected=None, expected_std_dev=None, obs_id="O-A"),
        _row(variable="B", expected=None, expected_std_dev=None, obs_id="O-B"),
        _row(variable="C", expected=0.5, expected_std_dev=0.05, obs_id="O-C"),
    ]
    summary = _summary(rows)
    dossier = _dossier()
    assert ProcessFlag.SPECS_MOSTLY_MISSING in compute_flags(dossier, summary, [])


def test_specs_mostly_missing_silent_at_half():
    """Boundary: exactly 50% missing does NOT fire (rule is strictly greater)."""
    rows = [
        _row(variable="A", expected=None, expected_std_dev=None, obs_id="O-A"),
        _row(variable="B", expected=0.5, expected_std_dev=0.05, obs_id="O-B"),
    ]
    summary = _summary(rows)
    dossier = _dossier()
    assert ProcessFlag.SPECS_MOSTLY_MISSING not in compute_flags(dossier, summary, [])


def test_specs_mostly_missing_silent_when_no_variables():
    """Empty variables list never fires this flag."""
    summary = _summary(rows=[], run_ids=[], variables=[])
    dossier = _dossier()
    assert ProcessFlag.SPECS_MOSTLY_MISSING not in compute_flags(dossier, summary, [])


# ---------- Composite ----------


def test_multiple_flags_fire_together():
    """Empty dossier hits sparse_data + unknown_organism + unknown_process."""
    summary = _summary(rows=[], run_ids=[], variables=[])
    dossier = _dossier(organism=None, registered_provenance="unknown", coverage_pct=0)
    flags = compute_flags(dossier, summary, [])
    assert ProcessFlag.SPARSE_DATA in flags
    assert ProcessFlag.UNKNOWN_ORGANISM in flags
    assert ProcessFlag.UNKNOWN_PROCESS in flags
