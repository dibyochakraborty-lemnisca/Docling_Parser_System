"""Synthetic-data tests for the Tier A cross_run toolkit."""

from __future__ import annotations

import math

import pandas as pd

from fermdocs_characterize.toolkit.cross_run import (
    cross_run_kpi_table,
    pairwise_deviation,
    variance_decomposition,
)


# ---------- cross_run_kpi_table ----------


def test_kpi_table_indexed_by_run_id() -> None:
    df = cross_run_kpi_table([
        {"run_id": "RUN-0001", "mu_max": 0.42},
        {"run_id": "RUN-0002", "mu_max": 0.38},
    ])
    assert df.index.name == "run_id"
    assert list(df.index) == ["RUN-0001", "RUN-0002"]
    assert df.loc["RUN-0001", "mu_max"] == 0.42


def test_kpi_table_empty_when_no_records() -> None:
    df = cross_run_kpi_table([])
    assert df.empty


def test_kpi_table_preserves_nan_when_kpi_missing() -> None:
    df = cross_run_kpi_table([
        {"run_id": "A", "mu_max": 0.4, "yield": 0.5},
        {"run_id": "B", "mu_max": 0.3},  # no yield
    ])
    assert pd.isna(df.loc["B", "yield"])


# ---------- pairwise_deviation ----------


def test_pairwise_deviation_finds_largest_gap() -> None:
    df = cross_run_kpi_table([
        {"run_id": "A", "mu_max": 0.40},
        {"run_id": "B", "mu_max": 0.42},
        {"run_id": "C", "mu_max": 0.10},  # outlier
    ])
    pairs = pairwise_deviation(df, top_k=1)
    assert len(pairs) == 1
    # The biggest gap should involve C.
    assert "C" in (pairs[0].run_a, pairs[0].run_b)


def test_pairwise_deviation_returns_empty_for_single_run() -> None:
    df = cross_run_kpi_table([{"run_id": "A", "mu_max": 0.4}])
    assert pairwise_deviation(df) == []


def test_pairwise_deviation_skips_zero_joint_mean() -> None:
    df = cross_run_kpi_table([
        {"run_id": "A", "mu_max": 0.0},
        {"run_id": "B", "mu_max": 0.0},
    ])
    # Joint mean is 0 → pair excluded.
    assert pairwise_deviation(df) == []


def test_pairwise_deviation_top_k_caps_output() -> None:
    df = cross_run_kpi_table([
        {"run_id": chr(65 + i), "mu_max": 0.1 * (i + 1)} for i in range(5)
    ])
    pairs = pairwise_deviation(df, top_k=3)
    assert len(pairs) == 3
    # Sorted descending by relative_gap
    gaps = [p.relative_gap for p in pairs]
    assert gaps == sorted(gaps, reverse=True)


# ---------- variance_decomposition ----------


def test_variance_decomp_no_grouping_returns_total() -> None:
    df = cross_run_kpi_table([
        {"run_id": "A", "mu_max": 0.4},
        {"run_id": "B", "mu_max": 0.3},
        {"run_id": "C", "mu_max": 0.5},
    ])
    out = variance_decomposition(df)
    assert len(out) == 1
    row = out.iloc[0]
    assert row["kpi"] == "mu_max"
    assert row["within_var"] == 0.0
    assert math.isclose(row["between_frac"], 1.0)


def test_variance_decomp_with_grouping_splits_components() -> None:
    df = cross_run_kpi_table([
        {"run_id": "A", "mu_max": 0.4, "group": "X"},
        {"run_id": "B", "mu_max": 0.42, "group": "X"},
        {"run_id": "C", "mu_max": 0.1, "group": "Y"},
        {"run_id": "D", "mu_max": 0.12, "group": "Y"},
    ])
    out = variance_decomposition(df, grouping_column="group")
    row = out[out["kpi"] == "mu_max"].iloc[0]
    # Most variance should be between-group (X is high, Y is low).
    assert row["between_var"] > row["within_var"]
    assert row["between_frac"] > 0.9


def test_variance_decomp_skips_grouping_column() -> None:
    df = cross_run_kpi_table([
        {"run_id": "A", "mu_max": 0.4, "group": "X"},
        {"run_id": "B", "mu_max": 0.5, "group": "X"},
    ])
    out = variance_decomposition(df, grouping_column="group")
    assert "group" not in set(out["kpi"])


def test_variance_decomp_empty_input() -> None:
    out = variance_decomposition(pd.DataFrame())
    assert out.empty
    assert list(out.columns) == [
        "kpi", "total_var", "between_var", "within_var", "between_frac"
    ]
