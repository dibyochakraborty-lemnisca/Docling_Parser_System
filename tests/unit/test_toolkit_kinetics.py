"""Synthetic-data tests for the kinetics toolkit.

Each function gets a known-shape input and we assert the computed
output matches analytic expectations within reasonable tolerance. The
goal is twofold: (1) catch regressions if the smoothing strategy
changes, (2) document the expected behavior of each function for
future readers.
"""

from __future__ import annotations

import math

import numpy as np
import pandas as pd
import pytest

from fermdocs_characterize.toolkit.kinetics import (
    compute_mu,
    doubling_time,
    phasewise_mu,
    segment_growth_phases,
)


def _exponential(time_h: np.ndarray, mu_true: float, x0: float = 0.1) -> np.ndarray:
    return x0 * np.exp(mu_true * time_h)


def _logistic(time_h: np.ndarray, mu_true: float, k: float = 30.0) -> np.ndarray:
    # Logistic: lag-ish at start, exp middle, plateau end.
    return k / (1 + (k / 0.1 - 1) * np.exp(-mu_true * time_h))


# ---------- compute_mu ----------


def test_compute_mu_recovers_true_rate_on_clean_exponential() -> None:
    t = np.linspace(0, 20, 41)  # 0.5h spacing
    x = _exponential(t, mu_true=0.3)
    res = compute_mu(t, x, window=5)
    assert math.isclose(res.mu_max, 0.3, rel_tol=0.05)
    assert res.n_points == 41


def test_compute_mu_handles_pandas_series() -> None:
    t = pd.Series(np.linspace(0, 20, 21))
    x = pd.Series(_exponential(t.to_numpy(), mu_true=0.4))
    res = compute_mu(t, x)
    assert res.mu_max > 0.3


def test_compute_mu_drops_nonpositive_and_nan() -> None:
    t = [0, 1, 2, 3, 4, 5, 6, 7]
    x = [0.1, 0.2, np.nan, 0.4, -1.0, 0.7, 1.0, 1.5]  # 6 valid points
    res = compute_mu(t, x, window=3)
    assert res.n_points == 6  # nan + negative dropped


def test_compute_mu_raises_on_too_few_points() -> None:
    with pytest.raises(ValueError):
        compute_mu([0, 1, 2], [0.1, 0.2, 0.3])


def test_compute_mu_dedupe_keeps_first_at_duplicate_time() -> None:
    t = [0, 1, 1, 2, 3, 4, 5]
    x = [0.1, 0.2, 0.99, 0.3, 0.4, 0.5, 0.6]  # duplicate at t=1
    res = compute_mu(t, x, window=3)
    assert res.n_points == 6


# ---------- doubling_time ----------


def test_doubling_time_matches_analytic() -> None:
    t = np.linspace(0, 15, 31)
    mu_true = 0.5
    x = _exponential(t, mu_true=mu_true)
    res = doubling_time(t, x, window=5)
    expected = math.log(2) / mu_true
    # Allow ~10% slop because rolling mean understates mu_max slightly.
    assert math.isclose(res.t_doubling_h, expected, rel_tol=0.15)
    assert res.phase_end_h > res.phase_start_h


def test_doubling_time_raises_when_no_growth() -> None:
    t = np.linspace(0, 10, 21)
    x = np.full_like(t, 1.0)
    # constant biomass → mu_max ≈ 0 → doubling_time fails
    with pytest.raises(ValueError):
        doubling_time(t, x)


# ---------- segment_growth_phases ----------


def test_segment_phases_logistic_has_lag_exp_stationary() -> None:
    t = np.linspace(0, 30, 61)
    x = _logistic(t, mu_true=0.5, k=30.0)
    df = segment_growth_phases(t, x, window=5)
    assert not df.empty
    phases = set(df["phase"].tolist())
    # Logistic curve must produce exp + stationary at minimum.
    assert "exp" in phases
    assert "stationary" in phases


def test_segment_phases_returns_empty_below_min_points() -> None:
    df = segment_growth_phases([0, 1, 2, 3, 4, 5, 6], [0.1, 0.2, 0.4, 0.8, 1.6, 3.2, 6.4])
    assert df.empty


def test_segment_phases_columns() -> None:
    t = np.linspace(0, 20, 41)
    x = _exponential(t, mu_true=0.3)
    df = segment_growth_phases(t, x)
    assert list(df.columns) == [
        "phase",
        "start_h",
        "end_h",
        "mean_mu",
        "biomass_delta_g_l",
    ]


def test_segment_phases_rows_are_chronologically_ordered() -> None:
    t = np.linspace(0, 30, 61)
    x = _logistic(t, mu_true=0.5)
    df = segment_growth_phases(t, x)
    starts = df["start_h"].tolist()
    assert starts == sorted(starts)
    # Each phase ends at-or-after it starts.
    for _, row in df.iterrows():
        assert row["end_h"] >= row["start_h"]


# ---------- phasewise_mu ----------


def test_phasewise_mu_aligns_with_segmentation() -> None:
    t = np.linspace(0, 30, 61)
    x = _logistic(t, mu_true=0.5)
    seg = segment_growth_phases(t, x)
    pw = phasewise_mu(t, x)
    assert len(pw) == len(seg)
    # Every phase has at least one point counted.
    assert (pw["n_points"] >= 1).all()


def test_phasewise_mu_empty_when_too_few_points() -> None:
    pw = phasewise_mu([0, 1, 2, 3, 4], [0.1, 0.2, 0.4, 0.8, 1.6])
    assert pw.empty
