"""Bootstrap CI on Pearson correlation.

Plan ref: plans/2026-05-07-rigour-and-actionability.md commit 2.

The 7.5/10 review flagged that r=-0.90 with n=6 should never ship
without a CI. compute_correlation returns r, n, ci_low, ci_high, and
weak_n_flag deterministically from a fixed seed.
"""

from __future__ import annotations

import numpy as np
import pytest

from fermdocs_characterize.toolkit.cross_run import (
    WEAK_N_THRESHOLD,
    CorrelationResult,
    compute_correlation,
)


# ---------- shape ----------


def test_returns_correlation_result_dataclass() -> None:
    rng = np.random.default_rng(0)
    x = rng.normal(size=20)
    y = 2 * x + rng.normal(scale=0.1, size=20)
    res = compute_correlation(x, y)
    assert isinstance(res, CorrelationResult)
    assert res.n == 20
    assert res.r > 0.95
    assert res.weak_n_flag is False  # n=20 >= 8


# ---------- weak n ----------


def test_weak_n_flag_set_when_n_below_threshold() -> None:
    """The IndPenSim case: n=6 must be flagged regardless of r magnitude."""
    x = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]
    y = [2.0, 4.0, 6.0, 8.0, 10.0, 12.0]
    res = compute_correlation(x, y)
    assert res.n == 6
    assert res.weak_n_flag is True
    assert res.r > 0.99  # perfect linear


def test_weak_n_threshold_boundary() -> None:
    """n == WEAK_N_THRESHOLD is NOT weak; n == threshold-1 is weak."""
    rng = np.random.default_rng(0)
    x = rng.normal(size=WEAK_N_THRESHOLD)
    y = rng.normal(size=WEAK_N_THRESHOLD)
    assert compute_correlation(x, y).weak_n_flag is False

    x = rng.normal(size=WEAK_N_THRESHOLD - 1)
    y = rng.normal(size=WEAK_N_THRESHOLD - 1)
    assert compute_correlation(x, y).weak_n_flag is True


# ---------- CI behavior ----------


def test_strong_correlation_large_n_has_tight_ci() -> None:
    rng = np.random.default_rng(0)
    x = rng.normal(size=200)
    y = 0.95 * x + rng.normal(scale=0.05, size=200)
    res = compute_correlation(x, y)
    assert res.ci_high - res.ci_low < 0.05
    assert res.ci_low > 0.9


def test_weak_n_widens_ci_relative_to_strong_n() -> None:
    """Same true correlation, fewer points → wider CI. Pin the relationship."""
    rng = np.random.default_rng(0)
    x_small = rng.normal(size=6)
    y_small = 0.9 * x_small + rng.normal(scale=0.3, size=6)
    res_small = compute_correlation(x_small, y_small)

    rng2 = np.random.default_rng(0)
    x_big = rng2.normal(size=100)
    y_big = 0.9 * x_big + rng2.normal(scale=0.3, size=100)
    res_big = compute_correlation(x_big, y_big)

    width_small = res_small.ci_high - res_small.ci_low
    width_big = res_big.ci_high - res_big.ci_low
    assert width_small > width_big


# ---------- determinism ----------


def test_deterministic_across_calls() -> None:
    rng = np.random.default_rng(0)
    x = rng.normal(size=15)
    y = rng.normal(size=15)
    a = compute_correlation(x, y, seed=42)
    b = compute_correlation(x, y, seed=42)
    assert a == b


def test_different_seeds_give_different_ci() -> None:
    rng = np.random.default_rng(0)
    x = rng.normal(size=15)
    y = rng.normal(size=15)
    a = compute_correlation(x, y, seed=1)
    b = compute_correlation(x, y, seed=2)
    assert a.r == b.r  # point estimate identical
    assert (a.ci_low, a.ci_high) != (b.ci_low, b.ci_high)


# ---------- edge cases ----------


def test_drops_paired_nans() -> None:
    x = [1.0, 2.0, float("nan"), 4.0, 5.0, 6.0, 7.0, 8.0]
    y = [2.0, 4.0, 99.0, 8.0, 10.0, 12.0, 14.0, 16.0]
    res = compute_correlation(x, y)
    assert res.n == 7  # the nan-paired entry dropped


def test_zero_variance_returns_zero_r() -> None:
    """Constant input → r undefined; surface as 0.0 with weak_n_flag."""
    x = [1.0] * 10
    y = list(range(10))
    res = compute_correlation(x, y)
    assert res.r == 0.0
    assert res.weak_n_flag is True


def test_too_few_points_raises() -> None:
    with pytest.raises(ValueError, match=r"≥2 finite"):
        compute_correlation([1.0], [2.0])


def test_mismatched_shapes_raise() -> None:
    with pytest.raises(ValueError, match="same shape"):
        compute_correlation([1, 2, 3], [1, 2])


# ---------- REGRESSION ----------


def test_existing_cross_run_helpers_still_importable() -> None:
    """Adding compute_correlation must not break A19/A20/A21 imports."""
    from fermdocs_characterize.toolkit.cross_run import (  # noqa: F401
        cross_run_kpi_table,
        pairwise_deviation,
        variance_decomposition,
    )
