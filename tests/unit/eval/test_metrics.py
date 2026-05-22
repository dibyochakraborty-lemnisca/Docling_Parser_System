from __future__ import annotations

from fermdocs_eval.metrics import (
    bootstrap_ci,
    per_axis_delta,
    per_axis_means,
    preference_rate,
)


def test_preference_rate_basic() -> None:
    rows = [
        {"winner": "A"},
        {"winner": "A"},
        {"winner": "B"},
        {"winner": "tie"},
    ]
    out = preference_rate(rows, treatment="A")
    assert out["n"] == 4
    assert out["treatment_wins"] == 2
    assert out["baseline_wins"] == 1
    assert out["ties"] == 1
    assert out["rate"] == 0.5


def test_preference_rate_empty() -> None:
    out = preference_rate([], treatment="A")
    assert out["n"] == 0
    assert out["rate"] == 0.0


def test_bootstrap_ci_collapses_on_unanimous() -> None:
    rows = [{"winner": "A"}] * 10
    lo, hi = bootstrap_ci(rows, treatment="A", n_resamples=500, seed=42)
    assert lo == 1.0 and hi == 1.0


def test_bootstrap_ci_brackets_truth() -> None:
    rows = [{"winner": "A"}] * 7 + [{"winner": "B"}] * 3
    lo, hi = bootstrap_ci(rows, treatment="A", n_resamples=2000, seed=7)
    assert 0.3 <= lo <= 0.7
    assert 0.7 <= hi <= 1.0


def test_per_axis_means_basic() -> None:
    rows = [
        {"scores": {"treatment": {"specificity": 8, "grounding": 7}}},
        {"scores": {"treatment": {"specificity": 6, "grounding": 9}}},
    ]
    out = per_axis_means(rows, axes=("specificity", "grounding"), role="treatment")
    assert out["specificity"]["n"] == 2
    assert out["specificity"]["mean"] == 7.0
    assert out["grounding"]["mean"] == 8.0


def test_per_axis_means_missing_role_returns_zeros() -> None:
    rows = [{"scores": {"treatment": {"specificity": 8}}}]
    out = per_axis_means(rows, axes=("specificity",), role="baseline")
    assert out["specificity"]["n"] == 0
    assert out["specificity"]["mean"] == 0.0


def test_per_axis_delta_wins_losses_ties() -> None:
    rows = [
        {"scores": {"treatment": {"specificity": 8}, "baseline": {"specificity": 5}}},
        {"scores": {"treatment": {"specificity": 6}, "baseline": {"specificity": 8}}},
        {"scores": {"treatment": {"specificity": 7}, "baseline": {"specificity": 7}}},
    ]
    out = per_axis_delta(rows, axes=("specificity",))
    assert out["specificity"]["n"] == 3
    assert out["specificity"]["wins"] == 1
    assert out["specificity"]["losses"] == 1
    assert out["specificity"]["ties"] == 1
    # mean delta = (3 + -2 + 0) / 3 = 1/3
    assert abs(out["specificity"]["mean_delta"] - (1.0 / 3.0)) < 1e-9
