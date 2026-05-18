from __future__ import annotations

from fermdocs_eval.metrics import (
    bootstrap_ci,
    confusion_matrix,
    over_fire_rate,
    per_axis_precision_recall,
    preference_rate,
)
from fermdocs_eval.synthetic import CRITIC_AXES


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
    # 7/10 A wins. CI should bracket 0.7 reasonably.
    rows = [{"winner": "A"}] * 7 + [{"winner": "B"}] * 3
    lo, hi = bootstrap_ci(rows, treatment="A", n_resamples=2000, seed=7)
    assert 0.3 <= lo <= 0.7
    assert 0.7 <= hi <= 1.0


def test_per_axis_pr_perfect() -> None:
    rows = [
        {"labeled_axis": "trajectory-axis", "fired_axes": ["trajectory-axis"]},
        {"labeled_axis": "trajectory-axis", "fired_axes": ["trajectory-axis"]},
        {"labeled_axis": "robustness-axis", "fired_axes": ["robustness-axis"]},
        {"labeled_axis": "clean", "fired_axes": []},
    ]
    out = per_axis_precision_recall(rows, CRITIC_AXES)
    assert out["trajectory-axis"]["precision"] == 1.0
    assert out["trajectory-axis"]["recall"] == 1.0
    assert out["robustness-axis"]["precision"] == 1.0
    assert out["robustness-axis"]["recall"] == 1.0


def test_per_axis_pr_with_misfire() -> None:
    rows = [
        # critic flagged trajectory but the truth is robustness — FP on trajectory, FN on robustness
        {"labeled_axis": "robustness-axis", "fired_axes": ["trajectory-axis"]},
        # correct trajectory fire — TP on trajectory
        {"labeled_axis": "trajectory-axis", "fired_axes": ["trajectory-axis"]},
    ]
    out = per_axis_precision_recall(rows, CRITIC_AXES)
    # trajectory: 1 TP + 1 FP -> precision 0.5
    assert out["trajectory-axis"]["precision"] == 0.5
    assert out["trajectory-axis"]["recall"] == 1.0
    # robustness: 0 TP + 1 FN -> recall 0
    assert out["robustness-axis"]["recall"] == 0.0


def test_over_fire_rate() -> None:
    rows = [
        {"labeled_axis": "clean", "fired_axes": []},
        {"labeled_axis": "clean", "fired_axes": ["robustness-axis"]},
        {"labeled_axis": "trajectory-axis", "fired_axes": ["trajectory-axis"]},
    ]
    out = over_fire_rate(rows)
    assert out["n_clean"] == 2
    assert out["any_fire"] == 1
    assert out["rate"] == 0.5


def test_confusion_matrix_shape() -> None:
    rows = [
        {"labeled_axis": "trajectory-axis", "fired_axes": ["trajectory-axis", "robustness-axis"]},
        {"labeled_axis": "clean", "fired_axes": []},
    ]
    m = confusion_matrix(rows, CRITIC_AXES)
    assert m["trajectory-axis"]["trajectory-axis"] == 1
    assert m["trajectory-axis"]["robustness-axis"] == 1
    assert m["clean"]["none"] == 1
