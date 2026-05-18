"""Pure metric functions over result rows. No I/O, no LLM calls.

Kept deterministic so paper figures can be regenerated from results.jsonl
without re-running anything.
"""

from __future__ import annotations

import random
from collections import Counter, defaultdict
from typing import Any, Iterable


def preference_rate(
    rows: Iterable[dict[str, Any]],
    *,
    treatment: str = "A",
) -> dict[str, float | int]:
    """Compute preference rate for `treatment` across judge rows.

    Each row must have a `winner` field ("A", "B", or "tie").
    """
    rows = list(rows)
    if not rows:
        return {"n": 0, "treatment_wins": 0, "baseline_wins": 0, "ties": 0, "rate": 0.0}
    winners = Counter(r["winner"] for r in rows)
    other = "B" if treatment == "A" else "A"
    n = len(rows)
    return {
        "n": n,
        "treatment_wins": winners[treatment],
        "baseline_wins": winners[other],
        "ties": winners["tie"],
        "rate": winners[treatment] / n,
    }


def bootstrap_ci(
    rows: list[dict[str, Any]],
    *,
    treatment: str = "A",
    n_resamples: int = 2000,
    seed: int = 0,
    alpha: float = 0.05,
) -> tuple[float, float]:
    """Bootstrap (lo, hi) CI for preference rate. Returns (0.0, 0.0) if empty."""
    if not rows:
        return (0.0, 0.0)
    rng = random.Random(seed)
    rates: list[float] = []
    n = len(rows)
    for _ in range(n_resamples):
        sample = [rows[rng.randrange(n)] for _ in range(n)]
        wins = sum(1 for r in sample if r["winner"] == treatment)
        rates.append(wins / n)
    rates.sort()
    lo = rates[int(n_resamples * (alpha / 2))]
    hi = rates[int(n_resamples * (1 - alpha / 2))]
    return (lo, hi)


def catch_rate(rows: list[dict]) -> dict[str, float | int]:
    """E2 headline metric A: did the critic catch the defect at all?

    A defect-labeled fixture (labeled_axis != 'clean') is a CATCH if
    ANY axis fired. A clean-labeled fixture is a FALSE POSITIVE if any
    axis fired.

    Reports catch rate over defect fixtures and false-positive rate over
    clean fixtures separately so they aren't conflated.
    """
    defects = [r for r in rows if r["labeled_axis"] != "clean"]
    cleans = [r for r in rows if r["labeled_axis"] == "clean"]
    caught = sum(1 for r in defects if r.get("fired_axes"))
    false_pos = sum(1 for r in cleans if r.get("fired_axes"))
    return {
        "n_defect": len(defects),
        "n_caught": caught,
        "catch_rate": caught / len(defects) if defects else 0.0,
        "n_clean": len(cleans),
        "n_false_positive": false_pos,
        "false_positive_rate": false_pos / len(cleans) if cleans else 0.0,
    }


def tag_accuracy(rows: list[dict]) -> dict[str, float | int]:
    """E2 headline metric B: when the critic caught the defect, did it
    tag the correct axis?

    Only counts CAUGHT defect fixtures (labeled_axis != 'clean' and
    fired_axes non-empty). Among those, a fixture has CORRECT TAG if
    labeled_axis appears in fired_axes (multi-tag is fine — partial
    credit for at least getting the right axis in the mix).

    catch_rate * tag_accuracy gives the strict per-axis recall.
    """
    caught = [
        r for r in rows
        if r["labeled_axis"] != "clean" and r.get("fired_axes")
    ]
    correct = sum(1 for r in caught if r["labeled_axis"] in r["fired_axes"])
    return {
        "n_caught": len(caught),
        "n_correct_tag": correct,
        "tag_accuracy": correct / len(caught) if caught else 0.0,
    }


def per_axis_precision_recall(
    rows: Iterable[dict[str, Any]],
    axes: list[str],
) -> dict[str, dict[str, float | int]]:
    """E2 metric: per critic axis precision and recall.

    Each row must have:
      - `labeled_axis`: ground-truth axis name (or "clean" for negatives)
      - `fired_axes`: list of axes the critic flagged

    Precision_a = correct fires on axis a / total fires on axis a
    Recall_a    = correct fires on axis a / total labeled axis-a cases
    """
    rows = list(rows)
    tp: dict[str, int] = defaultdict(int)
    fp: dict[str, int] = defaultdict(int)
    fn: dict[str, int] = defaultdict(int)
    labeled_counts: dict[str, int] = defaultdict(int)

    for r in rows:
        labeled = r["labeled_axis"]
        fired = set(r.get("fired_axes", []))
        labeled_counts[labeled] += 1
        for axis in axes:
            if axis in fired and labeled == axis:
                tp[axis] += 1
            elif axis in fired and labeled != axis:
                fp[axis] += 1
            elif axis not in fired and labeled == axis:
                fn[axis] += 1

    out: dict[str, dict[str, float | int]] = {}
    for axis in axes:
        denom_p = tp[axis] + fp[axis]
        denom_r = tp[axis] + fn[axis]
        out[axis] = {
            "tp": tp[axis],
            "fp": fp[axis],
            "fn": fn[axis],
            "precision": tp[axis] / denom_p if denom_p else 0.0,
            "recall": tp[axis] / denom_r if denom_r else 0.0,
            "labeled_count": labeled_counts.get(axis, 0),
        }
    return out


def over_fire_rate(rows: Iterable[dict[str, Any]]) -> dict[str, float | int]:
    """E2: how often any axis fires on a labeled-clean hypothesis."""
    rows = list(rows)
    clean = [r for r in rows if r["labeled_axis"] == "clean"]
    if not clean:
        return {"n_clean": 0, "any_fire": 0, "rate": 0.0}
    any_fire = sum(1 for r in clean if r.get("fired_axes"))
    return {"n_clean": len(clean), "any_fire": any_fire, "rate": any_fire / len(clean)}


def confusion_matrix(
    rows: Iterable[dict[str, Any]],
    axes: list[str],
) -> dict[str, dict[str, int]]:
    """E2: nested dict [labeled_axis][fired_axis] = count.

    A row contributes to every (labeled, fired) pair it presents — multi-fire
    rows count in multiple columns.
    """
    matrix: dict[str, dict[str, int]] = {
        labeled: {fired: 0 for fired in axes + ["none"]}
        for labeled in axes + ["clean"]
    }
    for r in rows:
        labeled = r["labeled_axis"]
        fired = r.get("fired_axes", [])
        if not fired:
            matrix[labeled]["none"] += 1
        else:
            for f in fired:
                if f in matrix[labeled]:
                    matrix[labeled][f] += 1
    return matrix
