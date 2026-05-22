"""Pure metric functions over result rows. No I/O, no LLM calls.

Kept deterministic so paper figures can be regenerated from results.jsonl
without re-running anything.

Scope: head-to-head agent-vs-baseline preference rate, multi-axis score
aggregation, and bootstrap CIs. Earlier axis-based metrics (per-axis P/R,
confusion matrix, catch_rate, tag_accuracy) were removed when the eval
scope was narrowed to head-to-head only.
"""

from __future__ import annotations

import random
import statistics
from collections import Counter
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


def per_axis_means(
    rows: Iterable[dict[str, Any]],
    *,
    axes: tuple[str, ...] = ("specificity", "grounding", "actionability", "honesty"),
    role: str = "treatment",
) -> dict[str, dict[str, float | int]]:
    """Aggregate multi-axis scores from judge rows.

    Each row must carry per-axis scores under
        row["scores"][role][axis_name]  ->  int in 1-10.

    `role` is "treatment" or "baseline" — we compute both by calling twice.

    Returns {axis: {n, mean, stdev, min, max}}.
    """
    out: dict[str, dict[str, float | int]] = {}
    rows = list(rows)
    for axis in axes:
        values: list[float] = []
        for r in rows:
            scores = r.get("scores") or {}
            role_scores = scores.get(role) or {}
            v = role_scores.get(axis)
            if isinstance(v, (int, float)):
                values.append(float(v))
        if not values:
            out[axis] = {"n": 0, "mean": 0.0, "stdev": 0.0, "min": 0.0, "max": 0.0}
        else:
            out[axis] = {
                "n": len(values),
                "mean": statistics.fmean(values),
                "stdev": statistics.stdev(values) if len(values) > 1 else 0.0,
                "min": min(values),
                "max": max(values),
            }
    return out


def per_axis_delta(
    rows: Iterable[dict[str, Any]],
    *,
    axes: tuple[str, ...] = ("specificity", "grounding", "actionability", "honesty"),
) -> dict[str, dict[str, float | int]]:
    """Per-axis treatment-minus-baseline score delta.

    Each row must carry both treatment and baseline scores under
        row["scores"]["treatment"][axis] and row["scores"]["baseline"][axis].
    Returns {axis: {n, mean_delta, wins, losses, ties}}.
    """
    out: dict[str, dict[str, float | int]] = {}
    rows = list(rows)
    for axis in axes:
        deltas: list[float] = []
        wins = losses = ties = 0
        for r in rows:
            scores = r.get("scores") or {}
            t = (scores.get("treatment") or {}).get(axis)
            b = (scores.get("baseline") or {}).get(axis)
            if isinstance(t, (int, float)) and isinstance(b, (int, float)):
                d = float(t) - float(b)
                deltas.append(d)
                if d > 0:
                    wins += 1
                elif d < 0:
                    losses += 1
                else:
                    ties += 1
        out[axis] = {
            "n": len(deltas),
            "mean_delta": statistics.fmean(deltas) if deltas else 0.0,
            "wins": wins,
            "losses": losses,
            "ties": ties,
        }
    return out
