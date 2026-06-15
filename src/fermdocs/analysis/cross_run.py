"""Cross-run comparative intervention engine.

The dynamic bake-off (brewtwin families + discovered mechanistic) fits ONE
run's trajectory and scores held-out R^2. On sparse fed-batch data (5-9
points/run) those fits are unidentifiable and the rubric honestly refuses.

But the signal in a multi-run dataset lives ACROSS runs: many batches at
different operating conditions, each with an outcome (final titer). 15 runs
= 15 points in knob-space. This module reads the per-run operating-condition
knobs the layout detector pulled from sheet metadata
(``dossier["run_conditions"]``) and relates each knob to the run outcome, then
emits the controllable levers as interventions.

This is OBSERVATIONAL, not causal: it reports associations across runs that
were not designed as a controlled experiment. Every intervention carries that
caveat. The engine clears its gate only when there are enough runs, the knob
actually varies, and the association is large enough to matter — otherwise it
stays silent rather than invent a lever.

Pure + deterministic (numpy only); no LLM, no brewtwin, no I/O beyond the
DataFrame it is handed. Wired into the recommendation in agent.py: it WINS the
recommendation when the dynamic bake-off refuses but it clears its gate, and
SUPPLEMENTS the interventions when a dynamic model wins.
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np
import pandas as pd

# Minimum runs with both a knob value and an outcome to attempt a comparison.
MIN_RUNS = 4
# A knob's association must move the outcome by at least this fraction of the
# observed outcome spread to be worth recommending (filters noise).
MIN_EFFECT_FRAC = 0.15
# Default outcome variable (the thing we want to maximize).
DEFAULT_OBJECTIVE = "product_g_l"
# An association moving the objective by less than this fraction of the
# run-to-run spread is WEAK/preliminary — likely within noise on a small,
# confounded campaign. Used to label such associations so they aren't argued as
# confident levers (a +2 g/L swing on a ~60 g/L spread is not a validated effect).
WEAK_EFFECT_FRAC = 0.2


def is_weak_effect(norm_effect: float | None) -> bool:
    return (norm_effect or 0.0) < WEAK_EFFECT_FRAC


def run_outcomes(obs_df: pd.DataFrame, objective: str) -> dict[str, float]:
    """Peak value of the objective variable per run (robust to end dropout)."""
    df = obs_df.copy()
    if "variable" not in df or "run_id" not in df or "value" not in df:
        return {}
    df = df[df["variable"].astype(str) == objective]
    df["value"] = pd.to_numeric(df["value"], errors="coerce")
    df = df.dropna(subset=["value"])
    if df.empty:
        return {}
    return {str(rid): float(g["value"].max()) for rid, g in df.groupby("run_id")}


def _numeric_knob_effect(
    pairs: list[tuple[float, float]], outcome_spread: float,
    *, min_frac: float = MIN_EFFECT_FRAC,
) -> dict[str, Any] | None:
    """Linear association of a numeric knob with the outcome across runs.

    `min_frac` is the suppression gate: an effect smaller than this fraction of
    the outcome spread returns None. The recommend stage uses the default (only
    surface levers worth acting on); ranking callers pass 0.0 to get an UNgated
    number for every varying lever."""
    xs = np.array([p[0] for p in pairs], dtype=float)
    ys = np.array([p[1] for p in pairs], dtype=float)
    if len(set(xs.tolist())) < 2:
        return None  # knob does not vary
    slope, intercept = np.polyfit(xs, ys, 1)
    if not math.isfinite(slope):
        return None
    x_lo, x_hi = float(xs.min()), float(xs.max())
    # Predict at the in-range knob end the fit favors.
    best_x = x_hi if slope >= 0 else x_lo
    predicted = float(slope * best_x + intercept)
    baseline = float(np.median(ys))
    delta = predicted - baseline
    if outcome_spread <= 0 or abs(delta) < min_frac * outcome_spread:
        return None
    return {
        "best_setting": round(best_x, 4),
        "baseline_value": round(baseline, 4),
        "predicted_value": round(predicted, 4),
        "delta": round(delta, 4),
        "direction": "increase" if slope >= 0 else "decrease",
        "n": len(pairs),
        "observed_range": [round(x_lo, 4), round(x_hi, 4)],
    }


def _categorical_knob_effect(
    pairs: list[tuple[str, float]], outcome_spread: float,
    *, min_frac: float = MIN_EFFECT_FRAC,
) -> dict[str, Any] | None:
    """Best category vs overall mean for a categorical knob. `min_frac`: see
    `_numeric_knob_effect` (0.0 = ungated, for ranking)."""
    groups: dict[str, list[float]] = {}
    for cat, y in pairs:
        groups.setdefault(cat, []).append(y)
    if len(groups) < 2:
        return None  # knob does not vary
    means = {c: float(np.mean(v)) for c, v in groups.items()}
    overall = float(np.mean([y for _, y in pairs]))
    best_cat = max(means, key=means.get)
    delta = means[best_cat] - overall
    if outcome_spread <= 0 or abs(delta) < min_frac * outcome_spread:
        return None
    return {
        "best_setting": best_cat,
        "baseline_value": round(overall, 4),
        "predicted_value": round(means[best_cat], 4),
        "delta": round(delta, 4),
        "direction": "set_to",
        "n": len(pairs),
        "category_means": {c: round(m, 4) for c, m in means.items()},
    }


def analyze(
    dossier: dict[str, Any] | None,
    obs_df: pd.DataFrame,
    *,
    objective: str = DEFAULT_OBJECTIVE,
) -> dict[str, Any] | None:
    """Relate each per-run knob to the outcome across runs.

    Returns a dict ``{cleared, interventions, summary, n_runs, objective}`` or
    None when there are no usable conditions/outcomes. ``cleared`` is True only
    when at least one knob shows a large-enough association over >= MIN_RUNS
    runs; callers should treat a cleared result as a recommendable verdict and
    an uncleared one as "analyzed, nothing actionable".
    """
    conditions = (dossier or {}).get("run_conditions") or {}
    if not conditions:
        return None
    outcomes = run_outcomes(obs_df, objective)
    if len(outcomes) < MIN_RUNS:
        return {
            "cleared": False,
            "interventions": [],
            "n_runs": len(outcomes),
            "objective": objective,
            "summary": (
                f"cross-run analysis skipped: only {len(outcomes)} run(s) have "
                f"{objective}; need >= {MIN_RUNS}."
            ),
        }
    ys = list(outcomes.values())
    outcome_spread = float(max(ys) - min(ys))

    knob_pairs = _gather_knob_pairs(conditions, outcomes)

    interventions: list[dict[str, Any]] = []
    for knob, pairs in sorted(knob_pairs.items()):
        if len(pairs) < MIN_RUNS:
            continue
        numeric = all(isinstance(v, (int, float)) and not isinstance(v, bool) for v, _ in pairs)
        effect = (
            _numeric_knob_effect([(float(v), y) for v, y in pairs], outcome_spread)
            if numeric
            else _categorical_knob_effect([(str(v), y) for v, y in pairs], outcome_spread)
        )
        if effect is None:
            continue
        interventions.append(_to_intervention(knob, objective, effect))

    interventions.sort(key=lambda i: abs(i.get("delta") or 0.0), reverse=True)
    cleared = bool(interventions)
    if cleared:
        top = interventions[0]
        summary = (
            f"cross-run comparative over {len(outcomes)} runs: "
            f"{top['description']} (assoc. {top['delta']:+} {objective})."
        )
    else:
        summary = (
            f"cross-run analysis over {len(outcomes)} runs found no knob with a "
            f"large-enough association to {objective}."
        )
    return {
        "cleared": cleared,
        "interventions": interventions,
        "n_runs": len(outcomes),
        "objective": objective,
        "summary": summary,
    }


def _spec_value(spec: Any) -> Any:
    """Pull a usable value from a run_conditions cell: the extractor's clean
    `numeric` when present (set only when the whole cell was a number), else the
    verbatim `value` (categorical). Shared by analyze + lever_effects."""
    if isinstance(spec, dict):
        v = spec.get("numeric")
        return spec.get("value") if v is None else v
    return spec


def _gather_knob_pairs(
    conditions: dict[str, Any], outcomes: dict[str, float]
) -> dict[str, list[tuple[Any, float]]]:
    """Every knob seen across runs → its (value, outcome) pairs."""
    knob_pairs: dict[str, list[tuple[Any, float]]] = {}
    for run_id, knobs in conditions.items():
        outcome = outcomes.get(str(run_id))
        if outcome is None or not isinstance(knobs, dict):
            continue
        for knob, spec in knobs.items():
            val = _spec_value(spec)
            if val is None:
                continue
            knob_pairs.setdefault(knob, []).append((val, outcome))
    return knob_pairs


def lever_effects(
    dossier: dict[str, Any] | None,
    obs_df: pd.DataFrame,
    *,
    objective: str = DEFAULT_OBJECTIVE,
) -> dict[str, dict[str, Any]]:
    """Per-lever association with the objective across runs — UNGATED, so every
    varying metadata lever gets a number for topic ranking (not just the ones
    big enough to recommend). Returns ``{lever_name: {delta, direction, n,
    best_setting, norm_effect, ...}}`` where ``norm_effect`` in [0,1] is
    ``|delta| / outcome_spread``. Empty dict when there aren't enough runs or no
    outcomes — the caller then falls back to neutral ordering. Pure/deterministic.
    """
    conditions = (dossier or {}).get("run_conditions") or {}
    if not conditions:
        return {}
    outcomes = run_outcomes(obs_df, objective)
    if len(outcomes) < MIN_RUNS:
        return {}
    ys = list(outcomes.values())
    spread = float(max(ys) - min(ys))
    out: dict[str, dict[str, Any]] = {}
    for knob, pairs in _gather_knob_pairs(conditions, outcomes).items():
        if len(pairs) < MIN_RUNS:
            continue
        numeric = all(isinstance(v, (int, float)) and not isinstance(v, bool) for v, _ in pairs)
        effect = (
            _numeric_knob_effect([(float(v), y) for v, y in pairs], spread, min_frac=0.0)
            if numeric
            else _categorical_knob_effect([(str(v), y) for v, y in pairs], spread, min_frac=0.0)
        )
        if effect is None:
            continue
        norm = min(abs(float(effect["delta"])) / spread, 1.0) if spread > 0 else 0.0
        out[knob] = {**effect, "norm_effect": round(norm, 4)}
    return out


def _to_intervention(knob: str, objective: str, effect: dict[str, Any]) -> dict[str, Any]:
    direction = effect["direction"]
    if direction == "set_to":
        desc = f"Set {knob} to {effect['best_setting']}"
    else:
        desc = f"{direction.capitalize()} {knob} toward {effect['best_setting']}"
    caveat = (
        f"Observational association across {effect['n']} runs, not a controlled "
        "experiment; correlation is not proven causation — validate experimentally."
    )
    return {
        "knob": knob,
        "description": desc,
        "objective_metric": f"{objective}.peak",
        "baseline_value": effect["baseline_value"],
        "predicted_value": effect["predicted_value"],
        "delta": effect["delta"],
        "in_coverage": True,  # best setting is chosen within the observed range
        "caveat": caveat,
        "rationale": (
            f"Across runs, {knob} associates with {objective} "
            f"(n={effect['n']}); best observed setting {effect['best_setting']}."
        ),
    }
