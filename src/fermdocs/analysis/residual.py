"""B3 — residual variance reporting (the honesty layer).

After ranking the levers, say how much of the objective's variance they actually
explain. When a large fraction is unexplained, the system flags 'there is a lever
you are not capturing' instead of false-confidently crowning the best of an
incomplete set. This is the boundary-of-knowledge signal — the antidote to
crowning feed-window (or anything) as THE driver when most variance is elsewhere.

A linear fit of the objective on all varying levers (numeric + one-hot
categoricals), built inline from run_conditions so this module stays free of any
fermdocs_optimize dependency (analysis must not import optimize).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from fermdocs.analysis.cross_run import DEFAULT_OBJECTIVE, MIN_RUNS, _spec_value, run_outcomes

# Fraction of UNexplained variance above which we flag a likely-missing lever.
# Statistical threshold, not a domain constant; validate on a 2nd dataset (V).
MISSING_LEVER_RESIDUAL = 0.4


@dataclass(frozen=True)
class ResidualReport:
    objective: str
    explained_r2: float          # variance explained by the captured levers
    unexplained: float           # 1 - explained
    n_runs: int
    n_lever_features: int
    likely_missing_lever: bool
    note: str


def _design(conditions: dict, outcomes: dict[str, float]) -> tuple[np.ndarray, np.ndarray, int]:
    """(X, y, n_features): one-hot categoricals + numeric levers over runs with an
    outcome. Intercept column included. Only knobs that VARY contribute."""
    runs = [r for r in outcomes if r in {str(k) for k in conditions}]
    runs = [str(r) for r in conditions if str(r) in outcomes]
    if len(runs) < MIN_RUNS:
        return np.empty((0, 0)), np.empty((0,)), 0
    # collect knob -> {run: value}
    knob_vals: dict[str, dict[str, Any]] = {}
    for r in runs:
        knobs = conditions[r] if isinstance(conditions.get(r), dict) else {}
        for k, spec in knobs.items():
            v = _spec_value(spec)
            if v is not None:
                knob_vals.setdefault(k, {})[r] = v
    cols: list[list[float]] = []
    for k, rv in knob_vals.items():
        if len(rv) < len(runs):
            continue  # knob missing on some runs -> skip (keeps the matrix aligned)
        distinct = {(_spec_value(v) if not isinstance(v, (int, float)) else round(float(v), 9))
                    for v in rv.values()}
        if len(distinct) < 2:
            continue  # constant -> no information
        numeric = all(isinstance(v, (int, float)) and not isinstance(v, bool)
                      for v in rv.values())
        if numeric:
            cols.append([float(rv[r]) for r in runs])
        else:
            cats = sorted({str(v) for v in rv.values()})
            for c in cats[1:]:  # drop one level (reference) to avoid collinearity
                cols.append([1.0 if str(rv[r]) == c else 0.0 for r in runs])
    y = np.array([outcomes[r] for r in runs], dtype=float)
    n_features = len(cols)
    intercept = [1.0] * len(runs)
    X = np.array([intercept, *cols], dtype=float).T if cols else np.array([intercept]).T
    return X, y, n_features


def residual_report(
    dossier: dict[str, Any] | None,
    obs_df: pd.DataFrame,
    *,
    objective: str = DEFAULT_OBJECTIVE,
    outcomes: dict[str, float] | None = None,
) -> ResidualReport | None:
    """Fraction of objective variance the captured levers explain, + a
    missing-lever flag when most variance is unexplained. None when unusable."""
    conditions = (dossier or {}).get("run_conditions") or {}
    outs = outcomes if outcomes is not None else run_outcomes(obs_df, objective)
    if not conditions or len(outs) < MIN_RUNS:
        return None
    X, y, n_features = _design(conditions, outs)
    if X.size == 0 or n_features == 0 or len(y) < MIN_RUNS:
        return None
    ss_tot = float(np.sum((y - y.mean()) ** 2))
    if ss_tot <= 0:
        return None
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    resid = y - X @ beta
    ss_res = float(np.sum(resid ** 2))
    r2 = max(0.0, min(1.0, 1.0 - ss_res / ss_tot))
    unexplained = round(1.0 - r2, 4)
    missing = unexplained > MISSING_LEVER_RESIDUAL
    note = (
        f"captured levers explain {round(100*r2)}% of {objective} variance; "
        f"{round(100*unexplained)}% unexplained"
        + (" — a lever driving the objective is likely NOT represented in the data "
           "(narrative/operating parameter not yet structured)." if missing else "."))
    return ResidualReport(objective, round(r2, 4), unexplained, len(y), n_features,
                          missing, note)
