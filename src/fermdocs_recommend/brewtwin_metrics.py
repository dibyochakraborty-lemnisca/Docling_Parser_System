"""Fit-quality + plausibility metrics, vendored from the brewtwin skills.

These functions live ONLY as snippets inside brewtwin's `analyze-and-interpret`
SKILL.md — they are not shipped as importable brewtwin code and the skills are
not packaged into the wheel. Rather than have the recommendation agent paste
their bodies into every LLM-emitted script (unverifiable, NaN-handling left to
the model), we vendor them here as a tested module that both the in-sandbox
fit script and the pure-Python rubric import.

Source of truth: brewtwin-main/skills/analyze-and-interpret/SKILL.md (fit_metrics
lines 23-47, parameter_plausibility 64-77, build_report 89-127). Kept faithful
to the skill semantics, with two deliberate strengthenings flagged in-line:
  * parameter_plausibility enforces the Y (yield) range the skill *table* lists
    but the skill *code* omitted, and reports unknown keys instead of silently
    passing them (a plausibility gate that passes unknown keys is not a gate).
"""

from __future__ import annotations

from typing import Any

import numpy as np

# Typical literature ranges for batch/fed-batch fermentation. From the skill's
# plausibility table (analyze-and-interpret SKILL.md). `Y` is included here even
# though the skill's reference *code* omitted it — the table lists 0.3-0.6.
PLAUSIBILITY_RANGES: dict[str, tuple[float, float]] = {
    "mu_max": (0.05, 2.0),   # 1/h
    "Ks": (0.01, 5.0),       # g/L
    "Y": (0.3, 0.6),         # g_X / g_S
    "P_max": (20.0, 120.0),  # g/L
    "Ki": (0.01, 100.0),     # g/L (substrate/product inhibition constant)
    "Kc": (0.01, 5.0),       # Contois constant (dimensionless-ish)
}


def fit_metrics(
    y_pred: np.ndarray, y_obs: np.ndarray, species: list[str]
) -> dict[str, dict[str, float]]:
    """Per-species RMSE and R^2 over finite (non-NaN) observations.

    Parameters
    ----------
    y_pred : (T, n_species) predicted trajectory
    y_obs  : (T, n_species) observed data (NaN = not observed at that time)
    species: ordered species name list

    Returns
    -------
    dict keyed by species name, each with {"rmse": float, "r2": float,
    "n": int}. `r2` is NaN when a species has no finite observations or when
    the observed variance (ss_tot) is zero (a plateau) — R^2 is undefined
    there, which the rubric treats as "cannot validate on this species".
    """
    y_pred = np.asarray(y_pred, dtype=float)
    y_obs = np.asarray(y_obs, dtype=float)
    out: dict[str, dict[str, float]] = {}
    for j, sp in enumerate(species):
        m = np.isfinite(y_obs[:, j]) & np.isfinite(y_pred[:, j])
        n = int(m.sum())
        if n == 0:
            out[sp] = {"rmse": float("nan"), "r2": float("nan"), "n": 0}
            continue
        resid = y_pred[m, j] - y_obs[m, j]
        rmse = float(np.sqrt(np.mean(resid**2)))
        ss_tot = float(np.sum((y_obs[m, j] - y_obs[m, j].mean()) ** 2))
        r2 = float(1.0 - np.sum(resid**2) / ss_tot) if ss_tot > 0 else float("nan")
        out[sp] = {"rmse": rmse, "r2": r2, "n": n}
    return out


def parameter_plausibility(params: dict[str, float]) -> dict[str, dict[str, Any]]:
    """Check fitted parameters against typical fermentation ranges.

    Returns a dict keyed identically to `params`, each value carrying the
    value, a `plausible` flag, the `range` checked against, and `known`
    (False when the parameter name has no reference range — such a value is
    reported as not-plausible-checkable rather than silently passing, so the
    rubric can decide whether an unranged parameter blocks a mechanism claim).
    """
    result: dict[str, dict[str, Any]] = {}
    for name, value in params.items():
        rng = PLAUSIBILITY_RANGES.get(name)
        if rng is None:
            result[name] = {
                "value": value,
                "plausible": False,
                "known": False,
                "range": None,
            }
            continue
        lo, hi = rng
        result[name] = {
            "value": value,
            "plausible": bool(lo <= value <= hi),
            "known": True,
            "range": [lo, hi],
        }
    return result


def prediction_summary(
    y_pred: np.ndarray, t_pred: np.ndarray, species: list[str]
) -> dict[str, dict[str, float]]:
    """Peak value/time and final value per species — the counterfactual readout."""
    y_pred = np.asarray(y_pred, dtype=float)
    t_pred = np.asarray(t_pred, dtype=float)
    summary: dict[str, dict[str, float]] = {}
    for j, sp in enumerate(species):
        col = y_pred[:, j]
        if not np.isfinite(col).any():
            summary[sp] = {
                "peak_value": float("nan"),
                "peak_time": float("nan"),
                "final_value": float("nan"),
            }
            continue
        idx_peak = int(np.nanargmax(col))
        summary[sp] = {
            "peak_value": float(col[idx_peak]),
            "peak_time": float(t_pred[idx_peak]),
            "final_value": float(col[np.isfinite(col).nonzero()[0][-1]]),
        }
    return summary


def build_report(
    *,
    model_type: str,
    fitted_params: dict[str, float],
    loss_history: list[float],
    y_pred: np.ndarray,
    y_obs: np.ndarray,
    t_pred: np.ndarray,
    species: list[str],
    fit_window: tuple[float, float],
    predict_window: tuple[float, float],
    caveats: tuple[str, ...] = (),
) -> dict[str, Any]:
    """Structured interpretation dict — the per-candidate scorecard.

    JSON-serialisable. Mirrors the skill's build_report but adds the
    optimizer-movement signal (`loss_reduction_frac`) the rubric needs to tell
    a stalled fit (data does not constrain params) from a converged poor fit.
    """
    metrics = fit_metrics(y_pred, y_obs, species)
    plaus = parameter_plausibility(fitted_params) if fitted_params else {}
    summary = prediction_summary(y_pred, t_pred, species)

    lh = [float(x) for x in loss_history] if loss_history is not None else []
    loss_reduction_frac = (
        float((lh[0] - lh[-1]) / lh[0]) if len(lh) >= 2 and lh[0] != 0 else 0.0
    )

    return {
        "model_type": model_type,
        "fitted_parameters": plaus,
        "fit_quality": metrics,
        "final_loss": float(lh[-1]) if lh else float("nan"),
        "loss_reduction_frac": loss_reduction_frac,
        "prediction_summary": summary,
        "fit_window": {"t_start": fit_window[0], "t_end": fit_window[1]},
        "predict_window": {"t_start": predict_window[0], "t_end": predict_window[1]},
        "caveats": list(caveats),
    }
