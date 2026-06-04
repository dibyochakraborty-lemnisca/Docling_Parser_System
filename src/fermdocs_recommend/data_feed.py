"""Bridge the bundle's observations.csv into brewtwin-ready inputs.

`observations.csv` is long-format with columns run_id, variable, time_h, value,
imputed, unit (ragged per-variable grids: dense online sensors + sparse offline
assays). These helpers are imported BOTH by the agent's sandbox code (to build
trajectories and score held-out fits) and by get_data_feed (to hand the agent a
classified summary so it does not have to re-derive the state/control split).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

# NOTE: brewtwin (and its JAX stack) is imported lazily inside build_feed only,
# so the API process can call summarize()/classify_variables()/leave_one_run_out()
# without paying the ~8.6s JAX import. Only the sandbox subprocess, which calls
# build_feed, actually imports brewtwin.

# Name-driven state/control split (the CSV does not tag this). Controls are the
# operator knobs; everything else is a measured state/analyte the model fits.
_CONTROL_SUBSTRINGS = (
    "agitation", "_rpm", "temperature", "_bar", "pressure",
    "gas_flow", "feed_rate", "stir", "airflow", "aeration",
)
# Feed-rate-like channels that drive fed-batch dilution (the FedBatchReactor
# feed profile). The primary substrate feed is preferred when several exist.
_FEED_SUBSTRINGS = ("feed_rate", "_feed", "fs", "substrate_feed")


def _is_control(var: str) -> bool:
    v = var.lower()
    return any(s in v for s in _CONTROL_SUBSTRINGS)


def _is_feed(var: str) -> bool:
    v = var.lower()
    return any(s in v for s in _FEED_SUBSTRINGS)


def classify_variables(variables: list[str]) -> dict[str, list[str]]:
    """Split observed variables into states, controls, and feed candidates."""
    controls, states, feeds = [], [], []
    for v in variables:
        if _is_feed(v):
            feeds.append(v)
        if _is_control(v):
            controls.append(v)
        else:
            states.append(v)
    return {"states": states, "controls": controls, "feed_candidates": feeds}


def detect_feed_var(variables: list[str]) -> str | None:
    feeds = [v for v in variables if _is_feed(v)]
    if not feeds:
        return None
    # prefer an explicit substrate feed-rate channel
    for v in feeds:
        if "feed_rate" in v.lower():
            return v
    return feeds[0]


def summarize(obs_df: pd.DataFrame) -> dict:
    """Classified summary for the agent's get_data_feed tool.

    Reports per-variable finite-point counts so the agent can see which states
    are dense (online) vs sparse (offline ~20 pts) before choosing what to fit.
    """
    obs_df = obs_df.copy()
    obs_df["value"] = pd.to_numeric(obs_df["value"], errors="coerce")
    variables = sorted(obs_df["variable"].unique().tolist())
    run_ids = sorted(obs_df["run_id"].unique().tolist())
    cls = classify_variables(variables)
    counts: dict[str, dict] = {}
    for v in variables:
        sub = obs_df[obs_df["variable"] == v]
        finite = int(sub["value"].notna().sum())
        real = int(sub[sub.get("imputed", 0) == 0]["value"].notna().sum()) if "imputed" in sub else finite
        counts[v] = {"finite_points": finite, "real_points": real}
    return {
        "run_ids": [str(r) for r in run_ids],
        "n_runs": len(run_ids),
        "variables": variables,
        "states": cls["states"],
        "controls": cls["controls"],
        "feed_var": detect_feed_var(variables),
        "point_counts": counts,
        "units": {
            v: (obs_df[obs_df["variable"] == v]["unit"].dropna().iloc[0]
                if "unit" in obs_df and obs_df[obs_df["variable"] == v]["unit"].notna().any()
                else None)
            for v in variables
        },
    }


def leave_one_run_out(run_ids: list[str]) -> tuple[list[str], str]:
    """Train on all-but-last run, validate on the last. Single-run falls back
    to training and validating on the same run (the agent must then last-N split)."""
    run_ids = sorted(run_ids)
    if len(run_ids) >= 2:
        return run_ids[:-1], run_ids[-1]
    return run_ids, run_ids[0]


def build_feed(obs_df: pd.DataFrame, feed_var: str | None = None) -> dict:
    """Long-format observations.csv -> {run_id: brewtwin Trajectory}.

    The designated feed channel becomes a condition channel (a time-varying
    input), not a fitted species. Sparse offline points keep their NaN gaps;
    fit() / fit_metrics mask them. Imputed points are preserved; held-out
    scoring uses get_real_observations to keep imputed values out of validation.
    """
    df = obs_df.copy()
    df["value"] = pd.to_numeric(df["value"], errors="coerce")
    df["time_h"] = pd.to_numeric(df["time_h"], errors="coerce")
    df = df.dropna(subset=["time_h"]).sort_values(by=["run_id", "time_h"])

    from brewtwin.data.schemas import Trajectory  # lazy: only the sandbox needs JAX

    trajectories: dict = {}
    for run_id, run_df in df.groupby("run_id"):
        pivoted = run_df.pivot_table(
            index="time_h", columns="variable", values="value", aggfunc="first"
        )
        t_grid = pivoted.index.to_numpy(dtype=float)
        concentrations: dict[str, np.ndarray] = {}
        conditions: dict[str, tuple] = {}
        for var in pivoted.columns:
            if feed_var and var == feed_var:
                vals = pivoted[var].copy()
                if vals.isna().any():
                    vals = vals.interpolate(method="linear").bfill().ffill().fillna(0.0)
                conditions[var] = (t_grid, vals.to_numpy(dtype=float))
            else:
                concentrations[var] = pivoted[var].to_numpy(dtype=float)
        trajectories[str(run_id)] = Trajectory.from_dense(
            t=t_grid,
            concentrations=concentrations,
            conditions=conditions or None,
        )
    return trajectories


def get_real_observations(
    obs_df: pd.DataFrame, run_id: str, species: list[str]
) -> tuple[np.ndarray, np.ndarray]:
    """(t_eval, y_obs) of measured-only (imputed==0) points for held-out scoring.

    y_obs is (T, n_species) with NaN where a species was not measured at t.
    """
    df = obs_df.copy()
    df["value"] = pd.to_numeric(df["value"], errors="coerce")
    df["time_h"] = pd.to_numeric(df["time_h"], errors="coerce")
    mask = df["run_id"].astype(str) == str(run_id)
    if "imputed" in df:
        mask &= df["imputed"] == 0
    df = df[mask]
    if df.empty:
        return np.array([]), np.empty((0, len(species)))
    pivoted = df.pivot_table(
        index="time_h", columns="variable", values="value", aggfunc="first"
    )
    t_eval = pivoted.index.to_numpy(dtype=float)
    y_obs = np.full((len(t_eval), len(species)), np.nan)
    for j, sp in enumerate(species):
        if sp in pivoted.columns:
            y_obs[:, j] = pivoted[sp].to_numpy(dtype=float)
    return t_eval, y_obs
