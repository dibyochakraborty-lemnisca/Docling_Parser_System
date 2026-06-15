"""Data-backed oracle: verify against the uploaded experiment, not a simulator.

The LABS oracle answers "what titer at these knobs?" by running a process
simulator. This oracle answers the same question from the REAL runs: k-nearest
neighbours in (normalised) knob space, distance-weighted. It interpolates among
experiments that were actually run — it never invents physics outside the data,
so it can only speak to the regime you've explored. Use it when you want the
optimizer grounded in the uploaded data instead of a generic simulator.

Honesty notes:
  - Only the knobs the data actually VARIES are used for the neighbour
    distance (`active_knobs`). A knob the dataset doesn't have (e.g. malt_frac
    for a lactic-acid process) is neither used nor invented — it is pinned in
    the search box and ignored here.
  - There is no extrapolation: a query outside the observed envelope just maps
    to the nearest real runs. The optimizer's box should be bounded to the
    observed range so "optimum" means "best point your data supports".
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from fermdocs_optimize.schema import KNOB_NAMES, Box, Candidate

# Map an internal optimizer species to the canonical observation variable it
# corresponds to. Structural mapping only (which column is which) — no expected
# values. Drugs/products differ by process; product_g_l is the titer target.
_TARGET_VAR = "product_g_l"
# Candidate variables for each derived knob, first present wins. These are
# STRUCTURAL (which measurement stands in for which knob), not magic numbers.
_KNOB_SOURCES = {
    "biomass": ("biomass_g_l", "od600_au", "wcw_g_l"),
    "total_sub": ("substrate_g_l",),
    "dilution": ("volume_l",),
}


class DataSimulator:
    """Simulator-shaped oracle backed by observed runs (k-NN over knobs)."""

    def __init__(
        self,
        knob_matrix: np.ndarray,
        peaks: np.ndarray,
        *,
        active_knobs: list[str],
        target: str = "P",
        k: int = 3,
    ) -> None:
        self._active = list(active_knobs)
        self._target = target
        self._k = max(1, int(k))
        self._X = np.asarray(knob_matrix, dtype=float)  # (n_runs, n_active)
        self._peaks = np.asarray(peaks, dtype=float)
        if self._X.ndim != 2 or self._X.shape[0] == 0:
            raise ValueError("DataSimulator needs at least one observed run")
        self._lo = self._X.min(axis=0)
        self._span = np.maximum(self._X.max(axis=0) - self._lo, 1e-9)
        self._Xn = (self._X - self._lo) / self._span

    def simulate(self, candidates: list[Candidate], *, v0: float) -> pd.DataFrame:
        rows = []
        for i, c in enumerate(candidates):
            q = np.array([getattr(c, kn) for kn in self._active], dtype=float)
            qn = (q - self._lo) / self._span
            d = np.linalg.norm(self._Xn - qn, axis=1)
            k = min(self._k, len(d))
            order = np.argsort(d)[:k]
            w = 1.0 / (d[order] + 1e-9)
            w = w / w.sum()
            peak = float(np.dot(w, self._peaks[order]))
            # One row per candidate; peak_titer_per_batch takes max of the
            # target column per batch, so a single point suffices.
            rows.append({"batch": i, "t": 0.0, self._target: peak})
        return pd.DataFrame(rows)


def runs_from_observations(
    obs_df: pd.DataFrame, *, target_var: str = _TARGET_VAR
) -> tuple[list[dict], list[str]]:
    """Derive per-run knobs + outcome from long observations.

    Returns (runs, active_knobs). Each run dict has run_id, peak (the target's
    max), and each derived knob (initial inoculum, initial substrate loading,
    volume-based dilution). `active_knobs` are the knobs present in every run
    AND varying across runs — the only ones worth fitting/optimizing. Shared by
    the k-NN oracle, the surrogate equation, and the mechanistic optimizer.
    Raises ValueError if the data can't support an optimization.
    """
    df = obs_df.copy()
    for col in ("run_id", "variable", "time_h", "value"):
        if col not in df.columns:
            raise ValueError(f"observations missing required column {col!r}")
    df["value"] = pd.to_numeric(df["value"], errors="coerce")
    df["time_h"] = pd.to_numeric(df["time_h"], errors="coerce")

    def _first(run_df: pd.DataFrame, names: tuple[str, ...]) -> float | None:
        for name in names:
            s = run_df[run_df["variable"] == name].dropna(subset=["value"])
            if not s.empty:
                return float(s.sort_values("time_h")["value"].iloc[0])
        return None

    def _peak(run_df: pd.DataFrame, name: str) -> float | None:
        s = run_df[run_df["variable"] == name].dropna(subset=["value"])
        return float(s["value"].max()) if not s.empty else None

    def _dilution(run_df: pd.DataFrame) -> float | None:
        s = run_df[run_df["variable"] == "volume_l"].dropna(subset=["value"])
        if s.empty:
            return None
        v = s["value"]
        vmin = float(v.min())
        return float((v.max() - vmin) / vmin) if vmin > 0 else 0.0

    runs: list[dict] = []
    for run_id, run_df in df.groupby("run_id"):
        peak = _peak(run_df, target_var)
        if peak is None:
            continue  # no outcome -> can't use this run
        knobs = {
            "biomass": _first(run_df, _KNOB_SOURCES["biomass"]),
            "total_sub": _first(run_df, _KNOB_SOURCES["total_sub"]),
            "dilution": _dilution(run_df),
        }
        runs.append({"run_id": str(run_id), "peak": peak, **knobs})

    if len(runs) < 2:
        raise ValueError(f"need >=2 runs with {target_var}; got {len(runs)}")

    active: list[str] = []
    for kn in ("biomass", "total_sub", "dilution"):
        vals = [r[kn] for r in runs]
        if any(v is None for v in vals):
            continue
        if len({round(v, 9) for v in vals}) >= 2:
            active.append(kn)
    if not active:
        raise ValueError("no controllable knob varies across runs; nothing to optimize")
    return runs, active


def build_data_oracle(
    obs_df: pd.DataFrame, *, target_var: str = _TARGET_VAR, k: int = 3
) -> tuple[DataSimulator, Box, list[str]]:
    """Build a DataSimulator + observed-envelope Box from long observations.

    Returns (oracle, box, active_knobs). Raises ValueError if the data can't
    support an optimization (no target, <2 runs, or no varying knob).
    """
    runs, active = runs_from_observations(obs_df, target_var=target_var)
    knob_matrix = np.array([[r[kn] for kn in active] for r in runs], dtype=float)
    peaks = np.array([r["peak"] for r in runs], dtype=float)
    oracle = DataSimulator(knob_matrix, peaks, active_knobs=active, k=k)

    # Box: observed [min, max] for active knobs; pin everything else (incl.
    # malt_frac) to a neutral constant so it is never optimized or invented.
    box_kwargs: dict[str, tuple[float, float]] = {}
    for kn in KNOB_NAMES:
        if kn in active:
            col = knob_matrix[:, active.index(kn)]
            box_kwargs[kn] = (float(col.min()), float(col.max()))
        else:
            box_kwargs[kn] = (0.0, 0.0)  # pinned-inactive (not faked)
    return oracle, Box(**box_kwargs), active


def optimize_on_data(
    obs_df: pd.DataFrame, *, k: int = 3, n_lhs: int = 200, seed: int = 11, v0: float = 10.0
):
    """End-to-end: build the data oracle from observations and search the
    observed envelope for the best operating point. Returns
    (OracleSearchReport, active_knobs). The report's `knobs_on_boundary` tells
    you which knobs sit at the edge of what was actually run — a boundary
    optimum means the real best may lie beyond your data (you'd need a new
    experiment to know).
    """
    from fermdocs_optimize.oracle_search import oracle_global_search

    oracle, box, active = build_data_oracle(obs_df, k=k)
    report = oracle_global_search(
        oracle, box, objective_species="P", v0=v0, n_lhs=n_lhs, seed=seed
    )
    return report, active
