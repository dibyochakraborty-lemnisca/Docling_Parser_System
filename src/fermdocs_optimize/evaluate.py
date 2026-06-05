"""Evaluation helpers: peak titer, model-vs-oracle agreement, convergence."""
from __future__ import annotations

import numpy as np
import pandas as pd

from fermdocs_optimize.models.base import PredictiveModel
from fermdocs_optimize.schema import Candidate


def peak_titer_per_batch(sim_df: pd.DataFrame, species: str = "P") -> dict[int, float]:
    """Max of `species` per batch in a simulator output dataframe."""
    return {int(b): float(g[species].max()) for b, g in sim_df.groupby("batch")}


def best_achieved(sim_df: pd.DataFrame, candidates: list[Candidate],
                  species: str = "P") -> tuple[Candidate, float]:
    """Return the (candidate, peak titer) with the highest simulated titer.

    Candidate i maps to the i-th batch in sorted batch-id order — robust to
    simulators that number batches from 0 (stub) or 1 (LABS)."""
    peaks = peak_titer_per_batch(sim_df, species)
    obs = _ordered_peaks(peaks, len(candidates))
    best_i = int(np.argmax(obs))
    return candidates[best_i], float(obs[best_i])


def _ordered_peaks(peaks: dict[int, float], n: int) -> np.ndarray:
    """Peaks aligned to candidate order via sorted batch ids."""
    ids = sorted(peaks)
    if len(ids) != n:
        raise ValueError(f"simulator returned {len(ids)} batches for {n} candidates")
    return np.array([peaks[b] for b in ids])


def model_vs_oracle_r2(model: PredictiveModel, candidates: list[Candidate],
                       sim_df: pd.DataFrame, *, v0: float,
                       species: str = "P") -> float:
    """R^2 of model-predicted peak titer vs oracle-simulated peak titer over the
    proposed candidates. Low value => the model is wrong where we're sampling =>
    fold the new data in and refit."""
    peaks = peak_titer_per_batch(sim_df, species)
    pred = np.array([model.predict_peak_titer(c, v0=v0) for c in candidates])
    obs = _ordered_peaks(peaks, len(candidates))
    ss_res = float(np.sum((obs - pred) ** 2))
    ss_tot = float(np.sum((obs - obs.mean()) ** 2))
    if ss_tot == 0:  # all candidates simulate to ~same titer
        return 1.0 if ss_res < 1e-6 else 0.0
    return 1.0 - ss_res / ss_tot


def sim_to_training_rows(sim_df: pd.DataFrame) -> pd.DataFrame:
    """Normalize a simulator output to the training schema (batch,t,X,S,P,M,V).

    Reconciles O2/DO naming; keeps only columns the model fits on + V."""
    keep = ["batch", "t", "X", "S", "P", "M", "V"]
    return sim_df[[c for c in keep if c in sim_df.columns]].copy()


def append_training(train: pd.DataFrame, new_rows: pd.DataFrame) -> pd.DataFrame:
    """Append new oracle data, renumbering new batches to avoid id collisions."""
    offset = int(train["batch"].max()) + 1 if len(train) else 0
    nr = new_rows.copy()
    nr["batch"] = nr["batch"].astype(int) + offset
    return pd.concat([train, nr], ignore_index=True)


def converged(trajectory: list[float], threshold: float) -> tuple[bool, float | None]:
    """Convergence when the last improvement in best-achieved titer < threshold."""
    if len(trajectory) < 2:
        return False, None
    delta = trajectory[-1] - trajectory[-2]
    return delta < threshold, delta
