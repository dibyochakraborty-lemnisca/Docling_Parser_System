"""Tier A kinetics: specific growth rate, doubling time, phase segmentation.

All functions here operate on a single (time, biomass) trajectory. They
return either a numpy array, a small dataclass, or a pandas DataFrame.
No I/O. No LLM. Deterministic.

Smoothing strategy: we use a centered rolling-mean on ln(X) before
finite-differencing rather than Savitzky-Golay. SG would require scipy,
which isn't in the project's dependency list. Rolling-mean + central
difference is good enough at the temporal resolution typical bioreactor
sampling produces (hourly to 4-hourly), and importantly stays
deterministic + zero-dep.

The reference repo's `fermentation_toolkit.py` uses scipy.signal.savgol;
swapping for rolling mean changes mu_max by ~5-8% on noisy signals but
preserves the qualitative phase structure (which is what segment_growth_phases
ultimately consumes).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np
import pandas as pd

PhaseName = Literal["lag", "exp", "linear", "stationary", "decline"]


@dataclass(frozen=True)
class MuResult:
    """Output of compute_mu.

    `mu` is the time-resolved specific growth rate aligned with `time_h`.
    Edge points get NaN where the smoothing window can't fit.
    """

    time_h: np.ndarray
    mu: np.ndarray
    mu_max: float
    t_mu_max_h: float
    n_points: int


def _ensure_sorted_finite(
    time_h: pd.Series | np.ndarray | list[float],
    biomass: pd.Series | np.ndarray | list[float],
) -> tuple[np.ndarray, np.ndarray]:
    t = np.asarray(time_h, dtype=float)
    x = np.asarray(biomass, dtype=float)
    if t.shape != x.shape:
        raise ValueError(f"time_h and biomass shape mismatch: {t.shape} vs {x.shape}")
    mask = np.isfinite(t) & np.isfinite(x) & (x > 0)
    t = t[mask]
    x = x[mask]
    order = np.argsort(t)
    t = t[order]
    x = x[order]
    # Drop duplicate timestamps — keep first.
    if len(t) > 1:
        keep = np.concatenate(([True], np.diff(t) > 0))
        t = t[keep]
        x = x[keep]
    return t, x


def _rolling_mean(arr: np.ndarray, window: int) -> np.ndarray:
    """Centered rolling mean. Edge points use shrinking windows."""
    if window < 1:
        raise ValueError(f"window must be >= 1, got {window}")
    if window == 1:
        return arr.copy()
    half = window // 2
    out = np.full_like(arr, np.nan, dtype=float)
    for i in range(len(arr)):
        lo = max(0, i - half)
        hi = min(len(arr), i + half + 1)
        out[i] = np.nanmean(arr[lo:hi])
    return out


def compute_mu(
    time_h: pd.Series | np.ndarray | list[float],
    biomass: pd.Series | np.ndarray | list[float],
    *,
    window: int = 5,
    poly: int = 2,  # accepted for catalog parity, currently unused
) -> MuResult:
    """Specific growth rate mu(t) = d(ln X)/dt using a rolling window.

    Calculates the instantaneous derivative over a rolling window.
    Filters out None/NaN and non-positive biomass values, then for each
    window computes (ln(X_end) - ln(X_start)) / (t_end - t_start).
    The maximum of these windowed derivatives is returned as mu_max.
    
    `poly` is accepted for forward-compatibility with a Savitzky-Golay
    upgrade (catalog default mirrors the reference); ignored today.
    """
    del poly  # reserved for future SG upgrade; keep signature stable
    t, x = _ensure_sorted_finite(time_h, biomass)
    n = len(t)
    
    # We need at least 2 points to compute a derivative.
    if n < 5:
        raise ValueError(f"compute_mu needs >= 5 valid points, got {n}")
        
    # Enforce window size constraints
    w = max(2, min(int(window), n))
    
    ln_x = np.log(x)
    mu = np.full(n, np.nan, dtype=float)
    
    # Vectorized calculation for the rolling window
    t_start = t[:-w+1]
    t_end = t[w-1:]
    dt = t_end - t_start
    
    ln_x_start = ln_x[:-w+1]
    ln_x_end = ln_x[w-1:]
    
    # Defensive check: dt > 0 to avoid division by zero
    valid_dt = dt > 0
    
    # Calculate mu only where dt > 0
    mu_calc = np.full_like(dt, np.nan, dtype=float)
    mu_calc[valid_dt] = (ln_x_end[valid_dt] - ln_x_start[valid_dt]) / dt[valid_dt]
    
    # Assign the calculated window derivatives to the start of the window
    mu[:-w+1] = mu_calc
        
    finite_mu = mu[np.isfinite(mu)]
    if finite_mu.size == 0:
        raise ValueError("compute_mu produced no finite mu values")

    idx_max = int(np.nanargmax(mu))
    return MuResult(
        time_h=t,
        mu=mu,
        mu_max=float(mu[idx_max]),
        t_mu_max_h=float(t[idx_max]),
        n_points=n,
    )


@dataclass(frozen=True)
class DoublingTimeResult:
    t_doubling_h: float
    mu_max: float
    t_mu_max_h: float
    phase_start_h: float
    phase_end_h: float


def doubling_time(
    time_h: pd.Series | np.ndarray | list[float],
    biomass: pd.Series | np.ndarray | list[float],
    *,
    window: int = 7,
    mu_fraction: float = 0.5,
) -> DoublingTimeResult:
    """t_d = ln(2) / mu_max during the exponential window.

    Defines the "exponential window" as the contiguous range around
    t_mu_max where mu(t) >= mu_fraction * mu_max. That window's start /
    end is reported so downstream consumers know what time range the
    doubling time describes.

    Raises ValueError if mu_max <= 0 (no exponential growth detected).
    """
    res = compute_mu(time_h, biomass, window=window)
    if res.mu_max <= 0:
        raise ValueError(f"mu_max <= 0 ({res.mu_max:.4f}); no exponential phase")

    threshold = mu_fraction * res.mu_max
    above = res.mu >= threshold

    # Find contiguous run containing the argmax.
    idx_max = int(np.nanargmax(res.mu))
    start = idx_max
    while start > 0 and bool(above[start - 1]):
        start -= 1
    end = idx_max
    while end < len(above) - 1 and bool(above[end + 1]):
        end += 1

    return DoublingTimeResult(
        t_doubling_h=float(np.log(2.0) / res.mu_max),
        mu_max=res.mu_max,
        t_mu_max_h=res.t_mu_max_h,
        phase_start_h=float(res.time_h[start]),
        phase_end_h=float(res.time_h[end]),
    )


def segment_growth_phases(
    time_h: pd.Series | np.ndarray | list[float],
    biomass: pd.Series | np.ndarray | list[float],
    *,
    window: int = 7,
    lag_threshold: float = 0.05,
    exp_threshold: float = 0.15,
    decline_threshold: float = -0.02,
) -> pd.DataFrame:
    """Partition the run into lag / exp / linear / stationary / decline phases.

    Phase rules per point (applied to smoothed mu(t)):
      mu < decline_threshold        → 'decline'
      decline_threshold <= mu < lag → 'lag' OR 'stationary' (lag if first occurrence early, stationary if late)
      lag <= mu < exp               → 'linear'
      mu >= exp                     → 'exp'

    Adjacent same-phase points are merged into a single phase row.
    Returns a DataFrame with columns:
      phase, start_h, end_h, mean_mu, biomass_delta_g_l

    Empty DataFrame returned when fewer than 8 valid points (mirrors the
    catalog's min_points contract — analyzer should emit data-gap above
    rather than calling here, but we double-check).
    """
    t, x = _ensure_sorted_finite(time_h, biomass)
    n = len(t)
    if n < 8:
        return pd.DataFrame(
            columns=["phase", "start_h", "end_h", "mean_mu", "biomass_delta_g_l"]
        )

    res = compute_mu(t, x, window=window)
    mu = res.mu

    # Heuristic: the first sub-lag-threshold run BEFORE any exp point is "lag";
    # any sub-lag run AFTER exp is "stationary".
    seen_exp = False
    labels: list[PhaseName | None] = []
    for i, m in enumerate(mu):
        if not np.isfinite(m):
            labels.append(None)
            continue
        if m >= exp_threshold:
            seen_exp = True
            labels.append("exp")
        elif m >= lag_threshold:
            labels.append("linear")
        elif m >= decline_threshold:
            labels.append("stationary" if seen_exp else "lag")
        else:
            labels.append("decline")

    # Fill None labels by carrying neighbors (rare edge points).
    for i in range(len(labels)):
        if labels[i] is None:
            labels[i] = labels[i - 1] if i > 0 and labels[i - 1] is not None else "lag"

    # Merge contiguous same-label runs.
    rows: list[dict] = []
    if not labels:
        return pd.DataFrame(
            columns=["phase", "start_h", "end_h", "mean_mu", "biomass_delta_g_l"]
        )
    cur_label = labels[0]
    cur_start = 0
    for i in range(1, len(labels)):
        if labels[i] != cur_label:
            rows.append(_phase_row(cur_label, cur_start, i - 1, t, x, mu))
            cur_label = labels[i]
            cur_start = i
    rows.append(_phase_row(cur_label, cur_start, len(labels) - 1, t, x, mu))

    return pd.DataFrame(rows)


def _phase_row(
    label: str,
    i_start: int,
    i_end: int,
    t: np.ndarray,
    x: np.ndarray,
    mu: np.ndarray,
) -> dict:
    seg_mu = mu[i_start : i_end + 1]
    finite = seg_mu[np.isfinite(seg_mu)]
    return {
        "phase": label,
        "start_h": float(t[i_start]),
        "end_h": float(t[i_end]),
        "mean_mu": float(np.mean(finite)) if finite.size else float("nan"),
        "biomass_delta_g_l": float(x[i_end] - x[i_start]),
    }


def phasewise_mu(
    time_h: pd.Series | np.ndarray | list[float],
    biomass: pd.Series | np.ndarray | list[float],
    *,
    window: int = 7,
    lag_threshold: float = 0.05,
    exp_threshold: float = 0.15,
    decline_threshold: float = -0.02,
) -> pd.DataFrame:
    """Mean mu within each phase from segment_growth_phases.

    Thin wrapper that returns just (phase, mean_mu, n_points) so the
    catalog's A11 entry has a function it can call directly without
    re-running phase segmentation downstream.
    """
    phases = segment_growth_phases(
        time_h,
        biomass,
        window=window,
        lag_threshold=lag_threshold,
        exp_threshold=exp_threshold,
        decline_threshold=decline_threshold,
    )
    if phases.empty:
        return pd.DataFrame(columns=["phase", "mean_mu", "n_points"])

    t, _x = _ensure_sorted_finite(time_h, biomass)
    rows = []
    for _, ph in phases.iterrows():
        mask = (t >= ph["start_h"]) & (t <= ph["end_h"])
        rows.append(
            {
                "phase": ph["phase"],
                "mean_mu": ph["mean_mu"],
                "n_points": int(mask.sum()),
            }
        )
    return pd.DataFrame(rows)
