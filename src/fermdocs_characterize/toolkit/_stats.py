"""Robust statistics helpers: median, IQR, skew detection.

Plan ref: plans/2026-05-07-characterize-determinism.md commit 4.

The IndPenSim feedback flagged that mean RQ 1.21 and median RQ 0.98
tell different stories. Mean is sensitive to transient spikes;
for RQ time-series with feed-event spikes, mean overestimates
overflow signal. Median is the honest summary.

This module provides:
  - central_tendency(arr): both mean and median + IQR + recommended
  - is_skewed(arr): heuristic for 'mean disagrees with median enough
    that median is the better summary'

Used by toolkit B10 (and any other adapter that aggregates a
time-series with potential transients) to surface BOTH statistics so
the synthesizer can prefer median for skewed signals.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np


@dataclass(frozen=True)
class CentralTendency:
    """Both mean and median, plus the IQR for spread, and a
    `recommended_summary` field naming which one to surface.

    `recommended_summary`: 'median' when the series is skewed enough
    that mean misrepresents the typical value; 'mean' otherwise.
    Synthesizer prompt rule (commit 4) says: when both are present,
    surface the recommended one and cite the other when they tell
    different stories.
    """

    mean: float
    median: float
    p25: float
    p75: float
    iqr: float
    n_points: int
    recommended_summary: str  # 'mean' or 'median'


def _to_array(arr: Iterable[float]) -> np.ndarray:
    a = np.asarray(list(arr), dtype=float)
    return a[np.isfinite(a)]


def central_tendency(values: Iterable[float]) -> CentralTendency:
    """Compute mean, median, p25, p75, IQR + recommend which one to
    surface based on skew.

    Raises ValueError on fewer than 2 finite values (no meaningful
    summary).
    """
    arr = _to_array(values)
    if len(arr) < 2:
        raise ValueError(f"need >= 2 finite values; got {len(arr)}")
    mean = float(np.mean(arr))
    median = float(np.median(arr))
    p25 = float(np.percentile(arr, 25))
    p75 = float(np.percentile(arr, 75))
    iqr = p75 - p25
    recommended = "median" if _is_skewed(arr, mean=mean, median=median) else "mean"
    return CentralTendency(
        mean=mean,
        median=median,
        p25=p25,
        p75=p75,
        iqr=iqr,
        n_points=int(len(arr)),
        recommended_summary=recommended,
    )


def is_skewed(values: Iterable[float]) -> bool:
    """True when the series is skewed enough that median is a better
    summary than mean. Public surface; same heuristic as the internal
    check in central_tendency."""
    arr = _to_array(values)
    if len(arr) < 2:
        return False
    return _is_skewed(arr, mean=float(np.mean(arr)), median=float(np.median(arr)))


def _is_skewed(arr: np.ndarray, *, mean: float, median: float) -> bool:
    """Heuristic: mean / median ratio > 1.15 (or < 0.87) means a
    >15% disagreement between the two summaries — skew is meaningful.

    Falls back to a robust skew measure when median is near zero (avoid
    divide-by-zero): use mean - median normalized by IQR.

    Threshold 1.15 was picked because:
      - 1.0 (perfectly symmetric) is the no-skew baseline.
      - For RQ in IndPenSim: mean 1.21, median 0.98 → ratio 1.23, well
        past 1.15 → flagged. Synthesizer surfaces median.
      - For RQ in clean aerobic runs: mean 0.96, median 0.96 →
        ratio 1.0 → not flagged. Synthesizer surfaces mean (and shows
        median equals it, no contradiction).
      - 15% is the threshold above which a non-specialist reader of
        the prose summary would notice the discrepancy.
    """
    if abs(median) < 1e-9:
        # Median near zero — ratio is unstable. Use IQR-normalized
        # mean-median offset instead.
        p25, p75 = np.percentile(arr, [25, 75])
        iqr = float(p75 - p25)
        if iqr < 1e-9:
            return False
        return abs(mean - median) / iqr > 0.5
    ratio = mean / median
    return ratio > 1.15 or ratio < 0.87
