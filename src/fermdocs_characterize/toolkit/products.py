"""Tier P toolkit: product-side KPIs that answer "which run was better?"

Plan ref: plans/2026-05-07-characterize-determinism.md commit 2.

The IndPenSim feedback exposed a systemic gap: the catalog had no
product-side metrics, so the synthesizer never reported final titer
even when it sat in an uncomputed column. This module fixes that.

Five metrics:
  P1 — final_product_titer:           last observed product value (g/L)
  P2 — peak_product_titer:            max product value + time of peak
  P3 — titer_decline_after_peak:     fractional drop from peak to final
                                       (the RUN-1 lysis signature: 21.6 → 14.3)
  P4 — integral_productivity:        ∫P/t dt (g/L/h area under curve)
  P5 — precursor_utilization:        (peak - final) / peak for a precursor
                                       variable; high = consumed efficiently
                                       (the IndPenSim PAA polarity fix)

All functions take simple lists / arrays and return a result dataclass.
They raise ValueError when their preconditions can't be met (e.g. fewer
than 2 valid points). Adapters in catalog_runner_adapters.py wrap these
into Findings.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Result dataclasses (frozen so adapters can hash if needed)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class FinalTiterResult:
    """P1 final product titer.

    `final_titer_g_l`: last non-NaN value in the trajectory.
    `t_final_h`: time at that observation, hours from run start.
    """

    final_titer_g_l: float
    t_final_h: float
    n_points: int


@dataclass(frozen=True)
class PeakTiterResult:
    """P2 peak product titer."""

    peak_titer_g_l: float
    t_peak_h: float
    n_points: int


@dataclass(frozen=True)
class TiterDeclineResult:
    """P3 titer decline after peak.

    `decline_fraction` = (peak - final) / peak.
    Range: [0, 1]. 0 = monotone (held at peak); 0.5 = lost half.
    Negative impossible (final ≤ peak by construction).
    `is_declining`: True when decline_fraction > 0.05 (5% loss is the
    threshold for 'something happened post-peak' — lysis, hydrolysis,
    or product re-uptake). 5% is conservative; real lysis events show
    20-40% drops.
    """

    decline_fraction: float
    peak_titer_g_l: float
    final_titer_g_l: float
    t_peak_h: float
    t_final_h: float
    is_declining: bool


@dataclass(frozen=True)
class IntegralProductivityResult:
    """P4 integral productivity ∫P/t dt.

    `integral_g_l_h`: trapezoidal integral of P over time.
    `mean_productivity_g_l_per_h`: integral / total_time. The
    'time-averaged titer' — useful for comparing runs of different
    durations.
    """

    integral_g_l_h: float
    mean_productivity_g_l_per_h: float
    duration_h: float
    n_points: int


@dataclass(frozen=True)
class PrecursorUtilizationResult:
    """P5 precursor utilization fraction.

    `utilization_fraction` = (peak - final) / peak. Polarity OPPOSITE
    of byproduct yield: high means the precursor was consumed
    (good for productivity); low means it accumulated (waste).

    `utilization_class`:
      'efficient'   if utilization_fraction >= 0.7
      'partial'     if 0.3 <= utilization_fraction < 0.7
      'wasted'      if utilization_fraction < 0.3 (precursor accumulating)

    The IndPenSim case: RUN-2 PAA went from peak → 634 mg/L (utilization
    high), RUN-1 stayed at 5203 mg/L (wasted).
    """

    utilization_fraction: float
    peak_value: float
    final_value: float
    utilization_class: str
    precursor_variable: str


# ---------------------------------------------------------------------------
# Compute functions
# ---------------------------------------------------------------------------


def _clean_pair(
    time_h, values
) -> tuple[np.ndarray, np.ndarray]:
    """Drop NaN rows, sort by time. Returns (t, v) numpy arrays.

    Raises ValueError when fewer than 2 surviving points (the minimum
    for any P-tier metric to be meaningful).
    """
    t_arr = np.asarray(time_h, dtype=float)
    v_arr = np.asarray(values, dtype=float)
    if t_arr.shape != v_arr.shape:
        raise ValueError(
            f"time and values shape mismatch: {t_arr.shape} vs {v_arr.shape}"
        )
    mask = np.isfinite(t_arr) & np.isfinite(v_arr)
    t_arr = t_arr[mask]
    v_arr = v_arr[mask]
    if len(t_arr) < 2:
        raise ValueError(
            f"need >= 2 finite (time, value) pairs; got {len(t_arr)}"
        )
    order = np.argsort(t_arr)
    return t_arr[order], v_arr[order]


def compute_final_titer(time_h, product_g_l) -> FinalTiterResult:
    """P1: last observed product titer."""
    t, v = _clean_pair(time_h, product_g_l)
    return FinalTiterResult(
        final_titer_g_l=float(v[-1]),
        t_final_h=float(t[-1]),
        n_points=int(len(t)),
    )


def compute_peak_titer(time_h, product_g_l) -> PeakTiterResult:
    """P2: max product titer + time of peak."""
    t, v = _clean_pair(time_h, product_g_l)
    idx = int(np.argmax(v))
    return PeakTiterResult(
        peak_titer_g_l=float(v[idx]),
        t_peak_h=float(t[idx]),
        n_points=int(len(t)),
    )


def compute_titer_decline(
    time_h, product_g_l, *, decline_threshold: float = 0.05
) -> TiterDeclineResult:
    """P3: fractional drop from peak titer to final titer.

    The lysis / hydrolysis signature: when product peaks mid-run and
    declines by run-end, something is consuming or degrading product.
    `decline_threshold` (default 5%) is the floor for `is_declining`;
    smaller drops are within sampling noise.
    """
    t, v = _clean_pair(time_h, product_g_l)
    peak_idx = int(np.argmax(v))
    peak = float(v[peak_idx])
    final = float(v[-1])
    if peak <= 0:
        # No product produced — decline is undefined; report 0.0 and
        # let the synthesizer interpret peak=0 separately.
        return TiterDeclineResult(
            decline_fraction=0.0,
            peak_titer_g_l=0.0,
            final_titer_g_l=final,
            t_peak_h=float(t[peak_idx]),
            t_final_h=float(t[-1]),
            is_declining=False,
        )
    decline = max(0.0, (peak - final) / peak)
    return TiterDeclineResult(
        decline_fraction=float(decline),
        peak_titer_g_l=peak,
        final_titer_g_l=final,
        t_peak_h=float(t[peak_idx]),
        t_final_h=float(t[-1]),
        is_declining=bool(decline > decline_threshold),
    )


def compute_integral_productivity(
    time_h, product_g_l
) -> IntegralProductivityResult:
    """P4: trapezoidal area under the product curve.

    Useful for comparing runs of different durations: a run that hits a
    high peak briefly may have lower integral than one that holds a
    moderate level for longer. mean_productivity = integral / duration.
    """
    t, v = _clean_pair(time_h, product_g_l)
    integral = float(np.trapezoid(v, t))
    duration = float(t[-1] - t[0])
    if duration <= 0:
        raise ValueError(f"non-positive duration {duration}")
    return IntegralProductivityResult(
        integral_g_l_h=integral,
        mean_productivity_g_l_per_h=integral / duration,
        duration_h=duration,
        n_points=int(len(t)),
    )


def compute_precursor_utilization(
    time_h,
    precursor_values,
    *,
    precursor_variable: str,
) -> PrecursorUtilizationResult:
    """P5: how much of a precursor was consumed during the run.

    Polarity OPPOSITE byproduct yield. Precursors get fed in (peak
    visible mid-run from the feed event), then drawn down as cells
    incorporate them into product. High utilization = good. Low
    utilization = waste / unproductive feeding.
    """
    t, v = _clean_pair(time_h, precursor_values)
    peak = float(np.max(v))
    final = float(v[-1])
    if peak <= 0:
        # No precursor ever present — not feeding it; precursor
        # utilization is undefined for this run.
        raise ValueError(
            f"precursor {precursor_variable!r}: peak value <= 0;"
            " no feeding event detected in trajectory"
        )
    fraction = max(0.0, (peak - final) / peak)
    if fraction >= 0.7:
        cls = "efficient"
    elif fraction >= 0.3:
        cls = "partial"
    else:
        cls = "wasted"
    return PrecursorUtilizationResult(
        utilization_fraction=float(fraction),
        peak_value=peak,
        final_value=final,
        utilization_class=cls,
        precursor_variable=precursor_variable,
    )
