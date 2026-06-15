"""Tier A operational metrics: bioreactor running conditions.

These are organism-agnostic — they describe what the equipment did,
not what the cells did. Most are simple geometric or threshold-based
calculations against a single trajectory.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import pandas as pd


# -----------------------------------------------------------------------------
# A14 — DO margin profile
# -----------------------------------------------------------------------------


@dataclass(frozen=True)
class DOMarginResult:
    """Dissolved-oxygen margin summary over the run.

    Margin at each time = DO(t) - critical_threshold. Negative margin
    means the cells were O2-limited.

    `frac_below`: fraction of timepoints with DO < critical_threshold.
    `min_margin`: deepest deficit seen.
    `time_below_h`: time-weighted hours spent below threshold (linear
        interpolation between samples; reasonable for hourly sampling).
    `never_aerobic`: DO never rose above the threshold at ANY timepoint
        (max_do <= threshold). When true, "below threshold" is the operating
        regime (consistent with anaerobic/microaerophilic operation), NOT an
        oxygen-limitation event — you cannot be O2-limited relative to a demand
        the process never aerated for. Callers must NOT report this as a
        bottleneck. This is a data signal, not an organism assumption.
    """

    frac_below: float
    min_margin: float
    min_do: float
    max_do: float
    mean_do: float
    time_below_h: float
    n_points: int
    never_aerobic: bool


def compute_do_margin(
    time_h: pd.Series | np.ndarray | list[float],
    do_pct: pd.Series | np.ndarray | list[float],
    *,
    critical_threshold_pct: float = 30.0,
) -> DOMarginResult:
    """Profile DO against an O2-limitation threshold.

    Default 30% saturation matches the textbook lower bound for aerobic
    fermentation; can be overridden when the bundle's organism has a
    documented different sensitivity.
    """
    t = np.asarray(time_h, dtype=float)
    do = np.asarray(do_pct, dtype=float)
    if t.shape != do.shape:
        raise ValueError(f"time_h and do_pct shape mismatch: {t.shape} vs {do.shape}")

    mask = np.isfinite(t) & np.isfinite(do)
    t = t[mask]
    do = do[mask]
    if len(t) < 2:
        raise ValueError(f"compute_do_margin needs >= 2 valid points, got {len(t)}")

    order = np.argsort(t)
    t = t[order]
    do = do[order]

    margin = do - critical_threshold_pct
    below = do < critical_threshold_pct

    # Time below: trapezoidal integration of the boolean mask against time.
    # Each interval contributes (t[i+1] - t[i]) when both endpoints are
    # below the threshold; (t[i+1] - t[i]) / 2 when exactly one is.
    time_below = 0.0
    for i in range(len(t) - 1):
        dt = t[i + 1] - t[i]
        if below[i] and below[i + 1]:
            time_below += dt
        elif below[i] or below[i + 1]:
            time_below += dt / 2.0

    return DOMarginResult(
        frac_below=float(below.sum()) / float(len(do)),
        min_margin=float(np.min(margin)),
        min_do=float(np.min(do)),
        max_do=float(np.max(do)),
        mean_do=float(np.mean(do)),
        time_below_h=float(time_below),
        n_points=int(len(t)),
        # DO never exceeded the limitation threshold at any sample -> there was
        # never an aerobic regime to be limited relative to. Data-derived, not an
        # organism prior: this gates the "O2 bottleneck" interpretation downstream.
        never_aerobic=bool(np.max(do) <= critical_threshold_pct),
    )


# -----------------------------------------------------------------------------
# A15 — Controller excursion count
# -----------------------------------------------------------------------------


@dataclass(frozen=True)
class ExcursionResult:
    """Count and duration of windows outside ±tolerance from setpoint.

    `excursions`: list of (start_h, end_h, peak_deviation) tuples.
    Each entry is a contiguous run of out-of-band points; entering and
    exiting the band ends one excursion.
    """

    n_excursions: int
    total_time_out_h: float
    max_abs_deviation: float
    excursions: list[tuple[float, float, float]]


def controller_excursions(
    time_h: pd.Series | np.ndarray | list[float],
    measured: pd.Series | np.ndarray | list[float],
    setpoint: pd.Series | np.ndarray | list[float] | float,
    *,
    tolerance: float,
) -> ExcursionResult:
    """Find windows where |measured - setpoint| > tolerance.

    Setpoint can be either a constant scalar or a per-time series. The
    tolerance is in the same units as the measurement (e.g. ±0.05 pH
    units, ±2°C, ±5% DO).
    """
    t = np.asarray(time_h, dtype=float)
    m = np.asarray(measured, dtype=float)
    if isinstance(setpoint, (int, float)):
        sp = np.full_like(t, float(setpoint))
    else:
        sp = np.asarray(setpoint, dtype=float)
    if not (t.shape == m.shape == sp.shape):
        raise ValueError("time_h / measured / setpoint shape mismatch")
    if tolerance <= 0:
        raise ValueError(f"tolerance must be > 0, got {tolerance}")

    mask = np.isfinite(t) & np.isfinite(m) & np.isfinite(sp)
    t = t[mask]
    m = m[mask]
    sp = sp[mask]
    if len(t) < 2:
        raise ValueError(f"controller_excursions needs >= 2 valid points, got {len(t)}")

    order = np.argsort(t)
    t = t[order]
    m = m[order]
    sp = sp[order]

    deviation = m - sp
    out_of_band = np.abs(deviation) > tolerance

    excursions: list[tuple[float, float, float]] = []
    in_excursion = False
    start_idx = 0
    peak = 0.0
    for i, oob in enumerate(out_of_band):
        if oob and not in_excursion:
            in_excursion = True
            start_idx = i
            peak = float(deviation[i])
        elif oob and in_excursion:
            if abs(deviation[i]) > abs(peak):
                peak = float(deviation[i])
        elif not oob and in_excursion:
            in_excursion = False
            excursions.append((float(t[start_idx]), float(t[i - 1]), peak))
            peak = 0.0
    if in_excursion:
        excursions.append((float(t[start_idx]), float(t[-1]), peak))

    total_time = sum(end - start for start, end, _ in excursions)
    max_abs = float(np.max(np.abs(deviation))) if deviation.size else 0.0

    return ExcursionResult(
        n_excursions=len(excursions),
        total_time_out_h=float(total_time),
        max_abs_deviation=max_abs,
        excursions=excursions,
    )


# -----------------------------------------------------------------------------
# A17 — Impeller tip speed
# -----------------------------------------------------------------------------


def tip_speed(
    rpm: pd.Series | np.ndarray | list[float] | float,
    impeller_diameter_m: float,
) -> float | np.ndarray:
    """v_tip = π · D · N (N in rev/s, D in m → v in m/s).

    Accepts a scalar RPM or a timecourse and returns the same shape.
    Tip speeds above ~5 m/s flag potential shear stress for shear-
    sensitive cells; we don't apply that threshold here, just return
    the number.
    """
    if impeller_diameter_m <= 0:
        raise ValueError(f"impeller_diameter_m must be > 0, got {impeller_diameter_m}")
    if isinstance(rpm, (int, float)):
        return math.pi * impeller_diameter_m * (float(rpm) / 60.0)
    arr = np.asarray(rpm, dtype=float)
    return math.pi * impeller_diameter_m * (arr / 60.0)


# -----------------------------------------------------------------------------
# A18 — Volumetric power input P/V
# -----------------------------------------------------------------------------


def power_per_volume(
    rpm: pd.Series | np.ndarray | list[float] | float,
    impeller_diameter_m: float,
    fluid_density_kg_m3: float,
    working_volume_l: float,
    *,
    power_number: float = 5.0,
) -> float | np.ndarray:
    """P/V from agitation power draw against working volume.

    P = Np · ρ · N³ · D⁵     (ungassed power, N in rev/s)
    P/V → W/m³.

    `power_number` defaults to 5.0 for a Rushton turbine in turbulent
    regime (Bates 1963); pitched-blade impellers run ~1.3-2.0. Override
    when the bundle reports a different impeller geometry.
    """
    if impeller_diameter_m <= 0:
        raise ValueError(f"impeller_diameter_m must be > 0, got {impeller_diameter_m}")
    if fluid_density_kg_m3 <= 0:
        raise ValueError(f"fluid_density_kg_m3 must be > 0, got {fluid_density_kg_m3}")
    if working_volume_l <= 0:
        raise ValueError(f"working_volume_l must be > 0, got {working_volume_l}")

    volume_m3 = working_volume_l / 1000.0

    def _one(n_rpm: float) -> float:
        n_rps = n_rpm / 60.0
        power_w = (
            power_number
            * fluid_density_kg_m3
            * (n_rps**3)
            * (impeller_diameter_m**5)
        )
        return power_w / volume_m3

    if isinstance(rpm, (int, float)):
        return _one(float(rpm))
    arr = np.asarray(rpm, dtype=float)
    return np.array([_one(v) for v in arr])
