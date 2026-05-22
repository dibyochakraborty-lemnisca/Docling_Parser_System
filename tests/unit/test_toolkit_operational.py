"""Synthetic-data tests for the Tier A operational toolkit."""

from __future__ import annotations

import math

import numpy as np
import pytest

from fermdocs_characterize.toolkit.operational import (
    compute_do_margin,
    controller_excursions,
    power_per_volume,
    tip_speed,
)


# ---------- compute_do_margin ----------


def test_do_margin_no_violation_when_above_threshold() -> None:
    t = np.linspace(0, 10, 11)
    do = np.full_like(t, 50.0)
    res = compute_do_margin(t, do, critical_threshold_pct=30.0)
    assert res.frac_below == 0.0
    assert res.min_margin == 20.0
    assert res.time_below_h == 0.0


def test_do_margin_full_violation_when_below_threshold() -> None:
    t = np.linspace(0, 10, 11)
    do = np.full_like(t, 10.0)
    res = compute_do_margin(t, do, critical_threshold_pct=30.0)
    assert res.frac_below == 1.0
    assert res.min_margin == -20.0
    assert math.isclose(res.time_below_h, 10.0)  # entire run below


def test_do_margin_partial_violation() -> None:
    # Half the run below: hours 0-5 at 10%, hours 5-10 at 50%.
    t = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
    do = np.array([10, 10, 10, 10, 10, 10, 50, 50, 50, 50, 50])
    res = compute_do_margin(t, do, critical_threshold_pct=30.0)
    # 6 of 11 points below.
    assert math.isclose(res.frac_below, 6 / 11)
    # Time below ≈ 5 hours of full + 0.5h transition.
    assert 5.0 < res.time_below_h < 6.0


# ---------- controller_excursions ----------


def test_excursions_constant_setpoint_no_excursions() -> None:
    t = np.linspace(0, 10, 11)
    measured = np.full_like(t, 7.0)
    res = controller_excursions(t, measured, setpoint=7.0, tolerance=0.1)
    assert res.n_excursions == 0
    assert res.total_time_out_h == 0.0


def test_excursions_one_window() -> None:
    t = np.array([0, 1, 2, 3, 4, 5, 6, 7])
    measured = np.array([7.0, 7.0, 7.5, 7.5, 7.5, 7.0, 7.0, 7.0])  # excursion at 2-4h
    res = controller_excursions(t, measured, setpoint=7.0, tolerance=0.1)
    assert res.n_excursions == 1
    assert res.excursions[0][0] == 2.0
    assert res.excursions[0][1] == 4.0
    assert math.isclose(res.excursions[0][2], 0.5)


def test_excursions_open_ended_at_run_end() -> None:
    t = np.array([0, 1, 2, 3, 4])
    measured = np.array([7.0, 7.0, 7.5, 7.5, 7.5])  # excursion runs to end
    res = controller_excursions(t, measured, setpoint=7.0, tolerance=0.1)
    assert res.n_excursions == 1
    assert res.excursions[0][1] == 4.0  # end of last point


def test_excursions_raises_on_invalid_tolerance() -> None:
    with pytest.raises(ValueError):
        controller_excursions([0, 1], [7.0, 7.0], setpoint=7.0, tolerance=0)


# ---------- tip_speed ----------


def test_tip_speed_scalar() -> None:
    # D = 0.1 m, N = 600 RPM = 10 rev/s → v = π·0.1·10 ≈ 3.14 m/s
    v = tip_speed(600, 0.1)
    assert math.isclose(v, math.pi, rel_tol=1e-6)


def test_tip_speed_array() -> None:
    arr = tip_speed(np.array([300.0, 600.0, 900.0]), 0.1)
    assert isinstance(arr, np.ndarray)
    assert math.isclose(arr[1], math.pi, rel_tol=1e-6)


def test_tip_speed_raises_on_bad_diameter() -> None:
    with pytest.raises(ValueError):
        tip_speed(600, 0)


# ---------- power_per_volume ----------


def test_power_per_volume_scales_as_n_cubed() -> None:
    p1 = power_per_volume(
        300, impeller_diameter_m=0.1,
        fluid_density_kg_m3=1000.0,
        working_volume_l=10.0,
    )
    p2 = power_per_volume(
        600, impeller_diameter_m=0.1,
        fluid_density_kg_m3=1000.0,
        working_volume_l=10.0,
    )
    # Doubling RPM should octuple P/V (N³ scaling).
    assert math.isclose(p2 / p1, 8.0, rel_tol=1e-6)


def test_power_per_volume_array_shape_matches_input() -> None:
    arr = power_per_volume(
        np.array([300.0, 600.0]),
        impeller_diameter_m=0.1,
        fluid_density_kg_m3=1000.0,
        working_volume_l=10.0,
    )
    assert isinstance(arr, np.ndarray)
    assert arr.shape == (2,)


def test_power_per_volume_raises_on_bad_inputs() -> None:
    with pytest.raises(ValueError):
        power_per_volume(300, impeller_diameter_m=0, fluid_density_kg_m3=1000, working_volume_l=10)
    with pytest.raises(ValueError):
        power_per_volume(300, impeller_diameter_m=0.1, fluid_density_kg_m3=0, working_volume_l=10)
    with pytest.raises(ValueError):
        power_per_volume(300, impeller_diameter_m=0.1, fluid_density_kg_m3=1000, working_volume_l=0)
