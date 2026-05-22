"""Synthetic-data tests for the Tier B balances toolkit."""

from __future__ import annotations

import math

import numpy as np
import pytest

from fermdocs_characterize.toolkit.balances import (
    CARBON_MASS_FRACTION,
    compute_byproduct_yield,
    compute_carbon_balance_closure,
    compute_rq,
)


# ---------- compute_rq ----------


def test_rq_constant_unit_means_no_overflow() -> None:
    t = np.linspace(0, 10, 11)
    our = np.full_like(t, 1.0)
    cer = np.full_like(t, 1.0)
    res = compute_rq(t, our, cer)
    assert math.isclose(res.rq_mean, 1.0)
    assert math.isclose(res.rq_max, 1.0)
    assert res.frac_over_overflow_threshold == 0.0
    assert res.overflow_flag is False


def test_rq_high_cer_flags_overflow() -> None:
    t = np.linspace(0, 10, 21)
    our = np.full_like(t, 1.0)
    cer = np.full_like(t, 1.5)  # RQ=1.5 everywhere → 100% over threshold
    res = compute_rq(t, our, cer)
    assert math.isclose(res.rq_mean, 1.5)
    assert res.frac_over_overflow_threshold == 1.0
    assert res.overflow_flag is True


def test_rq_drops_zero_and_negative_our() -> None:
    t = [0, 1, 2, 3, 4, 5]
    our = [1.0, 0.0, -1.0, 1.0, 1.0, 1.0]  # 4 valid points
    cer = [1.0, 1.0, 1.0, 1.0, 1.0, 1.0]
    res = compute_rq(t, our, cer)
    assert res.n_points == 4


def test_rq_raises_on_too_few_points() -> None:
    with pytest.raises(ValueError):
        compute_rq([0, 1], [1.0, 1.0], [1.0, 1.0])


def test_rq_overflow_fraction_floor_respected() -> None:
    # Only 10% of points above threshold → flag stays False even though some overflow happens.
    t = np.arange(10)
    our = np.full(10, 1.0)
    cer = np.array([1.5] + [0.9] * 9)  # 1/10 above threshold
    res = compute_rq(t, our, cer, overflow_fraction_floor=0.2)
    assert res.frac_over_overflow_threshold == 0.1
    assert res.overflow_flag is False


# ---------- compute_byproduct_yield ----------


def test_byproduct_yield_simple() -> None:
    t = [0, 5, 10]
    biomass = [1.0, 5.0, 11.0]  # ΔX = 10
    ethanol = [0.0, 2.0, 4.0]   # ΔP = 4
    res = compute_byproduct_yield(t, biomass, ethanol, byproduct_name="ethanol")
    assert res.byproduct == "ethanol"
    assert math.isclose(res.delta_x, 10.0)
    assert math.isclose(res.delta_p, 4.0)
    assert math.isclose(res.yield_g_per_g, 0.4)


def test_byproduct_yield_raises_on_no_growth() -> None:
    with pytest.raises(ValueError):
        compute_byproduct_yield([0, 5, 10], [5.0, 5.0, 5.0], [0.0, 1.0, 2.0])


def test_byproduct_yield_handles_unsorted_input() -> None:
    t = [10, 0, 5]
    biomass = [11.0, 1.0, 5.0]
    ethanol = [4.0, 0.0, 2.0]
    res = compute_byproduct_yield(t, biomass, ethanol)
    assert math.isclose(res.yield_g_per_g, 0.4)


# ---------- compute_carbon_balance_closure ----------


def test_carbon_balance_closes_when_inputs_balanced() -> None:
    # Glucose 10 g/L consumed → 4.0 g C consumed.
    # Outputs that should close: 5 g/L biomass (2.44 g C), 2 g/L ethanol (1.043 g C),
    # plus enough CO2 to round it out: target = 4.0 - 2.44 - 1.043 = 0.517 g C
    # → 0.517 / 12 = 0.0431 mol CO2 / L.
    res = compute_carbon_balance_closure(
        substrate_consumed_g_per_l=10.0,
        substrate_name="glucose",
        biomass_produced_g_per_l=5.0,
        products_produced_g_per_l={"ethanol": 2.0},
        co2_evolved_mol_per_l=0.0431,
    )
    assert math.isclose(res.closure, 1.0, abs_tol=0.005)
    assert res.c_consumed_g == pytest.approx(4.0, abs=1e-3)


def test_carbon_balance_low_when_co2_unmeasured() -> None:
    # Same inputs as above but co2 is None → closure should drop noticeably.
    res = compute_carbon_balance_closure(
        substrate_consumed_g_per_l=10.0,
        substrate_name="glucose",
        biomass_produced_g_per_l=5.0,
        products_produced_g_per_l={"ethanol": 2.0},
        co2_evolved_mol_per_l=None,
    )
    assert res.closure < 0.95
    assert res.c_in_co2_g == 0.0


def test_carbon_balance_raises_unknown_substrate() -> None:
    with pytest.raises(ValueError):
        compute_carbon_balance_closure(
            substrate_consumed_g_per_l=10.0,
            substrate_name="unobtainium",
            biomass_produced_g_per_l=5.0,
            products_produced_g_per_l={},
            co2_evolved_mol_per_l=None,
        )


def test_carbon_balance_raises_unknown_product() -> None:
    with pytest.raises(ValueError):
        compute_carbon_balance_closure(
            substrate_consumed_g_per_l=10.0,
            substrate_name="glucose",
            biomass_produced_g_per_l=5.0,
            products_produced_g_per_l={"made_up_byproduct": 1.0},
            co2_evolved_mol_per_l=None,
        )


def test_carbon_mass_fraction_table_sanity() -> None:
    # Spot-check: glucose is 40% C by mass.
    assert math.isclose(CARBON_MASS_FRACTION["glucose"], 0.40, abs_tol=1e-3)
    # Ethanol is more carbon-rich per gram than glucose.
    assert CARBON_MASS_FRACTION["ethanol"] > CARBON_MASS_FRACTION["glucose"]
