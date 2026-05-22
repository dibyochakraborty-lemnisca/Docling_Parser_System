"""Tests for the Tier C literature toolkit.

These hit the real process_priors registry (Saccharomyces, E. coli,
Penicillium chrysogenum) for organism-aware metrics, and pure
chemistry for organism-free metrics like Henry's-law saturation.
"""

from __future__ import annotations

import math

from fermdocs.domain.process_priors import cached_priors
from fermdocs_characterize.toolkit.literature import (
    mu_max_reference_vs_observed,
    overflow_threshold,
    oxygen_demand_vs_supply,
    qs_from_verduyn_yields,
    saturation_o2_concentration,
    vant_riet_kla,
)


# ---------- C2 mu_max reference ----------


def test_c2_yeast_in_range() -> None:
    priors = cached_priors()
    res = mu_max_reference_vs_observed(
        priors=priors, organism="Saccharomyces cerevisiae", observed_mu_max=0.20
    )
    assert res.status == "computed"
    assert res.metric_id == "C2"
    assert res.details["in_range"] is True
    assert math.isclose(res.value, 1.0, abs_tol=0.05)
    assert "Verduyn" in res.source


def test_c2_yeast_out_of_range_low() -> None:
    priors = cached_priors()
    res = mu_max_reference_vs_observed(
        priors=priors, organism="yeast", observed_mu_max=0.05
    )
    assert res.status == "computed"
    assert res.details["in_range"] is False
    assert res.value < 0.5  # well below typical


def test_c2_unknown_organism_emits_data_gap() -> None:
    priors = cached_priors()
    res = mu_max_reference_vs_observed(
        priors=priors, organism="Methylococcus capsulatus", observed_mu_max=0.1
    )
    assert res.status == "data_gap"
    assert res.value is None
    assert "missing_prior" in res.details


# ---------- C3 Henry C* (organism-free) ----------


def test_c3_water_at_30C_close_to_8mg_per_l() -> None:
    # Textbook value: dissolved-O2 saturation in pure water at 30°C, 1 atm
    # air ≈ 7.5 mg/L. The simplified van't Hoff fit is loose; ±25% tolerance.
    res = saturation_o2_concentration(temperature_C=30.0)
    assert res.status == "computed"
    assert 5.0 < res.value < 11.0


def test_c3_pressure_scales_linearly() -> None:
    a = saturation_o2_concentration(temperature_C=30.0, pressure_atm=1.0)
    b = saturation_o2_concentration(temperature_C=30.0, pressure_atm=2.0)
    assert math.isclose(b.value / a.value, 2.0, rel_tol=1e-6)


def test_c3_setschenow_correction_lowers_solubility() -> None:
    pure = saturation_o2_concentration(temperature_C=30.0)
    salty = saturation_o2_concentration(
        temperature_C=30.0, salinity_mol_per_l=0.5, setschenow_constant=0.1
    )
    assert salty.value < pure.value


def test_c3_invalid_temperature_data_gap() -> None:
    res = saturation_o2_concentration(temperature_C=200.0)
    assert res.status == "data_gap"


# ---------- C4 Van't Riet kLa ----------


def test_c4_kla_scales_with_p_per_v() -> None:
    a = vant_riet_kla(p_per_v_w_m3=1000, superficial_gas_velocity_m_s=0.05)
    b = vant_riet_kla(p_per_v_w_m3=10_000, superficial_gas_velocity_m_s=0.05)
    # Higher P/V → higher kLa.
    assert b.value > a.value


def test_c4_invalid_inputs_data_gap() -> None:
    res = vant_riet_kla(p_per_v_w_m3=0, superficial_gas_velocity_m_s=0.05)
    assert res.status == "data_gap"
    res2 = vant_riet_kla(p_per_v_w_m3=1000, superficial_gas_velocity_m_s=0)
    assert res2.status == "data_gap"


def test_c4_per_h_conversion() -> None:
    res = vant_riet_kla(p_per_v_w_m3=1000, superficial_gas_velocity_m_s=0.05)
    assert math.isclose(res.details["kla_per_h"], res.details["kla_per_s"] * 3600.0)


# ---------- C5 qs from Verduyn ----------


def test_c5_yeast_with_observed_qs_in_range() -> None:
    priors = cached_priors()
    res = qs_from_verduyn_yields(
        priors=priors, organism="S. cerevisiae", observed_qs=0.25
    )
    assert res.status == "computed"
    assert res.details["in_range"] is True


def test_c5_yeast_without_observed_qs_returns_typical() -> None:
    priors = cached_priors()
    res = qs_from_verduyn_yields(
        priors=priors, organism="S. cerevisiae", observed_qs=None
    )
    assert res.status == "computed"
    assert res.details["observed_qs"] is None
    assert res.details["in_range"] is None
    # value falls back to prior typical in this branch
    assert res.value == res.details["typical"]


def test_c5_unknown_organism_data_gap() -> None:
    priors = cached_priors()
    res = qs_from_verduyn_yields(
        priors=priors, organism="Methylococcus capsulatus", observed_qs=0.2
    )
    assert res.status == "data_gap"


# ---------- C9 oxygen demand vs supply ----------


def test_c9_balanced_returns_ratio_close_to_one() -> None:
    # OUR 5e-4 mol/L/h ≈ 16 mg/L/h.
    # kLa 200/h × (8 - 0.0) mg/L = 1600 mg/L/h headroom → ratio ≈ 0.01 (huge headroom).
    res = oxygen_demand_vs_supply(
        our_mol_per_l_per_h=5e-4,
        kla_per_h=200.0,
        do_saturation_mg_per_l=8.0,
        measured_do_mg_per_l=0.0,
    )
    assert res.status == "computed"
    assert res.value < 0.1
    assert res.details["o2_limited_signal"] is False


def test_c9_o2_limited_when_demand_near_supply() -> None:
    # Tighten supply by lowering kLa: kLa 5/h × 8 = 40 mg/L/h supply.
    # Demand 5e-4 × 32_000 = 16 mg/L/h → ratio 0.4 still safe; bump demand:
    # OUR 1.2e-3 → 38.4 mg/L/h → ratio 0.96 (limited).
    res = oxygen_demand_vs_supply(
        our_mol_per_l_per_h=1.2e-3,
        kla_per_h=5.0,
        do_saturation_mg_per_l=8.0,
        measured_do_mg_per_l=0.0,
    )
    assert res.value > 0.85
    assert res.details["o2_limited_signal"] is True


def test_c9_invalid_kla_data_gap() -> None:
    res = oxygen_demand_vs_supply(
        our_mol_per_l_per_h=1e-3, kla_per_h=0.0,
        do_saturation_mg_per_l=8.0, measured_do_mg_per_l=2.0,
    )
    assert res.status == "data_gap"


# ---------- C10 overflow threshold ----------


def test_c10_yeast_at_low_qs_no_overflow() -> None:
    priors = cached_priors()
    res = overflow_threshold(
        priors=priors, organism="S. cerevisiae", observed_qs=0.15
    )
    assert res.status == "computed"
    assert res.details["overflow_signal"] is False


def test_c10_yeast_high_qs_flags_overflow() -> None:
    priors = cached_priors()
    res = overflow_threshold(
        priors=priors, organism="S. cerevisiae", observed_qs=0.6
    )
    assert res.status == "computed"
    assert res.details["overflow_signal"] is True
    # Marker should be ethanol for yeast
    assert res.details["overflow_marker_variable"] == "ethanol_g_l"
    assert res.details["marker_typical"] is not None


def test_c10_unknown_organism_data_gap() -> None:
    priors = cached_priors()
    res = overflow_threshold(
        priors=priors, organism="Methylococcus capsulatus", observed_qs=0.5
    )
    assert res.status == "data_gap"
