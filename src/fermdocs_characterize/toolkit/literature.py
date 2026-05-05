"""Tier C literature-assisted estimators.

Where Tiers A and B compute *what was measured*, Tier C compares
measurements against *what literature predicts*. The reference values
come from `fermdocs.domain.process_priors` — a YAML registry of
organism × process_family → variable bounds with citations.

Architectural rule: Tier C functions accept a ProcessPriors document
plus an organism name; they return a result that includes the prior
source string for audit. Unknown organisms emit a "data_gap" status
rather than raising, so the analyzer can keep going and emit a
data-gap finding for that metric_id instead of crashing the run.

Why no hardcoded constants here: hardcoding s_cerevisiae/e_coli/etc
makes adding a new organism a code change. Routing through priors
makes it a YAML edit. Plus dossiers can override per-run via the
existing kinetic_estimates fields without touching this code.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Literal

from fermdocs.domain.process_priors import (
    ProcessPriors,
    ResolvedPrior,
    resolve_priors,
)

LiteratureStatus = Literal["computed", "data_gap"]


@dataclass(frozen=True)
class LiteratureResult:
    """Common return shape for Tier C functions.

    `status="computed"` means the prior was found and the estimate is
    valid. `status="data_gap"` means we couldn't resolve the prior;
    `details` contains the missing-prior summary. `value` is None on
    data_gap.
    """

    metric_id: str
    status: LiteratureStatus
    value: float | None
    organism: str | None
    process_family: str | None
    source: str | None
    details: dict


def _first_prior(
    priors: ProcessPriors,
    *,
    organism: str,
    variable: str,
    process_family: str | None = None,
) -> ResolvedPrior | None:
    matches = resolve_priors(
        priors,
        organism=organism,
        process_family=process_family,
        variable=variable,
    )
    return matches[0] if matches else None


def _data_gap(metric_id: str, organism: str, missing_variable: str) -> LiteratureResult:
    return LiteratureResult(
        metric_id=metric_id,
        status="data_gap",
        value=None,
        organism=organism,
        process_family=None,
        source=None,
        details={
            "missing_prior": missing_variable,
            "reason": (
                f"no prior for organism={organism!r}, variable={missing_variable!r} "
                "in process_priors registry"
            ),
        },
    )


# -----------------------------------------------------------------------------
# C2 — Reference mu_max vs observed
# -----------------------------------------------------------------------------


def mu_max_reference_vs_observed(
    *,
    priors: ProcessPriors,
    organism: str,
    observed_mu_max: float,
    process_family: str | None = None,
    variable: str = "mu_x_max_per_h",
) -> LiteratureResult:
    """Z-score-like comparison of observed mu_max against the prior range.

    Returns the ratio observed / typical and the prior range so the
    analyzer can flag "observed mu_max=0.05 is 4x below the Verduyn
    typical 0.20" for a yeast bundle.
    """
    prior = _first_prior(
        priors, organism=organism, process_family=process_family, variable=variable
    )
    if prior is None:
        return _data_gap("C2", organism, variable)

    low, high = prior.range
    in_range = low <= observed_mu_max <= high
    ratio = observed_mu_max / prior.typical if prior.typical > 0 else float("nan")

    return LiteratureResult(
        metric_id="C2",
        status="computed",
        value=ratio,
        organism=prior.organism,
        process_family=prior.process_family,
        source=prior.source,
        details={
            "observed_mu_max": observed_mu_max,
            "typical": prior.typical,
            "range": [low, high],
            "in_range": in_range,
            "ratio_observed_to_typical": ratio,
        },
    )


# -----------------------------------------------------------------------------
# C3 — Henry's-law-derived O2 saturation (C*)
# -----------------------------------------------------------------------------


# Henry's-law constant for O2 in pure water at 1 atm air (mol O2 / L · atm).
# Schumpe 1982: H_O2(298 K) ≈ 1.3e-3 mol/L/atm; for fermentation the
# air partial pressure of O2 is 0.2095 atm.
_HENRY_O2_298K_MOL_PER_L_PER_ATM = 1.3e-3
_AIR_FRACTION_O2 = 0.2095
# Van't Hoff relation, simplified: H(T) ≈ H(298) · exp(-ΔH_sol/R · (1/T - 1/298))
# with ΔH_sol/R for O2 ≈ 1700 K (Sander 2015 review).
_VAN_T_HOFF_K = 1700.0


def saturation_o2_concentration(
    *,
    temperature_C: float,
    pressure_atm: float = 1.0,
    salinity_mol_per_l: float = 0.0,
    setschenow_constant: float = 0.0,
) -> LiteratureResult:
    """C* = H(T, P) · p_O2, with optional Setschenow salt correction.

    Returns mg O2 / L (the unit dossiers report DO saturation in).
    Pure-water dilute-broth case: salinity_mol_per_l=0 collapses the
    Setschenow term out. Salt-laden broths can pass an estimated ionic
    strength (mol/L) to apply the Setschenow correction.

    Returns status="computed" always — this is pure chemistry, no
    organism prior needed. Kept in literature.py because the audit
    trail (Schumpe / Sander / Setschenow citations) belongs here.
    """
    if temperature_C < -10 or temperature_C > 80:
        return LiteratureResult(
            metric_id="C3",
            status="data_gap",
            value=None,
            organism=None,
            process_family=None,
            source=None,
            details={"reason": f"temperature_C={temperature_C} outside reasonable range"},
        )

    t_kelvin = 273.15 + temperature_C
    h_t = _HENRY_O2_298K_MOL_PER_L_PER_ATM * math.exp(
        -_VAN_T_HOFF_K * (1.0 / t_kelvin - 1.0 / 298.15)
    )
    p_o2 = pressure_atm * _AIR_FRACTION_O2
    c_star_mol_per_l = h_t * p_o2

    # Setschenow: log10(C*_pure / C*_salt) = K_S · I  →  C*_salt = C*_pure · 10^(-K_S · I)
    if salinity_mol_per_l > 0 and setschenow_constant > 0:
        c_star_mol_per_l *= 10 ** (-setschenow_constant * salinity_mol_per_l)

    c_star_mg_per_l = c_star_mol_per_l * 32_000.0  # 32 g/mol O2 → 32_000 mg/mol

    return LiteratureResult(
        metric_id="C3",
        status="computed",
        value=c_star_mg_per_l,
        organism=None,
        process_family=None,
        source="Schumpe 1982 (Henry's law); Sander 2015 (van't Hoff coefficient)",
        details={
            "temperature_C": temperature_C,
            "pressure_atm": pressure_atm,
            "henry_constant_mol_per_l_per_atm": h_t,
            "p_o2_atm": p_o2,
            "c_star_mg_per_l": c_star_mg_per_l,
            "setschenow_applied": salinity_mol_per_l > 0 and setschenow_constant > 0,
        },
    )


# -----------------------------------------------------------------------------
# C4 — Van't Riet kLa
# -----------------------------------------------------------------------------


def vant_riet_kla(
    *,
    p_per_v_w_m3: float,
    superficial_gas_velocity_m_s: float,
    alpha: float = 0.026,
    beta: float = 0.4,
    gamma: float = 0.5,
) -> LiteratureResult:
    """Van't Riet 1979 correlation: kLa = α · (P/V)^β · vs^γ (1/s).

    Defaults α=0.026, β=0.4, γ=0.5 are for coalescing aqueous systems
    (low-electrolyte broths). For non-coalescing electrolyte broths
    α=0.002, β=0.7, γ=0.2 (Van't Riet 1979 Table II); pass explicitly.
    """
    if p_per_v_w_m3 <= 0:
        return LiteratureResult(
            metric_id="C4",
            status="data_gap",
            value=None,
            organism=None,
            process_family=None,
            source=None,
            details={"reason": "P/V must be > 0"},
        )
    if superficial_gas_velocity_m_s <= 0:
        return LiteratureResult(
            metric_id="C4",
            status="data_gap",
            value=None,
            organism=None,
            process_family=None,
            source=None,
            details={"reason": "superficial gas velocity must be > 0"},
        )

    kla_per_s = alpha * (p_per_v_w_m3**beta) * (superficial_gas_velocity_m_s**gamma)

    return LiteratureResult(
        metric_id="C4",
        status="computed",
        value=kla_per_s,
        organism=None,
        process_family=None,
        source="Van't Riet 1979 Table II coalescing aqueous correlation",
        details={
            "p_per_v_w_m3": p_per_v_w_m3,
            "superficial_gas_velocity_m_s": superficial_gas_velocity_m_s,
            "alpha": alpha,
            "beta": beta,
            "gamma": gamma,
            "kla_per_s": kla_per_s,
            "kla_per_h": kla_per_s * 3600.0,
        },
    )


# -----------------------------------------------------------------------------
# C5 — qs from Verduyn yields (specific glucose uptake reference)
# -----------------------------------------------------------------------------


def qs_from_verduyn_yields(
    *,
    priors: ProcessPriors,
    organism: str,
    observed_qs: float | None = None,
    process_family: str | None = None,
    variable: str = "qs_glucose_g_per_g_per_h",
) -> LiteratureResult:
    """Compare observed qs (g substrate / g cells / h) to the Verduyn-yield
    reference range from priors.

    When observed_qs is None, returns the prior typical/range without a
    comparison — useful when the analyzer wants to surface "expected qs"
    for a bundle that doesn't yet have biomass-normalized substrate
    uptake.
    """
    prior = _first_prior(
        priors, organism=organism, process_family=process_family, variable=variable
    )
    if prior is None:
        return _data_gap("C5", organism, variable)

    low, high = prior.range
    in_range = (
        low <= observed_qs <= high if observed_qs is not None else None
    )
    ratio = (
        observed_qs / prior.typical
        if observed_qs is not None and prior.typical > 0
        else None
    )

    return LiteratureResult(
        metric_id="C5",
        status="computed",
        value=ratio if ratio is not None else prior.typical,
        organism=prior.organism,
        process_family=prior.process_family,
        source=prior.source,
        details={
            "observed_qs": observed_qs,
            "typical": prior.typical,
            "range": [low, high],
            "in_range": in_range,
            "ratio_observed_to_typical": ratio,
        },
    )


# -----------------------------------------------------------------------------
# C9 — Oxygen demand vs supply ratio
# -----------------------------------------------------------------------------


def oxygen_demand_vs_supply(
    *,
    our_mol_per_l_per_h: float,
    kla_per_h: float,
    do_saturation_mg_per_l: float,
    measured_do_mg_per_l: float,
) -> LiteratureResult:
    """Ratio of measured OUR to maximum possible OTR at current DO.

    OTR_max = kLa · (C* - DO)        [mg O2 / L / h, after unit-aligning]
    ratio   = OUR / OTR_max

    ratio < 1: the system has spare oxygen-transfer headroom.
    ratio ≈ 1: O2 transfer is the rate limit; cells are likely O2-limited.
    ratio > 1: numerically impossible at steady state — flags either an
               OUR overestimate or a kLa underestimate.

    Inputs are unit-aligned: caller must convert OUR to mol/L/h and
    kLa to 1/h before calling. Returns the ratio plus the OTR_max it
    derived so the audit trail is complete.
    """
    if kla_per_h <= 0:
        return LiteratureResult(
            metric_id="C9",
            status="data_gap",
            value=None,
            organism=None,
            process_family=None,
            source=None,
            details={"reason": "kla_per_h must be > 0"},
        )
    if do_saturation_mg_per_l <= 0:
        return LiteratureResult(
            metric_id="C9",
            status="data_gap",
            value=None,
            organism=None,
            process_family=None,
            source=None,
            details={"reason": "do_saturation must be > 0"},
        )

    # Convert OUR mol/L/h → mg O2 / L / h: × 32_000 mg/mol.
    our_mg_per_l_per_h = our_mol_per_l_per_h * 32_000.0
    driving_force = max(do_saturation_mg_per_l - measured_do_mg_per_l, 0.0)
    otr_max = kla_per_h * driving_force

    ratio = our_mg_per_l_per_h / otr_max if otr_max > 0 else float("inf")

    return LiteratureResult(
        metric_id="C9",
        status="computed",
        value=ratio,
        organism=None,
        process_family=None,
        source="OTR ≤ kLa·(C*-DO) mass-transfer envelope (textbook)",
        details={
            "our_mg_per_l_per_h": our_mg_per_l_per_h,
            "otr_max_mg_per_l_per_h": otr_max,
            "driving_force_mg_per_l": driving_force,
            "ratio_demand_over_supply": ratio,
            "o2_limited_signal": bool(ratio > 0.85),
        },
    )


# -----------------------------------------------------------------------------
# C10 — Overflow threshold (Crabtree / acetate switch)
# -----------------------------------------------------------------------------


def overflow_threshold(
    *,
    priors: ProcessPriors,
    organism: str,
    observed_qs: float,
    process_family: str | None = None,
    overflow_qs_variable: str = "qs_glucose_g_per_g_per_h",
    overflow_marker_variable: str = "ethanol_g_l",
) -> LiteratureResult:
    """Flag overflow metabolism when observed qs exceeds the critical
    qs reference (top of prior range) AND a corresponding marker prior
    exists.

    The "critical qs" from priors is an organism-specific threshold
    (Sonnleitner 1986 for yeast, ~0.30 g/g/h for E. coli, etc).
    Observed qs > critical means overflow is thermodynamically allowed;
    the marker prior (e.g. ethanol_g_l for yeast) names the byproduct
    to look for.
    """
    qs_prior = _first_prior(
        priors,
        organism=organism,
        process_family=process_family,
        variable=overflow_qs_variable,
    )
    if qs_prior is None:
        return _data_gap("C10", organism, overflow_qs_variable)

    critical_qs = qs_prior.range[1]  # top of range = critical threshold
    overflow_signal = observed_qs > critical_qs

    marker_prior = _first_prior(
        priors,
        organism=organism,
        process_family=process_family,
        variable=overflow_marker_variable,
    )
    marker_typical = marker_prior.typical if marker_prior else None
    marker_source = marker_prior.source if marker_prior else None

    return LiteratureResult(
        metric_id="C10",
        status="computed",
        value=critical_qs,
        organism=qs_prior.organism,
        process_family=qs_prior.process_family,
        source=qs_prior.source,
        details={
            "observed_qs": observed_qs,
            "critical_qs": critical_qs,
            "overflow_signal": overflow_signal,
            "overflow_marker_variable": overflow_marker_variable,
            "marker_typical": marker_typical,
            "marker_source": marker_source,
        },
    )
