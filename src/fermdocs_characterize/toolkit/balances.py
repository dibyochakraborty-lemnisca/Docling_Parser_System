"""Tier B mass-balance toolkit.

These metrics need MEASURED inputs the trajectory must contain — RQ
needs OUR + CER, yields need substrate + product trajectories, carbon
balance needs every C-bearing species. When inputs are missing the
analyzer is expected to emit a data-gap finding instead of calling
these functions.

Carbon-content table is molecular (glucose has 40% C by mass regardless
of which organism is eating it), so it lives here as a frozen dict.
Organism-specific yield references (Verduyn coefficients etc) belong
in `literature.py` and are PR 3 work.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

# Molecular carbon-mass fraction (g C per g of species). Sources:
#   glucose C6H12O6 = 72/180   = 0.4000
#   ethanol C2H6O   = 24/46    = 0.5217
#   acetate (acetic acid) C2H4O2 = 24/60 = 0.4000
#   lactate (lactic acid) C3H6O3 = 36/90 = 0.4000
#   pyruvate C3H4O3 = 36/88   = 0.4091
#   succinate (succinic acid) C4H6O4 = 48/118 = 0.4068
#   glycerol C3H8O3 = 36/92   = 0.3913
#   biomass — typical CH1.8O0.5N0.2 cell formula → 24/24.6 = 0.488 (Roels 1980)
# Adding species: must be molecular, must cite the formula in the comment.
CARBON_MASS_FRACTION: dict[str, float] = {
    "glucose": 0.4000,
    "ethanol": 0.5217,
    "acetate": 0.4000,
    "lactate": 0.4000,
    "pyruvate": 0.4091,
    "succinate": 0.4068,
    "glycerol": 0.3913,
    "biomass": 0.488,
}

CO2_C_MASS_FRACTION = 12.0 / 44.0  # 0.2727 — for converting mol CO2 → g C


# -----------------------------------------------------------------------------
# B10 — Respiratory quotient
# -----------------------------------------------------------------------------


@dataclass(frozen=True)
class RQResult:
    """Per-run RQ summary.

    `rq_mean`: time-weighted mean RQ over the analyzed window.
    `rq_max`: max instantaneous RQ.
    `frac_over_overflow_threshold`: fraction of the time series with
        RQ > overflow_threshold (default 1.1, the standard signal for
        respiro-fermentative metabolism).
    `overflow_flag`: True if frac_over_overflow_threshold > 0.2.
    """

    rq_mean: float
    rq_max: float
    rq_min: float
    frac_over_overflow_threshold: float
    overflow_flag: bool
    n_points: int


def compute_rq(
    time_h: pd.Series | np.ndarray | list[float],
    our: pd.Series | np.ndarray | list[float],
    cer: pd.Series | np.ndarray | list[float],
    *,
    overflow_threshold: float = 1.1,
    overflow_fraction_floor: float = 0.2,
) -> RQResult:
    """RQ(t) = CER(t) / OUR(t).

    OUR + CER are expected in the same molar units (mol/L/h or mmol/L/h —
    cancels out). Negative or zero OUR values are dropped (no respiration =
    undefined RQ).

    Raises ValueError when fewer than 3 valid (OUR > 0, finite) points
    remain.
    """
    t = np.asarray(time_h, dtype=float)
    o = np.asarray(our, dtype=float)
    c = np.asarray(cer, dtype=float)
    if not (t.shape == o.shape == c.shape):
        raise ValueError(
            f"shape mismatch: time_h={t.shape}, our={o.shape}, cer={c.shape}"
        )

    mask = np.isfinite(t) & np.isfinite(o) & np.isfinite(c) & (o > 0)
    t = t[mask]
    o = o[mask]
    c = c[mask]
    if len(t) < 3:
        raise ValueError(f"compute_rq needs >= 3 valid points, got {len(t)}")

    order = np.argsort(t)
    t = t[order]
    o = o[order]
    c = c[order]

    rq = c / o
    finite = np.isfinite(rq)
    rq_f = rq[finite]
    if rq_f.size < 3:
        raise ValueError("compute_rq: fewer than 3 finite RQ points after division")

    over = rq_f > overflow_threshold
    frac_over = float(over.sum()) / float(rq_f.size)

    return RQResult(
        rq_mean=float(np.mean(rq_f)),
        rq_max=float(np.max(rq_f)),
        rq_min=float(np.min(rq_f)),
        frac_over_overflow_threshold=frac_over,
        overflow_flag=bool(frac_over > overflow_fraction_floor),
        n_points=int(rq_f.size),
    )


# -----------------------------------------------------------------------------
# B6 — Byproduct yield per biomass increase
# -----------------------------------------------------------------------------


@dataclass(frozen=True)
class ByproductYieldResult:
    """Yield of byproduct per biomass produced.

    Defined as ΔP / ΔX between the first and last sampled time. If
    biomass increases <= 0 over the run, yield is undefined.
    """

    byproduct: str
    delta_p: float
    delta_x: float
    yield_g_per_g: float
    n_points: int


def compute_byproduct_yield(
    time_h: pd.Series | np.ndarray | list[float],
    biomass_g_l: pd.Series | np.ndarray | list[float],
    byproduct_g_l: pd.Series | np.ndarray | list[float],
    *,
    byproduct_name: str = "byproduct",
) -> ByproductYieldResult:
    """ΔP / ΔX across the run.

    Expects biomass and byproduct on the same time grid. Aligns by
    nearest-time inner-join; drops points where either is NaN. Uses
    the first and last surviving points to compute the delta — yields
    are run-level, not instantaneous.
    """
    t = np.asarray(time_h, dtype=float)
    x = np.asarray(biomass_g_l, dtype=float)
    p = np.asarray(byproduct_g_l, dtype=float)
    if not (t.shape == x.shape == p.shape):
        raise ValueError("time_h / biomass / byproduct shape mismatch")

    mask = np.isfinite(t) & np.isfinite(x) & np.isfinite(p)
    t = t[mask]
    x = x[mask]
    p = p[mask]
    if len(t) < 2:
        raise ValueError(f"compute_byproduct_yield needs >= 2 valid points, got {len(t)}")

    order = np.argsort(t)
    t = t[order]
    x = x[order]
    p = p[order]

    delta_x = float(x[-1] - x[0])
    delta_p = float(p[-1] - p[0])
    if delta_x <= 0:
        raise ValueError(
            f"compute_byproduct_yield: biomass did not grow (Δx={delta_x:.3f} g/L)"
        )

    return ByproductYieldResult(
        byproduct=byproduct_name,
        delta_p=delta_p,
        delta_x=delta_x,
        yield_g_per_g=delta_p / delta_x,
        n_points=int(len(t)),
    )


# -----------------------------------------------------------------------------
# B16 — Carbon balance closure
# -----------------------------------------------------------------------------


@dataclass(frozen=True)
class CarbonBalanceResult:
    """Closure = (C in biomass + C in products + C in CO2 evolved) / C consumed.

    Ideal value is 1.0; published fermentation balances close to 0.90-1.05.
    Below 0.85 or above 1.10 flags a measurement gap or an unaccounted
    species (e.g. unmeasured byproduct carbon). The result captures the
    components so downstream interpretation can attribute the gap.
    """

    closure: float
    c_consumed_g: float
    c_in_biomass_g: float
    c_in_products_g: float
    c_in_co2_g: float
    components: dict[str, float]


def compute_carbon_balance_closure(
    *,
    substrate_consumed_g_per_l: float,
    substrate_name: str,
    biomass_produced_g_per_l: float,
    products_produced_g_per_l: dict[str, float],
    co2_evolved_mol_per_l: float | None,
) -> CarbonBalanceResult:
    """Run-level carbon balance.

    Inputs are deltas across the whole run, in g/L (or mol/L for CO2).
    All species names must be keys in CARBON_MASS_FRACTION; unknown
    species raise ValueError (silent ignoring would hide carbon).

    `co2_evolved_mol_per_l` may be None when CER wasn't measured —
    closure is then computed without it and the result will read low,
    which is the correct signal that the balance is incomplete.
    """
    if substrate_name not in CARBON_MASS_FRACTION:
        raise ValueError(
            f"unknown substrate '{substrate_name}'; "
            f"add it to CARBON_MASS_FRACTION with its molecular C-fraction"
        )
    for product_name in products_produced_g_per_l:
        if product_name not in CARBON_MASS_FRACTION:
            raise ValueError(
                f"unknown product '{product_name}'; "
                f"add it to CARBON_MASS_FRACTION with its molecular C-fraction"
            )

    c_consumed = substrate_consumed_g_per_l * CARBON_MASS_FRACTION[substrate_name]
    if c_consumed <= 0:
        raise ValueError(f"substrate consumption must be > 0, got {c_consumed:.4f} g C/L")

    c_in_biomass = biomass_produced_g_per_l * CARBON_MASS_FRACTION["biomass"]
    c_in_products = sum(
        amt * CARBON_MASS_FRACTION[name]
        for name, amt in products_produced_g_per_l.items()
    )
    c_in_co2 = (
        0.0
        if co2_evolved_mol_per_l is None
        else co2_evolved_mol_per_l * 12.0  # 12 g C per mol CO2
    )

    components = {
        "biomass": c_in_biomass,
        "co2": c_in_co2,
        **{f"product_{n}": amt * CARBON_MASS_FRACTION[n]
           for n, amt in products_produced_g_per_l.items()},
    }

    return CarbonBalanceResult(
        closure=float((c_in_biomass + c_in_products + c_in_co2) / c_consumed),
        c_consumed_g=float(c_consumed),
        c_in_biomass_g=float(c_in_biomass),
        c_in_products_g=float(c_in_products),
        c_in_co2_g=float(c_in_co2),
        components=components,
    )
