"""Per-metric adapters for the deterministic catalog runner.

Plan ref: plans/2026-05-07-characterize-determinism.md commit 1.

One adapter per catalog entry. Each:
  - pulls the right Trajectory rows for (run_id, variable)
  - aligns time grids if needed
  - calls the toolkit_fn
  - returns a flat statistics dict (or None on input precondition fail)

Adapters NEVER fabricate data. They return None to signal 'inputs not
present or insufficient'; the runner converts that into a `data_gap`
Finding. They raise on real toolkit errors; the runner converts that
into a tool-error data_gap with the exception message.

Scope: this commit registers adapters for the metrics the IndPenSim
feedback loop demands answers on (A8, A9, A10, A14, B6, B10, B16).
The rest fall through to the LLM analyzer (warned at runner pre-flight).
More adapters land in commit 2 (product KPIs) and follow-up work.
"""

from __future__ import annotations

import numpy as np

from fermdocs_characterize.agents.catalog_runner import (
    _BundleView,
    _register_adapter,
)
from fermdocs_characterize.agents.metric_catalog import BIOMASS_PROXIES, DO_PROXIES
from fermdocs.domain.process_families import (
    UNKNOWN_FAMILY_NAME,
    ProcessFamilyConfig,
    lookup_family,
)
from fermdocs_characterize.toolkit._stats import central_tendency
from fermdocs_characterize.toolkit.balances import (
    compute_byproduct_yield,
    compute_carbon_balance_closure,
    compute_rq,
)
from fermdocs_characterize.toolkit.kinetics import (
    compute_mu,
    doubling_time,
    phasewise_mu,
    segment_growth_phases,
)
from fermdocs_characterize.toolkit.operational import compute_do_margin
from fermdocs_characterize.toolkit.products import (
    compute_final_titer,
    compute_integral_productivity,
    compute_peak_titer,
    compute_precursor_utilization,
    compute_titer_decline,
)


def _resolve_variable(
    bundle: _BundleView, run_id: str, primary: str, proxies: tuple[str, ...] = ()
) -> tuple[str, list[float], list[float]] | None:
    """Find the (variable_name, time_grid, values) for the first present
    candidate. Returns None if none of (primary + proxies) are in the
    bundle for this run.

    Filters out None values from the trajectory; aligned time_grid.
    Caller is responsible for `min_points` checks.
    """
    for candidate in (primary, *proxies):
        traj = bundle.trajectory(run_id, candidate)
        if traj is None:
            continue
        time_h: list[float] = []
        values: list[float] = []
        for t, v in zip(traj.time_grid, traj.values):
            if v is None:
                continue
            time_h.append(t)
            values.append(v)
        if not time_h:
            return None
        return candidate, time_h, values
    return None


def _aligned_pair(
    bundle: _BundleView, run_id: str, var_a: str, var_b: str
) -> tuple[list[float], list[float], list[float]] | None:
    """Pull two variables on the same run, return the (time_h, a, b)
    triplet on the inner-join time grid. None when either variable is
    missing or the join is empty.
    """
    ta = bundle.trajectory(run_id, var_a)
    tb = bundle.trajectory(run_id, var_b)
    if ta is None or tb is None:
        return None
    # Trajectories share a uniform time grid if they came from the same
    # ingestion path; defensive check.
    if ta.time_grid != tb.time_grid:
        # Inner join by time; rare path.
        common = sorted(set(ta.time_grid) & set(tb.time_grid))
        if not common:
            return None
        a_map = dict(zip(ta.time_grid, ta.values))
        b_map = dict(zip(tb.time_grid, tb.values))
        time_h: list[float] = []
        a_vals: list[float] = []
        b_vals: list[float] = []
        for t in common:
            av = a_map.get(t)
            bv = b_map.get(t)
            if av is None or bv is None:
                continue
            time_h.append(t)
            a_vals.append(av)
            b_vals.append(bv)
        return (time_h, a_vals, b_vals) if time_h else None
    time_h = []
    a_vals = []
    b_vals = []
    for t, av, bv in zip(ta.time_grid, ta.values, tb.values):
        if av is None or bv is None:
            continue
        time_h.append(t)
        a_vals.append(av)
        b_vals.append(bv)
    return (time_h, a_vals, b_vals) if time_h else None


# ---------------------------------------------------------------------------
# Tier A — kinetics
# ---------------------------------------------------------------------------


@_register_adapter("A8")
def _adapter_a8(bundle: _BundleView, run_id: str | None) -> dict | None:
    """A8: specific growth rate μ(t)."""
    if run_id is None:
        return None
    resolved = _resolve_variable(bundle, run_id, "biomass_g_l", BIOMASS_PROXIES)
    if resolved is None:
        return None
    var, time_h, values = resolved
    if len(time_h) < 5:
        return None
    result = compute_mu(time_h, values)
    return {
        "mu_max": result.mu_max,
        "t_mu_max_h": result.t_mu_max_h,
        "n_observations": len(time_h),
        "_variables_used": [var],
        "_summary": (
            f"μ_max = {result.mu_max:.4f} 1/h at t={result.t_mu_max_h:.1f}h"
            f" on {run_id} (variable={var})."
        ),
    }


@_register_adapter("A9")
def _adapter_a9(bundle: _BundleView, run_id: str | None) -> dict | None:
    """A9: doubling time during exponential window."""
    if run_id is None:
        return None
    resolved = _resolve_variable(bundle, run_id, "biomass_g_l", BIOMASS_PROXIES)
    if resolved is None:
        return None
    var, time_h, values = resolved
    if len(time_h) < 5:
        return None
    try:
        result = doubling_time(time_h, values)
    except ValueError:
        # No exponential growth detected — emit data_gap rather than
        # tool-error so synthesizer treats it as a real biological
        # observation (no exponential phase).
        return None
    return {
        "t_doubling_h": result.t_doubling_h,
        "mu_max": result.mu_max,
        "phase_start_h": result.phase_start_h,
        "phase_end_h": result.phase_end_h,
        "n_observations": len(time_h),
        "_variables_used": [var],
        "_summary": (
            f"doubling time = {result.t_doubling_h:.2f}h"
            f" (μ_max={result.mu_max:.4f} 1/h, phase {result.phase_start_h:.1f}"
            f"–{result.phase_end_h:.1f}h) on {run_id}."
        ),
    }


@_register_adapter("A10")
def _adapter_a10(bundle: _BundleView, run_id: str | None) -> dict | None:
    """A10: phase segmentation (lag/exp/stat/decline).

    Returns counts per phase as flat keys for downstream consumers.
    """
    if run_id is None:
        return None
    resolved = _resolve_variable(bundle, run_id, "biomass_g_l", BIOMASS_PROXIES)
    if resolved is None:
        return None
    var, time_h, values = resolved
    if len(time_h) < 8:
        return None
    df = segment_growth_phases(time_h, values)
    if df.empty:
        return None
    counts = df["phase"].value_counts().to_dict()
    return {
        "n_lag": int(counts.get("lag", 0)),
        "n_exp": int(counts.get("exp", 0)),
        "n_linear": int(counts.get("linear", 0)),
        "n_stationary": int(counts.get("stationary", 0)),
        "n_decline": int(counts.get("decline", 0)),
        "n_observations": len(time_h),
        "_variables_used": [var],
        "_summary": (
            f"phase counts on {run_id}: lag={counts.get('lag',0)},"
            f" exp={counts.get('exp',0)}, linear={counts.get('linear',0)},"
            f" stat={counts.get('stationary',0)}, decline={counts.get('decline',0)}."
        ),
    }


@_register_adapter("A11")
def _adapter_a11(bundle: _BundleView, run_id: str | None) -> dict | None:
    """A11: phase-wise mean μ."""
    if run_id is None:
        return None
    resolved = _resolve_variable(bundle, run_id, "biomass_g_l", BIOMASS_PROXIES)
    if resolved is None:
        return None
    var, time_h, values = resolved
    if len(time_h) < 8:
        return None
    df = phasewise_mu(time_h, values)
    if df.empty:
        return None
    out: dict = {
        "n_observations": len(time_h),
        "_variables_used": [var],
    }
    for _, row in df.iterrows():
        out[f"mean_mu_{row['phase']}"] = float(row["mean_mu"])
    out["_summary"] = (
        f"phase-wise μ on {run_id}: "
        + ", ".join(
            f"{row['phase']}={float(row['mean_mu']):.3f}" for _, row in df.iterrows()
        )
        + " 1/h."
    )
    return out


# ---------------------------------------------------------------------------
# Tier A — operational
# ---------------------------------------------------------------------------


@_register_adapter("A14")
def _adapter_a14(bundle: _BundleView, run_id: str | None) -> dict | None:
    """A14: DO margin against a threshold."""
    if run_id is None:
        return None
    resolved = _resolve_variable(
        bundle, run_id, "dissolved_o2_mg_l", DO_PROXIES
    )
    if resolved is None:
        return None
    var, time_h, values = resolved
    if len(time_h) < 3:
        return None
    result = compute_do_margin(time_h, values)
    return {
        "frac_below_threshold": result.frac_below,
        "min_do": result.min_do,
        "min_margin": result.min_margin,
        "time_below_h": result.time_below_h,
        "n_observations": len(time_h),
        "_variables_used": [var],
        "_summary": (
            f"DO margin on {run_id} (var={var}):"
            f" {result.frac_below*100:.1f}% of run below threshold,"
            f" min DO={result.min_do:.2f}, time below={result.time_below_h:.1f}h."
        ),
    }


# ---------------------------------------------------------------------------
# Tier B — balances
# ---------------------------------------------------------------------------


@_register_adapter("B6")
def _adapter_b6(bundle: _BundleView, run_id: str | None) -> dict | None:
    """B6: byproduct yield ΔP/ΔX.

    Tries known byproduct variable names: ethanol_g_l, acetate_g_l,
    lactate_g_l. Reports the first one present. PAA is intentionally
    NOT here — it's a precursor, not a byproduct (the IndPenSim
    polarity bug). Commit 2 ships P5 for precursor-as-input metrics.
    """
    if run_id is None:
        return None
    biomass_resolved = _resolve_variable(bundle, run_id, "biomass_g_l")
    if biomass_resolved is None:
        return None
    biomass_var, _, _ = biomass_resolved

    BYPRODUCTS = ("ethanol_g_l", "acetate_g_l", "lactate_g_l")
    available_byproduct: str | None = None
    for candidate in BYPRODUCTS:
        if bundle.trajectory(run_id, candidate) is not None:
            available_byproduct = candidate
            break
    if available_byproduct is None:
        return None

    pair = _aligned_pair(bundle, run_id, biomass_var, available_byproduct)
    if pair is None:
        return None
    time_h, biomass_vals, byproduct_vals = pair
    if len(time_h) < 2:
        return None
    result = compute_byproduct_yield(
        time_h, biomass_vals, byproduct_vals,
        byproduct_name=available_byproduct,
    )
    return {
        "byproduct": result.byproduct,
        "delta_p": result.delta_p,
        "delta_x": result.delta_x,
        "yield_g_per_g": result.yield_g_per_g,
        "n_observations": len(time_h),
        "_variables_used": [biomass_var, available_byproduct],
        "_summary": (
            f"{available_byproduct} yield on {run_id}:"
            f" Y={result.yield_g_per_g:.3f} g/g"
            f" (ΔP={result.delta_p:.2f}, ΔX={result.delta_x:.2f})."
        ),
    }


@_register_adapter("B10")
def _adapter_b10(bundle: _BundleView, run_id: str | None) -> dict | None:
    """B10: respiratory quotient + overflow flag. The mean-vs-median
    split addressed in commit 4 lives here; for now we emit just the
    mean + frac_over_threshold."""
    if run_id is None:
        return None
    pair = _aligned_pair(
        bundle, run_id, "our_mmol_per_l_per_h", "cer_mmol_per_l_per_h"
    )
    if pair is None:
        return None
    time_h, our, cer = pair
    if len(time_h) < 3:
        return None
    try:
        result = compute_rq(time_h, our, cer)
    except ValueError:
        return None
    # Commit 4 (robust stats): mean RQ over-reports overflow when there
    # are transient spikes (IndPenSim case: mean 1.21, median 0.98).
    # Compute the pointwise RQ time-series ourselves and report
    # central_tendency stats so the synthesizer can prefer median when
    # the signal is skewed.
    pointwise = []
    for o, c in zip(our, cer):
        if o > 0 and c is not None:
            pointwise.append(c / o)
    ct = central_tendency(pointwise) if len(pointwise) >= 2 else None
    out = {
        "mean_rq": result.rq_mean,
        "max_rq": result.rq_max,
        "min_rq": result.rq_min,
        "frac_over_overflow_threshold": result.frac_over_overflow_threshold,
        "overflow_flag": bool(result.overflow_flag),
        "n_observations": len(time_h),
        "_variables_used": ["our_mmol_per_l_per_h", "cer_mmol_per_l_per_h"],
    }
    if ct is not None:
        out["median_rq"] = ct.median
        out["p25_rq"] = ct.p25
        out["p75_rq"] = ct.p75
        out["iqr_rq"] = ct.iqr
        out["recommended_summary"] = ct.recommended_summary
        # Synthesizer reads recommended_summary; surface the actual
        # 'preferred' value as a top-level key for the prose summary.
        preferred = ct.median if ct.recommended_summary == "median" else ct.mean
        out["_summary"] = (
            f"RQ on {run_id}: "
            f"{ct.recommended_summary}={preferred:.2f}"
            f" (mean={ct.mean:.2f}, median={ct.median:.2f},"
            f" IQR={ct.iqr:.2f}),"
            f" {result.frac_over_overflow_threshold*100:.1f}% over 1.1"
            f" → overflow={'yes' if result.overflow_flag else 'no'}."
        )
    else:
        out["_summary"] = (
            f"RQ on {run_id}: mean={result.rq_mean:.2f},"
            f" max={result.rq_max:.2f},"
            f" {result.frac_over_overflow_threshold*100:.1f}% over 1.1"
            f" → overflow={'yes' if result.overflow_flag else 'no'}."
        )
    return out


# ---------------------------------------------------------------------------
# Tier P — product KPIs (process-family routed)
# ---------------------------------------------------------------------------
#
# Plan ref: plans/2026-05-07-characterize-determinism.md commit 2.
#
# These adapters resolve the product/precursor variable through
# `lookup_family()` at runtime. When the family is `unknown` (no
# routing) or the routed variable isn't present in the bundle, the
# adapter returns a special config-mismatch sentinel that the runner
# converts into ONE [CONFIG_MISMATCH] data_gap (NOT five per-metric
# data_gaps — A3 fix from plan-eng-review).


_CONFIG_MISMATCH_SENTINEL = "_CONFIG_MISMATCH_"


def _resolve_family(bundle: _BundleView) -> ProcessFamilyConfig:
    return lookup_family(bundle.process_family)


def _config_mismatch(reason: str) -> dict:
    """Return a stats dict that the runner recognizes as 'this should
    be a CONFIG_MISMATCH data_gap, not a tool-error or precondition
    data_gap'. The runner uses the `_config_mismatch_reason` key to
    surface the helpful message instead of a generic 'precondition not
    met'.

    Returning None would also work (becomes a generic precondition
    data_gap), but [CONFIG_MISMATCH] is more diagnostic for the user
    debugging YAML config typos.
    """
    return {
        "_config_mismatch": True,
        "_config_mismatch_reason": reason,
        "_summary": f"[CONFIG_MISMATCH] {reason}",
    }


@_register_adapter("P1")
def _adapter_p1(bundle: _BundleView, run_id: str | None) -> dict | None:
    """P1: final product titer."""
    if run_id is None:
        return None
    family = _resolve_family(bundle)
    if family.product_variable is None:
        # Unknown family or family without product → not applicable.
        # Fall through to None (precondition not met) rather than
        # CONFIG_MISMATCH because there's nothing to be mismatched.
        return None
    traj = bundle.trajectory(run_id, family.product_variable)
    if traj is None:
        avail = sorted(bundle.variables_for(run_id))
        return _config_mismatch(
            f"process_families.yaml routes {family.name} to"
            f" product_variable={family.product_variable!r}, but this"
            f" bundle has no such trajectory on {run_id}."
            f" Available variables: {avail or '(none)'}."
        )
    time_h = [t for t, v in zip(traj.time_grid, traj.values) if v is not None]
    values = [v for v in traj.values if v is not None]
    if len(time_h) < 2:
        return None
    result = compute_final_titer(time_h, values)
    return {
        "final_titer_g_l": result.final_titer_g_l,
        "t_final_h": result.t_final_h,
        "product_variable": family.product_variable,
        "n_observations": result.n_points,
        "_variables_used": [family.product_variable],
        "_summary": (
            f"final {family.product_variable} on {run_id}:"
            f" {result.final_titer_g_l:.2f} g/L at {result.t_final_h:.1f}h."
        ),
    }


@_register_adapter("P2")
def _adapter_p2(bundle: _BundleView, run_id: str | None) -> dict | None:
    """P2: peak product titer."""
    if run_id is None:
        return None
    family = _resolve_family(bundle)
    if family.product_variable is None:
        return None
    traj = bundle.trajectory(run_id, family.product_variable)
    if traj is None:
        avail = sorted(bundle.variables_for(run_id))
        return _config_mismatch(
            f"process_families.yaml routes {family.name} to"
            f" product_variable={family.product_variable!r}, but this"
            f" bundle has no such trajectory on {run_id}."
            f" Available variables: {avail or '(none)'}."
        )
    time_h = [t for t, v in zip(traj.time_grid, traj.values) if v is not None]
    values = [v for v in traj.values if v is not None]
    if len(time_h) < 2:
        return None
    result = compute_peak_titer(time_h, values)
    return {
        "peak_titer_g_l": result.peak_titer_g_l,
        "t_peak_h": result.t_peak_h,
        "product_variable": family.product_variable,
        "n_observations": result.n_points,
        "_variables_used": [family.product_variable],
        "_summary": (
            f"peak {family.product_variable} on {run_id}:"
            f" {result.peak_titer_g_l:.2f} g/L at {result.t_peak_h:.1f}h."
        ),
    }


@_register_adapter("P3")
def _adapter_p3(bundle: _BundleView, run_id: str | None) -> dict | None:
    """P3: titer decline after peak (lysis / hydrolysis flag)."""
    if run_id is None:
        return None
    family = _resolve_family(bundle)
    if family.product_variable is None:
        return None
    traj = bundle.trajectory(run_id, family.product_variable)
    if traj is None:
        # P1/P2 already emit the CONFIG_MISMATCH; P3 follows suit so the
        # user sees the same diagnostic on all three lines.
        avail = sorted(bundle.variables_for(run_id))
        return _config_mismatch(
            f"process_families.yaml routes {family.name} to"
            f" product_variable={family.product_variable!r}, but this"
            f" bundle has no such trajectory on {run_id}."
            f" Available variables: {avail or '(none)'}."
        )
    time_h = [t for t, v in zip(traj.time_grid, traj.values) if v is not None]
    values = [v for v in traj.values if v is not None]
    if len(time_h) < 2:
        return None
    result = compute_titer_decline(time_h, values)
    return {
        "decline_fraction": result.decline_fraction,
        "peak_titer_g_l": result.peak_titer_g_l,
        "final_titer_g_l": result.final_titer_g_l,
        "t_peak_h": result.t_peak_h,
        "t_final_h": result.t_final_h,
        "is_declining": bool(result.is_declining),
        "product_variable": family.product_variable,
        "n_observations": len(time_h),
        "_variables_used": [family.product_variable],
        "_summary": (
            f"{family.product_variable} decline on {run_id}:"
            f" peak={result.peak_titer_g_l:.2f} → final={result.final_titer_g_l:.2f}"
            f" g/L ({result.decline_fraction*100:.1f}% drop;"
            f" {'flagged' if result.is_declining else 'held'})."
        ),
    }


@_register_adapter("P4")
def _adapter_p4(bundle: _BundleView, run_id: str | None) -> dict | None:
    """P4: integral productivity."""
    if run_id is None:
        return None
    family = _resolve_family(bundle)
    if family.product_variable is None:
        return None
    traj = bundle.trajectory(run_id, family.product_variable)
    if traj is None:
        avail = sorted(bundle.variables_for(run_id))
        return _config_mismatch(
            f"process_families.yaml routes {family.name} to"
            f" product_variable={family.product_variable!r}, but this"
            f" bundle has no such trajectory on {run_id}."
            f" Available variables: {avail or '(none)'}."
        )
    time_h = [t for t, v in zip(traj.time_grid, traj.values) if v is not None]
    values = [v for v in traj.values if v is not None]
    if len(time_h) < 2:
        return None
    result = compute_integral_productivity(time_h, values)
    return {
        "integral_g_l_h": result.integral_g_l_h,
        "mean_productivity_g_l_per_h": result.mean_productivity_g_l_per_h,
        "duration_h": result.duration_h,
        "product_variable": family.product_variable,
        "n_observations": result.n_points,
        "_variables_used": [family.product_variable],
        "_summary": (
            f"{family.product_variable} integral productivity on {run_id}:"
            f" mean={result.mean_productivity_g_l_per_h:.2f} g/L/h"
            f" over {result.duration_h:.1f}h."
        ),
    }


@_register_adapter("P5")
def _adapter_p5(bundle: _BundleView, run_id: str | None) -> dict | None:
    """P5: precursor utilization. Iterates the family's precursor list
    and reports the FIRST precursor present in the bundle for this run.
    Multiple precursors aren't combined here — each gets its own future
    metric if needed; for v1 we surface one."""
    if run_id is None:
        return None
    family = _resolve_family(bundle)
    if not family.precursor_variables:
        # Family doesn't declare a precursor — P5 is not applicable.
        return None
    available = bundle.variables_for(run_id)
    chosen: str | None = None
    for precursor in family.precursor_variables:
        if precursor in available:
            chosen = precursor
            break
    if chosen is None:
        avail = sorted(available)
        return _config_mismatch(
            f"process_families.yaml routes {family.name} to"
            f" precursor_variables={list(family.precursor_variables)},"
            f" but this bundle has none of them on {run_id}."
            f" Available variables: {avail or '(none)'}."
        )
    traj = bundle.trajectory(run_id, chosen)
    if traj is None:
        return None  # defense-in-depth; can't reach
    time_h = [t for t, v in zip(traj.time_grid, traj.values) if v is not None]
    values = [v for v in traj.values if v is not None]
    if len(time_h) < 2:
        return None
    try:
        result = compute_precursor_utilization(
            time_h, values, precursor_variable=chosen
        )
    except ValueError:
        return None
    return {
        "utilization_fraction": result.utilization_fraction,
        "peak_value": result.peak_value,
        "final_value": result.final_value,
        "utilization_class": result.utilization_class,
        "precursor_variable": result.precursor_variable,
        "n_observations": len(time_h),
        "_variables_used": [chosen],
        "_summary": (
            f"{chosen} utilization on {run_id}:"
            f" {result.utilization_fraction*100:.1f}% consumed"
            f" (peak={result.peak_value:.0f} → final={result.final_value:.0f},"
            f" class={result.utilization_class})."
        ),
    }


# ---------------------------------------------------------------------------
# Tier B — carbon balance (continued)
# ---------------------------------------------------------------------------


@_register_adapter("B16")
def _adapter_b16(bundle: _BundleView, run_id: str | None) -> dict | None:
    """B16: carbon balance closure. Run-level (uses delta across whole run).

    Returns None when substrate or biomass deltas are not computable
    (need at least 2 valid points each).
    """
    if run_id is None:
        return None
    biomass = _resolve_variable(bundle, run_id, "biomass_g_l")
    substrate = _resolve_variable(bundle, run_id, "substrate_g_l")
    if biomass is None or substrate is None:
        return None
    _, _, b_vals = biomass
    _, _, s_vals = substrate
    if len(b_vals) < 2 or len(s_vals) < 2:
        return None
    biomass_produced = float(np.nanmax(b_vals)) - float(np.nanmin(b_vals))
    substrate_consumed = float(np.nanmax(s_vals)) - float(np.nanmin(s_vals))
    if substrate_consumed <= 0:
        return None

    # Optional CO2 flux from CER if available.
    co2_evolved: float | None = None
    cer_traj = bundle.trajectory(run_id, "cer_mmol_per_l_per_h")
    if cer_traj is not None:
        cer_vals = [v for v in cer_traj.values if v is not None]
        if len(cer_vals) >= 2:
            # Trapezoidal integration; mmol/L/h × h → mmol/L → mol/L.
            valid = [
                (t, v) for t, v in zip(cer_traj.time_grid, cer_traj.values)
                if v is not None
            ]
            t_arr = np.array([t for t, _ in valid])
            c_arr = np.array([v for _, v in valid])
            co2_mmol_per_l = float(np.trapezoid(c_arr, t_arr))
            co2_evolved = co2_mmol_per_l / 1000.0

    try:
        result = compute_carbon_balance_closure(
            substrate_consumed_g_per_l=substrate_consumed,
            substrate_name="glucose",
            biomass_produced_g_per_l=biomass_produced,
            products_produced_g_per_l={},
            co2_evolved_mol_per_l=co2_evolved,
        )
    except ValueError:
        return None

    return {
        "closure": result.closure,
        "c_consumed_g": result.c_consumed_g,
        "c_in_biomass_g": result.c_in_biomass_g,
        "c_in_co2_g": result.c_in_co2_g,
        "n_observations": len(b_vals) + len(s_vals),
        "_variables_used": ["biomass_g_l", "substrate_g_l"]
        + (["cer_mmol_per_l_per_h"] if co2_evolved is not None else []),
        "_summary": (
            f"carbon balance on {run_id}: closure={result.closure*100:.1f}%"
            f" (consumed {result.c_consumed_g:.1f}g → biomass"
            f" {result.c_in_biomass_g:.1f}g + CO2 {result.c_in_co2_g:.1f}g)."
        ),
    }
