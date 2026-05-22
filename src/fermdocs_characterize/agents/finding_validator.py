"""Physicality validator: catch non-physical values that pass schema.

Plan ref: plans/2026-05-07-characterize-determinism.md commit 3.

The IndPenSim feedback exposed a stack-wide gap: a finding claiming
'PAA yield 204.5 g/g' passed citation discipline, prompt invariants,
critic, and judge. Yields > 1 g/g are non-physical (you can't make
200x as much byproduct as biomass). Nothing in the existing pipeline
rejected it.

This module is the defense-in-depth layer. Every Finding from the
catalog runner AND the LLM trajectory analyzer is piped through
`validate_finding` before it lands in the bundle. Out-of-bound values
are converted to data_gap with reason 'computed value violated
physical bounds: <field>=<value>'. The original computed value is
preserved in `statistics['raw_invalid']` for audit; consumers see
data_gap with a clear reason rather than poisoned evidence.

Design:
  - Only validates well-known unit semantics. Yields, percentages,
    fractions, ratios with clear bounds. Anything ambiguous passes
    through unchanged.
  - Field-name based dispatch. The catalog metric_id determines which
    fields to check (B6 → yield_g_per_g; A14 → frac_below_threshold;
    B16 → closure; etc.).
  - Idempotent: running twice on the same finding produces the same
    result. Already-converted data_gaps pass through untouched.
"""

from __future__ import annotations

import logging
from typing import Any

from fermdocs_characterize.schema import Finding, FindingType, Severity

_log = logging.getLogger(__name__)


# Bounds dictionary: maps statistics-field name to (lower, upper).
# A value outside [lower, upper] is non-physical and triggers conversion.
# None means 'no bound on that side' (e.g. integrals can be any positive
# number, but never negative).
#
# Adding a new bound: pick a statistics field name from a catalog entry's
# output_columns. If the metric is well-known (yield, percentage, ratio
# with a domain ceiling), add it here. Otherwise leave it alone—over-
# rejection is worse than under-rejection.
_PHYSICAL_BOUNDS: dict[str, tuple[float | None, float | None]] = {
    # Yields: g per g, can't exceed 1.0 (mass-balance limit).
    "yield_g_per_g": (0.0, 1.0),
    "utilization_fraction": (0.0, 1.0),
    # Fractions of time / observations: in [0, 1].
    "frac_below_threshold": (0.0, 1.0),
    "frac_over_overflow_threshold": (0.0, 1.0),
    "decline_fraction": (0.0, 1.0),
    "between_frac": (0.0, 1.0),
    # Carbon balance closure: typical 0.85-1.10. Below 0 or above 1.5
    # is a measurement error; consider unconverted.
    "closure": (0.0, 1.5),
    # Specific growth rate mu: organism-dependent ceiling, but no real
    # organism in liquid culture exceeds 2.0 1/h (E. coli max ~0.7
    # under ideal conditions). Anything above 2 is a units/sign bug.
    "mu_max": (-0.5, 2.0),
    "mean_mu": (-0.5, 2.0),
    # Doubling time: must be positive. Negative or zero = math bug.
    "t_doubling_h": (0.0, None),
    # Times: hours from run start, never negative.
    "t_mu_max_h": (0.0, None),
    "t_peak_h": (0.0, None),
    "t_final_h": (0.0, None),
    "phase_start_h": (0.0, None),
    "phase_end_h": (0.0, None),
    "duration_h": (0.0, None),
    # Concentrations: never negative (NaN handled separately).
    "final_titer_g_l": (0.0, None),
    "peak_titer_g_l": (0.0, None),
    "min_do": (0.0, None),
    # RQ: physically [0, ~3.0]. Above 3 means more CO2 evolved than O2
    # consumed by 3x — thermodynamically suspicious unless heavy
    # anaerobic byproduct production. Below 0 = sign bug.
    "mean_rq": (0.0, 3.0),
    "max_rq": (0.0, 5.0),  # Allow slightly higher for transients.
    "min_rq": (0.0, 3.0),
    # Productivity: trapezoidal integral over time, can be 0 but never
    # negative.
    "integral_g_l_h": (0.0, None),
    "mean_productivity_g_l_per_h": (0.0, None),
    # Intracellular yield: mg product per g dry cell weight. Carotenoids
    # in S. cerevisiae usually 0.1–50 mg/g; oleaginous yeasts hit ~200 mg/g
    # for lipids; intracellular protein in well-engineered strains can
    # approach ~300 mg/g. 500 mg/g is a generous ceiling that catches
    # units mismatches (someone reporting g/g as mg/g → 1000× over) and
    # garbage data while admitting every realistic process.
    "final_yield_mg_per_g_dcw": (0.0, 500.0),
    "peak_yield_mg_per_g_dcw": (0.0, 500.0),
    "yield_decline_after_peak": (0.0, 1.0),
    "final_volumetric_yield_mg_per_l": (0.0, None),
}


def validate_finding(finding: Finding) -> Finding:
    """Inspect a Finding's statistics for non-physical values. Returns
    the original finding unchanged when all checks pass; returns a new
    Finding converted to data_gap with reason naming the violation
    when any field is out of bounds.

    Idempotent: data_gap input passes through unchanged (no
    re-validation needed; data_gaps don't carry computed values).

    NaN values in numeric fields are also rejected (NaN ∈ nothing).
    """
    stats = finding.statistics or {}
    # Already a data_gap or config_mismatch → nothing computed to validate.
    pattern_kind = stats.get("pattern_kind")
    if pattern_kind in ("data_gap", "config_mismatch"):
        return finding

    violations: list[str] = []
    raw_invalid: dict[str, Any] = {}

    for field, value in stats.items():
        if field not in _PHYSICAL_BOUNDS:
            continue
        if not isinstance(value, (int, float)):
            continue
        # NaN handling: x != x is a NaN check that doesn't import math.
        if value != value:  # noqa: PLR0124 — deliberate NaN check
            violations.append(f"{field}=NaN")
            raw_invalid[field] = "NaN"
            continue
        lower, upper = _PHYSICAL_BOUNDS[field]
        if lower is not None and value < lower:
            violations.append(f"{field}={value:g} < {lower}")
            raw_invalid[field] = value
            continue
        if upper is not None and value > upper:
            violations.append(f"{field}={value:g} > {upper}")
            raw_invalid[field] = value
            continue

    if not violations:
        return finding

    metric_id = stats.get("metric_id", "unknown")
    reason = (
        f"computed value violated physical bounds: {'; '.join(violations)}"
    )
    _log.warning(
        "finding_validator: %s on %s rejected (%s)",
        metric_id, finding.run_ids or "cross-run", reason,
    )

    # Build a new Finding marked as data_gap. Keep the same finding_id,
    # type, evidence chain. Drop the (now-known-bogus) computed values
    # from the top-level summary but preserve them under raw_invalid for
    # audit traceability.
    new_stats = dict(stats)
    new_stats["pattern_kind"] = "data_gap"
    new_stats["reason"] = reason
    new_stats["raw_invalid"] = raw_invalid
    return finding.model_copy(update={
        "severity": Severity.INFO,
        "summary": (
            f"{metric_id}: {reason}."
            f" Original computed values flagged invalid; see"
            f" statistics.raw_invalid for audit."
        ),
        "statistics": new_stats,
    })
