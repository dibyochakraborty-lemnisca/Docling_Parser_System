"""Data-derived search box: let the agent decide the search space from the data
instead of a hardcoded config.

The agent can legitimately infer the EVIDENCE envelope — the range of operating
conditions actually present in the training data, where the fitted model is
trustworthy. It CANNOT infer true physical limits (substrate solubility, pump
minimums, safety) from a CSV. So the search box is:

    box = (data envelope ± margin) ∩ physical_box

`margin` allows controlled exploration just past what's been observed (the point
of optimization is to beat the best-seen condition), while the intersection with
an optional physical box keeps it inside real hard limits. This stops the
optimizer from recommending conditions in a no-data region (e.g. a dilution far
below anything ever run), which is where model extrapolation is least reliable.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from fermdocs_optimize.schema import KNOB_NAMES, Box


def _reconstruct_knobs(train_df: pd.DataFrame) -> dict[str, list[float]]:
    """Recover each batch's operating knobs from its initial state + volume slope,
    matching how the mechanistic model reconstructs conditions."""
    out: dict[str, list[float]] = {k: [] for k in KNOB_NAMES}
    for _, g in train_df.groupby("batch"):
        g = g.sort_values("t")
        f = g.iloc[0]
        total = float(f["S"] + f["M"])
        v0 = float(f["V"])
        slope = float(np.polyfit(g["t"].to_numpy(float), g["V"].to_numpy(float), 1)[0])
        out["biomass"].append(float(f["X"]))
        out["total_sub"].append(total)
        out["malt_frac"].append(float(f["M"]) / total if total > 0 else 0.0)
        out["dilution"].append(max(slope / v0, 0.0) if v0 > 0 else 0.0)
    return out


def box_from_data(train_df: pd.DataFrame, *, margin: float = 0.0,
                  physical: Box | None = None) -> Box:
    """Derive the search box from the data envelope, expanded by `margin` (a
    fraction of each knob's observed range) and intersected with `physical` (an
    optional hard-limit Box the data can't know about).

    Floors: all knobs >= 0; malt_frac <= 1. Where the margin-expanded envelope
    would exceed a physical limit, the physical limit wins (intersection)."""
    knobs = _reconstruct_knobs(train_df)
    bounds: dict[str, tuple[float, float]] = {}
    for k in KNOB_NAMES:
        arr = np.asarray(knobs[k], dtype=float)
        lo, hi = float(arr.min()), float(arr.max())
        span = max(hi - lo, 1e-9)
        lb, ub = lo - margin * span, hi + margin * span
        lb = max(lb, 0.0)  # no negative biomass/substrate/dilution
        if k == "malt_frac":
            ub = min(ub, 1.0)
        if physical is not None:  # intersect with real hard limits
            plb, pub = getattr(physical, k)
            lb, ub = max(lb, plb), min(ub, pub)
        if lb > ub:  # degenerate (physical cap below data) — pin to the cap
            lb = ub
        bounds[k] = (lb, ub)
    return Box(**bounds)
