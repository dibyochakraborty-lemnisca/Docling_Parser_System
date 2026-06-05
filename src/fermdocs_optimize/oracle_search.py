"""Oracle-direct global search — use the simulator (ground truth) to find the
TRUE within-box maximum, not the surrogate.

The model-guided loop is cheap, but its surrogate is peak-blind (held-out peak
R² ~ 0), so it can plateau on a local point. This module spends oracle calls
directly:

  1. DENSE SWEEP   — a Latin-hypercube sample over the whole box, every point
                     evaluated on the oracle. (LABS batches the entire sweep into
                     ONE subprocess call, so this is one oracle round, not N.)
  2. REFINE        — pattern search (Hooke-Jeeves style) around the sweep's best
                     point, each step evaluated on the oracle, climbing to the
                     true local max. Each step is one batched oracle call.
  3. BOUNDARY FLAG — report which knobs sit on the box edge; a corner solution
                     means the box is the binding constraint and the real optimum
                     may lie OUTSIDE the limits.

Because every evaluation is the oracle, the returned maximum is ground truth, not
a model guess. This is the honest way to "fully utilise the simulator."
"""
from __future__ import annotations

import logging

import numpy as np
from scipy.stats import qmc

from fermdocs_optimize import evaluate as ev
from fermdocs_optimize.schema import KNOB_NAMES, Box, Candidate, OracleSearchReport
from fermdocs_optimize.simulators.base import Simulator

log = logging.getLogger(__name__)

_BOUNDARY_FRAC = 0.02  # within 2% of a bound counts as "on the boundary"


def _candidates(points: np.ndarray) -> list[Candidate]:
    return [Candidate(**dict(zip(KNOB_NAMES, row))) for row in points]


def _best_on_oracle(simulator: Simulator, cands: list[Candidate], *, v0: float,
                    target: str) -> tuple[Candidate, float]:
    """Evaluate all candidates on the oracle (one batched call) and return the
    (candidate, peak titer) with the highest ground-truth titer."""
    sim_df = simulator.simulate(cands, v0=v0)
    peaks = ev._ordered_peaks(ev.peak_titer_per_batch(sim_df, target), len(cands))
    i = int(np.argmax(peaks))
    return cands[i], float(peaks[i])


def oracle_global_search(
    simulator: Simulator,
    box: Box,
    *,
    objective_species: str = "P",
    v0: float = 10.0,
    n_lhs: int = 200,
    refine_iters: int = 10,
    seed: int = 11,
    warm_start: list[Candidate] | None = None,
) -> OracleSearchReport:
    """Find the true within-box maximum titer by searching the oracle directly.

    `warm_start` seeds the search with already-known good points (e.g. the
    model-guided loop's best), evaluated alongside the sweep so refinement begins
    from the best of {sweep, warm-start}. This guarantees the oracle search never
    does worse than the points handed to it."""
    bounds = np.array(box.as_list(), dtype=float)
    lo, hi = bounds[:, 0], bounds[:, 1]
    spans = np.maximum(hi - lo, 1e-12)

    # 1. DENSE SWEEP (+ warm starts) — one batched oracle call.
    sample = qmc.LatinHypercube(d=len(KNOB_NAMES), seed=seed).random(n_lhs)
    pts = qmc.scale(sample, lo, hi)
    cands = (list(warm_start) if warm_start else []) + _candidates(pts)
    best, best_titer = _best_on_oracle(simulator, cands, v0=v0, target=objective_species)
    n_evals = len(cands)

    # 2. REFINE — pattern search on the oracle around the sweep's best point.
    x = np.array([getattr(best, k) for k in KNOB_NAMES], dtype=float)
    step = 0.1 * spans
    for _ in range(refine_iters):
        trials = []
        for i in range(len(KNOB_NAMES)):
            for s in (+1.0, -1.0):
                y = x.copy()
                y[i] = float(np.clip(x[i] + s * step[i], lo[i], hi[i]))
                trials.append(y)
        cand, titer = _best_on_oracle(simulator, _candidates(np.array(trials)),
                                      v0=v0, target=objective_species)
        n_evals += len(trials)
        if titer > best_titer + 1e-9:
            best, best_titer = cand, titer
            x = np.array([getattr(best, k) for k in KNOB_NAMES], dtype=float)
        else:
            step *= 0.5  # contract around the current best
            if np.all(step < 1e-3 * spans):
                break

    # 3. BOUNDARY FLAG.
    on_boundary: dict[str, str] = {}
    for i, k in enumerate(KNOB_NAMES):
        v = getattr(best, k)
        if abs(v - lo[i]) <= _BOUNDARY_FRAC * spans[i]:
            on_boundary[k] = "lower"
        elif abs(v - hi[i]) <= _BOUNDARY_FRAC * spans[i]:
            on_boundary[k] = "upper"

    log.info("oracle search: %d evals, true max %.2f g/L, boundary=%s",
             n_evals, best_titer, on_boundary)
    return OracleSearchReport(
        best_candidate=best, best_titer=round(best_titer, 3),
        n_oracle_evals=n_evals, n_lhs=n_lhs, knobs_on_boundary=on_boundary,
    )
