"""scipy.optimize global search over the knob box.

A drop-in alternative to `oracle_search.oracle_global_search` (the hand-rolled
Latin-hypercube + Hooke-Jeeves pattern search). Same boundary: it takes a
`Simulator` (the oracle, or a `ModelSimulator` wrapping the discovered equation)
and returns an `OracleSearchReport`. The difference is the optimizer:

* method="de"     — scipy `differential_evolution`, global + bounded + derivative
                    free. Run VECTORIZED, so the whole population is evaluated in
                    ONE batched simulator call per generation (maps onto LABS /
                    ModelSimulator batching — thousands of trials, few calls).
* method="direct" — scipy `direct` (the DIRECT algorithm): global, bounded,
                    deterministic, no seed sensitivity. One eval at a time.

This is the right tool when the objective is CHEAP (searching the discovered
equation) — you can afford a thorough global method. On the real oracle, where
each eval is costly, prefer few-eval strategies; this is for the equation.
"""
from __future__ import annotations

import logging

import numpy as np
from scipy.optimize import differential_evolution, direct

from fermdocs_optimize import evaluate as ev
from fermdocs_optimize.schema import KNOB_NAMES, Box, Candidate, OracleSearchReport
from fermdocs_optimize.simulators.base import Simulator

log = logging.getLogger(__name__)

_BOUNDARY_FRAC = 0.02


def scipy_global_search(
    simulator: Simulator,
    box: Box,
    *,
    objective_species: str = "P",
    v0: float = 10.0,
    method: str = "de",
    maxiter: int = 30,
    popsize: int = 15,
    seed: int = 11,
    warm_start: list[Candidate] | None = None,
) -> OracleSearchReport:
    """Maximize peak titer over the box with scipy.optimize. Returns the same
    report shape as the LHS search so callers can swap freely."""
    bounds = [tuple(map(float, b)) for b in box.as_list()]
    lo = np.array([b[0] for b in bounds]); hi = np.array([b[1] for b in bounds])
    spans = np.maximum(hi - lo, 1e-12)
    n_evals = {"n": 0}

    def _neg_peaks(cands: list[Candidate]) -> np.ndarray:
        n_evals["n"] += len(cands)
        df = simulator.simulate(cands, v0=v0)
        peaks = ev._ordered_peaks(ev.peak_titer_per_batch(df, objective_species), len(cands))
        return -np.asarray(peaks, float)

    def objective(x: np.ndarray):
        x = np.asarray(x, float)
        if x.ndim == 1:  # single point (DIRECT, or DE's polish step)
            return float(_neg_peaks([Candidate(**dict(zip(KNOB_NAMES, x)))])[0])
        # vectorized: x is (d, S) — one column per population member
        cands = [Candidate(**dict(zip(KNOB_NAMES, x[:, j]))) for j in range(x.shape[1])]
        return _neg_peaks(cands)

    if method == "de":
        x0 = None
        if warm_start:
            x0 = np.array([getattr(warm_start[0], k) for k in KNOB_NAMES], float)
        res = differential_evolution(
            objective, bounds, vectorized=True, maxiter=maxiter, popsize=popsize,
            seed=seed, tol=1e-4, polish=True, init="latinhypercube", x0=x0)
        best_x, best_titer = res.x, float(-res.fun)
    elif method == "direct":
        res = direct(objective, bounds, maxiter=maxiter, locally_biased=False)
        best_x, best_titer = res.x, float(-res.fun)
    else:
        raise ValueError(f"unknown method {method!r} (expected 'de' or 'direct')")

    best = Candidate(**dict(zip(KNOB_NAMES, best_x)))
    on_boundary: dict[str, str] = {}
    for i, k in enumerate(KNOB_NAMES):
        v = getattr(best, k)
        if abs(v - lo[i]) <= _BOUNDARY_FRAC * spans[i]:
            on_boundary[k] = "lower"
        elif abs(v - hi[i]) <= _BOUNDARY_FRAC * spans[i]:
            on_boundary[k] = "upper"

    log.info("scipy %s search: %d evals, max %.2f g/L, boundary=%s",
             method, n_evals["n"], best_titer, on_boundary)
    return OracleSearchReport(
        best_candidate=best, best_titer=round(best_titer, 3),
        n_oracle_evals=n_evals["n"], n_lhs=0, knobs_on_boundary=on_boundary)
