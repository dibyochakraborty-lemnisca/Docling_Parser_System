"""Global-optimization proposer: maximize model-predicted peak titer.

Uses scipy `differential_evolution` (derivative-free, global) over the 4-knob
box. Returns the optimum plus a few diverse near-optimal points so the oracle
samples more than a single spot each round (helps the model learn the response
surface faster).
"""
from __future__ import annotations

import numpy as np
from scipy.optimize import differential_evolution

from fermdocs_optimize.models.base import PredictiveModel
from fermdocs_optimize.schema import KNOB_NAMES, Box, Candidate


class OptimizeProposer:
    def __init__(self, *, seed: int = 11, maxiter: int = 40, popsize: int = 15):
        self._seed, self._maxiter, self._popsize = seed, maxiter, popsize

    def propose(self, model: PredictiveModel, box: Box, *, k: int, v0: float) -> list[Candidate]:
        bounds = box.as_list()

        def neg_titer(x: np.ndarray) -> float:
            cand = Candidate(**dict(zip(KNOB_NAMES, x)))
            return -model.predict_peak_titer(cand, v0=v0)

        res = differential_evolution(
            neg_titer, bounds, seed=self._seed, maxiter=self._maxiter,
            popsize=self._popsize, tol=1e-4, polish=True)
        best = Candidate(**dict(zip(KNOB_NAMES, res.x)),
                         predicted_peak_titer=float(-res.fun))
        out = [best]
        # add k-1 diverse perturbations around the optimum, clamped to the box
        if k > 1:
            rng = np.random.default_rng(self._seed)
            for _ in range(k - 1):
                x = []
                for name, (lb, ub) in zip(KNOB_NAMES, bounds):
                    span = ub - lb
                    v = getattr(best, name) + rng.normal(0, 0.1 * span)
                    x.append(float(np.clip(v, lb, ub)))
                cand = Candidate(**dict(zip(KNOB_NAMES, x)))
                cand.predicted_peak_titer = model.predict_peak_titer(cand, v0=v0)
                out.append(cand)
        return out
