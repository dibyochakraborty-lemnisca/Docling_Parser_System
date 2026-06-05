"""Grid / Latin-hypercube proposer (robust fallback / warm-start).

Scores an LHS sample of the box with the model and returns the top-k points.
Derivative-free and global without relying on the optimizer converging —
useful when the model surface is rough or as a warm-start for OptimizeProposer.
"""
from __future__ import annotations

import numpy as np
from scipy.stats import qmc

from fermdocs_optimize.models.base import PredictiveModel
from fermdocs_optimize.schema import KNOB_NAMES, Box, Candidate


class GridProposer:
    def __init__(self, *, n_samples: int = 256, seed: int = 11):
        self._n, self._seed = n_samples, seed

    def propose(self, model: PredictiveModel, box: Box, *, k: int, v0: float) -> list[Candidate]:
        bounds = np.array(box.as_list(), float)
        sample = qmc.LatinHypercube(d=4, seed=self._seed).random(self._n)
        pts = qmc.scale(sample, bounds[:, 0], bounds[:, 1])
        scored = []
        for x in pts:
            cand = Candidate(**dict(zip(KNOB_NAMES, x)))
            cand.predicted_peak_titer = model.predict_peak_titer(cand, v0=v0)
            scored.append(cand)
        scored.sort(key=lambda c: c.predicted_peak_titer or -1e18, reverse=True)
        return scored[:k]
