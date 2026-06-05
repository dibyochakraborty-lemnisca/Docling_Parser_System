"""In-process stub oracle for tests — no LABS, no subprocess.

Integrates the same kinetic ODE with a fixed "true" parameter set (different
from the model's fit defaults) so the loop can be exercised deterministically in
CI. Lets us test the full fit->propose->simulate->evaluate->converge loop and
the integrity invariant without installing LABS.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from fermdocs_optimize.models.mechanistic import (
    C_FEED_TOTAL,
    DEFAULT_KLA,
    DEFAULT_O2,
    _compute_kla,
    _rhs,
)
from fermdocs_optimize.schema import Candidate
from scipy.integrate import odeint

# A "ground-truth" param set the model must rediscover from data.
TRUE_PARAMS = np.array([0.30, 0.4, 160.0, 1.6, 2.5, 0.05, 0.01])


class StubSimulator:
    """Deterministic in-process oracle (no external deps)."""

    def __init__(self, true_params=TRUE_PARAMS, t_end=75.0, n=76, seed=0):
        self._p = np.asarray(true_params, float)
        self._t = np.linspace(0, t_end, n)
        self._o2, self._kla_h = dict(DEFAULT_O2), _compute_kla(DEFAULT_KLA)
        self._rng = np.random.default_rng(seed)

    def _run(self, c: Candidate, v0: float) -> np.ndarray:
        k = c.knobs()
        s0 = k["total_sub"] * (1 - k["malt_frac"]); m0 = k["total_sub"] * k["malt_frac"]
        y0 = [k["biomass"], s0, 0.0, m0, self._o2["O2_sat"], v0]
        return odeint(_rhs, y0, self._t,
                      args=(self._p, k["dilution"] * v0,
                            C_FEED_TOTAL * (1 - k["malt_frac"]),
                            C_FEED_TOTAL * k["malt_frac"], self._o2, self._kla_h))

    def simulate(self, candidates: list[Candidate], *, v0: float) -> pd.DataFrame:
        rows = []
        for i, c in enumerate(candidates):
            sol = self._run(c, v0)
            for j, t in enumerate(self._t):
                rows.append({"batch": i, "t": t, "X": sol[j, 0], "S": sol[j, 1],
                             "P": sol[j, 2], "M": sol[j, 3], "O2": sol[j, 4],
                             "V": sol[j, 5]})
        return pd.DataFrame(rows)
