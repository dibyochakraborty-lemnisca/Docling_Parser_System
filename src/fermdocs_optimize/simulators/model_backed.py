"""A Simulator backed by the agent's own discovered model.

When there is NO process oracle (real lab data, no LABS), the equation the agent
discovered IS the only thing we can search. This adapter wraps a fitted model
(anything exposing `predict_P_trajectory`) behind the `Simulator` interface, so
the exact same global search (`oracle_global_search` — the LHS sweep + pattern
search) runs on the discovered equation instead of a real oracle.

The result is a MODEL-predicted optimum, not ground truth. It is a setpoint to
verify in the lab — the lab is the (slow) oracle that closes the loop.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from fermdocs_optimize.schema import Candidate

_T_END, _N_T = 75.0, 76


class ModelSimulator:
    """Adapt a fitted PredictiveModel (with `predict_P_trajectory`) to a Simulator.

    Returns a long dataframe (batch, t, P) — enough for the peak-titer search.
    The model must already be fit; this never refits.
    """

    def __init__(self, model, *, t_end: float = _T_END, n_t: int = _N_T):
        if not hasattr(model, "predict_P_trajectory"):
            raise TypeError("ModelSimulator needs a model with predict_P_trajectory")
        self._model = model
        self._t = np.linspace(0, t_end, n_t)

    def simulate(self, candidates: list[Candidate], *, v0: float) -> pd.DataFrame:
        rows = []
        for i, c in enumerate(candidates):
            p = self._model.predict_P_trajectory(c, v0=v0, t_end=float(self._t[-1]),
                                                  n=len(self._t))
            for t, pv in zip(self._t, p):
                rows.append({"batch": i, "t": float(t), "P": float(pv)})
        return pd.DataFrame(rows)
