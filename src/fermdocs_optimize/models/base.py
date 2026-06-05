"""The agent's own predictive model — the cheap surrogate that guides proposals.

A model is fit on observed batch data only and predicts the peak titer for a
candidate operating point. The Protocol lets the loop swap a mechanistic fit for
a neural surrogate without changing the orchestration (DIP).
"""
from __future__ import annotations

from typing import Protocol

import pandas as pd

from fermdocs_optimize.schema import Candidate


class PredictiveModel(Protocol):
    """Fit on data, predict peak titer for proposed knobs.

    Implementations MUST fit only on the provided observation dataframe; they
    must never read the simulator's true parameters (the answer key).
    """

    def fit(self, observations: pd.DataFrame) -> dict[str, float]:
        """Fit on long/wide batch observations. Returns per-species R^2 (X,S,P,M)."""
        ...

    def predict_peak_titer(self, candidate: Candidate, *, v0: float) -> float:
        """Predict the peak of the target species for one operating point."""
        ...

    @property
    def fitted_params(self) -> dict[str, float]:
        """The model's own fitted parameters (for reporting/plausibility)."""
        ...
