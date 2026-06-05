"""The ground-truth ORACLE boundary.

Optimization needs something to evaluate proposed operating points against.
A `Simulator` takes candidate batches and returns their trajectories. LABS is
the first concrete one; other process families plug in their own simulator (or
a "no simulator -> design-of-experiments / refuse" strategy).

This is the only seam through which the optimizer touches the simulator. The
agent's own model never imports the simulator's internals — keeping the model
an independent approximation and the trust boundary clean.
"""
from __future__ import annotations

from typing import Protocol

import pandas as pd

from fermdocs_optimize.schema import Candidate


class Simulator(Protocol):
    def simulate(self, candidates: list[Candidate], *, v0: float) -> pd.DataFrame:
        """Run the proposed batches. Returns a long dataframe with at least
        columns `batch, t, X, S, P, M, V` (one row per batch-time)."""
        ...
