"""The PROPOSE step — search the knob box for high-titer operating points.

A proposer turns the current model + feasible box into candidate batches to try
on the oracle next. Strategies (global optimize vs grid/LHS) are interchangeable
behind this Protocol.
"""
from __future__ import annotations

from typing import Protocol

from fermdocs_optimize.models.base import PredictiveModel
from fermdocs_optimize.schema import Box, Candidate


class Proposer(Protocol):
    def propose(
        self, model: PredictiveModel, box: Box, *, k: int, v0: float
    ) -> list[Candidate]:
        """Return up to `k` candidates that maximize model-predicted peak titer."""
        ...
