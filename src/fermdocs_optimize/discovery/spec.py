"""Typed contracts for equation discovery.

A `ModelSpec` is what the agent proposes: the *structure* of a kinetic model —
which parameters exist, the intermediate rate laws, and the per-state ODEs — as
math strings. The discovery loop compiles it, fits its parameters to data, and
scores it against the oracle. Over rounds the agent rewrites the spec to shrink
the gap to the oracle.
"""
from __future__ import annotations

from pydantic import BaseModel, Field


class ParamSpec(BaseModel):
    """One fittable parameter the agent introduces."""

    init: float
    lb: float
    ub: float


class ModelSpec(BaseModel):
    """An agent-proposed kinetic model structure (compiled by `expr.compile_spec`)."""

    name: str = "candidate"
    params: dict[str, ParamSpec]
    aux: dict[str, str] = Field(default_factory=dict)   # mu, growth terms, ...
    odes: dict[str, str]                                 # dX..dM required; O2,V default
    notes: str = ""                                      # the agent's reasoning

    def param_names(self) -> list[str]:
        return list(self.params.keys())


class DiscoveryRound(BaseModel):
    """One iteration: a proposed spec, how it fit data, how wrong vs the oracle."""

    round_index: int
    spec: ModelSpec
    fitted_params: dict[str, float]
    r2_by_species: dict[str, float]                # fit quality on DATA (X,S,P,M)
    oracle_peak_rmse: float                        # g/L, agent-model vs oracle peak P
    oracle_peak_r2: float                          # graded against between-condition variance
    oracle_traj_r2: float                          # trajectory P R^2 vs oracle at probes
    score: float                                   # combined objective (higher = better)
    compile_ok: bool = True
    error: str = ""


class DiscoveryReport(BaseModel):
    """Outcome of the discovery loop: the best structure the agent found."""

    best_spec: ModelSpec | None
    best_round: int
    rounds: list[DiscoveryRound]
    exit_reason: str                               # converged | max_rounds | no_compile
    oracle_peak_rmse: float | None = None
    oracle_peak_r2: float | None = None
    n_oracle_evals: int = 0
    improved: bool = False                          # better than the fixed baseline model
    baseline_peak_rmse: float | None = None
