"""Typed contracts for the optimizer system.

The optimizer drives a closed loop — fit a model, propose operating points,
evaluate them on a simulator oracle, fold surprises back in, repeat — to push a
target variable (product titer) as high as possible within the feasible box.

These DTOs are the boundary between the deterministic loop (`loop.py`), the
agentic shell (`agent.py`), and the API/CLI. Knob names match the LABS
explicit-batch config so a `Candidate` serializes straight into a batch.
"""
from __future__ import annotations

from pydantic import BaseModel, Field, model_validator

# The four decision variables, named exactly as LABS explicit-batch fields.
KNOB_NAMES = ("biomass", "total_sub", "malt_frac", "dilution")


class Box(BaseModel):
    """Feasible search box: per-knob [lb, ub]. Sourced from a process family's
    var_params (e.g. config.json). Proposals are clamped to this box."""

    biomass: tuple[float, float]
    total_sub: tuple[float, float]
    malt_frac: tuple[float, float]
    dilution: tuple[float, float]

    def as_list(self) -> list[tuple[float, float]]:
        return [getattr(self, k) for k in KNOB_NAMES]

    @model_validator(mode="after")
    def _ordered(self) -> "Box":
        for k in KNOB_NAMES:
            lb, ub = getattr(self, k)
            if lb > ub:
                raise ValueError(f"box[{k}]: lb {lb} > ub {ub}")
        return self


class Candidate(BaseModel):
    """One proposed operating point (a batch's controllable knobs)."""

    biomass: float
    total_sub: float
    malt_frac: float
    dilution: float
    predicted_peak_titer: float | None = None  # model's prediction at proposal time

    def knobs(self) -> dict[str, float]:
        return {k: getattr(self, k) for k in KNOB_NAMES}


class FitReport(BaseModel):
    """Quality of the agent's model fit on its current training data."""

    n_train_batches: int
    r2_by_species: dict[str, float]  # X,S,P,M
    fitted_params: dict[str, float]  # the agent's own params (never the true ones)
    target_species_r2: float  # R^2 on the optimization target (P)


class RoundResult(BaseModel):
    """Everything that happened in one loop round."""

    round_index: int
    fit: FitReport
    proposals: list[Candidate]
    # oracle outcomes for the proposals:
    best_candidate: Candidate
    achieved_peak_titer: float  # simulator-verified best P this round
    model_vs_oracle_r2: float  # how well the model predicted the simulated proposals
    augmented_training: bool  # did we fold new_data back in (R2 below gate)?
    n_training_after: int


class ConvergenceReport(BaseModel):
    reason: str  # "delta_below_threshold" | "max_rounds" | "budget_exhausted" | "no_model"
    converged: bool
    titer_trajectory: list[float]  # achieved peak P per round: P0, P1, ...
    final_delta: float | None = None


class OracleSearchReport(BaseModel):
    """Result of an oracle-direct global search — the simulator (ground truth),
    not the surrogate, searches the box. This is how we find/confirm the TRUE
    within-box maximum rather than the model-guided loop's local plateau."""

    best_candidate: Candidate
    best_titer: float                       # oracle-verified peak at best_candidate
    n_oracle_evals: int                     # total simulator evaluations spent
    n_lhs: int                              # size of the dense Latin-hypercube sweep
    knobs_on_boundary: dict[str, str] = Field(default_factory=dict)  # knob -> "lower"|"upper"
    improved_over_loop: bool | None = None  # did it beat the model-guided loop's best?


class OptimizationInput(BaseModel):
    """What the loop needs to run."""

    objective_species: str = "P"  # maximize peak of this species
    box: Box
    max_rounds: int = 8
    proposals_per_round: int = Field(default=4, ge=1)
    delta_titer_threshold: float = 2.0  # stop when ΔP_max < this (g/L)
    good_fit_r2: float = 0.8  # below this on the proposals → augment + refit
    v0: float = 10.0  # initial reactor volume (lab scale)
    # Oracle-direct global search: spend simulator calls to find the TRUE box
    # maximum (dense LHS sweep + pattern-search refinement, all on the oracle).
    oracle_search: bool = False
    n_lhs: int = Field(default=200, ge=1)   # dense sweep size (one batched oracle call)
    refine_iters: int = Field(default=10, ge=0)  # pattern-search refinement steps


class OptimizationOutput(BaseModel):
    """The result contract surfaced to API/CLI/agent."""

    meta: dict = Field(default_factory=dict)  # schema/agent version, run_id, timestamp
    confident: bool
    refusal_reason: str | None = None  # set iff not confident
    best_candidate: Candidate | None = None
    best_achieved_titer: float | None = None
    baseline_titer: float | None = None  # titer of the seed/starting point
    improvement: float | None = None  # best - baseline
    rounds: list[RoundResult] = Field(default_factory=list)
    convergence: ConvergenceReport | None = None
    selection_rationale: str = ""
    # Transparency: how the agent used the model — governing equations + the
    # per-round fit logs (params learned, R²). Surfaced in the UI's model log.
    model_log: list[dict] = Field(default_factory=list)
    # Oracle-direct global search result (when enabled): the true within-box max
    # found by searching the simulator directly, not the surrogate.
    oracle_search: OracleSearchReport | None = None

    @model_validator(mode="after")
    def _coherent(self) -> "OptimizationOutput":
        # Refusal <-> no confident result; mirrors the recommend module's discipline.
        if not self.confident:
            if self.refusal_reason is None:
                raise ValueError("not confident but no refusal_reason")
            if self.best_candidate is not None:
                raise ValueError("refusal must not carry a best_candidate")
        else:
            if self.refusal_reason is not None:
                raise ValueError("confident result must not carry a refusal_reason")
            if self.best_candidate is None:
                raise ValueError("confident result must carry a best_candidate")
        return self
