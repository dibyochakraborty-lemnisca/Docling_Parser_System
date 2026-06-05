"""Oracle-direct global search: the simulator (ground truth) finds the true box
maximum, not the surrogate. Uses the in-process StubSimulator (no LABS)."""
from __future__ import annotations

import numpy as np
import pandas as pd

from fermdocs_optimize.loop import run_optimization
from fermdocs_optimize.models.mechanistic import MechanisticModel
from fermdocs_optimize.oracle_search import oracle_global_search
from fermdocs_optimize.proposers.optimize import OptimizeProposer
from fermdocs_optimize.schema import Box, Candidate, OptimizationInput
from fermdocs_optimize.simulators.stub import StubSimulator


def _box() -> Box:
    return Box(biomass=(0.5, 5.0), total_sub=(10.0, 100.0),
               malt_frac=(0.05, 0.95), dilution=(0.005, 0.1))


def _seed(sim, box, n=6, seed=0):
    rng = np.random.default_rng(seed)
    cands = [Candidate(**{k: float(rng.uniform(*getattr(box, k)))
                          for k in ("biomass", "total_sub", "malt_frac", "dilution")})
             for _ in range(n)]
    return sim.simulate(cands, v0=10.0)[["batch", "t", "X", "S", "P", "M", "V"]]


def test_oracle_search_finds_true_max_and_counts_evals():
    sim = StubSimulator()
    box = _box()
    rep = oracle_global_search(sim, box, v0=10.0, n_lhs=40, refine_iters=2)

    assert rep.best_titer > 0
    assert rep.n_oracle_evals == 40 + 2 * 8  # sweep + 2 refine steps × (4 knobs × 2 dirs)
    assert rep.n_lhs == 40
    # the reported max is real: re-simulating the best knobs reproduces it (oracle)
    df = sim.simulate([rep.best_candidate], v0=10.0)
    assert abs(float(df["P"].max()) - rep.best_titer) < 1e-2  # report rounds to 3 dp
    # best point is inside the box
    for k in ("biomass", "total_sub", "malt_frac", "dilution"):
        lb, ub = getattr(box, k)
        assert lb - 1e-9 <= getattr(rep.best_candidate, k) <= ub + 1e-9


def test_oracle_search_matches_or_beats_model_loop():
    """The whole point: the oracle search should not do WORSE than the peak-blind
    model loop — it finds the true max by searching ground truth directly."""
    sim = StubSimulator()
    box = _box()
    train = _seed(sim, box)
    baseline = float(train.groupby("batch")["P"].max().max())

    spec = OptimizationInput(box=box, max_rounds=1, proposals_per_round=2,
                             oracle_search=True, n_lhs=48, refine_iters=2)
    out = run_optimization(training_data=train, model=MechanisticModel(),
                           proposer=OptimizeProposer(maxiter=6, popsize=4),
                           simulator=sim, spec=spec, baseline_titer=baseline)

    assert out.confident
    assert out.oracle_search is not None
    # the final best is at least the oracle search's true max
    assert out.best_achieved_titer >= out.oracle_search.best_titer - 1e-6
    # and it beats the seed baseline
    assert out.best_achieved_titer >= baseline - 1e-6
    assert "oracle global search" in out.selection_rationale.lower()


def test_no_oracle_search_leaves_field_none():
    sim = StubSimulator()
    box = _box()
    train = _seed(sim, box)
    spec = OptimizationInput(box=box, max_rounds=1, proposals_per_round=2,
                             oracle_search=False)
    out = run_optimization(training_data=train, model=MechanisticModel(),
                           proposer=OptimizeProposer(maxiter=6, popsize=4),
                           simulator=sim, spec=spec)
    assert out.oracle_search is None
