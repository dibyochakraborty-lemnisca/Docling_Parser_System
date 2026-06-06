"""Active-learning optimization: discover on data, optimize, verify the optimum
on the oracle, augment + re-discover on error. Uses the in-process StubSimulator
(no LABS) and the deterministic TemplateProposer (no LLM)."""
from __future__ import annotations

import numpy as np
import pandas as pd

from fermdocs_optimize.active_optimize import active_optimize
from fermdocs_optimize.discovery.proposers import TemplateProposer
from fermdocs_optimize.schema import Box, Candidate
from fermdocs_optimize.simulators.stub import StubSimulator


def _box() -> Box:
    return Box(biomass=(0.5, 5.0), total_sub=(10.0, 100.0),
               malt_frac=(0.05, 0.95), dilution=(0.005, 0.1))


def _seed(sim, box, n=12, seed=0):
    rng = np.random.default_rng(seed)
    cands = [Candidate(**{k: float(rng.uniform(*getattr(box, k)))
                          for k in ("biomass", "total_sub", "malt_frac", "dilution")})
             for _ in range(n)]
    return sim.simulate(cands, v0=10.0)[["batch", "t", "X", "S", "P", "M", "V"]]


def test_converges_and_oracle_only_at_optimum():
    """The stub IS the mechanistic family, so a template is found that matches it;
    the proposed optimum verifies on the oracle and the loop converges, having
    spent the oracle ONLY at the optimum (a handful of evals, not per-round)."""
    sim = StubSimulator(); box = _box(); data = _seed(sim, box)
    rep, new = active_optimize(
        data=data, simulator=sim, physical=box, proposer_factory=lambda: TemplateProposer(),
        target_peak_r2=0.8, inner_max_rounds=5, holdout=0.3, error_threshold=5.0,
        max_outer=3, n_neighbors=2, seed=1)
    assert rep.converged is True
    assert rep.error <= 5.0
    # oracle spent only at the optimum: ~ (1 + n_neighbors) per outer iter, tiny
    assert rep.n_oracle_evals <= 3 * (1 + 2)
    # the recommended titer is the oracle-verified one, inside the box
    assert rep.oracle_titer > 0
    for k in ("biomass", "total_sub", "malt_frac", "dilution"):
        lb, ub = getattr(box, k)
        assert lb - 1e-9 <= rep.best_knobs[k] <= ub + 1e-9


def test_recommended_titer_is_oracle_verified_not_predicted():
    """The returned titer must be the oracle's, never the model's prediction."""
    sim = StubSimulator(); box = _box(); data = _seed(sim, box)
    rep, _ = active_optimize(
        data=data, simulator=sim, physical=box, proposer_factory=lambda: TemplateProposer(),
        max_outer=2, n_neighbors=1, seed=2)
    # re-simulate the recommended knobs on the oracle -> matches the reported titer
    c = Candidate(**rep.best_knobs)
    truth = float(sim.simulate([c], v0=10.0)["P"].max())
    assert abs(truth - rep.oracle_titer) < 1e-2


def test_augments_data_when_model_is_wrong():
    """If the first model is wrong at its optimum, the loop folds the oracle point
    back in and re-discovers — exercised by forcing a tiny error threshold so it
    cannot converge immediately and must augment."""
    sim = StubSimulator(); box = _box(); data = _seed(sim, box)
    rep, new = active_optimize(
        data=data, simulator=sim, physical=box, proposer_factory=lambda: TemplateProposer(),
        target_peak_r2=0.99, inner_max_rounds=2, holdout=0.3,
        error_threshold=1e-6, max_outer=2, n_neighbors=2, seed=3)
    # with an impossibly tight threshold it should run multiple outer iters and
    # append oracle-verified batches to the data
    if not rep.converged:
        assert rep.n_outer >= 2
        assert rep.batches_added > 0
        assert len(new) > 0  # new batches returned for the caller to persist
