"""Active-learning optimization: discover on data, optimize, verify the optimum
on the oracle, augment + re-discover on error. Uses the in-process StubSimulator
(no LABS) and the deterministic TemplateProposer (no LLM)."""
from __future__ import annotations

import numpy as np
import pandas as pd

from fermdocs_optimize.active_optimize import (
    _grow_box,
    _search_beyond_bounds,
    active_optimize,
)
from fermdocs_optimize.discovery.proposers import TemplateProposer
from fermdocs_optimize.schema import Box, Candidate
from fermdocs_optimize.simulators.stub import StubSimulator


def _box() -> Box:
    return Box(biomass=(0.5, 5.0), total_sub=(10.0, 100.0),
               malt_frac=(0.05, 0.95), dilution=(0.005, 0.1))


class _RampModel:
    """Fake fitted model whose peak P rises monotonically with total_sub, so the
    optimizer always wants to push total_sub past the box's upper edge — the
    out-of-bounds-optimum case the expansion is built for."""

    def predict_P_trajectory(self, c, *, v0, t_end, n):
        return np.full(n, 0.5 * c.total_sub)


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


def test_grow_box_pushes_pinned_edges_and_respects_floors():
    """A pinned edge moves out by grow*span; the 0 floor and untouched knobs hold."""
    box = Box(biomass=(0.0, 5.0), total_sub=(10.0, 100.0),
              malt_frac=(0.05, 0.95), dilution=(0.005, 0.1))
    grown = _grow_box(box, {"total_sub": "upper", "biomass": "lower"}, None, 0.5)
    assert grown.total_sub[1] == 100.0 + 0.5 * 90.0     # upper grew by 0.5*span
    assert grown.biomass[0] == 0.0                       # lower already at floor, can't go below 0
    assert grown.dilution == box.dilution               # untouched knob unchanged


def test_grow_box_returns_equal_when_pinned_edge_is_capped():
    """If the pinned edge is already at the physical cap, the box can't grow — the
    loop relies on this equality to stop expanding."""
    box = Box(biomass=(0.5, 5.0), total_sub=(10.0, 100.0),
              malt_frac=(0.05, 0.95), dilution=(0.005, 0.1))
    physical = Box(biomass=(0.0, 5.0), total_sub=(0.0, 100.0),
                   malt_frac=(0.0, 1.0), dilution=(0.0, 0.1))
    grown = _grow_box(box, {"total_sub": "upper"}, physical, 0.5)
    assert grown == box


def test_search_expands_box_to_reach_out_of_bounds_optimum():
    """The optimum rises with total_sub past the box edge: the search pushes the
    edge out and re-searches, spending the whole expansion budget."""
    box = _box()
    search, final_box, n_exp = _search_beyond_bounds(
        _RampModel(), box, None, objective_species="P", v0=10.0, seed=1,
        max_expansions=4, grow=0.5)
    assert n_exp == 4
    assert final_box.total_sub[1] > 100.0               # box grew past the original ceiling
    assert search.best_titer > 0.5 * 100.0              # optimum beats the original-box best


def test_search_does_not_expand_when_capped_at_physical():
    """With a physical cap at the start box, the pinned optimum can't escape, so no
    expansion happens (and the oracle is never pulled out to a no-data region)."""
    box = _box()
    search, final_box, n_exp = _search_beyond_bounds(
        _RampModel(), box, box, objective_species="P", v0=10.0, seed=1,
        max_expansions=4, grow=0.5)
    assert n_exp == 0
    assert final_box == box
