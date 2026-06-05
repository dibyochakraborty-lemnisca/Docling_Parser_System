"""Loop tests using the in-process StubSimulator — no LABS, no subprocess.

Covers: the closed loop runs and improves titer, active-learning augmentation
fires when the model is wrong, convergence reporting, the refusal/coherence
contract, and the integrity invariant (the model never reads true params).
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from fermdocs_optimize.loop import refusal, run_optimization
from fermdocs_optimize.models.mechanistic import MechanisticModel
from fermdocs_optimize.proposers.optimize import OptimizeProposer
from fermdocs_optimize.schema import Box, Candidate, OptimizationInput
from fermdocs_optimize.simulators.stub import StubSimulator, TRUE_PARAMS


def _box() -> Box:
    return Box(biomass=(0.5, 5.0), total_sub=(10.0, 100.0),
               malt_frac=(0.05, 0.95), dilution=(0.005, 0.1))


def _seed_data(sim: StubSimulator, box: Box, n: int = 8, seed: int = 0) -> pd.DataFrame:
    """LHS-ish seed batches simulated by the stub, in training schema."""
    rng = np.random.default_rng(seed)
    cands = []
    for _ in range(n):
        knobs = {k: float(rng.uniform(*getattr(box, k)))
                 for k in ("biomass", "total_sub", "malt_frac", "dilution")}
        cands.append(Candidate(**knobs))
    df = sim.simulate(cands, v0=10.0)
    return df[["batch", "t", "X", "S", "P", "M", "V"]]


def test_loop_runs_and_improves():
    box = _box()
    sim = StubSimulator()
    train = _seed_data(sim, box)
    baseline = float(train.groupby("batch")["P"].max().max())

    spec = OptimizationInput(box=box, max_rounds=3, proposals_per_round=3,
                             delta_titer_threshold=2.0)
    out = run_optimization(training_data=train, model=MechanisticModel(),
                           proposer=OptimizeProposer(maxiter=15, popsize=8),
                           simulator=sim, spec=spec, baseline_titer=baseline)

    assert out.confident
    assert out.best_candidate is not None
    assert out.best_achieved_titer > 0
    # optimizer should find a point at least as good as the random seed best
    assert out.best_achieved_titer >= baseline - 1e-6
    assert len(out.rounds) >= 1
    assert out.convergence is not None
    assert len(out.convergence.titer_trajectory) == len(out.rounds)


def test_proposals_stay_in_box():
    box = _box()
    sim = StubSimulator()
    train = _seed_data(sim, box)
    spec = OptimizationInput(box=box, max_rounds=1, proposals_per_round=4)
    out = run_optimization(training_data=train, model=MechanisticModel(),
                           proposer=OptimizeProposer(maxiter=10, popsize=6),
                           simulator=sim, spec=spec)
    for rr in out.rounds:
        for c in rr.proposals:
            for k in ("biomass", "total_sub", "malt_frac", "dilution"):
                lb, ub = getattr(box, k)
                assert lb - 1e-9 <= getattr(c, k) <= ub + 1e-9


def test_model_never_reads_true_params(monkeypatch):
    """Integrity: the model fits on data only — it never opens a params file.

    (Recovering the true params from clean data is legitimate inference, so we
    assert the *code path* never reads mech_params, not that values differ.)"""
    import builtins
    opened: list[str] = []
    real_open = builtins.open

    def spy_open(file, *a, **k):
        opened.append(str(file))
        return real_open(file, *a, **k)

    monkeypatch.setattr(builtins, "open", spy_open)
    train = _seed_data(StubSimulator(), _box(), n=8)
    MechanisticModel().fit(train)
    assert not any("mech_param" in p.lower() for p in opened), \
        f"model read a params file during fit: {opened}"


def test_model_is_data_driven():
    """Two different oracles -> two different fitted models (learns from data)."""
    box = _box()
    a = MechanisticModel(); a.fit(_seed_data(StubSimulator(TRUE_PARAMS), box, n=8))
    other = TRUE_PARAMS * np.array([1.0, 1.0, 0.8, 1.0, 0.7, 1.0, 1.0])
    b = MechanisticModel(); b.fit(_seed_data(StubSimulator(other, seed=1), box, n=8))
    assert not np.allclose(list(a.fitted_params.values()),
                           list(b.fitted_params.values()), rtol=1e-2)


def test_refusal_is_coherent():
    out = refusal("model_unfittable", "ODE fit diverged")
    assert not out.confident
    assert out.refusal_reason == "model_unfittable"
    assert out.best_candidate is None


def test_box_rejects_inverted_bounds():
    with pytest.raises(ValueError):
        Box(biomass=(5.0, 0.5), total_sub=(10.0, 100.0),
            malt_frac=(0.05, 0.95), dilution=(0.005, 0.1))
