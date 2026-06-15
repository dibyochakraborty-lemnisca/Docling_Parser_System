"""General mechanistic discovery — an ODE over ALL measured variables, not the
fixed LABS species. Recovers a 3-state system + CV-clears, refuses on noise, and
the generalized compiler integrates an arbitrary state vector (batch model).
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from scipy.integrate import odeint

from fermdocs_optimize.data_equation import discover_and_optimize
from fermdocs_optimize.discovery.expr import compile_spec
from fermdocs_optimize.discovery.general_mech import (
    discover_general_mechanistic,
    load_state_runs,
)

# True 3-state LINEAR system: substrate A decays, biomass B fed by A, product C
# produced from A. (Linear so the affine-linear template recovers it exactly.)
_K = {"kA": 0.06, "kBA": 0.02, "kB": 0.01, "kCA": 0.05}


def _rhs(y, t):
    A, B, C = y
    return [-_K["kA"] * A, _K["kBA"] * A - _K["kB"] * B, _K["kCA"] * A]


def _make_obs(n_runs=8, seed=0):
    rng = np.random.default_rng(seed)
    t = np.array([0, 6, 12, 18, 24, 30, 36, 42, 48], float)
    rows = []
    for k in range(n_runs):
        A0 = 80.0 + 12.0 * k          # substrate loading varies across runs
        sol = odeint(_rhs, [A0, 5.0, 0.0], t)
        for ti, (A, B, C) in zip(t, sol):
            rows += [
                (f"R{k}", "substrate_g_l", ti, float(A)),
                (f"R{k}", "biomass_g_l", ti, float(B)),
                (f"R{k}", "product_g_l", ti, float(C)),
            ]
    return pd.DataFrame(rows, columns=["run_id", "variable", "time_h", "value"])


def test_compiler_integrates_arbitrary_state_batch_model():
    # Two arbitrary states, batch model (no conditions/constants/defaults).
    compiled = compile_spec(
        ["k"], {}, {"a": "-k*a", "b": "k*a"},
        state=("a", "b"), conditions=(), constants=(), ode_defaults={},
        required=("a", "b"))
    sol = odeint(compiled.rhs, [10.0, 0.0], np.linspace(0, 10, 11),
                 args=(np.array([0.1]), {}, {}))
    assert np.all(np.isfinite(sol))
    assert sol[-1, 0] < 10.0 and sol[-1, 1] > 0.0  # a decays into b


def test_load_state_runs_uses_all_relevant_variables():
    runs, states, sym_to_var = load_state_runs(_make_obs(), objective="product_g_l")
    modeled = set(sym_to_var.values())
    # all three measured channels are modeled, not just substrate+product
    assert modeled == {"substrate_g_l", "biomass_g_l", "product_g_l"}
    assert len(runs) == 8
    assert runs[0].Y.shape[1] == 3


def test_discovers_multistate_ode_and_clears_cv():
    fit = discover_general_mechanistic(_make_obs(), objective="product_g_l")
    assert fit is not None
    assert fit.objective_r2 > 0.9               # linear system generalizes
    assert len(fit.states) == 3                 # modeled over all 3 variables
    assert set(fit.sym_to_var.values()) == {"substrate_g_l", "biomass_g_l", "product_g_l"}


def test_end_to_end_mechanistic_wins_and_optimizes_initial_condition():
    out = discover_and_optimize(_make_obs(), objective="product_g_l")
    assert out.cleared is True
    assert out.family == "mechanistic"          # ODE generalized -> beats surrogate
    # the controllable lever is the substrate initial condition (a real variable)
    assert "substrate_g_l.initial" in out.best_knobs
    # product rises with substrate loading -> optimum at the top of observed range
    assert out.best_knobs["substrate_g_l.initial"] == pytest.approx(80 + 12 * 7, abs=10)
    assert out.cv_r2 > 0.9


def test_refuses_on_noise():
    rng = np.random.default_rng(5)
    t = np.array([0, 12, 24, 36, 48], float)
    rows = []
    for k in range(8):
        for ti in t:
            rows += [
                (f"R{k}", "substrate_g_l", ti, float(rng.normal(100, 20))),
                (f"R{k}", "product_g_l", ti, float(rng.normal(100, 30))),
            ]
    obs = pd.DataFrame(rows, columns=["run_id", "variable", "time_h", "value"])
    # objective uncorrelated with anything -> no ODE generalizes -> None.
    assert discover_general_mechanistic(obs, objective="product_g_l") is None
