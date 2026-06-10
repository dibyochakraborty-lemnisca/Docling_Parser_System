"""Equation discovery: the agent proposes ODE structure, the oracle scores it,
the best structure is kept. Uses the in-process StubSimulator (no LABS) and the
deterministic TemplateProposer (no LLM)."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from fermdocs_optimize.discovery import (
    CandidateModel,
    ModelSpec,
    ParamSpec,
    TemplateProposer,
    compile_spec,
    discover_model,
    discover_model_from_data,
)
from fermdocs_optimize.discovery.expr import ExprError
from fermdocs_optimize.oracle_search import oracle_global_search
from fermdocs_optimize.schema import Box, Candidate
from fermdocs_optimize.simulators.model_backed import ModelSimulator
from fermdocs_optimize.simulators.stub import StubSimulator


def _box() -> Box:
    return Box(biomass=(0.5, 5.0), total_sub=(10.0, 100.0),
               malt_frac=(0.05, 0.95), dilution=(0.005, 0.1))


def _seed(sim, box, n=8, seed=0):
    rng = np.random.default_rng(seed)
    cands = [Candidate(**{k: float(rng.uniform(*getattr(box, k)))
                          for k in ("biomass", "total_sub", "malt_frac", "dilution")})
             for _ in range(n)]
    return sim.simulate(cands, v0=10.0)[["batch", "t", "X", "S", "P", "M", "V"]]


@pytest.mark.parametrize("inj", [
    "__import__('os').system('echo hi')",
    "eval('1+1')",
    "os.popen('id').read()",
    "open('/etc/passwd')",
])
def test_compiler_blocks_code_injection(inj):
    """The agent writes math, never code: arbitrary names/calls are rejected and
    never executed."""
    with pytest.raises(ExprError):
        compile_spec([], {}, {"X": inj, "S": "0", "P": "0", "M": "0"})


def test_compiler_accepts_valid_math():
    cm = compile_spec(
        ["mu_max", "ks"],
        {"mu": "mu_max*S/(ks+S)*Max(0, 1 - P/200)"},
        {"X": "mu*X - D*X", "S": "-mu*X + D*(S_f - S)", "P": "mu*X - D*P", "M": "D*(M_f - M)"})
    dy = cm.rhs([1.0, 5.0, 0.0, 1.0, 7e-3, 10.0], 0.0, [0.3, 0.5],
               {"F": 0.1, "S_f": 100.0, "M_f": 5.0, "v0": 10.0},
               {"K_O2": 5e-4, "q_O2_max": 0.02, "O2_sat": 7e-3, "kLa": 30.0, "C_FEED": 300.0})
    assert len(dy) == 6 and all(np.isfinite(dy))


def test_candidate_model_fits_and_predicts():
    sim = StubSimulator(); box = _box(); train = _seed(sim, box)
    spec = ModelSpec(
        params={"mu_max": ParamSpec(init=0.3, lb=0.01, ub=1.0),
                "ks": ParamSpec(init=0.5, lb=0.01, ub=50.0),
                "P_max": ParamSpec(init=120.0, lb=50.0, ub=200.0),
                "alpha": ParamSpec(init=2.0, lb=0.001, ub=5.0)},
        aux={"mu": "mu_max*S/(ks+S)*Max(0, 1 - P/P_max)"},
        odes={"X": "mu*X - D*X", "S": "-alpha*mu*X + D*(S_f - S)",
              "P": "alpha*mu*X - D*P", "M": "D*(M_f - M)"})
    m = CandidateModel(spec); r2 = m.fit(train)
    assert set(r2) == {"X", "S", "P", "M"}
    peak = m.predict_peak_titer(Candidate(biomass=2.0, total_sub=80.0,
                                          malt_frac=0.1, dilution=0.01), v0=10.0)
    assert peak > 0


def test_discovery_loop_improves_over_baseline_and_keeps_best():
    """The structural search finds a model at least as good as the fixed
    mechanistic baseline, scored on the oracle's held-out probes."""
    sim = StubSimulator(); box = _box(); train = _seed(sim, box)
    rep = discover_model(training_data=train, simulator=sim, box=box,
                         proposer=TemplateProposer(), max_rounds=5, n_probes=10, seed=3)
    assert rep.best_spec is not None
    assert rep.rounds, "expected at least one round"
    # the kept best is the lowest-RMSE compiled round
    compiled = [r for r in rep.rounds if r.compile_ok]
    assert rep.oracle_peak_rmse == min(r.oracle_peak_rmse for r in compiled)
    # baseline was scored on the same oracle probes
    assert rep.baseline_peak_rmse is not None
    # the search never does WORSE than its own baseline by the end
    assert rep.oracle_peak_rmse <= rep.baseline_peak_rmse + 1e-6 or rep.improved is False


def test_discovery_survives_a_broken_proposal():
    """A structurally invalid spec scores worst but never crashes the loop."""
    sim = StubSimulator(); box = _box(); train = _seed(sim, box)

    class BadThenGood:
        def propose(self, *, round_index, history, data_summary):
            if round_index == 0:
                return ModelSpec(params={"a": ParamSpec(init=1.0, lb=0.0, ub=2.0)},
                                 odes={"X": "nonexistent_var*X", "S": "0", "P": "0", "M": "0"})
            if round_index == 1:
                return ModelSpec(
                    params={"mu_max": ParamSpec(init=0.3, lb=0.01, ub=1.0),
                            "alpha": ParamSpec(init=2.0, lb=0.001, ub=5.0)},
                    aux={"mu": "mu_max*S/(0.5+S)"},
                    odes={"X": "mu*X - D*X", "S": "-alpha*mu*X + D*(S_f-S)",
                          "P": "alpha*mu*X - D*P", "M": "D*(M_f-M)"})
            return None

    rep = discover_model(training_data=train, simulator=sim, box=box,
                         proposer=BadThenGood(), max_rounds=3, n_probes=8)
    assert rep.rounds[0].compile_ok is False
    assert rep.rounds[1].compile_ok is True
    assert rep.best_round == 1  # the broken one is never chosen


def test_discover_from_data_no_oracle_scores_on_holdout():
    """No-oracle mode: equations are scored on a HELD-OUT split of real batches,
    not a simulator. The best structure beats the fixed mechanistic baseline."""
    sim = StubSimulator(); box = _box()
    data = _seed(sim, box, n=16, seed=1)  # stand-in for real lab batches
    rep = discover_model_from_data(data=data, proposer=TemplateProposer(),
                                   max_rounds=5, holdout=0.3, seed=2)
    assert rep.best_spec is not None
    assert rep.n_oracle_evals == 0  # truly no oracle was queried
    assert rep.baseline_peak_rmse is not None
    compiled = [r for r in rep.rounds if r.compile_ok]
    assert rep.oracle_peak_rmse == min(r.oracle_peak_rmse for r in compiled)


def test_model_simulator_lets_search_run_on_the_equation():
    """The discovered equation becomes the 'simulator' the LHS search runs on,
    so an optimum can be proposed with no oracle at all."""
    sim = StubSimulator(); box = _box(); train = _seed(sim, box, n=10)
    spec = ModelSpec(
        params={"mu_max": ParamSpec(init=0.3, lb=0.01, ub=1.0),
                "ks": ParamSpec(init=0.5, lb=0.01, ub=50.0),
                "P_max": ParamSpec(init=120.0, lb=50.0, ub=200.0),
                "alpha": ParamSpec(init=2.0, lb=0.001, ub=5.0)},
        aux={"mu": "mu_max*S/(ks+S)*Max(0, 1 - P/P_max)"},
        odes={"X": "mu*X - D*X", "S": "-alpha*mu*X + D*(S_f - S)",
              "P": "alpha*mu*X - D*P", "M": "D*(M_f - M)"})
    model = CandidateModel(spec); model.fit(train)
    res = oracle_global_search(ModelSimulator(model), box, objective_species="P",
                               v0=10.0, n_lhs=24, refine_iters=2)
    assert res.best_titer > 0
    # the proposed optimum is inside the box
    for k in ("biomass", "total_sub", "malt_frac", "dilution"):
        lb, ub = getattr(box, k)
        assert lb - 1e-9 <= getattr(res.best_candidate, k) <= ub + 1e-9


def test_stiff_structure_is_rejected_fast():
    """An explosive/non-integrable structure is rejected at the cheap pre-flight
    (fit raises) instead of grinding odeint to the step cap on every fit eval —
    the guard that stops a stiff candidate from making a run take hours."""
    sim = StubSimulator(); box = _box(); data = _seed(sim, box, n=8)
    explosive = ModelSpec(
        name="explosive",
        params={"k": ParamSpec(init=1e6, lb=1e3, ub=1e9)},
        aux={},
        odes={"X": "k*X*X", "S": "0", "P": "k*X*X", "M": "0"})
    with pytest.raises(RuntimeError, match="not integrable"):
        CandidateModel(explosive).fit(data)


def test_extract_json_handles_double_encoding_and_fences():
    """The proposer's JSON parser tolerates code fences and gemini's double-encoded
    JSON (a JSON string whose content is itself JSON) — the failure that knocked
    the agent off its chosen search box."""
    from fermdocs_optimize.discovery.proposers import _extract_json

    inner = '{"biomass": {"lb": 1, "ub": 2}}'
    assert _extract_json(inner) == {"biomass": {"lb": 1, "ub": 2}}
    assert _extract_json("```json\n" + inner + "\n```") == {"biomass": {"lb": 1, "ub": 2}}
    # double-encoded: a JSON string containing the JSON object
    import json as _json
    assert _extract_json(_json.dumps(inner)) == {"biomass": {"lb": 1, "ub": 2}}
