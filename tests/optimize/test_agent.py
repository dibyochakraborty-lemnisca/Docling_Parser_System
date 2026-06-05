"""Agentic-shell tests: the OptimizerAgent wraps the deterministic loop.

Covers: the ReAct loop drives the tools and narrates a confident result; the
agent CANNOT fabricate the numbers (they come from the oracle-backed loop); an
emit before any loop run is an honest refusal; the no-client path degrades to the
deterministic loop; the integrity invariant holds end-to-end (the agent never
reads the oracle's true params); and the deterministic honesty annotations fire.
"""
from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest

from fermdocs_optimize import agent as agent_mod
from fermdocs_optimize.agent import OptimizerAgent
from fermdocs_optimize.proposers.optimize import OptimizeProposer
from fermdocs_optimize.schema import (
    Box,
    Candidate,
    ConvergenceReport,
    FitReport,
    OptimizationOutput,
    RoundResult,
)
from fermdocs_optimize.simulators.stub import StubSimulator
from fermdocs_optimize.tools_bundle import factory as tb


def _box() -> Box:
    return Box(biomass=(0.5, 5.0), total_sub=(10.0, 100.0),
               malt_frac=(0.05, 0.95), dilution=(0.005, 0.1))


def _seed(sim: StubSimulator, box: Box, n: int = 6, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    cands = [Candidate(**{k: float(rng.uniform(*getattr(box, k)))
                          for k in ("biomass", "total_sub", "malt_frac", "dilution")})
             for _ in range(n)]
    return sim.simulate(cands, v0=10.0)[["batch", "t", "X", "S", "P", "M", "V"]]


@pytest.fixture(autouse=True)
def _fast_proposers(monkeypatch):
    """Keep the real loop but make the global search cheap for tests."""
    monkeypatch.setitem(tb.PROPOSER_REGISTRY, "optimize",
                        lambda: OptimizeProposer(maxiter=6, popsize=4))
    monkeypatch.setattr(agent_mod, "OptimizeProposer",
                        lambda: OptimizeProposer(maxiter=6, popsize=4))


class ScriptedClient:
    """A fake LLM that replays a fixed list of ReAct responses."""

    def __init__(self, script: list[dict]):
        self._script = list(script)
        self.calls = 0

    def call(self, system, messages):
        self.calls += 1
        if not self._script:
            return {"action": "emit", "payload_json": json.dumps({"rationale": "done"})}
        return self._script.pop(0)


def _happy_script(loop_args: dict | None = None, narration: dict | None = None) -> list[dict]:
    loop_args = loop_args or {"objective_species": "P", "model": "mechanistic",
                              "proposer": "optimize", "max_rounds": 1,
                              "proposals_per_round": 2, "delta_titer_threshold": 2.0,
                              "oracle_search": True, "n_lhs": 24, "refine_iters": 1}
    narration = narration or {"rationale": "Run dilution low, substrate high.",
                              "confidence_note": "converged in one round"}
    return [
        {"action": "tool_call", "tool": "get_experiment", "args": {}},
        {"action": "tool_call", "tool": "get_box", "args": {}},
        {"action": "tool_call", "tool": "get_skill", "args": {"name": "optimize-titer"}},
        {"action": "tool_call", "tool": "run_optimization_loop", "args": loop_args},
        {"action": "tool_call", "tool": "submit_optimization",
         "args": {"payload_json": json.dumps(narration)}},
    ]


def test_agent_runs_and_narrates():
    sim = StubSimulator()
    box = _box()
    train = _seed(sim, box)
    client = ScriptedClient(_happy_script())
    out = OptimizerAgent(client).optimize(training_data=train, box=box, simulator=sim)

    assert out.confident
    assert out.best_candidate is not None
    assert out.best_achieved_titer and out.best_achieved_titer > 0
    # the agent's prose is grafted in over the loop's authoritative basis
    assert "Run dilution low" in out.selection_rationale
    assert "basis:" in out.selection_rationale
    assert out.meta.get("provider") == "gemini"


def test_numbers_come_from_loop_not_narration():
    """The agent cannot inflate the titer: the value comes from the oracle-backed
    loop, not from whatever the LLM writes in its narration."""
    sim = StubSimulator()
    box = _box()
    train = _seed(sim, box)
    liar = {"rationale": "We achieved an incredible 9999 g/L!!", "confidence_note": "trust me"}
    client = ScriptedClient(_happy_script(narration=liar))
    out = OptimizerAgent(client).optimize(training_data=train, box=box, simulator=sim)

    # the number comes from a REAL source — the per-round oracle results or the
    # oracle global search — never from the LLM's narration.
    real_sources = {round(r.achieved_peak_titer, 3) for r in out.rounds}
    if out.oracle_search is not None:
        real_sources.add(round(out.oracle_search.best_titer, 3))
    assert round(out.best_achieved_titer, 3) in real_sources
    assert out.best_achieved_titer < 1000  # physical titer, not the fabricated 9999


def test_emit_before_loop_is_refusal():
    sim = StubSimulator()
    box = _box()
    train = _seed(sim, box)
    client = ScriptedClient([{"action": "emit",
                              "payload_json": json.dumps({"rationale": "skip the work"})}])
    out = OptimizerAgent(client).optimize(training_data=train, box=box, simulator=sim)

    assert not out.confident
    assert out.refusal_reason == "no_loop_run"
    assert out.best_candidate is None


def test_no_client_runs_deterministic():
    sim = StubSimulator()
    box = _box()
    train = _seed(sim, box)
    out = OptimizerAgent(None).optimize(training_data=train, box=box, simulator=sim)

    assert out.confident
    assert out.best_candidate is not None
    assert out.meta.get("provider") == "none"
    assert "deterministic loop" in out.selection_rationale.lower()


def test_agent_never_reads_true_params(monkeypatch):
    """Integrity end-to-end: across a full agent run, no mech-params file is opened
    (the model fits on data; the oracle is the in-process stub here)."""
    import builtins
    opened: list[str] = []
    real_open = builtins.open
    monkeypatch.setattr(builtins, "open",
                        lambda f, *a, **k: (opened.append(str(f)), real_open(f, *a, **k))[1])

    sim = StubSimulator()
    box = _box()
    train = _seed(sim, box)
    OptimizerAgent(ScriptedClient(_happy_script())).optimize(
        training_data=train, box=box, simulator=sim)
    assert not any("mech_param" in p.lower() for p in opened), opened


def _fake_output(*, mvo_r2: float, improvement: float) -> OptimizationOutput:
    cand = Candidate(biomass=1.0, total_sub=50.0, malt_frac=0.1, dilution=0.01)
    rnd = RoundResult(
        round_index=0,
        fit=FitReport(n_train_batches=6, r2_by_species={"P": 0.9},
                      fitted_params={"mu_max": 0.3}, target_species_r2=0.9),
        proposals=[cand], best_candidate=cand, achieved_peak_titer=100.0,
        model_vs_oracle_r2=mvo_r2, augmented_training=True, n_training_after=8)
    return OptimizationOutput(
        confident=True, best_candidate=cand, best_achieved_titer=100.0,
        baseline_titer=100.0 - improvement, improvement=improvement, rounds=[rnd],
        convergence=ConvergenceReport(reason="max_rounds", converged=False,
                                      titer_trajectory=[100.0]),
        selection_rationale="base.")


def test_debate_levers_inform_but_do_not_constrain(tmp_path):
    """The optimizer reads debated levers via get_levers, surfaces them in the
    rationale (inform-only), but the search and result still come from the loop."""
    # a minimal optimization_debate.json (HypothesisOutput shape)
    debate = {"final_hypotheses": [
        {"hyp_id": "H-0001", "summary": "lower dilution to avoid washout",
         "affected_variables": ["V", "S", "P"], "actionable_recommendation": "drop dilution",
         "confidence": 0.7, "supporting_specialists": ["mass_transfer"]},
    ]}
    path = tmp_path / "optimization_debate.json"
    path.write_text(json.dumps(debate))

    sim = StubSimulator()
    box = _box()
    train = _seed(sim, box)
    script = [
        {"action": "tool_call", "tool": "get_experiment", "args": {}},
        {"action": "tool_call", "tool": "get_levers", "args": {}},
        {"action": "tool_call", "tool": "run_optimization_loop",
         "args": {"max_rounds": 1, "proposals_per_round": 2}},
        {"action": "tool_call", "tool": "submit_optimization",
         "args": {"payload_json": json.dumps(
             {"rationale": "Drop dilution.", "lever_reconciliation": "confirms H-0001"})}},
    ]
    out = OptimizerAgent(ScriptedClient(script)).optimize(
        training_data=train, box=box, simulator=sim, debate_output_path=str(path))

    assert out.confident
    assert out.meta.get("grounding_levers") == ["H-0001"]
    # the LLM's reconciliation + the deterministic lever appendix both surface
    assert "confirms H-0001" in out.selection_rationale
    assert "debated levers vs verified optimum" in out.selection_rationale
    assert "H-0001" in out.selection_rationale
    # inform-only: the box was the full feasible box (proposals stayed in it)
    for rr in out.rounds:
        for c in rr.proposals:
            for k in ("biomass", "total_sub", "malt_frac", "dilution"):
                lb, ub = getattr(box, k)
                assert lb - 1e-9 <= getattr(c, k) <= ub + 1e-9


def test_get_levers_empty_without_debate():
    sim = StubSimulator()
    box = _box()
    train = _seed(sim, box)
    out = OptimizerAgent(ScriptedClient(_happy_script())).optimize(
        training_data=train, box=box, simulator=sim)
    assert out.meta.get("grounding_levers") == []
    assert "debated levers vs verified optimum" not in out.selection_rationale


def test_model_log_carries_equations_and_fits():
    """Transparency: the optimizer output logs the governing equations (model
    card) plus a per-round fit entry (params learned + R²)."""
    from fermdocs_optimize.models.mechanistic import MechanisticModel

    card = MechanisticModel.model_card()
    assert card["kind"] == "equations"
    assert any("mu_max" in eq for eq in card["equations"])
    assert card["fitted_parameters"] and "least_squares" in card["method"]

    sim = StubSimulator()
    box = _box()
    train = _seed(sim, box)
    out = OptimizerAgent(None).optimize(training_data=train, box=box, simulator=sim)
    assert out.model_log, "expected a model log"
    assert out.model_log[0]["kind"] == "equations"  # leads with the equations
    fits = [e for e in out.model_log if e["kind"] == "fit"]
    assert fits, "expected at least one fit log"
    assert fits[0]["fitted_params"] and fits[0]["r2_by_species"]


def test_honesty_suffix_flags_low_agreement_and_no_improvement():
    suffix = OptimizerAgent._honesty_suffix(_fake_output(mvo_r2=0.1, improvement=5.0))
    assert "model-vs-oracle" in suffix and "lead, not a confirmed global optimum" in suffix

    suffix2 = OptimizerAgent._honesty_suffix(_fake_output(mvo_r2=0.95, improvement=0.0))
    assert "no improvement over baseline" in suffix2

    clean = OptimizerAgent._honesty_suffix(_fake_output(mvo_r2=0.95, improvement=5.0))
    assert clean == ""
