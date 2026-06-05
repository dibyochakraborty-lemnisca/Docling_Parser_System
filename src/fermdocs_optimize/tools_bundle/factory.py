"""Tool registry for the optimizer ReAct loop.

The agent reads the seed experiment (get_experiment), the feasible box (get_box),
the vendored skills (get_skill), and drives the closed loop (run_optimization_loop).
The loop is the trustworthy core: it fits the model, proposes knobs, and simulates
them on the ground-truth oracle, so the achieved titer and best operating point
cannot be fabricated by the LLM — the agent only chooses the configuration and
narrates the result.

Integrity: the simulator is INJECTED (constructed by the orchestrator from the
mech_params path). Nothing in this module reads the oracle's true parameters; the
agent's model fits on training data only.
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

import pandas as pd

from fermdocs_optimize.evaluate import peak_titer_per_batch
from fermdocs_optimize.loop import refusal, run_optimization
from fermdocs_optimize.models.base import PredictiveModel
from fermdocs_optimize.models.mechanistic import MechanisticModel
from fermdocs_optimize.proposers.base import Proposer
from fermdocs_optimize.proposers.grid import GridProposer
from fermdocs_optimize.proposers.optimize import OptimizeProposer
from fermdocs_optimize.schema import Box, OptimizationInput, OptimizationOutput
from fermdocs_optimize.simulators.base import Simulator
from fermdocs_optimize import skill_loader

_log = logging.getLogger(__name__)

# Swappable strategies, selected by name in run_optimization_loop. New families
# are new entries here (open-closed); the loop and agent are unchanged.
MODEL_REGISTRY: dict[str, Callable[[], PredictiveModel]] = {
    "mechanistic": MechanisticModel,
}
PROPOSER_REGISTRY: dict[str, Callable[[], Proposer]] = {
    "optimize": OptimizeProposer,
    "grid": GridProposer,
}

_MAX_LOOP_RUNS = 4  # the loop shells out to the oracle per proposal; cap re-runs


@dataclass
class _AgentState:
    best_result: OptimizationOutput | None = None  # authoritative loop result
    narration: dict | None = None                  # the agent's submitted prose
    submitted: bool = False
    tool_calls: int = 0
    loop_runs: int = 0


@dataclass
class OptimizeToolBundle:
    training_data: pd.DataFrame
    box: Box
    simulator: Simulator
    baseline_titer: float | None = None
    objective_species: str = "P"
    v0: float = 10.0
    # Inform-only seam: debated optimization levers (from the opportunity debate)
    # the agent reads to narrate + reconcile. They NEVER constrain the search.
    levers: list[dict] = field(default_factory=list)
    state: _AgentState = field(default_factory=_AgentState)

    def _gate(self, tool_name: str) -> dict | None:
        self.state.tool_calls += 1
        if self.state.submitted and tool_name != "submit_optimization":
            return {"error": "already_finalized", "tool": tool_name}
        return None

    # --- read tools --------------------------------------------------------
    def get_experiment(self) -> dict:
        gated = self._gate("get_experiment")
        if gated:
            return gated
        df = self.training_data
        species = [c for c in ("X", "S", "P", "M") if c in df.columns]
        peaks = peak_titer_per_batch(df, self.objective_species) if self.objective_species in df.columns else {}
        baseline = self.baseline_titer
        if baseline is None and peaks:
            baseline = max(peaks.values())
        knob_ranges = {}
        for k in ("biomass", "total_sub", "malt_frac", "dilution"):
            if k in df.columns:
                knob_ranges[k] = [float(df[k].min()), float(df[k].max())]
        return {
            "n_batches": int(df["batch"].nunique()),
            "n_rows": int(len(df)),
            "species_present": species,
            "objective_species": self.objective_species,
            "baseline_peak_titer": None if baseline is None else round(float(baseline), 3),
            "t_range": [float(df["t"].min()), float(df["t"].max())] if "t" in df.columns else None,
            "observed_knob_ranges": knob_ranges or None,
        }

    def get_box(self) -> dict:
        gated = self._gate("get_box")
        if gated:
            return gated
        return {k: list(getattr(self.box, k)) for k in
                ("biomass", "total_sub", "malt_frac", "dilution")}

    def get_levers(self) -> dict:
        """The debated optimization levers (prior over the box). Advisory only:
        the search still covers the full box and the oracle judges everything."""
        gated = self._gate("get_levers")
        if gated:
            return gated
        return {"levers": self.levers, "n": len(self.levers),
                "note": "advisory prior from the opportunity debate; the oracle still verifies every proposal"}

    def get_skill(self, name: str) -> dict:
        gated = self._gate("get_skill")
        if gated:
            return gated
        text = skill_loader.load_skill(name)
        if text is None:
            return {"error": f"unknown skill {name!r}", "available": skill_loader.available_skills()}
        return {"skill": name, "content": text}

    # --- the core: run the deterministic loop on the oracle ----------------
    def run_optimization_loop(
        self,
        objective_species: str | None = None,
        model: str = "mechanistic",
        proposer: str = "optimize",
        max_rounds: int = 6,
        proposals_per_round: int = 4,
        delta_titer_threshold: float = 2.0,
        oracle_search: bool = True,
        n_lhs: int = 200,
        refine_iters: int = 10,
    ) -> dict:
        gated = self._gate("run_optimization_loop")
        if gated:
            return gated
        if self.state.loop_runs >= _MAX_LOOP_RUNS:
            return {"error": "loop_budget_exhausted",
                    "detail": f"run_optimization_loop may be called at most {_MAX_LOOP_RUNS} times"}
        self.state.loop_runs += 1

        model_factory = MODEL_REGISTRY.get(model)
        proposer_factory = PROPOSER_REGISTRY.get(proposer)
        if model_factory is None:
            return {"error": f"unknown model {model!r}", "available": list(MODEL_REGISTRY)}
        if proposer_factory is None:
            return {"error": f"unknown proposer {proposer!r}", "available": list(PROPOSER_REGISTRY)}

        spec = OptimizationInput(
            box=self.box,
            objective_species=objective_species or self.objective_species,
            max_rounds=max(1, int(max_rounds)),
            proposals_per_round=max(1, int(proposals_per_round)),
            delta_titer_threshold=float(delta_titer_threshold),
            v0=self.v0,
            oracle_search=bool(oracle_search),
            n_lhs=max(1, int(n_lhs)),
            refine_iters=max(0, int(refine_iters)),
        )
        try:
            out = run_optimization(
                training_data=self.training_data, model=model_factory(),
                proposer=proposer_factory(), simulator=self.simulator,
                spec=spec, baseline_titer=self.baseline_titer)
        except Exception as exc:  # noqa: BLE001 — a failed run is an honest refusal, never a crash
            _log.exception("optimization loop raised")
            out = refusal("loop_error", f"{model}/{proposer}: {exc}")

        self._track(out)
        return self._summarize(out, model, proposer)

    def _track(self, out: OptimizationOutput) -> None:
        """Keep the best confident run; otherwise keep the first refusal seen."""
        best = self.state.best_result
        if out.confident:
            if best is None or not best.confident or \
                    (out.best_achieved_titer or -1) > (best.best_achieved_titer or -1):
                self.state.best_result = out
        elif best is None:
            self.state.best_result = out

    @staticmethod
    def _summarize(out: OptimizationOutput, model: str, proposer: str) -> dict:
        if not out.confident:
            return {"confident": False, "refusal_reason": out.refusal_reason,
                    "detail": out.selection_rationale, "model": model, "proposer": proposer}
        rounds = [
            {"round": r.round_index, "fit_target_r2": r.fit.target_species_r2,
             "model_vs_oracle_r2": r.model_vs_oracle_r2,
             "achieved_peak_titer": r.achieved_peak_titer, "augmented": r.augmented_training}
            for r in out.rounds
        ]
        bc = out.best_candidate
        summary = {
            "confident": True,
            "model": model, "proposer": proposer,
            "best_achieved_titer": out.best_achieved_titer,
            "baseline_titer": out.baseline_titer,
            "improvement": out.improvement,
            "best_knobs": None if bc is None else bc.knobs(),
            "trajectory": out.convergence.titer_trajectory if out.convergence else [],
            "convergence_reason": out.convergence.reason if out.convergence else None,
            "converged": out.convergence.converged if out.convergence else False,
            "rounds": rounds,
        }
        if out.oracle_search is not None:
            r = out.oracle_search
            summary["oracle_search"] = {
                "true_box_max": r.best_titer, "n_oracle_evals": r.n_oracle_evals,
                "improved_over_loop": r.improved_over_loop,
                "knobs_on_boundary": r.knobs_on_boundary,
            }
        return summary

    # --- terminator --------------------------------------------------------
    def submit_optimization(self, payload: dict) -> dict:
        self.state.tool_calls += 1
        if self.state.submitted:
            return {"error": "already_submitted"}
        if self.state.best_result is None:
            return {"error": "no_loop_run",
                    "detail": "call run_optimization_loop before submitting"}
        self.state.narration = payload if isinstance(payload, dict) else {}
        self.state.submitted = True
        return {"ok": True}

    def dispatch(self) -> dict[str, Any]:
        return {
            "get_experiment": self.get_experiment,
            "get_box": self.get_box,
            "get_levers": self.get_levers,
            "get_skill": self.get_skill,
            "run_optimization_loop": self.run_optimization_loop,
            "submit_optimization": self.submit_optimization,
        }


def make_optimize_tools(
    training_data: pd.DataFrame,
    box: Box,
    simulator: Simulator,
    *,
    baseline_titer: float | None = None,
    objective_species: str = "P",
    v0: float = 10.0,
    levers: list[dict] | None = None,
) -> OptimizeToolBundle:
    return OptimizeToolBundle(
        training_data=training_data, box=box, simulator=simulator,
        baseline_titer=baseline_titer, objective_species=objective_species, v0=v0,
        levers=levers or [])
