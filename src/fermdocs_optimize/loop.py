"""The deterministic optimization loop.

Closed-loop, active-learning optimization of product titer:

    fit the model on data  ─►  propose knobs maximizing predicted titer
       ▲                                  │
       │                                  ▼
    augment training  ◄── evaluate ◄── simulate proposals on the oracle
    (if model wrong)        │
                            ▼
                       converged?  (Δ best-titer < threshold)  ─► stop

This module is pure orchestration over injected collaborators (model, proposer,
simulator) — no LLM, fully testable with a stub simulator. The agentic shell
(`agent.py`) wraps this with objective interpretation and narration.
"""
from __future__ import annotations

import logging

import pandas as pd

from fermdocs_optimize import evaluate as ev
from fermdocs_optimize.models.base import PredictiveModel
from fermdocs_optimize.oracle_search import oracle_global_search
from fermdocs_optimize.proposers.base import Proposer
from fermdocs_optimize.schema import (
    Candidate,
    ConvergenceReport,
    FitReport,
    OptimizationInput,
    OptimizationOutput,
    RoundResult,
)
from fermdocs_optimize.simulators.base import Simulator

log = logging.getLogger(__name__)


def run_optimization(
    *,
    training_data: pd.DataFrame,
    model: PredictiveModel,
    proposer: Proposer,
    simulator: Simulator,
    spec: OptimizationInput,
    baseline_titer: float | None = None,
) -> OptimizationOutput:
    """Run the model-guided loop; when `spec.oracle_search` is set, follow it with
    an oracle-direct global search so the result is the TRUE within-box maximum,
    not the surrogate's local plateau."""
    out = _active_learning(
        training_data=training_data, model=model, proposer=proposer,
        simulator=simulator, spec=spec, baseline_titer=baseline_titer)
    if spec.oracle_search:
        out = _augment_with_oracle(out, simulator=simulator, spec=spec,
                                   baseline=baseline_titer)
    return out


def _active_learning(
    *,
    training_data: pd.DataFrame,
    model: PredictiveModel,
    proposer: Proposer,
    simulator: Simulator,
    spec: OptimizationInput,
    baseline_titer: float | None = None,
) -> OptimizationOutput:
    """The closed loop: fit → propose → simulate → evaluate → converge.

    `training_data`: seed batches (wide schema batch,t,X,S,P,M,V).
    `baseline_titer`: titer of the starting/seed point, for the improvement metric.
    """
    train = training_data.copy()
    rounds: list[RoundResult] = []
    trajectory: list[float] = []
    target = spec.objective_species

    # Model transparency: lead the log with the governing equations (if the model
    # publishes a card), then append a fit log per round.
    model_log: list[dict] = []
    card = getattr(model, "model_card", None)
    if callable(card):
        try:
            model_log.append(card())
        except Exception:  # noqa: BLE001 — a missing card never breaks the loop
            pass

    for r in range(spec.max_rounds):
        # 1. FIT (data only; never the oracle's true params)
        r2 = model.fit(train)
        target_r2 = r2.get(target, float("nan"))
        fit_log = getattr(model, "fit_log", None)
        if callable(fit_log):
            try:
                entry = fit_log(r2, int(train["batch"].nunique()))
                entry["title"] = f"Round {r}: " + entry.get("title", "fit")
                model_log.append(entry)
            except Exception:  # noqa: BLE001
                pass
        fit = FitReport(
            n_train_batches=int(train["batch"].nunique()),
            r2_by_species={k: round(v, 4) for k, v in r2.items()},
            fitted_params=model.fitted_params,
            target_species_r2=round(target_r2, 4),
        )

        # 2. PROPOSE knobs maximizing predicted peak titer
        proposals = proposer.propose(model, spec.box, k=spec.proposals_per_round, v0=spec.v0)

        # 3. SIMULATE proposals on the ground-truth oracle
        sim_df = simulator.simulate(proposals, v0=spec.v0)

        # 4. EVALUATE: best achieved titer + model-vs-oracle agreement
        best_cand, achieved = ev.best_achieved(sim_df, proposals, species=target)
        mvo_r2 = ev.model_vs_oracle_r2(model, proposals, sim_df, v0=spec.v0, species=target)
        trajectory.append(achieved)

        # 4b. ACTIVE LEARNING: if the model was wrong where we sampled, fold it in
        augment = mvo_r2 < spec.good_fit_r2
        if augment:
            train = ev.append_training(train, ev.sim_to_training_rows(sim_df))

        rounds.append(RoundResult(
            round_index=r, fit=fit, proposals=proposals, best_candidate=best_cand,
            achieved_peak_titer=round(achieved, 3),
            model_vs_oracle_r2=round(mvo_r2, 4), augmented_training=augment,
            n_training_after=int(train["batch"].nunique()),
        ))
        log.info("round %d: fit P R2=%.3f  best titer=%.2f  model/oracle R2=%.3f  %s",
                 r, target_r2, achieved, mvo_r2, "augmented" if augment else "")

        # 5. CONVERGE on best-titer plateau
        done, delta = ev.converged(trajectory, spec.delta_titer_threshold)
        if done:
            return _finalize(rounds, trajectory, "delta_below_threshold", True,
                             delta, baseline_titer, model_log)

    return _finalize(rounds, trajectory, "max_rounds", False, None, baseline_titer, model_log)


def _finalize(rounds, trajectory, reason, converged_flag, delta, baseline,
              model_log=None) -> OptimizationOutput:
    # overall best across all rounds
    best_round = max(rounds, key=lambda rr: rr.achieved_peak_titer)
    best_titer = best_round.achieved_peak_titer
    best_cand = best_round.best_candidate
    conv = ConvergenceReport(reason=reason, converged=converged_flag,
                             titer_trajectory=[round(t, 3) for t in trajectory],
                             final_delta=None if delta is None else round(delta, 4))
    improvement = None if baseline is None else round(best_titer - baseline, 3)
    rationale = (
        f"Best simulated peak titer {best_titer:.1f} g/L at "
        f"biomass={best_cand.biomass:.2f}, total_sub={best_cand.total_sub:.1f}, "
        f"malt_frac={best_cand.malt_frac:.3f}, dilution={best_cand.dilution:.4f}. "
        f"Converged via {reason} over {len(rounds)} round(s)."
    )
    if baseline is not None:
        rationale += f" Improvement over baseline ({baseline:.1f}): {improvement:+.1f} g/L."
    return OptimizationOutput(
        confident=True, best_candidate=best_cand, best_achieved_titer=best_titer,
        baseline_titer=baseline, improvement=improvement, rounds=rounds,
        convergence=conv, selection_rationale=rationale, model_log=model_log or [],
    )


def _augment_with_oracle(out: OptimizationOutput, *, simulator: Simulator,
                         spec: OptimizationInput, baseline: float | None) -> OptimizationOutput:
    """Run the oracle-direct global search and fold its TRUE maximum into `out`.

    Promotes the result to the oracle's best whenever it beats the model-guided
    loop (or whenever the loop refused). Adds a boundary note when the optimum
    sits on the box edge — a signal the real optimum may lie outside the limits."""
    try:
        # Warm-start from the loop's best so the global search can only improve.
        warm = [out.best_candidate] if (out.confident and out.best_candidate) else None
        rep = oracle_global_search(
            simulator, spec.box, objective_species=spec.objective_species,
            v0=spec.v0, n_lhs=spec.n_lhs, refine_iters=spec.refine_iters,
            warm_start=warm)
    except Exception as exc:  # noqa: BLE001 — never let the polish step break the run
        log.exception("oracle global search failed")
        out.selection_rationale += f" [oracle search skipped: {exc}]"
        return out

    loop_best = out.best_achieved_titer if out.confident else None
    rep.improved_over_loop = loop_best is None or rep.best_titer > loop_best
    boundary = ""
    if rep.knobs_on_boundary:
        boundary = (f" [box-binding: {rep.knobs_on_boundary} at the box edge — the true "
                    "optimum may lie OUTSIDE these limits; widen var_params to test.]")

    if rep.improved_over_loop:
        improvement = None if baseline is None else round(rep.best_titer - baseline, 3)
        note = (
            f" Oracle global search ({rep.n_oracle_evals} simulator evals over the box) "
            f"found the true maximum {rep.best_titer:.1f} g/L at "
            f"biomass={rep.best_candidate.biomass:.2f}, total_sub={rep.best_candidate.total_sub:.1f}, "
            f"malt_frac={rep.best_candidate.malt_frac:.3f}, dilution={rep.best_candidate.dilution:.4f}"
            + (f" (+{improvement:.1f} over baseline)." if improvement is not None else ".")
        )
        rationale = (out.selection_rationale + note + boundary).strip()
        return OptimizationOutput(
            meta=out.meta, confident=True, best_candidate=rep.best_candidate,
            best_achieved_titer=rep.best_titer, baseline_titer=baseline,
            improvement=improvement, rounds=out.rounds, convergence=out.convergence,
            selection_rationale=rationale, model_log=out.model_log, oracle_search=rep,
        )

    # Loop already matched/beat the oracle search — confirm it.
    out.oracle_search = rep
    out.selection_rationale += (
        f" Oracle global search ({rep.n_oracle_evals} simulator evals) confirmed the "
        f"optimum: no better point found in the box ({rep.best_titer:.1f} g/L)." + boundary)
    return out


def refusal(reason: str, detail: str = "") -> OptimizationOutput:
    """Honest refusal (e.g. model can't fit, oracle unavailable)."""
    return OptimizationOutput(
        confident=False, refusal_reason=reason,
        selection_rationale=detail or f"Optimization refused: {reason}.",
    )
