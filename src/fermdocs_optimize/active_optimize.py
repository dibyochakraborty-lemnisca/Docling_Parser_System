"""Active-learning optimization (EGO-style) with the oracle spent only at the optimum.

    discover model on DATA (held-out scored) until peak R2 >= target
        │
        ▼
    optimize the model (scipy); if the optimum pins to a box edge, push the
    edge outward and re-search the model -> proposed optimum + predicted titer
        │
        ▼
    verify ONLY that point (+ a few neighbors) on the ORACLE
        │
   error = |predicted - oracle| at the optimum
        │
   < threshold ?  ── yes ─►  done: return the oracle-verified optimum
        │ no
        ▼
   fold the oracle point(s) into the data, widen the data box, re-discover
        ▲___________________________________________________________________│

The oracle is never used to score discovery rounds — only to verify the proposed
optimum. That makes each oracle call land where it changes the answer, which is
the only affordable shape on a real process (one call = one real experiment).
"""
from __future__ import annotations

import logging
import os

import numpy as np
import pandas as pd
from pydantic import BaseModel

from fermdocs_optimize import evaluate as ev
from fermdocs_optimize.discovery.candidate_model import CandidateModel
from fermdocs_optimize.discovery.loop import discover_model_from_data
from fermdocs_optimize.discovery.proposers import TemplateProposer
from fermdocs_optimize.discovery.spec import DiscoveryReport
from fermdocs_optimize.schema import KNOB_NAMES, Box, Candidate
from fermdocs_optimize.scipy_search import scipy_global_search
from fermdocs_optimize.search_space import box_from_data
from fermdocs_optimize.simulators.base import Simulator
from fermdocs_optimize.simulators.model_backed import ModelSimulator

log = logging.getLogger(__name__)

_WIDE = ("X", "S", "P", "M", "V")

# DE thoroughness for the (now cheap, ODE-guarded) MODEL search. Tunable so you
# can trade search depth for speed WITHOUT editing code; defaults preserve the
# original exploration. The ODE step-guard makes each eval cheap, so these stay
# high rather than being cut to work around stiff integrations.
_DE_MAXITER = int(os.environ.get("FERMDOCS_OPTIMIZE_DE_MAXITER", "30"))
_DE_POPSIZE = int(os.environ.get("FERMDOCS_OPTIMIZE_DE_POPSIZE", "15"))


class OuterIteration(BaseModel):
    iteration: int
    discovery: DiscoveryReport
    box: dict[str, list[float]]          # the search box this round (agent-decided)
    proposed_knobs: dict[str, float]
    predicted_titer: float               # model's prediction at its optimum
    oracle_titer: float                  # oracle truth at that point
    error: float                         # |predicted - oracle|
    converged: bool
    n_points_added: int
    knobs_on_boundary: dict[str, str] = {}  # which knobs sat on the FINAL search edge
    n_box_expansions: int = 0               # times the box was grown to chase an out-of-bounds optimum


class ActiveOptimizeReport(BaseModel):
    converged: bool
    n_outer: int
    best_knobs: dict[str, float]         # recommended optimum (best oracle-verified)
    oracle_titer: float                  # its oracle-verified titer (the number to trust)
    predicted_titer: float               # the model's prediction there
    error: float
    best_spec_name: str
    best_equations: list[str]
    best_fitted_params: dict[str, float]
    iterations: list[OuterIteration]
    n_oracle_evals: int
    batches_added: int
    knobs_on_boundary: dict[str, str] = {}  # of the recommended point, vs its search box


def _perturb(c: Candidate, box: Box, rng, n: int, frac: float = 0.04) -> list[Candidate]:
    """A few small jitters of `c`, clipped to the box — a tiny local DoE so the
    refit gets gradient information around the optimum, not just one point."""
    out = []
    for _ in range(n):
        knobs = {}
        for k in KNOB_NAMES:
            lb, ub = getattr(box, k)
            span = max(ub - lb, 1e-12)
            knobs[k] = float(np.clip(getattr(c, k) + rng.normal(0, frac * span), lb, ub))
        out.append(Candidate(**knobs))
    return out


def _apply_floors(lb: float, ub: float, knob: str) -> tuple[float, float]:
    """Universal physical sanity the data can't override (mirrors box_from_data):
    no negative knobs; malt_frac is a fraction so it cannot exceed 1."""
    lb = max(lb, 0.0)
    if knob == "malt_frac":
        ub = min(ub, 1.0)
    return lb, ub


def _grow_box(box: Box, edges: dict[str, str], physical: Box | None, grow: float) -> Box:
    """Push the box outward on the knobs whose optimum pinned to an edge. Each
    pinned edge moves out by `grow` * current span, then clipped to the floors and
    (if given) the physical hard limits. Returns a new Box equal to the input when
    no pinned edge could actually move (all already at a cap) — the caller uses that
    equality to stop expanding."""
    bounds: dict[str, tuple[float, float]] = {}
    for k in KNOB_NAMES:
        lb, ub = getattr(box, k)
        span = max(ub - lb, 1e-9)
        side = edges.get(k)
        if side == "upper":
            ub = ub + grow * span
        elif side == "lower":
            lb = lb - grow * span
        lb, ub = _apply_floors(lb, ub, k)
        if physical is not None:
            plb, pub = getattr(physical, k)
            lb, ub = max(lb, plb), min(ub, pub)
        if lb > ub:
            lb = ub
        bounds[k] = (lb, ub)
    return Box(**bounds)


def _search_beyond_bounds(
    model, box: Box, physical: Box | None, *,
    objective_species: str, v0: float, seed: int,
    max_expansions: int, grow: float,
):
    """Optimize the discovered MODEL (cheap, no oracle) and let the optimum escape
    a box the agent drew too tight. A best point pinned to a box edge means the
    objective was still climbing outward when it ran out of room — so push those
    edges out and re-search. Repeat until the optimum is interior, no pinned edge
    can grow (a physical cap or floor), or `max_expansions` is hit. The oracle is
    NEVER touched here; the caller verifies the single final point once.

    Returns (final OracleSearchReport, the box that produced it, n_expansions)."""
    sim = ModelSimulator(model)
    cur = box
    n_exp = 0
    search = scipy_global_search(
        sim, cur, method="de", objective_species=objective_species,
        v0=v0, maxiter=_DE_MAXITER, popsize=_DE_POPSIZE, seed=seed)
    while search.knobs_on_boundary and n_exp < max_expansions:
        pushed = dict(search.knobs_on_boundary)
        grown = _grow_box(cur, pushed, physical, grow)
        if grown == cur:  # every pinned edge already at a cap — can't explore further
            log.info("expand: optimum pinned at %s but capped; stop expanding",
                     list(pushed))
            break
        cur = grown
        n_exp += 1
        search = scipy_global_search(
            sim, cur, method="de", objective_species=objective_species,
            v0=v0, maxiter=_DE_MAXITER, popsize=_DE_POPSIZE, seed=seed)
        log.info("expand %d: pushed %s -> model max %.2f (boundary now %s)",
                 n_exp, list(pushed), search.best_titer,
                 list(search.knobs_on_boundary) or "interior")
    return search, cur, n_exp


def _relabel(sim_df: pd.DataFrame, start_id: int) -> pd.DataFrame:
    """Renumber a simulator output's batches to fresh ids and keep the wide schema."""
    ids = sorted(sim_df["batch"].unique(),
                 key=lambda x: int("".join(ch for ch in str(x) if ch.isdigit()) or 0))
    mapping = {old: start_id + i for i, old in enumerate(ids)}
    df = sim_df.copy()
    df["batch"] = df["batch"].map(mapping)
    return df[["batch", "t", *_WIDE]]


def active_optimize(
    *,
    data: pd.DataFrame,
    simulator: Simulator,
    physical: Box | None = None,
    proposer_factory=None,
    objective_species: str = "P",
    v0: float = 10.0,
    target_peak_r2: float = 0.8,
    inner_max_rounds: int = 6,
    holdout: float = 0.3,
    error_threshold: float = 5.0,
    max_outer: int = 4,
    n_neighbors: int = 3,
    box_margin: float = 0.0,
    box_fn=None,
    max_expansions: int = 4,
    expand_grow: float = 0.5,
    seed: int = 7,
) -> tuple[ActiveOptimizeReport, pd.DataFrame]:
    """Run the active-learning loop. Returns (report, new_batches) where
    new_batches are the oracle-verified batches appended across the run (for the
    caller to persist to train.csv)."""
    proposer_factory = proposer_factory or (lambda: TemplateProposer())
    rng = np.random.default_rng(seed)
    work = data.copy()
    new_batches: list[pd.DataFrame] = []
    iterations: list[OuterIteration] = []
    n_oracle = 0
    # track the best oracle-verified point across iterations
    best = None  # (oracle_titer, knobs, predicted, err, spec, model)

    for it in range(max_outer):
        # The agent decides the search box from the (growing) data when a box_fn
        # is given; it may extend beyond observed ranges (the oracle verify is the
        # safety net). Fall back to the data envelope if it declines/fails.
        box = None
        if box_fn is not None:
            try:
                box = box_fn(work)
            except Exception:  # noqa: BLE001
                box = None
        if box is None:
            box = box_from_data(work, margin=box_margin, physical=physical)
        rep = discover_model_from_data(
            data=work, proposer=proposer_factory(), max_rounds=inner_max_rounds,
            holdout=holdout, seed=seed, v0=v0, target_peak_r2=target_peak_r2)
        if rep.best_spec is None:
            log.warning("outer %d: discovery produced no compilable model", it)
            break

        model = CandidateModel(rep.best_spec)
        model.fit(work)
        # Optimize the model and let the optimum escape the box if it pins to an
        # edge (the true max may lie beyond the range the agent drew). Expansion
        # runs on the cheap model only; the box that produced the optimum is what
        # we record and verify.
        search, box, n_exp = _search_beyond_bounds(
            model, box, physical, objective_species=objective_species, v0=v0,
            seed=seed, max_expansions=max_expansions, grow=expand_grow)
        c = search.best_candidate
        predicted = float(search.best_titer)
        if n_exp:
            log.info("outer %d: expanded box %d time(s) to chase an out-of-bounds optimum",
                     it, n_exp)

        # verify ONLY the optimum (+ neighbors) on the oracle — one batched call
        cands = [c, *_perturb(c, box, rng, n_neighbors)]
        sim_df = simulator.simulate(cands, v0=v0)
        n_oracle += len(cands)
        peaks = ev._ordered_peaks(ev.peak_titer_per_batch(sim_df, objective_species), len(cands))
        oracle_titer = float(peaks[0])
        err = abs(predicted - oracle_titer)
        converged = err <= error_threshold

        iterations.append(OuterIteration(
            iteration=it, discovery=rep,
            box={k: [round(v, 5) for v in getattr(box, k)] for k in KNOB_NAMES},
            proposed_knobs={k: round(getattr(c, k), 5) for k in KNOB_NAMES},
            predicted_titer=round(predicted, 3), oracle_titer=round(oracle_titer, 3),
            error=round(err, 3), converged=converged,
            n_points_added=(0 if converged else len(cands)),
            knobs_on_boundary=dict(search.knobs_on_boundary),
            n_box_expansions=n_exp))
        log.info("outer %d '%s': predicted %.2f, oracle %.2f, err %.2f%s",
                 it, rep.best_spec.name, predicted, oracle_titer, err,
                 "  CONVERGED" if converged else "")

        if best is None or oracle_titer > best[0]:
            best = (oracle_titer, c, predicted, err, rep.best_spec, model,
                    dict(search.knobs_on_boundary))

        if converged:
            break

        # fold the verified point(s) into the data and re-discover with a wider box
        start = int(pd.to_numeric(work["batch"], errors="coerce").max()) + 1
        labeled = _relabel(sim_df, start)
        work = pd.concat([work, labeled], ignore_index=True)
        new_batches.append(labeled)

    oracle_titer, c, predicted, err, spec, model, boundary = best
    eqs = [f"{k} = {v}" for k, v in spec.aux.items()] + \
          [f"d{k}/dt = {v}" for k, v in spec.odes.items()]
    report = ActiveOptimizeReport(
        converged=bool(iterations and iterations[-1].converged),
        n_outer=len(iterations),
        best_knobs={k: round(getattr(c, k), 5) for k in KNOB_NAMES},
        oracle_titer=round(oracle_titer, 3), predicted_titer=round(predicted, 3),
        error=round(err, 3), best_spec_name=spec.name, best_equations=eqs,
        best_fitted_params={k: round(v, 5) for k, v in model.fitted_params.items()},
        iterations=iterations, n_oracle_evals=n_oracle,
        batches_added=sum(int(b["batch"].nunique()) for b in new_batches),
        knobs_on_boundary=boundary)
    new_df = pd.concat(new_batches, ignore_index=True) if new_batches else pd.DataFrame()
    return report, new_df
