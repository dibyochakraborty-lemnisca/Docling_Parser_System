"""Discover a lever->titer equation from the real runs, validate it, optimize it.

The levers are whatever THIS experiment actually varied — design factors pulled
from sheet metadata (``dossier["run_conditions"]``: nitrogen source, feed conc,
timing) plus varying observation initial conditions — discovered by
``lever_discovery``, never a hardcoded knob list. Numeric levers enter the
equation directly; categorical ones (e.g. nitrogen source) are one-hot encoded.

Two model families, tried in order (per the chosen design):

  1. MECHANISTIC (preferred) — kinetic ODEs fit to the run trajectories, gated
     by leave-run-out cross-validation. Biology-grounded and extrapolates, but
     needs identifiable dynamics; on sparse fed-batch data it usually can't be
     identified and the gate refuses. Delegated to the mechanistic discovery
     that already exists (fermdocs_recommend.mech_discovery), imported softly.

  2. SURROGATE (fallback) — an LLM (or template) proposes/iterates a STATIC
     algebraic surface peak_titer = f(knobs), fit by least-squares across runs,
     gated by the same leave-run-out CV. Empirical, but tractable on ~15 points.

The winner (whichever clears the CV gate with higher held-out R^2) is optimized
over the observed knob envelope to propose the best operating point. If neither
generalizes, we refuse honestly — the data can't support a predictive model.

No hardcoded domain values: the only constants are method thresholds (the CV
R^2 gate). Expectations come from the data.
"""

from __future__ import annotations

import logging
import os
from typing import Any, Protocol

import numpy as np
import sympy as sp
from pydantic import BaseModel, Field
from scipy.optimize import differential_evolution, least_squares

log = logging.getLogger(__name__)

# Held-out R^2 a model must clear to be trusted (method threshold, not a domain
# value). Below this the model doesn't generalize -> refuse rather than optimize
# a fit that only memorized the runs.
GATE_R2 = float(os.environ.get("FERMDOCS_DATA_EQUATION_GATE_R2", "0.5"))

# Data-relative sanity guards (method thresholds, NOT hardcoded domain ceilings).
# A prediction this many times above the OBSERVED objective maximum is implausible
# vs the data envelope — usually an upstream unit-canonicalization error (e.g. a
# mechanistic model fit on unit-corrupted data recommending 1809 g/L when runs
# peak near 150). We refuse rather than report it.
_IMPLAUSIBLE_FACTOR = float(os.environ.get("FERMDOCS_OPTIMIZE_IMPLAUSIBLE_FACTOR", "2.0"))
# Within-run working-volume change above this fraction => fed-batch operating mode
# (the batch ODE has no dilution term, so flag the mismatch).
_FEDBATCH_VOL_FRAC = 0.15

# Whitelisted math for the algebraic surrogate (no arbitrary callables).
_MATH = {"exp": sp.exp, "log": sp.log, "sqrt": sp.sqrt, "Abs": sp.Abs}


class EquationSpec(BaseModel):
    """A static algebraic equation peak_titer = expr(knobs; params)."""

    name: str
    expr: str
    params: dict[str, tuple[float, float, float]] = Field(default_factory=dict)
    # param name -> (init, lb, ub)
    notes: str | None = None


class EqProposer(Protocol):
    def propose(self, *, round_index: int, history: list, knob_names: list[str],
                data_summary: dict) -> EquationSpec: ...


# -----------------------------------------------------------------------------
# Compile / fit / cross-validate
# -----------------------------------------------------------------------------

def _compile(expr: str, knob_names: list[str], param_names: list[str]):
    """Safely turn an algebraic expression over knobs + params into a vectorized
    callable f(X[n,k], theta[p]) -> y[n]. Rejects any symbol outside the
    whitelist so the LLM can rewrite the equation but not run arbitrary code."""
    allowed = set(knob_names) | set(param_names)
    symtab = {n: sp.Symbol(n) for n in allowed}
    e = sp.sympify(expr, locals={**symtab, **_MATH})
    unknown = {s.name for s in e.free_symbols} - allowed
    if unknown:
        raise ValueError(f"equation uses unknown symbols {sorted(unknown)}")
    knob_syms = [symtab[k] for k in knob_names]
    param_syms = [symtab[p] for p in param_names]
    fn = sp.lambdify(knob_syms + param_syms, e, modules="numpy")

    def f(X: np.ndarray, theta: np.ndarray) -> np.ndarray:
        cols = [np.asarray(X[:, j], dtype=float) for j in range(X.shape[1])]
        out = fn(*cols, *[float(t) for t in theta])
        return np.broadcast_to(np.asarray(out, dtype=float), (X.shape[0],))

    return f


def _fit(f, X: np.ndarray, y: np.ndarray, p0: np.ndarray, bounds):
    res = least_squares(lambda p: f(X, p) - y, p0, bounds=bounds, max_nfev=2000)
    return res.x


def _r2(y: np.ndarray, pred: np.ndarray) -> float:
    ss_tot = float(((y - y.mean()) ** 2).sum())
    if ss_tot <= 0:
        return float("nan")
    return 1.0 - float(((y - pred) ** 2).sum()) / ss_tot


def _cv_r2(f, X: np.ndarray, y: np.ndarray, p0: np.ndarray, bounds) -> float:
    """Leave-one-run-out pooled held-out R^2."""
    n = len(y)
    if n < 3:
        return float("nan")
    preds = np.empty(n)
    idx = np.arange(n)
    for i in range(n):
        m = idx != i
        try:
            theta = _fit(f, X[m], y[m], p0, bounds)
            preds[i] = float(f(X[i:i + 1], theta)[0])
        except Exception:  # noqa: BLE001 — a fold that won't fit counts as a miss
            preds[i] = float("nan")
    ok = np.isfinite(preds)
    if ok.sum() < 3:
        return float("nan")
    return _r2(y[ok], preds[ok])


# -----------------------------------------------------------------------------
# Proposers
# -----------------------------------------------------------------------------

class TemplateEqProposer:
    """Deterministic forms of rising complexity: linear -> quadratic-per-knob ->
    with pairwise interactions. Used as the default and as the offline test
    proposer (no LLM)."""

    def propose(self, *, round_index, history, knob_names, data_summary) -> EquationSpec:
        forms = [self._linear, self._quadratic, self._interactions]
        return forms[min(round_index, len(forms) - 1)](knob_names)

    @staticmethod
    def _params(n: int) -> dict[str, tuple[float, float, float]]:
        return {f"p{i}": (0.0, -1e9, 1e9) for i in range(n)}

    def _linear(self, ks):
        terms = ["p0"] + [f"p{i+1}*{k}" for i, k in enumerate(ks)]
        return EquationSpec(name="linear", expr=" + ".join(terms),
                            params=self._params(len(ks) + 1))

    def _quadratic(self, ks):
        terms = ["p0"]
        i = 1
        for k in ks:
            terms += [f"p{i}*{k}", f"p{i+1}*{k}**2"]
            i += 2
        return EquationSpec(name="quadratic", expr=" + ".join(terms),
                            params=self._params(i))

    def _interactions(self, ks):
        terms = ["p0"]
        i = 1
        for k in ks:
            terms += [f"p{i}*{k}", f"p{i+1}*{k}**2"]
            i += 2
        for a in range(len(ks)):
            for b in range(a + 1, len(ks)):
                terms.append(f"p{i}*{ks[a]}*{ks[b]}")
                i += 1
        return EquationSpec(name="interactions", expr=" + ".join(terms),
                            params=self._params(i))


# -----------------------------------------------------------------------------
# Surrogate discovery + optimization
# -----------------------------------------------------------------------------

class EquationResult(BaseModel):
    family: str  # "mechanistic" | "surrogate"
    cleared: bool
    cv_r2: float
    spec_name: str | None = None
    expr: str | None = None
    best_knobs: dict[str, Any] = Field(default_factory=dict)
    predicted_peak: float | None = None
    levers: list[dict] = Field(default_factory=list)  # discovered levers (name/kind/source)
    # True when the optimum sits at the edge of the observed data: NOT a validated
    # optimum, but a signal that the best point may lie beyond the explored range.
    boundary_limited: bool = False
    on_boundary: dict[str, str] = Field(default_factory=dict)
    rationale: str = ""


def discover_surrogate(
    design, *, proposer: EqProposer | None = None, max_rounds: int = 4,
) -> tuple[EquationSpec, np.ndarray, float] | None:
    """Propose/iterate algebraic equations over the discovered-lever design,
    keep the one with the best leave-run-out CV R^2. The equation is fit over
    the design's feature columns (numeric levers + one-hot categorical levels),
    so it generalizes to whatever this experiment actually varied. Returns
    (spec, fitted_params, cv_r2) or None if nothing compiles/fits."""
    proposer = proposer or TemplateEqProposer()
    feats = design.feature_names
    X = design.X
    y = design.y
    summary = {"n_runs": len(y), "knobs": feats,
               "peak_range": [float(y.min()), float(y.max())]}
    history: list[dict] = []
    best = None  # (spec, theta, cv_r2)
    for rnd in range(max_rounds):
        try:
            spec = proposer.propose(round_index=rnd, history=history,
                                    knob_names=feats, data_summary=summary)
            pnames = list(spec.params)
            f = _compile(spec.expr, feats, pnames)
            p0 = np.array([spec.params[p][0] for p in pnames], dtype=float)
            lb = np.array([spec.params[p][1] for p in pnames], dtype=float)
            ub = np.array([spec.params[p][2] for p in pnames], dtype=float)
            theta = _fit(f, X, y, p0, (lb, ub))
            cv = _cv_r2(f, X, y, p0, (lb, ub))
        except Exception as exc:  # noqa: BLE001
            history.append({"round": rnd, "error": str(exc)})
            continue
        history.append({"round": rnd, "name": spec.name, "cv_r2": cv})
        if best is None or (np.isfinite(cv) and cv > best[2]):
            best = (spec, theta, cv)
    return best


def optimize_surrogate(
    design, spec: EquationSpec, theta: np.ndarray,
) -> tuple[dict[str, Any], float, dict[str, str]]:
    """Maximize the fitted equation over the OBSERVED lever envelope. Numeric
    levers vary continuously within their observed [min, max]; categorical
    levers are held to one of the category COMBINATIONS actually run (we never
    invent an unseen mix of categories — that's extrapolation). Returns
    (best_knobs, predicted_peak, levers_on_boundary), where best_knobs maps each
    lever to its best numeric value or chosen category."""
    feats = design.features
    X = design.X
    f = _compile(spec.expr, design.feature_names, list(spec.params))
    n = len(feats)

    num_idx = [i for i, ft in enumerate(feats) if ft.kind == "numeric"]
    oh_idx = [i for i, ft in enumerate(feats) if ft.kind == "onehot"]
    lo = X.min(axis=0)
    hi = X.max(axis=0)

    # Observed one-hot combinations: the distinct categorical assignments that
    # actually occurred across runs. Each is a fixed sub-vector over oh columns.
    if oh_idx:
        combos = sorted({tuple(X[r, oh_idx].tolist()) for r in range(X.shape[0])})
    else:
        combos = [()]

    # Numeric dims we actually optimize (skip ones that don't vary).
    nondegen = [i for i in num_idx if hi[i] - lo[i] > 1e-12]

    def evaluate(num_vals: dict[int, float], oh_vec: tuple) -> float:
        x = np.empty(n)
        for i in num_idx:
            x[i] = num_vals.get(i, lo[i])
        for j, i in enumerate(oh_idx):
            x[i] = oh_vec[j]
        return float(f(x.reshape(1, -1), theta)[0])

    best_val = -np.inf
    best_x = np.array([lo[i] if i in num_idx else 0.0 for i in range(n)])
    for oh_vec in combos:
        if nondegen:
            res = differential_evolution(
                lambda xs, _oh=oh_vec: -evaluate(
                    {i: xs[nondegen.index(i)] for i in nondegen}, _oh),
                [(float(lo[i]), float(hi[i])) for i in nondegen],
                seed=7, maxiter=60, tol=1e-6, polish=True,
            )
            num_vals = {i: float(res.x[nondegen.index(i)]) for i in nondegen}
        else:
            num_vals = {}
        val = evaluate(num_vals, oh_vec)
        if val > best_val:
            best_val = val
            xb = np.empty(n)
            for i in num_idx:
                xb[i] = num_vals.get(i, lo[i])
            for j, i in enumerate(oh_idx):
                xb[i] = oh_vec[j]
            best_x = xb

    # Read the chosen feature vector back into lever-named settings.
    best_knobs: dict[str, Any] = {}
    for lev in design.levers:
        if lev.kind == "numeric":
            i = next(k for k, ft in enumerate(feats) if ft.lever == lev.name)
            best_knobs[lev.name] = round(float(best_x[i]), 4)
        else:
            group = design.onehot_groups.get(lev.name, {})
            chosen = None
            for ft in feats:
                if ft.lever == lev.name and best_x[feats.index(ft)] >= 0.5:
                    chosen = ft.category
                    break
            best_knobs[lev.name] = chosen if chosen is not None else (
                lev.categories[0] if lev.categories else None)

    span = np.maximum(hi - lo, 1e-12)
    on_boundary: dict[str, str] = {}
    for i in nondegen:
        lever_name = feats[i].lever
        if abs(best_x[i] - lo[i]) <= 0.02 * span[i]:
            on_boundary[lever_name] = "lower"
        elif abs(best_x[i] - hi[i]) <= 0.02 * span[i]:
            on_boundary[lever_name] = "upper"
    return best_knobs, float(best_val), on_boundary


# -----------------------------------------------------------------------------
# Mechanistic attempt (preferred) — reuse the existing ODE discovery + CV
# -----------------------------------------------------------------------------

def _lever_base_channel(lev) -> str:
    """The observation channel a lever refers to. A derived initial-condition
    lever 'substrate_g_l.initial' refers to channel 'substrate_g_l'; a metadata
    lever refers to its own name (which may or may not be a measured channel)."""
    return lev.name[:-len(".initial")] if lev.name.endswith(".initial") else lev.name


def _mechanistic_attempt(obs_df, levers, objective, *, proposer=None):
    """Discover a coupled kinetic ODE over ALL relevant measured variables (not a
    fixed LABS species set), gated by leave-runs-out held-out R^2 on the
    objective. If it generalizes AND at least one controllable lever is the
    initial condition of a modeled state, optimize those initial conditions by
    simulating the fitted ODE to peak objective. Returns an EquationResult or None
    (no multi-state model / nothing generalizes / no state-lever to move -> fall
    back to the surrogate, which can also handle categorical levers). Never raises."""
    try:
        from fermdocs_optimize.discovery.general_mech import (
            discover_general_mechanistic,
            optimize_initial_conditions,
        )

        fit = discover_general_mechanistic(obs_df, objective=objective,
                                           proposer=proposer, gate_r2=GATE_R2)
        if fit is None:
            return None

        # Which discovered numeric levers ARE the initial condition of a modeled
        # state? Those are the ones the ODE can actually be optimized over.
        var_to_sym = {v: s for s, v in fit.sym_to_var.items()}
        lever_states: dict[str, tuple[float, float]] = {}
        sym_to_lever: dict[str, str] = {}
        for lev in levers:
            if lev.kind != "numeric":
                continue
            sym = var_to_sym.get(_lever_base_channel(lev))
            rng = lev.observed_range
            if sym is not None and rng is not None:
                lever_states[sym] = rng
                sym_to_lever[sym] = lev.name
        if not lever_states:
            return None  # ODE generalizes but no controllable state-lever to move

        best_y0, pred, on_b_syms = optimize_initial_conditions(fit, lever_states)
        best_knobs = {sym_to_lever[s]: v for s, v in best_y0.items()}
        on_boundary = {sym_to_lever[s]: side for s, side in on_b_syms.items()}
        # Render the ODEs over the real variable names for the model log.
        odes_named = {fit.sym_to_var[s]: fit.spec.odes[s] for s in fit.states}
        modeled = ", ".join(fit.sym_to_var[s] for s in fit.states)
        return EquationResult(
            family="mechanistic", cleared=True, cv_r2=round(fit.objective_r2, 3),
            spec_name=fit.spec.name, expr=str(odes_named),
            best_knobs=best_knobs, predicted_peak=round(float(pred), 3),
            on_boundary=on_boundary,
            rationale=(f"Mechanistic ODE '{fit.spec.name}' over the measured "
                       f"variables ({modeled}) generalized (held-out objective "
                       f"R^2={fit.objective_r2:.2f}); optimized the controllable "
                       "initial conditions by simulating to peak."),
        )
    except Exception:  # noqa: BLE001 — mechanistic is best-effort; fall back
        log.exception("mechanistic attempt failed; falling back to surrogate")
        return None


def _observed_objective_max(obs_df, objective: str) -> float | None:
    """Max observed value of the objective across all runs (the data envelope's
    ceiling). Used to reject predictions that are implausible vs the data."""
    import pandas as pd

    df = obs_df[obs_df["variable"].astype(str) == objective]
    vals = pd.to_numeric(df["value"], errors="coerce").dropna()
    return float(vals.max()) if len(vals) else None


def _within_run_volume_change(obs_df) -> float:
    """Largest within-run relative change of working volume (max-min)/min across
    runs. >0 means volume moves during a run -> fed-batch (the batch ODE has no
    dilution term). Returns 0.0 if no volume channel is present."""
    import pandas as pd

    df = obs_df[obs_df["variable"].astype(str) == "volume_l"]
    if df.empty:
        return 0.0
    worst = 0.0
    for _, g in df.groupby("run_id"):
        v = pd.to_numeric(g["value"], errors="coerce").dropna()
        if len(v) >= 2 and float(v.min()) > 0:
            worst = max(worst, (float(v.max()) - float(v.min())) / float(v.min()))
    return worst


def _apply_sanity_guards(result: EquationResult, obs_df, objective: str) -> EquationResult:
    """Physical/operating-mode sanity on a cleared optimization result:

      1. Reject predictions implausible vs the OBSERVED envelope (data-relative,
         not a hardcoded ceiling) — catches models fit on unit-corrupted data.
      2. Mark boundary-sitting optima as insufficient-data, not validated optima.
      3. Flag a fed-batch operating mode the batch ODE doesn't represent.
    """
    if not result.cleared or result.predicted_peak is None:
        return result
    obs_max = _observed_objective_max(obs_df, objective)
    if obs_max is not None and obs_max > 0 and result.predicted_peak > _IMPLAUSIBLE_FACTOR * obs_max:
        return result.model_copy(update={
            "cleared": False, "best_knobs": {}, "predicted_peak": None,
            "on_boundary": {}, "boundary_limited": False,
            "rationale": (
                f"Predicted peak {result.predicted_peak:g} exceeds "
                f"{_IMPLAUSIBLE_FACTOR:g}x the observed maximum ({obs_max:g}) — "
                "implausible against the data envelope (commonly an upstream "
                "unit-canonicalization error). Refusing rather than report a "
                "physically implausible optimum; re-check units and refit."),
        })
    rationale = result.rationale
    boundary_limited = bool(result.on_boundary)
    if boundary_limited:
        rationale += (
            f" NOTE: the optimum sits at the edge of the observed data "
            f"({result.on_boundary}) — treat this as insufficient data in that "
            "region, not a validated optimum; the true best may lie beyond the "
            "explored range and needs a new experiment to confirm.")
    if result.family == "mechanistic" and _within_run_volume_change(obs_df) > _FEDBATCH_VOL_FRAC:
        rationale += (
            " CAVEAT: working volume changes materially within runs (fed-batch), "
            "but the discovered ODE is a batch model with no dilution term; its "
            "dynamics may not match the actual operating mode.")
    return result.model_copy(update={"rationale": rationale,
                                     "boundary_limited": boundary_limited})


def discover_and_optimize(
    obs_df, *, dossier: dict | None = None, objective: str | None = None,
    surrogate_proposer: EqProposer | None = None, mech_proposer=None,
) -> EquationResult:
    """Mechanistic-first, surrogate-fallback discovery + optimization over the
    experiment's OWN levers (discovered from run_conditions metadata + varying
    observation channels — never a hardcoded knob list). Returns the winning
    EquationResult, or a refusal (cleared=False) when no lever varied or no model
    generalizes (the honest outcome on data too sparse / too confounded to fit)."""
    from fermdocs_optimize.lever_discovery import (
        DEFAULT_OBJECTIVE,
        build_design,
        discover_levers,
    )

    objective = objective or DEFAULT_OBJECTIVE
    levers = discover_levers(dossier, obs_df, objective=objective)
    lever_dump = [{"name": lev.name, "kind": lev.kind, "source": lev.source}
                  for lev in levers]
    if not levers:
        return EquationResult(
            family="surrogate", cleared=False, cv_r2=float("nan"), levers=lever_dump,
            rationale="no controllable lever varied across runs; nothing to optimize.",
        )

    # Mechanistic (biology-grounded) first: discover a coupled ODE over ALL the
    # measured variables and optimize the controllable initial conditions. Falls
    # through to the surrogate when the data can't identify a generalizing ODE or
    # the only levers are categorical (which the ODE can't represent).
    mech = _mechanistic_attempt(obs_df, levers, objective, proposer=mech_proposer)
    if mech is not None:
        mech.levers = lever_dump
        return _apply_sanity_guards(mech, obs_df, objective)

    design = build_design(levers, obs_df, objective=objective)
    if design is None or len(design.y) < 3:
        return EquationResult(
            family="surrogate", cleared=False, cv_r2=float("nan"), levers=lever_dump,
            rationale=("too few runs with both lever values and an outcome to fit "
                       "a generalizable model."),
        )

    found = discover_surrogate(design, proposer=surrogate_proposer)
    if found is None:
        return EquationResult(family="surrogate", cleared=False, cv_r2=float("nan"),
                              levers=lever_dump,
                              rationale="no surrogate equation compiled/fit")
    spec, theta, cv = found
    if not np.isfinite(cv) or cv < GATE_R2:
        return EquationResult(
            family="surrogate", cleared=False, levers=lever_dump,
            cv_r2=round(float(cv), 3) if np.isfinite(cv) else float("nan"),
            spec_name=spec.name, expr=spec.expr,
            rationale=(f"Best equation '{spec.name}' did not generalize "
                       f"(leave-run-out R^2={cv:.2f} < {GATE_R2}); refusing rather "
                       "than optimize a fit that only memorized the runs."),
        )
    best_knobs, pred, on_b = optimize_surrogate(design, spec, theta)
    result = EquationResult(
        family="surrogate", cleared=True, cv_r2=round(float(cv), 3), levers=lever_dump,
        spec_name=spec.name, expr=spec.expr, best_knobs=best_knobs,
        predicted_peak=round(float(pred), 3), on_boundary=on_b,
        rationale=(f"Static equation '{spec.name}' over the discovered levers "
                   f"({', '.join(lev.name for lev in levers)}) generalized "
                   f"(leave-run-out R^2={cv:.2f}); optimized over the observed envelope."),
    )
    return _apply_sanity_guards(result, obs_df, objective)
