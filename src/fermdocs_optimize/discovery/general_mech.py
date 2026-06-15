"""General mechanistic ODE discovery — over WHATEVER variables a dataset has.

The recommend-stage `mech_discovery` models exactly two LABS species (substrate
S, product P) and zeroes the rest. This module removes that assumption: the state
vector is built from every relevant time-varying channel the data actually
measured (biomass, substrate, product, byproducts, OD, ...), and the agent (or a
generic template family) writes a coupled ODE over those real variables. It reuses
the same safe symbolic compiler (`expr.compile_spec`, now state-agnostic) so the
agent rewrites EQUATIONS, never code.

Honesty, unchanged from the LABS loop:
  * Fit ONE shared structure across ALL runs (pool the points so params are
    identifiable on sparse trajectories).
  * Validate by holding out WHOLE RUNS (leave-runs-out): a structure that overfits
    scores worse held-out, so the loop converges to the simplest ODE that
    generalizes. The held-out R^2 on the OBJECTIVE is the gate. If nothing clears
    it, the caller refuses — the loop raises the ceiling, it cannot invent signal.

No hardcoded species, no reactor physics: a batch model (no feed/aeration) over
the discovered states. Conditions/constants are empty; every state gets an ODE.
"""
from __future__ import annotations

import json
import logging
import os
from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.integrate import odeint
from scipy.optimize import least_squares

from fermdocs_optimize.discovery.expr import ExprError, compile_spec
from fermdocs_optimize.discovery.spec import ModelSpec, ParamSpec

log = logging.getLogger(__name__)

_MXSTEP = int(os.environ.get("FERMDOCS_OPTIMIZE_ODE_MXSTEP", "500"))
# Cap the state vector so the ODE stays identifiable on sparse lab data. The
# objective plus the (K-1) channels best co-measured with it. Dropped channels
# are logged (no silent truncation).
_MAX_STATES = int(os.environ.get("FERMDOCS_GENERAL_MECH_MAX_STATES", "4"))
_MIN_ALIGNED = 3      # timepoints a run needs (after aligning states) to be usable
_MIN_RUNS = 3         # runs needed to fit + hold out


@dataclass
class StateRun:
    """One run's aligned multi-variable trajectory."""

    run_id: str
    t: np.ndarray                 # (n_t,)
    Y: np.ndarray                 # (n_t, n_states) observed, columns in state order
    y0: np.ndarray                # (n_states,) initial condition


def _sanitize(name: str, used: set[str]) -> str:
    base = "".join(ch if (ch.isalnum() or ch == "_") else "_" for ch in name)
    if not base or not (base[0].isalpha() or base[0] == "_"):
        base = "v_" + base
    sym = base
    i = 1
    while sym in used:
        sym = f"{base}_{i}"
        i += 1
    used.add(sym)
    return sym


def load_state_runs(
    obs_df: pd.DataFrame, *, objective: str, max_states: int = _MAX_STATES,
) -> tuple[list[StateRun], list[str], dict[str, str]]:
    """Build aligned multi-variable runs + the chosen state set.

    Returns (runs, state_syms, sym_to_var). State variables are the objective
    plus the channels most co-measured with it (capped at `max_states`), each
    time-varying and present (>= _MIN_ALIGNED aligned points) in >= _MIN_RUNS
    runs. `state_syms` are sympy-safe identifiers; `sym_to_var` maps each back to
    its observation variable name. Raises ValueError if the data can't support a
    multi-state model with the objective present."""
    df = obs_df.copy()
    for col in ("run_id", "variable", "time_h", "value"):
        if col not in df.columns:
            raise ValueError(f"observations missing required column {col!r}")
    df["value"] = pd.to_numeric(df["value"], errors="coerce")
    df["time_h"] = pd.to_numeric(df["time_h"], errors="coerce")
    df = df.dropna(subset=["value", "time_h"])

    # Per-run pivot var x time.
    pivots: dict[str, pd.DataFrame] = {}
    for run_id, g in df.groupby("run_id"):
        piv = g.pivot_table(index="time_h", columns="variable", values="value",
                            aggfunc="mean").sort_index()
        pivots[str(run_id)] = piv
    if not any(objective in p.columns for p in pivots.values()):
        raise ValueError(f"objective {objective!r} not measured in any run")

    # Candidate channels: time-varying, co-measured with the objective. Rank by
    # how many runs have >= _MIN_ALIGNED points with BOTH the channel and the
    # objective present (coverage), so we keep the most identifiable couplings.
    coverage: dict[str, int] = {}
    for var in {c for p in pivots.values() for c in p.columns}:
        if var == objective:
            continue
        n = 0
        for piv in pivots.values():
            if var in piv.columns and objective in piv.columns:
                aligned = piv[[var, objective]].dropna()
                if len(aligned) >= _MIN_ALIGNED and aligned[var].nunique() > 1:
                    n += 1
        if n >= _MIN_RUNS:
            coverage[var] = n
    ranked = sorted(coverage, key=lambda v: (coverage[v], v), reverse=True)
    chosen_vars = [objective] + ranked[: max(0, max_states - 1)]
    dropped = ranked[max(0, max_states - 1):]
    if dropped:
        log.info("general-mech: capping states at %d; modeling %s, dropping %s",
                 max_states, chosen_vars, dropped)

    # Build aligned runs over the chosen states (intersection timepoints).
    used: set[str] = set()
    sym_to_var = {_sanitize(v, used): v for v in chosen_vars}
    var_to_sym = {v: s for s, v in sym_to_var.items()}
    state_syms = [var_to_sym[v] for v in chosen_vars]

    runs: list[StateRun] = []
    for run_id, piv in pivots.items():
        if not all(v in piv.columns for v in chosen_vars):
            continue
        aligned = piv[chosen_vars].dropna().sort_index()
        if len(aligned) < _MIN_ALIGNED:
            continue
        t = aligned.index.to_numpy(float)
        Y = aligned.to_numpy(float)  # columns follow chosen_vars order
        runs.append(StateRun(str(run_id), t, Y, Y[0].copy()))

    if len(runs) < _MIN_RUNS:
        raise ValueError(
            f"need >= {_MIN_RUNS} runs with {chosen_vars} aligned; got {len(runs)}")
    return runs, state_syms, sym_to_var


# -----------------------------------------------------------------------------
# Generic, domain-agnostic proposers
# -----------------------------------------------------------------------------

def _affine_linear(states: list[str]) -> ModelSpec:
    """dX_i/dt = c_i + sum_j a_ij * X_j — a linear dynamical system."""
    params, odes = {}, {}
    for i, si in enumerate(states):
        terms = [f"c_{i}"]
        params[f"c_{i}"] = ParamSpec(init=0.0, lb=-1e3, ub=1e3)
        for j, sj in enumerate(states):
            p = f"a_{i}_{j}"
            params[p] = ParamSpec(init=0.0, lb=-10.0, ub=10.0)
            terms.append(f"{p}*{sj}")
        odes[si] = " + ".join(terms)
    return ModelSpec(name="affine_linear", params=params, aux={}, odes=odes,
                     notes="Linear dynamical system over all measured states.")


def _generalized_lv(states: list[str]) -> ModelSpec:
    """dX_i/dt = X_i*(r_i + sum_j b_ij * X_j) — generalized Lotka-Volterra; captures
    growth, consumption and product-coupling without naming the chemistry."""
    params, odes = {}, {}
    for i, si in enumerate(states):
        inner = [f"r_{i}"]
        params[f"r_{i}"] = ParamSpec(init=0.0, lb=-5.0, ub=5.0)
        for j, sj in enumerate(states):
            p = f"b_{i}_{j}"
            params[p] = ParamSpec(init=0.0, lb=-1.0, ub=1.0)
            inner.append(f"{p}*{sj}")
        odes[si] = f"{si}*(" + " + ".join(inner) + ")"
    return ModelSpec(name="generalized_lotka_volterra", params=params, aux={},
                     odes=odes, notes="Generalized Lotka-Volterra over all states.")


_TEMPLATES = [_affine_linear, _generalized_lv]


class GeneralTemplateProposer:
    """Deterministic generic family (no LLM): linear -> generalized LV."""

    def propose(self, *, round_index, history, states, summary):
        if round_index >= len(_TEMPLATES):
            return None
        return _TEMPLATES[round_index](states)


class LLMGeneralProposer:
    """LLM writes a coupled ODE over the dataset's ACTUAL variables (named), not a
    fixed LABS species set. Compounding multi-turn conversation from held-out
    feedback, same pattern as the LABS discovery proposer."""

    def __init__(self, model: str | None = None, api_key: str | None = None,
                 temperature: float = 0.3):
        self._model = (model or os.environ.get("FERMDOCS_OPTIMIZE_MODEL")
                       or os.environ.get("FERMDOCS_GEMINI_MODEL", "gemini-3-pro"))
        self._api_key = api_key or os.environ.get("GEMINI_API_KEY")
        self._temperature = temperature
        self._messages: list[dict] = []

    def _system(self, states: list[str]) -> str:
        return (
            "You are a kinetic-modeling agent. Discover the ODE structure of a "
            "fermentation/bioprocess by trial against held-out RUNS. The measured "
            f"state variables are: {', '.join(states)} (concentrations/levels in "
            "their own units, modeled as a batch system — no feed or aeration "
            "terms). Write one ODE per state. Available in expressions: the state "
            "names above; functions Max, Min, exp, log, sqrt, Abs, and ** for "
            "powers; plus any parameters you declare. Each round you see your "
            "current equations and how well they predicted HELD-OUT runs (R^2 on "
            "the objective). Revise the STRUCTURE to raise held-out objective R^2: "
            "change uptake/production laws, add coupling or saturation, adjust "
            "yields. Do NOT just add parameters — held-out R^2 punishes overfitting. "
            'Return ONLY JSON: {"name": str, "notes": str, '
            '"params": {"<p>": {"init": float, "lb": float, "ub": float}, ...}, '
            '"aux": {"<name>": "<expr>", ...}, '
            '"odes": {' + ", ".join(f'"{s}": "<dX/dt expr>"' for s in states) + "}}"
        )

    def _turn(self, round_index, history, summary) -> str:
        if round_index == 0 or not history:
            return (f"Data summary: {json.dumps(summary)}\n\nPropose your first "
                    "coupled ODE structure as the JSON spec.")
        r = history[-1]
        return (f"Result of round {r['round']} '{r['name']}': held-out objective "
                f"R^2={r['obj_r2']:.3f}" + ("" if r["ok"] else f" [ERROR: {r['error']}]")
                + ". Reason about why, then revise the STRUCTURE to raise held-out "
                "objective R^2. Return the JSON spec.")

    def propose(self, *, round_index, history, states, summary):
        from google import genai
        from google.genai import types

        from fermdocs_optimize.discovery.proposers import _extract_json
        self._messages.append({"role": "user",
                               "parts": [{"text": self._turn(round_index, history, summary)}]})
        client = genai.Client(api_key=self._api_key)
        resp = client.models.generate_content(
            model=self._model, contents=self._messages,
            config=types.GenerateContentConfig(
                system_instruction=self._system(states),
                response_mime_type="application/json", temperature=self._temperature))
        if not resp.text:
            return None
        self._messages.append({"role": "model", "parts": [{"text": resp.text}]})
        raw = _extract_json(resp.text)
        return ModelSpec(
            name=raw.get("name", f"llm_r{round_index}"), notes=raw.get("notes", ""),
            params={k: ParamSpec(**v) for k, v in raw["params"].items()},
            aux=raw.get("aux", {}) or {}, odes=raw["odes"])


# -----------------------------------------------------------------------------
# Fit / simulate / score (batch model: no conditions, no constants)
# -----------------------------------------------------------------------------

def _compile(spec: ModelSpec, states: list[str]):
    return compile_spec(spec.param_names(), spec.aux, spec.odes, state=tuple(states),
                        conditions=(), constants=(), ode_defaults={},
                        required=tuple(states))


def _simulate(compiled, theta, run: StateRun) -> np.ndarray | None:
    try:
        sol = odeint(compiled.rhs, list(run.y0), run.t, args=(theta, {}, {}),
                     mxstep=_MXSTEP)
    except Exception:  # noqa: BLE001
        return None
    return sol if np.all(np.isfinite(sol)) else None


def _fit(compiled, x0, lb, ub, train: list[StateRun]) -> np.ndarray:
    def resid(theta):
        out = []
        for run in train:
            sol = _simulate(compiled, theta, run)
            if sol is None:
                out.append(np.full(run.Y.size, 1e3)); continue
            scale = np.maximum(np.ptp(run.Y, axis=0), 1.0)  # per-state range scale
            out.append(((sol - run.Y) / scale).ravel())
        return np.concatenate(out)

    sol = least_squares(resid, x0, bounds=(lb, ub), method="trf",
                        xtol=1e-8, ftol=1e-8, max_nfev=300)
    return sol.x


def _r2(obs: np.ndarray, pred: np.ndarray) -> float:
    sst = float(np.sum((obs - obs.mean()) ** 2))
    return 1.0 - float(np.sum((obs - pred) ** 2)) / sst if sst > 0 else float("nan")


def _score_objective(compiled, theta, held: list[StateRun], obj_col: int) -> float:
    obs_all, pred_all = [], []
    for run in held:
        sol = _simulate(compiled, theta, run)
        if sol is None:
            continue
        obs_all.append(run.Y[:, obj_col]); pred_all.append(sol[:, obj_col])
    if not obs_all:
        return -1e9
    return _r2(np.concatenate(obs_all), np.concatenate(pred_all))


def _split(runs: list[StateRun], holdout: float, seed: int):
    rng = np.random.default_rng(seed)
    idx = rng.permutation(len(runs))
    n_test = max(1, int(round(len(runs) * holdout)))
    test = {int(i) for i in idx[:n_test]}
    return ([r for i, r in enumerate(runs) if i not in test],
            [r for i, r in enumerate(runs) if i in test])


@dataclass
class GeneralMechFit:
    spec: ModelSpec
    states: list[str]               # sympy-safe state symbols, in order
    sym_to_var: dict[str, str]      # state sym -> observation variable name
    theta: np.ndarray               # params fit on ALL runs (for optimization)
    objective_r2: float             # held-out objective R^2 (the gate)
    runs: list[StateRun]
    obj_col: int


def discover_general_mechanistic(
    obs_df: pd.DataFrame, *, objective: str, proposer=None, max_rounds: int = 5,
    holdout: float = 0.3, seed: int = 7, target_r2: float | None = 0.75,
    gate_r2: float = 0.5,
) -> GeneralMechFit | None:
    """Discover a coupled ODE over all relevant measured variables, gated by
    leave-runs-out held-out R^2 on the objective. Returns a GeneralMechFit (theta
    refit on all runs, ready to simulate/optimize) or None if the data can't
    support a multi-state model or nothing generalizes. Never raises."""
    try:
        runs, states, sym_to_var = load_state_runs(obs_df, objective=objective)
    except ValueError as exc:
        log.info("general-mech: %s", exc)
        return None
    obj_sym = next(s for s, v in sym_to_var.items() if v == objective)
    obj_col = states.index(obj_sym)
    proposer = proposer or GeneralTemplateProposer()
    train, held = _split(runs, holdout, seed)
    summary = {"n_runs": len(runs), "n_train": len(train), "n_held_out": len(held),
               "states": [sym_to_var[s] for s in states], "objective": objective,
               "scored_against": "held_out_runs"}

    history: list[dict] = []
    best: tuple[float, ModelSpec] | None = None
    for r in range(max_rounds):
        try:
            spec = proposer.propose(round_index=r, history=history, states=states,
                                    summary=summary)
        except Exception as exc:  # noqa: BLE001
            log.warning("general-mech proposer failed at round %d: %s", r, exc); break
        if spec is None:
            break
        rec = {"round": r, "name": spec.name, "ok": False, "error": "", "obj_r2": -1e9}
        try:
            compiled = _compile(spec, states)
            names = spec.param_names()
            x0 = np.array([spec.params[n].init for n in names], float)
            lb = np.array([spec.params[n].lb for n in names], float)
            ub = np.array([spec.params[n].ub for n in names], float)
            theta = _fit(compiled, x0, lb, ub, train)
            r2 = _score_objective(compiled, theta, held, obj_col)
            rec.update(ok=True, obj_r2=r2)
            if best is None or r2 > best[0]:
                best = (r2, spec)
        except ExprError as exc:
            rec.update(error=f"compile: {exc}")
        except Exception as exc:  # noqa: BLE001
            rec.update(error=f"{type(exc).__name__}: {exc}")
        history.append(rec)
        log.info("general-mech round %d '%s': held-out objective R2=%.3f%s",
                 r, spec.name, rec["obj_r2"], "" if rec["ok"] else f" [{rec['error']}]")
        if best is not None and target_r2 is not None and best[0] >= target_r2:
            break

    if best is None or not np.isfinite(best[0]) or best[0] < gate_r2:
        return None
    # Refit the winning structure on ALL runs for downstream optimization.
    r2, spec = best
    compiled = _compile(spec, states)
    names = spec.param_names()
    x0 = np.array([spec.params[n].init for n in names], float)
    lb = np.array([spec.params[n].lb for n in names], float)
    ub = np.array([spec.params[n].ub for n in names], float)
    theta = _fit(compiled, x0, lb, ub, runs)
    return GeneralMechFit(spec=spec, states=states, sym_to_var=sym_to_var, theta=theta,
                          objective_r2=float(r2), runs=runs, obj_col=obj_col)


# -----------------------------------------------------------------------------
# Optimization: simulate the fitted ODE, push the controllable initial conditions
# -----------------------------------------------------------------------------

def optimize_initial_conditions(
    fit: GeneralMechFit, lever_states: dict[str, tuple[float, float]],
) -> tuple[dict[str, float], float, dict[str, str]]:
    """Maximize peak objective by moving the controllable state INITIAL conditions
    over their observed ranges (everything else held at the across-run median
    start). `lever_states` maps a state SYMBOL to its observed (lo, hi). Returns
    (best_y0_by_state_sym, predicted_peak, on_boundary). No extrapolation: each
    initial condition stays inside the observed envelope."""
    from scipy.optimize import differential_evolution

    compiled = _compile(fit.spec, fit.states)
    y0_med = np.median(np.array([r.y0 for r in fit.runs]), axis=0)
    t_end = float(max(r.t.max() for r in fit.runs))
    t_grid = np.linspace(0.0, t_end, 60)
    opt_idx = [fit.states.index(s) for s in lever_states]
    los = [lever_states[fit.states[i]][0] for i in opt_idx]
    his = [lever_states[fit.states[i]][1] for i in opt_idx]

    def peak_for(vals) -> float:
        y0 = y0_med.copy()
        for k, i in enumerate(opt_idx):
            y0[i] = vals[k]
        try:
            sol = odeint(compiled.rhs, list(y0), t_grid, args=(fit.theta, {}, {}),
                         mxstep=_MXSTEP)
            col = sol[:, fit.obj_col]
            return float(np.nanmax(col)) if np.all(np.isfinite(col)) else -1e9
        except Exception:  # noqa: BLE001
            return -1e9

    nondegen = [k for k in range(len(opt_idx)) if his[k] - los[k] > 1e-12]
    if nondegen:
        res = differential_evolution(
            lambda xs: -peak_for([xs[nondegen.index(k)] if k in nondegen else los[k]
                                  for k in range(len(opt_idx))]),
            [(los[k], his[k]) for k in nondegen], seed=7, maxiter=60, tol=1e-6, polish=True)
        best_vals = [res.x[nondegen.index(k)] if k in nondegen else los[k]
                     for k in range(len(opt_idx))]
    else:
        best_vals = list(los)
    pred = peak_for(best_vals)
    best = {fit.states[opt_idx[k]]: round(float(best_vals[k]), 4) for k in range(len(opt_idx))}
    on_b: dict[str, str] = {}
    for k in nondegen:
        s = fit.states[opt_idx[k]]
        span = max(his[k] - los[k], 1e-12)
        if abs(best_vals[k] - los[k]) <= 0.02 * span:
            on_b[s] = "lower"
        elif abs(best_vals[k] - his[k]) <= 0.02 * span:
            on_b[s] = "upper"
    return best, float(pred), on_b
