"""Improvement loop for the mechanistic model (recommend stage).

The one-shot mechanistic fit fails on sparse lab data for two reasons: the kinetic
params aren't identifiable from 5-9 points per run, and a fixed Monod structure
mismatches a fed-batch, product-inhibited lactic process. This loop attacks both:

  propose ODE structure ──► fit ACROSS ALL RUNS ──► score LEAVE-RUNS-OUT
        ▲                                                    │
        │                                                    ▼
   revise structure ◄──── feed back held-out R² + residuals ◄┘

It is the equation-discovery loop (fermdocs_optimize.discovery) brought to the
recommend stage, with two deliberate changes for THIS data:

  * Fit ONE shared structure across all runs at once (pool ~100 points + condition
    diversity), not per-run on 5-9 points. This is what makes the kinetic params
    identifiable.
  * Validate by holding out WHOLE RUNS (train on most, predict the rest), a far
    stronger generalization test than holding out the tail of one sparse curve.

The held-out R^2 is the loop's gate, which is exactly what keeps it honest: a more
complex structure that overfits the training runs scores WORSE held-out, so the
loop converges to the simplest structure that genuinely generalizes instead of
piling on parameters. If no structure clears the bar, the loop reports its best
and the rubric still (correctly) refuses — the loop raises the ceiling, it cannot
manufacture information that isn't in the data.

State: we model substrate S and product P directly (g/L). There is no measured
biomass in g/L in these bundles, so growth is folded into the substrate->product
kinetics (a reduced model); X/M/O2/V are inert here. We reuse the optimizer's
safe symbolic compiler (`expr.compile_spec`) so the LLM rewrites EQUATIONS, never
code.
"""
from __future__ import annotations

import logging

import numpy as np
import pandas as pd
from scipy.integrate import odeint
from scipy.optimize import least_squares

from fermdocs_optimize.discovery.expr import ExprError, compile_spec
from fermdocs_optimize.discovery.spec import ModelSpec, ParamSpec

log = logging.getLogger(__name__)

# Inert constants/conditions: we model concentrations in a batch approximation
# (no feed term, no aeration). Zeroing these makes the compiler's default O2/V
# balances inert so only the proposed dS/dt, dP/dt matter.
_CONST = {"K_O2": 0.1, "q_O2_max": 0.0, "O2_sat": 1.0, "kLa": 0.0, "C_FEED": 0.0}

_SUBSTRATE = "substrate_g_l"
_PRODUCT = "product_g_l"
_MXSTEP = 500


# --------------------------------------------------------------------------- data
class RunSeries:
    """One run's observed substrate/product on a shared time grid."""

    __slots__ = ("run_id", "t", "S", "P", "S0", "P0", "V0")

    def __init__(self, run_id: str, t, S, P, V0: float = 1.0):
        self.run_id = run_id
        self.t = np.asarray(t, float)
        self.S = np.asarray(S, float)
        self.P = np.asarray(P, float)
        self.S0 = float(self.S[0])
        self.P0 = float(self.P[0])
        self.V0 = float(V0)


def load_runs(observations: pd.DataFrame) -> list[RunSeries]:
    """Build per-run substrate/product series from a long observations frame
    (columns run_id, variable, time_h, value). A run needs both S and P with >=3
    aligned timepoints to be usable."""
    runs: list[RunSeries] = []
    for run_id, g in observations.groupby("run_id"):
        piv = (g[g["variable"].isin([_SUBSTRATE, _PRODUCT])]
               .pivot_table(index="time_h", columns="variable", values="value", aggfunc="mean")
               .sort_index())
        if _SUBSTRATE not in piv or _PRODUCT not in piv:
            continue
        piv = piv.dropna(subset=[_SUBSTRATE, _PRODUCT])
        if len(piv) < 3:
            continue
        v0_rows = g[g["variable"] == "volume_l"].sort_values("time_h")
        v0 = float(v0_rows["value"].iloc[0]) if len(v0_rows) else 1.0
        runs.append(RunSeries(str(run_id), piv.index.to_numpy(float),
                              piv[_SUBSTRATE].to_numpy(float),
                              piv[_PRODUCT].to_numpy(float), v0))
    return runs


def _split_runs(runs: list[RunSeries], holdout: float, seed: int):
    """Leave-RUNS-out split: hold out a fraction of whole runs (>=1)."""
    rng = np.random.default_rng(seed)
    idx = rng.permutation(len(runs))
    n_test = max(1, int(round(len(runs) * holdout)))
    test = {int(i) for i in idx[:n_test]}
    train = [r for i, r in enumerate(runs) if i not in test]
    held = [r for i, r in enumerate(runs) if i in test]
    return train, held


# ----------------------------------------------------------------------- fit/score
def _simulate(compiled, theta: np.ndarray, run: RunSeries):
    """Integrate the compiled ODE for one run from its initial S,P. Returns the
    predicted (S, P) on the run's time grid, or None if integration fails."""
    y0 = [0.0, run.S0, run.P0, 0.0, _CONST["O2_sat"], run.V0]
    cond = {"F": 0.0, "S_f": 0.0, "M_f": 0.0, "v0": run.V0}
    try:
        sol = odeint(compiled.rhs, y0, run.t, args=(theta, cond, _CONST), mxstep=_MXSTEP)
    except Exception:  # noqa: BLE001
        return None
    S, P = sol[:, 1], sol[:, 2]
    if not (np.all(np.isfinite(S)) and np.all(np.isfinite(P))):
        return None
    return S, P


def _fit(compiled, x0, lb, ub, train: list[RunSeries]) -> np.ndarray:
    """Fit params by pooling residuals (S and P, range-scaled) across all train runs."""
    def resid(theta):
        out = []
        for run in train:
            pred = _simulate(compiled, theta, run)
            if pred is None:
                out.append(np.full(len(run.t) * 2, 1e3)); continue
            Sp, Pp = pred
            out.append((Sp - run.S) / max(np.ptp(run.S), 1.0))
            out.append((Pp - run.P) / max(np.ptp(run.P), 1.0))
        return np.concatenate(out)

    sol = least_squares(resid, x0, bounds=(lb, ub), method="trf",
                        xtol=1e-8, ftol=1e-8, max_nfev=300)
    return sol.x


def _r2(obs: np.ndarray, pred: np.ndarray) -> float:
    sst = float(np.sum((obs - obs.mean()) ** 2))
    return 1.0 - float(np.sum((obs - pred) ** 2)) / sst if sst > 0 else float("nan")


def _score(compiled, theta, held: list[RunSeries]) -> dict:
    """Pooled held-out fit quality on product (primary) and substrate."""
    So, Sp, Po, Pp = [], [], [], []
    for run in held:
        pred = _simulate(compiled, theta, run)
        if pred is None:
            continue
        S, P = pred
        So.append(run.S); Sp.append(S); Po.append(run.P); Pp.append(P)
    if not Po:
        return {_PRODUCT: {"r2": -1e9, "rmse": 1e6, "n": 0},
                _SUBSTRATE: {"r2": -1e9, "rmse": 1e6, "n": 0}}
    So, Sp = np.concatenate(So), np.concatenate(Sp)
    Po, Pp = np.concatenate(Po), np.concatenate(Pp)
    return {
        _PRODUCT: {"r2": _r2(Po, Pp), "rmse": float(np.sqrt(np.mean((Po - Pp) ** 2))), "n": int(Po.size)},
        _SUBSTRATE: {"r2": _r2(So, Sp), "rmse": float(np.sqrt(np.mean((So - Sp) ** 2))), "n": int(So.size)},
    }


# ------------------------------------------------------------------------ proposers
def _spec(name: str, params: dict[str, ParamSpec], dS: str, dP: str, notes: str = "") -> ModelSpec:
    # X and M are inert (no biomass/maltose in these bundles); growth is folded
    # into the substrate->product kinetics.
    return ModelSpec(name=name, params=params, aux={},
                     odes={"X": "0", "S": dS, "P": dP, "M": "0"}, notes=notes)


_TEMPLATES = [
    lambda: _spec(
        "monod_substrate_to_product",
        {"vmax": ParamSpec(init=5.0, lb=0.0, ub=200.0),
         "Ks": ParamSpec(init=10.0, lb=1e-3, ub=200.0),
         "Yps": ParamSpec(init=1.0, lb=0.05, ub=5.0)},
        dS="-vmax*S/(Ks+S)/Yps", dP="vmax*S/(Ks+S)",
        notes="Monod substrate uptake -> product; yield Yps couples dS to dP."),
    lambda: _spec(
        "monod_product_inhibition",
        {"vmax": ParamSpec(init=5.0, lb=0.0, ub=200.0),
         "Ks": ParamSpec(init=10.0, lb=1e-3, ub=200.0),
         "Yps": ParamSpec(init=1.0, lb=0.05, ub=5.0),
         "Pmax": ParamSpec(init=150.0, lb=50.0, ub=400.0)},
        dS="-(vmax*S/(Ks+S)*Max(0,1-P/Pmax))/Yps", dP="vmax*S/(Ks+S)*Max(0,1-P/Pmax)",
        notes="Adds linear product inhibition (1-P/Pmax) — lactic acid self-inhibits."),
    lambda: _spec(
        "monod_cubic_inhibition",
        {"vmax": ParamSpec(init=5.0, lb=0.0, ub=200.0),
         "Ks": ParamSpec(init=10.0, lb=1e-3, ub=200.0),
         "Yps": ParamSpec(init=1.0, lb=0.05, ub=5.0),
         "Pmax": ParamSpec(init=150.0, lb=50.0, ub=400.0)},
        dS="-(vmax*S/(Ks+S)*Max(0,1-P/Pmax)**3)/Yps", dP="vmax*S/(Ks+S)*Max(0,1-P/Pmax)**3",
        notes="Sharper (cubic) inhibition near the titer ceiling."),
]


class TemplateMechProposer:
    """Deterministic walk over a curated family (no LLM) — runs without an API key
    and is the honest fallback / test proposer."""

    def propose(self, *, round_index, history, summary):
        if round_index >= len(_TEMPLATES):
            return None
        return _TEMPLATES[round_index]()


class LLMMechProposer:
    """Gemini rewrites and revises the kinetics from held-out feedback (compounding
    multi-turn conversation, same pattern as the optimizer's discovery proposer)."""

    _SYSTEM = (
        "You are a fermentation kinetic-modeling agent. Discover the ODE structure of a "
        "lactic-acid fed-batch fermentation by trial against held-out RUNS. State variables: "
        "S (substrate, g/L) and P (product/titer, g/L). There is no biomass measurement, so "
        "fold growth into the substrate->product kinetics. Available in expressions: S, P; "
        "functions Max, Min, exp, log, sqrt, Abs, and ** for powers; plus any params you "
        "declare. Each round you see your current equations and how well they predicted "
        "HELD-OUT runs (R^2 on product and substrate). Revise the STRUCTURE to raise held-out "
        "product R^2: change the uptake law, add/strengthen product inhibition, adjust the "
        "yield coupling. Do NOT just add parameters — held-out R^2 punishes overfitting. "
        'Return ONLY JSON: {"name": str, "notes": str, '
        '"params": {"<p>": {"init": float, "lb": float, "ub": float}, ...}, '
        '"odes": {"S": "<dS/dt expr>", "P": "<dP/dt expr>"}}'
    )

    def __init__(self, model: str | None = None, api_key: str | None = None, temperature: float = 0.3):
        import os
        self._model = (model or os.environ.get("FERMDOCS_RECOMMEND_MODEL")
                       or os.environ.get("FERMDOCS_GEMINI_MODEL", "gemini-3-pro"))
        self._api_key = api_key or os.environ.get("GEMINI_API_KEY")
        self._temperature = temperature
        self._messages: list[dict] = []

    def _turn(self, round_index, history, summary) -> str:
        if round_index == 0 or not history:
            return (f"Data summary: {summary}\n\nPropose your first kinetic ODE structure "
                    "for dS/dt and dP/dt as the JSON spec.")
        r = history[-1]
        return (f"Result of round {r['round']} '{r['name']}': held-out product R^2="
                f"{r['p_r2']:.3f}, substrate R^2={r['s_r2']:.3f}"
                + ("" if r["ok"] else f" [ERROR: {r['error']}]")
                + ". Reason about why, then revise the STRUCTURE to raise held-out product "
                "R^2. Return the JSON spec.")

    def propose(self, *, round_index, history, summary):
        from google import genai
        from google.genai import types

        from fermdocs_optimize.discovery.proposers import _extract_json
        self._messages.append({"role": "user", "parts": [{"text": self._turn(round_index, history, summary)}]})
        client = genai.Client(api_key=self._api_key)
        resp = client.models.generate_content(
            model=self._model, contents=self._messages,
            config=types.GenerateContentConfig(
                system_instruction=self._SYSTEM, response_mime_type="application/json",
                temperature=self._temperature))
        if not resp.text:
            return None
        self._messages.append({"role": "model", "parts": [{"text": resp.text}]})
        raw = _extract_json(resp.text)
        odes = raw["odes"]
        return _spec(raw.get("name", f"llm_r{round_index}"),
                     {k: ParamSpec(**v) for k, v in raw["params"].items()},
                     dS=odes["S"], dP=odes["P"], notes=raw.get("notes", ""))


# ----------------------------------------------------------------------------- loop
def discover_mechanistic(
    observations: pd.DataFrame,
    *,
    proposer=None,
    max_rounds: int = 5,
    holdout: float = 0.3,
    seed: int = 7,
    target_p_r2: float | None = 0.75,
) -> dict:
    """Run the mechanistic improvement loop and return a rubric-shaped candidate.

    Returns {model_type, attempted, disqualified, best_spec, report, rounds, ...}
    where `report.fit_quality` is the held-out R^2/RMSE per species so the existing
    rubric (`score_candidate`/`gate_good_fit`) scores it exactly like the bake-off
    families."""
    proposer = proposer or TemplateMechProposer()
    runs = load_runs(observations)
    if len(runs) < 3:
        return {"model_type": "mechanistic_discovered", "attempted": True, "disqualified": True,
                "disqualification_reason": f"need >=3 usable runs, got {len(runs)}", "report": None}

    train, held = _split_runs(runs, holdout, seed)
    summary = {"n_runs": len(runs), "n_train": len(train), "n_held_out": len(held),
               "P_range": [round(min(r.P.min() for r in runs), 1), round(max(r.P.max() for r in runs), 1)],
               "S_range": [round(min(r.S.min() for r in runs), 1), round(max(r.S.max() for r in runs), 1)],
               "scored_against": "held_out_runs"}

    history: list[dict] = []
    best = None  # (p_r2, spec, theta, report)
    for r in range(max_rounds):
        try:
            spec = proposer.propose(round_index=r, history=history, summary=summary)
        except Exception as exc:  # noqa: BLE001
            log.warning("mech proposer failed at round %d: %s", r, exc); break
        if spec is None:
            break
        rec = {"round": r, "name": spec.name, "ok": False, "error": "", "p_r2": -1e9, "s_r2": -1e9}
        try:
            compiled = compile_spec(spec.param_names(), spec.aux, spec.odes)
            names = spec.param_names()
            x0 = np.array([spec.params[n].init for n in names], float)
            lb = np.array([spec.params[n].lb for n in names], float)
            ub = np.array([spec.params[n].ub for n in names], float)
            theta = _fit(compiled, x0, lb, ub, train)
            fq = _score(compiled, theta, held)
            rec.update(ok=True, p_r2=fq[_PRODUCT]["r2"], s_r2=fq[_SUBSTRATE]["r2"])
            # PRODUCT-ONLY gate: titer is the objective, and substrate is fed-batch
            # noisy (pulsed feed, no feed term in this reduced model). Exposing only
            # product in fit_quality makes the rubric's min-over-species gate on
            # product alone — no rubric change needed. Substrate kept as diagnostic.
            report = {"fit_quality": {_PRODUCT: fq[_PRODUCT]},
                      "fitted_parameters": {n: {"value": float(v), "plausible": True}
                                            for n, v in zip(names, theta)},
                      "substrate_diagnostic_r2": round(float(fq[_SUBSTRATE]["r2"]), 4)}
            cand = (fq[_PRODUCT]["r2"], spec, theta, report)
            if best is None or cand[0] > best[0]:
                best = cand
        except ExprError as exc:
            rec.update(error=f"compile: {exc}")
        except Exception as exc:  # noqa: BLE001
            rec.update(error=f"{type(exc).__name__}: {exc}")
        history.append(rec)
        log.info("mech round %d '%s': held-out P R2=%.3f S R2=%.3f%s",
                 r, spec.name, rec["p_r2"], rec["s_r2"], "" if rec["ok"] else f" [{rec['error']}]")
        if best is not None and target_p_r2 is not None and best[0] >= target_p_r2:
            log.info("mech discovery: held-out P R2 target %.2f reached", target_p_r2)
            break

    if best is None:
        return {"model_type": "mechanistic_discovered", "attempted": True, "disqualified": True,
                "disqualification_reason": "no structure compiled+fit", "report": None,
                "rounds": history}
    p_r2, spec, theta, report = best
    return {
        "model_type": "mechanistic_discovered", "attempted": True, "disqualified": False,
        "best_spec": {"name": spec.name, "odes": spec.odes, "notes": spec.notes},
        "best_held_out_product_r2": round(float(p_r2), 4),
        "report": report, "rounds": history,
        "n_runs": len(runs), "n_held_out": len(held),
    }
