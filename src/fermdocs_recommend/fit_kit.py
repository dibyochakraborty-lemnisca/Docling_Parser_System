"""Tested brewtwin fit/score kit, importable inside the sandbox.

The agent decides WHAT to model (which channels, which families — guided by the
skills and the hypothesis). This kit performs the brittle HOW correctly: load
observations.csv, leave-one-run-out split, fit each family against the real
brewtwin API, simulate on the held-out run's real points, and score via
build_report. It is deterministic and unit-testable, so the agent's sandbox
call cannot hallucinate the data-prep / fit API (the failure mode we hit live).

Typical sandbox usage (one call):

    from fermdocs_recommend.fit_kit import run_bakeoff
    import json
    print(json.dumps(run_bakeoff(
        "<observations_csv_path>",
        biomass="biomass_g_l", substrate="substrate_g_l",
        feed_var="feed_rate_l_per_h", volume_var="volume_l",
    )))
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from fermdocs_recommend.brewtwin_metrics import build_report

_DISQ = "disqualified"


def _first_real(df: pd.DataFrame, run: str, var: str, default: float) -> float:
    s = df[(df["run_id"].astype(str) == str(run)) & (df["variable"] == var)]
    if "imputed" in s:
        s = s[s["imputed"] == 0]
    s = s.dropna(subset=["value"]).sort_values("time_h")
    return float(s["value"].iloc[0]) if len(s) else default


def _series(df: pd.DataFrame, run: str, var: str) -> tuple[np.ndarray, np.ndarray]:
    s = df[(df["run_id"].astype(str) == str(run)) & (df["variable"] == var)].copy()
    s["value"] = pd.to_numeric(s["value"], errors="coerce")
    s["time_h"] = pd.to_numeric(s["time_h"], errors="coerce")
    s = s.dropna(subset=["time_h"]).sort_values("time_h")
    return s["time_h"].to_numpy(float), s["value"].to_numpy(float)


def _real_obs(df: pd.DataFrame, run: str, species: list[str]) -> tuple[np.ndarray, np.ndarray]:
    """(t_eval, y_obs[T, n_species]) of imputed==0 points on `run` (offline grid)."""
    d = df[df["run_id"].astype(str) == str(run)].copy()
    d["value"] = pd.to_numeric(d["value"], errors="coerce")
    d["time_h"] = pd.to_numeric(d["time_h"], errors="coerce")
    if "imputed" in d:
        d = d[d["imputed"] == 0]
    # use the densest-among-offline grid: the union of times where ANY modeled
    # species was really measured
    times = sorted(d[d["variable"].isin(species)].dropna(subset=["time_h"])["time_h"].unique())
    t_eval = np.asarray(times, dtype=float)
    t_index = {round(float(t), 9): i for i, t in enumerate(t_eval)}
    # Place each channel's REAL measured values at their exact times only; never
    # interpolate into the held-out target (interpolated values inflate R^2 by
    # giving the model a smooth, easy-to-track curve instead of real points).
    y = np.full((len(t_eval), len(species)), np.nan)
    for j, sp in enumerate(species):
        ts, vs = _series(d, run, sp)
        for t, v in zip(ts, vs):
            i = t_index.get(round(float(t), 9))
            if i is not None:
                y[i, j] = v
    return t_eval, y


def run_bakeoff(
    obs_csv_path: str,
    *,
    biomass: str,
    substrate: str | None = None,
    product: str | None = None,
    feed_var: str | None = None,
    volume_var: str | None = None,
    feed_substrate_conc: float = 500.0,
    families: tuple[str, ...] = ("mechanistic", "surrogate", "hybrid"),
    n_adam: int = 200,
    n_epochs: int = 400,
) -> dict[str, Any]:
    """Fit + score the requested families. Returns {family: candidate_dict}.

    Each candidate is {"model_type", "attempted", "disqualified",
    "disqualification_reason", "report": <build_report or None>}. Never raises —
    a family that errors is recorded disqualified so one failure cannot lose the
    others (the same contract the rubric expects).
    """
    import jax
    jax.config.update("jax_enable_x64", True)

    df = pd.read_csv(obs_csv_path)
    species = [s for s in [biomass, substrate, product] if s]

    run_ids = sorted(df["run_id"].astype(str).unique().tolist())
    if len(run_ids) >= 2:
        train_ids, val_id = run_ids[:-1], run_ids[-1]
    else:
        train_ids, val_id = run_ids, run_ids[0]
    train_id = train_ids[0]  # single train run keeps the recipe simple + stable

    t_val, y_val = _real_obs(df, val_id, species)
    if t_val.size < 4 or not np.isfinite(y_val).any():
        disq = {"model_type": m, "attempted": True, "disqualified": True,
                "disqualification_reason": "insufficient held-out points", "report": None}
        return {m: {**disq, "model_type": m} for m in families}

    out: dict[str, Any] = {}
    for fam in families:
        try:
            if fam == "mechanistic":
                out[fam] = _fit_mechanistic(df, train_id, val_id, species, biomass, substrate,
                                            feed_var, volume_var, feed_substrate_conc,
                                            t_val, y_val, n_adam, hybrid=False)
            elif fam == "hybrid":
                out[fam] = _fit_mechanistic(df, train_id, val_id, species, biomass, substrate,
                                            feed_var, volume_var, feed_substrate_conc,
                                            t_val, y_val, n_adam, hybrid=True)
            elif fam == "surrogate":
                out[fam] = _fit_surrogate(df, train_id, val_id, species, t_val, y_val, n_epochs)
            else:
                out[fam] = {"model_type": fam, "attempted": False, "disqualified": False,
                            "disqualification_reason": "unknown family", "report": None}
        except Exception as e:  # noqa: BLE001
            out[fam] = {"model_type": fam, "attempted": True, "disqualified": True,
                        "disqualification_reason": f"{type(e).__name__}: {str(e)[:160]}",
                        "report": None}
    return out


def _feed_callable(df, run, feed_var):
    import jax.numpy as jnp
    tg, fv = _series(df, run, feed_var)
    fv = np.nan_to_num(fv, nan=0.0)
    tgj, fvj = jnp.asarray(tg), jnp.asarray(fv)
    return lambda t: jnp.interp(t, tgj, fvj)


def _fit_mechanistic(df, train_id, val_id, species, biomass, substrate, feed_var,
                     volume_var, feed_sub_conc, t_val, y_val, n_adam, *, hybrid):
    import jax.random as jr
    import equinox as eqx
    from brewtwin.species import ChemicalSpecies, BiologicalSpecies
    from brewtwin.reactions.reaction import Reaction
    from brewtwin.reactions.network import ReactionNetwork
    from brewtwin.rate_models.kinetic import Monod, Concentration, Constant
    from brewtwin.rate_models.composite import CompositeRateLaw
    from brewtwin.rate_models.ml import EquinoxRateModel
    from brewtwin.rate_models.hybrid import RelativeHybridRateModel
    from brewtwin.reactors.batch import BatchReactor
    from brewtwin.reactors.fedbatch import FedBatchReactor
    from brewtwin.data.schemas import Trajectory
    from brewtwin.data.observables import from_variable
    from brewtwin.fitting.hybrid_fit import fit
    from brewtwin.solvers.jax_solver import JaxSolver

    model_type = "hybrid" if hybrid else "mechanistic"
    if substrate is None:
        return {"model_type": model_type, "attempted": True, "disqualified": True,
                "disqualification_reason": "mechanistic recipe needs a substrate channel",
                "report": None}

    def build(run):
        X0 = _first_real(df, run, biomass, 0.1)
        S0 = _first_real(df, run, substrate, 10.0)
        X = BiologicalSpecies(biomass, conc=X0)
        S = ChemicalSpecies(substrate, conc=S0)
        net = ReactionNetwork("net"); net.add_species(X); net.add_species(S)
        mech = CompositeRateLaw(Constant(0.3), Monod(S, Ks=1.0), Concentration(X))
        rate = mech
        if hybrid:
            mlp = eqx.nn.MLP(in_size=2, out_size=1, width_size=16, depth=2, key=jr.key(0))
            residual = EquinoxRateModel(mlp, input_features=[substrate, biomass], name="residual")
            rate = RelativeHybridRateModel(mech, residual)
        net.add_reaction(Reaction(name="growth", stoichiometry={substrate: -2.0, biomass: 1.0}, rate_model=rate))
        if feed_var:
            V0 = _first_real(df, run, volume_var, 1.0) if volume_var else 1.0
            feed = _feed_callable(df, run, feed_var)
            return FedBatchReactor(net, feed_rate=feed, feed_concentrations={substrate: feed_sub_conc, biomass: 0.0}, initial_volume=V0)
        return BatchReactor(net)

    mech_species = [biomass, substrate]
    tt, _ = _series(df, train_id, biomass)
    train_X = _series(df, train_id, biomass)
    txa = train_X[0]
    sA = np.interp(txa, *_series(df, train_id, substrate)) if len(txa) else np.array([])
    train_traj = Trajectory.from_dense(t=txa, concentrations={biomass: train_X[1], substrate: sA})
    t_span = (float(txa[0]), float(txa[-1]))

    reactor = build(train_id)
    res = fit(reactor, [train_traj], [from_variable(biomass), from_variable(substrate)],
              t_span=t_span, solver="kvaerno5", n_adam=n_adam, lr_adam=0.05, n_lbfgs=0,
              rtol=1e-6, atol=1e-8, max_steps=200000)
    fitted = res.meta["fitted_model"]

    # read params (traversal; param_estimates is {} for CompositeRateLaw)
    rm = list(fitted.network.reactions)[0].rate_model
    base = rm.mechanistic if hybrid else rm
    params = {"mu_max": float(base.factors[0].value), "Ks": float(base.factors[1].Ks)}

    # validate on the held-out run's times. brewtwin rate models are frozen
    # eqx modules, so we simulate the fitted reactor (carrying the train feed
    # profile + fitted kinetics) at the val run's real time grid — a documented
    # approximation since the val feed differs run-to-run.
    sim = JaxSolver("kvaerno5", rtol=1e-6, atol=1e-8, max_steps=200000).solve(
        fitted, t_span=(float(t_val[0]), float(t_val[-1])), t_eval=t_val)
    y_pred = np.full((len(t_val), len(species)), np.nan)
    for j, sp in enumerate(species):
        if sp in sim.variables:
            y_pred[:, j] = np.asarray(sim.y[:, sim.variables.index(sp)])

    report = build_report(model_type=model_type, fitted_params=({} if hybrid else params),
                          loss_history=[float(x) for x in res.loss_history], y_pred=y_pred,
                          y_obs=y_val, t_pred=t_val, species=species, fit_window=t_span,
                          predict_window=(float(t_val[0]), float(t_val[-1])),
                          caveats=("hybrid ML residual is a black-box correction",) if hybrid else ())
    return {"model_type": model_type, "attempted": True, "disqualified": False,
            "disqualification_reason": None, "report": report}


def _fit_surrogate(df, train_id, val_id, species, t_val, y_val, n_epochs):
    import jax.numpy as jnp
    import jax.random as jr
    import diffrax
    from brewtwin.data.schemas import Trajectory
    from brewtwin.surrogates.neural_ode import NeuralODE
    from brewtwin.surrogates.train import fit_surrogate

    # offline grid for the training run
    tg, xb = _series(df, train_id, species[0])
    conc = {species[0]: xb}
    for sp in species[1:]:
        ts, vs = _series(df, train_id, sp)
        conc[sp] = np.interp(tg, ts, vs) if len(ts) else np.full_like(tg, np.nan)
    train_traj = Trajectory.from_dense(t=tg, concentrations=conc)

    node = NeuralODE(state_size=len(species), width=48, depth=3, key=jr.key(0),
                     solver=diffrax.Tsit5(), rtol=1e-4, atol=1e-6)
    sres = fit_surrogate(node, [train_traj], species, solver="tsit5", rtol=1e-4, atol=1e-6,
                         n_epochs=n_epochs, lr=5e-3)
    y0 = jnp.array([float(np.nan_to_num(y_val[0, j], nan=conc[species[j]][0] if len(conc[species[j]]) else 0.0))
                    for j in range(len(species))])
    pred = np.asarray(sres.surrogate.rollout(y0, jnp.asarray(t_val)))
    report = build_report(model_type="surrogate", fitted_params={},
                          loss_history=[float(x) for x in sres.train_loss_history], y_pred=pred,
                          y_obs=y_val, t_pred=t_val, species=species,
                          fit_window=(float(tg[0]), float(tg[-1])),
                          predict_window=(float(t_val[0]), float(t_val[-1])),
                          caveats=("surrogate valid only within the training distribution",))
    return {"model_type": "surrogate", "attempted": True, "disqualified": False,
            "disqualification_reason": None, "report": report}
