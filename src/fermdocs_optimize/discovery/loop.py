"""The equation-discovery loop.

    propose ODE structure ─► compile + fit params on DATA ─► score vs ORACLE
        ▲                                                          │
        │                                                          ▼
    revise equations  ◄──── feed back peak/trajectory error ◄── (probe conditions)
                                                                   │
                                          converged? (RMSE plateau) ─► keep best

The agent rewrites the *structure*; the oracle judges every structure on
held-out conditions. The agent never sees the oracle's parameters — only how
wrong its own equations are. This is the honest "make the equations, ask the
oracle, fix them, repeat" loop.
"""
from __future__ import annotations

import logging

import numpy as np
import pandas as pd
from scipy.stats import qmc

from fermdocs_optimize.discovery.candidate_model import CandidateModel
from fermdocs_optimize.discovery.proposers import SpecProposer, TemplateProposer
from fermdocs_optimize.discovery.spec import DiscoveryReport, DiscoveryRound
from fermdocs_optimize.models.mechanistic import MechanisticModel
from fermdocs_optimize.schema import KNOB_NAMES, Box, Candidate
from fermdocs_optimize.simulators.base import Simulator

log = logging.getLogger(__name__)

_T_END, _N_T = 75.0, 76


def _probe_candidates(box: Box, n: int, seed: int) -> list[Candidate]:
    bounds = np.array(box.as_list(), float)
    lo, hi = bounds[:, 0], bounds[:, 1]
    pts = qmc.scale(qmc.LatinHypercube(d=len(KNOB_NAMES), seed=seed).random(n), lo, hi)
    return [Candidate(**dict(zip(KNOB_NAMES, row))) for row in pts]


def _oracle_truth(simulator: Simulator, probes: list[Candidate], v0: float):
    """Per-probe oracle peak P and the P trajectory on a common time grid."""
    df = simulator.simulate(probes, v0=v0)
    peaks, trajs = [], []
    grid = np.linspace(0, _T_END, _N_T)
    for b in sorted(df["batch"].unique(),
                    key=lambda x: int("".join(c for c in str(x) if c.isdigit()) or 0)):
        g = df[df["batch"] == b].sort_values("t")
        t, p = g["t"].to_numpy(float), g["P"].to_numpy(float)
        peaks.append(float(p.max()))
        trajs.append(np.interp(grid, t, p))
    return np.array(peaks), np.array(trajs)


def _score(model, probes, oracle_peaks, oracle_trajs, v0):
    """Peak RMSE/R^2 and trajectory R^2 of a model vs the oracle. Works for any
    PredictiveModel (peak only) and adds trajectory R^2 when the model exposes
    `predict_P_trajectory` (the discovery CandidateModel)."""
    traj_fn = getattr(model, "predict_P_trajectory", None)
    if traj_fn is not None:
        pred_trajs = np.array([traj_fn(c, v0=v0, t_end=_T_END, n=_N_T) for c in probes])
        pred_peaks = pred_trajs.max(axis=1)
    else:
        pred_trajs = None
        pred_peaks = np.array([model.predict_peak_titer(c, v0=v0) for c in probes])
    finite = np.isfinite(pred_peaks) & (pred_peaks > -1e5)
    if pred_trajs is not None:
        finite &= np.all(np.isfinite(pred_trajs), axis=1)
    if finite.sum() < 2:
        return 1e6, -1e9, -1e9
    op, pp = oracle_peaks[finite], pred_peaks[finite]
    rmse = float(np.sqrt(np.mean((pp - op) ** 2)))
    ss_tot = float(np.sum((op - op.mean()) ** 2))
    peak_r2 = 1.0 - float(np.sum((op - pp) ** 2)) / ss_tot if ss_tot > 0 else float("nan")
    traj_r2 = float("nan")
    if pred_trajs is not None:
        ot, pt = oracle_trajs[finite].ravel(), pred_trajs[finite].ravel()
        tt = float(np.sum((ot - ot.mean()) ** 2))
        traj_r2 = 1.0 - float(np.sum((ot - pt) ** 2)) / tt if tt > 0 else float("nan")
    return rmse, peak_r2, traj_r2


def _batch_truth(df: pd.DataFrame):
    """Reconstruct (Candidate, actual peak P, t-grid, actual P trajectory) for each
    batch from real data — the held-out truth when there is no oracle."""
    out = []
    for exp, obs in zip(*_reconstruct_pairs(df)):
        v0 = float(exp["y0"][5])
        x0, s0, p0, m0 = (float(exp["y0"][i]) for i in range(4))
        total = s0 + m0
        cand = Candidate(biomass=x0, total_sub=total,
                         malt_frac=(m0 / total if total > 0 else 0.0),
                         dilution=(exp["F"] / v0 if v0 > 0 else 0.0))
        p_traj = obs[:, 2]  # FIT_SPECIES = (X,S,P,M) -> P is column 2
        out.append((cand, float(p_traj.max()), np.asarray(exp["t"], float), p_traj, v0))
    return out


def _reconstruct_pairs(df):
    batches = MechanisticModel._reconstruct(df)
    return [b[0] for b in batches], [b[1] for b in batches]


def _r2(obs: np.ndarray, pred: np.ndarray) -> float:
    sst = float(np.sum((obs - obs.mean()) ** 2))
    return 1.0 - float(np.sum((obs - pred) ** 2)) / sst if sst > 0 else float("nan")


def _collect_preds(model, truths, v0):
    """Per-batch predicted vs observed peaks and P-trajectories for held-out
    batches. Batches the model can't integrate (non-finite) are dropped. Returns
    four parallel lists: (pred_peaks, obs_peaks, pred_trajs, obs_trajs)."""
    pred_peaks, obs_peaks, pred_trajs, obs_trajs = [], [], [], []
    for cand, peak, t, p_traj, v0b in truths:
        traj = model.predict_P_trajectory(cand, v0=v0b, t_end=float(t[-1]), n=len(t)) \
            if hasattr(model, "predict_P_trajectory") else None
        if traj is None:
            pp = model.predict_peak_titer(cand, v0=v0b)
        else:
            if not np.all(np.isfinite(traj)) or traj.max() < -1e5:
                continue
            pp = float(traj.max())
            pred_trajs.append(traj); obs_trajs.append(p_traj)
        if not np.isfinite(pp) or pp < -1e5:
            continue
        pred_peaks.append(pp); obs_peaks.append(peak)
    return pred_peaks, obs_peaks, pred_trajs, obs_trajs


def _metrics(pred_peaks, obs_peaks, pred_trajs, obs_trajs):
    """Peak RMSE/R^2 + trajectory R^2 from collected predictions (needs >=2 points)."""
    if len(pred_peaks) < 2:
        return 1e6, -1e9, -1e9
    pp = np.array(pred_peaks); op = np.array(obs_peaks)
    rmse = float(np.sqrt(np.mean((pp - op) ** 2)))
    peak_r2 = _r2(op, pp)
    traj_r2 = float("nan")
    if pred_trajs:
        traj_r2 = _r2(np.concatenate(obs_trajs), np.concatenate(pred_trajs))
    return rmse, peak_r2, traj_r2


def _split_batches(df: pd.DataFrame, holdout: float, seed: int):
    """Single train/test split at the batch level. `holdout` is the test fraction
    (>= 2 batches). Returns (train_df, test_df)."""
    ids = sorted(df["batch"].unique(),
                 key=lambda x: int("".join(c for c in str(x) if c.isdigit()) or 0))
    rng = np.random.default_rng(seed)
    perm = rng.permutation(len(ids))
    n_test = max(2, int(round(len(ids) * holdout)))
    test_ids = {ids[i] for i in perm[:n_test]}
    train = df[~df["batch"].isin(test_ids)].copy()
    test = df[df["batch"].isin(test_ids)].copy()
    return train, test


def _score_on_data(model, truths, v0):
    """Peak RMSE/R^2 + trajectory R^2 of a model on held-out REAL batches."""
    return _metrics(*_collect_preds(model, truths, v0))


def discover_model_from_data(
    *,
    data: pd.DataFrame,
    proposer: SpecProposer | None = None,
    max_rounds: int = 5,
    holdout: float = 0.3,
    seed: int = 7,
    v0: float = 10.0,
    rmse_tol: float = 0.5,
    target_peak_r2: float | None = None,
) -> DiscoveryReport:
    """Discover the ODE structure with NO oracle: fit each proposed structure on a
    training split of REAL batches and score peak prediction on a held-out split.
    The held-out data is the only ground truth available. The best structure is
    what the caller searches for an optimum (it refits the spec on all the data)."""
    proposer = proposer or TemplateProposer()
    train_df, test_df = _split_batches(data, holdout, seed)
    truths = _batch_truth(test_df)

    base = MechanisticModel(); base.fit(train_df)
    base_rmse, _, _ = _score_on_data(base, truths, v0)

    data_summary = {
        "n_train_batches": int(train_df["batch"].nunique()),
        "n_test_batches": int(test_df["batch"].nunique()),
        "P_range": [round(float(data["P"].min()), 2), round(float(data["P"].max()), 2)],
        "species": ["X", "S", "P", "M"], "knobs": list(KNOB_NAMES),
        "baseline_peak_rmse": round(base_rmse, 3), "scored_against": "held_out_real_batches",
    }

    rounds: list[DiscoveryRound] = []
    best: DiscoveryRound | None = None
    exit_reason = "max_rounds"
    for r in range(max_rounds):
        try:
            spec = proposer.propose(round_index=r, history=rounds, data_summary=data_summary)
        except Exception as exc:  # noqa: BLE001 — a malformed proposal ends discovery with best-so-far
            log.warning("proposer failed at round %d: %s", r, exc)
            exit_reason = "proposer_error"; break
        if spec is None:
            exit_reason = "proposer_done"; break
        try:
            model = CandidateModel(spec); r2 = model.fit(train_df)
            rmse, peak_r2, traj_r2 = _score_on_data(model, truths, v0)
            rd = DiscoveryRound(
                round_index=r, spec=spec, fitted_params=model.fitted_params,
                r2_by_species={k: round(v, 4) for k, v in r2.items()},
                oracle_peak_rmse=round(rmse, 4), oracle_peak_r2=round(peak_r2, 4),
                oracle_traj_r2=round(traj_r2, 4), score=round(-rmse, 4), compile_ok=True)
        except Exception as exc:  # noqa: BLE001
            log.warning("discovery (data) round %d failed: %s", r, exc)
            rd = DiscoveryRound(
                round_index=r, spec=spec, fitted_params={}, r2_by_species={},
                oracle_peak_rmse=1e6, oracle_peak_r2=-1e9, oracle_traj_r2=-1e9,
                score=-1e6, compile_ok=False, error=f"{type(exc).__name__}: {exc}")
        rounds.append(rd)
        log.info("round %d '%s': held-out peak RMSE=%.3f R2=%.3f (train P R2=%.3f)",
                 r, spec.name, rd.oracle_peak_rmse, rd.oracle_peak_r2,
                 rd.r2_by_species.get("P", float("nan")))
        if best is None or rd.oracle_peak_rmse < best.oracle_peak_rmse:
            best = rd
        if best.oracle_peak_rmse <= rmse_tol:
            exit_reason = "converged"; break
        # held-out peak R² is good enough -> stop discovering, go optimize
        if target_peak_r2 is not None and best.oracle_peak_r2 >= target_peak_r2:
            exit_reason = "r2_target_reached"; break

    return DiscoveryReport(
        best_spec=best.spec if best else None,
        best_round=best.round_index if best else -1,
        rounds=rounds, exit_reason=exit_reason,
        oracle_peak_rmse=best.oracle_peak_rmse if best else None,
        oracle_peak_r2=best.oracle_peak_r2 if best else None,
        n_oracle_evals=0, baseline_peak_rmse=round(base_rmse, 4),
        improved=bool(best and best.oracle_peak_rmse < base_rmse),
    )


def discover_model(
    *,
    training_data: pd.DataFrame,
    simulator: Simulator,
    box: Box,
    proposer: SpecProposer | None = None,
    max_rounds: int = 5,
    n_probes: int = 24,
    v0: float = 10.0,
    seed: int = 7,
    rmse_tol: float = 0.5,
) -> DiscoveryReport:
    """Run the discovery loop. Returns the best structure found, scored on the
    oracle's held-out probe conditions."""
    proposer = proposer or TemplateProposer()
    probes = _probe_candidates(box, n_probes, seed)
    oracle_peaks, oracle_trajs = _oracle_truth(simulator, probes, v0)
    n_oracle_evals = len(probes)

    # Baseline: the fixed hand-written mechanistic model, same probes.
    base = MechanisticModel()
    base.fit(training_data)
    base_rmse, _, _ = _score(base, probes, oracle_peaks, oracle_trajs, v0)

    data_summary = {
        "n_batches": int(training_data["batch"].nunique()),
        "P_range": [round(float(training_data["P"].min()), 2),
                    round(float(training_data["P"].max()), 2)],
        "species": ["X", "S", "P", "M"],
        "knobs": list(KNOB_NAMES),
        "oracle_peak_P_range": [round(float(oracle_peaks.min()), 2),
                                round(float(oracle_peaks.max()), 2)],
        "baseline_peak_rmse": round(base_rmse, 3),
    }

    rounds: list[DiscoveryRound] = []
    best: DiscoveryRound | None = None
    exit_reason = "max_rounds"

    for r in range(max_rounds):
        try:
            spec = proposer.propose(round_index=r, history=rounds, data_summary=data_summary)
        except Exception as exc:  # noqa: BLE001 — a malformed proposal ends discovery with best-so-far
            log.warning("proposer failed at round %d: %s", r, exc)
            exit_reason = "proposer_error"
            break
        if spec is None:
            exit_reason = "proposer_done"
            break
        try:
            model = CandidateModel(spec)
            r2 = model.fit(training_data)
            rmse, peak_r2, traj_r2 = _score(model, probes, oracle_peaks, oracle_trajs, v0)
            rd = DiscoveryRound(
                round_index=r, spec=spec, fitted_params=model.fitted_params,
                r2_by_species={k: round(v, 4) for k, v in r2.items()},
                oracle_peak_rmse=round(rmse, 4), oracle_peak_r2=round(peak_r2, 4),
                oracle_traj_r2=round(traj_r2, 4), score=round(-rmse, 4), compile_ok=True)
        except Exception as exc:  # noqa: BLE001 — a bad structure scores worst, never crashes
            log.warning("discovery round %d failed: %s", r, exc)
            rd = DiscoveryRound(
                round_index=r, spec=spec, fitted_params={}, r2_by_species={},
                oracle_peak_rmse=1e6, oracle_peak_r2=-1e9, oracle_traj_r2=-1e9,
                score=-1e6, compile_ok=False, error=f"{type(exc).__name__}: {exc}")
        rounds.append(rd)
        n_oracle_evals += len(probes)
        log.info("round %d '%s': oracle peak RMSE=%.3f R2=%.3f (data P R2=%.3f)",
                 r, spec.name, rd.oracle_peak_rmse, rd.oracle_peak_r2,
                 rd.r2_by_species.get("P", float("nan")))

        if best is None or rd.oracle_peak_rmse < best.oracle_peak_rmse:
            best = rd
        if best.oracle_peak_rmse <= rmse_tol:
            exit_reason = "converged"
            break

    return DiscoveryReport(
        best_spec=best.spec if best else None,
        best_round=best.round_index if best else -1,
        rounds=rounds, exit_reason=exit_reason,
        oracle_peak_rmse=best.oracle_peak_rmse if best else None,
        oracle_peak_r2=best.oracle_peak_r2 if best else None,
        n_oracle_evals=n_oracle_evals,
        baseline_peak_rmse=round(base_rmse, 4),
        improved=bool(best and best.oracle_peak_rmse < base_rmse),
    )
