"""Self-contained mechanistic model of the lactic-acid fed-batch.

This re-implements the standard bioprocess kinetic ODE (Monod growth +
product inhibition + maltose uptake + dissolved-O2 dynamics) and fits its 7
parameters with `scipy.optimize.least_squares`. It is deliberately decoupled
from the LABS package: the optimizer reaches the simulator only through the
`Simulator` boundary (subprocess), so the agent's model is its own approximation.

Integrity: this fits ONLY on observation data. It never reads the simulator's
true parameter file (mech_params.json). The ODE *form* is general bioprocess
knowledge; the parameter *values* are what the loop must learn from data.

ODE state y = [X, S, P, M, O2, V]:
    mu   = mu_max * S/(ks+S) * max(0, 1 - P/P_max)^3 * O2/(K_O2+O2)
    dX   = mu*X - D*X
    dS   = -Y_inv*(alpha*mu*X + beta*X) + km*M + D*(S_f - S)
    dP   =  alpha*mu*X + beta*X - D*P
    dM   = -km*M + D*(M_f - M)
    dO2  = kLa*(O2_sat - O2) - q_O2_max*f_O2*X - D*O2
    dV   = F                                  (D = F/V, F = dilution*V0)
with kLa = a*(P_g_over_V)^b*(v_s)^c * 3600  [1/h].
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.integrate import odeint
from scipy.optimize import least_squares

from fermdocs_optimize.schema import Candidate

C_FEED_TOTAL = 300.0  # feed substrate concentration [g/L]
FIT_SPECIES = ("X", "S", "P", "M")
PARAM_ORDER = ("mu_max", "ks", "P_max", "Y_inv", "alpha", "beta", "km")
# Plausible bounds for the scipy fit (standard kinetic ranges).
_LB = np.array([0.01, 0.01, 50.0, 0.1, 0.001, 0.001, 1e-5])
_UB = np.array([1.0, 50.0, 200.0, 10.0, 5.0, 2.0, 1.0])
_X0 = np.array([0.28, 0.5, 101.75, 1.4, 1.75, 0.12, 0.02])

# Default oxygen / mass-transfer physics (fixed, not fitted). Matches the lab
# singlezone config; overridable per process family.
DEFAULT_O2 = {"K_O2": 5e-4, "q_O2_max": 0.02, "O2_sat": 7e-3}
DEFAULT_KLA = {"a": 0.0083, "b": 0.62, "c": 0.49, "P_g_over_V": 1000.0, "v_s": 0.0015}


def _compute_kla(kla: dict) -> float:
    return kla["a"] * (kla["P_g_over_V"] ** kla["b"]) * (kla["v_s"] ** kla["c"]) * 3600.0


def _rhs(y, t, p, F, S_f, M_f, o2, kla_h):
    X, S, P, M, O2, V = y
    D = F / V
    f_O2 = O2 / (o2["K_O2"] + O2 + 1e-12)
    mu = p[0] * S / (p[1] + S + 1e-12) * max(0.0, 1.0 - P / p[2]) ** 3 * f_O2
    growth_assoc = p[4] * mu * X + p[5] * X  # alpha*mu*X + beta*X
    dX = mu * X - D * X
    dS = -p[3] * growth_assoc + p[6] * M + D * (S_f - S)
    dP = growth_assoc - D * P
    dM = -p[6] * M + D * (M_f - M)
    dO2 = kla_h * (o2["O2_sat"] - O2) - o2["q_O2_max"] * f_O2 * X - D * O2
    dV = F
    return [dX, dS, dP, dM, dO2, dV]


class MechanisticModel:
    """7-param kinetic model fit with scipy. Implements PredictiveModel."""

    def __init__(self, o2_params: dict | None = None, kla_params: dict | None = None):
        self._o2 = o2_params or dict(DEFAULT_O2)
        self._kla = kla_params or dict(DEFAULT_KLA)
        self._kla_h = _compute_kla(self._kla)
        self._params: np.ndarray | None = None

    # ---- internal: reconstruct per-batch operating conditions from data -----
    @staticmethod
    def _reconstruct(df: pd.DataFrame):
        batches = []
        for _, g in df.sort_values("t").groupby("batch"):
            g = g.reset_index(drop=True)
            t = g["t"].to_numpy(float)
            x0, s0, p0, m0, v0 = (float(g[c].iloc[0]) for c in ("X", "S", "P", "M", "V"))
            total = s0 + m0
            malt_frac = m0 / total if total > 0 else 0.0
            slope = np.polyfit(t, g["V"].to_numpy(float), 1)[0]  # V = V0 + D*V0*t
            D = max(slope / v0, 0.0)
            exp = {
                "t": t, "y0": [x0, s0, p0, m0, 0.0, v0],  # O2(0) set below
                "F": D * v0, "S_f": C_FEED_TOTAL * (1 - malt_frac),
                "M_f": C_FEED_TOTAL * malt_frac,
            }
            obs = np.column_stack([g[s].to_numpy(float) for s in FIT_SPECIES])
            batches.append((exp, obs))
        return batches

    def _simulate(self, exp, p):
        y0 = list(exp["y0"]); y0[4] = self._o2["O2_sat"]
        return odeint(_rhs, y0, exp["t"],
                      args=(p, exp["F"], exp["S_f"], exp["M_f"], self._o2, self._kla_h))

    def _residuals(self, theta, batches):
        res = []
        for exp, obs in batches:
            try:
                sol = self._simulate(exp, theta)
            except Exception:
                res.append(np.full(obs.size, 1e3)); continue
            pred = sol[:, :4]  # X,S,P,M are the first four states
            res.append(((pred - obs) / np.maximum(np.abs(obs), 1.0)).ravel())
        return np.concatenate(res)

    # ---- PredictiveModel interface -----------------------------------------
    def fit(self, observations: pd.DataFrame) -> dict[str, float]:
        batches = self._reconstruct(observations)
        sol = least_squares(self._residuals, _X0, bounds=(_LB, _UB), method="trf",
                            args=(batches,), xtol=1e-8, ftol=1e-8, max_nfev=300)
        self._params = sol.x
        return self._r2(batches)

    def _r2(self, batches) -> dict[str, float]:
        preds = {s: [] for s in FIT_SPECIES}; obss = {s: [] for s in FIT_SPECIES}
        for exp, obs in batches:
            sol = self._simulate(exp, self._params)
            for j, s in enumerate(FIT_SPECIES):
                preds[s].append(sol[:, j]); obss[s].append(obs[:, j])
        out = {}
        for s in FIT_SPECIES:
            p = np.concatenate(preds[s]); o = np.concatenate(obss[s])
            ss_res = float(np.sum((o - p) ** 2)); ss_tot = float(np.sum((o - o.mean()) ** 2))
            out[s] = 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")
        return out

    def predict_peak_titer(self, candidate: Candidate, *, v0: float,
                           t_end: float = 75.0, n: int = 76) -> float:
        if self._params is None:
            raise RuntimeError("model not fit")
        k = candidate.knobs()
        s0 = k["total_sub"] * (1 - k["malt_frac"]); m0 = k["total_sub"] * k["malt_frac"]
        exp = {"t": np.linspace(0, t_end, n),
               "y0": [k["biomass"], s0, 0.0, m0, 0.0, v0],
               "F": k["dilution"] * v0,
               "S_f": C_FEED_TOTAL * (1 - k["malt_frac"]),
               "M_f": C_FEED_TOTAL * k["malt_frac"]}
        try:
            sol = self._simulate(exp, self._params)
            return float(np.max(sol[:, 2]))  # P is state index 2
        except Exception:
            return -1e6

    @property
    def fitted_params(self) -> dict[str, float]:
        if self._params is None:
            return {}
        return {k: float(v) for k, v in zip(PARAM_ORDER, self._params)}

    # ---- transparency: what equations the agent's model uses, and how ---------
    @classmethod
    def model_card(cls) -> dict:
        """Human-readable governing equations + method. Callable without fitting,
        so the UI can show 'how the agent is using the model' up front."""
        return {
            "kind": "equations",
            "title": "Mechanistic kinetic model (7-parameter ODE)",
            "state": "y = [X biomass, S substrate, P product, M maltose, O2, V volume]",
            "equations": [
                "mu  = mu_max · S/(ks+S) · max(0, 1 − P/P_max)^3 · O2/(K_O2+O2)   (Monod growth × product inhibition × O2 limitation)",
                "dX/dt = mu·X − D·X",
                "dS/dt = −Y_inv·(alpha·mu·X + beta·X) + km·M + D·(S_f − S)",
                "dP/dt =  alpha·mu·X + beta·X − D·P                              (growth- + non-growth-associated formation)",
                "dM/dt = −km·M + D·(M_f − M)                                     (maltose uptake)",
                "dO2/dt = kLa·(O2_sat − O2) − q_O2_max·f_O2·X − D·O2",
                "dV/dt = F,   D = F/V,   F = dilution·V0",
                "kLa = a·(P_g_over_V)^b·(v_s)^c · 3600  [1/h]",
            ],
            "fitted_parameters": list(PARAM_ORDER),
            "fixed_physics": {"o2": dict(DEFAULT_O2), "kla": dict(DEFAULT_KLA)},
            "method": ("scipy.optimize.least_squares (TRF, bounded) over the 7 kinetic "
                       "parameters; the ODE is integrated per batch with scipy.integrate.odeint "
                       "and fit to measured X,S,P,M. Parameter values are LEARNED from data — "
                       "the model never reads the simulator's true parameters."),
            "param_bounds": {k: [float(lb), float(ub)]
                             for k, lb, ub in zip(PARAM_ORDER, _LB, _UB)},
        }

    def fit_log(self, r2_by_species: dict[str, float], n_batches: int) -> dict:
        """A 'fit' log entry: what the model learned this fit and how well."""
        return {
            "kind": "fit",
            "title": f"Fit mechanistic ODE on {n_batches} batch(es)",
            "detail": ("least_squares over 7 kinetic params; per-species R² on the "
                       "fit data below."),
            "method": "scipy.optimize.least_squares (TRF, bounded), odeint per batch",
            "fitted_params": self.fitted_params,
            "r2_by_species": {k: round(float(v), 4) for k, v in r2_by_species.items()},
        }
