"""A predictive model built from an agent-proposed ModelSpec.

Same contract as the fixed MechanisticModel (fit on data → predict peak titer),
but the equations come from the spec, not from hardcoded source. It reuses the
mechanistic model's per-batch condition reconstruction and fixed-physics
constants so a candidate sees exactly the same inputs — the only thing that
varies is the structure the agent proposed.

Integrity: fits on observation data ONLY. Never reads the oracle's parameters.
"""
from __future__ import annotations

import os
import warnings

import numpy as np
import pandas as pd
from scipy.integrate import odeint
from scipy.optimize import least_squares

# Per-integration step cap. Stiff LLM-proposed structures otherwise grind to the
# old 2000-step ceiling on every one of hundreds of fit/search evaluations, which
# is what makes a run take hours. 500 is scipy's own default and is exactly what
# the trusted hand-written MechanisticModel runs at, so healthy structures are
# unaffected; only pathological ones bail sooner and get scored worst.
_MXSTEP = int(os.environ.get("FERMDOCS_OPTIMIZE_ODE_MXSTEP", "500"))

from fermdocs_optimize.discovery.expr import CompiledModel, compile_spec
from fermdocs_optimize.discovery.spec import ModelSpec
from fermdocs_optimize.models.mechanistic import (
    C_FEED_TOTAL,
    DEFAULT_KLA,
    DEFAULT_O2,
    FIT_SPECIES,
    MechanisticModel,
    _compute_kla,
)
from fermdocs_optimize.schema import Candidate


class CandidateModel:
    """Compile + fit an agent-proposed kinetic structure. Implements PredictiveModel."""

    def __init__(self, spec: ModelSpec, o2_params: dict | None = None,
                 kla_params: dict | None = None):
        self.spec = spec
        self._o2 = o2_params or dict(DEFAULT_O2)
        self._kla = kla_params or dict(DEFAULT_KLA)
        self._kla_h = _compute_kla(self._kla)
        self._const = {
            "K_O2": self._o2["K_O2"], "q_O2_max": self._o2["q_O2_max"],
            "O2_sat": self._o2["O2_sat"], "kLa": self._kla_h, "C_FEED": C_FEED_TOTAL,
        }
        self._compiled: CompiledModel = compile_spec(
            spec.param_names(), spec.aux, spec.odes)
        self._theta: np.ndarray | None = None

    # ---- integration --------------------------------------------------------
    def _cond(self, exp) -> dict:
        return {"F": exp["F"], "S_f": exp["S_f"], "M_f": exp["M_f"],
                "v0": float(exp["y0"][5])}

    def _simulate(self, exp, theta):
        y0 = list(exp["y0"]); y0[4] = self._o2["O2_sat"]
        cond = self._cond(exp)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            return odeint(self._compiled.rhs, y0, exp["t"],
                          args=(theta, cond, self._const), mxstep=_MXSTEP)

    def _integrable(self, batches, theta, *, min_ok_frac: float = 0.5) -> bool:
        """Cheap pre-flight: can this structure even be integrated at `theta`?

        A stiff/divergent structure otherwise costs up to max_nfev * n_batches
        full integrations before least_squares gives up. This runs ONE capped
        integration per batch and bails the whole candidate if too many fail to
        converge — milliseconds to reject garbage instead of minutes."""
        ok = 0
        for exp, _ in batches:
            y0 = list(exp["y0"]); y0[4] = self._o2["O2_sat"]
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                sol, info = odeint(
                    self._compiled.rhs, y0, exp["t"],
                    args=(theta, self._cond(exp), self._const),
                    mxstep=_MXSTEP, full_output=True)
            if str(info.get("message", "")).startswith("Integration successful") \
                    and np.all(np.isfinite(sol)):
                ok += 1
        return ok >= max(1, int(min_ok_frac * len(batches)))

    def _residuals(self, theta, batches):
        res = []
        for exp, obs in batches:
            try:
                sol = self._simulate(exp, theta)
                pred = sol[:, :4]
                if not np.all(np.isfinite(pred)):
                    raise ValueError("non-finite")
            except Exception:
                res.append(np.full(obs.size, 1e3)); continue
            res.append(((pred - obs) / np.maximum(np.abs(obs), 1.0)).ravel())
        return np.concatenate(res)

    # ---- PredictiveModel interface -----------------------------------------
    def fit(self, observations: pd.DataFrame) -> dict[str, float]:
        batches = MechanisticModel._reconstruct(observations)
        p = self.spec.params
        x0 = np.array([p[n].init for n in self.spec.param_names()], float)
        lb = np.array([p[n].lb for n in self.spec.param_names()], float)
        ub = np.array([p[n].ub for n in self.spec.param_names()], float)
        if not self._integrable(batches, x0):
            raise RuntimeError(
                "candidate structure not integrable (stiff/non-convergent at init)")
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            sol = least_squares(self._residuals, x0, bounds=(lb, ub), method="trf",
                                args=(batches,), xtol=1e-8, ftol=1e-8, max_nfev=300)
        self._theta = sol.x
        return self._r2(batches)

    def _r2(self, batches) -> dict[str, float]:
        preds = {s: [] for s in FIT_SPECIES}; obss = {s: [] for s in FIT_SPECIES}
        for exp, obs in batches:
            sol = self._simulate(exp, self._theta)
            for j, s in enumerate(FIT_SPECIES):
                preds[s].append(sol[:, j]); obss[s].append(obs[:, j])
        out = {}
        for s in FIT_SPECIES:
            pr = np.concatenate(preds[s]); ob = np.concatenate(obss[s])
            ss_res = float(np.sum((ob - pr) ** 2))
            ss_tot = float(np.sum((ob - ob.mean()) ** 2))
            out[s] = 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")
        return out

    def predict_P_trajectory(self, candidate: Candidate, *, v0: float,
                             t_end: float = 75.0, n: int = 76) -> np.ndarray:
        """Full product (P) trajectory for one operating point — used to score
        the candidate against the oracle on both peak and shape."""
        if self._theta is None:
            raise RuntimeError("model not fit")
        k = candidate.knobs()
        s0 = k["total_sub"] * (1 - k["malt_frac"]); m0 = k["total_sub"] * k["malt_frac"]
        exp = {"t": np.linspace(0, t_end, n),
               "y0": [k["biomass"], s0, 0.0, m0, 0.0, v0],
               "F": k["dilution"] * v0,
               "S_f": C_FEED_TOTAL * (1 - k["malt_frac"]),
               "M_f": C_FEED_TOTAL * k["malt_frac"]}
        try:
            p = self._simulate(exp, self._theta)[:, 2]
            return p if np.all(np.isfinite(p)) else np.full(n, -1e6)
        except Exception:
            return np.full(n, -1e6)

    def predict_peak_titer(self, candidate: Candidate, *, v0: float,
                           t_end: float = 75.0, n: int = 76) -> float:
        return float(np.max(self.predict_P_trajectory(
            candidate, v0=v0, t_end=t_end, n=n)))

    @property
    def fitted_params(self) -> dict[str, float]:
        if self._theta is None:
            return {}
        return {n: float(v) for n, v in zip(self.spec.param_names(), self._theta)}
