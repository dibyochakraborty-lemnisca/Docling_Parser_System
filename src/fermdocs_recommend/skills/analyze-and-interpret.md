---
name: analyze-and-interpret
description: >
  Assess fit quality, sanity-check parameters and predictions, and emit a
  structured scientific report for downstream agents. Use after any fit-* skill.
---

## Purpose

Turn raw fit results into a structured interpretation: goodness-of-fit metrics,
parameter plausibility, prediction summary, and a JSON-serialisable report dict
for the orchestrator.

---

## Fit quality metrics

NaN-safe per-species RMSE and R²:

```python
import numpy as np

def fit_metrics(y_pred: np.ndarray, y_obs: np.ndarray, species: list[str]) -> dict:
    """Per-species RMSE and R² over finite (non-NaN) observations.

    Parameters
    ----------
    y_pred : (T, n_species) predicted trajectory
    y_obs  : (T, n_species) observed data (NaN = not observed at that time)
    species: ordered species name list

    Returns
    -------
    dict keyed by species name, each with {"rmse": float, "r2": float}
    """
    out = {}
    for j, sp in enumerate(species):
        m = np.isfinite(y_obs[:, j])
        if not m.any():
            out[sp] = {"rmse": float("nan"), "r2": float("nan")}
            continue
        resid  = y_pred[m, j] - y_obs[m, j]
        rmse   = float(np.sqrt(np.mean(resid ** 2)))
        ss_tot = float(np.sum((y_obs[m, j] - y_obs[m, j].mean()) ** 2))
        r2     = float(1.0 - np.sum(resid ** 2) / ss_tot) if ss_tot > 0 else float("nan")
        out[sp] = {"rmse": rmse, "r2": r2}
    return out
```

---

## Parameter plausibility check

Typical literature ranges for batch fermentation:

| Parameter | Typical range | Unit |
|-----------|---------------|------|
| µ_max | 0.05 – 2.0 | 1/h |
| Ks (substrate half-saturation) | 0.01 – 5.0 | g/L |
| Y (biomass yield) | 0.3 – 0.6 | g_X/g_S |
| P_max (product inhibition) | 20 – 120 | g/L |

```python
def parameter_plausibility(params: dict) -> dict:
    """Check fitted parameters against typical fermentation ranges.
    Returns a dict with the same keys, each containing the value and a plausible flag.
    """
    ranges = {
        "mu_max": (0.05, 2.0),
        "Ks":     (0.01, 5.0),
        "P_max":  (20.0, 120.0),
    }
    result = {}
    for name, value in params.items():
        lo, hi = ranges.get(name, (0.0, float("inf")))
        result[name] = {"value": value, "plausible": bool(lo <= value <= hi)}
    return result
```

---

## Structured report

Build this dict and return it to the orchestrator. All fields are JSON-serialisable.

```python
import numpy as np

def build_report(
    model_type: str,           # "mechanistic", "surrogate", or "hybrid"
    fitted_params: dict,       # e.g. {"mu_max": 0.45, "Ks": 0.8}  (mechanistic/hybrid)
    loss_history: list,        # result.loss_history or result.train_loss_history
    y_pred: np.ndarray,        # (T, n_species) predictions on fit window
    y_obs: np.ndarray,         # (T, n_species) observed values (NaN where unobserved)
    t_pred: np.ndarray,        # time points of y_pred
    species: list[str],
    fit_window: tuple,         # (t_start, t_end)
    predict_window: tuple,     # (t_start, t_end)
    caveats: list[str] = (),
    recommended_next: str = "",
) -> dict:

    metrics = fit_metrics(y_pred, y_obs, species)
    plaus   = parameter_plausibility(fitted_params) if fitted_params else {}

    # Simple prediction summary: peak and time-to-exhaustion per species
    summary = {}
    for j, sp in enumerate(species):
        col = y_pred[:, j]
        idx_peak = int(np.argmax(col))
        summary[sp] = {
            "peak_value": float(col[idx_peak]),
            "peak_time":  float(t_pred[idx_peak]),
            "final_value": float(col[-1]),
        }

    return {
        "model_type": model_type,
        "fitted_parameters": plaus,
        "fit_quality": metrics,
        "final_loss": float(loss_history[-1]),
        "prediction_summary": summary,
        "fit_window":     {"t_start": fit_window[0],     "t_end": fit_window[1]},
        "predict_window": {"t_start": predict_window[0], "t_end": predict_window[1]},
        "caveats": list(caveats),
        "recommended_next": recommended_next,
    }
```

### Example usage after `fit-mechanistic-model`

```python
# (continuing from fit-mechanistic-model recipe)
t_pred = np.linspace(0.0, 24.0, 200)
sim = JaxSolver("kvaerno5", rtol=1e-7, atol=1e-9, max_steps=200_000).solve(
    fitted, t_span=(0.0, 24.0), t_eval=t_pred)
y_pred = np.array(sim.y)

t_obs, y_obs = traj.to_fit_arrays(["X", "S"])   # NaN-sparse observed matrix

report = build_report(
    model_type="mechanistic",
    fitted_params={"mu_max": mu_max_fit, "Ks": Ks_fit},
    loss_history=result.loss_history,
    y_pred=y_pred,
    y_obs=y_obs,
    t_pred=t_pred,
    species=["X", "S"],
    fit_window=(float(t_array[0]), float(t_array[-1])),
    predict_window=(0.0, 24.0),
    caveats=["extrapolation beyond fit window unvalidated"],
    recommended_next="collect denser samples during exponential phase",
)

import json
print(json.dumps(report, indent=2))
```

---

## Interpretation rules

1. **Parameters**: flag any value outside plausibility range; suggest likely cause
   (bad initial guess, insufficient data coverage, wrong model structure).
2. **Fit quality**: R² > 0.95 indicates a good fit; RMSE should be compared to the
   typical measurement error (±5–10% for most HPLC / OD assays).
3. **Surrogate and hybrid**: never report ML weights as mechanisms. State "black-box
   correction" with only the fit-quality improvement as evidence.
4. **Extrapolation**: mechanistic models extrapolate better; always state the fit
   window in the report and flag predictions outside it.
5. **Hybrid residual effect**: compare RMSE of the hybrid vs a pure mechanistic fit
   on the same data — justify the added complexity only if the improvement is clear.

---

## Gotchas

- `traj.to_fit_arrays(species)` returns a `(T, n_species)` numpy array with `NaN`
  where a species was not observed at a given time. `fit_metrics` handles this.
- `y_pred` from `JaxSolver.solve` is the solver-output Trajectory `.y`; convert
  to numpy with `np.array(sim.y)`.
- For surrogate models the loss key is `result.train_loss_history` (not
  `result.loss_history`).
