---
name: fit-mechanistic-model
description: >
  Fit a mechanistic kinetic model (Monod, Contois, Andrews/Haldane, mass-action,
  etc.) to fermentation time-series data and recover interpretable rate parameters.
  Use when the reaction mechanism is known or hypothesizable. Preferred over hybrid.
  For unknown mechanisms use fit-surrogate-model.
---

## Purpose

Recover interpretable kinetic rate constants (µ_max, Ks, P_max, …) from
concentration time-series by differentiating through the ODE solver with JAX autodiff.

## When to use

- Mechanism is known or can be hypothesized (Monod growth, substrate inhibition, etc.)
- Interpretable parameters are needed for the downstream report
- Extrapolation beyond the fit window is important

## When NOT to use

- Mechanism is completely unknown → use **fit-surrogate-model**
- Pure surrogate already fits well → do not add a hybrid on top

---

## Build and fit

```python
import numpy as np, jax
jax.config.update("jax_enable_x64", True)   # MUST be first (see conventions C1)

from brewtwin.species import ChemicalSpecies, BiologicalSpecies
from brewtwin.reactions.reaction import Reaction
from brewtwin.reactions.network import ReactionNetwork
from brewtwin.rate_models.kinetic import Monod, Concentration, Constant
from brewtwin.rate_models.composite import CompositeRateLaw
from brewtwin.reactors.batch import BatchReactor
from brewtwin.data.schemas import Trajectory          # data-schema Trajectory (C2)
from brewtwin.data.observables import from_variable
from brewtwin.fitting.hybrid_fit import fit


def build_growth(mu_max, Ks, Y=0.5):
    """Monod batch growth: rate = mu_max * S/(Ks+S) * X"""
    X = BiologicalSpecies("X", conc=0.1)   # conc = initial condition (C3)
    S = ChemicalSpecies("S", conc=10.0)
    net = ReactionNetwork("growth_net")
    net.add_species(X)
    net.add_species(S)
    rate = CompositeRateLaw(
        Constant(mu_max),          # mu_max: trainable scalar multiplier
        Monod(S, Ks=Ks),           # S/(Ks+S) saturation
        Concentration(X),          # proportional to biomass
    )
    net.add_reaction(Reaction(
        name="growth",
        stoichiometry={"S": -1.0 / Y, "X": 1.0},
        rate_model=rate,
    ))
    return BatchReactor(net)


# Prepare data — use data-schema Trajectory, not solver-output Trajectory (C2)
traj = Trajectory.from_dense(
    t=t_array,                            # numpy array, ascending
    concentrations={"X": x_obs, "S": s_obs},
)

# Build reactor with initial guess (deliberately off from truth)
reactor = build_growth(mu_max=0.2, Ks=3.0)

result = fit(
    reactor,
    traj,
    [from_variable("X"), from_variable("S")],   # C4: bare reactor → no prefix
    t_span=(float(t_array[0]), float(t_array[-1])),
    solver="kvaerno5",      # C5: stiff-safe
    n_adam=300,
    lr_adam=0.05,
    n_lbfgs=0,
    rtol=1e-7,
    atol=1e-9,
    max_steps=200_000,
    progress=True,
)
fitted = result.meta["fitted_model"]   # BatchReactor with trained params
```

---

## Read fitted parameters

**CRITICAL**: `result.param_estimates` is `{}` for `CompositeRateLaw` — traverse
factors in the same order they were passed to `CompositeRateLaw(...)` (C6):

```python
rate = list(fitted.network.reactions)[0].rate_model   # CompositeRateLaw
mu_max_fit = float(rate.factors[0].value)             # Constant.value
Ks_fit     = float(rate.factors[1].Ks)               # Monod.Ks
# factors[2] is Concentration — no trainable parameter

print(f"mu_max = {mu_max_fit:.4f} 1/h")
print(f"Ks     = {Ks_fit:.4f} g/L")
print(f"loss history: {result.loss_history[0]:.4e} → {result.loss_history[-1]:.4e}")
```

For **`MassAction`** rate models (chemical kinetics with `log_scale=True`):
```python
# result.param_estimates IS populated: {"_.<reaction>.k": value}
k_fit = list(result.param_estimates.values())[0]
```

---

## Predict

```python
from brewtwin.solvers.jax_solver import JaxSolver

t_pred = np.linspace(0.0, 24.0, 200)
sim = JaxSolver("kvaerno5", rtol=1e-7, atol=1e-9, max_steps=200_000).solve(
    fitted,
    t_span=(0.0, 24.0),
    t_eval=t_pred,
)
# sim.y    → numpy array (T, n_species)
# sim.variables → ["X", "S"]  (column order)
X_pred = np.array(sim.y[:, sim.variables.index("X")])
S_pred = np.array(sim.y[:, sim.variables.index("S")])
```

---

## Rate model menu

Available in `brewtwin.rate_models.kinetic`:

| Class | Formula | Parameters |
|-------|---------|------------|
| `Constant(v)` | `v` | `value` |
| `Concentration(sp)` | `[sp]` | none |
| `Monod(sp, Ks)` | `S/(Ks+S)` | `Ks` |
| `ContoisKinetics(sp, bio, Kc)` | `S/(Kc·X+S)` | `Kc` |
| `AndrewsKinetics(sp, Ks, Ki)` | `S/(Ks+S+S²/Ki)` | `Ks`, `Ki` |
| `ProductInhibition(sp, P_max, n)` | `max(0, 1-P/P_max)^n` | `P_max`, `n` |
| `SubstrateInhibition(sp, Ki)` | `Ki/(Ki+S)` | `Ki` |
| `MassAction(k, species_orders)` | `k·∏[Si]^ni` | `k` (use `log_scale=True` for k spanning decades) |

Combine with `CompositeRateLaw(*factors)` (product) or `AdditiveRateLaw(*terms)` (sum).

---

## Gotchas

- Initial guess must be on a gradient slope; a guess in a dynamically-insensitive
  region (e.g. k₂ so large that B is instantly consumed regardless of its value)
  will produce zero gradients and no movement.
- `CompositeRateLaw` factor order determines `factors[i]` indices — keep it consistent
  between `build_*` and parameter readback.
- `from_variable("X")` is correct for a bare `BatchReactor`; add the reactor id
  only when using `ReactorNetwork`.
- For rate constants spanning >3 orders of magnitude use `MassAction(k, ..., log_scale=True)`.
