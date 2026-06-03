---
name: fit-hybrid-model
description: >
  Fit a hybrid mechanistic + ML rate model when a pure mechanistic model is
  structurally incomplete but the known structure is worth preserving. LAST RESORT:
  try fit-mechanistic-model and fit-surrogate-model first.
---

## Purpose

Augment a mechanistic rate law with a small neural correction term that absorbs
unknown effects (unmeasured inhibitors, unmodelled interactions). The mechanistic
part remains interpretable; the ML part is a black box.

## When to use

- Mechanistic model has the right structure but consistently misfits in one regime
- Interpretable parameters (µ_max, Ks) are needed alongside the correction
- A pure surrogate would discard valuable structural prior knowledge

## When NOT to use (prefer these first)

- Try **fit-mechanistic-model** → if it fits, use it
- Try **fit-surrogate-model** → if interpretability isn't needed, use it
- Only fall back here when both are inadequate

---

## Two hybrid forms

| Class | Formula | When |
|-------|---------|------|
| `RelativeHybridRateModel(mech, residual)` | `r = r_mech · (1 + δ)` | relative multiplicative correction; δ=0 recovers pure mech |
| `AdditiveHybridRateModel(mech, correction)` | `r = r_mech + δ` | absolute additive correction |

---

## Build and fit

Uses the **same `fit()` pipeline** as mechanistic — NN weights and mechanistic
parameters are co-optimized in a single gradient pass.

```python
import jax, jax.random as jr, equinox as eqx
jax.config.update("jax_enable_x64", True)   # MUST be first (C1)

from brewtwin.species import ChemicalSpecies, BiologicalSpecies
from brewtwin.reactions.reaction import Reaction
from brewtwin.reactions.network import ReactionNetwork
from brewtwin.rate_models.kinetic import Monod, Concentration, Constant
from brewtwin.rate_models.composite import CompositeRateLaw
from brewtwin.rate_models.ml import EquinoxRateModel
from brewtwin.rate_models.hybrid import RelativeHybridRateModel
from brewtwin.reactors.batch import BatchReactor
from brewtwin.data.schemas import Trajectory          # data-schema Trajectory (C2)
from brewtwin.data.observables import from_variable
from brewtwin.fitting.hybrid_fit import fit


def build_hybrid(mu_max_init=0.3, Ks_init=2.0):
    X = BiologicalSpecies("X", conc=0.1)
    S = ChemicalSpecies("S", conc=10.0)
    net = ReactionNetwork("hybrid_net")
    net.add_species(X)
    net.add_species(S)

    # Mechanistic base rate
    mech = CompositeRateLaw(
        Constant(mu_max_init),
        Monod(S, Ks=Ks_init),
        Concentration(X),
    )

    # ML correction: MLP residual
    # in_size MUST equal len(input_features) (C9); out_size=1 always
    input_features = ["S", "X"]
    mlp = eqx.nn.MLP(
        in_size=len(input_features),
        out_size=1,
        width_size=16,
        depth=2,
        key=jr.key(0),
    )
    residual = EquinoxRateModel(mlp, input_features=input_features, name="residual")

    rate = RelativeHybridRateModel(mech, residual)  # r = r_mech * (1 + δ_ml)

    net.add_reaction(Reaction(
        name="growth",
        stoichiometry={"S": -2.0, "X": 1.0},
        rate_model=rate,
    ))
    return BatchReactor(net)


traj = Trajectory.from_dense(t=t_array, concentrations={"X": x_obs, "S": s_obs})

result = fit(
    build_hybrid(),
    traj,
    [from_variable("X"), from_variable("S")],   # C4: bare reactor → no prefix
    t_span=(float(t_array[0]), float(t_array[-1])),
    solver="kvaerno5",      # C5: stiff-safe
    n_adam=300,
    lr_adam=0.02,
    n_lbfgs=0,
    rtol=1e-7,
    atol=1e-9,
    max_steps=200_000,
    progress=True,
)
fitted = result.meta["fitted_model"]
```

---

## Read fitted parameters

```python
hyb = list(fitted.network.reactions)[0].rate_model   # RelativeHybridRateModel
# Mechanistic part — interpretable (same as CompositeRateLaw traversal, C6)
mu_max_fit = float(hyb.mechanistic.factors[0].value)  # Constant.value
Ks_fit     = float(hyb.mechanistic.factors[1].Ks)     # Monod.Ks
# ML part (hyb.residual) — not interpretable; report only its effect on fit quality
```

---

## Predict

Identical to the mechanistic skill:

```python
from brewtwin.solvers.jax_solver import JaxSolver
import numpy as np

sim = JaxSolver("kvaerno5", rtol=1e-7, atol=1e-9, max_steps=200_000).solve(
    fitted, t_span=(0.0, 24.0), t_eval=np.linspace(0.0, 24.0, 200)
)
X_pred = np.array(sim.y[:, sim.variables.index("X")])
S_pred = np.array(sim.y[:, sim.variables.index("S")])
```

---

## Gotchas

- `EquinoxRateModel` `in_size` must equal `len(input_features)` exactly (C9);
  `out_size` must be `1`.
- Every name in `input_features` must be a species name or an environment variable
  declared in the `ReactionNetwork`.
- The ML residual is not interpretable — never report `δ` as a mechanism. Report
  only the mechanistic factor values and the reduction in fit error.
- `hyb.mechanistic.factors[i]` traversal: index order must match
  `CompositeRateLaw(*args)` call order (same rule as mechanistic skill, C6).
- Use `AdditiveHybridRateModel` when the residual is expected to be small and
  additive; use `RelativeHybridRateModel` when a multiplicative fold-change is more
  natural. Both are fitted identically.
