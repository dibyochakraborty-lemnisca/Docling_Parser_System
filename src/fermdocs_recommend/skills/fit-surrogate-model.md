---
name: fit-surrogate-model
description: >
  Train a black-box surrogate (Neural ODE or LSTM) on fermentation concentration
  time-series when the mechanism is unknown. Prefer Neural ODE (continuous dynamics,
  handles irregular sampling); use LSTM for long or regularly-sampled sequences.
  Preferred over hybrid. For known mechanisms use fit-mechanistic-model.
---

## Purpose

Learn dynamics directly from data, with no assumed kinetic mechanism. The trained
surrogate can predict trajectories from any initial condition within the training
distribution.

## When to use

- Mechanism is unknown or too complex to write down
- Black-box prediction accuracy matters more than parameter interpretability

## When NOT to use

- Mechanism is known → use **fit-mechanistic-model** (interpretable, extrapolates better)
- Hybrid is being considered → try mechanistic or surrogate alone first

## Which surrogate

| Surrogate | When |
|-----------|------|
| **Neural ODE** (default) | continuous dynamics; irregular or log-spaced time grids |
| **LSTM** | long sequences (>200 steps), regularly-sampled, uniform Δt |
| CTGRUSurrogate | irregular Δt with explicit time-gap input |
| LatentODE | sparse / partially observed data |

---

## Neural ODE (default)

```python
import numpy as np, jax, jax.numpy as jnp, jax.random as jr, diffrax
jax.config.update("jax_enable_x64", True)   # MUST be first (C1)

from brewtwin.surrogates.neural_ode import NeuralODE
from brewtwin.surrogates.train import fit_surrogate
from brewtwin.data.schemas import Trajectory          # data-schema Trajectory (C2)

# Prepare data
traj = Trajectory.from_dense(
    t=t_array,                               # numpy array, ascending
    concentrations={"X": x_obs, "S": s_obs},
)
species = ["X", "S"]                         # order defines the state vector (C8)

# Build model — EXPLICIT solver (C5)
node = NeuralODE(
    state_size=len(species),   # must equal len(species) (C8)
    width=64,
    depth=3,
    key=jr.key(0),
    solver=diffrax.Tsit5(),    # EXPLICIT — never Kvaerno5 on learned field (C5)
    rtol=1e-4,
    atol=1e-6,
)

result = fit_surrogate(
    node, traj, species,
    solver="tsit5",            # explicit (C5)
    rtol=1e-4,
    atol=1e-6,
    max_steps=200_000,
    n_epochs=500,
    lr=3e-3,
    progress=True,
)
model = result.surrogate
print(f"loss: {result.train_loss_history[0]:.4e} → {result.train_loss_history[-1]:.4e}")
```

### Predict (Neural ODE)

```python
y0 = jnp.array([x0, s0])                              # initial condition
t_eval = jnp.asarray(np.linspace(0.0, 24.0, 100))
pred = model.rollout(y0, t_eval)                       # (len(t_eval), n_species)
X_pred = np.array(pred[:, species.index("X")])
S_pred = np.array(pred[:, species.index("S")])
```

---

## LSTM (long / regular sequences)

```python
import jax, jax.numpy as jnp, jax.random as jr
jax.config.update("jax_enable_x64", True)

from brewtwin.surrogates.rnn import LSTMSurrogate
from brewtwin.surrogates.train import fit_surrogate
from brewtwin.data.schemas import Trajectory

traj = Trajectory.from_dense(t=t_array, concentrations={"X": x_obs, "S": s_obs})
species = ["X", "S"]

lstm = LSTMSurrogate(
    state_size=len(species),   # must equal len(species) (C8)
    input_size=0,              # 0 = no time-varying control inputs
    hidden_size=64,
    key=jr.key(0),
)

result = fit_surrogate(lstm, traj, species, n_epochs=500, lr=1e-3, progress=True)
model = result.surrogate
```

### Predict (LSTM)

```python
# RNNs require a control-sequence argument even with input_size=0 (C7)
n_steps = len(t_eval) - 1
u = jnp.zeros((n_steps, 1))              # sentinel; ignored when input_size=0
pred = model.rollout(jnp.array([x0, s0]), u)   # shape (n_steps+1, n_species)
# Note: output has n_steps+1 rows (includes y0); first row is y0.
X_pred = np.array(pred[:, species.index("X")])
```

---

## Multiple replicates

Pass a list of `Trajectory` objects to train on multiple runs simultaneously:

```python
result = fit_surrogate(
    node,
    [traj_rep1, traj_rep2, traj_rep3],   # list of data-schema Trajectories
    species,
    solver="tsit5",
    n_epochs=500,
    lr=3e-3,
    progress=True,
)
```

---

## Gotchas

- Neural ODE: **always** `solver=diffrax.Tsit5()` at construction **and**
  `solver="tsit5"` in `fit_surrogate`. Using an implicit solver (Kvaerno) on a
  learned vector field causes `EquinoxRuntimeError` (NaN in Newton solver).
- `state_size` must exactly equal `len(species)`. Species order in the list defines
  the state vector column order.
- LSTM / GRU predict: pass `jnp.zeros((len(t_eval)-1, 1))` when `input_size=0`.
  The output length is `n_steps+1`, not `len(t_eval)` — index accordingly.
- float64 must be enabled before building the model (C1); dtype mismatch between
  model weights (float64) and data (float32) causes diffrax buffer errors.
