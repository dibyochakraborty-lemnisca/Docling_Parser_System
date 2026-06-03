# brewtwin Skills — Fermentation-Expert Agent

Reference index for an agent that generates and runs brewtwin code to fit models,
make predictions, and interpret results.

---

## Model-selection hierarchy

Use this to choose a skill:

```
Is the kinetic mechanism known or hypothesizable?
├── YES → fit-mechanistic-model     interpretable params, best extrapolation   ◄ PREFER
└── NO  → fit-surrogate-model       black box                                  ◄ PREFER
          ├── Neural ODE  → default: continuous dynamics, handles irregular t
          └── LSTM        → long or regularly-sampled sequences (>200 steps)

Partial mechanism + one unknown rate term?  → fit-hybrid-model                 ◄ LAST RESORT
```

**Rule**: prefer mechanistic or surrogate. Use hybrid only when a pure mechanistic
model is structurally incomplete *and* the known structure is worth preserving.
Within surrogates, default to Neural ODE; use LSTM for long sequences.

---

## Skills

| Skill | Use when |
|-------|----------|
| [fit-mechanistic-model](fit-mechanistic-model/SKILL.md) | mechanism known/hypothesizable; interpretable k's wanted |
| [fit-surrogate-model](fit-surrogate-model/SKILL.md) | mechanism unknown; neural ODE or LSTM |
| [fit-hybrid-model](fit-hybrid-model/SKILL.md) | partial mechanism + unknown rate term (last resort) |
| [analyze-and-interpret](analyze-and-interpret/SKILL.md) | assess fit quality; emit structured report |

---

## Installation

brewtwin is a local package (not on PyPI). Python ≥ 3.11.

```bash
cd /path/to/brew-twin        # directory containing pyproject.toml
pip install -e .             # editable install; pulls jax, diffrax, equinox, optax, tqdm, …

python -c "import brewtwin; print('brewtwin', brewtwin.__version__)"
```

GPU optional: `pip install -e ".[gpu]"`. CPU is sufficient for all skills.

---

## Shared conventions

**Read this before using any skill.**

### C1 — Enable float64 first
```python
import jax
jax.config.update("jax_enable_x64", True)   # MUST be before any heavy JAX call
```
Required for stiff kinetics tolerances and to avoid dtype mismatches between
surrogate weights and data arrays.

### C2 — Two different `Trajectory` classes (do not confuse)
- `brewtwin.data.schemas.Trajectory` — **experimental data**; input to `fit()` and
  `fit_surrogate()`. Build with `Trajectory.from_dense(t=..., concentrations={...})`.
- `brewtwin.solvers.base.Trajectory` — **solver output** from `JaxSolver.solve()`;
  has `.t`, `.y` (shape `(T, n_species)`), `.variables` (list of names).

Convert solver output to data schema: `Trajectory.from_dense(t=traj.t, concentrations={...})`.

### C3 — Species `conc=` sets the initial condition
```python
BiologicalSpecies("X", conc=0.1)   # biomass starts at 0.1 g/L
ChemicalSpecies("S", conc=10.0)    # substrate starts at 10.0 g/L
```

### C4 — Observable reactor id
- Bare `BatchReactor` → `from_variable("X")` (default id `"_"`)
- Multi-compartment `ReactorNetwork` → `from_variable("X", "fermenter")`

### C5 — Solver choice
| Context | Solver |
|---------|--------|
| Mechanistic / hybrid fitting & simulation | `"kvaerno5"` (stiff-safe, default) |
| Neural ODE training | `"tsit5"` + `diffrax.Tsit5()` (explicit — never implicit) |

Never use `Kvaerno5` on a learned vector field: its Newton solver receives non-finite
input during training and raises `EquinoxRuntimeError`.

### C6 — Reading fitted mechanistic parameters (CRITICAL)
`result.param_estimates` is **empty `{}`** for `CompositeRateLaw` models (the
normal form of Monod growth). Read parameters by traversing the rate-model factors:

```python
rate = list(fitted.network.reactions)[0].rate_model   # CompositeRateLaw
mu_max = float(rate.factors[0].value)                 # Constant.value
Ks     = float(rate.factors[1].Ks)                    # Monod.Ks
```

`param_estimates` is only populated for flat models like `MassAction`
(`{"_.<reaction>.k": value}`).

---

## Gotcha quick-reference

| # | Trap | Fix |
|---|------|-----|
| 1 | float64 not enabled | `jax.config.update("jax_enable_x64", True)` first |
| 2 | wrong Trajectory class | use `from_dense` for fitting input |
| 3 | `param_estimates` empty | traverse `.factors[i].<attr>` on CompositeRateLaw |
| 4 | implicit solver on NeuralODE | use `"tsit5"` / `diffrax.Tsit5()` only |
| 5 | stiff mechanistic diverges | use `"kvaerno5"` |
| 6 | wrong observable reactor id | `from_variable("X")` for bare reactor (no prefix) |
| 7 | LSTM predict crashes | pass `jnp.zeros((n_steps, 1))` when `input_size=0` |
| 8 | state_size mismatch | `state_size == len(species)` |
| 9 | EquinoxRateModel size wrong | MLP `in_size == len(input_features)`, `out_size=1` |
| 10 | rate constants span many decades | `MassAction(..., log_scale=True)` |
