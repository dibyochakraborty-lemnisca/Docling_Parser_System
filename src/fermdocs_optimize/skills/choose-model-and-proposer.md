# Skill: choose-model-and-proposer

How to pick the `model` and `proposer` arguments to `run_optimization_loop`.

## Models (the agent's own cheap surrogate, fit on data)

- `"mechanistic"` (default) — a 7-parameter kinetic ODE fit with
  `scipy.optimize.least_squares`. Physically interpretable, extrapolates
  sensibly near the data, and is the right default for fed-batch fermentation.
  Use it unless you have a specific reason not to.

The model only guides the *proposals*; the oracle still judges every proposal,
so a mediocre fit costs extra rounds, not correctness. If `model_vs_oracle_r2`
stays low after augmentation, that is a signal to report low confidence — not to
distrust the achieved titer (the oracle ran it).

## Proposers (the search over the knob box)

- `"optimize"` (default) — global search (`scipy.differential_evolution`)
  maximizing predicted peak P, plus a few diverse perturbations. Best when the
  model is a reasonable guide and you want the strongest single point.
- `"grid"` — Latin-hypercube sweep, top-k by predicted titer. More robust when
  the model surface is rough or you want broad coverage early. A good warm-start
  for round 0, or a fallback when `"optimize"` keeps proposing the same corner.

## When to re-run

- Trajectory still climbing at `max_rounds` → same config, more `max_rounds`.
- `"optimize"` collapses onto one box corner with poor oracle agreement → try
  `"grid"` for broader coverage.
- Otherwise, one run is enough.
