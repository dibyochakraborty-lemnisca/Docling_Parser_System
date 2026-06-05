# Optimizer Agent — Architecture & Implementation Plan

**Branch:** `optimizer-agent` (off `Lemnisca-design-refactor`)
**Date:** 2026-06-04
**Status:** **Phase 0 + Phase 1 + Phase 2 DONE.** Grounded in the LABS repo
(`~/Downloads/lactic_acid-main`) + the `README for human.txt` workflow + the
provided config/data files.

## Progress log

- **Phase 0 spike (de-risked, all green):**
  - IPOPT-free oracle: `generate-batches --mech-params` runs without IPOPT;
    seed batch → peak P 103.7 g/L.
  - scipy 7-param fit (reusing LABS's exact ODE for the spike): held-out **test
    R²: P=0.83, S=0.81, M=0.96, X=0.73** — clears the 0.8 gate on the target.
  - one PROPOSE→SIMULATE round: optimizer found knobs → oracle-verified 108.8 g/L,
    and exposed the model-vs-oracle gap that motivates the refit loop.
- **Phase 1 MVP (`src/fermdocs_optimize/`, built + verified):**
  - Modules: `schema`, `models/{base,mechanistic}`, `simulators/{base,labs,stub}`,
    `proposers/{base,optimize,grid}`, `evaluate`, `loop`, `cli`. Registered
    `fermdocs-optimize` in pyproject.
  - Model is **self-contained** (own ODE re-impl), decoupled from LABS; reaches
    the oracle only via the `generate-batches` subprocess (`Simulator` seam).
  - **Stub unit tests 6/6 green** (loop runs+improves, in-box clamping,
    data-driven, never-reads-params integrity via open()-spy, refusal coherence,
    box validation).
  - **Real-LABS e2e green:** baseline 72.06 → **best 108.56 g/L (+36.5)** over 2
    rounds; active-learning augmentation fired (fit P R² 0.62→0.75 as oracle data
    folded in); converged on ΔP<2. Caught + fixed a real bug (LABS 1-based vs
    stub 0-based batch ids → order-based candidate↔batch mapping).
- **Phase 2 agentic shell (`agent.py` + `tools_bundle/` + skills, built + verified):**
  - `OptimizerAgent` runs a ReAct loop (same two-tool `tool_call`/`emit` shape as
    diagnose/recommend) over 5 gated tools: `get_experiment`, `get_box`,
    `get_skill`, `run_optimization_loop`, `submit_optimization`.
  - **The loop stays authoritative.** The LLM interprets the objective, picks the
    model/proposer (`MODEL_REGISTRY`/`PROPOSER_REGISTRY` — open-closed) and the
    round/convergence budget, and narrates — but `run_optimization_loop` fits +
    proposes + simulates on the oracle, so the achieved titer and best point come
    from the loop, not the LLM. Test `test_numbers_come_from_loop_not_narration`
    pins this (LLM claims 9999 g/L → output still the oracle's ~100).
  - **Integrity preserved end-to-end:** the `Simulator` is injected (the orchestrator
    builds `LABSSimulator` from the mech-params path); nothing in `agent.py`/tools
    reads the true params. `test_agent_never_reads_true_params` spies `open()`.
  - **Honest failure:** emit-before-loop → `no_loop_run` refusal; LLM error/step
    exhaustion finalizes any loop result or refuses; deterministic honesty suffix
    flags low model-vs-oracle agreement and "no improvement over baseline."
  - **Graceful degradation:** `provider=none` runs the deterministic loop directly
    (the core works without an LLM). `gemini`/`anthropic` clients in `llm_clients.py`.
  - Skills vendored under `src/fermdocs_optimize/skills/` (README injected into the
    system prompt; `optimize-titer` + `choose-model-and-proposer` served via
    `get_skill`); registered in pyproject force-include. CLI gains `--provider`.
  - **Tests 12/12 green** (6 loop + 6 agent), no LABS needed (StubSimulator).
- **Next:** Phase 3 (platform lift of shared model-fit/LLM seams + thin parent
  intent router: diagnostic vs optimizer).

---

## 1. What we're building

A new **child agentic system** — the *optimizer* — sibling to the existing
diagnostic system (fermdocs). Where fermdocs asks *"what went wrong, fix it,"*
the optimizer asks *"how do we push the target variable (product titer P) as high
as possible,"* even on a perfectly healthy experiment.

The lactic-acid case is the **first concrete instance** because it ships a
**simulator** (LABS) that acts as the ground-truth oracle. That makes this a
*closed-loop, simulator-in-the-loop* optimization, not just a one-shot
recommendation.

## 2. The loop (from `README for human.txt`, concretized)

This is model-based **active-learning optimization**: fit a cheap model, use it to
propose promising operating points, evaluate them on the expensive simulator,
fold surprises back into the model, repeat until titer stops improving.

```
 seed: train_data.csv (the only data the agent's model may fit on)
 ┌──────────────────────────────────────────────────────────────────────┐
 │ ROUND n                                                                │
 │                                                                        │
 │  1. FIT       fit the agent's OWN model on current training data       │
 │               (7-param mechanistic via scipy least_squares,            │
 │                or a LABS node/rnn surrogate). NEVER reads mech_params.  │
 │                         │                                              │
 │  2. PROPOSE   search the 4-knob box to MAXIMIZE predicted peak P:      │
 │               total_sub, malt_frac, dilution, biomass                  │
 │               (grid sweep OR scipy.optimize). Emit K candidate batches.│
 │                         │                                              │
 │  3. SIMULATE  write candidates into edit_config.json "batches",       │
 │     (ORACLE)  run LABS:                                                 │
 │               generate-batches --config edit_config.json \             │
 │                 --mech-params mech_params.json --output new_data.csv   │
 │               → ground-truth trajectories for the proposals.           │
 │                         │                                              │
 │  4. EVALUATE  R²/MAE of agent-model prediction vs new_data.            │
 │               if R² < 0.8 → append new_data to training data (model    │
 │                              was wrong there → it must learn).         │
 │               record best achieved peak P this round.                  │
 │                         │                                              │
 │  5. CONVERGE  if  P_best(n) − P_best(n−1) < ΔP_thresh (e.g. 2 g/L)     │
 │                  → STOP. else → refit (back to 1).                     │
 └──────────────────────────────────────────────────────────────────────┘
 output: best operating point (4 knobs), achieved + predicted titer,
         the improvement trajectory P_0→P_n, final model fit quality.
```

**Integrity invariant (from the README):** the agent's *model* is fit only on
data; it must never read `mech_params.json` (the answer key). The orchestration
*passes the file path* to the LABS subprocess (the simulator needs it) but never
ingests its values into the agent's model. This is the same "earn trust, don't
cheat" discipline as the rest of the system.

## 3. How LABS plugs in (verified against the repo)

- **Simulator** = `generate-batches --config <explicit_config> --mech-params
  mech_params.json --output new_data.csv` (or the API `run_explicit_batches(...)`).
  With `--mech-params`, LABS skips fitting and uses those exact params → true oracle.
- **Explicit batch config** (`edit_config.json`): `reactor_model: "singlezone"`,
  `o2_params`, `kla_params`, `noise_fracs`, `seed`, and a `batches` list. Each batch:
  `{name, V0, biomass, total_sub, malt_frac, dilution}`. Derived by LABS:
  `S = total_sub*(1-malt_frac)`, `M = total_sub*malt_frac`, `F = dilution*V0`.
- **Output schema**: `batch, t, X, S, P, M, O2, V` (note: data files use `DO`;
  reconcile O2/DO at the IO boundary).
- **The agent's model.** ⚠ LABS's *built-in* mechanistic fit uses **Pyomo +
  IPOPT** (`synthetic_data/fitting.py`) — that's the very IPOPT the README says
  to skip. So we do **not** call LABS's fitter. Instead the agent fits its own
  model with **scipy**, two options: (a) a **scipy re-implementation** of the
  same 7-param ODE (`models.py:160-194`) fit via `scipy.optimize.least_squares`
  + `scipy.integrate.odeint` to X,S,P,M — IPOPT-free, matches the README "use
  scipy.optimize"; or (b) a LABS **node/rnn surrogate** (`run_training`, pure
  torch, also IPOPT-free). Mechanistic is the README default.
- **The oracle never fits**: `generate-batches --mech-params mech_params.json`
  loads the true params and skips fitting, so the simulator path is IPOPT-free too.
- **Net install**: `pip install -e .` in the LABS repo; **skip IPOPT entirely**.
  The only thing that needed IPOPT was LABS's own fitter, which we bypass.
- **Fitting data shape**: our scipy fitter consumes the wide CSV (`batch,t,X,S,
  P,M,...`) directly; LABS's Pyomo fitter wanted long `species,time,value` — not
  our path. We fit to X,S,P,M (LABS ignores O2/V in the kinetic fit too).
- **Param bounds** (for the scipy fit + plausibility): mu_max[0.01,1],
  ks[0.01,50], P_max[50,200], Y_inv[0.1,10], alpha[0.001,5], beta[0.001,2],
  km[1e-5,1]. Feed: `F=D*V0`, `S_f=300*(1-malt_frac)`, `M_f=300*malt_frac`.

## 4. System shape — where this lives

Per the system-of-systems vision: shared platform (ingest → characterize →
bundle) feeds intent-specific child systems; a parent routes intent.

```
                  PARENT SUPER-AGENT (router; thin first cut)
                          │ intent: "maximize titer" → optimizer
   shared platform: ingest → characterize → BUNDLE  (intent-agnostic)
                          │ the bundle
        ┌─────────────────┴───────────────────┐
        ▼                                      ▼
  DIAGNOSTIC SYSTEM (fermdocs)        OPTIMIZER SYSTEM  ← NEW
  diagnose→hypothesize→recommend      fit→propose→simulate→evaluate→converge
```

New package `src/fermdocs_optimize/`:

```
fermdocs_optimize/
  schema.py        OptimizationInput, OptimizationOutput, Candidate,
                   RoundResult, ConvergenceReport   (Pydantic, coherence-validated)
  loop.py          deterministic optimization loop (fit→propose→simulate→eval→converge)
  agent.py         OptimizerAgent — LLM orchestrator over the loop (ReAct-ish),
                   honest refusal / low-confidence reporting
  models/          the agent's OWN predictor (fit on data only)
    base.py          PredictiveModel Protocol: fit(df), predict_peak_titer(knobs)
    mechanistic.py   7-param kinetic fit via scipy.optimize.least_squares
    surrogate.py     wraps LABS run_training (node/rnn)  [fallback]
  proposers/       the PROPOSE step over the 4-knob box
    base.py          Proposer Protocol: propose(model, bounds, k) -> [Candidate]
    grid.py          LHS / grid sweep
    optimize.py      scipy.optimize (differential_evolution / L-BFGS) maximizing P
  simulators/      the ground-truth ORACLE abstraction (the genuinely new seam)
    base.py          Simulator Protocol: simulate(candidates) -> trajectories df
    labs.py          LABSSimulator: ensure-install, write edit_config.json,
                     run generate-batches with --mech-params, read new_data.csv
  evaluate.py      R²/MAE (model vs simulated), best-titer tracking, ΔP convergence
  tools_bundle/    agent tools: get_data, fit_model, propose, run_simulator,
                   evaluate, submit_result  (gated, like diagnose/recommend)
  cli.py           fermdocs-optimize
tests/optimize/    rubric/loop/simulator-stub/convergence + integrity (no param peek)
```

The `Simulator` Protocol is the key new abstraction: optimization needs a
ground-truth evaluator. LABS is the first concrete one; other process families
plug in their own simulators (or "no simulator → DoE / refuse" like recommend).

## 5. Build phases

- **Phase 0 — Spike (de-risk LABS):** install LABS **without IPOPT**, run
  `generate-batches --config edit_config.json --mech-params mech_params.json
  --output new_data.csv`, confirm the oracle works + `new_data.csv` shape. Then
  fit our **scipy** 7-param ODE on `train_data.csv` and sanity-check R² on
  `test_data.csv`. Pure scripts, no package yet. **Proves both hard integrations
  (IPOPT-free oracle + scipy fit) before any architecture.**
- **Phase 1 — Deterministic loop MVP:** `simulators/labs.py`, `models/mechanistic.py`,
  `proposers/{grid,optimize}.py`, `evaluate.py`, `loop.py`, `schema.py`, `cli.py`.
  Reproduces the README end-to-end and prints the P_0→P_n improvement + best knobs.
  Fully testable with a simulator stub (no LABS needed in CI).
- **Phase 2 — Agentic shell:** `agent.py` + `tools_bundle/` wrap the loop with an
  LLM orchestrator (objective interpretation, model/proposer choice, narrating
  rounds, honest low-confidence reporting). Mirrors the recommend agent pattern.
- **Phase 3 — Platform + parent:** lift the shared model-fit/LLM/memory seams into
  the platform layer; add the thin parent router (intent → optimizer vs diagnostic).
  (Overlaps the deferred SOLID WS1; coordinate.)

## 6. Guardrails / honest-failure (system ethos)
- Proposals clamped to the var_params box `[lb, ub]` for all 4 knobs.
- Agent model fits on data only; **never reads `mech_params.json`** (enforced +
  unit-tested).
- Convergence is explicit (ΔP threshold OR max rounds OR token/compute budget);
  the improvement trajectory is reported honestly, including "did not improve."
- If the model can't fit (poor R² after augmentation), report low confidence
  rather than a confident-sounding optimum.

## 7. Decisions (locked 2026-06-04)
1. **Agent's predictor:** scipy 7-param mechanistic (re-impl of LABS ODE),
   LABS node/rnn surrogate as fallback.
2. **Proposer:** support both; default scipy global (differential_evolution),
   grid/LHS as robust fallback / warm-start.
3. **Build approach:** Phase-0 spike → deterministic MVP → agentic shell → platform/parent.
4. **Placement:** new `src/fermdocs_optimize/` sibling package; `Simulator`
   Protocol is the new seam (LABS first); shared model-fit lifts to platform later.
