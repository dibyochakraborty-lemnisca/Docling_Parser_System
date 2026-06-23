# De-LABS: make the system data-native by default

**Date:** 2026-06-16
**Branch:** `optimizer-agent-explore` (suggest new branch `delabs-data-native`)
**Goal:** Real uploaded data drives everything — objective channel, levers, state
variables, oracle. LABS becomes one *opt-in* synthetic/benchmark backend behind an
explicit flag. Never the default, never a silent fallback.

---

## Review decisions (2026-06-16 /plan-eng-review, full-scope PR)

1. **Objective is fully data-derived** — no hardcoded `product_g_l`. New `resolve_objective(bundle, user_question)`: (a) explicit user_question/API target wins, (b) else the channel the ingest schema canonically tags as the product objective, (c) else **refuse**. Never `process_priors.yaml` (that file is slated for removal).
2. **Data always wins for real bundles** — if `observations.csv` exists, the API uses the data oracle unconditionally; `FERMDOCS_OPTIMIZE_ORACLE=labs` is **ignored** on the API path. LABS runs against real data only via the standalone CLI on synthetic input.
3. **Quarantine = package boundary, not rewrite** — Gen-2 (data path) has **zero** Gen-1 imports (verified). Move Gen-1 into `fermdocs_optimize/benchmark/` + an import-guard test. Do NOT rewrite Gen-1 solvers to be knob-general.
4. **Routing layer has 0% test coverage today** — the two regression tests (real bundle → never LABS symbols; API never substitutes the LABS CSV) are non-negotiable additions.

Corrected framing from review: the **data optimizer is already general** (`product_g_l`, discovered levers, state-agnostic). The wild-numbers bug is purely the **routing default** (`labs`). Phase 0 fixes the misbehavior; Phases 1–2 fix debate/chart framing; Phase 3 is hygiene.

## The core problem

There are **two parallel optimizer stacks**, and the LABS one is the default:

| Stack | Code | State / objective | Status |
|---|---|---|---|
| **Data path** | `_run_data_optimization` → `data_equation.discover_and_optimize` → `discovery/general_mech.py` | state-agnostic, `product_g_l`, discovered levers, k-fold CV gate, sanity guards | **correct, keep** |
| **LABS path** | `_run_active_optimization` + `_run_closed_loop_optimizer` → `simulators/labs.LABSSimulator` | hardcoded `X,S,P,M,V`, `KNOB_NAMES`, `objective_species="P"`, `FERMDOCS_OPTIMIZE_TRAIN` CSV | **the contamination** |

The router `_optimize_oracle_mode()` ([runner_pipeline.py:948](../apps/api/fermdocs_api/runner_pipeline.py#L948)) **defaults to `"labs"`**, so every real run goes into the LABS stack unless `.env` overrides it. Worse: `_seed_training_from_bundle` returns `None` for praaj long-format data, so the LABS stack **silently loads the LABS CSV** (`FERMDOCS_OPTIMIZE_TRAIN`) — that's the f2942b56 "wild numbers" bug (biomass 1.69M, titer 126897 g/L). The system optimized a synthetic LABS experiment and labeled it as the user's result.

The debate layer is also LABS-framed: `objective_species="P"` is the default in ~10 call sites, and `DEFAULT_LEVERS` (the LABS knob set) is the fallback when lever discovery finds nothing. That's why charts cite `'P'` and `biomass_g_l` (LABS names) instead of `product_g_l` / `od600_au` (real channel names) and silently fail to render.

---

## Inventory — every LABS fingerprint

### A. Routing / oracle (the bleed)
- [runner_pipeline.py:948](../apps/api/fermdocs_api/runner_pipeline.py#L948) — `FERMDOCS_OPTIMIZE_ORACLE` defaults to `"labs"`.
- [runner_pipeline.py:1102-1131](../apps/api/fermdocs_api/runner_pipeline.py#L1102) — `_seed_training_from_bundle` requires wide-schema `{t,X,S,P,M,V}`; returns `None` for long-format → silent LABS CSV fallback.
- [runner_pipeline.py:1160-1165, 1196-1199, 1237, 1354, 1387-1388](../apps/api/fermdocs_api/runner_pipeline.py#L1160) — `FERMDOCS_OPTIMIZE_TRAIN` LABS-CSV fallbacks.
- [runner_pipeline.py:1134-1180](../apps/api/fermdocs_api/runner_pipeline.py#L1134) `_run_closed_loop_optimizer`, [1371+](../apps/api/fermdocs_api/runner_pipeline.py#L1371) `_run_active_optimization` — both instantiate `LABSSimulator`, hardcode `("biomass","total_sub","malt_frac","dilution")` box and `X,S,P,M,V`.

### B. Objective species = "P" (kills chart grounding + topic framing)
- [optimize_debate/loader.py:127](../src/fermdocs_optimize_debate/loader.py#L127) `objective_species="P"` default — the debate is never told the real objective channel.
- [optimize/agent_box.py:41](../src/fermdocs_optimize/agent_box.py#L41), [scipy_search.py:39](../src/fermdocs_optimize/scipy_search.py#L39), [oracle_search.py:55](../src/fermdocs_optimize/oracle_search.py#L55), [schema.py:99](../src/fermdocs_optimize/schema.py#L99), [agent.py:141](../src/fermdocs_optimize/agent.py#L141), [active_optimize.py:187](../src/fermdocs_optimize/active_optimize.py#L187), [tools_bundle/factory.py:65](../src/fermdocs_optimize/tools_bundle/factory.py#L65), [simulators/data_backed.py:184](../src/fermdocs_optimize/simulators/data_backed.py#L184), [runner_pipeline.py:1399](../apps/api/fermdocs_api/runner_pipeline.py#L1399) — `objective_species="P"` defaults.

### C. Hardcoded LABS knobs / state
- [optimize/schema.py:16](../src/fermdocs_optimize/schema.py#L16) — `KNOB_NAMES = ("biomass","total_sub","malt_frac","dilution")`.
- [optimize_debate/levers.py:33-42](../src/fermdocs_optimize_debate/levers.py#L33) — `DEFAULT_LEVERS` (LABS lever set) + [levers.py:45-61](../src/fermdocs_optimize_debate/levers.py#L45) `knobs_for_variables` (LABS map).
- [optimize_debate/schema.py:63](../src/fermdocs_optimize_debate/schema.py#L63) — `levers_from_output` falls back to `knobs_for_variables` (LABS) when a hypothesis cites no association.
- [optimize/active_optimize.py:46](../src/fermdocs_optimize/active_optimize.py#L46) `_WIDE=("X","S","P","M","V")`, [evaluate.py:56](../src/fermdocs_optimize/evaluate.py#L56), [discovery/cli.py:49](../src/fermdocs_optimize/discovery/cli.py#L49), [discovery/loop.py:189,269](../src/fermdocs_optimize/discovery/loop.py#L189), [models/mechanistic.py:81](../src/fermdocs_optimize/models/mechanistic.py#L81) — hardcoded `X,S,P,M,V` state.

### D. LABS simulator + synthetic backend
- [optimize/simulators/labs.py](../src/fermdocs_optimize/simulators/labs.py), `models/mechanistic.py`, `discovery/loop.py` — the LABS mechanistic stack.

---

## Plan (ordered by leverage; each phase shippable on its own)

### Phase 0 — Stop the bleed (highest leverage, ~1 file)
**Make `data` the default and kill the silent LABS substitution.** This alone fixes the wild-numbers bug for every real run.

1. **Data always wins (decision 2):** if `characterization/observations.csv` exists, the API run path uses the data oracle unconditionally. `FERMDOCS_OPTIMIZE_ORACLE=labs` is *ignored* on the API path — it only routes the standalone CLI on synthetic data. `_optimize_oracle_mode()` becomes CLI-only (or is removed from the API entirely).
2. Delete the silent `FERMDOCS_OPTIMIZE_TRAIN` fallbacks from the **API** functions (4 sites: 1160, 1196, 1237, 1387). The wide-schema slice moves to the LABS CLI. The API never substitutes a synthetic CSV.
3. `_run_data_optimization` already uses `product_g_l` and discovered levers — make it the only path a real bundle can reach; refuse cleanly (debate-only) when the data can't be modeled.

**Acceptance:** a praaj bundle with `FERMDOCS_OPTIMIZE_ORACLE` unset runs the data path; titer is on the ingested product scale; no `X/S/P/M/V` anywhere in the output. A LABS run still works only when explicitly opted in.

### Phase 1 — Derive the objective channel (kills "P")
1. New `resolve_objective(bundle, user_question)` (decision 1): explicit user_question/API target → else the channel the ingest schema canonically tags as the product objective → else **refuse**. No hardcoded `product_g_l`, no `process_priors.yaml`. `lever_discovery.DEFAULT_OBJECTIVE` stops being an unconditional default.
2. `load_optimization_bundle(bundle_dir)` calls the resolver and passes the objective through instead of `objective_species="P"`. Thread it into the synthesizer view + topic framing.
3. Leave `"P"` as the default only inside the quarantined Gen-1/LABS modules (they genuinely use `P`).

**Acceptance:** topics and chart specs reference `product_g_l`, not `P`; previously-dropped chart specs now ground (combined with the chart-grounding fix).

### Phase 2 — Derive levers + state; neutralize LABS fallbacks
1. `topics.py`: the `discovered_levers is not None` branch already handles the data path. Make the `else` (LABS `DEFAULT_LEVERS`) branch reachable **only** under the explicit LABS backend; for a real bundle with no discovered levers, run a **trends-only** debate (no fabricated LABS knobs).
2. `levers_from_output` ([optimize_debate/schema.py:63](../src/fermdocs_optimize_debate/schema.py#L63)): when a hypothesis cites no association, return `[]` (or the discovered knobs), **not** `knobs_for_variables` LABS knobs.
3. Quarantine `KNOB_NAMES`, `DEFAULT_LEVERS`, `_WIDE`, and the `X/S/P/M/V` mechanistic model into the LABS backend module; the data path goes exclusively through `discovery/general_mech.py` (state-agnostic).

**Acceptance:** with LABS not configured, no code path can emit `biomass/total_sub/malt_frac/dilution` or `X/S/P/M/V` for a real bundle.

### Phase 3 — Quarantine Gen-1/LABS into a benchmark subpackage (decision 3)
Gen-1 is a fixed-contract stack: `schema.KNOB_NAMES` (4 knobs, 13 consumers) + `models/mechanistic.MechanisticModel` (X/S/P/M/V, 12 consumers) + `active_optimize`/`oracle_search`/`scipy_search`/`agent`/`loop`/`proposers/*`/`simulators/{labs,data_backed,model_backed,stub}`. Gen-2 (`data_equation`, `general_mech`, `lever_discovery`) imports **none** of it.
1. Move Gen-1 into `fermdocs_optimize/benchmark/` (package boundary). Do NOT rewrite its solvers to be knob-general — no behavior gain for real data, large blast radius.
2. Add an **import-guard test**: assert the data path (`data_equation`, `general_mech`, `lever_discovery`, `fermdocs_optimize_debate/*`) imports no `benchmark/` symbol.
3. The benchmark backend is reachable only via the standalone CLI (`cli.py`, `discovery/cli.py`) on synthetic data — never the API run path.
4. `fermdocs_eval` has no direct LABS import (verified) — low risk; re-check after the move.

### Phase 4 — Tests + docs (routing layer is 0% covered today)
**Regression (iron rule, non-negotiable):**
1. Real praaj-shaped long-format bundle → optimization output has NO `X/S/P/M/V`, NO `biomass/total_sub/malt_frac/dilution`, titer on `product_g_l` scale.
2. `FERMDOCS_OPTIMIZE_TRAIN` set to a LABS CSV + real bundle present → API run path NEVER loads it (the f2942b56 wild-numbers bug).

**New/changed path coverage:**
3. `resolve_objective`: override / canonical-product / refuse (3 branches).
4. Oracle routing: obs present → data; `=labs` ignored on real bundle; no obs → debate-only.
5. `topics.py`: real bundle, no discovered levers → trends-only, not `DEFAULT_LEVERS`.
6. `levers_from_output`: no cited association → `[]`, not LABS knobs.
7. Import-guard test (Phase 3.2).
8. Update tests asserting LABS defaults (`test_active_optimize`, `test_discovery`, `test_topics`).

**Docs:** `docs/architecture.md` + `README.md` — data-derived by default; LABS is a CLI-only benchmark backend.

---

## Out of scope (flag, don't fix here)
- `process_priors.yaml` hardcoded domain values ([src/fermdocs/schema/process_priors.yaml](../src/fermdocs/schema/process_priors.yaml)) — separate "no hardcoded domain values" cleanup, tracked in memory. Not LABS, leave for its own pass.
- The chart-grounding fix (renderable variable names into the synthesizer view) — complementary; Phase 1 here makes it land cleanly.

## Risk
- Phase 0 changes the default behavior — any workflow relying on the implicit LABS oracle must now set `FERMDOCS_OPTIMIZE_ORACLE=labs` explicitly. Acceptable: that path was producing wrong results for real data.
- LABS evals must keep working — Phase 3 isolation must not break `fermdocs_eval`.

## GSTACK REVIEW REPORT

| Review | Trigger | Why | Runs | Status | Findings |
|--------|---------|-----|------|--------|----------|
| CEO Review | `/plan-ceo-review` | Scope & strategy | 0 | — | — |
| Codex Review | `/codex review` | Independent 2nd opinion | 0 | — | — |
| Eng Review | `/plan-eng-review` | Architecture & tests (required) | 1 | ISSUES_OPEN | 3 issues, 1 critical gap (routing 0% tested) |
| Design Review | `/plan-design-review` | UI/UX gaps | 0 | — | — |
| DX Review | `/plan-devex-review` | Developer experience gaps | 0 | — | — |

- **UNRESOLVED:** 0 decisions (all 4 answered: objective=data-derived, data-always-wins, package-boundary quarantine, full scope).
- **CRITICAL GAP:** the optimize routing layer has 0% test coverage today; closed by Phase 4 regression tests (non-negotiable).
- **VERDICT:** ENG reviewed — plan ready to implement once Phase 4 regression tests are committed alongside the code. Run /ship when implementation is done.
