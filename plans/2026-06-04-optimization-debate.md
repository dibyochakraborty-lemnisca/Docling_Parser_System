# Optimization Debate — Architecture & Implementation Plan

**Branch:** `optimizer-agent`
**Date:** 2026-06-04
**Status:** **Phases A + B + C DONE** (deterministic + wiring verified; live debate run pending tokens).
**Topic source (locked):** knob-anchored + trends, diagnosis-optional.
**Decision locked:** the debate **informs**, it does not constrain. The optimizer
always searches the full feasible box; the oracle judges every proposal. The
debate sets the objective, prioritizes levers, and interprets the result — a
wrong specialist can never hide the true optimum.

## 1. The shape

Make the two child systems symmetric:

```
DIAGNOSTIC:  diagnose → hypothesis debate ("what went wrong?")       → recommend (model-backed fix)
OPTIMIZER:   characterize → opportunity debate ("what can we push?")  → optimize  (simulator-backed knobs)
```

The optimization debate is the qualitative front half (specialists argue about
where the titer headroom is and why); the closed-loop optimizer we already built
is the quantitative back half (oracle-verified knob settings). A final synthesis
reconciles the verified optimum back through the debate's levers.

## 2. The key finding: the debate engine is narrative-agnostic

`run_stage(*, hyp_input: HypothesisInput, hooks: RunnerHooks, ...)` (runner.py:1028)
is a pure state machine. It depends on two abstractions we can supply our own
implementations of:

- **`hooks: RunnerHooks`** — `LiveHooks` (live_hooks.py:57) builds each agent via
  `build_kinetics_specialist(client, tools)` etc., and a `SpecialistAgent`
  (specialist_base.py:132) is fully driven by a `SPECIALIST_SPEC` dict
  (`system_identity` / `invariants` / `task_spec` / `tool_hints` / `recap`).
- **`hyp_input.seed_topics: list[SeedTopic]`** — the engine consumes topics; it
  does not care that today they come from `DiagnosisOutput`.

The diagnosis-specificity lives in exactly two places:
1. `seed_topic_extractor.py` — projects `DiagnosisOutput.{failures,analyses,trends,
   open_questions}` into topics.
2. The per-specialist SPEC prompts (task framing).

The orchestrator, synthesizer, critic, judge, ranker, projector, state, budget,
event_log, llm_clients, and tools_bundle are evidence-quality machinery with no
fault-vs-opportunity assumption.

**Consequence:** we build a sibling package that *imports the engine* and supplies
(a) optimization seed topics and (b) optimization specialist specs. We modify
**nothing** in `fermdocs_hypothesis` (open-closed; the stable diagnostic stage
stays stable).

## 3. New package: `src/fermdocs_optimize_debate/`

```
fermdocs_optimize_debate/
  topics.py          opportunity-topic extractor → list[SeedTopic]
                     (knob-anchored + characterization trends; diagnosis-optional)
  specs/             optimization SPECIALIST_SPEC dicts (reframed task/identity)
    kinetics.py        "where is the kinetic headroom?" (substrate saturation, μ ceiling)
    mass_transfer.py   "where does O2/feed transfer cap output?"
    metabolic.py       "what flux/maltose lever raises product?"
  hooks.py           OptimizeHooks(LoadedBundle) — subclasses/mirrors LiveHooks,
                     swaps the 3 specialist builders; reuses synth/critic/judge
  loader.py          bundle → HypothesisInput (characterization + optional prior
                     diagnosis/hypothesis); requires NO fault signals
  schema.py          OptimizationLever (thin view over FinalHypothesis) +
                     OptimizationDebateOutput (reuses HypothesisOutput shape)
  cli.py             fermdocs-optimize-debate run <bundle> → optimization_debate.json
```

Reused from `fermdocs_hypothesis` by import (unchanged): `run_stage`,
`SpecialistAgent`, `build_synthesizer/critic/judge`, `OrchestratorAgent`,
`RunnerHooks`, `HypothesisInput`, `HypothesisOutput`, `SeedTopic`, ranker,
projector, state, budget, event_log, tools_bundle, llm_clients.

## 4. The inform-only seam to the optimizer

The debate writes `optimization_debate.json` (final levers, each carrying
`summary`, `affected_variables`, `actionable_recommendation`, `confidence`,
`supporting_specialists`). The optimizer agent (already built) gains:

- a `get_levers()` tool — reads the debate output, surfaces the prioritized
  levers + objective so the LLM narrates against the debated mechanisms;
- **no change to the search**: the box stays the full feasible box from
  `config.json`; the oracle still judges everything;
- a **reconciliation** line in the final rationale: explain the oracle-verified
  optimum through the levers it confirms/contradicts (mirrors how `recommend`
  cites `grounding_hyp_ids`).

Levers map onto the four knobs (`biomass`, `total_sub`, `malt_frac`, `dilution`),
so the debate is a *reasoned prior over the box* and the loop is the *verified
posterior*.

## 5. Phasing

- **Phase A — topics + schema (deterministic, no LLM):** `topics.py`, `schema.py`,
  `loader.py`, with stub-driven tests (a fake bundle → expected opportunity
  topics). Proves the input seam without burning tokens.
- **Phase B — specs + hooks + cli:** optimization specialist specs, `OptimizeHooks`,
  wire `run_stage`, write `optimization_debate.json`. Run end-to-end on a real
  bundle with the stub/canned agents first, then live.
- **Phase C — seam:** `get_levers()` tool + reconciliation in the optimizer agent;
  one end-to-end (debate → optimize) on the lactic-acid bundle.
- **Phase D (Phase 3 overlap) — parent router:** intent → diagnostic vs optimizer
  debate; shared engine lifts to a platform package.

## 6. Progress log

- **Phase A (deterministic, no LLM) — DONE.** `levers.py` (knob→effect-variable
  spec, shared by generation + the seam), `topics.py` (knob-anchored OPEN_QUESTION
  topics + objective-driven TREND topics, reusing existing engine source types —
  zero schema change), `schema.py` (`OptimizationLever` + `levers_from_output` +
  `levers_from_debate_json`, dict/object tolerant), `loader.py` (bundle →
  HypothesisInput, diagnosis-OPTIONAL, reuses the hypothesis pool builders). 6 tests.
- **Phase B (engine reuse) — DONE.** `specs.py` (3 optimization specialist SPECs:
  same domain expertise, flipped mission — hunt headroom + name the lever, with the
  honesty rule "direction not magnitude, the oracle verifies"), `hooks.py`
  (`OptimizeHooks` subclasses `LiveHooks`, swaps ONLY the 3 specialists; orchestrator/
  synthesizer/critic/judge/projectors reused untouched), `run.py` + `cli.py`
  (`fermdocs-optimize-debate run`). 5 tests. **Confirmed: `fermdocs_hypothesis` is
  unmodified.**
- **Phase C (inform-only seam) — DONE.** Optimizer agent gains a `get_levers()`
  tool (advisory prior), a `--debate` CLI flag + `debate_output_path` param, a
  deterministic reconcile appendix (debated levers vs the oracle-verified optimum)
  and graft-in of the LLM's `lever_reconciliation`; `grounding_levers` recorded in
  meta. **The search still covers the full box — a test asserts proposals stay
  in-box regardless of levers.** Knob back-mapping excludes the objective species so
  levers map to discriminating knobs, not all of them. 2 tests (14 total on optimizer).
- **Pending:** one live end-to-end (debate → optimize) on a real lactic-acid bundle
  (needs a characterized bundle + Gemini tokens); Phase D parent router.

## 7. Decision (locked 2026-06-04)
- Topic source: **knob-anchored + trends, diagnosis-optional** (the debate works on
  a healthy run; levers line up 1:1 with the optimizer's box).
- Engine: **import + extend, never fork** — new package supplies topics + specs +
  hooks; the diagnostic stage is closed for modification.
