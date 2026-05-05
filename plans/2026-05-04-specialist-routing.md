# Specialist routing plan (deferred)

**Status:** Deferred — do not build until trigger condition met.
**Trigger:** About to add specialist #4 (or planning many specialists in next 2 weeks).
**Estimated cost:** ~600-1000 LOC, isolated PR.

---

## Problem

Today's hypothesis stage has 3 specialists (`kinetics`, `mass_transfer`,
`metabolic`) hard-coded into `runner.SPECIALIST_ORDER`. Every specialist
runs sequentially on every topic regardless of relevance. Specialists
self-filter their views via `specialist_domain_tags` but always emit a
facet (or a degenerate one). Synthesizer integrates ALL facets per the
"preserve each facet's distinguishing claim" rule.

At 3 specialists this tolerates one being slightly off-topic — the other
two carry signal. **At 20 specialists this collapses:**

1. **Token blowup.** 20 specialists × ~4K tokens/view × per turn = ~80K
   tokens/topic on facets alone. Burns through the 200K
   `max_total_input_tokens` budget in 2-3 topics.
2. **Synthesis dilution.** If 17 of 20 emit weak facets, the synthesizer
   either preserves them all (kitchen-sink hypothesis the critic rightly
   rejects) or silently drops them (loses audit trail).
3. **Latency.** 20 sequential calls at ~5-10s each = 100-200s/turn just
   on facets. Frontend feels broken.

The user's framing: "if I bring 20 specialists but only 3 are useful for
a given topic, how do I avoid running the other 17?"

---

## Solution — three layers, build in order

### Layer A — Topic routing (ship first)

Add a routing step BEFORE facet contribution. Top-K (default 3-5) of M
specialists run.

**A1 — Deterministic.** Each specialist declares domain tags (already
exists: `specialist_domain_tags(role)`). The router scores each
specialist by tag overlap with `topic.affected_variables` +
`topic.summary`. Top-K by score participate.

**A2 — LLM-driven.** A new lightweight router agent (or extension of
the orchestrator) sees the topic + every specialist's one-line domain
description and picks K relevant ones. Costs one extra LLM call per
topic but handles novel topics tag-matching can't catch.

**Recommendation:** ship A1 first (fast, deterministic, easy to test).
Upgrade to A2 if tag-matching proves brittle on real bundles.

### Layer B — Confidence-weighted synthesis

Synthesizer already sees `facet.confidence`. Today it doesn't act on it.

- **Prompt rule:** facets below threshold T (start with 0.4) get a
  one-sentence "specialist X had no strong signal" mention rather than
  integrated into the claim. Drop from `cited_findings` entirely if
  confidence < 0.4.
- **Structural cap:** synthesizer only integrates top-K by confidence,
  regardless of how many came in. So even if Layer A misroutes and 10
  specialists run, only the 3 strongest shape the final hypothesis.

### Layer C — Specialist self-pass

Specialist tool schema gains a `pass` / `no_signal` action distinct
from facet emission.

- When a specialist passes, emit `SpecialistPassedEvent` (cheap, no
  content tokens).
- Runner skips passed specialists in the facet list.
- Synthesizer never sees noisy facets from off-topic specialists.

---

## "Only 3 of 20 useful" — full flow with all three layers

```
Topic selected
      ↓
Router (Layer A) ──▶ picks 5 most likely relevant specialists
      ↓                     (deterministic tag overlap)
For each of 5:
  ├─ specialist computes view, calls LLM
  ├─ if no signal → emit `pass` action (Layer C) ─▶ SpecialistPassedEvent
  └─ if signal     → emit facet with confidence
      ↓
3 facets contributed (2 passed)
      ↓
Synthesizer (Layer B):
  - integrates top-3 by confidence
  - drops low-conf facets to one-line mentions
  - writes hypothesis citing only the strong evidence
```

Total LLM calls: ~6 (1 router + 5 specialists, 2 short-circuit).
Audit trail records: who was considered, who passed, who contributed.

---

## Files this would touch

- `src/fermdocs_hypothesis/runner.py` — `SPECIALIST_ORDER` (static
  tuple) → per-topic routed list. `contribute_facet` phase short-circuits
  on pass.
- `src/fermdocs_hypothesis/agents/router.py` (new, Layer A2 only) OR
  extend `orchestrator.py`. Emits `SpecialistRoutedEvent`.
- `src/fermdocs_hypothesis/events.py` — new `SpecialistRoutedEvent` and
  `SpecialistPassedEvent` in the discriminated union.
- `src/fermdocs_hypothesis/schema.py` — `SpecialistRole` Literal (3
  values today) → open string with registry. Each specialist declares:
  - `role_id` (string identifier)
  - `domain_tags` (set of strings for routing)
  - `description` (one-line, used by Layer A2 router prompt)
- `src/fermdocs_hypothesis/projector.py` — Layer B confidence threshold
  enters here for filtering low-confidence facets out of synthesizer view.
- `src/fermdocs_hypothesis/agents/synthesizer.py` — Layer B explicit
  prompt rules about confidence-weighted integration + structural cap.
- `src/fermdocs_hypothesis/state.py` — new projection
  `specialist_routes_for_topic(events, topic_id)` for audit.

---

## What NOT to do

- **Don't build this now.** Current failure modes (carotenoid loop,
  IndPenSim spec-mismatch noise) are prompt + projection problems, not
  specialist-count problems. The system just started working end-to-end
  on `caisc-langgraph`.
- **Don't over-engineer the router.** A1 is enough until proven
  insufficient. The user already has tag taxonomy that works for 3
  specialists; trust it for ~10 specialists too.
- **Don't break the existing 3-specialist contract during scaffolding.**
  Layer A as a no-op (always picks all specialists) is the right
  intermediate state for the "scaffold before second specialist" path.

---

## Trigger condition for revisit

Build when **about to add specialist #4**. That's the natural moment:

- You can't responsibly add a 4th without routing because synthesis
  dilution starts immediately at N=4 (one of four facets being weak
  has 25% influence on the hypothesis vs 33% at N=3 — sounds smaller
  but compounds).
- Schema changes (`SpecialistRole` from Literal to registry) are
  cheaper to do once with the 4th specialist than to refactor twice.

If multiple specialists are coming in a single PR, ship Layer A first
as a no-op scaffold (always picks all M), then flip the gating on once
the second specialist arrives. That gives you the routing infrastructure
without committing to specific routing logic on Day 1.

---

## Tests to write when this lands

- Layer A1 router: tag-overlap scoring picks expected specialists for
  fixture topics (DO crash → mass_transfer + kinetics, biomass plateau
  → kinetics + metabolic).
- Layer A1 router back-compat: with M=3 specialists registered and
  K=3 default, output matches today's `SPECIALIST_ORDER` deterministically.
- Layer B confidence threshold: facets at 0.39 dropped, 0.41 kept.
- Layer B top-K cap: with 8 facets contributed, synthesizer integrates
  only top-3 by confidence.
- Layer C pass: specialist's `pass` action emits
  `SpecialistPassedEvent`, runner skips facet count, synthesizer view's
  facet list excludes the passed specialist.
- E2E: full debate with 10 registered specialists, K=4 routing, 1
  passing — assert exactly 3 facets reach the synthesizer.
