# Richer debate: grounding + topics

Goal: make the hypothesis / opportunity debate argue from **structure and real
cross-run signal**, over the **right topics** — instead of asserting vibes over
non-actionable byproduct levers. Scope = clusters 1 (grounding) and 2 (topics).
Out of scope here: debate *dynamics* (adversarial roles, cross-examination).

Anchored to a real failure (run `3cfc2aa6`): the opportunity debate spent its
whole run on `acetate.initial` and `ethanol.initial` (non-controllable byproduct
initials), all three specialists "deferred", and the accepted hypotheses leaned
on a DO=0 "oxygen bottleneck" story that's wrong for an anaerobic lactic process.

---

## Root causes (verified in code)

1. **Byproduct/output channels are treated as controllable levers.**
   `lever_discovery._derived_levers` makes a `<channel>.initial` lever for *every*
   varying observation channel, including outputs (acetate/ethanol/glycerol/
   product). You don't control initial acetate; it's an outcome.

2. **The topic ranker structurally prefers byproduct initials over real design
   factors.** `ranker._score_seed`:
   `severity_weight × priority + 0.3·citation_density + 0.2·q_overlap − 0.5·attempts − 1.0·rejections`.
   - Metadata design lever (e.g. `main_fermentation_nitrogen_source`): priority
     0.7, **no measured trajectory → citation_density 0 → score ≈ 0.7**.
   - Derived byproduct lever (`acetate.initial`): priority 0.7 + cites the
     acetate channel's many findings → **score ≈ 1.0**. It wins.
   Effect size on titer is **not** in the score at all.

3. **Specialists can only assert, never check.** Their tool set is read-only
   lookup (`query_bundle`, `get_priors`, `get_narrative_observations`); there is
   **no compute** and **no cross-run relationship query**. So they "defer to
   peers" because they have no way to test a claim.

4. **No within-experiment cross-run signal reaches the debate.** The
   `fermdocs_recommend/cross_run.py` engine already relates each per-run lever to
   peak titer across runs (linear fit / group means, effect size + caveat) — but
   that signal is invisible to the debate. (Note: the existing
   `cross_run_lessons` in the view is *memory from prior experiments*, a
   different thing.)

---

## Keystone: cross-run associations as evidence + ranking signal

One new artifact underpins both clusters. Compute the within-experiment
lever→titer associations once at debate-load time and expose them two ways:

- **As citeable evidence** (`CrossRunAssociationRef`) in each specialist's view,
  so a specialist can argue "across 15 runs, nitrogen source X associates with
  +N g/L titer (n=…)" and cite it (basis `cross_run`, which already exists in
  `ConfidenceBasis`).
- **As the topic-ranking signal**, so the levers that actually move titer lead.

Source: reuse `fermdocs_recommend.cross_run.analyze(dossier, obs_df, objective)`
→ `interventions` (knob, delta, direction, n, observed_range, caveat). It's pure
numpy, no LLM, already tested.

**Decisions locked in review (2026-06-13):**
- **Lift `cross_run` to a shared module** (e.g. `src/fermdocs/analysis/cross_run.py`)
  that both recommend and the debate import — avoids a `debate → recommend`
  package dependency (A3).
- **Naming (A1):** the new evidence channel is `within_run_associations`, NOT
  "cross_run_*". The view already has `cross_run_lessons` = memory from PRIOR
  experiments; reusing "cross-run" for within-experiment associations would
  confuse devs and the LLM. Keep the two names distinct.
- **Degenerate fallback (A3):** when `cross_run` yields no associations (<4 runs,
  no objective), lever topics fall back to neutral ordering and TREND topics
  carry the debate. The debate still runs; it just loses effect-size ordering.

---

## Cluster 2 — Better topics

### 2.1 Debate metadata design factors only  *(topics.py / loader.py)*  — **A2 decision**
- **No output/input classifier** (the t0≈0-and-rising heuristic is too fragile:
  fed-batch substrate also rises; a product with a nonzero baseline isn't ≈0 at
  t0). Rejected in review.
- Instead: **debate only the `run_conditions` metadata design factors** (nitrogen
  source, feed, inoculum, etc. — unambiguously controllable) plus TREND topics.
  **Drop derived observation-channel `.initial` levers from debate topics
  entirely.** You control the recipe, not "initial acetate" or "initial
  substrate" independently of the recipe.
- Derived `.initial` levers stay available to the **optimizer surrogate**
  (`data_equation`/`lever_discovery`) — only their promotion to *debate topics*
  is removed. Scope: a filter in `optimize_debate.topics`, not a change to
  `lever_discovery`'s output.
- Result: `acetate.initial` / `ethanol.initial` / `product.initial` never become
  debate topics; the nitrogen/feed/inoculum design factors do.

### 2.2 Rank lever topics by cross-run effect size  *(topics.py + ranker.py)*  — **D4 decision**
- Attach each lever's cross-run association (|delta| normalized by the outcome
  spread, and `n`) to its `SeedTopic` (new optional `effect_size` field).
- **Add an explicit, gated term** to `ranker._score_seed`: `+ w·effect_size`,
  where `effect_size` is 0 for any topic that doesn't carry one. The
  hypothesis-stage topics (failures/analyses) have none → term is 0 → **that
  stage's ranking is provably unchanged** (covered by a test). Keeps `priority`
  meaning one thing.
- Result: a design factor with a large measured titer swing (nitrogen source)
  now outranks the trend topics and any remaining levers; metadata levers no
  longer lose for lack of a measured trajectory.

### 2.3 Ground metadata-lever topics with the association  *(topics.py)*
- A metadata lever topic currently has no citation → starved. Cite its
  `CrossRunAssociationRef` so it carries evidence (and the specialist starts
  from "this lever moved titer", not a blank prompt).

### 2.4 Mid-debate topic proposal  *(orchestrator + live_hooks + ranker)*  — phase 2, optional
- The orchestrator can already emit `add_open_question`, and the ranker already
  synthesizes a topic from an open question (`_synthetic_topic_from_question`),
  BUT `live_hooks.pick_topic` ignores any action that isn't `select_topic`, so
  the path is dead. Wire it: when the orchestrator adds a question, seed it and
  let it compete next turn. Lets the debate follow surprises instead of exiting
  at 2 topics.
- Interaction / counterfactual seed topics ("does nitrogen × feed-timing
  interact?", "what's the titer ceiling?") generated when the cross-run engine
  sees two strong single levers. Smaller, do after 2.1–2.3 land.

---

## Cluster 1 — Better grounding

### 1.1 Within-run associations in the specialist view  *(schema + projector + loaders + live_hooks)*
- New `WithinRunAssociationRef` (frozen): `assoc_id`, `lever`, `summary`,
  `delta`, `direction`, `n`, `observed_range`, `variables_involved`,
  `objective`. Citeable like a finding. (Named to NOT collide with the existing
  `cross_run_lessons` = prior-experiment memory — A1.)
- Add `relevant_within_run: list[WithinRunAssociationRef]` to `SpecialistView`
  (cap ~6 in `VIEW_CAPS`), filtered by topic/domain in `project_specialist`.
- Populate a `cross_run_pool` in both bundle loaders
  (`fermdocs_hypothesis.bundle_loader` and `fermdocs_optimize_debate.loader`)
  from the keystone engine; thread through `live_hooks.contribute_facet` exactly
  like the existing pools.
- Specialists may cite `assoc_id` in `cited_finding_ids` (or a new
  `cited_association_ids`); `_build_facet` records it; `confidence_basis`
  `cross_run` already supported.

### 1.2 `query_relationship` tool  *(tools_bundle/factory.py + specialist_base schema + ToolHint)*
- Add one read-only tool: `query_relationship(lever, objective?)` → runs the
  cross-run association on demand and returns delta/direction/n/range. Lets a
  specialist *test* "does substrate loading track titer?" instead of asserting.
- Add via the established 4-step pattern: constant → method on
  `HypothesisToolBundle` → `dispatch` branch → `SPECIALIST_SCHEMA` tool enum +
  args + `ToolHint` in each specialist spec.

### 1.3 Constrained read-only `execute_python` for specialists  *(decision-gated)*
- The sandbox already exists (`fermdocs_diagnose/tools_bundle/execute_python.py`:
  120s/2GB/output-capped, project-root cwd) and the critic already uses it.
  Offer specialists a tighter variant (e.g. 60s, 10KB output, observations CSV
  path injected, no new fetches).
- **Decision D3 (locked yes):** build it in Phase B alongside `query_relationship`.
  Reverses "no execute_python in specialists (v0)". Constraints to keep the
  debate loop fast and safe: read-only (observations injected, **no fetch, no
  network**), output ≤10KB, ≤60s, and a per-specialist code-run budget so it
  can't blow the loop's wall-clock/token budget. Perf is the watch item, not
  safety (sandbox is the same one the critic already uses).

### 1.4 Domain compute tools  *(tools_bundle)*  — phase 2
- Deterministic per-domain helpers as tools: kinetics yield-ceiling / Monod fit;
  mass-transfer kLa/OUR envelope (the C9 math already exists in the toolkit).
  Thin wrappers over existing `fermdocs_characterize.toolkit` functions.

### 1.5 cobrapy / FBA for the metabolic specialist  — **CUT (not doing)**
(Left for context only; Phase D is out of scope per the user.)

- Highest-value grounding for the metabolic agent, but **blocked on a model**
  (see the cobrapy discussion: needs an organism-appropriate SBML — AGORA/BiGG,
  CarveMe draft, or a small hand-built lactic network). Implement as Option B
  (cobra-backed deterministic tools: `fba_growth`, `single_knockout`,
  `production_envelope`), NOT a code-writing skill.
- **Decision D2:** confirm the model source before building. Do this phase LAST;
  it does not block 1.1–1.4 or any of cluster 2.

---

## Sequencing (by leverage / dependency)

- **Phase A — SHIPPED (2026-06-13).** keystone lift + 2.1 (metadata-only) + 2.2
  (gated effect-size ranker term) + 1.1 (within_run_associations in the view) +
  the regression test (nitrogen leads, byproduct initials gone).
- **Phase A.5 — SHIPPED.** associations made citeable end-to-end
  (`cited_association_ids` on facet/hypothesis/final; counts as grounding so a
  design-factor hypothesis isn't muted to schema_only; validated at the facet
  boundary, carried by union through the synthesizer).
- **Phase B — SHIPPED (2026-06-15).** `query_relationship` tool (look up a design
  factor's cross-run effect, incl. levers beyond the top-N view) + constrained
  read-only `execute_python` for specialists (shared sandbox, 60s/10KB, `obs`
  preloaded). Wired into the optimize specialists' tool hints. Tests pass.
  Remaining: Phase C (depth).
- **Phase A (keystone + cluster 2 core):** lift `cross_run.analyze` to shared;
  2.1 (drop output levers), 2.2 (effect-size ranking), 2.3 (ground metadata
  levers), 1.1 (associations in the view). This alone fixes the `3cfc2aa6`
  failure: real design levers lead, every specialist has cross-run evidence.
- **Phase B (grounding interactivity):** 1.2 (`query_relationship`), 1.3
  (read-only execute_python, if D3=yes).
- **Phase C (depth):** 2.4 (mid-debate topics, interaction/counterfactual),
  1.4 (domain compute tools).
- **Phase D:** CUT (cobrapy not in scope).

Each phase is independently shippable and testable. **Phase A is a hard gate:**
verify it moves real-bundle quality before investing in B/C.

## Risks / guardrails to keep
- Effect-size ranking must keep the deterministic tie-break (topic_id) — no LLM
  jitter upstream of agent calls.
- The output-vs-input lever classifier is data-relative; add tests on praaj-like
  data (acetate excluded, nitrogen/substrate kept) and a degenerate case.
- Cross-run associations are observational — carry the existing "not causation"
  caveat into the evidence text so specialists don't overclaim.
- execute_python for specialists stays read-only (inject data, no fetch/network)
  and keeps the audit-never-read invariant.

## Decisions — RESOLVED in eng review (2026-06-13)
- **D1 (lever classification):** RESOLVED → **no classifier.** Debate metadata
  design factors only; drop derived `.initial` levers from debate topics (2.1).
- **D2 (cobra model):** N/A — **Phase D cut** (not doing cobrapy).
- **D3 (specialist compute):** RESOLVED → **yes, build both** `query_relationship`
  and a constrained read-only `execute_python` in Phase B (1.2 + 1.3).
- **D4 (ranking shape):** RESOLVED → **explicit gated ranker term** (2.2).
- **A1:** RESOLVED → name the new channel `within_run_associations`.
- **A3:** RESOLVED → lift `cross_run` to a shared module; neutral fallback +
  trends carry the debate when no associations exist.

## Test plan (per phase)
- **A (incl. the mandatory regression test):**
  - **REGRESSION (CRITICAL, IRON RULE):** reproduce run `3cfc2aa6`'s topic set on
    praaj-shaped fixtures (run_conditions with a varying nitrogen factor +
    acetate/ethanol channels + per-run titer) and assert: (1) `acetate.initial` /
    `ethanol.initial` are NOT debate topics, (2) the nitrogen design factor IS a
    topic, (3) it ranks above the trend topics. This proves the failure is fixed.
  - effect-size ranker term: a design factor with a large titer swing outranks a
    flat one; **and a hypothesis-stage ranking test proving order is unchanged
    when no effect_size is present** (gated-term guarantee).
  - degenerate fallback: <4 runs / no objective → debate still runs on trends, no
    crash, no effect-size ordering.
  - `within_run_associations` appear in the specialist view and are citeable.
- **B:** `query_relationship` dispatch + schema + memoization test; specialist
  `execute_python` **read-only enforcement** test (no network/fetch, output cap,
  budget cap) + a graceful-failure test (bad code → tool error, debate continues).
- **C:** open-question→topic wiring test; interaction/counterfactual topic
  generation test.
