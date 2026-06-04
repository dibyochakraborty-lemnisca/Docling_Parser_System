# FASSO Architecture

This document describes the current architecture of FASSO (Fermentation Agentic Scientific Synthesis and Observation). It is the
repo-root design reference. Historical notes in `plans/` are useful for
intent, but the current contracts are the code, schemas, tests, and this
document.

## System Shape

FASSO is a staged analysis system for fermentation reports:

```text
source files
  -> ingest                      (raw -> dossier)
  -> bundle                      (dossier + characterization + diagnosis)
  -> characterize                (deterministic-first metrics)
  -> diagnose                    (observational ReAct loop)
  -> hypothesize                 (multi-agent causal debate)
       <-> MemoryBackend         (cross-run priors, optional)
  -> recommend                   (model bake-off + intervention sim, or refuse)
  -> local API/web app
```

The stages are intentionally separated by typed JSON artifacts. This keeps
each agent from needing direct access to every upstream implementation detail
and makes the output inspectable after every stage.

## Core Packages

```text
src/fermdocs
  Parsing, header mapping, unit normalization, storage, dossier creation,
  process identity (closed-vocab process_family), PDF segmentation,
  narrative extraction, and bundle I/O.

src/fermdocs_characterize
  Deterministic trajectory construction, metric catalog execution, anomaly
  detection (instrument change, h0 outlier, header inconsistency, scale
  change, bioreactor change), narrative observation materialization,
  optional LLM trajectory analysis, and validation (physicality bounds +
  closed-vocab process_family routing).

src/fermdocs_diagnose
  Observational diagnosis agent. It uses a bounded ReAct loop over bundle
  tools and emits failures, trends, analyses, and open questions.

src/fermdocs_hypothesis
  Multi-agent causal hypothesis stage. State machine, specialist agents,
  typed projector views, synthesis/critique/judgment, HITL resume,
  follow-up, chart specs, Plotly rendering, lessons summarizer, and
  cross-run memory wire-up.

src/fermdocs_recommend
  Model-based recommendation stage. A ReAct agent fits three model
  families (mechanistic / surrogate / hybrid) plus a pre-trained
  IndPenSim path; a pure deterministic rubric selects the winner or
  refuses; the winner is used as an oracle to line-search the best
  next-batch intervention. Tools, sandbox, schema, and rubric are split.

src/fermdocs_memory
  MemoryBackend Protocol with Noop / Stub / Synap adapters. Tier 1
  (lessons memory) is live; Tiers 2-5 are scaffolded but not wired.

apps/api
  Local FastAPI wrapper around the full pipeline. Reads FERMDOCS_MEMORY
  to construct the memory backend per-run, and runs the recommend stage
  after a run reaches DONE (errors there never flip the run to FAILED).

apps/web
  Next.js UI for upload, run progress, websocket events, hypotheses,
  charts, recommendations, follow-up, and print-to-PDF. Uses the
  Lemnisca dark "instrument panel" design system.
```

## Artifact Flow

### 1. Raw Inputs

Supported user-facing inputs:

```text
.csv
.xlsx
.pdf
.zip bundle
```

CSV/XLSX/PDF inputs run the full pipeline. Zip inputs are treated as existing
bundles and skip directly to hypothesis.

**Operator-supplied process family.** The upload UI exposes a closed-vocab
dropdown sourced from `src/fermdocs/schema/process_families.yaml`. When the
operator picks anything other than "Auto-detect", the API writes a manifest
YAML next to the upload and passes `--process-manifest` to ingest. This
bypasses the LLM identity extractor entirely — required for CSV-only
bundles where the LLM has no narrative to read.

### 2. Dossier

The ingest stage emits a dossier JSON. It is the first downstream artifact and
contains observations, provenance, residual material, process identity, and
document mapping information.

Important ingest principles:

- The LLM maps headers and gives unit hints; it should not invent numeric
  values.
- Source material that does not map cleanly is preserved as residual data.
- Provenance is part of the data model, not a debug feature.
- Process identity separates observed identity from registered identity.
- **`RegisteredProcess.process_family` is the canonical key** downstream
  routing reads (memory layer, catalog runner). Sourced either from the
  closed-enum LLM call or from an operator manifest.
- Operator manifests override LLM identity extraction. Manifest writes
  `provenance=MANIFEST` on both observed and registered facts so the
  source is auditable.

### 3. Bundle

The bundle is the central artifact boundary.

Typical structure:

```text
bundle_<id>/
|-- meta.json
|-- dossier.json                 includes registered.process_family
|-- characterization/
|   |-- characterization.json
|   |-- observations.csv
|   `-- narrative_observations.json
|-- diagnosis/
|   `-- diagnosis.json
|-- audit/
`-- user_question.json
```

`meta.json` is written last and is the readiness signal. Readers reject
missing or incompatible metadata. This prevents later stages from consuming
half-written bundles.

Runtime code should not use `audit/` as evidence. Audit files are for traces,
debugging, and post-run inspection.

### 4. Characterization Output

Characterization converts dossier observations into:

```text
trajectories
findings
narrative observations
timeline events
expected-vs-observed summaries
facts graph
open questions
kinetic estimates
metadata anomalies     instrument-change, h0-outlier, header-inconsistency,
                       scale-change, bioreactor-change (all deterministic)
```

The stage is deliberately deterministic-first. Metric catalog execution,
toolkit functions, robust statistics, metadata anomaly detectors, product
KPIs, and physicality validators run before optional LLM analysis.

Product-family-specific KPI routing reads `RegisteredProcess.process_family`
to select the right adapters from `process_families.yaml` (e.g. P1-P5
penicillin KPIs vs P_INTRACELLULAR_YIELD for yeast carotenoid).

### 5. Diagnosis Output

Diagnosis is observational. It should answer:

```text
What failed?
What trended?
What was observed?
What is uncertain?
```

It should not speculate causal mechanisms beyond what the evidence supports.
The causal debate belongs to hypothesis.

Diagnosis claims cite upstream evidence by IDs: finding IDs, trajectory IDs,
or narrative observation IDs. Bundle-backed diagnosis uses tools such as
`get_findings`, `get_trajectory`, `get_narrative_observations`, and
`execute_python`.

The ReAct loop has hard tool-use enforcement. A model that tries to emit a
diagnosis without first using tools gets retried once; if it still does not
use tools, the result goes down an error path.

### 6. Hypothesis Output

Hypothesis is causal and argumentative. It proposes mechanisms, compares
alternatives, exposes uncertainty, and produces actionable recommendations.

The stage emits:

```text
final hypotheses                with actionable_recommendation + chart_specs
rejected hypotheses
open questions
debate summary
token report
global.md event log path        canonical human-readable log
Plotly chart JSON
LessonsSummarizedEvent          structured Lesson[] for memory persistence
```

`global.md` is the canonical human-readable event log. The JSON output is the
machine contract.

### 7. Memory Layer (Phase 1)

`src/fermdocs_memory` ships three Protocol implementations:

```text
NoopBackend     write no-op; fetch returns []. Default; preserves byte-
                identical behavior to pre-memory-layer FASSO.

StubBackend     in-memory dict; substring ranking. For unit tests.

SynapBackend    wraps maximem_synap.MaximemSynapSDK. Async-to-sync via
                a dedicated event-loop thread (the SDK is async; the
                Protocol is sync). Maps MemoryRecord -> sdk.memories.create:
                  document     <- record.summary
                  document_id  <- record.memory_id (idempotency key)
                  user_id      <- record.process_family
                  customer_id  <- record.tenant_id
                  metadata     <- organism, tags, provenance, etc.
                Retrieval via sdk.user.context.fetch returns 5 typed
                buckets (facts, preferences, episodes, emotions,
                temporal_events) which we flatten into MemoryRecords.
                Client-side filters on organism / variables_overlap /
                finding_classes_overlap (Synap doesn't filter on metadata).
                Failure absorption: any SDK exception during fetch
                returns []; during write logs + skips. Memory failures
                never break runs.
```

**Frozen Protocol contract (`src/fermdocs_memory/base.py`):**

```python
class MemoryBackend(Protocol):
    def write(self, record: MemoryRecord) -> None: ...
    def fetch(self, query: MemoryQuery) -> list[MemoryRecord]: ...
    def supersede(self, memory_id: str, by: str) -> None: ...
```

**Hard invariants:**

- `MemoryQuery(kind="lesson", process_family=None)` raises `ValueError` at
  `validate_query()` time. Cross-strain lesson retrieval is structurally
  impossible without an explicit opt-in.
- `MemoryRecord` is frozen; `provenance` is stored as `MappingProxyType` so
  the audit trail can't be mutated post-construction.
- Tenant isolation lives in `MemoryRecord.tenant_id`. SynapBackend routes
  to per-tenant Synap Customer scope.

**Write timing (D6 + D10):**

```text
during run    lessons buffered in RunnerState
HITL pause    buffer serialized to <bundle>/lesson_buffer.json
HITL resume   buffer reloaded
clean exit    {consensus_reached, no_topics_left} -> memory.write per Lesson
failed exit   {budget_exhausted, max_turns_reached, exception} -> SKIP
```

**Retrieval at view-build (in `projector.py` via `LiveHooks`):**

```text
_fetch_cross_run(topic_summary) -> LessonsDigest | None
  cached per topic_summary within a turn so 4 view-build sites share
  one Synap round-trip. process_family from hyp_input drives the
  user_id scope. semantic_query = topic.summary.
```

**Prompt surfaces (synthesizer + critic):**

```text
SYNTHESIZER_INVARIANTS includes "CROSS-RUN LESSONS (memory-layer Phase 1)":
  When view.cross_run_lessons is populated, treat each lesson as a
  prior — not ground truth. Bundle evidence overrides. Surface
  contradictions explicitly.

CRITIC_INVARIANTS includes "[MEMORY-AXIS]":
  Reject when a hypothesis cites a prior lesson but the cited evidence
  is from a different run/strain.
```

**Deferred tiers (with explicit triggers in `plans/2026-05-10-memory-layer.md`):**

```text
Tier 2  ratified-hypothesis store      after Phase 1 has 2 weeks of data
Tier 3  rejected-hypothesis store      bundle with Tier 2
Tier 4  strain-conditional KPI priors  when n >= 15 runs per family
Tier 5  human-correction memory        first time HITL feedback fails to
                                       apply on a follow-up run
```

### 8. Recommendation Output

The recommend stage (`src/fermdocs_recommend`) runs after the hypothesis
run reaches DONE. It turns the final hypotheses into a concrete,
fit-tested intervention for the next batch — or refuses. It writes
`<bundle>/recommend/recommendation.json` and is wired so any failure logs
and leaves the run DONE (never FAILED).

**Shape:** a bounded ReAct agent (`RecommendationAgent.recommend()`, max
~20 steps) fits models in a sandbox via a gated tool registry
(`get_hypotheses`, `get_data_feed`, `get_skill`, `execute_python`,
`submit_recommendation`); a pure rubric judges; the schema enforces
coherence.

**Model families (the bake-off):**

```text
mechanistic   ODE with fitted parameters; parameters are plausibility-checked
surrogate     neural ODE / LSTM; no parameters to vet (any architecture allowed)
hybrid        ODE + learned residual (MLP) correction
```

For `penicillin_fedbatch`, a **pre-trained IndPenSim path**
(`registry.py`) loads a control-augmented LSTM trained offline on the
800-batch dataset (cached under `src/fermdocs_recommend/models/`,
held-out penicillin R² ≈ 0.91) and supplies it as the surrogate
candidate, optionally fine-tuning on all-but-last run and scoring the
held-out run.

**Deterministic rubric (`rubric.py`, no LLM, no I/O). The rubric, not the
LLM, makes the final call.** Per-candidate scoring uses the worst eligible
species (min R², max RMSE) over species with ≥ 4 held-out points. Gates:

```text
GOOD_FIT_R2          = 0.75   R² must exceed this on every eligible species
                             (lowered from 0.95 for noisy bioprocess data)
RMSE_FLOOR_MULTIPLIER = 2.0   RMSE <= 2 x measurement floor (when supplied)
MIN_HELDOUT_POINTS    = 4     species below this are ineligible (not scored)
STALL_REDUCTION_FRAC  = 0.05  optimizer loss must drop > 5%, else data does
                             not constrain the model -> refuse
DISPLACE_MARGIN_FRAC  = 0.10  a challenger must beat a passing mechanistic
                             model on RMSE by > 10% to displace it
```

Selection: a mechanistic model that passes (good fit + plausible params +
moved) wins unless displaced; else a passing surrogate wins; else a
passing hybrid; else honest refusal. Mechanistic/hybrid parameters are
checked against known ranges (out-of-range names reported in
`offending_params`); surrogates skip plausibility.

**Refusal codes** (recommended_model = "none", interventions = []):

```text
brewtwin_not_installed     fit toolkit import failed
insufficient_data          too few points / optimizer stalled
poor_fit_all_models        every attempted model R² <= 0.75
implausible_parameters     only good fits were mechanistic with bad params
mechanism_not_supported    mechanistic fit the curve but offered no mechanism
compute_budget_exhausted   agent ran out of steps
stage_error                unexpected error inside the stage
```

**Intervention simulation (`registry._simulate_interventions`).** The
winning model becomes an oracle. Hypotheses' `affected_variables` map to
control knobs (e.g. `dissolved_o2 -> [Fg, RPM]`); each knob is
line-searched over `0.8–1.25x` baseline (19 points) to maximize predicted
peak titer, staying within the observed envelope ± 3σ and capping the
predicted titer at `mean + 3σ` of the training data to block oracle
extrapolation. Changes with delta > 0.1 g/L are reported.

**Output contract (`schema.py`, `RecommendationOutput`):**

```text
recommended_model    mechanistic | surrogate | hybrid | none
confident            True iff recommended_model != none
refusal_reason       set iff not confident (codes above)
selection_rationale  why the rubric picked / refused
candidates[]         one per family: attempted, disqualified, selection_r2,
                     selection_rmse, good_fit, plausible, offending_params,
                     stalled, fitted_parameters, report
interventions[]      knob, objective_metric, baseline -> predicted, delta,
                     in_coverage, caveat, rationale  (empty on refusal)
grounding_hyp_ids[]  hypotheses that motivated the interventions
```

A model validator rejects incoherent payloads (refusal with interventions,
or confidence with a refusal_reason), so the contract cannot be violated.

## Runtime State Machine

The hypothesis runner is an explicit state machine, not LangGraph today.

```text
init
  -> select_topic
  -> retry_topic
  -> contribute_facet
  -> synthesize
  -> critique
  -> judge
  -> finalize_turn
  -> exit
  -> done
```

The live agent order is:

```text
orchestrator
  -> kinetics specialist
  -> mass_transfer specialist
  -> metabolic specialist
  -> synthesizer
  -> critic
  -> judge
  -> lessons summarizer, when retry patterns need compression
```

The current specialist set is fixed at three. Routing is intentionally static.
The planned routing rule is: add deterministic specialist routing only when a
fourth specialist exists.

## User Question and HITL Model

User questions are a lens over the evidence. They should bias topic selection,
ranking, views, and summaries, but they should not suppress bottom-up analysis.

The question classifier resolves:

```text
question shape
affected runs
affected variables
```

This is written to `user_question.json` in the bundle when applicable.

There are two interactive modes:

```text
answers/resume (HITL)
  Triggered via CLI (`--hitl`) or API (`POST /api/runs/{run_id}/answers`).
  The run pauses when agents emit open questions. The CLI prompts the user,
  captures their answers as `HumanInputReceivedEvent`s, and the same
  debate resumes. The lesson buffer survives the pause via 
  <bundle>/lesson_buffer.json. This ensures human feedback is organically 
  integrated into the final synthesis and cross-run memory.

follow-up
  The bundle is frozen. A new user question overwrites user_question.json and
  only the hypothesis stage runs again. Memory layer reads/writes are
  per-followup-run, not per-original-run.
```


*Note on `HumanInputRecord`: While the system supports interactive open-question answering (via event injection), the `HumanInputRecord` schema object attached to individual debate attempts is currently a stub reserved for future explicit trajectory steering (e.g., a human forcing a "red flag" override directly into a specific specialist's output context).*


Diagnosis open questions intentionally do not include `re_run_from`.
Hypothesis open questions may include `re_run_from` because only hypothesis
knows whether the right next action is diagnosis or hypothesis.

## Chart Model

Hypothesis agents emit declarative `chart_specs`. Local deterministic code
turns those specs into Plotly JSON.

Supported chart families include:

```text
time_series_overlay
scatter_correlation       includes weak-n bootstrap CI badge for n < 8
faceted_time_series
```

This split matters: the LLM can request the chart it needs, but rendering,
data lookup, regression details, and JSON shape stay deterministic and
testable.

## Frontend (Lemnisca instrument panel)

The web UI uses the **Lemnisca design system** — a dark "instrument
panel" aesthetic. (It replaced an earlier light "Editorial Scientific"
direction; some plan docs still describe the old one.)

Type stack (`next/font/google` + system grotesk):

```text
Helvetica Neue   body + headings (system grotesk, set in globals.css)
JetBrains Mono   --font-ui — every label, eyebrow, tag, date, stat
Newsreader       --font-display — sparing serif-italic accents
```

Color palette (CSS variables in `apps/web/src/app/globals.css`):

```text
--color-bg          #000000   true-black canvas
--color-surface-1   #0b0c0d   card surfaces (-2 #111315, -3 #181b1d)
--color-ink         #f4f5f6   primary text (ramp -> #c7cbce / #92989c / #62686c)
--color-rule        rgba(255,255,255,0.10)   hairline dividers
--color-accent      #38AFD8   signature teal — used sparingly
--color-ok / -warn / -error   #3fbfa6 / #e3a552 / #e5484d
```

Token wiring: shadcn-compat aliases are HSL triples
(`hsl(var(--x) / <alpha-value>)`); accent + semantic colors also expose
RGB-channel triples (`--accent-rgb`, etc.) consumed as
`rgb(var(--x-rgb) / <alpha-value>)` so Tailwind alpha modifiers (`/10`,
`/40`) work on tinted fills and borders. Plain-hex vars do NOT support
alpha modifiers — a known footgun.

Signature surfaces:

```text
home hero        animated lemniscate (∞) motif over a teal glow;
                 two-column split (hero left, upload well right)
chrome           sticky blurred header with the ∞ wordmark, a
                 scroll-progress bar, teal glow instead of drop shadows
run debate       Timeline renders as a conversation: each specialist
                 agent (kinetics / mass_transfer / metabolic + synthesizer
                 / critic / judge) gets a colored avatar + live token
                 count and contributes dialog bubbles; synthesis is a
                 neutral card; accept/reject are status pills
final hypotheses two-up — reasoning text left, Plotly charts right
recommendation   the RecommendationOutput card (model, confidence/refusal,
                 candidate bake-off, interventions)
```

## API and Web Architecture

The API is a local FastAPI service:

```text
POST /api/uploads         form fields: files[], process_family (optional)
POST /api/runs
GET  /api/runs
GET  /api/runs/{id}
WS   /api/runs/{id}/events
POST /api/runs/{id}/answers
POST /api/runs/{id}/followup
```

Raw file runs execute:

```text
fermdocs ingest                      with --process-manifest if operator
                                     supplied a process_family
  -> fermdocs-characterize --bundle
  -> fermdocs-diagnose --bundle
  -> in-process hypothesis run       with memory backend per FERMDOCS_MEMORY
  -> recommend stage                 after DONE; writes recommend/recommendation.json
```

Bundle zip runs execute:

```text
unzip bundle
  -> load_bundle
  -> in-process hypothesis run
  -> recommend stage                 after DONE
```

The recommend stage runs in a thread pool (the fit toolkit is blocking),
status flips through `RunStatus.RECOMMENDING`, and the run returns to DONE
whether the recommender succeeds, refuses, or errors. The result is
surfaced on `GET /api/runs/{id}` as `recommendation_output`.

The API publishes status messages and hypothesis events to websocket
subscribers. It stores uploads and run outputs under `FERMDOCS_API_ROOT`,
defaulting to `out/api`.

The web app is a development UI. It is not a production SaaS shell. It has no
authentication, no tenant isolation, and no durable background worker.

## Data and Schema Boundaries

The most important contracts are:

```text
fermdocs.domain.models
  Golden schema, parsed tables, observations, residual payloads, dossier,
  document maps, process identity (RegisteredProcess.process_family =
  closed enum from process_families.yaml).

fermdocs_characterize.schema
  CharacterizationOutput, Finding, Trajectory, NarrativeObservation, facts,
  expected-vs-observed, timeline, open questions, metadata anomalies.

fermdocs_diagnose.schema
  DiagnosisOutput, failures, trends, analyses, diagnosis open questions.

fermdocs_hypothesis.schema
  HypothesisInput, topics, facets, hypotheses, critiques, judgments,
  HITL records, open questions, chart specs, final output. Lesson model
  with stable lesson_id (memory-layer Phase 1). LessonsDigest carries
  structured lessons list alongside the legacy digest string.

fermdocs_recommend.schema
  RecommendationOutput, CandidateReport, Intervention, RecommendationMeta.
  A model validator enforces refusal <-> no-interventions coherence.

fermdocs_memory.base
  MemoryRecord, MemoryQuery, MemoryBackend Protocol, MemoryKind enum.
  validate_query() enforces the D7 cross-strain guard.
```

Prefer extending these schemas deliberately over passing loose dictionaries
between stages.

## Validation and Invariants

Important invariants:

- Bundle metadata gates bundle readability.
- IDs are namespaced and cross-references are validated.
- Findings must cite real observations when possible.
- LLM-judged evidence has confidence caps.
- Characterization validates physical plausibility and data coverage.
- Diagnosis claims must cite upstream evidence.
- Hypothesis validators check citation integrity and provenance downgrades.
- Production code should not import test stubs.
- Hypothesis agents should not import each other directly except through
  approved shared base abstractions.
- Runtime modules should not read from `audit/`.
- **Memory is opt-in**: NoopBackend default; SynapBackend only when
  `FERMDOCS_MEMORY=synap`.
- **Memory failures never break runs**: fetch returns []; write logs +
  skips. The D6 invariant that failed runs don't write is enforced in
  the runner's `_persist_lessons_to_memory` helper.
- **Closed-vocab process_family**: the enum is enforced at the LLM
  schema layer (Gemini structured output), at the manifest loader,
  and at the memory query layer.
- **The recommendation rubric overrides the LLM.** Model selection is the
  deterministic `rubric.py`, not the agent's self-report; refusal ↔
  no-interventions coherence is a schema validator; and the recommend
  stage never flips a run to FAILED.

Useful scripts:

```bash
python scripts/check_audit_invariant.py
python scripts/check_hypothesis_invariants.py
```

## LLM Provider Boundaries

Current provider reality:

```text
ingest mapper:        Gemini, Anthropic, fake
identity extractor:   Gemini, Anthropic (both with closed-enum schema)
unit normalizer:      rule-based plus optional Gemini/Anthropic fallback
narrative extraction: Gemini/Anthropic paths exist in ingest code
characterization:     deterministic plus optional Gemini trajectory analyzer
diagnosis:            Gemini or Anthropic clients, fake/none error path
hypothesis:           Gemini live path
recommendation:       Gemini or Anthropic (FERMDOCS_RECOMMEND_PROVIDER,
                      falls back to hypothesis/mapper provider); fake path
                      for offline tests. Model fitting itself is non-LLM.
memory:               Synap (managed; embedding handled by Synap internally)
embeddings (memory):  Synap-managed (no separate provider config needed)
```

Prompt composition in the hypothesis package is cache-friendly: stable policy
and contract layers are kept separate from volatile view payloads. That does
not mean every provider path uses explicit prompt caching.

## Budgets

Default hypothesis CLI budget:

```text
max_turns = 10
max_critic_cycles_per_topic = 3
max_tool_calls_total = 80
max_total_input_tokens = 200000
```

The API uses a larger budget for interactive runs:

```text
max_turns = 20
max_critic_cycles_per_topic = 6
max_tool_calls_total = 160
max_total_input_tokens = 400000
max_open_questions = 30
```

Raising one budget usually requires raising the related budgets. For example,
more turns generally means more tool calls and more token budget.

## Testing Strategy

The project uses layered testing:

```text
unit tests
  Fast deterministic checks for parsing, schema validation, metric toolkit,
  bundle behavior, agent contracts, chart generation, API offline paths,
  memory backends (Noop + Stub + SynapBackend with mocked SDK).

integration tests
  Pipeline-level tests with fakes or fixtures. Live Synap tests under
  tests/integration/memory/ are gated on SYNAP_API_KEY in env and skip
  cleanly when the key is absent.

evals
  Scripted characterization, diagnosis, and hypothesis reliability
  fixtures. Memory-layer eval harness (Phase 1 D9) for measuring
  prompt regression with priors injected vs disabled.

live_llm tests
  Opt-in tests that require API keys and cost tokens.
```

The default pytest config deselects `live_llm`.

Common commands:

```bash
pytest tests/unit -v
pytest tests/unit/memory tests/unit/hypothesis/memory -v
pytest tests/integration/memory -v           # requires SYNAP_API_KEY
cd apps/web && npm run typecheck && npm run build
```

## Operational Limitations

Current known limitations:

- The API run store is not durable across backend restarts.
- No auth, tenants (beyond logical tenant_id on memory), quotas, secrets
  management, or production deployment story is implemented.
- Full raw ingest requires Postgres.
- PDF quality depends on Docling extraction and source layout.
- Browser print is the current PDF export path.
- Hypothesis live execution is Gemini-only today.
- The pre-trained recommendation path only exists for `penicillin_fedbatch`
  (IndPenSim). Other families fall back to fitting models from the run's own
  data, which needs enough held-out points or the recommender refuses.
- Memory layer uses Synap (US-hosted). Customers requiring data residency
  would need an alternative backend (Postgres+pgvector adapter is
  scoped in the plan but not implemented).
- Synap dashboard's Memories panel is a mock-data preview; live records
  are inspectable via the SDK only.
- Some historical docs and package docstrings still reflect earlier stages
  of the project.

## Extension Guidance

### Adding a Golden Column

Edit:

```text
src/fermdocs/schema/golden_schema.yaml
```

Add real observed header examples. Examples are usually more useful to the
mapper than long prose descriptions.

### Adding a Process Family

Edit `src/fermdocs/schema/process_families.yaml`. The Gemini structured-output
schema and the manifest loader both read this dynamically — no other code
change required. The upload UI dropdown picks up the new family automatically
via `PROCESS_FAMILY_OPTIONS` in `apps/web/src/lib/api.ts` (add a readable
label there).

If the new family needs different KPI routing (product_variable,
precursor_variables, intracellular_product_variable, overflow_byproducts),
add those fields to the YAML entry. The catalog runner adapters will pick
them up.

### Adding a Characterization Metric

Add a deterministic toolkit function, register it in the metric catalog, and
ensure it emits stable evidence and `metric_id` metadata. Downstream routing
and hypothesis views rely on those metric IDs.

### Adding a Specialist

Add the specialist, projector view rules, prompts, schema/test coverage, and
then introduce deterministic top-K specialist routing. Do not add routing
while the specialist set remains the current fixed three.

### Adding a Memory Backend

Implement `MemoryBackend` Protocol in a new file under `src/fermdocs_memory/`.
Wire it into `_build_memory_backend()` in `apps/api/fermdocs_api/runner_pipeline.py`
behind a new `FERMDOCS_MEMORY=...` value. The Protocol's hard guarantees
(D7 raise, frozen records, write absorption) are enforced by `validate_query()`
and the runner's `_persist_lessons_to_memory` — your adapter just implements
the three Protocol methods.

### Extending the Recommendation Stage

The model recipes the agent fits live as vendored brewtwin skill cards
under `src/fermdocs_recommend/skills/` (loaded by `skill_loader.py` and
exposed through `tools_bundle/factory.py`). To change selection behavior,
edit the gates/thresholds in `rubric.py` — keep it LLM-free and I/O-free so
it stays the authoritative, testable arbiter. To add a pre-trained path for
a new process family, follow the `penicillin_fedbatch` pattern in
`registry.py` (cached artifact under `models/`, `has_base_model()` /
`cached_candidate()` wiring) and train offline via a `pretrain.py`-style
script. Add tests under `tests/recommend/` — the rubric tests assert the
verdict overrides the LLM's self-report.

### Adding a Provider

Keep provider-specific SDK code behind client factories. The rest of the
stage should consume typed request/response objects or small protocols, not
raw provider payloads.

### Changing Bundle Schema

Treat bundle schema changes as compatibility work:

1. Update schema/version metadata.
2. Update writer and reader together.
3. Add tests for old/new behavior.
4. Decide whether older bundles hard-fail or warn.

## Mental Model for Future Work

When changing the system, ask which boundary owns the behavior:

```text
raw extraction or provenance?               src/fermdocs
deterministic observations/findings?        src/fermdocs_characterize
observational summary?                      src/fermdocs_diagnose
causal mechanism debate?                    src/fermdocs_hypothesis
model fit / intervention / refusal?         src/fermdocs_recommend
cross-run priors / memory?                  src/fermdocs_memory
upload/run orchestration?                   apps/api
human workflow and display?                 apps/web
```

Keeping that separation is what lets the system evolve without every agent
and UI surface becoming coupled to every upstream implementation detail.
