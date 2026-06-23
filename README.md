# FASSO — Fermentation Agentic Scientific Synthesis and Observation

FASSO is a fermentation document intelligence pipeline. It ingests
CSV, Excel, PDF, or pre-built bundle uploads; preserves raw observations
with provenance; characterizes trajectories deterministically; runs an
observational diagnosis agent; runs a multi-agent hypothesis stage with
citations and charts; fits/selects a process model and simulates the
intervention to recommend for the next run; runs an optimization stage
that debates the controllable levers and searches for the best operating
point (or honestly refuses); and persists distilled lessons to a managed
memory layer so successive runs on the same process family compound
rather than re-debate from scratch.

The current system is no longer just a parser. The live flow is:

```text
raw files or bundle zip
  -> ingest
  -> characterize
  -> diagnose
  -> hypothesize  <-- reads/writes cross-run memory
  -> recommend    <-- fits models, simulates interventions, or refuses
  -> optimize     <-- debates levers, discovers a model, searches for the best
                      operating point on the data, or honestly refuses
  -> FastAPI + Next.js UI
```

For first-time setup, read [`SETUP.md`](SETUP.md). For the deeper
design, read [`ARCHITECTURE.md`](ARCHITECTURE.md). For how the system
stays trustworthy as the agents evolve, read
[`docs/agent-safety-and-trust.md`](docs/agent-safety-and-trust.md).

## Current Capabilities

- Ingest `.csv`, `.xlsx`, `.pdf`, or a zipped existing bundle.
- Map raw headers to a canonical fermentation schema with Gemini,
  Anthropic, or a deterministic fake mapper for offline runs.
- Normalize units with pint plus optional LLM fallback for difficult unit
  strings.
- Preserve provenance back to source files, sheets, cells, pages, and
  narrative blocks.
- Segment multi-run PDFs and support operator-supplied process identity
  manifests.
- Extract narrative evidence from PDF prose, including closure events,
  deviations, interventions, observations, conclusions, and protocol notes.
- Operator-supplied **process family** dropdown at upload time. Closed
  enum from `process_families.yaml` — required for CSV-only bundles
  where the LLM identity extractor has no narrative to read from.
- Produce a versioned bundle that downstream stages treat as the artifact
  boundary.
- Characterize observations into trajectories, findings, metadata anomalies,
  product/process KPIs, open questions, and narrative observations.
- Run an observational diagnosis agent with hard tool-use enforcement.
- Run a hypothesis debate with orchestrator, three specialists, synthesizer,
  critic, judge, lessons summarizer, HITL resume, and follow-up questions.
- **Cross-run memory layer (Phase 1).** At clean-exit, the runner persists
  distilled lessons to a managed memory backend keyed by `process_family`.
  On the next run on the same family, those priors are retrieved and
  injected into the synthesizer/critic prompts as `[CROSS-RUN LESSONS]`.
  Memory is opt-in via `FERMDOCS_MEMORY=synap`; default is a noop.
- Render deterministic Plotly charts from hypothesis `chart_specs`.
- **Model-based recommendation stage.** After the run is DONE, the
  recommender bakes off candidate process models (mechanistic / surrogate /
  hybrid, plus a pre-trained IndPenSim path for `penicillin_fedbatch`),
  scores them on held-out runs with a pure deterministic rubric, and either
  simulates the best intervention to apply next or **honestly refuses** when
  no model is trustworthy. See [Recommendation Stage](#recommendation-stage).
- **Optimization stage.** Discovers the experiment's own controllable
  **levers** from the data (metadata design factors + varying observation
  channels — never a hardcoded knob list), runs an **opportunity debate**
  (the hypothesis engine seeded from those levers + observed trends), and
  searches for the best operating point. Against a process simulator it runs
  an active-learning loop with agent-written equation discovery; against the
  uploaded data alone it discovers a lever→titer model (a coupled ODE over the
  measured variables first, a static surrogate as fallback), validates it by
  leave-run-out cross-validation, and optimizes only within the observed
  envelope — or **honestly refuses**. See [Optimization Stage](#optimization-stage).
- **Lemnisca instrument-panel frontend** — dark "instrument panel" design
  system: true-black canvas, signature teal accent (`#38AFD8`), hairline
  rules, the lemniscate (∞) motif, and a Helvetica Neue / JetBrains Mono /
  Newsreader type stack. See [Frontend](#frontend-lemnisca-instrument-panel).
- Use a local FastAPI backend and Next.js frontend for upload, live events,
  hypotheses, charts, recommendations, follow-up, and browser print-to-PDF.

## Repository Layout

```text
.
|-- src/fermdocs/                 ingest, parsing, mapping, storage, dossier, bundle
|-- src/fermdocs_characterize/    deterministic characterization + optional analyzer
|-- src/fermdocs_diagnose/        observational ReAct diagnosis agent
|-- src/fermdocs_hypothesis/      multi-agent causal hypothesis stage
|-- src/fermdocs_recommend/       model bake-off + cross-run engine + intervention simulation + honest refusal
|-- src/fermdocs_optimize/        lever discovery, model/equation discovery, data + simulator oracles, search
|-- src/fermdocs_optimize_debate/ opportunity debate (hypothesis engine seeded from discovered levers + trends)
|-- src/fermdocs_memory/          MemoryBackend Protocol + Noop/Stub/Synap adapters
|-- apps/api/                     FastAPI local backend
|-- apps/web/                     Next.js local frontend
|-- tests/                        unit, integration, eval-oriented tests
|-- evals/                        scripted evaluation fixtures/runners
|-- scripts/                      invariant and reliability checks
|-- plans/                        historical planning notes, useful but not canonical
|-- migrations/                   Postgres schema migrations
|-- ARCHITECTURE.md               current architecture and design decisions
|-- SETUP.md                      clone-and-run guide
```

## Prerequisites

- Python 3.11+
- Node.js 18+ (or Bun) for the web app
- Postgres 15+ for raw CSV/PDF/XLSX ingest
- Gemini API key for the current full live pipeline
- Optional: Anthropic API key for supported mapper/diagnosis paths
- Optional: **Synap API key** for the memory layer (sign up at
  https://synap.maximem.ai — free tier available)

The hypothesis stage currently uses the Gemini client. Anthropic support
exists for some earlier stages but is not the live hypothesis path.

## Quick Start

For full setup with screenshots and troubleshooting, see [`SETUP.md`](SETUP.md).

The short version:

```bash
git clone https://github.com/dibyochakraborty-lemnisca/FASSO.git
cd fermdocs

# Python
python3.11 -m venv .venv
source .venv/bin/activate
pip install -e ".[dev,gemini,pdf]"
pip install -e "apps/api[dev]"

# Database
docker run -d --name fermdocs-pg \
  -e POSTGRES_USER=fermdocs -e POSTGRES_PASSWORD=fermdocs \
  -e POSTGRES_DB=fermdocs -p 5432:5432 postgres:16
alembic upgrade head

# Config
cp .env.example .env
# Edit .env: set GEMINI_API_KEY, optionally SYNAP_API_KEY + FERMDOCS_MEMORY=synap

# Frontend
cd apps/web && npm install && cd ../..

# Run
set -a; source .env; set +a
fermdocs-api &
cd apps/web && npm run dev
# Open http://localhost:3000
```

## Memory Layer

The hypothesis stage can read and write across-run memory. The design
is a frozen `MemoryBackend` Protocol with multiple adapters:

```text
src/fermdocs_memory/
|-- base.py       MemoryRecord, MemoryQuery, MemoryBackend Protocol
|-- noop.py       NoopBackend (default; off)
|-- stub.py       StubBackend (in-memory dict; for tests)
`-- synap.py      SynapBackend (production; wraps maximem-synap SDK)
```

**What gets persisted (Phase 1):** Distilled lessons from the
`lessons_summarizer` agent, written only when the run reaches a clean
`exit_reason` (`consensus_reached` or `no_topics_left`). Failed runs
and budget-exhausted runs do not pollute the store.

**Retrieval primary key:** `process_family` (closed vocab from
`process_families.yaml`). Maps to Synap's `user_id` scope. This is why
the upload UI requires a process family pick — CSV-only bundles have
no narrative for the LLM identity extractor to classify from.

**How to turn it on:**

```bash
# In your .env
FERMDOCS_MEMORY=synap
SYNAP_API_KEY=synap_...
FERMDOCS_TENANT_ID=lemnisca-internal   # optional; default "default"
```

**Without those env vars,** the system uses `NoopBackend` and behaves
exactly as it did before Phase 1 landed. Memory is fully opt-in.

**Deferred to later phases (with explicit triggers in the plan):**

- Tier 2: ratified-hypothesis store
- Tier 3: rejected-hypothesis store (negative examples)
- Tier 4: strain-conditional KPI prior table for the finding_validator
- Tier 5: human-correction memory (HITL responses become durable priors)

See `plans/2026-05-10-memory-layer.md` for the full roadmap and the
[Synap onboarding markdown](plans/synap_setup/fermdocs-dev-usecase.md).

## Recommendation Stage

After a run reaches DONE, the API runs a fifth stage that turns the
hypotheses into a concrete, tested recommendation for the next batch. It
lives in `src/fermdocs_recommend/` and writes
`<bundle>/recommend/recommendation.json`. It never flips a run to FAILED —
on any error it logs and the run stays DONE.

**What it does:**

1. **Candidate bake-off.** A ReAct agent (`RecommendationAgent`) fits three
   model families in a sandbox — **mechanistic** (ODE with fitted
   parameters), **surrogate** (neural ODE / LSTM, no parameters to vet),
   and **hybrid** (ODE + learned residual) — using a leave-one-run-out
   split and held-out scoring.
2. **Pre-trained IndPenSim path.** For `penicillin_fedbatch`, a
   control-augmented LSTM pre-trained on the 800-batch IndPenSim dataset is
   loaded directly (cached under `src/fermdocs_recommend/models/`,
   held-out penicillin R² ≈ 0.91) and used as the surrogate candidate.
3. **Deterministic rubric.** `rubric.py` — no LLM, no I/O — picks the
   winner. Gates: good fit (R² > 0.75 on every eligible species and RMSE
   within 2× the measurement floor), parameter plausibility (mechanistic /
   hybrid only), and optimizer movement (loss must drop > 5%, else the data
   doesn't constrain the model). A mechanistic model that passes wins unless
   a challenger beats it on RMSE by > 10%.
4. **Intervention simulation.** The winning model is used as an oracle: a
   line-search over the relevant control knobs (e.g. `Fs`, `Fg`, `RPM`,
   `Fpaa`) at 0.8–1.25× baseline, capped at mean + 3σ of the training titer
   to prevent oracle extrapolation, reporting the change that maximizes
   predicted peak titer.
5. **Honest refusal.** If no candidate clears the rubric, the stage refuses
   with a code (`poor_fit_all_models`, `insufficient_data`,
   `implausible_parameters`, `mechanism_not_supported`,
   `compute_budget_exhausted`, `brewtwin_not_installed`, `stage_error`) and
   emits **zero** interventions. A schema-level validator enforces the
   refusal ↔ no-interventions coherence.

The output (`RecommendationOutput`) carries `recommended_model`,
`confident`, `refusal_reason`, `selection_rationale`, the per-family
`candidates` (with R²/RMSE, fit/plausibility verdicts, offending params),
and the `interventions` (knob, baseline → predicted, delta, in-coverage
flag, caveat). The web run page renders this as the **Recommendation**
card. See [`ARCHITECTURE.md`](ARCHITECTURE.md#8-recommendation-output) for
the rubric constants and refusal codes.

## Optimization Stage

The optimization stage turns the analysis into a forward-looking question:
*given everything we measured, where is the headroom and what operating point
should the next batch use?* It lives in `src/fermdocs_optimize/` (the optimizer)
and `src/fermdocs_optimize_debate/` (the debate), and has two halves.

**1. Lever discovery.** `lever_discovery.py` reads the experiment's *own*
controllable inputs from the data — design factors in `run_conditions`
metadata (e.g. nitrogen source, feed concentration; numeric or categorical)
plus the initial conditions of observation channels that vary across runs.
There is no hardcoded knob list; a source becomes a lever only if it varies,
and the objective is never a lever.

**2. Opportunity debate** (`fermdocs_optimize_debate`). Reuses the hypothesis
debate engine, seeded forward-looking: one topic per discovered lever plus
observed **trend** topics from characterization. The specialists argue where
the titer headroom is. Observed trends (evidence-grounded) outrank speculative
levers in the ranker.

**3. Model discovery + search.** The API run path **always verifies against the
uploaded data itself** — real data wins, the LABS simulator is never used there
(de-LABS, 2026-06-16). The objective channel is resolved from the data + the
user's question (`fermdocs/analysis/objective.py`), not a fixed species.

- **data path (default)** — verify against the uploaded data, no simulator
  (`data_equation.py`). Discovers a **lever→objective model**: a coupled mechanistic
  ODE over *all* the measured variables first (`discovery/general_mech.py`),
  falling back to a static algebraic surrogate (numeric + one-hot categorical
  levers). Either model must clear a **leave-run-out cross-validation** gate, is
  optimized only over the **observed envelope**, and is checked by data-relative
  sanity guards: reject a prediction implausible vs the observed maximum, mark a
  boundary-sitting optimum as *insufficient data in that region* (not a validated
  optimum), and flag a fed-batch operating-mode mismatch. If nothing generalizes,
  it **refuses**.
- **LABS benchmark backend** (`fermdocs_optimize/benchmark/`, opt-in, CLI-only) —
  a synthetic process simulator for benchmarking the optimizer against a known
  answer. Reachable only via the standalone optimize CLIs (`FERMDOCS_OPTIMIZE_ORACLE`
  + a configured simulator); an import-guard test keeps it out of the data path.

Like recommend, the decision contract is coherent: a confident result carries a
best operating point; a refusal carries a reason and zero knobs. On sparse data
the honest outcome is often the surrogate or a refusal — the loop raises the
ceiling, it does not manufacture signal that isn't in the data.

## Setup at a Glance

```bash
git clone https://github.com/dibyochakraborty-lemnisca/FASSO.git
cd fermdocs

python3.11 -m venv .venv
source .venv/bin/activate

pip install -e ".[dev,gemini]"

# Optional, needed for PDF table extraction.
pip install -e ".[pdf]"

# Optional API package.
pip install -e "apps/api[dev]"

cp .env.example .env
```

Edit `.env`. For a normal live local run, the high-leverage values are:

```bash
DATABASE_URL=postgresql+psycopg://fermdocs:fermdocs@localhost:5432/fermdocs
FERMDOCS_MAPPER_PROVIDER=gemini
GEMINI_API_KEY=...
FERMDOCS_DATA_DIR=./data
FERMDOCS_API_ROOT=out/api

# Optional memory layer:
FERMDOCS_MEMORY=synap
SYNAP_API_KEY=...
FERMDOCS_TENANT_ID=lemnisca-internal
```

The Python CLIs do not all load `.env` in the same way. For shell use, the
least surprising approach is:

```bash
set -a
source .env
set +a
```

## Database

Raw ingest writes to Postgres. Bundle-only hypothesis runs do not need a
database, but full CSV/PDF/XLSX runs do.

```bash
docker run -d --name fermdocs-pg \
  -e POSTGRES_USER=fermdocs \
  -e POSTGRES_PASSWORD=fermdocs \
  -e POSTGRES_DB=fermdocs \
  -p 5432:5432 postgres:16

alembic upgrade head
```

If you already run Postgres locally:

```bash
createuser -s fermdocs
createdb -O fermdocs fermdocs
psql -d fermdocs -c "ALTER USER fermdocs WITH PASSWORD 'fermdocs';"
alembic upgrade head
```

## Run the Full Local App

Start the backend:

```bash
source .venv/bin/activate
set -a; source .env; set +a
fermdocs-api
```

Start the frontend:

```bash
cd apps/web
npm install
npm run dev
```

Open:

```text
http://localhost:3000
```

The UI accepts one or more raw `.csv`, `.xlsx`, or `.pdf` files. It also
accepts a single `.zip` containing an existing bundle. Zip uploads bypass
ingest/characterize/diagnose and run the hypothesis stage directly.

**Pick a process family on the upload page.** The dropdown is the
operator's way of telling the system what biology this is, especially
critical for CSV uploads where the LLM has no prose to classify from.
Options come from `src/fermdocs/schema/process_families.yaml`. Picking
"Auto-detect" runs the LLM identity extractor as before (works on PDFs).

## Run the Pipeline from the CLI

### 1. Ingest Raw Files

Offline deterministic smoke test:

```bash
fermdocs ingest \
  --experiment-id EXP-001 \
  --files tests/fixtures/sample_run.csv \
  --fake-mapper \
  --out out/EXP-001.dossier.json
```

Live mapper:

```bash
fermdocs ingest \
  --experiment-id EXP-001 \
  --files path/to/report.pdf path/to/data.xlsx \
  --out out/EXP-001.dossier.json
```

For CSV-only bundles, supply a process manifest so the dossier carries
a canonical `process_family`:

```bash
echo "process_family: penicillin_fedbatch" > manifest.yaml
echo "organism: Penicillium chrysogenum" >> manifest.yaml

fermdocs ingest \
  --experiment-id EXP-001 \
  --files path/to/data.csv \
  --process-manifest manifest.yaml \
  --out out/EXP-001.dossier.json
```

Useful ingest options:

```bash
--provider gemini|anthropic|fake
--no-llm-normalizer
--no-extract-narrative
--process-manifest path/to/process_manifest.yaml
--segment-pdfs / --no-segment-pdfs
--manifest-run-id RUN-001
--extract-narrative-insights / --no-extract-narrative-insights
```

### 2. Characterize and Write a Bundle

```bash
fermdocs-characterize out/EXP-001.dossier.json \
  --bundle out
```

This writes a bundle directory under `out/`, including the dossier,
characterization JSON, flattened observations CSV, optional narrative
observations, and bundle metadata.

To disable the optional LLM trajectory analyzer:

```bash
fermdocs-characterize out/EXP-001.dossier.json \
  --bundle out \
  --no-trajectory-analyzer
```

### 3. Diagnose the Bundle

```bash
fermdocs-diagnose run \
  --bundle out/bundle_<id>
```

With a bundle, diagnosis writes:

```text
out/bundle_<id>/diagnosis/diagnosis.json
```

### 4. Run Hypothesis

```bash
fermdocs-hypothesize run out/bundle_<id> \
  --out-root out/hypothesis
```

With an initial user question:

```bash
fermdocs-hypothesize run out/bundle_<id> \
  --question "Why did RUN-2 lose carotenoid productivity after 80 hours?"
```

Outputs include:

```text
out/hypothesis/<hypothesis_run_id>/global.md
out/hypothesis/<hypothesis_run_id>/hypothesis_output.json
```

`global.md` is the human-readable event log. The JSON output is the
structured contract.

The CLI uses `NoopBackend` for memory by default. The API runner reads
`FERMDOCS_MEMORY=synap` to enable the live memory backend.

### 5. Recommend

Bake off process models and simulate the next-batch intervention:

```bash
fermdocs-recommend \
  --bundle out/bundle_<id> \
  --hypothesis-output out/hypothesis/<hyp_run_id>/hypothesis_output.json \
  --provider gemini
```

`--hypothesis-output` is optional — it points the recommender at the final
hypotheses so interventions stay grounded in `affected_variables`. Output
defaults to `<bundle>/recommend/recommendation.json`; override with
`--output`. In the live app this stage runs automatically once a run is
DONE, so the CLI is mainly for offline reproduction and debugging.

### 6. Human-in-the-Loop (HITL) Interaction

When running the hypothesis stage, you can explicitly enable HITL mode. If the agents encounter ambiguity and emit `open_questions`, the system will pause and ask the operator for ground-truth answers.

```bash
fermdocs-hypothesize run \
  --bundle-dir out/bundle-EXP-001 \
  --hitl \
  --out-dir out/hyp-EXP-001
```

If open questions are generated, the CLI prompts:
`N open question(s). Answer them and re-run? [Y/n]:`

The user's answers are captured as `HumanInputReceivedEvent`s and `QuestionResolvedEvent`s, appended to the event log, and the stage automatically resumes. The agents then factor the operator's input into the next round of debate.

### 7. Optimize

Run the opportunity debate over a bundle (discovers the levers, argues where the
headroom is):

```bash
fermdocs-optimize-debate run out/bundle_<id> \
  --objective P \
  --out-dir out/optimize
```

The model-search half has two entry points. Against the **uploaded data** (no
simulator) it runs inside the API runner once a run is DONE — automatically,
whenever the bundle has observations — discovering a lever→objective model,
validating it by leave-run-out CV, and optimizing within the observed envelope or
refusing. The **LABS benchmark backend** (equation discovery + oracle-verified
search against a synthetic simulator) is opt-in and CLI-only — it needs a process
simulator, its true params, and a search box, and is never used on the API path:

```bash
fermdocs-optimize \
  --train train_data.csv \
  --mech-params mech_params.json \
  --box config.json \
  --rounds 6 --proposals 4 \
  --out out/optimization.json
```

In the live app the optimize stage runs automatically off the bundle; the CLIs
are for offline reproduction and debugging.

## API Surface

The local API lives under `/api`:

```text
GET  /api/health
POST /api/uploads          form fields: files[], process_family (optional)
POST /api/runs
GET  /api/runs
GET  /api/runs/{run_id}
WS   /api/runs/{run_id}/events
POST /api/runs/{run_id}/answers
POST /api/runs/{run_id}/followup
```

The API is local-only by design. It has no auth, no tenancy model, and no
durable job queue.

`GET /api/runs/{run_id}` returns the full run detail, including the
hypothesis `output`, any `followups`, and — once the recommend stage has
run — a `recommendation_output` object (`recommended_model`, `confident`,
`refusal_reason`, `selection_rationale`, `candidates`, `interventions`).
The recommendation is a field on the run detail, not a separate endpoint.

## Frontend (Lemnisca instrument panel)

The web UI uses the **Lemnisca design system** — a dark "instrument
panel" aesthetic, replacing the earlier light editorial direction.

Type stack (via `next/font/google` + system grotesk):

- **Helvetica Neue** — body + headings (system grotesk).
- **JetBrains Mono** (`--font-ui`) — every label, eyebrow, tag, stat.
- **Newsreader** (`--font-display`) — sparing serif-italic accents.

Palette (CSS variables in `apps/web/src/app/globals.css`):

```text
--color-bg        #000000   true-black canvas
--color-surface-1 #0b0c0d   card surfaces (2/3 for hover, wells)
--color-ink       #f4f5f6   primary text (ramp down to #62686c)
--color-rule      rgba(255,255,255,0.10)  hairline dividers
--color-accent    #38AFD8   signature teal — used sparingly
--color-ok/-warn/-error      #3fbfa6 / #e3a552 / #e5484d
```

Signature elements: the animated **lemniscate (∞)** motif on the home
hero, teal **glow** instead of drop shadows, a scroll-progress bar, and
the run-page **debate stream** where each specialist agent gets a colored
avatar + live token count and contributes dialog bubbles. Final
hypotheses render two-up (reasoning left, charts right). Accent and
semantic colors expose RGB-channel triples so Tailwind alpha modifiers
(`/10`, `/40`) work on tinted fills and borders.

## Configuration Cheat Sheet

Common environment variables:

```bash
# Database and storage
DATABASE_URL=postgresql+psycopg://fermdocs:fermdocs@localhost:5432/fermdocs
FERMDOCS_DATA_DIR=./data
FERMDOCS_API_ROOT=out/api
FERMDOCS_REPO_ROOT=/absolute/path/to/fermdocs

# Mapper / ingest
FERMDOCS_MAPPER_PROVIDER=gemini
FERMDOCS_GEMINI_MODEL=gemini-3-flash
FERMDOCS_MAPPER_MODEL=claude-haiku-4-5-20251001
FERMDOCS_SCHEMA_PATH=/path/to/golden_schema.yaml
FERMDOCS_USE_LLM_NORMALIZER=true
FERMDOCS_NORMALIZER_PROVIDER=gemini

# PDF and narrative extraction
FERMDOCS_PDF_OCR=false
FERMDOCS_PDF_SEGMENT=true
FERMDOCS_SEGMENTER_PROVIDER=gemini
FERMDOCS_SEGMENTER_MODEL=gemini-3-pro
FERMDOCS_EXTRACT_NARRATIVE=true
FERMDOCS_EXTRACT_NARRATIVE_INSIGHTS=true
FERMDOCS_NARRATIVE_PROVIDER=gemini
FERMDOCS_NARRATIVE_MODEL=gemini-3-pro

# Later stages
FERMDOCS_CHARACTERIZE_PROVIDER=gemini
FERMDOCS_CHARACTERIZE_MODEL=gemini-3-pro
FERMDOCS_DIAGNOSIS_PROVIDER=gemini
FERMDOCS_DIAGNOSIS_MODEL=gemini-3-pro
FERMDOCS_HYPOTHESIS_PROVIDER=gemini
FERMDOCS_HYPOTHESIS_MODEL=gemini-3-pro
FERMDOCS_QUESTION_CLASSIFIER_MODEL=gemini-3-flash

# Recommendation stage (falls back to hypothesis/mapper provider when unset)
FERMDOCS_RECOMMEND_PROVIDER=gemini       # gemini | anthropic
FERMDOCS_RECOMMEND_MODEL=gemini-3-pro    # falls back to FERMDOCS_GEMINI_MODEL
FERMDOCS_RECOMMEND_MAX_OUTPUT_TOKENS=    # optional override

# Unit-mislabel guard: if a "g/L"-tagged column actually holds mg/L, values
# inflate 1000x; this rejects conversions beyond N x the nominal range.
FERMDOCS_UNIT_PLAUSIBILITY_FACTOR=50

# Memory layer
FERMDOCS_MEMORY=synap                 # noop (default) | synap
SYNAP_API_KEY=...
SYNAP_INSTANCE_ID=                    # optional; resolved from API key when blank
FERMDOCS_TENANT_ID=lemnisca-internal  # multi-tenant scope; default "default"

# Keys
GEMINI_API_KEY=...
ANTHROPIC_API_KEY=...

# Debug prompt/response payloads
FERMDOCS_DEBUG_MAPPER=1
FERMDOCS_DEBUG_IDENTITY=1
FERMDOCS_DEBUG_SEGMENTER=1
FERMDOCS_DEBUG_CHARACTERIZE=1
FERMDOCS_DEBUG_DIAGNOSIS=1
FERMDOCS_DEBUG_HYPOTHESIS=1
```

## Testing and Checks

The suite is large by design — ~1,392 unit tests and ~1,628 tests total across
178 files. Most safety mechanisms (claim guard, finding validator, citation
integrity, refusal coherence, optimizer sanity guards) carry regression tests
written from real bad outputs, so the same mistake can't silently return. See
[`docs/agent-safety-and-trust.md`](docs/agent-safety-and-trust.md).

Fast default test run:

```bash
pytest tests/unit -v
```

Broader local run:

```bash
pytest tests -v
```

Memory layer tests:

```bash
pytest tests/unit/memory tests/unit/hypothesis/memory -v
# Live Synap integration test (requires SYNAP_API_KEY in env):
pytest tests/integration/memory -v
```

Frontend:

```bash
cd apps/web
npm run typecheck
npm run build
```

Useful invariant/eval scripts:

```bash
python scripts/check_audit_invariant.py
python scripts/check_hypothesis_invariants.py
python scripts/eval_hypothesis_reliability.py --help
```

The default pytest config deselects `live_llm`. Live LLM tests/evals require
keys and will cost tokens.

## Design Decisions to Preserve

- The bundle is the artifact boundary between stages.
- `meta.json` is the bundle readiness signal and is written last.
- Runtime code should not read `audit/` artifacts as evidence.
- Characterization should prefer deterministic metrics and validators before
  optional LLM analysis.
- Diagnosis is observational: what happened, what trended, what is uncertain.
- Hypothesis is causal: mechanisms, alternatives, critiques, judgments, and
  actionable recommendations.
- Diagnosis open questions do not carry `re_run_from`; hypothesis open
  questions do.
- User questions are a lens over the evidence, not a replacement for the
  bottom-up analysis.
- **The recommendation rubric is deterministic and authoritative.** The LLM
  fits models and self-reports, but `rubric.py` (no LLM) makes the final
  call. A model that lies about its fit is overruled.
- **Honest refusal is a feature.** When no model clears the rubric, the
  recommender emits a refusal code and zero interventions rather than a
  confident-sounding guess. Schema-level coherence is enforced.
- Specialist routing is intentionally static while there are only three
  specialists. Add routing when there is a fourth specialist.
- **No hardcoded domain values.** No baked-in nominal/spec/expected constants;
  data is judged against its own distribution (data-relative). The optimizer's
  levers are discovered from the data, not from a fixed knob list.
- **Claim guard.** A shared deterministic check (`src/fermdocs/claim_guard.py`)
  rejects agent claims that contradict the data (false "unavailable", oxygen
  limitation on an anaerobic run, scale confound when scale is constant, a rate
  "at t=0"). Wired into characterize (finding source) and hypothesis (output).
- **The optimizer refuses too.** Like recommend, it emits a best operating point
  only when a model clears cross-validation and the optimum is inside the
  observed data; otherwise it refuses with a reason.
- **Memory is opt-in and gated on a closed-vocab `process_family`.**
  `NoopBackend` is the default; `SynapBackend` activates via env var.
  No `process_family` → memory silently no-ops.
- **Memory failures never break runs.** Backend outage on fetch returns
  empty priors; outage on write logs a warning and continues. The D6
  invariant: failed/budget-exhausted runs never write.
- The closed-enum `process_family` (penicillin_fedbatch,
  yeast_intracellular_product_fedbatch, yeast_aerobic_fedbatch,
  ecoli_recombinant_protein, melanin_batch) is enforced at the LLM
  schema level, in the manifest loader, and at the memory layer.

## Current Limitations

- The API run store is in-memory plus files on disk. Restarting the backend
  loses active run state.
- The web/API stack is local development software: no auth, RBAC,
  multi-tenancy enforcement, rate limiting, or production deployment
  hardening. Memory layer carries a `tenant_id` field but isolation
  is currently logical (no separate Synap instances per tenant).
- Hypothesis live execution currently uses Gemini.
- PDF extraction quality depends on Docling and on source document quality.
- Browser print-to-PDF is the current PDF export path.
- Prompt structure is cache-friendly, but provider-level prompt caching is
  not a universal cross-stage feature.
- Some historical docs and package docstrings may still describe older v1
  behavior; prefer code, schemas, tests, and this document.
- Synap dashboard's Memories tab is currently a mock-data preview; live
  records are inspectable via the SDK only.

## Where to Start in Code

Read in this order:

```text
src/fermdocs/domain/models.py
src/fermdocs/pipeline.py
src/fermdocs/bundle/writer.py
src/fermdocs/bundle/reader.py
src/fermdocs_characterize/schema.py
src/fermdocs_characterize/pipeline.py
src/fermdocs_diagnose/schema.py
src/fermdocs_diagnose/agent.py
src/fermdocs_hypothesis/schema.py
src/fermdocs_hypothesis/runner.py
src/fermdocs_hypothesis/projector.py
src/fermdocs_recommend/schema.py
src/fermdocs_recommend/agent.py
src/fermdocs_recommend/rubric.py
src/fermdocs_recommend/registry.py
src/fermdocs_recommend/cross_run.py
src/fermdocs_optimize/lever_discovery.py
src/fermdocs_optimize/data_equation.py
src/fermdocs_optimize/discovery/general_mech.py
src/fermdocs_optimize_debate/topics.py
src/fermdocs_memory/base.py
src/fermdocs_memory/synap.py
src/fermdocs/claim_guard.py
apps/api/fermdocs_api/runner_pipeline.py
apps/web/src/app/runs/[id]/page.tsx
```
