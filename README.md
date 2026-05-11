# fermdocs

`fermdocs` is a fermentation document intelligence pipeline. It ingests
CSV, Excel, PDF, or pre-built bundle uploads; preserves raw observations
with provenance; characterizes trajectories deterministically; runs an
observational diagnosis agent; runs a multi-agent hypothesis stage with
citations, charts, and recommendations; and now persists distilled
lessons to a managed memory layer so successive runs on the same
process family compound rather than re-debate from scratch.

The current system is no longer just a parser. The live flow is:

```text
raw files or bundle zip
  -> ingest
  -> characterize
  -> diagnose
  -> hypothesize  <-- reads/writes cross-run memory
  -> FastAPI + Next.js UI
```

For first-time setup, read [`SETUP.md`](SETUP.md). For the deeper
design, read [`ARCHITECTURE.md`](ARCHITECTURE.md).

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
- **Editorial Scientific frontend** — Phase 1 type + color foundations are
  live (Fraunces + Hanken Grotesk, paper/ink/forest palette, light-only).
  Phase 2 layouts + Phase 3 chart styling are scheduled.
- Use a local FastAPI backend and Next.js frontend for upload, live events,
  hypotheses, charts, follow-up, and browser print-to-PDF.

## Repository Layout

```text
.
|-- src/fermdocs/                 ingest, parsing, mapping, storage, dossier, bundle
|-- src/fermdocs_characterize/    deterministic characterization + optional analyzer
|-- src/fermdocs_diagnose/        observational ReAct diagnosis agent
|-- src/fermdocs_hypothesis/      multi-agent causal hypothesis stage
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
git clone https://github.com/Lemniscabio/fermdocs.git
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

## Setup at a Glance

```bash
git clone https://github.com/Lemniscabio/fermdocs.git
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

## Frontend (Editorial Scientific)

The web UI is being redesigned in the Editorial Scientific direction —
Quanta/Nature register, generous whitespace, single forest accent,
asymmetric two-column grid. Phase 1 (typography + color tokens) is
live as of `frontend-styling` branch. Phase 2 (layouts) and Phase 3
(editorial chart styling) are scheduled.

Type stack: **Fraunces** (variable serif) + **Hanken Grotesk** (UI).
Both free, hosted via `next/font/google`. Color: paper #FBFAF7, ink
#0F1B2D, forest accent #1B4D3E. Light-only — dark mode was dropped
for v1.

See `plans/2026-05-11-frontend-redesign-editorial.md` for the full
design doc.

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
- Specialist routing is intentionally static while there are only three
  specialists. Add routing when there is a fourth specialist.
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
src/fermdocs_memory/base.py
src/fermdocs_memory/synap.py
apps/api/fermdocs_api/runner_pipeline.py
apps/web/src/app/runs/[id]/page.tsx
```
