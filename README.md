# fermdocs

`fermdocs` is a fermentation document intelligence pipeline. It ingests
CSV, Excel, PDF, or pre-built bundle uploads; preserves raw observations
with provenance; characterizes trajectories deterministically; runs an
observational diagnosis agent; and then runs a multi-agent hypothesis stage
with citations, human follow-up, charts, and a local web UI.

The current system is no longer just a parser. The live flow is:

```text
raw files or bundle zip
  -> ingest
  -> characterize
  -> diagnose
  -> hypothesize
  -> FastAPI + Next.js UI
```

For the deeper design, read [`ARCHITECTURE.md`](ARCHITECTURE.md).

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
- Produce a versioned bundle that downstream stages treat as the artifact
  boundary.
- Characterize observations into trajectories, findings, metadata anomalies,
  product/process KPIs, open questions, and narrative observations.
- Run an observational diagnosis agent with hard tool-use enforcement.
- Run a hypothesis debate with orchestrator, three specialists, synthesizer,
  critic, judge, lessons summarizer, HITL resume, and follow-up questions.
- Render deterministic Plotly charts from hypothesis `chart_specs`.
- Use a local FastAPI backend and Next.js frontend for upload, live events,
  hypotheses, charts, follow-up, and browser print-to-PDF.

## Repository Layout

```text
.
|-- src/fermdocs/                 ingest, parsing, mapping, storage, dossier, bundle
|-- src/fermdocs_characterize/    deterministic characterization + optional analyzer
|-- src/fermdocs_diagnose/        observational ReAct diagnosis agent
|-- src/fermdocs_hypothesis/      multi-agent causal hypothesis stage
|-- apps/api/                     FastAPI local backend
|-- apps/web/                     Next.js local frontend
|-- tests/                        unit, integration, eval-oriented tests
|-- evals/                        scripted evaluation fixtures/runners
|-- scripts/                      invariant and reliability checks
|-- plans/                        historical planning notes, useful but not canonical
|-- migrations/                   Postgres schema migrations
|-- ARCHITECTURE.md               current architecture and design decisions
```

## Prerequisites

- Python 3.11+
- Node.js 18+ for the web app
- Postgres 15+ for raw CSV/PDF/XLSX ingest
- Gemini API key for the current full live pipeline
- Optional Anthropic API key for supported mapper/diagnosis paths

The hypothesis stage currently uses the Gemini client. Anthropic support
exists for some earlier stages but is not the live hypothesis path.

## Setup

```bash
git clone https://github.com/dibyochakraborty-lemnisca/Docling_Parser_System.git
cd Docling_Parser_System

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

## API Surface

The local API lives under `/api`:

```text
GET  /api/health
POST /api/uploads
POST /api/runs
GET  /api/runs
GET  /api/runs/{run_id}
WS   /api/runs/{run_id}/events
POST /api/runs/{run_id}/answers
POST /api/runs/{run_id}/followup
```

The API is local-only by design. It has no auth, no tenancy model, and no
durable job queue.

## Configuration Cheat Sheet

Common environment variables:

```bash
# Database and storage
DATABASE_URL=postgresql+psycopg://fermdocs:fermdocs@localhost:5432/fermdocs
FERMDOCS_DATA_DIR=./data
FERMDOCS_API_ROOT=out/api
FERMDOCS_REPO_ROOT=/absolute/path/to/Docling_Parse

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

## Current Limitations

- The API run store is in-memory plus files on disk. Restarting the backend
  loses active run state.
- The web/API stack is local development software: no auth, RBAC,
  multi-tenancy, rate limiting, or production deployment hardening.
- Hypothesis live execution currently uses Gemini.
- PDF extraction quality depends on Docling and on source document quality.
- Browser print-to-PDF is the current PDF export path.
- Prompt structure is cache-friendly, but provider-level prompt caching is
  not a universal cross-stage feature.
- Some historical docs and package docstrings may still describe older v1
  behavior; prefer code, schemas, tests, and this document.

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
apps/api/fermdocs_api/runner_pipeline.py
apps/web/src/app/runs/[id]/page.tsx
```
