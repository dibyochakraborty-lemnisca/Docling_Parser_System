# apps

Local frontend and backend for the full fermdocs pipeline.

## Layout

```text
apps/api
  FastAPI backend. It accepts uploads, starts runs, streams events over
  websockets, stores local run outputs, resumes paused runs, and launches
  follow-up hypothesis runs.

apps/web
  Next.js 14 app. It provides upload, optional user question input, live
  event timeline, hypothesis cards, inline Plotly charts, follow-up UI, and
  browser print-to-PDF.
```

## Run Locally

From the repo root:

```bash
source .venv/bin/activate
pip install -e ".[dev,gemini]"
pip install -e "apps/api[dev]"

# Optional, needed for PDF extraction.
pip install -e ".[pdf]"

set -a; source .env; set +a
alembic upgrade head
fermdocs-api
```

In another terminal:

```bash
cd apps/web
npm install
npm run dev
```

Open:

```text
http://localhost:3000
```

## Supported Uploads

- One or more raw `.csv`, `.xlsx`, or `.pdf` files. These run the full
  ingest -> characterize -> diagnose -> hypothesize pipeline.
- A single `.zip` containing an existing bundle. Zip uploads bypass upstream
  stages and run hypothesis directly.

Zip uploads cannot be mixed with raw files in the same request.

## API Endpoints

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

## Limitations

- Local development only.
- No auth, user accounts, tenant isolation, or production job queue.
- Run state is in memory plus files under `FERMDOCS_API_ROOT`.
- Restarting the backend loses active in-memory run state.
