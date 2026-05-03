# apps/ — frontend + backend for the hypothesis stage

Plan ref: [plans/2026-05-03-hypothesis-debate-v0.md](../plans/2026-05-03-hypothesis-debate-v0.md) (v0.5).

## Layout

- **`apps/api/`** — FastAPI backend (`fermdocs-api`) that wraps the
  Python pipeline (ingest → characterize → diagnose → hypothesize)
  behind an HTTP/WebSocket surface. Local-only by default.
- **`apps/web/`** — Next.js 14 (App Router) + Tailwind + shadcn/ui
  frontend. Polished UI for upload, live debate timeline, open-question
  answers, final hypothesis card, token report.

## Run locally

```bash
# Backend (port 8000)
pip install -e apps/api/[dev]
fermdocs-api

# Frontend (port 3000) — proxies /api/* to the backend
cd apps/web
npm install
npm run dev
```

Then open <http://localhost:3000>.

## What v0.5a/b ship

- Upload a `.zip` of an existing diagnose bundle (containing `meta.json`)
- Backend kicks off the hypothesis stage, streams events over WebSocket
- Frontend shows live debate timeline (topic → facets → synthesis →
  critique → judge → accept/reject), final hypotheses, rejected
  hypotheses, open questions form, and per-agent token report
- Submit answers to open questions to trigger a resume round

## What's deferred to v1

- CSV/PDF upload that runs the full ingest+characterize+diagnose pipeline
  (today: must upload a pre-built bundle zip)
- Auth / multi-user / cloud deployment (today: localhost only)
- Persistent run store (today: in-process; restart = lose runs)
- SuperMemory / past-insight retrieval
- Visual debate replay (today: events stream live; replay works but UX
  could be polished)
