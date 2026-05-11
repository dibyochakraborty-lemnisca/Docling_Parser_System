# SETUP

Clone-and-run guide for fermdocs. Walks through every dependency from scratch
to a working local UI in 15-20 minutes.

For background on what fermdocs *is*, read [README.md](README.md).
For the design rationale, read [ARCHITECTURE.md](ARCHITECTURE.md).

---

## Table of Contents

1. [Prerequisites](#1-prerequisites)
2. [Clone and Python environment](#2-clone-and-python-environment)
3. [Database](#3-database-postgres)
4. [API keys](#4-api-keys)
5. [Environment file](#5-environment-file)
6. [Frontend dependencies](#6-frontend-dependencies)
7. [First run](#7-first-run)
8. [Memory layer setup](#8-memory-layer-setup-optional-but-recommended)
9. [Verifying it all works](#9-verifying-it-all-works)
10. [Troubleshooting](#10-troubleshooting)

---

## 1. Prerequisites

You need these on your machine before starting:

| Tool | Version | How to check |
|---|---|---|
| Python | 3.11 or newer | `python3 --version` |
| Node.js | 18 or newer (or Bun ≥ 1.0) | `node --version` |
| Docker | any recent | `docker --version` |
| Git | any recent | `git --version` |

On macOS, the easiest install path is Homebrew:

```bash
brew install python@3.11 node git
brew install --cask docker
```

Open Docker Desktop once after install so the daemon is running.

On Linux:

```bash
# Debian/Ubuntu
sudo apt update
sudo apt install -y python3.11 python3.11-venv nodejs npm git docker.io
sudo systemctl enable --now docker
sudo usermod -aG docker $USER  # log out + back in for this to apply
```

You also need accounts to obtain API keys (see step 4):

- **Google AI Studio** (for Gemini) — https://aistudio.google.com/apikey
- **Maximem Synap** (optional, for the memory layer) — https://synap.maximem.ai
- **Anthropic** (optional) — https://console.anthropic.com

---

## 2. Clone and Python environment

```bash
git clone https://github.com/Lemniscabio/fermdocs.git
cd fermdocs

python3.11 -m venv .venv
source .venv/bin/activate
```

Install the Python packages. The base install + dev tools + Gemini support
is the minimum for a live run:

```bash
pip install --upgrade pip
pip install -e ".[dev,gemini]"
```

Add the optional extras you'll likely want:

```bash
# PDF table extraction (Docling). Needed for PDF uploads.
pip install -e ".[pdf]"

# API server.
pip install -e "apps/api[dev]"
```

Sanity-check the install:

```bash
fermdocs --help
fermdocs-characterize --help
fermdocs-diagnose --help
fermdocs-hypothesize --help
fermdocs-api --help
```

If any of those error out, see [Troubleshooting](#10-troubleshooting).

---

## 3. Database (Postgres)

Raw CSV/XLSX/PDF ingest writes to Postgres. Bundle-only hypothesis runs
(uploading a `.zip` bundle) do not need a database. For first-time setup,
get Postgres running anyway — most realistic uploads will be raw files.

The fastest path is a Docker container:

```bash
docker run -d --name fermdocs-pg \
  -e POSTGRES_USER=fermdocs \
  -e POSTGRES_PASSWORD=fermdocs \
  -e POSTGRES_DB=fermdocs \
  -p 5432:5432 \
  postgres:16
```

Verify it's running:

```bash
docker ps | grep fermdocs-pg
# Should show: fermdocs-pg ... 0.0.0.0:5432->5432/tcp
```

Apply schema migrations:

```bash
alembic upgrade head
```

You should see a series of `Running upgrade ... -> ...` lines ending in
"head". If you get an error about not being able to connect, the most
common cause is the env var isn't set yet — skip ahead to step 5, set
`DATABASE_URL`, then come back and re-run `alembic upgrade head`.

**Already running Postgres locally?** Use it directly:

```bash
createuser -s fermdocs
createdb -O fermdocs fermdocs
psql -d fermdocs -c "ALTER USER fermdocs WITH PASSWORD 'fermdocs';"
alembic upgrade head
```

---

## 4. API keys

### Required: Gemini

fermdocs's live hypothesis path uses Gemini.

1. Visit https://aistudio.google.com/apikey
2. Sign in with a Google account.
3. Click "Create API key" and pick or create a Google Cloud project.
4. Copy the key (starts with `AIzaSy...`).

Free tier is generous and covers normal development use. Heavy iteration
(re-running the same bundle 10+ times in a day) may bump into rate limits.

### Optional but recommended: Synap (memory layer)

Synap is the managed memory backend that lets successive runs on the same
process family compound rather than re-debating from scratch.

1. Visit https://synap.maximem.ai
2. Sign up (Google or GitHub auth).
3. Create an instance — name it `fermdocs-dev` for clarity.
4. Open the API Keys page in the dashboard sidebar.
5. Create a key. Copy it (starts with `synap_...`).

The free tier ships with $12 of credit, which is plenty for first
weeks of dev. You don't need to upload the use-case markdown unless
you want the dashboard's onboarding context (the file is at
`plans/synap_setup/fermdocs-dev-usecase.md` if you want to upload it).

### Optional: Anthropic

If you want to test the Anthropic mapper / identity / diagnosis paths:

1. Visit https://console.anthropic.com
2. Create an API key.

---

## 5. Environment file

```bash
cp .env.example .env
```

Open `.env` in your editor and set, at minimum:

```bash
DATABASE_URL=postgresql+psycopg://fermdocs:fermdocs@localhost:5432/fermdocs
GEMINI_API_KEY=AIzaSy...your-key-here...
FERMDOCS_DATA_DIR=./data
FERMDOCS_API_ROOT=out/api
```

If you signed up for Synap and want the memory layer on (recommended):

```bash
FERMDOCS_MEMORY=synap
SYNAP_API_KEY=synap_...your-key-here...
FERMDOCS_TENANT_ID=default
```

If you skipped Synap, leave `FERMDOCS_MEMORY=noop` and the system runs
exactly as it would without the memory layer — fully optional.

Make the file readable by your shell session:

```bash
set -a; source .env; set +a
```

(Add that line to your shell `rc` if you use this repo daily.)

Now re-run the migration if it didn't complete in step 3:

```bash
alembic upgrade head
```

---

## 6. Frontend dependencies

The web UI is Next.js. Install via npm or Bun:

```bash
cd apps/web
npm install
cd ../..
```

Or with Bun (faster):

```bash
cd apps/web
bun install
cd ../..
```

Verify it builds:

```bash
cd apps/web
npm run typecheck
npm run build
cd ../..
```

A clean `Compiled successfully` is what you want.

---

## 7. First run

You need two terminals.

**Terminal 1 — API:**

```bash
source .venv/bin/activate
set -a; source .env; set +a
fermdocs-api
```

Wait for the line `Uvicorn running on http://127.0.0.1:8000`. If you set
`FERMDOCS_MEMORY=synap`, you should also see (on the first real run, not
boot):

```
synap-backend: initialized instance_id='...' tenant='default'
```

**Terminal 2 — Web:**

```bash
cd apps/web
npm run dev
```

Wait for `Ready in ... ms`. Open http://localhost:3000.

You should see the upload page with:
- A "Your question" textarea
- A **Process family** dropdown (auto-detect + 5 closed-vocab options)
- A file picker
- A "Submit" button

If anything is missing, see [Troubleshooting](#10-troubleshooting).

### Your first upload

The simplest first test is the included penicillin synthetic fixture
(if it exists in your clone — it's used by some integration tests):

```bash
ls tests/fixtures/*.csv 2>/dev/null
```

If none exist, use any small fermentation CSV with these columns at minimum:
- a time column (`time_h` or similar)
- one or more measured variables (`biomass_g_l`, `dissolved_o2_mg_l`, etc.)

In the UI:

1. Type a question like *"Why did some batches outperform others?"*
2. **Pick a process family** from the dropdown matching your data:
   - Penicillin data → `Penicillin fed-batch`
   - Yeast carotenoid/lipid data → `Yeast — intracellular product`
   - Yeast biomass data → `Yeast — aerobic fed-batch`
   - E. coli recombinant protein data → `E. coli — recombinant protein`
   - Anything else / not sure → `Auto-detect (LLM)`
3. Click "Add file", pick your CSV
4. Click "Submit for analysis"

The run page opens immediately. Status updates stream in via WebSocket:
`ingesting → characterizing → diagnosing → hypothesizing → done`.

Total time depends on file size and Gemini latency: 8-15 minutes for a
typical 4-6 batch carotenoid dataset.

---

## 8. Memory layer setup (optional but recommended)

The memory layer ships as opt-in. If you set `FERMDOCS_MEMORY=synap` in
step 5, the live backend is already wired. There's no separate setup
step beyond providing the API key — the runner will:

- **On clean run completion**, persist distilled lessons to Synap keyed
  by `process_family` (i.e. the dropdown you picked at upload time).
- **On the next run on the same `process_family`**, retrieve up to 5
  priors and inject them into the synthesizer + critic prompts.

### Verifying memory writes

After your first successful run completes:

1. Open the Synap dashboard at https://synap.maximem.ai
2. Click **Requests** in the sidebar.
3. You should see several `ADD` rows with timestamps matching your run's
   end time. Each row is one distilled lesson being written.

If you see no `ADD` rows after a run completes:

```bash
# Check the API logs (Terminal 1)
# Look for either:
#   "synap-backend: initialized"   -> SDK connected
#   "synap-backend: write failed" -> outage / auth issue
# Or no synap line at all -> FERMDOCS_MEMORY isn't reaching the API
```

The most common cause of "no synap line" is the API server not having
the env var loaded. Make sure you ran `set -a; source .env; set +a`
in **Terminal 1** before starting `fermdocs-api`, not in another
terminal.

### Verifying memory reads

To confirm the read path works, run a *second* bundle of the same
process family. In the run's `global.md`:

```bash
grep -A 5 "CROSS-RUN LESSONS" out/api/runs/<RUN_UUID>/global.md
```

On the cold first run, the block is empty. On the warm second run, you
should see up to 5 priors retrieved from Synap and pasted into the
synthesizer prompt.

If the block is empty even on the second run, the most likely cause is
that `process_family` was never written to the dossier. Check:

```bash
parsevenv/bin/python -c "
import json
d = json.load(open('out/api/runs/<RUN_UUID>/...path to bundle/dossier.json'))
print(d['experiment']['process']['registered']['process_family'])
"
```

If that prints `None`, the upload's dropdown pick didn't make it
through. Re-upload the bundle and pick a specific process family
(not Auto-detect) on the upload page.

---

## 9. Verifying it all works

Run the test suite to make sure nothing regressed during install:

```bash
source .venv/bin/activate
pytest tests/unit -q
```

Expected: ~1300+ tests pass, 2-3 may skip without API keys, 0-2 may
fail (pre-existing on certain branches; check with `git log`).

Memory-layer-specific tests:

```bash
pytest tests/unit/memory tests/unit/hypothesis/memory -q
```

Live Synap integration test (requires `SYNAP_API_KEY` in env):

```bash
set -a; source .env; set +a
pytest tests/integration/memory -q
```

Frontend typecheck:

```bash
cd apps/web && npm run typecheck && cd ../..
```

Useful invariant scripts:

```bash
python scripts/check_audit_invariant.py
python scripts/check_hypothesis_invariants.py
```

---

## 10. Troubleshooting

### `fermdocs: command not found`

Your virtualenv isn't activated, or the install failed.

```bash
source .venv/bin/activate
pip install -e ".[dev,gemini]"
```

If `pip install` errors, check `python --version` is 3.11+ inside the venv.

### `psycopg.OperationalError: could not connect to server`

The Postgres container isn't running.

```bash
docker ps | grep fermdocs-pg
# If not listed:
docker start fermdocs-pg
# If the container doesn't exist:
# (re-run step 3's docker run command)
```

### `alembic.util.exc.CommandError: Can't locate revision identified by 'head'`

You're running alembic from the wrong directory. Run it from the repo root:

```bash
cd /path/to/fermdocs
alembic upgrade head
```

### `KeyError: 'GEMINI_API_KEY'` or `400 Bad Request` from Gemini

The env var isn't set in the shell running the API.

```bash
# In Terminal 1, before starting fermdocs-api:
set -a; source .env; set +a
fermdocs-api
```

### Frontend shows "Cannot connect to API"

The API server isn't running, or it's running on a different port.

```bash
# In another terminal:
curl http://localhost:8000/api/health
# Expected: {"status":"ok"}
```

If you get connection refused, restart `fermdocs-api` in Terminal 1.

### Process family dropdown not showing

You're on an older commit before the `upload-process-family-ui` branch
landed. Pull latest:

```bash
git pull origin main  # or whichever branch you're tracking
cd apps/web && npm run build && cd ../..
```

### `synap-backend: initialized` never appears in API logs

`FERMDOCS_MEMORY=synap` isn't reaching the API process.

```bash
# In Terminal 1 (where you launched fermdocs-api):
env | grep -E "FERMDOCS_MEMORY|SYNAP"
# Should show:
#   FERMDOCS_MEMORY=synap
#   SYNAP_API_KEY=synap_...
# If empty, source the .env file BEFORE launching fermdocs-api:
set -a; source .env; set +a
fermdocs-api
```

### Memory writes show in Synap dashboard but reads come back empty

The `process_family` on the dossier is `None`. Check:

```bash
# Find the bundle for your run:
ls out/api/uploads/*/bundles/*/dossier.json

# Inspect:
parsevenv/bin/python -c "
import json, sys
d = json.load(open(sys.argv[1]))
print('process_family:', d['experiment']['process']['registered'].get('process_family'))
" path/to/dossier.json
```

If `process_family` is `None`, the upload didn't go through the dropdown
path. Re-upload with a specific family pick (not Auto-detect).

### `pytest tests/unit` shows 1+ failures

Some failures are pre-existing on certain feature branches (notably
`followup-context` and `caisc-2026-submission`). Check whether the
failure exists on `main`:

```bash
git stash
git checkout main
pytest tests/unit/path/to/failing_test.py -q
git checkout -
git stash pop
```

If the failure is on `main`, it's a known pre-existing issue, not your install.

### `docling` install fails

Docling has heavy native dependencies. On Apple Silicon, you may need:

```bash
arch -arm64 pip install docling
```

On Linux, ensure `libgomp1` is installed:

```bash
sudo apt install -y libgomp1
```

You can also skip PDF table extraction by not installing the `[pdf]` extra.
The system will fall back to text-only PDF parsing.

### `next/font` fails to fetch Google Fonts during `npm run build`

This is a sandbox/network restriction. Bypass it by setting:

```bash
# In apps/web/.env.local
NEXT_FONT_GOOGLE_MOCKED_RESPONSES=true
```

Or run on a network that allows outbound to fonts.googleapis.com.

---

## Next Steps

Once everything is running:

- Read [README.md](README.md) for what each part of the system does
- Read [ARCHITECTURE.md](ARCHITECTURE.md) for why the system is shaped this way
- Browse `plans/` for the design rationale behind specific features
  (memory layer, charts, anomaly detection, frontend redesign)
- Try uploading a real bundle and see how the hypothesis stage handles it

For development, the most useful entry points are:

```text
apps/api/fermdocs_api/main.py           API routes
apps/api/fermdocs_api/runner_pipeline.py runs ingest -> hypothesis end-to-end
src/fermdocs_hypothesis/runner.py        hypothesis state machine
src/fermdocs_memory/synap.py             memory adapter
apps/web/src/app/page.tsx                upload page + dropdown
apps/web/src/app/runs/[id]/page.tsx     run + hypothesis viewer
```
