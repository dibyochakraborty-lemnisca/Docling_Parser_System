# fermdocs

Fermentation experiment document parser. Ingests PDF/Excel/CSV reports, maps headers to a canonical golden-column schema via LLM, stores observations with full provenance, and produces a versioned dossier JSON for downstream agents.

## Install

```bash
pip install -e ".[dev]"
# Optional PDF support (Docling pulls heavy ML deps):
pip install -e ".[pdf]"
```

## Configure

```bash
cp .env.example .env
# Edit .env: set ANTHROPIC_API_KEY and DATABASE_URL
```

## Run migrations

```bash
alembic upgrade head
```

## Ingest

```bash
fermdocs ingest \
  --experiment-id EXP-ASX-001 \
  --files path/to/run1.csv path/to/setup.pdf \
  --out /tmp/EXP-ASX-001.dossier.json
```

Exit codes: 0 ok, 1 usage, 2 input, 3 parse, 4 db, 5 partial.

## Build a dossier from already-ingested data

```bash
fermdocs dossier --experiment-id EXP-ASX-001 --out dossier.json
```

## Review queue

```bash
fermdocs review --next
```

## Print contracts

```bash
fermdocs --print-schema dossier
fermdocs --print-schema golden
```

See `docs/architecture.md` for the design.
