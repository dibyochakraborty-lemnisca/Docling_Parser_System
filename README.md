# fermdocs

Fermentation experiment document parser. Ingests PDF/Excel/CSV reports, maps headers to a canonical golden-column schema via LLM, stores observations with full provenance in Postgres, and produces a versioned dossier JSON for downstream agents.

This README is a tour for someone who just cloned the repo. If you want the design rationale, read [`docs/architecture.md`](docs/architecture.md) after this.

---

## What it does, in one paragraph

You hand `fermdocs` an experiment ID and a bunch of files. It parses each file into tables, asks an LLM (Anthropic or Gemini) to map raw headers like `"Titer (g/L)"` to canonical golden columns like `final_titer_g_l`, normalizes units via pint, stores every value with full provenance back to the source cell, and produces a JSON dossier for the next agent in your pipeline. Anything that doesn't map cleanly lands in a JSONB residual blob, never lost. Re-ingesting the same file is a no-op (sha256 dedup). The LLM never invents numeric values; it only emits mappings and unit-normalization hints.

---

## Get it running locally

### Prerequisites

- Python 3.11+
- Postgres 15+ (Docker or native)
- An Anthropic API key, a Gemini API key, or both

### Setup

```bash
# Clone
git clone https://github.com/dibyochakraborty-lemnisca/Docling_Parser_System.git
cd Docling_Parser_System

# Virtualenv
python3.11 -m venv .venv
source .venv/bin/activate

# Core install
pip install -e ".[dev]"

# Optional extras
pip install -e ".[pdf]"      # Docling for PDF table extraction (~1-2GB ML deps)
pip install -e ".[gemini]"   # google-genai SDK
```

### Database

```bash
# Either: Docker
docker run -d --name fermdocs-pg \
  -e POSTGRES_USER=fermdocs -e POSTGRES_PASSWORD=fermdocs -e POSTGRES_DB=fermdocs \
  -p 5432:5432 postgres:16

# Or: local Postgres
createuser -s fermdocs && createdb -O fermdocs fermdocs
psql -d fermdocs -c "ALTER USER fermdocs WITH PASSWORD 'fermdocs';"
```

### Configure

```bash
cp .env.example .env
# Edit .env: at minimum set DATABASE_URL and one LLM API key.
set -a; source .env; set +a
```

### Migrate and verify

```bash
alembic upgrade head
pytest tests/unit -v   # 47 tests, no DB or LLM needed
```

If `pytest` is green, you're set up.

---

## First ingest, end to end

Try it offline first (no LLM cost, deterministic):

```bash
fermdocs ingest \
  --experiment-id EXP-001 \
  --files tests/fixtures/sample_run.csv \
  --fake-mapper \
  --out /tmp/EXP-001.dossier.json

cat /tmp/EXP-001.dossier.json | head -50
```

What happened:

1. The CSV got parsed into one `ParsedTable`.
2. `FakeHeaderMapper` matched headers like `"Strain"` -> `strain_id` by exact synonym lookup.
3. `UnitConverter` ran each numeric value through pint.
4. `Repository` wrote 16 observations into Postgres.
5. `build_dossier` projected those rows into a versioned JSON envelope.

Now with a real LLM:

```bash
fermdocs ingest \
  --experiment-id EXP-002 \
  --files tests/fixtures/sample_run.csv \
  --provider gemini \
  --out /tmp/EXP-002.dossier.json
```

Compare the dossiers. Confidence scores will differ; the mappings should mostly agree.

---

## How to read the codebase

Start in this order:

```
1. src/fermdocs/domain/models.py     -- the data shapes
2. src/fermdocs/pipeline.py          -- the full sequence in one file
3. src/fermdocs/dossier.py           -- the read path
4. src/fermdocs/cli.py               -- how it all wires up
5. docs/architecture.md              -- everything else
```

The one-paragraph layering rule: **`domain/` imports nothing from the project. Everything else imports from `domain/`. `storage/` is the only module that imports SQLAlchemy. `mapping/client.py` and `mapping/gemini_client.py` are the only ones that talk to LLMs.** This is enforced by code review and (eventually) a CI grep check.

---

## Project layout, annotated

```
Docling_Parse/
|
+- pyproject.toml         hatchling, deps split: core / dev / pdf / gemini
+- alembic.ini            points at migrations/
+- .env.example           every env var you might need
|
+- src/fermdocs/
|  |
|  +- domain/             the only module that has zero internal deps
|  |  +- models.py        Pydantic. ParsedTable, Observation, Dossier shapes.
|  |  +- golden_schema.py loads golden_schema.yaml; --schema CLI override hooks here
|  |
|  +- parsing/            file -> list[ParsedTable]
|  |  +- base.py          FileParser ABC. supports() + parse().
|  |  +- csv_parser.py    pandas. dtype=str, no NA inference.
|  |  +- excel_parser.py  pandas. one ParsedTable per sheet.
|  |  +- pdf_parser.py    Docling. Lazy-imports docling so the [pdf] extra is optional.
|  |  +- router.py        FormatRouter dispatches by extension.
|  |
|  +- mapping/            ParsedTable -> MappingResult
|  |  +- mapper.py        HeaderMapper Protocol; FakeHeaderMapper for offline tests.
|  |  +- prompt.py        the system + user prompts (provider-agnostic).
|  |  +- client.py        LLMHeaderMapper -- Anthropic Claude.
|  |  +- gemini_client.py GeminiHeaderMapper -- Google Gemini.
|  |  +- factory.py       build_mapper(provider) -- single seam for picking impl.
|  |  +- confidence.py    auto / needs_review / residual band thresholds.
|  |
|  +- units/              value + raw_unit -> canonical value
|  |  +- converter.py     UnitConverter. ConversionResult carries 'via' field.
|  |  +- registry.txt     pint custom unit defs (pH, OD600, fold_change, log_cfu_per_ml).
|  |  +- normalizer.py    Rule-based + LLM normalizers. Hints, not values.
|  |
|  +- storage/            Postgres I/O
|  |  +- models.py        SQLAlchemy 2.0. JSONB for value_raw / value_canonical / source_locator.
|  |  +- repository.py    Repository class. The only place that calls SQLAlchemy.
|  |
|  +- file_store/         original-file persistence
|  |  +- base.py          FileStore Protocol; sha256_of helper.
|  |  +- local.py         LocalFileStore: <root>/files/<sha256>.<ext>.
|  |
|  +- pipeline.py         IngestionPipeline. The ONLY file that knows the full flow.
|  +- dossier.py          build_dossier(experiment_id, repo) -> dict.
|  +- cli.py              Click. Wires real impls; tests wire fakes.
|  +- __init__.py         exports: ingest, build_dossier, IngestionResult.
|  +- __main__.py         python -m fermdocs entry.
|  +- schema/
|     +- golden_schema.yaml  the canonical golden-column definitions. Edit me.
|
+- migrations/
|  +- env.py
|  +- versions/
|     +- 0001_initial.py            DDL for all tables
|     +- 0002_backfill_via_field.py adds via='pint' to pre-normalizer rows
|
+- tests/
|  +- fixtures/                     test data, no real PDFs committed
|  +- conftest.py                   fixes sys.path so 'fermdocs' resolves
|  +- unit/                         no DB, no LLM, runs in <1s
|  +- integration/                  pipeline tests using fakes for repo + store
|
+- docs/
   +- architecture.md               full design doc -- read after this
```

---

## The five most common things you'll do

### 1. Add a new golden column

Edit `src/fermdocs/schema/golden_schema.yaml`:

```yaml
  - name: my_new_metric_g_l
    description: One-line description for the LLM.
    data_type: float
    canonical_unit: g/L
    synonyms: [my_metric, mm, MM]
    examples:
      - {raw_header: "My Metric (g/L)"}
      - {raw_header: "MM [g/L]"}
```

That's it. No code change. Re-run `fermdocs ingest`; the LLM mapper will pick up the new column on the next call.

**The single highest-leverage thing in the system is the schema YAML. Examples are worth more than long descriptions.** Add 2-3 real `raw_header` strings you've seen in the wild.

### 2. Add a new file format

Write a new parser in `src/fermdocs/parsing/`:

```python
from pathlib import Path
from fermdocs.domain.models import ParsedTable
from fermdocs.parsing.base import FileParser

class MyParser(FileParser):
    def supports(self, path: Path) -> bool:
        return path.suffix.lower() == ".myext"

    def parse(self, path: Path) -> list[ParsedTable]:
        ...  # produce ParsedTable list with format-specific locator
```

Register it in `cli.py:_build_pipeline`:

```python
router = FormatRouter([CsvParser(), ExcelParser(), DoclingPdfParser(), MyParser()])
```

Write a test in `tests/unit/test_my_parser.py` mirroring `test_csv_parser.py`. Done.

### 3. Add a new LLM provider (e.g., OpenAI)

Copy `src/fermdocs/mapping/gemini_client.py` to `openai_client.py`. Replace the SDK calls with OpenAI's structured-output API. The response shape and Pydantic validation stay identical.

Register in `src/fermdocs/mapping/factory.py`:

```python
if name == "openai":
    from fermdocs.mapping.openai_client import OpenAIHeaderMapper
    return OpenAIHeaderMapper()
```

Add `"openai"` to the `click.Choice` in `cli.py` and document the env vars in `.env.example`. Done.

### 4. Add a new unit alias

If pint can't parse a unit you keep seeing, edit `src/fermdocs/units/registry.txt`:

```
@alias gram_per_liter_per_hour = g/(L*h) = g_per_liter_per_hour
my_new_dimensionless = []
```

Reload (re-run the CLI) and pint will accept the new strings.

If the unit is **structurally weird** (Unicode superscripts, embedded annotations), look at `src/fermdocs/units/normalizer.py:RuleBasedNormalizer` instead. The rule-based normalizer transforms unit strings into pint-parseable form before pint sees them.

### 5. Debug what happened on a specific ingest

Everything is queryable. Open a psql shell:

```sql
-- Which observations exist for an experiment?
SELECT column_name, raw_header,
       value_canonical->>'value' AS canonical,
       value_canonical->>'via' AS via,
       conversion_status,
       mapping_confidence
FROM golden_observations
WHERE experiment_id = 'EXP-001'
  AND superseded_by IS NULL
ORDER BY column_name, raw_header;

-- What fell to residual?
SELECT payload->'tables_partial' AS partial,
       payload->'tables_unmapped' AS unmapped
FROM residual_data
WHERE experiment_id = 'EXP-001';

-- What's awaiting human review?
SELECT column_name, raw_header, mapping_confidence,
       value_canonical->'normalization' AS norm
FROM golden_observations
WHERE needs_review AND superseded_by IS NULL
ORDER BY mapping_confidence ASC
LIMIT 20;
```

The CLI also surfaces the review queue:

```bash
fermdocs review --next
```

---

## How testing works

```
tests/unit/                 fast, deterministic, offline. Run on every save.
tests/integration/          pipeline tests using fakes for repo + store.
                            Still no DB, no LLM. Run before commit.
```

Key patterns:

- **`FakeHeaderMapper`** lets you exercise the whole pipeline without an API key. It matches by exact synonym; tests use small canonical schemas so matches are predictable.
- **`_FakeRepo` / `_FakeFileStore`** in `tests/integration/test_normalizer_pipeline.py` show the dependency-injection pattern. Real `IngestionPipeline` + fake collaborators = full flow tested with no infra.
- **Pydantic-validated LLM responses** mean tests can mock the LLM client by returning canned dicts; the rest of the code doesn't know the difference.

Run:

```bash
pytest tests/unit -v          # ~1s, runs anywhere
pytest tests -v               # full suite, still no DB or LLM
```

Add a test in the matching `tests/unit/test_<module>.py` when you change anything. The bar is 100% coverage for new code paths; it's cheap with AI-assisted writing.

---

## Configuration cheat sheet

All behavior is environment-configurable. The full list is in `.env.example`; the high-leverage ones:

```bash
DATABASE_URL=postgresql+psycopg://fermdocs:fermdocs@localhost:5432/fermdocs

# Mapper provider
FERMDOCS_MAPPER_PROVIDER=anthropic      # or 'gemini' or 'fake'
ANTHROPIC_API_KEY=sk-ant-...            # if using Anthropic
GEMINI_API_KEY=AI...                    # if using Gemini
FERMDOCS_MAPPER_MODEL=claude-haiku-4-5-20251001
FERMDOCS_GEMINI_MODEL=gemini-2.5-flash

# Storage
FERMDOCS_DATA_DIR=./data                # where original files land

# Optional schema override
FERMDOCS_SCHEMA_PATH=./my_custom_schema.yaml

# Unit normalizer
FERMDOCS_USE_LLM_NORMALIZER=false       # rule-based always on; LLM is opt-in
FERMDOCS_NORMALIZER_PROVIDER=anthropic  # falls back to MAPPER_PROVIDER

# PDF
FERMDOCS_PDF_OCR=false                  # only enable for image-based PDFs
```

---

## CLI reference

### Ingest

```bash
fermdocs ingest \
  --experiment-id EXP-001 \
  --files path/to/a.pdf path/to/b.csv \
  [--out /tmp/dossier.json] \
  [--schema /path/to/custom.yaml] \
  [--provider {anthropic|gemini|fake}] \
  [--fake-mapper] \
  [--llm-normalizer | --no-llm-normalizer]
```

Exit codes: `0` ok, `1` usage, `2` input, `3` parse, `4` db, `5` partial.

### Build a dossier from already-ingested data

```bash
fermdocs dossier --experiment-id EXP-001 --out dossier.json
```

This is a pure read of Postgres; safe to run any time.

### Review queue

```bash
fermdocs review --next   # one observation at a time, sorted by oldest unreviewed
```

### Inspect contracts

```bash
fermdocs --print-schema dossier   # full dossier JSON Schema
fermdocs --print-schema golden    # GoldenSchema YAML structure
fermdocs --print-schema mapper    # mapper response schema (JSON Schema)
```

Use these to validate downstream consumers and to lock contracts for a future Rust port.

---

## Common gotchas

- **`DATABASE_URL not set`** -- you ran the CLI without `set -a; source .env; set +a` first. Or you don't have a `.env` yet.
- **`role "fermdocs" does not exist`** -- you have Postgres but no `fermdocs` user. See the Database setup section above.
- **PDF ingest seems stuck** -- Docling is loading ~1GB of ML weights on first run. Subsequent runs are cached. The progress bar is below where you'd expect.
- **`ImportError: docling`** -- you didn't install the `[pdf]` extra. `pip install -e ".[pdf]"`.
- **All values come out wrong by 1000x** -- check `unit_canonical` vs `canonical_unit` in `golden_schema.yaml`. The mapper extracted `"mg/mL"` but your schema declared `"g/L"`; pint is doing its best but maybe the source unit string was misread. Inspect via `fermdocs review --next`.
- **Same experiment_id has wildly contradicting values** -- expected if multiple files report the same column with disagreeing numbers. Philosophy B: we preserve, never resolve. The next agent reasons about it.
- **`DimensionalityError: Cannot convert from ...`** -- the LLM mapped a column to a golden field whose canonical unit doesn't match the raw unit's dimension. Likely a schema design issue; either add a per-product golden column or accept that this header should land in residual.

---

## What's intentionally NOT in v1

- Narrative / free-text extraction (only structured tables today)
- Figure / chart data extraction (figures stored as references only)
- Multi-product breakdown columns (one product per experiment_id assumed; see `docs/architecture.md` for the workaround)
- Self-learning unit hints (one-shot LLM normalization without persistence)
- A review UI (CLI-only review queue)
- Cloud file storage (LocalFileStore only; FileStore Protocol is the swap point)
- Service mode / FastAPI wrapper (CLI + library only; pattern A or B integration)
- Authentication, multi-tenancy, RBAC

None of these are forced moves; the architecture is set up so each can be added without disturbing the rest.

---

## Where to learn more

- [`docs/architecture.md`](docs/architecture.md) -- full design, invariants, data model, failure modes, rollback runbook.
- The Pydantic models in `src/fermdocs/domain/models.py` -- read these and you understand 80% of the system.
- `src/fermdocs/pipeline.py` -- one file, end to end.
- The integration test in `tests/integration/test_normalizer_pipeline.py` -- shows the full flow with fakes.

If something in the codebase surprises you, that's a documentation bug. Open an issue or update the relevant doc as part of your PR.
