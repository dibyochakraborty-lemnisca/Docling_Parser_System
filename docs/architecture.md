# fermdocs architecture

Last updated: matches `0.1.0`. Source of truth for design intent. Update when you change a public contract.

---

## 1. What this is

`fermdocs` ingests fermentation experiment reports (PDF, Excel, CSV) and produces a versioned, agent-consumable dossier JSON for downstream LLM workflows. It maps raw column headers to a canonical golden-column schema using an LLM (Anthropic or Gemini), preserves full provenance for every value, normalizes units via pint with optional LLM fallback, and stores everything in Postgres with a re-extraction-friendly schema.

**The non-negotiables:**

- Every observation carries provenance back to a specific cell in a specific file.
- LLMs never emit numeric values, only header-to-column mappings and unit-normalization hints. Code does the math.
- Conflicting observations are preserved, never resolved away (Philosophy B).
- The dossier JSON is a stable, versioned contract for downstream agents.
- Source files are content-addressed by sha256 so re-ingest is idempotent.

---

## 2. Pipeline

```
        +-----------+
        |  caller   |  (CLI / LangGraph node / direct library import)
        +-----+-----+
              |
              | files: list[Path], experiment_id: str
              v
     +--------+---------+
     |  IngestionPipeline
     |  (pipeline.py)   |  the only file with full-sequence knowledge
     +--------+---------+
              |
              v
  +-----------+-----------+
  |   FileStore.put       |  hashes file, copies to <root>/files/<sha256>.<ext>
  |   (file_store/local)  |
  +-----------+-----------+
              |
              v
  +-----------+-----------+
  |   Repository.find_or_create_file
  |   (storage/repository) |  idempotent on (experiment_id, sha256)
  +-----------+-----------+
              |
     created? | yes
              v
  +-----------+-----------+
  |   FormatRouter.parse  |  PDF -> DoclingPdfParser
  |   (parsing/router)    |  XLSX -> ExcelParser (pandas)
  |                       |  CSV  -> CsvParser   (pandas)
  +-----------+-----------+
              |
              | list[ParsedTable]
              v
  +-----------+-----------+
  |   HeaderMapper.map    |  one batched LLM call per file
  |   (mapping/*)         |  Fake | Anthropic | Gemini
  +-----------+-----------+
              |
              | MappingResult: per-table list of MappingEntry
              v
  +-----------+-----------+
  |   for each cell:      |
  |   UnitConverter.convert(value, unit_raw, canonical_unit, normalizer)
  |   (units/converter)   |
  +-----------+-----------+
              |
              | pint succeeds -> via='pint'
              | pint fails + normalizer set:
              |    rule-based / llm normalizer
              |    -> apply_hint -> via='rule_based' or 'llm'
              v
  +-----------+-----------+
  |   Repository.write_observations
  |   Repository.write_residual
  +-----------+-----------+
              |
              v
        Postgres tables
        (experiments, source_files, golden_observations, residual_data)
              |
              v
  +-----------+-----------+
  |   build_dossier(experiment_id, repo)
  |   (dossier.py)        |  read-only projection -> versioned JSON
  +-----------+-----------+
              |
              v
        dossier JSON 1.0
        (consumed by next agent)
```

---

## 3. Module map and responsibilities

```
src/fermdocs/
  domain/                pure data + Pydantic models. NEVER imports from elsewhere
                         in the project. Everything else imports from here.
    models.py            ParsedTable, MappingEntry, MappingResult, Observation,
                         IngestionResult, GoldenSchema, GoldenColumn,
                         ConversionStatus, ObservationType, ConfidenceBand,
                         ResidualPayload
    golden_schema.py     loads + validates golden_schema.yaml; respects
                         FERMDOCS_SCHEMA_PATH env var and CLI override

  parsing/               file -> list[ParsedTable]. Only place that reads files.
    base.py              FileParser ABC
    csv_parser.py        CSV / TSV via pandas (dtype=str, no NA inference)
    excel_parser.py      .xlsx / .xls via pandas (one ParsedTable per sheet)
    pdf_parser.py        Docling-backed; lazy imports docling so the [pdf]
                         extra is optional. OCR off by default; toggle with
                         FERMDOCS_PDF_OCR=true.
    router.py            FormatRouter dispatches by extension

  mapping/               ParsedTable -> MappingResult. Only place that calls
                         LLM SDKs (anthropic, google.genai).
    mapper.py            HeaderMapper Protocol; FakeHeaderMapper for offline
                         tests (synonym-based deterministic match)
    prompt.py            shared prompt rendering (system + user)
    client.py            LLMHeaderMapper (Anthropic, structured tool-use)
    gemini_client.py     GeminiHeaderMapper (response_schema)
    confidence.py        bands (auto >=0.85, needs_review 0.6-0.85,
                         residual <0.6); thresholds are constants here
    factory.py           build_mapper(provider, use_fake) -- single seam
                         for picking impl. Adding a provider = new client
                         file + 2 lines here.

  units/                 value + raw_unit -> canonical value. Only place that
                         imports pint.
    converter.py         UnitConverter, ConversionResult (incl. via, hint),
                         apply_hint() for normalizer outputs
    registry.txt         pint custom unit definitions: pH, OD600,
                         fold_change, log_cfu_per_ml (all dimensionless)
    normalizer.py        UnitNormalizer Protocol, NormalizationHint,
                         RuleBasedNormalizer (regex transforms),
                         LLMUnitNormalizer (Anthropic/Gemini, opt-in,
                         per-run cache, validated via Pydantic),
                         ChainNormalizer

  storage/               Postgres I/O. Only place that imports SQLAlchemy.
    models.py            DeclarativeBase + ExperimentRow, SourceFileRow,
                         ObservationRow, ResidualRow. Indexes:
                         (experiment_id, column_name);
                         needs_review WHERE needs_review;
                         experiment_id WHERE superseded_by IS NULL
    repository.py        Repository class; FileRecord dataclass.
                         Methods: upsert_experiment, find_or_create_file,
                         mark_file_parsed, write_observations,
                         write_residual, fetch_*, next_review_observation,
                         row_to_observation

  file_store/            original-file persistence. Pluggable.
    base.py              FileStore Protocol, StoredFile dataclass,
                         sha256_of helper
    local.py             LocalFileStore: <root>/files/<sha256>.<ext>;
                         re-storing same bytes is a no-op

  pipeline.py            IngestionPipeline. The ONLY file that knows the full
                         sequence. New cross-cutting concerns belong here.
  dossier.py             build_dossier(experiment_id, repo) -> dict.
                         Read-only projection of storage; no side effects.
  cli.py                 Click-based CLI. Wires real impls; tests wire fakes.
  __init__.py            exports ingest, build_dossier, IngestionResult
  __main__.py            python -m fermdocs entry point
  schema/
    golden_schema.yaml   the editable schema. Version-controlled.

migrations/              alembic. Hand-written SQL; one migration per logical
                         change. Not auto-generated from ORM (so the DDL
                         survives the future Rust port).
  versions/
    0001_initial.py      experiments, source_files, golden_observations,
                         residual_data, indexes
    0002_backfill_via_field.py   adds via='pint' to existing observations

tests/
  fixtures/              committed test data
  unit/                  fast, no DB, no LLM
  integration/           pipeline-level tests using fakes for repo + store
```

---

## 4. Architectural invariants (what CI should enforce)

These hold today. If a future change breaks them, you have a regression.

### 4.1 Layer purity

- `domain/` imports from `pydantic`, `enum`, stdlib only. **Never** imports anything in `src/fermdocs/`.
  - Verify: `grep -r '^from fermdocs' src/fermdocs/domain/` returns nothing.
- `storage/` is the only module that imports `sqlalchemy`.
  - Verify: `grep -rn 'from sqlalchemy\|import sqlalchemy' src/fermdocs/ | grep -v storage/`
- `units/converter.py` is the only module that imports `pint`.
- `mapping/client.py` is the only module that imports `anthropic`.
- `mapping/gemini_client.py` and `units/normalizer.py` are the only modules that import `google.genai`.

### 4.2 LLM contract

Three LLM-using components, three different trust profiles. Read this carefully -- the safest one (mapper) and the riskiest one (narrative extractor) coexist in the codebase.

- **Header mapper LLM:** **never** emits values. Receives headers + sample rows for context; emits only header-to-column mappings, raw_unit strings, confidence, and rationale. Code applies the mapping to extract values from the parsed table.
- **Unit normalizer LLM:** **never** emits converted values. Output is restricted to one of three actions (`use_pint_expr | dimensionless | unconvertible`) with an optional pint expression string, rationale, and confidence. Code does the math via pint. The `use_factor` action was deliberately excluded so every successful path goes through pint's dimensionality check.
- **Narrative extractor LLM:** **does** emit numeric values. This is the only LLM in the system that produces numbers directly. The relaxation is deliberate -- it is the only way to extract values from prose that has no tabular structure -- and is paired with seven compensating safety mechanisms (see Section 8).

LLM responses go through `pydantic.model_validate` before hitting any business logic. Malformed responses degrade to safe defaults (table -> residual; conversion -> failed; narrative extraction -> empty list), never crash the pipeline.

### 4.3 Provenance

- Every `Observation` row carries `experiment_id`, `file_id`, `column_name`, `raw_header`, and `source_locator`. None are nullable.
- `source_locator` is a JSONB document. Format-agnostic shape:
  - PDF:  `{"format": "pdf", "file": "x.pdf", "page": N, "table_idx": N, "row": N, "col": N}`
  - Excel: `{"format": "xlsx", "file": "x.xlsx", "sheet": "Setup", "row": N, "col": N}`
  - CSV:  `{"format": "csv", "file": "x.csv", "row": N, "col": N}`

### 4.4 Idempotent re-ingest

- `source_files` has `UNIQUE (experiment_id, sha256)`. Same bytes uploaded twice for the same experiment = no-op.
- File contents at `<DATA_DIR>/files/<sha256>.<ext>`. `LocalFileStore.put` skips the copy if the path exists.
- `extractor_version` (`v0.1.0` today) is stamped on every observation and residual row. Re-extraction (after pipeline changes) creates new rows; old rows get `superseded_by` set.
- `fetch_active_observations` filters `superseded_by IS NULL`. Same for residuals. The dossier reads only active rows.

### 4.5 Polymorphic value storage

- `value_raw` and `value_canonical` are JSONB columns shaped:
  ```json
  {"value": <any>, "type": "float|int|text|bool", "via": "pint|rule_based|llm|not_applicable"}
  ```
  with optional `"normalization": {action, pint_expr, rationale, confidence, source}` when a normalizer fired.
- `via` is **always populated** post-migration 0002. Code that reads it should never expect NULL except where `value_canonical` itself is NULL (text columns that don't convert).

### 4.6 Dossier contract versioning

- Every dossier carries `"dossier_schema_version": "1.0"` at the root.
- Consumers branch on this field. Bump the version when you change the shape; never silently mutate the existing version.
- `Observation.to_dossier_observation()` is the single source of truth for the per-observation JSON shape. Never serialize observations any other way.

### 4.7 Confidence semantics

- LLM-emitted confidence scores are **ranks, not calibrated probabilities**. Use them to sort review queues; don't display them to users as percentages without calibrating against ground truth.
- `extraction_confidence` is currently 1.0 for clean numeric/text cells, 0.5 for cells that fall back to string after coerce failure. Bumped to reflect normalizer involvement when one fired.
- `combined_confidence` = `mapping_confidence * extraction_confidence`, computed at dossier-build time.

---

## 5. Data model

DDL is in `migrations/versions/0001_initial.py`. Conceptual view:

### 5.1 `experiments`

Caller-supplied identity. The pipeline upserts (no-op on conflict) and trusts the caller's `experiment_id`.

```
experiment_id  TEXT  PRIMARY KEY    -- caller-supplied; not extracted from documents
name           TEXT  NULL
uploaded_by    TEXT  NULL
created_at     TIMESTAMPTZ  default now()
status         TEXT  default 'ingesting'   -- 'ingesting' | 'ready' | 'failed'
notes          TEXT  NULL
```

### 5.2 `source_files`

One row per ingested file. Sha256 dedup is the idempotency primitive.

```
file_id        UUID  PRIMARY KEY
experiment_id  TEXT  REFERENCES experiments
filename       TEXT  NOT NULL
sha256         TEXT  NOT NULL              -- content hash
mime_type      TEXT  NULL
size_bytes     BIGINT NULL
page_count     INT   NULL                  -- PDF only
storage_path   TEXT  NOT NULL              -- where the original lives
parsed_at      TIMESTAMPTZ  NULL
parse_status   TEXT  default 'pending'     -- 'pending' | 'ok' | 'failed'
parse_error    TEXT  NULL

UNIQUE (experiment_id, sha256)
```

### 5.3 `golden_observations`

The heart. Every value with full provenance. Polymorphic via JSONB envelope.

```
observation_id        UUID  PRIMARY KEY
experiment_id         TEXT  REFERENCES experiments
file_id               UUID  REFERENCES source_files
column_name           TEXT  NOT NULL                -- canonical golden field name
raw_header            TEXT  NOT NULL                -- what the source document said
observation_type      TEXT  default 'unknown'      -- planned|measured|reported|derived|unknown

value_raw             JSONB NOT NULL                -- {"value": ..., "type": ...}
unit_raw              TEXT  NULL
value_canonical       JSONB NULL                   -- {"value": ..., "type": ..., "via": ..., "normalization": ...}
unit_canonical        TEXT  NULL
conversion_status     TEXT  default 'not_applicable' -- 'ok' | 'failed' | 'not_applicable'

source_locator        JSONB NOT NULL                -- format-agnostic location

mapping_confidence    NUMERIC NULL                  -- LLM-self-reported, 0..1
extraction_confidence NUMERIC NULL                  -- 1.0 clean / 0.5 messy
needs_review          BOOLEAN default false         -- triggered by 0.60-0.85 mapping confidence

extractor_version     TEXT  NOT NULL                -- 'v0.1.0' today
superseded_by         UUID  NULL REFERENCES golden_observations

extracted_at          TIMESTAMPTZ  default now()

-- indexes
INDEX (experiment_id, column_name)
INDEX (needs_review) WHERE needs_review
INDEX (experiment_id) WHERE superseded_by IS NULL  -- 'active observations' fast path
```

### 5.4 `residual_data`

Per-file JSONB bucket for everything that didn't map cleanly. Includes:

- Tables that fully failed (`tables_unmapped`)
- Tables where some columns mapped, some didn't (`tables_partial`)
- Free narrative text (currently always empty; v1 doesn't extract narrative)
- Figures (currently always empty; v1 doesn't extract figures)

```
residual_id        UUID  PRIMARY KEY
file_id            UUID  REFERENCES source_files
experiment_id      TEXT  REFERENCES experiments
extractor_version  TEXT  NOT NULL
payload            JSONB NOT NULL
superseded_by      UUID  NULL REFERENCES residual_data
created_at         TIMESTAMPTZ default now()

INDEX (file_id)
```

---

## 6. Mapping subsystem

### 6.1 Why a Protocol, not a class hierarchy

Protocol-based polymorphism (PEP 544) gives us swap-friendliness without the inheritance ceremony. Any object with a `.map(tables, schema) -> MappingResult` method is a `HeaderMapper`. New providers add a class; nothing else changes.

```
HeaderMapper Protocol
    .map(tables: list[ParsedTable], schema: GoldenSchema) -> MappingResult

Implementations:
  FakeHeaderMapper       -- exact match on synonyms; for tests and offline runs
  LLMHeaderMapper        -- Anthropic Claude, structured tool-use
  GeminiHeaderMapper     -- Google Gemini, response_schema
```

### 6.2 Batched, not per-table

`map()` accepts a list of tables and returns mappings for all of them. A 30-table PDF triggers **one** LLM call, not 30. This was a deliberate review-time decision to keep cost predictable.

### 6.3 Provider factory

`build_mapper(provider, use_fake)` resolves provider in this order:

1. Explicit arg (`--provider anthropic` on the CLI).
2. `FERMDOCS_MAPPER_PROVIDER` env var.
3. Default: `'gemini'` (cheap fast Flash model handles header mapping well; Anthropic stays available for users who prefer it).

`use_fake=True` short-circuits to `FakeHeaderMapper` regardless. The CLI uses this when `--fake-mapper` is set.

### 6.4 Confidence bands and the residual fallthrough

```
confidence >= 0.85    -> auto-accept, lands in golden_observations
0.60 <= conf < 0.85   -> stored with needs_review=true (still in golden_observations)
confidence < 0.60     -> mapping is dropped; the column falls into residual JSONB
```

The `needs_review` queue is surfaced via `fermdocs review --next`. Rule-of-thumb interpretation:

- Auto: trust it, ingest forward.
- Needs review: trust the data, but a human should look later.
- Residual: don't trust the LLM's mapping; preserve the raw column for the next agent to reason about.

Thresholds live in `mapping/confidence.py` as constants. Tune them once after calibration against real ground truth.

---

## 7. Unit conversion subsystem

### 7.1 Two-tier flow

```
UnitConverter.convert(value, unit_raw, canonical_unit, normalizer=None)
    |
    v
_convert_with_pint     -- handles canonical=None, value=None, parse, conversion
    |
    +-- success -> ConversionResult(via='pint')
    |
    +-- failure --(if normalizer set)--> normalizer.normalize(...)
                       |
                       v
                NormalizationHint(action, pint_expr?, rationale, confidence, source)
                       |
                       v
                  apply_hint(value, ..., hint):
                      use_pint_expr  -> retry pint with hint.pint_expr
                                        on success: via=hint.source
                      dimensionless  -> store value as-is, status=ok, via=hint.source
                      unconvertible  -> status=failed, via=hint.source, error=rationale
```

### 7.2 Why three actions, not four

An earlier design had `use_factor` (LLM emits a scalar, code multiplies). It was dropped because:

- Pint's dimensionality check is the strongest correctness guarantee in the system.
- A scalar factor bypasses it. The LLM can confidently say "factor 0.001" and turn a g/g (mass fraction) value into a g/L (volumetric concentration) value with no signal that anything went wrong.
- Every successful path now goes through pint. If pint refuses, the result is failure-with-rationale, not silent corruption.

If a future workload genuinely needs scalar factors, add them with a separate dimensional-sanity check at apply time.

### 7.3 Why two normalizers, not one

- `RuleBasedNormalizer` is **always on**. Deterministic regex transforms (Unicode superscripts, "of pellet" annotations, known dimensionless tokens). Zero LLM cost. Catches the predictable 70-80% of failures.
- `LLMUnitNormalizer` is **on by default** (`FERMDOCS_USE_LLM_NORMALIZER=true`); pass `--no-llm-normalizer` to disable. For the unpredictable rest. In-process cache prevents repeat calls on the same `(unit_raw, canonical_unit)` within a run. If the API key is missing or the call fails, it degrades to `UNCONVERTIBLE` -- the pipeline never crashes.

`ChainNormalizer` runs them in order: rule-based first, LLM only if rule-based returned `unconvertible`.

### 7.4 The pint custom registry

`units/registry.txt` declares fermentation-specific dimensionless units that pint doesn't ship with:

```
pH = [pH_unit]
OD600 = [OD_unit]
fold_change = [fold_unit]
log_cfu_per_ml = [log_cfu_unit]
```

These are loaded at converter construction. Add to this file when a new dimensionless unit shows up; expand sparingly (the file ships with the wheel).

### 7.5 LLM normalizer error handling

- API failure (network, auth, rate limit) -> `UNCONVERTIBLE` with `rationale="llm_error: <message>"`. Pipeline continues.
- Malformed JSON -> Pydantic validation fails -> caught -> `UNCONVERTIBLE`. Pipeline continues.
- Cached failures persist for the run. Next ingest tries fresh.

---

## 8. Narrative extraction (Tier 1 + Tier 2)

PDFs contain prose. Tables alone don't capture every value: methods sections describe vessel setup, observation paragraphs report measurements, captions describe figures. The narrative subsystem handles all of this.

### 8.1 Tier 1 -- always-on capture

The PDF parser walks `document.texts` (paragraphs, headings, list items, captions) in addition to `document.tables`. Every block >= 20 chars is captured into `ResidualPayload.narrative` and persisted in the `residual_data` JSONB column. This is unconditional -- no LLM cost, no opt-in.

This fixes a silent data-loss bug from the table-only era: prose that contained meaningful information used to be discarded by the parser before reaching the residual.

### 8.2 Tier 2 -- on-by-default LLM extraction

`LLMNarrativeExtractor` (default `gemini-3-pro`, override via `FERMDOCS_NARRATIVE_MODEL`) reads narrative blocks and emits structured candidates:

```
NarrativeExtraction = {
  column,                  -- must exist in golden_schema
  value,                   -- numeric or string
  unit,                    -- raw unit string from the prose
  evidence,                -- VERBATIM substring of source text
  source_paragraph_idx,    -- which block this came from
  confidence,              -- 0..1 self-reported
  rationale
}
```

Disable by setting `FERMDOCS_EXTRACT_NARRATIVE=false` or `--no-extract-narrative`. Tier 1 capture is unaffected.

### 8.3 The seven safety mechanisms

The narrative extractor is the only LLM that emits values directly. Each candidate extraction passes through these gates before becoming an `Observation`:

1. **Schema validation.** `column` must be a name in `golden_schema.yaml`. The LLM cannot invent column names.
2. **Source-block resolution.** `source_paragraph_idx` must exist in the chunk currently being processed. Cross-chunk references are rejected.
3. **Evidence substring.** `evidence` must be a verbatim substring of the source paragraph text.
4. **Value-string-in-evidence.** The numeric value, in any of its rendered forms (`14.2`, `14.20`, `14`), must appear inside the evidence span. This catches "yields ranged from 11.7 to 14.2" -> wrong-pick attacks.
5. **Sentence bound.** Evidence must be <= 200 chars and contain <= 2 sentence terminators (`.!?`). Catches paragraph-spanning quotes that lose context.
6. **Dedup against table observations.** If `(column_name, value_canonical)` already exists with `source_locator.section == 'table'` for this file, the prose extraction is dropped. Prevents *phantom corroboration* -- prose that summarizes a table value should not produce a second observation that looks like an independent confirmation.
7. **Confidence cap + needs_review forced.** Narrative observations have their `mapping_confidence` capped at `0.85` and always have `needs_review=true`. They never auto-accept; they always surface in `fermdocs review --next`.

Additionally:

- `observation_type` is **fixed** to `"reported"` for narrative observations. The LLM does not classify intent (planned vs measured vs reported). This was a v1 simplification; if downstream agents need finer signal, add it later.
- Per-call paragraph cap of `MAX_PARAGRAPHS_PER_CALL = 20`. Larger files are chunked. Cross-chunk associations are intentionally lost (prevents the LLM from incorrectly linking facts across distant pages).
- API failure (network, auth, malformed JSON) -> empty list. Pipeline continues. Tier 1 residual capture is unaffected.

### 8.4 Provenance for narrative observations

```json
"source_locator": {
  "format": "pdf",
  "file": "report.pdf",
  "page": 3,
  "section": "narrative",
  "paragraph_idx": 7,
  "evidence_quote": "grown at 30°C"
},
"value_canonical": {
  "value": 30.0,
  "type": "float",
  "via": "pint",
  "extracted_via": "narrative_llm"
}
```

`source_locator.section` is the discriminator: `"table"` vs `"narrative"`. Migration 0003 backfills `"table"` on pre-Tier-2 rows. Downstream consumers can filter by section.

`evidence_quote` is content-addressable provenance -- if Docling version drift later changes `paragraph_idx` indexing, the literal evidence string still pins where the value came from.

### 8.5 Observed behavior on real PDFs

On the Astaxanthin PDF in `tests/fixtures/`:

- Tier 1: 8 narrative blocks captured (previously dropped silently).
- Tier 2: extractions depend on whether the prose contains values mappable to the current schema. With a fermentation-process-focused schema and an analytical-chemistry PDF, most narrative content is intentionally ignored (no matching golden columns). With a richer schema covering analytical metrics, expect 5-15 narrative observations per typical CRO report.

## 9. The dossier

`build_dossier(experiment_id, repository)` reads from Postgres and returns a versioned dict. It's a pure projection; no side effects. Shape:

```
{
  "dossier_schema_version": "1.0",
  "experiment": {
    "experiment_id", "name", "uploaded_by", "created_at", "status",
    "source_files": [...]
  },
  "golden_columns": {
    "<column_name>": {
      "canonical_unit": "...",
      "observations": [<observation>, ...]   // multiple if multiple files reported it
    }
  },
  "residual": {
    "summary": {residual_records, tables_unmapped, tables_partial},
    "records": [<full residual payload>, ...]
  },
  "ingestion_summary": {
    "total_observations", "high_confidence", "needs_review",
    "fell_to_residual", "golden_coverage_percent",
    "files_failed_to_parse", "schema_version"
  }
}
```

Each `observation` (from `Observation.to_dossier_observation()`):

```
{
  "observation_id", "value", "unit",
  "value_raw", "unit_raw",
  "observation_type",
  "confidence": {"mapping", "extraction", "combined"},
  "needs_review",
  "source": {"file_id", "raw_header", "locator"},
  "conversion_status",
  "extractor_version",
  "via",                                  // 'pint' | 'rule_based' | 'llm' | None for legacy rows
  "normalization"                         // full hint when normalizer fired, else None
}
```

**Multiple observations per column is intentional.** When two files in the same experiment both report `biomass_g_l`, the dossier shows both, with their provenance. The next agent decides what to do (e.g., compare planned vs measured, flag contradictions). No resolver collapses them.

---

## 10. CLI contracts (Rust-port-friendly)

The CLI surface is part of the durable contract. A future Rust binary should accept the same flags and exit codes.

### 9.1 Subcommands

```
fermdocs ingest --experiment-id <str> --files <path>... [--out <path>]
                [--schema <path>] [--provider {gemini,anthropic,fake}]
                [--fake-mapper] [--llm-normalizer/--no-llm-normalizer]
                [--extract-narrative/--no-extract-narrative]

fermdocs dossier --experiment-id <str> [--out <path>]

fermdocs review --next

fermdocs --print-schema {dossier|golden|mapper}
```

### 9.2 Exit codes

```
0  ok
1  usage error (bad flags)
2  input error (missing file, unreadable)
3  parse error (parser crashed on a file)
4  db error
5  partial success (some files ok, some failed)
```

### 9.3 Environment variables

```
DATABASE_URL                    postgres connection string (required)
GEMINI_API_KEY                  required if FERMDOCS_MAPPER_PROVIDER=gemini (default)
ANTHROPIC_API_KEY               required if FERMDOCS_MAPPER_PROVIDER=anthropic
FERMDOCS_MAPPER_PROVIDER        gemini | anthropic | fake (default: gemini)
FERMDOCS_GEMINI_MODEL           override Gemini model id
FERMDOCS_MAPPER_MODEL           override Anthropic mapper model id
FERMDOCS_DATA_DIR               where source files are stored (default: ./data)
FERMDOCS_SCHEMA_PATH            override the bundled golden_schema.yaml
FERMDOCS_USE_LLM_NORMALIZER     true/false (default: false)
FERMDOCS_NORMALIZER_PROVIDER    falls back to FERMDOCS_MAPPER_PROVIDER
FERMDOCS_PDF_OCR                true/false; off by default for digital PDFs
```

---

## 11. Failure modes and mitigations

| Failure | Detection | Mitigation in code |
|---|---|---|
| LLM mapper returns malformed JSON | Pydantic validation | Caught -> table moves to residual, file marked ok |
| LLM mapper times out / network down | Exception | Caught -> file marked `parse_status=failed`, exit code 5 |
| Pint can't parse a unit | UndefinedUnitError | If normalizer set: try normalizer; else status=failed, value_canonical=NULL, observation still stored |
| Pint dimensionality mismatch | DimensionalityError | Same as above |
| Same `experiment_id` reused for unrelated experiments | None (caller's bug) | Visible in dossier as multiple observations for the same column with disagreeing values |
| Docling crashes on malformed PDF | Exception in pipeline | Caught -> file marked `parse_status=failed`, error stored, exit code 5 |
| Schema YAML malformed | Pydantic validation at load time | Fail fast at startup, exit 1 |
| Database connection lost mid-run | SQLAlchemy raises | Pipeline crashes; partial state in DB. Caller retries. (Re-ingest is idempotent.) |
| LLM normalizer returns invalid action | Pydantic enum check | Caught -> hint becomes UNCONVERTIBLE -> conversion fails cleanly |
| Narrative extractor LLM fails (network, auth, JSON) | Caught | Returns empty list; Tier 1 residual capture is unaffected |
| Narrative extractor returns hallucinated evidence | Substring + value-string + sentence-bound check | Per-extraction reject; other extractions in the same call kept; rejection counted in `IngestionFileResult.narrative_extractions_rejected` |
| Narrative extractor invents a column name | Schema validation against `golden_schema.yaml` | Per-extraction reject |
| Prose value duplicates a table value | Dedup against table observations | Per-extraction drop; counted in `narrative_extractions_deduped` |

---

## 12. Rust-portable contracts

The Python codebase is throwaway. The contracts that survive a Rust rewrite:

1. **CLI surface and exit codes** (Section 9). Tested via `--print-schema` introspection.
2. **Postgres DDL** (`migrations/versions/0001_initial.py`). Hand-written SQL. A Rust port runs the same alembic migrations or applies the SQL directly.
3. **Dossier JSON schema** (`fermdocs --print-schema dossier`). A future Rust binary must produce JSON validating against this. Add a CI job that diffs the two binaries' schema output.
4. **`golden_schema.yaml`** format. Plain YAML, loaded the same way by both languages.
5. **Mapper response JSON schema** (`fermdocs --print-schema mapper`). Provider-agnostic; same shape regardless of which LLM emits it.
6. **Normalizer hint schema** (implicit; serialized inside `value_canonical.normalization`). Three actions, fixed fields.
7. **Narrative extraction schema** (`NarrativeExtraction` Pydantic model). Column, value, unit, evidence, source_paragraph_idx, confidence, rationale. A future Rust port emits identical JSON.

Python-specific code stays encapsulated in:

- `mapping/client.py`, `mapping/gemini_client.py`, `units/normalizer.py` (LLM SDKs)
- `storage/` (SQLAlchemy)
- `domain/models.py` (Pydantic)

A Rust port swaps these modules; everything else is data shape.

---

## 13. Rollback runbook

If the LLM normalizer (or any extractor change) is producing bad values:

### 12.1 Stop the bleeding

```bash
export FERMDOCS_USE_LLM_NORMALIZER=false
```

Future ingests use rule-based only. No code change required.

### 12.2 Identify suspect observations

```sql
SELECT observation_id, experiment_id, file_id, column_name, raw_header,
       value_canonical->>'via' AS via,
       value_canonical->'normalization' AS normalization
FROM golden_observations
WHERE value_canonical->>'via' = 'llm'
  AND superseded_by IS NULL;
```

Filter further by experiment, column, or normalization rationale as needed.

### 12.3 Re-extract

Re-run `fermdocs ingest` for affected experiments. Source files are content-addressed so the originals are still on disk.

New observation rows are created with `via='pint'` (or `via='rule_based'` if rule-based fires). Mark the old LLM-tagged rows superseded:

```sql
UPDATE golden_observations old
SET superseded_by = (
    SELECT new.observation_id
    FROM golden_observations new
    WHERE new.experiment_id = old.experiment_id
      AND new.column_name   = old.column_name
      AND new.source_locator = old.source_locator
      AND new.observation_id != old.observation_id
    ORDER BY new.extracted_at DESC LIMIT 1
)
WHERE old.value_canonical->>'via' = 'llm'
  AND old.superseded_by IS NULL;
```

The dossier-builder filters `superseded_by IS NULL`, so once superseded, old LLM-tagged observations stop appearing in dossiers automatically.

---

## 14. Glossary

| Term | Meaning |
|---|---|
| **Golden column** | A canonical fermentation field defined in `golden_schema.yaml` (e.g., `biomass_g_l`) |
| **Tier 1 / Tier 2 / Tier 3** | Identity columns / common KPIs / residual JSONB. Implicit in the current schema. |
| **Observation** | One value extracted from one cell, with provenance. Multiple observations per (experiment, column) are normal. |
| **Provenance** | The chain `(experiment_id, file_id, source_locator, raw_header)` that points back to a specific cell. |
| **Dossier** | The versioned JSON output consumed by downstream agents. |
| **Confidence band** | Auto / needs_review / residual decision based on mapping confidence. |
| **via** | `value_canonical.via` field: which path produced the canonical value (`pint`, `rule_based`, `llm`). |
| **Hint** | The `NormalizationHint` returned by a `UnitNormalizer`; a recipe code follows to convert. |
| **Superseded** | Observation marked as replaced by a newer extraction. Dossier ignores superseded rows. |
| **Residual** | The JSONB bucket of everything that didn't map to a golden column. |
| **Idempotent re-ingest** | Same file uploaded twice = no duplicate observations. Enforced by `(experiment_id, sha256)` uniqueness. |
| **Tier 1 (narrative)** | Always-on capture of prose into `residual_data.payload.narrative`. No LLM. |
| **Tier 2 (narrative)** | On-by-default LLM-based entity extraction from prose into `golden_observations`. Sonnet-class model; seven safety guards. |
| **Evidence quote** | Verbatim substring of source prose used to verify (and justify) a narrative-derived observation. |
| **Phantom corroboration** | The bug Tier 2 dedup prevents: prose that summarizes a table producing a second observation that looks like an independent confirmation. |
