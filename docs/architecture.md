# fermdocs — architecture

## Pipeline

```
file (csv|xlsx|pdf)
      |
      v
+------------------+    +-------------------+    +-------------------+
| FormatRouter     | -> | HeaderMapper      | -> | UnitConverter     |
| (one per ext)    |    | (LLM, batched)    |    | (pint, custom reg)|
+------------------+    +-------------------+    +-------------------+
      |                          |                         |
      v                          v                         v
  ParsedTable[]             MappingResult              Observation[]
                                                            |
                                                            v
                                                  +------------------+
                                                  | Repository       |
                                                  | (Postgres only)  |
                                                  +------------------+
                                                            |
                                                            v
                                                  build_dossier(exp_id)
                                                            |
                                                            v
                                                     dossier JSON
                                                  (versioned 1.0)
```

## Module boundaries

```
src/fermdocs/
  domain/            pure data + Pydantic. imports nothing from project.
  parsing/           file -> ParsedTable. only place that reads files.
  mapping/           ParsedTable[] -> MappingResult. only place that calls LLM.
  units/             value + unit -> canonical. only place that imports pint.
  storage/           Postgres I/O. only place that imports SQLAlchemy.
  file_store/        original-file persistence. swap LocalFileStore -> S3 later.
  pipeline.py        the only file that knows the full sequence.
  dossier.py         read-only projection of storage -> versioned dossier JSON.
  cli.py             thin Click wrapper. wires real impls; tests wire fakes.
  schema/            golden_schema.yaml. the editable contract.
```

## Key invariants

1. `domain/` has no project-internal imports. Enforce in CI:
   `! grep -r '^from fermdocs' src/fermdocs/domain/`
2. `storage/` is the only module that imports SQLAlchemy.
3. `mapping/client.py` is the only module that imports the Anthropic SDK.
4. Pydantic models are the *only* thing serialized to dossier JSON. Never
   `.model_dump()` ORM rows directly. `Observation.to_dossier_observation()` is the
   single source of truth for dossier shape.
5. `dossier_schema_version` is part of every dossier. Consumers branch on it.

## Polymorphic value storage

`golden_observations.value_raw` and `value_canonical` are JSONB columns shaped:

```json
{"value": <any>, "type": "float|int|text|bool"}
```

This lets one table hold numeric titers and string strain IDs without a column
explosion. Numeric queries can still hit the JSONB efficiently with expression
indexes, e.g. `((value_canonical->>'value')::numeric)`.

## Locator

`source_locator` is JSONB and format-agnostic:

- PDF: `{"format": "pdf", "page": 4, "table_idx": 1, "row": 5, "col": 3}`
- Excel: `{"format": "xlsx", "sheet": "OnlineData", "row": 12, "col": 1}`
- CSV: `{"format": "csv", "row": 47, "col": 6}`

## Re-extraction

Observations and residuals carry `extractor_version` and `superseded_by`.
Re-running ingestion on the same source file creates new rows; old rows get
`superseded_by` set. `fetch_active_observations` returns only the unsuperseded.
No destructive updates.

## Confidence as a rank

LLM-self-reported confidence is treated as a rank, not a calibrated probability.
Three bands (constants in `mapping/confidence.py`):

- >= 0.85: auto-accept, lands in `golden_observations`
- 0.60 - 0.85: lands in `golden_observations` with `needs_review=true`
- < 0.60: falls through to residual JSONB

Calibrate thresholds against fixture set before user-facing copy says "X% sure".

## Unit normalization

When pint cannot parse a unit string, an optional `UnitNormalizer` is consulted.
The normalizer never emits a converted numeric value -- it tells the converter
what to do via a `NormalizationHint` (one of three actions):

- `use_pint_expr`: rewrite the unit string into something pint can parse, retry pint.
- `dimensionless`: store value as-is, no conversion.
- `unconvertible`: give up; status=failed.

Two implementations ship:

- `RuleBasedNormalizer` -- deterministic regex transforms (Unicode superscripts,
  'of pellet' annotations, known dimensionless tokens). Always on.
- `LLMUnitNormalizer` -- opt-in via `FERMDOCS_USE_LLM_NORMALIZER=true` (or the
  `--llm-normalizer` CLI flag). Caches per `(unit_raw, canonical_unit)` over the
  lifetime of one pipeline run. Failures (network, malformed JSON, auth)
  degrade to `unconvertible` rather than crashing.

The `via` field on `value_canonical` records the path: `'pint' | 'rule_based' |
'llm' | 'not_applicable'`. When a normalizer fired, `value_canonical.normalization`
holds the full hint (action, pint_expr, rationale, confidence) for audit.

## Rollback runbook

If the LLM normalizer is making bad hints:

1. Disable: `export FERMDOCS_USE_LLM_NORMALIZER=false`. Future ingests use
   rule-based only.

2. Identify suspect observations:
   ```sql
   SELECT observation_id, experiment_id, column_name, raw_header,
          value_canonical->>'via' AS via,
          value_canonical->'normalization' AS normalization
   FROM golden_observations
   WHERE value_canonical->>'via' = 'llm'
     AND superseded_by IS NULL;
   ```

3. Re-ingest affected experiments. Source files are content-addressed by sha256
   in `LocalFileStore`, so re-running `fermdocs ingest` on the same file produces
   new observation rows; mark old rows superseded:
   ```sql
   UPDATE golden_observations old
   SET superseded_by = (
       SELECT new.observation_id FROM golden_observations new
       WHERE new.experiment_id = old.experiment_id
         AND new.column_name   = old.column_name
         AND new.source_locator = old.source_locator
         AND new.observation_id != old.observation_id
       ORDER BY new.extracted_at DESC LIMIT 1
   )
   WHERE old.value_canonical->>'via' = 'llm' AND old.superseded_by IS NULL;
   ```

The dossier-builder filters `superseded_by IS NULL` so once superseded, old
LLM-tagged observations stop appearing in dossiers automatically.

## Rust-port-friendly

The Python codebase is throwaway. The contracts that survive a Rust rewrite:

- CLI flag set + exit codes (see README)
- Postgres DDL (`migrations/versions/0001_initial.py`)
- Dossier JSON schema (`fermdocs --print-schema dossier`)
- `golden_schema.yaml` (language-agnostic)
- Mapper response JSON schema (`fermdocs --print-schema mapper`)

Python-isms (Pydantic, SQLAlchemy, Anthropic SDK) stay encapsulated in
`mapping/client.py`, `storage/`, and `domain/models.py`. A Rust port swaps
those modules.
