# Data Flow: PDF → CharacterizationOutput

End-to-end trace of what happens when a fermentation PDF arrives, following the actual code paths in this repo. Two pipelines run back-to-back: **Ingestion** (the existing `fermdocs` package) and **Characterization** (the new `fermdocs_characterize` package). Each stage names the file, function, and shape of the data leaving it.

```
  user uploads file.pdf
         │
         ▼
  +-----------------------------------------------------------------+
  |                       INGESTION (fermdocs)                       |
  |                                                                  |
  |  CLI → Pipeline → Parser → Mapper → Converter → Repo + Residual |
  |                              ↓ LLM     ↓ (pint + Normalizer)    |
  |                          NarrativeExtractor (optional Tier 2)   |
  |                                                                  |
  +-----------------------------------------------------------------+
         │
         ▼  build_dossier()
     {dossier dict}     ← single artifact, content-addressed by experiment
         │
         ▼
  +-----------------------------------------------------------------+
  |                  CHARACTERIZATION (fermdocs_characterize)        |
  |                                                                  |
  |  CLI → Pipeline → Views → Candidates → Builders → Validator    |
  |              (SpecsProvider supplies nominal/std_dev)            |
  +-----------------------------------------------------------------+
         │
         ▼
  CharacterizationOutput.json    ← read by Diagnosis Agent (next)
```

---

## Stage 1 — CLI receives the upload

**Where**: `src/fermdocs/cli.py` (function `main` → `ingest`)

```
$ fermdocs ingest --experiment-id exp-2026-001 --provider gemini run42.pdf
```

The CLI:
- Reads `FERMDOCS_*` env vars (provider, USE_LLM_NORMALIZER, EXTRACT_NARRATIVE)
- Builds a `Mapper`, optional `LLMNarrativeExtractor`, optional `LLMUnitNormalizer` via factories in `src/fermdocs/mapping/factory.py` and `src/fermdocs/units/normalizer.py`
- Constructs `IngestionPipeline` (`src/fermdocs/pipeline.py`) with `router`, `mapper`, `converter`, `repo`, `file_store`, `schema`, `normalizer`, `narrative_extractor`
- Calls `pipeline.run(experiment_id, [Path('run42.pdf')])`

No data leaves the CLI; it's plumbing.

## Stage 2 — File registration + parser routing

**Where**: `src/fermdocs/pipeline.py:IngestionPipeline.run`

For each input file:

1. **Hash and store**: `FileStore.put(path) → (sha256, stored_path)`. Files are content-addressed; re-uploads are deduped.
2. **Register file**: `Repository.create_file(experiment_id, filename, sha256, ...)` writes a row in `source_files`. Returns `file_id` (UUID).
3. **Idempotency**: a UNIQUE constraint on `(experiment_id, sha256)` means re-running the same file is a no-op insert.
4. **Route to parser**: `Router.route(path)` (`src/fermdocs/parsing/router.py`) picks by extension:
   - `.pdf` → `PdfParser` (`src/fermdocs/parsing/pdf_parser.py`)
   - `.xlsx` → `ExcelParser`
   - `.csv` → `CsvParser`
5. **Parse**: `parser.parse(path) → ParseResult(tables: list[ParsedTable], narrative_blocks: list[NarrativeBlock])` (`src/fermdocs/domain/models.py`).

### What `PdfParser.parse` actually does

Lazy-imports `docling`, then:

```python
result = converter.convert(str(path))
document = result.document

# (a) tables: every document.tables[i] becomes a ParsedTable
#     with table_id, headers, rows, locator{type='table', page, table_idx}

# (b) narrative: every document.texts[i] of length ≥ 20 chars
#     becomes a NarrativeBlock with text, type (paragraph/heading/list_item/caption/other),
#     locator{type='narrative', page, paragraph_idx}
```

Output shape after Stage 2:

```
ParseResult
├─ tables: [ParsedTable(table_id="t0", headers=["Time", "Biomass", ...], rows=[[24, 0.75, ...], ...], locator={...}), ...]
└─ narrative_blocks: [NarrativeBlock(text="Foaming was observed at 18h ...", type=PARAGRAPH, locator={...}), ...]
```

## Stage 3 — Header mapping (LLM 1 of up to 3)

**Where**: `src/fermdocs/mapping/mapper.py:Mapper.map_tables`

For each `ParsedTable`:
- Builds a prompt via `src/fermdocs/mapping/prompt.py` containing the headers, sample rows, and the golden schema (loaded from `src/fermdocs/schema/golden_schema.yaml`).
- Calls `MappingClient.complete()` (Anthropic or Gemini, swapped via `factory.py`).
- Parses the model response into `MappingResult → [TableMapping(table_id, entries: [MappingEntry(raw_header, mapped_to | None, raw_unit, confidence, rationale)])]`.

**Trust boundary**: the LLM only proposes mappings, never values. Mapping confidence below 0.85 marks the row as `needs_review` later.

Output shape: a structured mapping for each header. `mapped_to=None` means the header didn't fit any golden column → row goes to residual.

## Stage 4 — Per-cell extraction + unit conversion

**Where**: `src/fermdocs/pipeline.py` (the per-table loop), `src/fermdocs/units/converter.py`, `src/fermdocs/units/normalizer.py`

For each cell in each table:

1. **Skip unmapped**: if `mapped_to is None`, the raw row is collected into `tables_partial` (residual) and we continue.
2. **Build raw value envelope**: `value_raw = {"value": cell, "raw_unit": header_unit}`.
3. **Convert units**: `Converter.convert(value, raw_unit, target_unit)` (`src/fermdocs/units/converter.py`):
   - Tries `pint` first.
   - On pint failure, asks the **Normalizer** for a hint: rule-based first (`RuleBasedNormalizer` strips Unicode superscripts, `"of pellet"` annotations, etc.), then optionally an `LLMUnitNormalizer` (LLM 2 of up to 3) that suggests a `use_pint_expr` rewrite or marks the unit `unconvertible` / `dimensionless`.
   - The normalizer's hint is **applied to the unit string and re-fed to pint**. The LLM never produces a numeric value.
   - Returns `ConversionResult(value, unit, status, via="pint"|"rule_based"|"llm"|"not_applicable", hint?)`.
4. **Persist**: `Repository.write_observation(...)` inserts a row into `golden_observations` with:
   - `observation_id` (UUID), `experiment_id`, `file_id`, `column_name` (golden), `raw_header`
   - `value_raw`, `unit_raw`
   - `value_canonical = {"value": converted, "via": "pint"|...}` (JSONB)
   - `unit_canonical`, `conversion_status`
   - `mapping_confidence`, `extraction_confidence`
   - `source_locator` (which table cell, plus `section: "table"`)
   - `extractor_version`, `superseded_by=None`
5. **Residual capture**: tables/cells that didn't fully map are stored in `residual_data.payload.tables_partial` or `tables_unmapped`.

Note: `source_locator` is intentionally a JSONB blob. v1 characterization expects it to also carry `run_id` and `timestamp_h` (forward-compatible fields). Today's ingestion does not yet emit those; the synthetic fixtures populate them so downstream code can be tested.

## Stage 5 — Narrative tier (Tier 1 always, Tier 2 optional LLM 3 of up to 3)

**Where**: `src/fermdocs/pipeline.py` narrative branch, `src/fermdocs/mapping/narrative_extractor.py`

### Tier 1 (always on)

Every `NarrativeBlock` from `ParseResult.narrative_blocks` is appended to `residual_data.payload.narrative` as raw text + locator. Nothing is interpreted.

### Tier 2 (only if `EXTRACT_NARRATIVE=true`, default in CLI)

`LLMNarrativeExtractor.extract(blocks, schema)`:
1. Chunks blocks into batches of ≤20 paragraphs (`MAX_PARAGRAPHS_PER_CALL`).
2. Calls the LLM with: schema, paragraphs, instructions to output `NarrativeExtraction(column, value, unit, evidence, source_paragraph_idx, confidence, rationale)`.
3. Each emitted extraction passes **seven gates**:
   - **G1**: schema validation (Pydantic).
   - **G2**: source-block resolution — `source_paragraph_idx` must point at a real block.
   - **G3**: evidence substring check — the `evidence` string must appear verbatim in the source block.
   - **G4**: value-string-in-evidence check — `str(value)` must appear in `evidence`.
   - **G5**: sentence bound — `evidence` must contain ≤2 sentence terminators and ≤200 chars.
   - **G6**: dedup against existing **table** observations on the same column at the same time.
   - **G7**: confidence cap at **0.85** for LLM-derived findings.
4. Survivors are written as `Observation` rows with `source_locator.section="narrative"`, `extracted_via="narrative_llm"`, `needs_review=True`.

Failures of any gate are counted into `IngestionFileResult.narrative_extractions_rejected` (or `_deduped`) for telemetry.

## Stage 6 — Build the dossier

**Where**: `src/fermdocs/dossier.py:build_dossier(experiment_id, repository)`

After all files are processed, the dossier is the single artifact every downstream agent reads. It is built fresh (not stored long-term) by:

1. Fetching the experiment row, the file rows, all active (non-superseded) observations, and all residual records from Postgres.
2. Grouping observations by `column_name` via `to_dossier_observation()` (`src/fermdocs/domain/models.py:Observation.to_dossier_observation`).
3. Computing the `ingestion_summary` (counts, coverage, narrative tier stats).
4. Returning a dict with:

```python
{
    "dossier_schema_version": "1.0",
    "experiment": {experiment_id, name, source_files: [...], ...},
    "golden_columns": {
        "Biomass (X)": {
            "canonical_unit": "g/L",
            "observations": [
                {"observation_id": "...", "value": 0.75, "unit": "g/L",
                 "source": {"file_id": "...", "raw_header": "Biomass",
                            "locator": {"section": "table", "run_id": "RUN-A001", "timestamp_h": 24.0}},
                 "confidence": {...}, "via": "pint", ...},
                ...
            ]
        },
        ...
    },
    "residual": {summary, records: [...]},
    "ingestion_summary": {total_observations, golden_coverage_percent, ...}
}
```

**This dict is the input to characterization.** It can be serialized to JSON, persisted, or passed as in-memory state through LangGraph.

---

## Stage 7 — Characterization CLI / LangGraph node

**Where**: `src/fermdocs_characterize/cli.py` or `src/fermdocs_characterize/langgraph_node.py`

```
$ fermdocs-characterize dossier.json --out characterization.json
```

The CLI:
- Parses the dossier JSON.
- Constructs `CharacterizationPipeline()` (`src/fermdocs_characterize/pipeline.py`).
- Calls `pipeline.run(dossier)`.
- Writes the resulting `CharacterizationOutput` JSON to stdout or a file.

The LangGraph node `characterize_node(state)` does the same with a typed state dict. State carries `dossier`, `output`, and `errors`.

## Stage 8 — Specs + summary view

**Where**: `src/fermdocs_characterize/specs.py`, `src/fermdocs_characterize/views/summary.py`

Characterization needs nominal+std_dev specs per variable. v1 reads them from `dossier["_specs"]` (synthetic fixture format) via `DictSpecsProvider.from_dossier(dossier)`. In production, ingestion will provide an `IngestionSpecsProvider` backed by a setpoint table; same `SpecsProvider` Protocol.

Then `build_summary(dossier, specs)` flattens `golden_columns` into a sorted list of `SummaryRow`:

```python
SummaryRow(
    observation_id="...",
    run_id="RUN-A001",      # from observation.source.locator.run_id
    time=24.0,              # from observation.source.locator.timestamp_h
    variable="Biomass (X)",
    value=0.75,
    unit="g/L",
    expected=0.5,           # from specs.get(variable).nominal
    expected_std_dev=0.05,  # from specs.get(variable).std_dev
)
```

Sort key is `(run_id, time, variable, observation_id)` — stable across re-runs. Observations whose locator lacks `run_id` or `timestamp_h` go into `Summary.dropped` (logged, skipped). The pipeline does not crash on partial data.

## Stage 9 — Trajectories

**Where**: `src/fermdocs_characterize/views/trajectories.py:build_trajectories`

Group `Summary.rows` by `(run_id, variable)`. For each group:

- If `dossier["_trajectory_grid"]` is set (`{dt_hours, start, end}`), build a regular grid and impute missing values with **carry-forward** (`imputation_method="carry_forward"`), tracking real vs imputed vs missing per grid point.
- Otherwise, use the observed timestamps directly (no imputation, no missing).

Emits:

```python
Trajectory(
    trajectory_id="T-0001",
    run_id="RUN-B001",
    variable="Biomass (X)",
    time_grid=[0.0, 12.0, 24.0, 36.0, 48.0],
    values=[0.5, 0.5, 0.7, 0.7, 0.85],
    imputation_flags=[False, True, False, True, False],
    imputation_method="carry_forward",
    source_observation_ids=["...", "...", "..."],
    unit="g/L",
    quality=0.6,                              # 60% of grid points are real
    data_quality=DataQuality(pct_missing=0.0, pct_imputed=0.4, pct_real=0.6),
)
```

`quality` and `data_quality` are what the Critic Agent will use to attack findings whose trajectory is sparse.

## Stage 10 — Range-violation candidate generation

**Where**: `src/fermdocs_characterize/candidates/range_violation.py:find_range_violations`

For each `SummaryRow` with `expected` and `expected_std_dev`:

1. Compute `sigmas = round((value - expected) / std_dev, 6)` (rounded to avoid FP boundary slips).
2. If `|sigmas| < 2.0`: skip (still recorded as a Deviation in Stage 11, just no Finding).
3. Severity by tier:
   - `[2.0, 3.0)` → `MINOR`, confidence 0.85
   - `[3.0, 5.0)` → `MAJOR`, confidence 0.95
   - `≥ 5.0` → `CRITICAL`, confidence 0.99
4. Look up the trajectory for `(run_id, variable)`. If `quality < 0.8`, attach a caveat: `"trajectory has 40% imputed/missing data on a 12h grid"`.
5. Build a `CandidateFinding` (no ID yet).

All candidates are pure, deterministic, and **never invoke an LLM**.

## Stage 11 — Sort + assign IDs + builders

**Where**: `src/fermdocs_characterize/pipeline.py:CharacterizationPipeline.run`

After Stage 10, the pipeline:

1. Sorts candidates by `(severity desc, |sigmas| desc, run_id, time, variable)` — worst-first.
2. Assigns finding IDs as `<characterization_id>:F-NNNN` (globally unique across all outputs).
3. **Builders run, each over its own input slice**:
   - `build_deviations(summary)` (`builders/expected_vs_observed.py`) emits one `Deviation` per row that has expected/std_dev (including in-spec rows where sigmas < 2). Sorted by `(run_id, time, variable)`. IDs `D-NNNN`.
   - `build_timeline(findings)` (`builders/timeline.py`) emits one `TimelineEvent` per finding, sorted by `(time asc, severity desc, run_id, variable)`. Computes `lag_to_next_seconds` within each run. IDs `E-NNNN`.
   - `build_open_questions(findings, trajectories, dt_hours)` (`builders/open_questions.py`) emits an `OpenQuestion(decision_type=evidence_request, ...)` for each `(run_id, variable)` whose trajectory has `quality < 0.8` and at least one finding. Includes `would_resolve_with` hints (e.g. `["finer_biomass_sampling", "online_OD_trace"]`). IDs `Q-NNNN`.
   - `build_facts_graph(summary)` (`builders/facts_graph.py`) returns an empty graph in v1; v2+ populates structural Sample/Measurement/Condition nodes.

## Stage 12 — Assemble + validate

**Where**: `src/fermdocs_characterize/pipeline.py` (assembly), `src/fermdocs_characterize/validators/output_validator.py` (cross-cutting validation)

Assemble `CharacterizationOutput`:

```python
CharacterizationOutput(
    meta=Meta(
        schema_version="1.0",
        characterization_version="v1.0.0",
        process_priors_version=None,
        characterization_id=<UUID>,
        generation_timestamp=<datetime>,
        supersedes=None,                    # set when re-running on same dossier
        source_dossier_ids=[experiment_id],
    ),
    findings=[...],
    timeline=[...],
    expected_vs_observed=[...],
    trajectories=[...],
    facts_graph=FactsGraph(nodes=[], edges=[]),
    kinetic_estimates=[],                    # populated in v3
    open_questions=[...],
)
```

**Pydantic validation runs at construction time** (`schema.py:CharacterizationOutput`):
- All finding IDs namespaced to `meta.characterization_id`
- No duplicate IDs across collections
- `TimelineEvent.finding_id` and `OpenQuestion.relevant_finding_ids` resolve to real findings
- All `FactsGraph` edges reference real nodes
- `Finding.confidence ≤ 0.85` when `extracted_via=LLM_JUDGED`
- All ID strings match their regex (`^F-\d{4,}$`, etc.)
- `DataQuality` percentages sum to 1.0
- `Trajectory.values` and `Trajectory.imputation_flags` have the same length as `Trajectory.time_grid`

Then **cross-cutting validation** (`validators/output_validator.py:validate_output`):
- `meta.schema_version` is current
- `meta.process_priors_version`, when set, matches the active priors version
- Every `evidence_observation_ids` / `source_observation_ids` resolves into the source dossier (follows ingestion's supersession chain)
- Returns `list[str]` of errors; pipeline raises `ValidationError` if non-empty

## Stage 13 — Emit

The CLI writes `output.model_dump_json(indent=2)` to stdout or `--out`. The LangGraph node returns the output in the typed state under `state["output"]`.

---

## What the Diagnosis Agent (next) reads

A single JSON file with these top-level keys:

| Key | Used for |
|---|---|
| `meta` | versioning, identity, supersession |
| `findings` | what failed, severity, confidence, evidence, caveats, competing explanations |
| `timeline` | what happened in what order, lags between events |
| `expected_vs_observed` | full setpoint-vs-actual table including in-spec rows |
| `trajectories` | TS-JEPA-friendly time series with `data_quality` |
| `facts_graph` | structural Sample/Measurement/Condition nodes (empty in v1) |
| `kinetic_estimates` | empty in v1–v2; populated in v3 for the simulation agent |
| `open_questions` | structured questions Diagnosis or the Orchestrator must resolve |

Diagnosis can answer:

- *What failed?* → `findings[].summary` + `severity` + `statistics`
- *Where in the data?* → `evidence_observation_ids`, `run_ids`, `time_window`
- *How sure?* → `confidence` interpreted under `extracted_via`, plus `evidence_strength`
- *What else might explain it?* → `competing_explanations`, `caveats`
- *What's the timeline?* → `timeline` ordered by `time` with lags within runs
- *What does the trajectory look like?* → `trajectories` with `imputation_flags`
- *What's still unresolved?* → `open_questions`

## Stable IDs across the chain

Every findable artifact has a stable, globally-unique ID:

| ID | Format | Globally unique? |
|---|---|---|
| Finding | `<characterization_id>:F-NNNN` | yes (UUID-namespaced) |
| TimelineEvent | `E-NNNN` | within a single output |
| Deviation | `D-NNNN` | within a single output |
| Trajectory | `T-NNNN` | within a single output |
| OpenQuestion | `Q-NNNN` | within a single output |
| KineticFit | `K-NNNN` | within a single output |
| Node | `N-NNNN` | within a single output |
| Edge | `G-NNNN` | within a single output |

When the debate loop says *"finding F-0042 contradicts F-0017"*, it means the namespaced IDs from a specific characterization output. If characterization is re-run, a new output supersedes the old (`meta.supersedes` chain) and IDs are re-issued.

## Files at a glance

```
INGESTION                                            CHARACTERIZATION
src/fermdocs/                                        src/fermdocs_characterize/
  cli.py                                               cli.py
  pipeline.py                                          pipeline.py
  dossier.py            ── dossier dict ───────▶     specs.py (SpecsProvider)
  domain/models.py                                     schema.py (contract)
  parsing/                                             vocabularies/*.md
    pdf_parser.py                                      views/
    excel_parser.py                                      summary.py
    csv_parser.py                                        trajectories.py
  mapping/                                             candidates/
    mapper.py                                            range_violation.py
    narrative_extractor.py                             builders/
    factory.py                                           expected_vs_observed.py
  units/                                                 timeline.py
    converter.py                                         open_questions.py
    normalizer.py                                        facts_graph.py
  storage/                                             validators/
    repository.py                                        output_validator.py
    models.py                                          langgraph_node.py
  file_store/
    local.py
  schema/
    golden_schema.yaml
```

## When the new golden schema lands

The ingestion side updates: `golden_schema.yaml`, the alembic migration, and any column-specific normalizer rules. **Characterization code does not change** — it iterates over whatever `golden_columns` the dossier carries. The only artifacts that need regenerating are the three synthetic fixtures (`evals/characterize/fixtures/`), which serves as the acceptance test that the schema swap didn't break the contract.
