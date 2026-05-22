# PDF Document Segmentation (LLM-driven)

Status: DRAFT — awaiting sign-off
Author: drafted in session, 2026-05-03
Branch: caisc-2026-submission

## Problem

PDFs ingested today are flattened into "tables + narrative" with no
document-structure context. Each table picks its own `run_id` via column
heuristics. On multi-batch PDFs (e.g. carotenoid: 6 batches across 18 pages),
the heuristic invents per-row fake run-ids from the time column. Downstream:
diagnose sees one-point trajectories, hypothesis has nothing to correlate,
output is shallow.

Probe of the carotenoid PDF (Docling output):
- 4 of 6 batch boundaries appear as `SectionHeaderItem("BATCH-NN REPORT")`
- 6 of 6 batches end with a `TextItem` starting `"Batch closure:"`
- Real run structure is fully recoverable from document context, just not
  from any single table in isolation.

The information needed to assign tables to runs lives one layer up from the
resolver, in the document tree. The resolver was being asked to recover
information that was discarded before it ran.

## Decision

Add an LLM-driven **document segmentation pre-pass** between PDF parsing and
ingest. The LLM reads the parsed document outline (headings + first-line of
text blocks + table positions, **not** table values) and returns a structured
map: list of runs, each with display name, source signal, and the table
indices that belong to it.

The map is injected into the existing `ManifestStrategy` path of
`RunIdResolver`. From the resolver's perspective, the output looks identical
to an operator-supplied manifest. No new strategy class needed.

This decision was reached after evaluating three options:
- **A (chosen)**: Always run the LLM segmenter. Robust to any document shape.
- **A'**: Try deterministic signals first (BATCH-NN headers + "Batch closure:"
  sentinels), fall back to LLM if signals disagree or absent. Cheaper but
  more code paths and more failure modes.
- **A''**: Pure deterministic. Brittle to PDFs that don't match the carotenoid
  pattern.

A was chosen for: simplest control flow (one path, not two), most generalizable
(no per-PDF assumptions baked in), the cost is bounded (one LLM call per PDF
at ~$0.005), and the determinism we'd save by going A' is not worth two
codepaths to maintain.

## Non-goals

- Replacing the existing `ColumnStrategy` / `FilenameStrategy` /
  `SyntheticStrategy` chain. Those still serve CSV/Excel and PDFs where the
  LLM declines to segment.
- LLM table-type classification (timecourse vs feed-plan vs composition).
  Handled by a separate deterministic filter (see Feed-plan stash below).
- Cross-document run-id reconciliation (same batch reported in two files).
- Identity extraction. The existing `build_identity_client` continues to
  classify organism/process/scale.

## Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│  CLI: fermdocs ingest                                               │
└──────────────────────────────┬──────────────────────────────────────┘
                               │
        ┌──────────────────────▼─────────────────────┐
        │  FormatRouter.parse(path)                  │
        │    → DoclingPdfParser (PDF only)           │
        │      returns ParseResult(tables, blocks)   │
        └──────────────────────┬─────────────────────┘
                               │
        ┌──────────────────────▼─────────────────────┐  NEW
        │  DocumentSegmenter.segment(parse_result)   │
        │    PDF only; CSV/Excel skip this step      │
        │    1. Build outline (headings + 1st line   │
        │       of each text block + table positions)│
        │    2. Call LLM with structured-output      │
        │       schema → DocumentMap                  │
        │    3. Validate: every table_idx assigned   │
        │       exactly once; runs are non-empty     │
        │    4. Return DocumentMap or None on error  │
        └──────────────────────┬─────────────────────┘
                               │
        ┌──────────────────────▼─────────────────────┐
        │  IngestionPipeline.ingest()                │
        │    For each ParsedTable:                   │
        │    - if DocumentMap present:               │
        │        run_id = doc_map.run_for(table_idx) │
        │        injected as manifest_run_id         │
        │    - else: existing strategy chain         │
        └────────────────────────────────────────────┘
```

### DocumentMap schema

```python
class RunSegment(BaseModel):
    run_id: str           # Canonical e.g. "RUN-0001"
    display_name: str     # As seen in PDF, e.g. "BATCH-01 REPORT"
    table_indices: list[int]  # Which table_idx values belong to this run
    source_signal: str    # "section_header" | "text_pattern" | "inferred"
    confidence: float     # LLM-reported, 0.0-1.0
    rationale: str        # Short LLM explanation, for debugging

class DocumentMap(BaseModel):
    schema_version: Literal["1.0"] = "1.0"
    file_id: str
    runs: list[RunSegment]
    unassigned_table_indices: list[int]  # Tables LLM couldn't place
    overall_confidence: float
    llm_model: str
    llm_provider: str
```

### LLM prompt shape (sketch)

Input to the LLM is **structural only** — no table values:

```
Document outline for {filename}:

[HEADER L1] CONFIDENTIAL                                  (page 1)
[HEADER L1] BATCH-01 REPORT  (inferred — not in tree)     (page 1)
[TEXT]      "Objective: To grow recombinant ..."          (page 1)
[TABLE 0]   headers: [Component, Concentration (g/L)]     (page 1, 5 rows)
...
[TEXT]      "Batch closure: Cultivation completed at 82h" (page 4)
[HEADER L1] BATCH-02 REPORT                               (page 5)
...

Identify experimental runs in this document. For each run, list which
TABLE indices belong to it. A "run" = one fermentation experiment (one
vessel, one timecourse). Skip composition/feed-plan tables; assign only
measurement tables (timecourse data with a time column).

Return structured JSON matching DocumentMap schema.
```

### Manifest interaction (loud-warning behavior)

Per design call: manifest wins, loudly.

```
Operator supplies manifest run_id "OPERATOR-RUN-42" for the file.
DocumentSegmenter still runs and produces a 6-run map.

Pipeline behavior:
  - All tables in the file get run_id = "OPERATOR-RUN-42"
  - WARN log: "Manifest pinned 1 run for {file}, segmenter detected
    6 runs ({display_names}). Manifest wins. If the segmenter is right,
    omit the manifest."
  - DocumentMap is still persisted to residual JSONB for inspection.
```

### Feed-plan stash

Separate from the segmenter, deterministic. When `_extract_tables` parses a
table, classify by header pattern:

```python
def _is_feed_plan_table(headers: list[str]) -> bool:
    norm = {h.strip().lower() for h in headers}
    has_segment = any("segment" in h for h in norm)
    has_batch_hours = any("batch hour" in h or "batch hours" in h for h in norm)
    has_feed_rate = any("feed rate" in h for h in norm)
    return has_segment and (has_batch_hours or has_feed_rate)
```

Feed-plan tables do **not** flow into `golden_columns`. They're stored under
`residual.process_recipe[file_id]` as raw header+row JSON. Future use:
diagnose can reference them as "operator-planned setpoints" rather than
flagging the planned 15 mL/h as an impossible 15 L/h measurement.

### Persistence

`DocumentMap` is written to two places:
1. `Repository.write_document_map(file_id, doc_map)` — new table, queryable.
2. `dossier.json` under `experiment.document_maps[file_id]` — visible to
   downstream agents, including diagnose.

`process_recipe` stash lives in residual JSONB on the dossier.

## Failure modes & fallbacks

| Failure | Behavior |
|---|---|
| LLM call times out | Skip segmenter; ingest falls through to existing column heuristics. WARN. |
| LLM returns malformed JSON | Same as timeout — skip + WARN, retry once. |
| LLM assigns same table_idx to two runs | Validation rejects map; skip + WARN. |
| LLM returns 0 runs | Treat as "single fermentation" — synthesize one run with all tables, source_signal="inferred". |
| LLM returns runs whose table_indices don't sum to all measurement tables | unassigned_table_indices is populated; those tables go through existing chain. |
| `GEMINI_API_KEY` missing | Skip segmenter entirely. Log INFO once per process. |
| Operator manifest present | Run segmenter for inspection only; manifest still wins (with loud-warning). |

## Test plan

### Unit tests

- `DocumentSegmenter` with a stubbed LLM client:
  - happy path: 6 batches, returns 6-run map
  - LLM returns malformed JSON → returns None, logs warning
  - LLM duplicate table_idx → returns None
  - LLM unassigned tables → map is valid but `unassigned_table_indices` populated
  - empty document (no tables) → returns map with 0 runs (no LLM call)

- `_is_feed_plan_table` header-classification:
  - segment + batch hours + feed rate → True
  - segment + duration → False (might be a different segment table)
  - timecourse table (time + OD + WCW) → False

- `ManifestStrategy` injection from DocumentMap:
  - DocumentMap maps table_idx 5 to RUN-0003 → resolver sees
    manifest_run_id="RUN-0003" for that table

### Integration tests

- **CRITICAL regression — carotenoid PDF**: ingest end-to-end, assert dossier
  has exactly 6 distinct run_ids matching BATCH-01..06, each with 9-21
  timepoints, WCW values match table content. With LLM stubbed deterministic.
- **CRITICAL regression — IndPenSim CSV**: ingest still produces correct
  run-ids via ColumnStrategy (segmenter must not run for CSV).
- **Manifest loud warning**: ingest carotenoid with manifest_run_id pinned;
  assert all observations get the manifest id AND a warning log line is
  produced naming the segmenter's count.
- **Feed-plan stash**: ingest carotenoid; assert feed-plan tables (BATCH-04
  page 10, BATCH-05 page 13, BATCH-06 page 16) are absent from
  `feed_rate_l_per_h` golden_column AND present in
  `residual.process_recipe`.

### Live LLM eval (separate, opt-in)

One eval run with the real Gemini call on the carotenoid PDF, gated
behind `pytest -m live_llm`. Asserts: 6 runs detected, all 6 BATCH names
in display_names, table assignments match the integration-test expectations
within tolerance.

## Migration / rollback

- New code is additive. Existing strategy chain stays intact for CSV/Excel
  and as fallback when segmenter fails or is disabled.
- Feature flag: `FERMDOCS_PDF_SEGMENT=true` (default true once shipped).
  Set to `false` to fully disable and revert to old behavior.
- DB schema: one new table `document_maps` (file_id, json). Additive
  migration, no data backfill needed.

## Cost & latency

- One `gemini-3.1-pro-preview` call per PDF, structured output. Input tokens
  scale with document length (full outline + every text-block first line +
  every table header row); cost is not a constraint per project decision.
- Latency: ~5-15 seconds added to ingest depending on PDF size. PDF ingest
  already takes 30-60s for Docling parse, so this is a small relative cost.

## Resolved decisions

- **LLM model**: `gemini-3.1-pro-preview`. Override via
  `FERMDOCS_SEGMENTER_MODEL` env var.
- **Outline truncation**: none. Send full outline regardless of length;
  budget is not a constraint. If a real PDF ever exceeds the model's context
  window, revisit then.
- **Feature flag default**: `FERMDOCS_PDF_SEGMENT=true` ships on. Set false
  to disable and revert to old column-heuristic behavior.

## Open questions (to resolve before merge)

1. **Segmenter on CSV-only ingest**: should be a no-op (no PDF in inputs) —
   confirm via type-check at the segmenter entry.

## Implementation sequence (for execution after sign-off)

1. Add `DocumentMap` + `RunSegment` Pydantic models. Tests for validation.
2. Add `_is_feed_plan_table` filter to `DoclingPdfParser._extract_tables`.
   Stash matched tables to a new `ParseResult.feed_plan_tables` field.
   Tests using a stub Docling document.
3. Add `DocumentSegmenter` class with stubbed LLM-call interface.
   Tests for all failure modes with fake clients.
4. Wire `DocumentSegmenter` into `IngestionPipeline.ingest()` between
   parse and per-table mapping. Inject DocumentMap as
   `manifest_run_id` per-table.
5. Add manifest-loud-warning logic when both manifest and DocumentMap exist.
6. Add `Repository.write_document_map`, dossier `document_maps` field.
7. Integration test: carotenoid PDF end-to-end with stubbed LLM.
8. Integration test: IndPenSim CSV regression.
9. Live LLM eval (opt-in marker).

Each step is one commit. Total ~5-7 commits. Each commit must pass tests
on its own.

## Decisions deferred to TODOS

- LLM table-type classifier (richer than the deterministic feed-plan filter).
  Pick up only if the deterministic filter misses real-world tables.
- Cross-document reconciliation.
- Operator UI to preview detected segmentation and override before ingest.
