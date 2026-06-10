from __future__ import annotations

import math
import mimetypes
import os
import uuid
from datetime import datetime
from pathlib import Path

from fermdocs.domain.golden_schema import load_schema
from typing import Any

from fermdocs.domain.models import (
    ConfidenceBand,
    ConversionStatus,
    GoldenSchema,
    IngestionFileResult,
    IngestionResult,
    MappingEntry,
    MappingResult,
    NarrativeBlock,
    NarrativeExtraction,
    Observation,
    ObservationType,
    ParsedTable,
    ResidualPayload,
    TableMapping,
)
from fermdocs.file_store.base import FileStore
from fermdocs.mapping.confidence import band
from fermdocs.mapping.layout_detector import LayoutDetector
from fermdocs.mapping.mapper import HeaderMapper
from fermdocs.mapping.narrative_extractor import (
    NARRATIVE_CONFIDENCE_CAP,
    NarrativeExtractor,
    chunk_blocks,
    is_dup_of_table_observations,
    verify_evidence,
)
from fermdocs.parsing.document_segmenter import DocumentSegmenter
from fermdocs.parsing.router import FormatRouter
from fermdocs.parsing.run_id_resolver import RunIdResolution, RunIdResolver
from fermdocs.storage.repository import FileRecord, Repository
from fermdocs.units.converter import UnitConverter
from fermdocs.units.normalizer import UnitNormalizer

EXTRACTOR_VERSION = "v0.1.0"

# Observations from a layout-detected table whose structural confidence is
# below this go to the review queue (the detector wasn't sure it read the
# sheet right). Env-tunable. The detector already caps confidence at 0.85.
_LAYOUT_REVIEW_THRESHOLD = float(
    os.environ.get("FERMDOCS_LAYOUT_REVIEW_THRESHOLD", "0.7")
)
# Data-relative scale guard. For a variable measured across several runs, a
# run whose values are more than this multiple of the cross-run median scale
# is a suspected unit/scale problem (e.g. one column tagged g/L but holding
# mg/L, inflating ~1000x) -> flag for review. This compares each run to the
# DATA's own distribution, not to any hardcoded expected value, and needs
# enough runs to form a baseline (below).
_SCALE_OUTLIER_FACTOR = float(os.environ.get("FERMDOCS_SCALE_OUTLIER_FACTOR", "50"))
_SCALE_OUTLIER_MIN_RUNS = int(os.environ.get("FERMDOCS_SCALE_OUTLIER_MIN_RUNS", "3"))

# File suffixes that go through DoclingPdfParser. The segmenter only runs
# for these — CSV/Excel paths skip the LLM call entirely and use the
# existing column-heuristic chain unchanged.
_PDF_SUFFIXES = {".pdf"}


class IngestionPipeline:
    def __init__(
        self,
        router: FormatRouter,
        mapper: HeaderMapper,
        unit_converter: UnitConverter,
        repository: Repository,
        file_store: FileStore,
        schema: GoldenSchema | None = None,
        normalizer: UnitNormalizer | None = None,
        narrative_extractor: NarrativeExtractor | None = None,
        run_id_resolver: "RunIdResolver | None" = None,
        document_segmenter: DocumentSegmenter | None = None,
        layout_detector: "LayoutDetector | None" = None,
    ):
        self._router = router
        self._mapper = mapper
        self._converter = unit_converter
        self._repo = repository
        self._files = file_store
        self._schema = schema or load_schema()
        self._schema_index = self._schema.by_name()
        self._normalizer = normalizer
        self._narrative_extractor = narrative_extractor
        # Optional LLM layout detector. None preserves the legacy path
        # exactly (naive parser tables); when wired, it finds the real
        # header row + data span on messy sheets so structure-agnostic
        # ingestion works. See mapping/layout_detector.py.
        self._layout_detector = layout_detector
        # Strategy-chain resolver. Tests can inject a custom chain (e.g.
        # only ColumnStrategy) to assert specific paths.
        from fermdocs.parsing.run_id_resolver import RunIdResolver as _RIR

        self._run_id_resolver = run_id_resolver or _RIR()
        # PDF-only LLM document segmenter. None disables segmentation and
        # the pipeline falls back to the existing column-heuristic chain
        # for every table. CSV/Excel paths never invoke the segmenter
        # regardless of this setting.
        self._segmenter = document_segmenter

    def ingest(
        self,
        experiment_id: str,
        files: list[Path],
        *,
        manifest_run_id: str | None = None,
    ) -> IngestionResult:
        """Ingest files for an experiment.

        `manifest_run_id`, when supplied, pins every observation in every
        file to that run-id. The segmenter still runs (its output is
        recorded for inspection) and emits a WARN if it disagrees with the
        manifest. CLI plumbing for this parameter lands in a follow-up
        commit; for now it defaults to None to preserve existing behavior.
        """
        self._repo.upsert_experiment(experiment_id)
        results: list[IngestionFileResult] = []
        for path in files:
            results.append(
                self._ingest_one(
                    experiment_id, path, manifest_run_id=manifest_run_id
                )
            )
        return IngestionResult(experiment_id=experiment_id, files=results)

    def _ingest_one(
        self,
        experiment_id: str,
        path: Path,
        *,
        manifest_run_id: str | None = None,
    ) -> IngestionFileResult:
        stored = self._files.put(path)
        mime, _ = mimetypes.guess_type(path.name)
        record = FileRecord(
            file_id=uuid.uuid4(),
            experiment_id=experiment_id,
            filename=path.name,
            sha256=stored.sha256,
            storage_path=stored.storage_path,
            size_bytes=stored.size_bytes,
            mime_type=mime,
        )
        file_id, created = self._repo.find_or_create_file(record)
        if not created:
            return IngestionFileResult(
                file_id=file_id, filename=path.name, parse_status="ok"
            )
        try:
            parsed = self._router.parse(path)
        except Exception as e:
            self._repo.mark_file_parsed(file_id, "failed", str(e))
            return IngestionFileResult(
                file_id=file_id, filename=path.name, parse_status="failed", parse_error=str(e)
            )

        tables = parsed.tables
        narrative_blocks = parsed.narrative_blocks

        # Detector-primary: when a layout detector is wired and the parser
        # exposed raw grids (CSV/Excel), the detector locates the real
        # header row + data span and replaces the naive row-0-is-header
        # tables. On any detector miss it returns [] and we fall back to the
        # parser's tables, so clean files still ingest even with no LLM.
        if self._layout_detector is not None and parsed.raw_grids:
            detected = self._layout_detector.detect(
                parsed.raw_grids, schema=self._schema
            )
            if detected:
                tables = detected

        # PDF-only: run the LLM segmenter to assign each table to an
        # experimental run. CSV/Excel inputs skip this entirely (the
        # column-heuristic chain works fine for tabular files with real
        # run-id columns). The segmenter is best-effort; on any failure
        # it returns None and the existing chain handles every table.
        # When manifest_run_id is pinned, segmenter still runs and logs a
        # disagreement warning if it detects multiple distinct runs —
        # operator's signal that the manifest may be wrong.
        doc_map = None
        if (
            self._segmenter is not None
            and path.suffix.lower() in _PDF_SUFFIXES
        ):
            doc_map = self._segmenter.segment(
                parsed,
                file_id=str(file_id),
                manifest_run_id=manifest_run_id,
            )

        mapping = self._map_tables(tables)
        mapping_by_table = {tm.table_id: tm for tm in mapping.tables}
        # Cross-run dominant source unit per variable: keeps every run on the
        # same scale (a lone g/L column in one run won't read 10x the runs that
        # only report %w/w). No conversion, no density, no hardcoded value.
        dominant_units = _dominant_source_units(mapping)

        table_observations: list[Observation] = []
        residual = ResidualPayload()

        # Stash any operator-supplied feeding-schedule tables (PDF only;
        # always empty for CSV/Excel). These were filtered out of the
        # observation stream by the parser to avoid polluting
        # feed_rate_l_per_h with planned-setpoint values.
        if parsed.feed_plan_tables:
            residual.process_recipe = [
                {
                    "table_id": t.table_id,
                    "headers": t.headers,
                    "rows": t.rows,
                    "locator": t.locator,
                }
                for t in parsed.feed_plan_tables
            ]

        # Persist DocumentMap to residual for inspection by downstream
        # agents (diagnose can cite "BATCH-04 REPORT (page 9)" instead of
        # raw run_ids).
        if doc_map is not None:
            residual.document_map = doc_map.model_dump(mode="json")

        for table in tables:
            tm = mapping_by_table.get(table.table_id)
            if tm is None:
                _add_unmapped(residual, table, reason="no_mapping_returned")
                continue
            obs, partial = self._observations_for_table(
                experiment_id,
                file_id,
                table,
                tm,
                doc_map=doc_map,
                manifest_run_id=manifest_run_id,
                dominant_units=dominant_units,
            )
            table_observations.extend(obs)
            if partial:
                _add_partial(residual, table, tm, partial)

        # Cross-run design factors: the experimental variables deliberately
        # varied across runs (nutrient source/amount, feed timing, titer
        # target) read from the metadata/highlights blocks. One pass over all
        # this file's runs so factor names line up and only varying factors
        # survive. These are the levers the cross-run recommendation engine
        # reasons over. Never breaks ingestion.
        if self._layout_detector is not None and parsed.raw_grids:
            # Bound the factor digest to each sheet's metadata region (above
            # the data table we just located) — smaller prompt, sharper read.
            table_starts: dict[str, int] = {}
            for t in tables:
                if isinstance(t.locator, dict) and isinstance(
                    t.locator.get("header_row"), int
                ):
                    sid = str(t.table_id).split("~", 1)[0]
                    hr = t.locator["header_row"]
                    table_starts[sid] = min(table_starts.get(sid, hr), hr)
            try:
                factors = self._layout_detector.extract_design_factors(
                    parsed.raw_grids, table_starts=table_starts
                )
            except Exception:  # noqa: BLE001 — conditions are best-effort
                factors = {}
            for run_id, knobs in factors.items():
                residual.run_conditions.setdefault(str(run_id), {}).update(knobs)

        # Tier 1: always capture narrative blocks in residual.
        if narrative_blocks:
            residual.narrative = [b.model_dump(mode="json") for b in narrative_blocks]

        # Tier 2: optional LLM extraction over narrative blocks.
        narrative_observations, narrative_stats = self._extract_narrative_observations(
            experiment_id, file_id, narrative_blocks, table_observations
        )

        all_observations = table_observations + narrative_observations
        # Data-relative scale guard (replaces the old hardcoded-nominal sanity):
        # flag runs whose values for a variable are wildly off the cross-run
        # median scale. Pure function of this file's own data.
        _flag_scale_outliers(all_observations)
        n_obs = self._repo.write_observations(all_observations)
        n_review = sum(1 for o in all_observations if getattr(o, "needs_review", False))
        # Coverage gate: a file that parsed and carried tabular data but
        # produced zero observations is a silent ingestion failure (the
        # praaj 0-obs case). Surface it loudly instead of returning a
        # clean-looking empty result.
        ingestion_warning: str | None = None
        had_tabular_input = bool(tables) or bool(parsed.raw_grids)
        if n_obs == 0 and had_tabular_input:
            ingestion_warning = "no_observations_extracted"
        n_res = 0
        if any(
            getattr(residual, f) for f in residual.model_fields if getattr(residual, f)
        ):
            self._repo.write_residual(file_id, experiment_id, residual, EXTRACTOR_VERSION)
            n_res = 1

        self._repo.mark_file_parsed(file_id, "ok", parsed_at=datetime.utcnow())
        return IngestionFileResult(
            file_id=file_id,
            filename=path.name,
            parse_status="ok",
            observations_written=n_obs,
            residuals_written=n_res,
            narrative_blocks_captured=len(narrative_blocks),
            narrative_extractions_kept=narrative_stats["kept"],
            narrative_extractions_rejected=narrative_stats["rejected"],
            narrative_extractions_deduped=narrative_stats["deduped"],
            observations_needing_review=n_review,
            ingestion_warning=ingestion_warning,
        )

    def _extract_narrative_observations(
        self,
        experiment_id: str,
        file_id: uuid.UUID,
        blocks: list[NarrativeBlock],
        table_observations: list[Observation],
    ) -> tuple[list[Observation], dict[str, int]]:
        stats = {"kept": 0, "rejected": 0, "deduped": 0}
        if not blocks or self._narrative_extractor is None:
            return [], stats

        out: list[Observation] = []
        for chunk in chunk_blocks(blocks):
            blocks_by_idx = {
                b.locator.get("paragraph_idx"): b for b in chunk
            }
            extractions = self._narrative_extractor.extract(chunk, self._schema)
            for ext in extractions:
                # 1. Schema validation: column must exist in golden schema.
                if ext.column not in self._schema_index:
                    stats["rejected"] += 1
                    continue
                # 2. Source paragraph must exist in this chunk.
                src_block = blocks_by_idx.get(ext.source_paragraph_idx)
                if src_block is None:
                    stats["rejected"] += 1
                    continue
                # 3. Evidence verification.
                ok, _reason = verify_evidence(ext.evidence, src_block.text, ext.value)
                if not ok:
                    stats["rejected"] += 1
                    continue
                # 4. Dedup against table observations.
                if is_dup_of_table_observations(ext, table_observations):
                    stats["deduped"] += 1
                    continue
                # 5. Build the observation.
                obs = self._build_narrative_observation(
                    experiment_id, file_id, ext, src_block
                )
                out.append(obs)
                stats["kept"] += 1
        return out, stats

    def _build_narrative_observation(
        self,
        experiment_id: str,
        file_id: uuid.UUID,
        ext: NarrativeExtraction,
        src_block: NarrativeBlock,
    ) -> Observation:
        golden = self._schema_index[ext.column]
        data_type = str(golden.data_type)
        conversion = self._converter.convert(
            ext.value, ext.unit, golden.canonical_unit, normalizer=self._normalizer,
        )
        value_raw = {"value": _coerce(ext.value, data_type), "type": data_type}
        value_canonical: dict[str, Any] | None = None
        if conversion.value_canonical is not None:
            value_canonical = {
                "value": conversion.value_canonical,
                "type": data_type,
                "via": conversion.via,
                "extracted_via": "narrative_llm",
            }
            if conversion.hint is not None:
                value_canonical["normalization"] = {
                    "action": conversion.hint.action.value,
                    "pint_expr": conversion.hint.pint_expr,
                    "rationale": conversion.hint.rationale,
                    "confidence": conversion.hint.confidence,
                    "source": conversion.hint.source,
                }
        elif conversion.status == ConversionStatus.NOT_APPLICABLE:
            # Text columns: store raw with extracted_via marker even though canonical is None.
            pass

        # Capped confidence: narrative observations never auto-accept.
        capped_conf = min(ext.confidence, NARRATIVE_CONFIDENCE_CAP)
        locator = {
            **src_block.locator,
            "section": "narrative",
            "evidence_quote": ext.evidence,
        }
        return Observation(
            observation_id=uuid.uuid4(),
            experiment_id=experiment_id,
            file_id=file_id,
            column_name=ext.column,
            raw_header=ext.evidence[:80],  # short label for the value source
            observation_type=ObservationType.REPORTED,  # fixed; LLM does not classify
            value_raw=value_raw,
            unit_raw=ext.unit,
            value_canonical=value_canonical,
            unit_canonical=conversion.unit_canonical,
            conversion_status=conversion.status,
            source_locator=locator,
            mapping_confidence=capped_conf,
            extraction_confidence=_extraction_confidence(ext.value, data_type),
            needs_review=True,  # narrative observations always go to review queue
            extractor_version=EXTRACTOR_VERSION,
            schema_version=self._schema.version,
            extracted_at=datetime.utcnow(),
        )

    def _map_tables(self, tables: list[ParsedTable]) -> MappingResult:
        """Map headers, deduping tables that share an identical header set.

        A templated workbook (e.g. praaj's 15 sheets with identical columns)
        otherwise sends every table to the mapper in one call, producing a huge
        response that can truncate and fail to parse. We map one representative
        per distinct header signature and copy its entries to the rest — same
        result, a fraction of the request, and no truncation. Entries are keyed
        by raw_header, so copying across identical-header tables is exact.
        """
        if not tables:
            return MappingResult(tables=[])
        order: list[tuple] = []
        members: dict[tuple, list[ParsedTable]] = {}
        for t in tables:
            sig = tuple(t.headers)
            if sig not in members:
                members[sig] = []
                order.append(sig)
            members[sig].append(t)
        reps = [members[sig][0] for sig in order]
        result = self._mapper.map(reps, self._schema)
        rep_tms = {tm.table_id: tm for tm in result.tables}
        # Fallback: if the mapper didn't echo table_ids, match by order.
        by_order = (
            dict(zip(order, result.tables))
            if len(result.tables) == len(order)
            else {}
        )
        out: list[TableMapping] = []
        for sig in order:
            rep = members[sig][0]
            tm = rep_tms.get(rep.table_id) or by_order.get(sig)
            if tm is None:
                continue
            for t in members[sig]:
                out.append(TableMapping(table_id=t.table_id, entries=tm.entries))
        return MappingResult(tables=out)

    def _observations_for_table(
        self,
        experiment_id: str,
        file_id: uuid.UUID,
        table: ParsedTable,
        mapping: TableMapping,
        *,
        doc_map: Any | None = None,
        manifest_run_id: str | None = None,
        dominant_units: dict[str, str] | None = None,
    ) -> tuple[list[Observation], list[dict]]:
        observations: list[Observation] = []
        unmapped_columns: list[dict] = []
        col_index = {h: i for i, h in enumerate(table.headers)}
        time_col_idx = _detect_time_column(table.headers)
        # Confidence routing: a layout-detected table the LLM wasn't sure
        # about sends every observation it produced to the review queue.
        # Legacy (non-detector) tables carry no structural_confidence, so
        # low_conf stays False and behavior is unchanged.
        struct_conf = (
            table.locator.get("structural_confidence")
            if isinstance(table.locator, dict)
            else None
        )
        low_conf = (
            isinstance(struct_conf, (int, float))
            and struct_conf < _LAYOUT_REVIEW_THRESHOLD
        )
        # Run-id precedence (highest first):
        #   1. Operator manifest (manifest_run_id)
        #   2. DocumentMap.run_for_table(idx)        — LLM segmenter output
        #   3. RunIdResolver chain                   — column / filename / synthetic
        # Manifest wins; segmenter output is the next-best signal; resolver
        # chain handles the fallthrough. The disagreement warning between
        # (1) and (2) is emitted by DocumentSegmenter.segment().
        seg_manifest_run_id: str | None = manifest_run_id
        # Layout-detector single-run grouping pins a per-sheet run id in the
        # table locator so a multi-sheet workbook becomes multiple runs
        # instead of collapsing to one filename-derived id. Manifest still
        # wins; this slots just below it.
        if seg_manifest_run_id is None and isinstance(table.locator, dict):
            detector_run = table.locator.get("run_id")
            if detector_run:
                seg_manifest_run_id = str(detector_run)
        if seg_manifest_run_id is None and doc_map is not None and isinstance(
            table.locator, dict
        ):
            table_idx = table.locator.get("table_idx")
            if isinstance(table_idx, int):
                run_segment = doc_map.run_for_table(table_idx)
                if run_segment is not None:
                    seg_manifest_run_id = run_segment.run_id
        # Run-id resolution via strategy chain. ColumnStrategy returns
        # column_idx → per-row read; other strategies return a single value
        # used for every row.
        run_resolution = self._run_id_resolver.resolve(
            headers=table.headers,
            rows=table.rows,
            filename=table.locator.get("file") if isinstance(table.locator, dict) else None,
            manifest_run_id=seg_manifest_run_id,
        )

        # One source column per canonical variable. When several columns map
        # to the same variable (e.g. praaj's "Lactic Acid(%w/w)" and
        # "Volume Corrected Lactic Acid (g/L)" both -> product_g_l), keep the
        # one already in the canonical unit (no conversion, unambiguous) and
        # demote the rest to residual. Without this the variable holds two
        # values per timepoint and the diagnosis sees a phantom "conflict".
        demoted = _reconcile_duplicate_mappings(
            mapping.entries, self._schema_index, dominant_units
        )

        for entry in mapping.entries:
            decision = band(entry.confidence)
            if entry.raw_header in demoted:
                unmapped_columns.append(
                    {
                        "raw_header": entry.raw_header,
                        "reason": "duplicate_mapping",
                        "mapped_to": entry.mapped_to,
                        "kept": demoted[entry.raw_header],
                    }
                )
                continue
            if entry.mapped_to is None or decision == ConfidenceBand.RESIDUAL:
                unmapped_columns.append(
                    {
                        "raw_header": entry.raw_header,
                        "reason": "no_mapping" if entry.mapped_to is None else "low_confidence",
                        "confidence": entry.confidence,
                    }
                )
                continue
            golden = self._schema_index.get(entry.mapped_to)
            if golden is None:
                unmapped_columns.append(
                    {"raw_header": entry.raw_header, "reason": "unknown_canonical"}
                )
                continue
            # Metadata-only golden columns (experiment_id, strain_id, organism,
            # product) carry canonical_unit=None and describe experiment
            # identity, not per-row measurements. Mappers occasionally route
            # batch-id columns here — refuse to write per-row observations
            # against them. Identity belongs in the dossier's process layer.
            if golden.canonical_unit is None:
                unmapped_columns.append(
                    {
                        "raw_header": entry.raw_header,
                        "reason": "metadata_column_not_observation",
                        "mapped_to": entry.mapped_to,
                    }
                )
                continue
            col_idx = col_index.get(entry.raw_header)
            if col_idx is None:
                unmapped_columns.append(
                    {"raw_header": entry.raw_header, "reason": "header_not_in_table"}
                )
                continue
            for row_idx, row in enumerate(table.rows):
                if col_idx >= len(row):
                    continue
                raw_value = row[col_idx]
                if raw_value is None or raw_value == "":
                    continue
                # Missing-value markers used in lab sheets ("-", "N/A", "ND",
                # em dash, etc.) are NOT measurements. Skip them so they never
                # become observations with a non-numeric value that crashes a
                # later float() (e.g. characterize build_summary).
                if _is_missing_marker(raw_value):
                    continue
                if isinstance(raw_value, float) and math.isnan(raw_value):
                    continue
                row_run_id = _resolve_row_run_id(run_resolution, row)
                row_time_h = (
                    _coerce_time_h(row[time_col_idx])
                    if time_col_idx is not None and time_col_idx < len(row)
                    else None
                )
                observations.append(
                    self._build_observation(
                        experiment_id=experiment_id,
                        file_id=file_id,
                        table=table,
                        entry=entry,
                        golden_unit=golden.canonical_unit,
                        data_type=str(golden.data_type),
                        raw_value=raw_value,
                        row_idx=row_idx,
                        col_idx=col_idx,
                        run_id=row_run_id,
                        time_h=row_time_h,
                        decision=decision,
                        force_review=low_conf,
                    )
                )
        return observations, unmapped_columns

    def _build_observation(
        self,
        *,
        experiment_id: str,
        file_id: uuid.UUID,
        table: ParsedTable,
        entry: MappingEntry,
        golden_unit: str | None,
        data_type: str,
        raw_value: object,
        row_idx: int,
        col_idx: int,
        decision: ConfidenceBand,
        run_id: str | None = None,
        time_h: float | None = None,
        force_review: bool = False,
    ) -> Observation:
        conversion = self._converter.convert(
            raw_value, entry.raw_unit, golden_unit, normalizer=self._normalizer,
        )
        value_raw = {"value": _coerce(raw_value, data_type), "type": data_type}
        value_canonical: dict[str, Any] | None = None
        if conversion.value_canonical is not None:
            value_canonical = {
                "value": conversion.value_canonical,
                "type": data_type,
                "via": conversion.via,
            }
            if conversion.hint is not None:
                value_canonical["normalization"] = {
                    "action": conversion.hint.action.value,
                    "pint_expr": conversion.hint.pint_expr,
                    "rationale": conversion.hint.rationale,
                    "confidence": conversion.hint.confidence,
                    "source": conversion.hint.source,
                }
        locator = {**table.locator, "row": row_idx, "col": col_idx, "section": "table"}
        if run_id is not None:
            locator["run_id"] = run_id
        if time_h is not None:
            locator["timestamp_h"] = time_h
        # Value-level review routing is decision-band + layout confidence only.
        # Scale/unit sanity is judged data-relatively across runs after all
        # observations are built (see _flag_scale_outliers) -- no hardcoded
        # expected value is consulted here.
        needs_review = decision == ConfidenceBand.NEEDS_REVIEW or force_review
        return Observation(
            observation_id=uuid.uuid4(),
            experiment_id=experiment_id,
            file_id=file_id,
            column_name=entry.mapped_to or "",
            raw_header=entry.raw_header,
            observation_type=ObservationType.UNKNOWN,
            value_raw=value_raw,
            unit_raw=entry.raw_unit,
            value_canonical=value_canonical,
            unit_canonical=conversion.unit_canonical,
            conversion_status=conversion.status,
            source_locator=locator,
            mapping_confidence=entry.confidence,
            extraction_confidence=_extraction_confidence(raw_value, data_type),
            needs_review=needs_review,
            extractor_version=EXTRACTOR_VERSION,
            schema_version=self._schema.version,
            extracted_at=datetime.utcnow(),
        )


# -----------------------------------------------------------------------------
# Time-column auto-detection + run-id resolution helpers
#
# Time-column detection stays a simple header match — there's no ambiguity
# in practice (a CSV either has a clearly-named time column or it doesn't).
# Run-id resolution moved to a strategy chain (parsing/run_id_resolver.py)
# because it has many failure modes and grows over time. See that module
# for the generalization rationale.
# -----------------------------------------------------------------------------

# Time-column bases, compared AFTER stripping any unit annotation. The unit
# is carried in parens/brackets and varies widely ("Time (Hours)", "Time (h)",
# "Time [min]"), so we strip it and match the bare base. Without this, a header
# like praaj's "Time (Hours)" goes undetected -> observations get no timestamp
# -> no trajectory forms -> the long-format observations.csv comes out empty
# even though the observations were ingested.
_TIME_HEADER_BASES = frozenset(
    {
        "time",
        "time_h",
        "elapsed_time",
        "elapsed time",
        "elapsed",
        "timestamp",
        "time point",
        "sampling time",
        "duration",
        "t",
    }
)


def _time_header_base(header: str | None) -> str:
    """Lowercased header with any trailing unit annotation stripped.

    'Time (Hours)' -> 'time'; 'Time [h]' -> 'time'; 'Elapsed_Time' -> 'elapsed_time'.
    """
    norm = (header or "").strip().lower()
    base = norm.split("(", 1)[0].split("[", 1)[0]
    return base.strip().rstrip(":").strip()


def _flag_scale_outliers(observations: list[Observation]) -> None:
    """Mark runs whose canonical values for a variable are wildly off the
    cross-run median scale (suspected unit/scale problem). Data-relative: the
    only reference is the dataset's own distribution, never a hardcoded value.

    Per variable, each run's scale = the max |canonical value| in that run.
    A run whose scale exceeds _SCALE_OUTLIER_FACTOR x the median run-scale is
    flagged needs_review. Requires >= _SCALE_OUTLIER_MIN_RUNS runs so a real
    baseline exists; with fewer, there is nothing to compare against and we
    leave the data untouched (no hardcoded fallback).
    """
    import statistics
    from collections import defaultdict

    by_var: dict[str, list[Observation]] = defaultdict(list)
    for o in observations:
        vc = o.value_canonical or {}
        val = vc.get("value")
        rid = (o.source_locator or {}).get("run_id")
        if isinstance(val, (int, float)) and not isinstance(val, bool) and rid and o.column_name:
            by_var[o.column_name].append(o)

    for _var, obs_list in by_var.items():
        run_scale: dict[str, float] = defaultdict(float)
        for o in obs_list:
            rid = str(o.source_locator["run_id"])
            run_scale[rid] = max(run_scale[rid], abs(float(o.value_canonical["value"])))
        scales = [s for s in run_scale.values() if s > 0]
        if len(scales) < _SCALE_OUTLIER_MIN_RUNS:
            continue
        med = statistics.median(scales)
        if med <= 0:
            continue
        for o in obs_list:
            rid = str(o.source_locator["run_id"])
            if run_scale[rid] > _SCALE_OUTLIER_FACTOR * med:
                o.needs_review = True
                o.source_locator["sanity_flag"] = "run_scale_outlier_vs_other_runs"


def _norm_unit(unit: str | None) -> str:
    """Loose unit comparison key: lowercased, whitespace stripped."""
    return "".join((unit or "").lower().split())


def _dominant_source_units(mapping) -> dict[str, str]:
    """Per canonical variable, the source unit used by the MOST runs (tables).

    Cross-run consistency beats per-run unit preference: if 14 runs report a
    variable in %w/w and one run also has a g/L column, the dataset is only
    self-consistent if every run uses %w/w. Counted per table so one run with
    several same-unit columns can't outvote the population.
    """
    from collections import Counter

    counts: dict[str, Counter] = {}
    for tm in mapping.tables:
        per_var: dict[str, set] = {}
        for e in tm.entries:
            if e.mapped_to:
                per_var.setdefault(e.mapped_to, set()).add(_norm_unit(e.raw_unit))
        for var, units in per_var.items():
            c = counts.setdefault(var, Counter())
            for u in units:
                c[u] += 1
    return {var: c.most_common(1)[0][0] for var, c in counts.items() if c}


def _reconcile_duplicate_mappings(entries, schema_index, dominant_units=None) -> dict[str, str]:
    """When >1 column maps to the same canonical variable, pick one and report
    the rest as demoted.

    Preference (in order): the column whose raw unit matches the unit that most
    RUNS use for this variable (cross-run consistency — so a lone off-template
    column can't put one run on a different scale than the rest), then the
    canonical unit, then mapping confidence, then stable header order. Returns
    ``{demoted_raw_header: kept_raw_header}``. A variable with one column here
    produces nothing.
    """
    dominant_units = dominant_units or {}
    groups: dict[str, list] = {}
    for e in entries:
        if e.mapped_to:
            groups.setdefault(e.mapped_to, []).append(e)
    demoted: dict[str, str] = {}
    for var, es in groups.items():
        if len(es) < 2:
            continue
        col = schema_index.get(var)
        canon = _norm_unit(getattr(col, "canonical_unit", None)) if col else ""
        dom = dominant_units.get(var)

        def _rank(e, _dom=dom):
            u = _norm_unit(e.raw_unit)
            dom_match = 1 if _dom and u == _dom else 0
            unit_match = 1 if canon and u == canon else 0
            return (dom_match, unit_match, e.confidence or 0.0, -len(e.raw_header or ""))

        winner = max(es, key=_rank)
        for e in es:
            if e.raw_header != winner.raw_header:
                demoted[e.raw_header] = winner.raw_header
    return demoted


def _detect_time_column(headers: list[str]) -> int | None:
    """Return the index of the first column whose header is a time column,
    matching on the unit-stripped base so any 'Time (<unit>)' is recognized.
    """
    for i, h in enumerate(headers):
        if _time_header_base(h) in _TIME_HEADER_BASES:
            return i
    return None


def _resolve_row_run_id(
    resolution: RunIdResolution, row: list[Any]
) -> str | None:
    """Apply a RunIdResolution to one row of the table.

    When the resolution carries a column_idx (ColumnStrategy), we read the
    cell. Otherwise we use the resolution's verbatim value (manifest /
    filename / synthetic). Coercion is identical to the column path so the
    resulting run_ids are interchangeable.
    """
    if resolution.column_idx is not None:
        if resolution.column_idx >= len(row):
            return None
        return _coerce_run_id(row[resolution.column_idx])
    return resolution.value or None


# Lab-sheet missing-value markers. A cell holding one of these means "not
# measured", not a value — kept out of the observation stream so it never
# becomes a non-numeric observation.
_MISSING_MARKERS = frozenset(
    {"nan", "-", "--", "---", "—", "–", "n/a", "na", "n.a.", "nd", "n.d.", "none", "null", "#n/a"}
)


def _is_missing_marker(value: object) -> bool:
    return isinstance(value, str) and value.strip().lower() in _MISSING_MARKERS


def _coerce_time_h(value: object) -> float | None:
    """CSV cells arrive as strings; the parser already filtered None / ""."""
    if value is None or value == "":
        return None
    try:
        f = float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return None
    if math.isnan(f) or math.isinf(f):
        return None
    return f


def _coerce_run_id(value: object) -> str | None:
    if value is None or value == "":
        return None
    s = str(value).strip()
    if not s or s.lower() == "nan":
        return None
    # Common case: numeric run id like "1" or "1.0" — normalize to int form.
    try:
        f = float(s)
        if math.isnan(f):
            return None
        if f.is_integer():
            return f"RUN-{int(f):04d}"
    except (TypeError, ValueError):
        pass
    return s


def _coerce(value: object, data_type: str) -> object:
    if data_type == "float":
        try:
            return float(value)  # type: ignore[arg-type]
        except (TypeError, ValueError):
            return value
    if data_type == "int":
        try:
            return int(float(value))  # type: ignore[arg-type]
        except (TypeError, ValueError):
            return value
    if data_type == "bool":
        if isinstance(value, str):
            return value.strip().lower() in {"true", "yes", "1"}
        return bool(value)
    return str(value) if value is not None else None


def _extraction_confidence(value: object, data_type: str) -> float:
    if data_type in {"float", "int"}:
        try:
            float(value)  # type: ignore[arg-type]
            return 1.0
        except (TypeError, ValueError):
            return 0.5
    return 1.0


def _add_unmapped(residual: ResidualPayload, table: ParsedTable, reason: str) -> None:
    residual.tables_unmapped.append(
        {
            "table_id": table.table_id,
            "headers": table.headers,
            "rows": table.rows,
            "locator": table.locator,
            "reason": reason,
        }
    )


def _add_partial(
    residual: ResidualPayload,
    table: ParsedTable,
    mapping: TableMapping,
    unmapped_columns: list[dict],
) -> None:
    residual.tables_partial.append(
        {
            "table_id": table.table_id,
            "locator": table.locator,
            "unmapped_columns": unmapped_columns,
        }
    )


def ingest(experiment_id: str, files: list[Path], pipeline: IngestionPipeline) -> IngestionResult:
    """Top-level convenience for library callers (tests, pattern B integrations)."""
    return pipeline.ingest(experiment_id, files)
