from __future__ import annotations

import math
import mimetypes
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

        mapping = self._mapper.map(tables, self._schema)
        mapping_by_table = {tm.table_id: tm for tm in mapping.tables}

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
            )
            table_observations.extend(obs)
            if partial:
                _add_partial(residual, table, tm, partial)

        # Tier 1: always capture narrative blocks in residual.
        if narrative_blocks:
            residual.narrative = [b.model_dump(mode="json") for b in narrative_blocks]

        # Tier 2: optional LLM extraction over narrative blocks.
        narrative_observations, narrative_stats = self._extract_narrative_observations(
            experiment_id, file_id, narrative_blocks, table_observations
        )

        all_observations = table_observations + narrative_observations
        n_obs = self._repo.write_observations(all_observations)
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
            ext.value, ext.unit, golden.canonical_unit, normalizer=self._normalizer
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

    def _observations_for_table(
        self,
        experiment_id: str,
        file_id: uuid.UUID,
        table: ParsedTable,
        mapping: TableMapping,
        *,
        doc_map: Any | None = None,
        manifest_run_id: str | None = None,
    ) -> tuple[list[Observation], list[dict]]:
        observations: list[Observation] = []
        unmapped_columns: list[dict] = []
        col_index = {h: i for i, h in enumerate(table.headers)}
        time_col_idx = _detect_time_column(table.headers)
        # Run-id precedence (highest first):
        #   1. Operator manifest (manifest_run_id)
        #   2. DocumentMap.run_for_table(idx)        — LLM segmenter output
        #   3. RunIdResolver chain                   — column / filename / synthetic
        # Manifest wins; segmenter output is the next-best signal; resolver
        # chain handles the fallthrough. The disagreement warning between
        # (1) and (2) is emitted by DocumentSegmenter.segment().
        seg_manifest_run_id: str | None = manifest_run_id
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

        for entry in mapping.entries:
            decision = band(entry.confidence)
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
                # Filter NaN sentinels so they never reach JSONB (Postgres
                # JSON spec forbids NaN). Common in offline-measurement
                # columns where most timestamps lack a sample.
                if isinstance(raw_value, str) and raw_value.strip().lower() == "nan":
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
    ) -> Observation:
        conversion = self._converter.convert(
            raw_value, entry.raw_unit, golden_unit, normalizer=self._normalizer
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
            needs_review=(decision == ConfidenceBand.NEEDS_REVIEW),
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

_TIME_HEADER_PATTERNS = (
    "time",
    "time (h)",
    "time(h)",
    "time_h",
    "time [h]",
    "elapsed_time",
    "elapsed time",
    "t",
)


def _detect_time_column(headers: list[str]) -> int | None:
    """Return the index of the first column whose header matches a time
    pattern, or None.
    """
    for i, h in enumerate(headers):
        norm = (h or "").strip().lower()
        if norm in _TIME_HEADER_PATTERNS:
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
