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
from fermdocs.parsing.router import FormatRouter
from fermdocs.storage.repository import FileRecord, Repository
from fermdocs.units.converter import UnitConverter
from fermdocs.units.normalizer import UnitNormalizer

EXTRACTOR_VERSION = "v0.1.0"


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

    def ingest(self, experiment_id: str, files: list[Path]) -> IngestionResult:
        self._repo.upsert_experiment(experiment_id)
        results: list[IngestionFileResult] = []
        for path in files:
            results.append(self._ingest_one(experiment_id, path))
        return IngestionResult(experiment_id=experiment_id, files=results)

    def _ingest_one(self, experiment_id: str, path: Path) -> IngestionFileResult:
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

        mapping = self._mapper.map(tables, self._schema)
        mapping_by_table = {tm.table_id: tm for tm in mapping.tables}

        table_observations: list[Observation] = []
        residual = ResidualPayload()

        for table in tables:
            tm = mapping_by_table.get(table.table_id)
            if tm is None:
                _add_unmapped(residual, table, reason="no_mapping_returned")
                continue
            obs, partial = self._observations_for_table(
                experiment_id, file_id, table, tm
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
    ) -> tuple[list[Observation], list[dict]]:
        observations: list[Observation] = []
        unmapped_columns: list[dict] = []
        col_index = {h: i for i, h in enumerate(table.headers)}
        time_col_idx, run_col_idx = _detect_time_and_run_columns(table.headers)

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
                row_run_id = (
                    _coerce_run_id(row[run_col_idx])
                    if run_col_idx is not None and run_col_idx < len(row)
                    else None
                )
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
# Time / run-id auto-detection
#
# Real fermentation CSVs almost always carry one column for elapsed time and
# one for the batch / run identifier. The characterization stage reads
# locator.run_id and locator.timestamp_h off every observation; without those
# fields, every row is dropped from the summary. We auto-detect once per
# table by matching the headers against a small allow-list. CLI overrides
# come later if a real file ever needs disambiguation.
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
_RUN_HEADER_PATTERNS = (
    "batch_ref",
    "batch",
    "batch_id",
    "run",
    "run_id",
    "run_ref",
)


def _detect_time_and_run_columns(
    headers: list[str],
) -> tuple[int | None, int | None]:
    """Return (time_col_idx, run_col_idx) by matching header strings.

    Case-insensitive, whitespace-trimmed. First match wins; duplicate columns
    (e.g. IndPenSim's 'Batch_ref' / 'Batch_ref.1') resolve to the first hit.
    """
    time_idx: int | None = None
    run_idx: int | None = None
    for i, h in enumerate(headers):
        norm = (h or "").strip().lower()
        if time_idx is None and norm in _TIME_HEADER_PATTERNS:
            time_idx = i
        if run_idx is None and norm in _RUN_HEADER_PATTERNS:
            run_idx = i
    return time_idx, run_idx


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
