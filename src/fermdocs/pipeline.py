from __future__ import annotations

import mimetypes
import uuid
from datetime import datetime
from pathlib import Path

from fermdocs.domain.golden_schema import load_schema
from fermdocs.domain.models import (
    ConfidenceBand,
    GoldenSchema,
    IngestionFileResult,
    IngestionResult,
    MappingEntry,
    Observation,
    ObservationType,
    ParsedTable,
    ResidualPayload,
    TableMapping,
)
from fermdocs.file_store.base import FileStore
from fermdocs.mapping.confidence import band
from fermdocs.mapping.mapper import HeaderMapper
from fermdocs.parsing.router import FormatRouter
from fermdocs.storage.repository import FileRecord, Repository
from fermdocs.units.converter import UnitConverter

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
    ):
        self._router = router
        self._mapper = mapper
        self._converter = unit_converter
        self._repo = repository
        self._files = file_store
        self._schema = schema or load_schema()
        self._schema_index = self._schema.by_name()

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
            tables = self._router.parse(path)
        except Exception as e:
            self._repo.mark_file_parsed(file_id, "failed", str(e))
            return IngestionFileResult(
                file_id=file_id, filename=path.name, parse_status="failed", parse_error=str(e)
            )

        mapping = self._mapper.map(tables, self._schema)
        mapping_by_table = {tm.table_id: tm for tm in mapping.tables}

        observations: list[Observation] = []
        residual = ResidualPayload()

        for table in tables:
            tm = mapping_by_table.get(table.table_id)
            if tm is None:
                _add_unmapped(residual, table, reason="no_mapping_returned")
                continue
            obs, partial = self._observations_for_table(
                experiment_id, file_id, table, tm
            )
            observations.extend(obs)
            if partial:
                _add_partial(residual, table, tm, partial)

        n_obs = self._repo.write_observations(observations)
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
    ) -> Observation:
        conversion = self._converter.convert(raw_value, entry.raw_unit, golden_unit)
        value_raw = {"value": _coerce(raw_value, data_type), "type": data_type}
        value_canonical = (
            {"value": conversion.value_canonical, "type": data_type}
            if conversion.value_canonical is not None
            else None
        )
        locator = {**table.locator, "row": row_idx, "col": col_idx}
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
            extracted_at=datetime.utcnow(),
        )


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
