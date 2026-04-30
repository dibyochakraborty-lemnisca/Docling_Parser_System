from __future__ import annotations

import uuid
from dataclasses import dataclass
from datetime import datetime
from typing import Any

from sqlalchemy import select
from sqlalchemy.dialects.postgresql import insert as pg_insert
from sqlalchemy.engine import Engine
from sqlalchemy.orm import Session

from fermdocs.domain.models import IngestionFileResult, Observation, ResidualPayload
from fermdocs.storage.models import (
    Base,
    ExperimentRow,
    ObservationRow,
    ResidualRow,
    SourceFileRow,
)


@dataclass
class FileRecord:
    file_id: uuid.UUID
    experiment_id: str
    filename: str
    sha256: str
    storage_path: str
    size_bytes: int | None = None
    mime_type: str | None = None
    page_count: int | None = None


class Repository:
    """Single seam between domain and Postgres. Nothing else imports SQLAlchemy."""

    def __init__(self, engine: Engine):
        self._engine = engine

    def create_all(self) -> None:
        Base.metadata.create_all(self._engine)

    def upsert_experiment(
        self, experiment_id: str, name: str | None = None, uploaded_by: str | None = None
    ) -> None:
        with Session(self._engine) as session:
            stmt = pg_insert(ExperimentRow).values(
                experiment_id=experiment_id, name=name, uploaded_by=uploaded_by
            )
            stmt = stmt.on_conflict_do_nothing(index_elements=["experiment_id"])
            session.execute(stmt)
            session.commit()

    def find_or_create_file(self, record: FileRecord) -> tuple[uuid.UUID, bool]:
        """Return (file_id, created). Idempotent on (experiment_id, sha256)."""
        with Session(self._engine) as session:
            existing = session.execute(
                select(SourceFileRow).where(
                    SourceFileRow.experiment_id == record.experiment_id,
                    SourceFileRow.sha256 == record.sha256,
                )
            ).scalar_one_or_none()
            if existing is not None:
                return existing.file_id, False
            row = SourceFileRow(
                file_id=record.file_id,
                experiment_id=record.experiment_id,
                filename=record.filename,
                sha256=record.sha256,
                mime_type=record.mime_type,
                size_bytes=record.size_bytes,
                page_count=record.page_count,
                storage_path=record.storage_path,
            )
            session.add(row)
            session.commit()
            return row.file_id, True

    def mark_file_parsed(
        self,
        file_id: uuid.UUID,
        status: str,
        error: str | None = None,
        parsed_at: datetime | None = None,
    ) -> None:
        with Session(self._engine) as session:
            row = session.get(SourceFileRow, file_id)
            if row is None:
                return
            row.parse_status = status
            row.parse_error = error
            row.parsed_at = parsed_at or datetime.utcnow()
            session.commit()

    def write_observations(self, observations: list[Observation]) -> int:
        if not observations:
            return 0
        with Session(self._engine) as session:
            session.add_all(
                [
                    ObservationRow(
                        observation_id=o.observation_id,
                        experiment_id=o.experiment_id,
                        file_id=o.file_id,
                        column_name=o.column_name,
                        raw_header=o.raw_header,
                        observation_type=str(o.observation_type),
                        value_raw=o.value_raw,
                        unit_raw=o.unit_raw,
                        value_canonical=o.value_canonical,
                        unit_canonical=o.unit_canonical,
                        conversion_status=str(o.conversion_status),
                        source_locator=o.source_locator,
                        mapping_confidence=o.mapping_confidence,
                        extraction_confidence=o.extraction_confidence,
                        needs_review=o.needs_review,
                        extractor_version=o.extractor_version,
                        schema_version=o.schema_version,
                    )
                    for o in observations
                ]
            )
            session.commit()
        return len(observations)

    def write_residual(
        self,
        file_id: uuid.UUID,
        experiment_id: str,
        payload: ResidualPayload,
        extractor_version: str,
    ) -> uuid.UUID:
        with Session(self._engine) as session:
            row = ResidualRow(
                residual_id=uuid.uuid4(),
                file_id=file_id,
                experiment_id=experiment_id,
                extractor_version=extractor_version,
                payload=payload.model_dump(),
            )
            session.add(row)
            session.commit()
            return row.residual_id

    def fetch_experiment(self, experiment_id: str) -> ExperimentRow | None:
        with Session(self._engine) as session:
            return session.get(ExperimentRow, experiment_id)

    def fetch_files(self, experiment_id: str) -> list[SourceFileRow]:
        with Session(self._engine) as session:
            return list(
                session.execute(
                    select(SourceFileRow).where(SourceFileRow.experiment_id == experiment_id)
                ).scalars()
            )

    def fetch_active_observations(self, experiment_id: str) -> list[ObservationRow]:
        with Session(self._engine) as session:
            return list(
                session.execute(
                    select(ObservationRow).where(
                        ObservationRow.experiment_id == experiment_id,
                        ObservationRow.superseded_by.is_(None),
                    )
                ).scalars()
            )

    def fetch_residuals(self, experiment_id: str) -> list[ResidualRow]:
        with Session(self._engine) as session:
            return list(
                session.execute(
                    select(ResidualRow).where(
                        ResidualRow.experiment_id == experiment_id,
                        ResidualRow.superseded_by.is_(None),
                    )
                ).scalars()
            )

    def next_review_observation(self) -> ObservationRow | None:
        with Session(self._engine) as session:
            return session.execute(
                select(ObservationRow)
                .where(
                    ObservationRow.needs_review.is_(True),
                    ObservationRow.superseded_by.is_(None),
                )
                .order_by(ObservationRow.extracted_at.asc())
                .limit(1)
            ).scalar_one_or_none()

    def row_to_observation(self, row: ObservationRow) -> Observation:
        return Observation.model_validate(
            {
                "observation_id": row.observation_id,
                "experiment_id": row.experiment_id,
                "file_id": row.file_id,
                "column_name": row.column_name,
                "raw_header": row.raw_header,
                "observation_type": row.observation_type,
                "value_raw": row.value_raw,
                "unit_raw": row.unit_raw,
                "value_canonical": row.value_canonical,
                "unit_canonical": row.unit_canonical,
                "conversion_status": row.conversion_status,
                "source_locator": row.source_locator,
                "mapping_confidence": float(row.mapping_confidence)
                if row.mapping_confidence is not None
                else None,
                "extraction_confidence": float(row.extraction_confidence)
                if row.extraction_confidence is not None
                else None,
                "needs_review": row.needs_review,
                "extractor_version": row.extractor_version,
                "schema_version": row.schema_version,
                "superseded_by": row.superseded_by,
                "extracted_at": row.extracted_at,
            }
        )
