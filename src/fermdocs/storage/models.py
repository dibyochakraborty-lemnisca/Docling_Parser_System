from __future__ import annotations

import uuid
from datetime import datetime
from typing import Any

from sqlalchemy import (
    BigInteger,
    Boolean,
    ForeignKey,
    Index,
    Integer,
    Numeric,
    String,
    UniqueConstraint,
    text,
)
from sqlalchemy.dialects.postgresql import JSONB, TIMESTAMP
from sqlalchemy.dialects.postgresql import UUID as PG_UUID
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column


class Base(DeclarativeBase):
    pass


class ExperimentRow(Base):
    __tablename__ = "experiments"

    experiment_id: Mapped[str] = mapped_column(String, primary_key=True)
    name: Mapped[str | None] = mapped_column(String)
    uploaded_by: Mapped[str | None] = mapped_column(String)
    created_at: Mapped[datetime] = mapped_column(
        TIMESTAMP(timezone=True), server_default=text("now()")
    )
    status: Mapped[str] = mapped_column(String, default="ingesting")
    notes: Mapped[str | None] = mapped_column(String)


class SourceFileRow(Base):
    __tablename__ = "source_files"

    file_id: Mapped[uuid.UUID] = mapped_column(PG_UUID(as_uuid=True), primary_key=True)
    experiment_id: Mapped[str] = mapped_column(
        String, ForeignKey("experiments.experiment_id"), nullable=False
    )
    filename: Mapped[str] = mapped_column(String, nullable=False)
    sha256: Mapped[str] = mapped_column(String, nullable=False)
    mime_type: Mapped[str | None] = mapped_column(String)
    size_bytes: Mapped[int | None] = mapped_column(BigInteger)
    page_count: Mapped[int | None] = mapped_column(Integer)
    storage_path: Mapped[str] = mapped_column(String, nullable=False)
    parsed_at: Mapped[datetime | None] = mapped_column(TIMESTAMP(timezone=True))
    parse_status: Mapped[str] = mapped_column(String, default="pending")
    parse_error: Mapped[str | None] = mapped_column(String)

    __table_args__ = (UniqueConstraint("experiment_id", "sha256", name="uq_file_dedup"),)


class ObservationRow(Base):
    __tablename__ = "golden_observations"

    observation_id: Mapped[uuid.UUID] = mapped_column(
        PG_UUID(as_uuid=True), primary_key=True, default=uuid.uuid4
    )
    experiment_id: Mapped[str] = mapped_column(
        String, ForeignKey("experiments.experiment_id"), nullable=False
    )
    file_id: Mapped[uuid.UUID] = mapped_column(
        PG_UUID(as_uuid=True), ForeignKey("source_files.file_id"), nullable=False
    )
    column_name: Mapped[str] = mapped_column(String, nullable=False)
    raw_header: Mapped[str] = mapped_column(String, nullable=False)
    observation_type: Mapped[str] = mapped_column(String, default="unknown")
    value_raw: Mapped[dict[str, Any]] = mapped_column(JSONB, nullable=False)
    unit_raw: Mapped[str | None] = mapped_column(String)
    value_canonical: Mapped[dict[str, Any] | None] = mapped_column(JSONB)
    unit_canonical: Mapped[str | None] = mapped_column(String)
    conversion_status: Mapped[str] = mapped_column(String, default="not_applicable")
    source_locator: Mapped[dict[str, Any]] = mapped_column(JSONB, nullable=False)
    mapping_confidence: Mapped[float | None] = mapped_column(Numeric)
    extraction_confidence: Mapped[float | None] = mapped_column(Numeric)
    needs_review: Mapped[bool] = mapped_column(Boolean, default=False)
    extractor_version: Mapped[str] = mapped_column(String, nullable=False)
    schema_version: Mapped[str | None] = mapped_column(String)
    superseded_by: Mapped[uuid.UUID | None] = mapped_column(
        PG_UUID(as_uuid=True), ForeignKey("golden_observations.observation_id")
    )
    extracted_at: Mapped[datetime] = mapped_column(
        TIMESTAMP(timezone=True), server_default=text("now()")
    )

    __table_args__ = (
        Index("idx_obs_exp_col", "experiment_id", "column_name"),
        Index(
            "idx_obs_review", "needs_review", postgresql_where=text("needs_review")
        ),
        Index(
            "idx_obs_active",
            "experiment_id",
            postgresql_where=text("superseded_by IS NULL"),
        ),
    )


class ResidualRow(Base):
    __tablename__ = "residual_data"

    residual_id: Mapped[uuid.UUID] = mapped_column(
        PG_UUID(as_uuid=True), primary_key=True, default=uuid.uuid4
    )
    file_id: Mapped[uuid.UUID] = mapped_column(
        PG_UUID(as_uuid=True), ForeignKey("source_files.file_id"), nullable=False
    )
    experiment_id: Mapped[str] = mapped_column(
        String, ForeignKey("experiments.experiment_id"), nullable=False
    )
    extractor_version: Mapped[str] = mapped_column(String, nullable=False)
    payload: Mapped[dict[str, Any]] = mapped_column(JSONB, nullable=False)
    superseded_by: Mapped[uuid.UUID | None] = mapped_column(
        PG_UUID(as_uuid=True), ForeignKey("residual_data.residual_id")
    )
    created_at: Mapped[datetime] = mapped_column(
        TIMESTAMP(timezone=True), server_default=text("now()")
    )

    __table_args__ = (Index("idx_residual_file", "file_id"),)
