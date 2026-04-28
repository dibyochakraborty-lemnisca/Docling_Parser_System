from __future__ import annotations

from datetime import datetime
from enum import Enum
from typing import Any, Literal
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field


class DataType(str, Enum):
    FLOAT = "float"
    INT = "int"
    TEXT = "text"
    BOOL = "bool"


class ObservationType(str, Enum):
    PLANNED = "planned"
    MEASURED = "measured"
    REPORTED = "reported"
    DERIVED = "derived"
    UNKNOWN = "unknown"


class ConversionStatus(str, Enum):
    OK = "ok"
    FAILED = "failed"
    NOT_APPLICABLE = "not_applicable"


class GoldenColumnExample(BaseModel):
    raw_header: str
    confidence: float | None = None


class GoldenColumn(BaseModel):
    name: str
    description: str
    data_type: DataType
    canonical_unit: str | None = None
    synonyms: list[str] = Field(default_factory=list)
    observation_types: list[ObservationType] = Field(default_factory=list)
    examples: list[GoldenColumnExample] = Field(default_factory=list)


class GoldenSchema(BaseModel):
    version: str
    columns: list[GoldenColumn]

    def by_name(self) -> dict[str, GoldenColumn]:
        return {c.name: c for c in self.columns}


class ParsedTable(BaseModel):
    """A table extracted from any source file. Locator is format-agnostic."""

    table_id: str
    headers: list[str]
    rows: list[list[Any]]
    locator: dict[str, Any]

    def sample_rows(self, n: int = 3) -> list[list[Any]]:
        return self.rows[:n]


class MappingEntry(BaseModel):
    raw_header: str
    mapped_to: str | None
    raw_unit: str | None = None
    confidence: float = Field(ge=0.0, le=1.0)
    rationale: str | None = None


class TableMapping(BaseModel):
    table_id: str
    entries: list[MappingEntry]


class MappingResult(BaseModel):
    """Output of one mapper call covering one or more tables."""

    tables: list[TableMapping]


class ConfidenceBand(str, Enum):
    AUTO = "auto"
    NEEDS_REVIEW = "needs_review"
    RESIDUAL = "residual"


class Observation(BaseModel):
    """A single value extracted from a source. Polymorphic via JSONB value envelopes."""

    model_config = ConfigDict(use_enum_values=True)

    observation_id: UUID
    experiment_id: str
    file_id: UUID
    column_name: str
    raw_header: str
    observation_type: ObservationType = ObservationType.UNKNOWN
    value_raw: dict[str, Any]
    unit_raw: str | None = None
    value_canonical: dict[str, Any] | None = None
    unit_canonical: str | None = None
    conversion_status: ConversionStatus = ConversionStatus.NOT_APPLICABLE
    source_locator: dict[str, Any]
    mapping_confidence: float | None = None
    extraction_confidence: float | None = None
    needs_review: bool = False
    extractor_version: str
    superseded_by: UUID | None = None
    extracted_at: datetime

    def to_dossier_observation(self) -> dict[str, Any]:
        """Project to the dossier JSON shape. Single source of truth for serialization."""
        combined = (
            (self.mapping_confidence or 0.0) * (self.extraction_confidence or 0.0)
            if self.mapping_confidence is not None and self.extraction_confidence is not None
            else None
        )
        return {
            "observation_id": str(self.observation_id),
            "value": (self.value_canonical or self.value_raw).get("value"),
            "unit": self.unit_canonical or self.unit_raw,
            "value_raw": self.value_raw.get("value"),
            "unit_raw": self.unit_raw,
            "observation_type": self.observation_type,
            "confidence": {
                "mapping": self.mapping_confidence,
                "extraction": self.extraction_confidence,
                "combined": combined,
            },
            "needs_review": self.needs_review,
            "source": {
                "file_id": str(self.file_id),
                "raw_header": self.raw_header,
                "locator": self.source_locator,
            },
            "conversion_status": self.conversion_status,
            "extractor_version": self.extractor_version,
        }


class ResidualPayload(BaseModel):
    tables_unmapped: list[dict[str, Any]] = Field(default_factory=list)
    tables_partial: list[dict[str, Any]] = Field(default_factory=list)
    narrative: list[dict[str, Any]] = Field(default_factory=list)
    figures: list[dict[str, Any]] = Field(default_factory=list)
    lists: list[dict[str, Any]] = Field(default_factory=list)
    raw_blocks: list[dict[str, Any]] = Field(default_factory=list)


class IngestionFileResult(BaseModel):
    file_id: UUID
    filename: str
    parse_status: Literal["ok", "failed"]
    parse_error: str | None = None
    observations_written: int = 0
    residuals_written: int = 0


class IngestionResult(BaseModel):
    experiment_id: str
    files: list[IngestionFileResult]

    @property
    def all_ok(self) -> bool:
        return all(f.parse_status == "ok" for f in self.files)

    @property
    def any_ok(self) -> bool:
        return any(f.parse_status == "ok" for f in self.files)
