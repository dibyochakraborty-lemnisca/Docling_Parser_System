from __future__ import annotations

from datetime import datetime
from enum import Enum
from typing import Any, Literal
from uuid import UUID

import math

from pydantic import BaseModel, ConfigDict, Field, field_validator


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


class IdentityProvenance(str, Enum):
    """How a ProcessIdentity was determined.

    Downstream agents reason about a violation differently depending on
    whether the identity came from an operator-supplied manifest (high
    trust), an LLM extraction validated against a closed registry (medium
    trust, capped at 0.85), or no identification at all.
    """

    MANIFEST = "manifest"
    LLM_WHITELISTED = "llm_whitelisted"
    UNKNOWN = "unknown"


class EvidenceLocator(BaseModel):
    """A pointer to the exact source span supporting a claim.

    Carries enough information for a future facts_graph builder to create
    a Source node and a `derived_from` edge from the cited Sample. Plain
    strings would force every consumer to re-derive provenance.
    """

    file_id: str
    paragraph_idx: int
    span_text: str = Field(max_length=200)
    span_start: int | None = None  # char offset within the paragraph; optional


class ScaleInfo(BaseModel):
    volume_l: float | None = None
    vessel_type: str | None = None


class ObservedFacts(BaseModel):
    """Surface facts extracted from prose. Populated whenever the LLM finds
    them in the source documents, regardless of whether the process matches
    the registry.

    Yeast experiment? `organism="Saccharomyces cerevisiae"` shows up here
    even though no yeast process is registered. This layer carries what
    the agents *can* know directly from the paper without a recipe match.

    Substring-evidence verification applies to every populated field.
    Confidence is capped at 0.85 (LLM_CONFIDENCE_CAP).
    """

    organism: str | None = None
    product: str | None = None
    process_family_hint: str | None = None  # "fed-batch" / "batch" / "perfusion"
    scale: ScaleInfo | None = None
    confidence: float = Field(ge=0.0, le=1.0, default=0.0)
    provenance: IdentityProvenance = IdentityProvenance.UNKNOWN
    evidence_locators: list[EvidenceLocator] = Field(
        default_factory=list, max_length=5
    )
    rationale: str | None = None


class RegisteredProcess(BaseModel):
    """Registry classification. Only populated on registry hit + fingerprint
    pass. Stays UNKNOWN for processes outside the registry.

    Failure of this layer does NOT nullify ObservedFacts: agents still see
    the organism even when the recipe is unknown.
    """

    process_id: str | None = None  # registry id, e.g. "penicillin_indpensim"
    confidence: float = Field(ge=0.0, le=1.0, default=0.0)
    provenance: IdentityProvenance = IdentityProvenance.UNKNOWN
    rationale: str | None = None


class ProcessIdentity(BaseModel):
    """Per-experiment identity carrying both layers.

    Downstream agents read both:
      - process.observed.organism: usually present even on non-registered runs
      - process.registered.process_id: only present on registry hit
    """

    observed: ObservedFacts = Field(default_factory=ObservedFacts)
    registered: RegisteredProcess = Field(default_factory=RegisteredProcess)


class GoldenColumnExample(BaseModel):
    raw_header: str
    confidence: float | None = None


class GoldenColumn(BaseModel):
    name: str
    description: str
    data_type: DataType
    canonical_unit: str | None = None
    nominal: float | None = None
    std_dev: float | None = None
    synonyms: list[str] = Field(default_factory=list)
    observation_types: list[ObservationType] = Field(default_factory=list)
    examples: list[GoldenColumnExample] = Field(default_factory=list)

    @field_validator("nominal", "std_dev")
    @classmethod
    def _finite(cls, v: float | None) -> float | None:
        if v is not None and not math.isfinite(v):
            raise ValueError("must be a finite number (got inf or nan)")
        return v

    @field_validator("std_dev")
    @classmethod
    def _non_negative(cls, v: float | None) -> float | None:
        if v is not None and v < 0:
            raise ValueError("std_dev must be >= 0")
        return v


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


class NarrativeBlockType(str, Enum):
    PARAGRAPH = "paragraph"
    HEADING = "heading"
    LIST_ITEM = "list_item"
    CAPTION = "caption"
    OTHER = "other"


class NarrativeBlock(BaseModel):
    """A non-table block of text extracted from a document (paragraph, heading, etc)."""

    text: str
    type: NarrativeBlockType = NarrativeBlockType.PARAGRAPH
    locator: dict[str, Any]


class NarrativeExtraction(BaseModel):
    """Output of LLMNarrativeExtractor: a candidate observation pulled from prose.

    Subject to evidence verification + dedup before becoming an Observation.
    """

    column: str
    value: Any
    unit: str | None = None
    evidence: str
    source_paragraph_idx: int
    confidence: float = Field(ge=0.0, le=1.0)
    rationale: str | None = None


class ParseResult(BaseModel):
    """What every FileParser returns. Tables always; narrative blocks for PDFs.

    `feed_plan_tables` carries operator-supplied feeding-strategy tables
    (Segment | Batch hours | Feed rate) that were detected during PDF parse.
    These are kept OUT of `tables` so they never enter the observation
    stream as fake measurements; the pipeline stashes them into
    residual.process_recipe instead. CSV / Excel parsers leave the field
    empty.
    """

    tables: list[ParsedTable] = Field(default_factory=list)
    narrative_blocks: list["NarrativeBlock"] = Field(default_factory=list)
    feed_plan_tables: list[ParsedTable] = Field(default_factory=list)


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
    schema_version: str | None = None
    superseded_by: UUID | None = None
    extracted_at: datetime

    def to_dossier_observation(self) -> dict[str, Any]:
        """Project to the dossier JSON shape. Single source of truth for serialization.

        Backward-compatible: pre-normalizer observations have value_canonical without
        a 'via' field. The .get() calls below tolerate missing keys.
        """
        combined = (
            (self.mapping_confidence or 0.0) * (self.extraction_confidence or 0.0)
            if self.mapping_confidence is not None and self.extraction_confidence is not None
            else None
        )
        canonical = self.value_canonical or {}
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
            "via": canonical.get("via"),
            "normalization": canonical.get("normalization"),
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
    narrative_blocks_captured: int = 0
    narrative_extractions_kept: int = 0
    narrative_extractions_rejected: int = 0
    narrative_extractions_deduped: int = 0


class IngestionResult(BaseModel):
    experiment_id: str
    files: list[IngestionFileResult]

    @property
    def all_ok(self) -> bool:
        return all(f.parse_status == "ok" for f in self.files)

    @property
    def any_ok(self) -> bool:
        return any(f.parse_status == "ok" for f in self.files)


# -----------------------------------------------------------------------------
# Document segmentation (PDF only)
#
# Plan ref: docs/design/2026-05-03-pdf-document-segmentation.md
#
# A DocumentMap labels each TableItem in a PDF with the experimental run it
# belongs to. Produced by an LLM segmenter that reads the document outline
# (headings, first-line previews of text blocks, table positions). Consumed
# by IngestionPipeline as a per-table manifest_run_id override.
#
# CSV / Excel inputs do not produce DocumentMaps — the existing column /
# filename / synthetic resolver chain stays unchanged for those paths.
# -----------------------------------------------------------------------------


class RunSegmentSource(str, Enum):
    """How the LLM identified a run boundary.

    Tracked for debugging and confidence calibration. A run grounded in a
    SectionHeaderItem (`section_header`) is more trustworthy than one
    inferred from prose patterns (`text_pattern`) or pure positional
    inference (`inferred`).
    """

    SECTION_HEADER = "section_header"
    TEXT_PATTERN = "text_pattern"
    INFERRED = "inferred"


class RunSegment(BaseModel):
    """One experimental run identified by the document segmenter.

    `table_indices` are the `table_idx` values (0-based, matching
    ParsedTable.locator["table_idx"]) of TableItems belonging to this run.
    Composition tables, feed-plan tables, and any table the LLM declines
    to assign are NOT listed here — they end up in DocumentMap.
    unassigned_table_indices.
    """

    model_config = ConfigDict(frozen=True)

    run_id: str = Field(min_length=1)
    display_name: str = Field(min_length=1)
    table_indices: list[int]
    source_signal: RunSegmentSource
    confidence: float = Field(ge=0.0, le=1.0)
    rationale: str = ""

    @field_validator("table_indices")
    @classmethod
    def _table_indices_unique_nonneg(cls, v: list[int]) -> list[int]:
        if any(i < 0 for i in v):
            raise ValueError("table_indices must be non-negative")
        if len(v) != len(set(v)):
            raise ValueError("table_indices must be unique within a run")
        return v


class DocumentMap(BaseModel):
    """LLM-produced map from a PDF's TableItems to experimental runs.

    Validation invariant: a given table_idx appears in at most one run's
    `table_indices`, AND is never simultaneously in `unassigned_table_indices`.
    Violation indicates an LLM bug or a malformed structured-output response;
    the DocumentSegmenter rejects such maps and falls through to the
    existing resolver chain.
    """

    model_config = ConfigDict(frozen=True)

    schema_version: Literal["1.0"] = "1.0"
    file_id: str = Field(min_length=1)
    runs: list[RunSegment]
    unassigned_table_indices: list[int] = Field(default_factory=list)
    overall_confidence: float = Field(ge=0.0, le=1.0)
    llm_model: str
    llm_provider: str

    @field_validator("unassigned_table_indices")
    @classmethod
    def _unassigned_unique_nonneg(cls, v: list[int]) -> list[int]:
        if any(i < 0 for i in v):
            raise ValueError("unassigned_table_indices must be non-negative")
        if len(v) != len(set(v)):
            raise ValueError("unassigned_table_indices must be unique")
        return v

    @field_validator("runs")
    @classmethod
    def _runs_have_unique_run_ids(cls, v: list[RunSegment]) -> list[RunSegment]:
        ids = [r.run_id for r in v]
        if len(ids) != len(set(ids)):
            raise ValueError("DocumentMap runs must have unique run_ids")
        return v

    def model_post_init(self, __context: Any) -> None:
        # Cross-field invariant: no table_idx appears in two runs OR in both
        # an assigned run and the unassigned list. Pydantic v2 cross-field
        # validation lives here (field validators only see one field at a time).
        seen: set[int] = set()
        for run in self.runs:
            for idx in run.table_indices:
                if idx in seen:
                    raise ValueError(
                        f"table_idx {idx} appears in multiple runs;"
                        " each table belongs to exactly one run"
                    )
                seen.add(idx)
        for idx in self.unassigned_table_indices:
            if idx in seen:
                raise ValueError(
                    f"table_idx {idx} is both assigned to a run AND listed"
                    f" as unassigned; document map is contradictory"
                )

    def run_for_table(self, table_idx: int) -> RunSegment | None:
        """Return the RunSegment containing this table_idx, or None.

        Used by IngestionPipeline to look up the run for each parsed table.
        Linear in number of runs; that's fine — typical PDFs have <20 runs.
        """
        for run in self.runs:
            if table_idx in run.table_indices:
                return run
        return None
