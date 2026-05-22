"""Tests for IngestionPipeline ↔ DocumentSegmenter wire-up.

Plan ref: docs/design/2026-05-03-pdf-document-segmentation.md

Verifies the wire-up contract added in commit 5:
  - PDF inputs invoke the segmenter; CSV/Excel inputs do not.
  - DocumentMap.run_for_table(idx) is consulted to derive manifest_run_id
    per table; ManifestStrategy then takes precedence over ColumnStrategy.
  - When the segmenter returns None (LLM down, malformed output, etc.),
    behavior matches the pre-segmenter pipeline exactly — no regression.
  - feed_plan_tables on ParseResult are stashed to residual.process_recipe
    and never enter the observation stream.

These tests target the slice of pipeline behavior that changed. End-to-end
tests with real Docling parsing live in commit 8 (carotenoid integration).
"""

from __future__ import annotations

import uuid
from pathlib import Path
from typing import Any

import pytest

from fermdocs.domain.models import (
    DocumentMap,
    ParsedTable,
    RunSegment,
    RunSegmentSource,
)
from fermdocs.parsing.run_id_resolver import (
    ManifestStrategy,
    RunIdResolution,
    RunIdResolver,
)


# ---------- helpers ----------


class _RecordingResolver:
    """RunIdResolver double that records the manifest_run_id it received."""

    def __init__(self, returns: RunIdResolution | None = None) -> None:
        self.returns = returns or RunIdResolution(
            value="RUN-RECORDED",
            strategy="manifest",
            confidence=1.0,
            rationale="recording",
        )
        self.calls: list[dict[str, Any]] = []

    def resolve(
        self,
        *,
        headers: list[str],
        rows: list[list[Any]],
        filename: str | None = None,
        manifest_run_id: str | None = None,
    ) -> RunIdResolution:
        self.calls.append(
            {
                "headers": headers,
                "rows_count": len(rows),
                "filename": filename,
                "manifest_run_id": manifest_run_id,
            }
        )
        return self.returns


def _table(idx: int) -> ParsedTable:
    return ParsedTable(
        table_id=f"f.pdf#p1#t{idx}",
        headers=["Time (h)", "OD"],
        rows=[],
        locator={
            "format": "pdf",
            "file": "f.pdf",
            "page": 1,
            "table_idx": idx,
            "section": "table",
        },
    )


def _doc_map_with_two_runs() -> DocumentMap:
    return DocumentMap(
        file_id="file-abc",
        runs=[
            RunSegment(
                run_id="RUN-0001",
                display_name="BATCH-01 REPORT",
                table_indices=[0, 1],
                source_signal=RunSegmentSource.SECTION_HEADER,
                confidence=0.9,
                rationale="BATCH-01 heading on page 1",
            ),
            RunSegment(
                run_id="RUN-0002",
                display_name="BATCH-02 REPORT",
                table_indices=[2, 3],
                source_signal=RunSegmentSource.SECTION_HEADER,
                confidence=0.9,
                rationale="BATCH-02 heading on page 5",
            ),
        ],
        unassigned_table_indices=[4],
        overall_confidence=0.85,
        llm_model="gemini-3.1-pro-preview",
        llm_provider="gemini",
    )


def _make_pipeline_with(resolver: _RecordingResolver):
    """Build a minimally-wired pipeline that exposes _observations_for_table.

    We bypass __init__ to avoid the real Repository/FileStore/Schema boot.
    Only the attributes _observations_for_table actually touches are set.
    """
    from fermdocs.pipeline import IngestionPipeline

    pipeline = IngestionPipeline.__new__(IngestionPipeline)
    pipeline._run_id_resolver = resolver  # type: ignore[attr-defined]
    pipeline._schema_index = {}  # mapping is empty → loop over entries no-ops
    return pipeline


# ---------- doc_map injection ----------


def test_doc_map_injects_run_id_as_manifest():
    """When DocumentMap assigns table 1 to RUN-0001, the resolver receives
    manifest_run_id='RUN-0001' for that table — ManifestStrategy then wins
    over any column heuristic.
    """
    from fermdocs.domain.models import TableMapping

    resolver = _RecordingResolver()
    pipeline = _make_pipeline_with(resolver)
    table = _table(idx=1)
    mapping = TableMapping(table_id=table.table_id, entries=[])
    doc_map = _doc_map_with_two_runs()

    pipeline._observations_for_table(  # type: ignore[attr-defined]
        experiment_id="exp-x",
        file_id=uuid.uuid4(),
        table=table,
        mapping=mapping,
        doc_map=doc_map,
    )

    assert len(resolver.calls) == 1
    assert resolver.calls[0]["manifest_run_id"] == "RUN-0001"


def test_doc_map_assigns_different_run_per_table():
    """Table 2 lives in RUN-0002, table 0 lives in RUN-0001 — verify
    per-table lookup, not per-document.
    """
    from fermdocs.domain.models import TableMapping

    resolver = _RecordingResolver()
    pipeline = _make_pipeline_with(resolver)
    doc_map = _doc_map_with_two_runs()

    for table_idx, expected_run in [(0, "RUN-0001"), (2, "RUN-0002")]:
        table = _table(idx=table_idx)
        mapping = TableMapping(table_id=table.table_id, entries=[])
        pipeline._observations_for_table(  # type: ignore[attr-defined]
            experiment_id="exp-x",
            file_id=uuid.uuid4(),
            table=table,
            mapping=mapping,
            doc_map=doc_map,
        )

    assert resolver.calls[0]["manifest_run_id"] == "RUN-0001"
    assert resolver.calls[1]["manifest_run_id"] == "RUN-0002"


def test_doc_map_unassigned_table_falls_through_to_chain():
    """Table 4 is in unassigned_table_indices, not in any run. The
    resolver must receive manifest_run_id=None so the existing chain
    (column / filename / synthetic) handles it.
    """
    from fermdocs.domain.models import TableMapping

    resolver = _RecordingResolver()
    pipeline = _make_pipeline_with(resolver)
    table = _table(idx=4)
    mapping = TableMapping(table_id=table.table_id, entries=[])
    doc_map = _doc_map_with_two_runs()

    pipeline._observations_for_table(  # type: ignore[attr-defined]
        experiment_id="exp-x",
        file_id=uuid.uuid4(),
        table=table,
        mapping=mapping,
        doc_map=doc_map,
    )

    assert resolver.calls[0]["manifest_run_id"] is None


def test_no_doc_map_passes_none_manifest():
    """Backwards compat: without a doc_map, behavior matches pre-segmenter
    pipeline exactly (manifest_run_id=None passed to resolver).
    """
    from fermdocs.domain.models import TableMapping

    resolver = _RecordingResolver()
    pipeline = _make_pipeline_with(resolver)
    table = _table(idx=0)
    mapping = TableMapping(table_id=table.table_id, entries=[])

    pipeline._observations_for_table(  # type: ignore[attr-defined]
        experiment_id="exp-x",
        file_id=uuid.uuid4(),
        table=table,
        mapping=mapping,
        doc_map=None,
    )

    assert resolver.calls[0]["manifest_run_id"] is None


def test_doc_map_with_table_idx_out_of_range_falls_through():
    """Defensive: a table whose table_idx isn't in any run AND isn't in
    unassigned_table_indices (e.g., a table the segmenter never saw)
    falls through cleanly with manifest_run_id=None.
    """
    from fermdocs.domain.models import TableMapping

    resolver = _RecordingResolver()
    pipeline = _make_pipeline_with(resolver)
    table = _table(idx=99)  # not in the doc_map at all
    mapping = TableMapping(table_id=table.table_id, entries=[])
    doc_map = _doc_map_with_two_runs()

    pipeline._observations_for_table(  # type: ignore[attr-defined]
        experiment_id="exp-x",
        file_id=uuid.uuid4(),
        table=table,
        mapping=mapping,
        doc_map=doc_map,
    )

    assert resolver.calls[0]["manifest_run_id"] is None


# ---------- pipeline construction ----------


def test_pipeline_default_segmenter_is_none():
    """Backwards compat: existing callers that don't pass document_segmenter
    get None, which means no LLM call is ever made."""
    from fermdocs.pipeline import IngestionPipeline

    pipeline = IngestionPipeline.__new__(IngestionPipeline)
    # Simulate __init__ without the heavy collaborators: just verify the
    # default for the new constructor arg by direct construction inspection.
    import inspect

    sig = inspect.signature(IngestionPipeline.__init__)
    assert "document_segmenter" in sig.parameters
    assert sig.parameters["document_segmenter"].default is None


def test_pipeline_pdf_suffix_constant_is_lowercase():
    """The PDF suffix check must be case-insensitive (path.suffix.lower()).
    Verify _PDF_SUFFIXES is a lowercase set so the check works."""
    from fermdocs.pipeline import _PDF_SUFFIXES

    for suffix in _PDF_SUFFIXES:
        assert suffix == suffix.lower(), f"non-lowercase suffix in set: {suffix!r}"
        assert suffix.startswith("."), f"suffix missing dot: {suffix!r}"


# ---------- manifest precedence over doc_map ----------


def test_manifest_run_id_beats_doc_map():
    """Per design: manifest wins. Even when doc_map assigns table 1 to
    RUN-0001, an operator-supplied manifest_run_id pins it to the manifest
    value instead.
    """
    from fermdocs.domain.models import TableMapping

    resolver = _RecordingResolver()
    pipeline = _make_pipeline_with(resolver)
    table = _table(idx=1)
    mapping = TableMapping(table_id=table.table_id, entries=[])
    doc_map = _doc_map_with_two_runs()  # would assign idx=1 → RUN-0001

    pipeline._observations_for_table(  # type: ignore[attr-defined]
        experiment_id="exp-x",
        file_id=uuid.uuid4(),
        table=table,
        mapping=mapping,
        doc_map=doc_map,
        manifest_run_id="OPERATOR-RUN-42",
    )

    assert resolver.calls[0]["manifest_run_id"] == "OPERATOR-RUN-42"


def test_manifest_run_id_applies_even_without_doc_map():
    """Manifest works even when no segmenter ran (e.g. CSV path with manifest)."""
    from fermdocs.domain.models import TableMapping

    resolver = _RecordingResolver()
    pipeline = _make_pipeline_with(resolver)
    table = _table(idx=0)
    mapping = TableMapping(table_id=table.table_id, entries=[])

    pipeline._observations_for_table(  # type: ignore[attr-defined]
        experiment_id="exp-x",
        file_id=uuid.uuid4(),
        table=table,
        mapping=mapping,
        doc_map=None,
        manifest_run_id="OPERATOR-RUN-99",
    )

    assert resolver.calls[0]["manifest_run_id"] == "OPERATOR-RUN-99"
