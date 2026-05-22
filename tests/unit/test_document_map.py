"""Tests for DocumentMap + RunSegment validation.

Plan ref: docs/design/2026-05-03-pdf-document-segmentation.md

Critical invariants (must reject malformed LLM output):
  - A table_idx may appear in at most one run.
  - A table_idx cannot be both assigned and unassigned.
  - Run ids within a DocumentMap are unique.
  - table_indices and unassigned_table_indices contain unique non-negative ints.
"""

from __future__ import annotations

import pytest

from fermdocs.domain.models import DocumentMap, RunSegment, RunSegmentSource


def _segment(
    run_id: str,
    indices: list[int],
    *,
    source: RunSegmentSource = RunSegmentSource.SECTION_HEADER,
    confidence: float = 0.9,
    display_name: str | None = None,
) -> RunSegment:
    return RunSegment(
        run_id=run_id,
        display_name=display_name or run_id,
        table_indices=indices,
        source_signal=source,
        confidence=confidence,
        rationale="test",
    )


# ---------- RunSegment ----------


def test_run_segment_happy_path():
    s = _segment("RUN-0001", [0, 1, 2])
    assert s.run_id == "RUN-0001"
    assert s.table_indices == [0, 1, 2]
    assert s.source_signal == RunSegmentSource.SECTION_HEADER


def test_run_segment_rejects_negative_table_idx():
    with pytest.raises(ValueError, match="non-negative"):
        _segment("RUN-0001", [0, -1])


def test_run_segment_rejects_duplicate_table_idx():
    with pytest.raises(ValueError, match="unique within a run"):
        _segment("RUN-0001", [0, 1, 1])


def test_run_segment_rejects_empty_run_id():
    with pytest.raises(ValueError):
        RunSegment(
            run_id="",
            display_name="x",
            table_indices=[0],
            source_signal=RunSegmentSource.SECTION_HEADER,
            confidence=0.9,
        )


def test_run_segment_confidence_bounds():
    with pytest.raises(ValueError):
        _segment("RUN-0001", [0], confidence=1.5)
    with pytest.raises(ValueError):
        _segment("RUN-0001", [0], confidence=-0.1)


# ---------- DocumentMap ----------


def test_document_map_happy_path():
    dm = DocumentMap(
        file_id="file-abc",
        runs=[
            _segment("RUN-0001", [0, 1]),
            _segment("RUN-0002", [2, 3]),
        ],
        unassigned_table_indices=[4, 5],
        overall_confidence=0.88,
        llm_model="gemini-3.1-pro-preview",
        llm_provider="gemini",
    )
    assert len(dm.runs) == 2
    assert dm.unassigned_table_indices == [4, 5]
    assert dm.schema_version == "1.0"


def test_document_map_rejects_table_idx_in_two_runs():
    """The carotenoid case where the LLM might claim table 5 belongs to
    both BATCH-04 and BATCH-05. Validation must reject this.
    """
    with pytest.raises(ValueError, match="multiple runs"):
        DocumentMap(
            file_id="file-abc",
            runs=[
                _segment("RUN-0001", [0, 1, 5]),
                _segment("RUN-0002", [5, 6]),
            ],
            overall_confidence=0.5,
            llm_model="gemini-3.1-pro-preview",
            llm_provider="gemini",
        )


def test_document_map_rejects_assigned_and_unassigned_overlap():
    """Table 3 cannot be both 'in RUN-0001' and 'unassigned'."""
    with pytest.raises(ValueError, match="contradictory|both assigned"):
        DocumentMap(
            file_id="file-abc",
            runs=[_segment("RUN-0001", [0, 1, 3])],
            unassigned_table_indices=[3, 4],
            overall_confidence=0.5,
            llm_model="gemini-3.1-pro-preview",
            llm_provider="gemini",
        )


def test_document_map_rejects_duplicate_run_ids():
    with pytest.raises(ValueError, match="unique run_ids"):
        DocumentMap(
            file_id="file-abc",
            runs=[
                _segment("RUN-0001", [0]),
                _segment("RUN-0001", [1]),
            ],
            overall_confidence=0.5,
            llm_model="gemini-3.1-pro-preview",
            llm_provider="gemini",
        )


def test_document_map_rejects_negative_unassigned():
    with pytest.raises(ValueError, match="non-negative"):
        DocumentMap(
            file_id="file-abc",
            runs=[],
            unassigned_table_indices=[-1, 0],
            overall_confidence=0.0,
            llm_model="gemini-3.1-pro-preview",
            llm_provider="gemini",
        )


def test_document_map_rejects_duplicate_unassigned():
    with pytest.raises(ValueError, match="unique"):
        DocumentMap(
            file_id="file-abc",
            runs=[],
            unassigned_table_indices=[3, 3],
            overall_confidence=0.0,
            llm_model="gemini-3.1-pro-preview",
            llm_provider="gemini",
        )


def test_document_map_allows_zero_runs():
    """An LLM saying 'I couldn't segment this' is valid output. The
    pipeline falls back to the existing resolver chain. No exception.
    """
    dm = DocumentMap(
        file_id="file-abc",
        runs=[],
        unassigned_table_indices=[0, 1, 2],
        overall_confidence=0.1,
        llm_model="gemini-3.1-pro-preview",
        llm_provider="gemini",
    )
    assert dm.runs == []
    assert dm.run_for_table(0) is None


def test_document_map_run_for_table_finds_assigned():
    dm = DocumentMap(
        file_id="file-abc",
        runs=[
            _segment("RUN-0001", [0, 1, 2]),
            _segment("RUN-0002", [3, 4]),
        ],
        overall_confidence=0.9,
        llm_model="gemini-3.1-pro-preview",
        llm_provider="gemini",
    )
    s1 = dm.run_for_table(1)
    assert s1 is not None and s1.run_id == "RUN-0001"
    s2 = dm.run_for_table(4)
    assert s2 is not None and s2.run_id == "RUN-0002"


def test_document_map_run_for_table_returns_none_for_unassigned():
    dm = DocumentMap(
        file_id="file-abc",
        runs=[_segment("RUN-0001", [0, 1])],
        unassigned_table_indices=[2, 3],
        overall_confidence=0.5,
        llm_model="gemini-3.1-pro-preview",
        llm_provider="gemini",
    )
    assert dm.run_for_table(2) is None
    assert dm.run_for_table(99) is None


def test_document_map_schema_version_pinned():
    """Adding schema_version=2.0 should fail at model construction.
    Future schema migrations must be explicit.
    """
    with pytest.raises(ValueError):
        DocumentMap(
            schema_version="2.0",  # type: ignore[arg-type]
            file_id="file-abc",
            runs=[],
            overall_confidence=0.0,
            llm_model="x",
            llm_provider="y",
        )
