"""Tests for DocumentSegmenter — happy path + every failure mode.

Plan ref: docs/design/2026-05-03-pdf-document-segmentation.md

The segmenter must NEVER raise into the pipeline. Every failure path
returns None so callers fall back to the existing column-heuristic chain.
This file enumerates every failure path documented in the design's
Failure-modes table.
"""

from __future__ import annotations

from typing import Any

import pytest

from fermdocs.domain.models import (
    NarrativeBlock,
    NarrativeBlockType,
    ParseResult,
    ParsedTable,
    RunSegmentSource,
)
from fermdocs.parsing.document_segmenter import (
    DocumentSegmenter,
    build_outline,
    build_user_prompt,
)


# ---------- helpers ----------


class _ScriptedClient:
    """Returns a pre-canned dict, or raises a pre-canned exception."""

    def __init__(
        self, response: dict[str, Any] | None = None, raises: Exception | None = None
    ) -> None:
        self._response = response
        self._raises = raises
        self.calls: list[tuple[str, str]] = []

    def call(self, system: str, user: str) -> dict[str, Any]:
        self.calls.append((system, user))
        if self._raises is not None:
            raise self._raises
        assert self._response is not None
        return self._response


def _table(idx: int, headers: list[str] | None = None, page: int = 1) -> ParsedTable:
    return ParsedTable(
        table_id=f"f.pdf#p{page}#t{idx}",
        headers=headers or ["Time (h)", "OD", "WCW"],
        rows=[],
        locator={
            "format": "pdf",
            "file": "f.pdf",
            "page": page,
            "table_idx": idx,
            "section": "table",
        },
    )


def _block(
    text: str,
    block_type: NarrativeBlockType = NarrativeBlockType.PARAGRAPH,
    page: int = 1,
    paragraph_idx: int = 0,
) -> NarrativeBlock:
    return NarrativeBlock(
        text=text,
        type=block_type,
        locator={
            "format": "pdf",
            "file": "f.pdf",
            "page": page,
            "section": "narrative",
            "paragraph_idx": paragraph_idx,
        },
    )


def _segmenter(client: _ScriptedClient | None) -> DocumentSegmenter:
    return DocumentSegmenter(
        client=client,
        model_name="gemini-3.1-pro-preview",
        provider="gemini",
    )


# ---------- happy paths ----------


def test_happy_path_six_batches():
    """Carotenoid-shaped response: 6 runs, each with 1 table assigned."""
    parse_result = ParseResult(
        tables=[_table(i) for i in range(6)],
        narrative_blocks=[
            _block(
                f"BATCH-0{i + 1} REPORT",
                NarrativeBlockType.HEADING,
                page=i * 3 + 1,
                paragraph_idx=i * 10,
            )
            for i in range(6)
        ],
    )
    client = _ScriptedClient(
        response={
            "runs": [
                {
                    "run_id": f"RUN-000{i + 1}",
                    "display_name": f"BATCH-0{i + 1} REPORT",
                    "table_indices": [i],
                    "source_signal": "section_header",
                    "confidence": 0.95,
                    "rationale": f"BATCH-0{i + 1} REPORT heading on page {i * 3 + 1}",
                }
                for i in range(6)
            ],
            "unassigned_table_indices": [],
            "overall_confidence": 0.92,
        }
    )
    dm = _segmenter(client).segment(parse_result, file_id="file-abc")
    assert dm is not None
    assert len(dm.runs) == 6
    assert dm.run_for_table(3).run_id == "RUN-0004"
    assert dm.llm_model == "gemini-3.1-pro-preview"
    assert dm.llm_provider == "gemini"
    assert len(client.calls) == 1


def test_empty_tables_returns_zero_run_map_no_llm_call():
    """Documents with zero parsed tables short-circuit; we still return a
    map (with 0 runs) so callers can persist provenance.
    """
    client = _ScriptedClient(response={"should_not_be_called": True})
    parse_result = ParseResult(tables=[], narrative_blocks=[])
    dm = _segmenter(client).segment(parse_result, file_id="file-empty")
    assert dm is not None
    assert dm.runs == []
    assert client.calls == []  # short-circuit verified


def test_zero_runs_with_unassigned_is_valid():
    """LLM may decide a document has no clear run boundaries. Map is valid;
    pipeline falls back to existing chain on a per-table basis.
    """
    parse_result = ParseResult(tables=[_table(0), _table(1)])
    client = _ScriptedClient(
        response={
            "runs": [],
            "unassigned_table_indices": [0, 1],
            "overall_confidence": 0.2,
        }
    )
    dm = _segmenter(client).segment(parse_result, file_id="file-x")
    assert dm is not None
    assert dm.runs == []
    assert dm.unassigned_table_indices == [0, 1]


# ---------- failure: no client configured ----------


def test_no_client_returns_none():
    parse_result = ParseResult(tables=[_table(0)])
    dm = _segmenter(None).segment(parse_result, file_id="file-x")
    assert dm is None


# ---------- failure: LLM raises ----------


def test_llm_raises_returns_none_does_not_propagate():
    parse_result = ParseResult(tables=[_table(0), _table(1)])
    client = _ScriptedClient(raises=RuntimeError("connection timeout"))
    dm = _segmenter(client).segment(parse_result, file_id="file-x")
    assert dm is None


def test_llm_raises_arbitrary_exception():
    """Even programmer errors from the client must not propagate — the
    segmenter is best-effort."""
    parse_result = ParseResult(tables=[_table(0)])
    client = _ScriptedClient(raises=KeyError("missing-key"))
    dm = _segmenter(client).segment(parse_result, file_id="file-x")
    assert dm is None


# ---------- failure: malformed response ----------


def test_missing_runs_field_treated_as_zero_runs():
    """`runs` missing entirely is equivalent to `runs: []`."""
    parse_result = ParseResult(tables=[_table(0)])
    client = _ScriptedClient(
        response={"unassigned_table_indices": [0], "overall_confidence": 0.1}
    )
    dm = _segmenter(client).segment(parse_result, file_id="file-x")
    assert dm is not None
    assert dm.runs == []


def test_run_missing_required_field_returns_none():
    """A run object missing run_id is invalid LLM output → fall back."""
    parse_result = ParseResult(tables=[_table(0)])
    client = _ScriptedClient(
        response={
            "runs": [{"display_name": "BATCH-01", "table_indices": [0]}],
            "overall_confidence": 0.5,
        }
    )
    dm = _segmenter(client).segment(parse_result, file_id="file-x")
    assert dm is None


def test_duplicate_table_idx_across_runs_rejected():
    """The validation invariant from commit 1: same table_idx in two runs
    is rejected by DocumentMap.model_post_init, segmenter returns None.
    """
    parse_result = ParseResult(tables=[_table(0), _table(1), _table(2)])
    client = _ScriptedClient(
        response={
            "runs": [
                {
                    "run_id": "RUN-0001",
                    "display_name": "BATCH-01",
                    "table_indices": [0, 1],
                    "source_signal": "section_header",
                    "confidence": 0.9,
                },
                {
                    "run_id": "RUN-0002",
                    "display_name": "BATCH-02",
                    "table_indices": [1, 2],  # 1 collides with RUN-0001
                    "source_signal": "section_header",
                    "confidence": 0.9,
                },
            ],
            "overall_confidence": 0.9,
        }
    )
    dm = _segmenter(client).segment(parse_result, file_id="file-x")
    assert dm is None


def test_assigned_and_unassigned_overlap_rejected():
    parse_result = ParseResult(tables=[_table(0), _table(1)])
    client = _ScriptedClient(
        response={
            "runs": [
                {
                    "run_id": "RUN-0001",
                    "display_name": "BATCH-01",
                    "table_indices": [0, 1],
                    "source_signal": "section_header",
                    "confidence": 0.9,
                }
            ],
            "unassigned_table_indices": [1],  # contradiction
            "overall_confidence": 0.9,
        }
    )
    dm = _segmenter(client).segment(parse_result, file_id="file-x")
    assert dm is None


# ---------- failure: out-of-range table_idx ----------


def test_unknown_table_idx_in_runs_rejected():
    """LLM hallucinates table_idx 99 in a doc with 2 tables → reject whole map."""
    parse_result = ParseResult(tables=[_table(0), _table(1)])
    client = _ScriptedClient(
        response={
            "runs": [
                {
                    "run_id": "RUN-0001",
                    "display_name": "BATCH-01",
                    "table_indices": [0, 99],  # 99 doesn't exist
                    "source_signal": "section_header",
                    "confidence": 0.9,
                }
            ],
            "overall_confidence": 0.9,
        }
    )
    dm = _segmenter(client).segment(parse_result, file_id="file-x")
    assert dm is None


def test_unknown_table_idx_in_unassigned_rejected():
    parse_result = ParseResult(tables=[_table(0)])
    client = _ScriptedClient(
        response={
            "runs": [],
            "unassigned_table_indices": [0, 5],  # 5 doesn't exist
            "overall_confidence": 0.1,
        }
    )
    dm = _segmenter(client).segment(parse_result, file_id="file-x")
    assert dm is None


# ---------- failure: bad source_signal coerces to inferred ----------


def test_unknown_source_signal_coerced_to_inferred():
    """An LLM-supplied source_signal we don't recognize defaults to inferred
    (least-trusted level) rather than rejecting the whole map.
    """
    parse_result = ParseResult(tables=[_table(0)])
    client = _ScriptedClient(
        response={
            "runs": [
                {
                    "run_id": "RUN-0001",
                    "display_name": "BATCH-01",
                    "table_indices": [0],
                    "source_signal": "psychic_intuition",  # unknown
                    "confidence": 0.9,
                }
            ],
            "overall_confidence": 0.9,
        }
    )
    dm = _segmenter(client).segment(parse_result, file_id="file-x")
    assert dm is not None
    assert dm.runs[0].source_signal == RunSegmentSource.INFERRED


# ---------- outline construction ----------


def test_outline_orders_by_page_then_idx():
    """Tables on a page sort after narrative on the same page (because
    headings introduce tables, then tables follow)."""
    parse_result = ParseResult(
        tables=[
            _table(0, page=2),
            _table(1, page=1),
        ],
        narrative_blocks=[
            _block(
                "BATCH-01 REPORT",
                NarrativeBlockType.HEADING,
                page=1,
                paragraph_idx=0,
            ),
            _block(
                "BATCH-02 REPORT",
                NarrativeBlockType.HEADING,
                page=2,
                paragraph_idx=10,
            ),
        ],
    )
    outline = build_outline(parse_result)
    lines = outline.split("\n")
    # Order should be: BATCH-01 (p1), TABLE idx=1 (p1), BATCH-02 (p2), TABLE idx=0 (p2)
    assert "BATCH-01" in lines[0]
    assert "TABLE idx=1" in lines[1]
    assert "BATCH-02" in lines[2]
    assert "TABLE idx=0" in lines[3]


def test_outline_truncates_long_text_blocks():
    long_text = "X" * 1000
    parse_result = ParseResult(
        narrative_blocks=[_block(long_text, paragraph_idx=0)]
    )
    outline = build_outline(parse_result)
    assert len(outline) < 500  # bounded


def test_outline_does_not_include_table_values():
    """Privacy-style invariant: only table_idx + headers + position go to LLM,
    never row values. Even if rows were populated they shouldn't leak.
    """
    parse_result = ParseResult(
        tables=[
            ParsedTable(
                table_id="f.pdf#p1#t0",
                headers=["Time", "OD"],
                rows=[[42.0, 999.999], [43.0, 1000.0]],
                locator={"page": 1, "table_idx": 0},
            )
        ]
    )
    outline = build_outline(parse_result)
    assert "999.999" not in outline
    assert "1000.0" not in outline
    assert "Time" in outline


def test_user_prompt_includes_outline_and_table_count():
    parse_result = ParseResult(tables=[_table(0), _table(1), _table(2)])
    prompt = build_user_prompt(parse_result, file_id="file-xyz")
    assert "file-xyz" in prompt
    assert "0..2" in prompt  # valid index range hint
    assert "Outline:" in prompt
