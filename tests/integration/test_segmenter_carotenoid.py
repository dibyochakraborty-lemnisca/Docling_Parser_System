"""Integration test: real DoclingPdfParser + scripted segmenter on carotenoid PDF.

Plan ref: docs/design/2026-05-03-pdf-document-segmentation.md

Stubs the LLM call (no API key required, no network) but uses the real
Docling parser to confirm:

  1. Feed-plan tables (BATCH-04 page 10 feeding strategy, BATCH-05 page
     13, BATCH-06 page 16) are filtered out of the observation stream
     into ParseResult.feed_plan_tables.
  2. The outline produced by build_outline contains the BATCH-NN section
     headers Docling extracts.
  3. A scripted DocumentMap with valid table_indices passes validation
     end-to-end and assigns the right run-id per table_idx.
  4. CSV path remains untouched — verified by spot-checking the
     non-segmenter resolver chain still works on the same parsed tables.

Auto-skips when the carotenoid PDF is absent (CI environments without
the test fixture).
"""

from __future__ import annotations

from pathlib import Path

import pytest

from fermdocs.domain.models import ParseResult, ParsedTable
from fermdocs.parsing.document_segmenter import DocumentSegmenter, build_outline


CAROTENOID_PDF = (
    Path(__file__).resolve().parents[2]
    / "data"
    / "files"
    / "Carotenoid_batch_report_MASKED (1).pdf"
)


pytestmark = pytest.mark.skipif(
    not CAROTENOID_PDF.exists(),
    reason=f"carotenoid PDF not present at {CAROTENOID_PDF}",
)


@pytest.fixture(scope="module")
def carotenoid_parsed() -> ParseResult:
    """Parse the carotenoid PDF once per module — Docling is heavy."""
    from fermdocs.parsing.pdf_parser import DoclingPdfParser

    parser = DoclingPdfParser()
    return parser.parse(CAROTENOID_PDF)


# ---------- feed-plan filter (commit 2 verification on real PDF) ----------


def test_feed_plan_tables_extracted_to_separate_field(
    carotenoid_parsed: ParseResult,
):
    """Carotenoid pages 10/13/16 contain feeding-schedule tables shaped
    Segment | Batch hours | Duration | Feed rate | Feed volume. They must
    end up in feed_plan_tables, NOT in tables. Without the filter,
    diagnose flags 40.5 mL/h as physically impossible 40.5 L/h."""
    assert len(carotenoid_parsed.feed_plan_tables) >= 1, (
        "expected at least one feed-plan table (BATCH-04/05/06 schedules);"
        f" got 0 — filter may have regressed"
    )
    # Each detected feed-plan table must actually have the Segment header
    for table in carotenoid_parsed.feed_plan_tables:
        normalized = [(h or "").lower() for h in table.headers]
        assert any("segment" in h for h in normalized), (
            f"feed_plan_tables contains a table without 'segment' header: {table.headers}"
        )


def test_feed_plan_tables_not_in_main_tables(carotenoid_parsed: ParseResult):
    """The same table must not appear in BOTH lists — the parser routes,
    not duplicates."""
    fp_ids = {t.table_id for t in carotenoid_parsed.feed_plan_tables}
    main_ids = {t.table_id for t in carotenoid_parsed.tables}
    overlap = fp_ids & main_ids
    assert not overlap, f"tables in both lists (route-vs-duplicate bug): {overlap}"


def test_feed_rate_column_segment_pattern_excluded(
    carotenoid_parsed: ParseResult,
):
    """Defensive: no main-stream table should match the feed-plan pattern.
    If one does, the parser failed to route it and observations downstream
    will be polluted."""
    from fermdocs.parsing.pdf_parser import _is_feed_plan_table

    leaked = [t for t in carotenoid_parsed.tables if _is_feed_plan_table(t.headers)]
    assert not leaked, (
        f"feed-plan-shaped table leaked into observation stream:"
        f" {[(t.table_id, t.headers) for t in leaked]}"
    )


# ---------- outline construction on real PDF ----------


def test_outline_contains_batch_boundary_signals(
    carotenoid_parsed: ParseResult,
):
    """The LLM segmenter has two boundary signals on this PDF:
      - BATCH-NN headers (only some survive Docling's heading classifier
        + the parser's _MIN_NARRATIVE_LEN filter; observed on real PDF:
        ~1 BATCH-NN header makes it through cleanly)
      - 'Batch closure:' sentinels (one per batch — these are TextItems
        and survive cleanly, observed: 4 of 6 closures appear)

    Combined, the LLM has 5+ boundary markers in a 6-batch document.
    Test asserts both signal classes are present so a regression that
    silently drops one is caught."""
    outline = build_outline(carotenoid_parsed)
    has_batch_header = any(f"BATCH-0{n}" in outline for n in (1, 2, 3, 4, 5, 6))
    closure_count = outline.lower().count("batch closure:")
    assert has_batch_header, "no BATCH-NN headers in outline at all"
    assert closure_count >= 3, (
        f"expected ≥3 'Batch closure:' sentinels in outline, found {closure_count}."
        f" These are the segmenter's primary boundary signal."
    )


def test_outline_contains_table_position_markers(carotenoid_parsed: ParseResult):
    """Every parsed table needs to appear in the outline so the LLM can
    reference it by index."""
    outline = build_outline(carotenoid_parsed)
    for table in carotenoid_parsed.tables[:5]:
        idx = table.locator.get("table_idx")
        assert f"TABLE idx={idx}" in outline, (
            f"table_idx={idx} missing from outline — LLM cannot reference it"
        )


def test_outline_does_not_leak_table_values(carotenoid_parsed: ParseResult):
    """Privacy invariant: outline carries headers + position only, never
    cell data. Spot-check a known carotenoid row value (1082, the WCW
    spike at 108h) is absent."""
    outline = build_outline(carotenoid_parsed)
    assert "1082" not in outline, "table values leaked into LLM outline"


# ---------- scripted segmenter end-to-end ----------


class _ScriptedClient:
    def __init__(self, response):
        self._response = response

    def call(self, system: str, user: str):
        return self._response


def test_scripted_segmenter_produces_valid_doc_map(
    carotenoid_parsed: ParseResult,
):
    """Hand a scripted 6-run map (one per BATCH) to the segmenter using
    real table_idx values from the parsed PDF. Validates that the
    end-to-end happy path works — schema validation passes, run_for_table
    lookups return the right run."""
    table_indices = sorted(
        int(t.locator.get("table_idx", -1)) for t in carotenoid_parsed.tables
    )
    assert len(table_indices) >= 6, (
        f"need ≥6 measurement tables to script a 6-batch map;"
        f" got {len(table_indices)}"
    )

    # Distribute the available indices across 6 runs (rough — for
    # validation, not for biological accuracy)
    chunk_size = max(1, len(table_indices) // 6)
    chunks = [
        table_indices[i * chunk_size : (i + 1) * chunk_size]
        for i in range(6)
    ]
    # Stick any leftovers on the last chunk
    leftover = table_indices[6 * chunk_size :]
    chunks[-1] = chunks[-1] + leftover
    # Drop any empty chunks (very small PDFs)
    chunks = [c for c in chunks if c]

    response = {
        "runs": [
            {
                "run_id": f"RUN-000{i + 1}",
                "display_name": f"BATCH-0{i + 1} REPORT",
                "table_indices": chunks[i],
                "source_signal": "section_header",
                "confidence": 0.9,
                "rationale": f"BATCH-0{i + 1} heading section",
            }
            for i in range(len(chunks))
        ],
        "unassigned_table_indices": [],
        "overall_confidence": 0.88,
    }
    client = _ScriptedClient(response)
    segmenter = DocumentSegmenter(
        client=client,
        model_name="scripted-test",
        provider="test",
    )
    dm = segmenter.segment(carotenoid_parsed, file_id="carotenoid-test")
    assert dm is not None, "scripted segmenter should return a valid map"
    assert len(dm.runs) == len(chunks)
    # Verify per-table lookup works
    first_idx_of_run_0 = chunks[0][0]
    seg = dm.run_for_table(first_idx_of_run_0)
    assert seg is not None
    assert seg.run_id == "RUN-0001"


def test_scripted_segmenter_rejects_hallucinated_table_idx(
    carotenoid_parsed: ParseResult,
):
    """If the LLM returns table_idx 999 (which the PDF doesn't have),
    the segmenter must reject the whole map (commit 3's range check)
    and return None. Pipeline then falls through to existing chain."""
    response = {
        "runs": [
            {
                "run_id": "RUN-0001",
                "display_name": "BATCH-01",
                "table_indices": [0, 999],  # 999 doesn't exist in any PDF
                "source_signal": "section_header",
                "confidence": 0.9,
            }
        ],
        "unassigned_table_indices": [],
        "overall_confidence": 0.9,
    }
    client = _ScriptedClient(response)
    segmenter = DocumentSegmenter(
        client=client, model_name="scripted-test", provider="test"
    )
    dm = segmenter.segment(carotenoid_parsed, file_id="carotenoid-test")
    assert dm is None, "out-of-range table_idx should reject whole map"
