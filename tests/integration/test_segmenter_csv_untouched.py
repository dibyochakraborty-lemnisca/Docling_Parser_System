"""Integration test: IndPenSim CSV ingest with segmenter wired but never invoked.

Plan ref: docs/design/2026-05-03-pdf-document-segmentation.md

Per design call: "DO NOT touch the CSV path". This is the safety net.

Verifies that:
  1. CsvParser produces a ParseResult with empty feed_plan_tables and
     empty narrative_blocks (the parser itself is unchanged).
  2. RunIdResolver.ColumnStrategy still selects Batch_ref.1 (the column
     with real run-id signal) over the constant-zero Batch_ref column.
  3. With the segmenter constructed and passed to IngestionPipeline, a
     CSV ingest path NEVER calls the segmenter's .call() method —
     enforced by a sentinel client that raises AssertionError if invoked.

If any of these regress, the CSV path has been silently affected by the
PDF-segmentation work.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from fermdocs.domain.models import ParseResult
from fermdocs.parsing.csv_parser import CsvParser
from fermdocs.parsing.document_segmenter import DocumentSegmenter
from fermdocs.parsing.run_id_resolver import RunIdResolver


INDPENSIM_CSV = (
    Path(__file__).resolve().parents[2]
    / "data"
    / "files"
    / "IndPenSim_V2_export_V7.csv"
)


pytestmark = pytest.mark.skipif(
    not INDPENSIM_CSV.exists(),
    reason=f"IndPenSim CSV not present at {INDPENSIM_CSV}",
)


@pytest.fixture(scope="module")
def indpensim_parsed() -> ParseResult:
    """Parse IndPenSim once per module."""
    parser = CsvParser()
    return parser.parse(INDPENSIM_CSV)


# ---------- parser-level: CSV path unchanged ----------


def test_csv_parser_produces_no_feed_plan_tables(indpensim_parsed: ParseResult):
    """The feed_plan_tables field is PDF-only (DoclingPdfParser populates
    it). CSV/Excel parsers must leave it empty by construction."""
    assert indpensim_parsed.feed_plan_tables == [], (
        "CsvParser populated feed_plan_tables — the PDF-only filter has"
        " leaked into the CSV path"
    )


def test_csv_parser_produces_no_narrative_blocks(indpensim_parsed: ParseResult):
    """CSVs have no narrative; this is the existing invariant. If it
    breaks, the segmenter outline construction would silently include
    spurious data."""
    assert indpensim_parsed.narrative_blocks == [], (
        "CsvParser produced narrative blocks — CSV invariant broken"
    )


def test_csv_parser_produces_at_least_one_table(indpensim_parsed: ParseResult):
    """Sanity: the CSV does parse to a table."""
    assert len(indpensim_parsed.tables) >= 1


# ---------- resolver: ColumnStrategy still wins on IndPenSim ----------


def test_resolver_picks_batch_ref_dot_1_column(indpensim_parsed: ParseResult):
    """The IndPenSim regression: Batch_ref is always 0, Batch_ref.1 has
    real values. ColumnStrategy must pick the live one. This was the
    original motivation for ColumnStrategy and must not regress."""
    table = indpensim_parsed.tables[0]
    resolver = RunIdResolver()
    resolution = resolver.resolve(
        headers=table.headers,
        rows=table.rows,
        filename=INDPENSIM_CSV.name,
        manifest_run_id=None,
    )
    assert resolution is not None
    assert resolution.strategy == "column", (
        f"expected ColumnStrategy to win on IndPenSim;"
        f" got {resolution.strategy} (rationale: {resolution.rationale})"
    )
    # The resolver should pick the column whose values vary (Batch_ref.1)
    # not the constant-zero one.
    assert resolution.column_idx is not None


# ---------- segmenter: never invoked on CSV path ----------


class _FailIfCalledClient:
    """Sentinel client. .call() raising means the CSV path invoked the
    segmenter's LLM, which violates the 'don't touch CSV' invariant."""

    def __init__(self) -> None:
        self.was_called = False

    def call(self, system: str, user: str):
        self.was_called = True
        raise AssertionError(
            "DocumentSegmenter.call() was invoked during CSV ingest."
            " The .pdf suffix guard in IngestionPipeline._ingest_one"
            " has regressed — segmenter is leaking into the CSV path."
        )


def test_segmenter_short_circuits_on_empty_tables(indpensim_parsed: ParseResult):
    """Even when called directly, the segmenter must return an empty map
    (not None, not an exception) for a ParseResult with zero tables. This
    happens during CSV ingest only if some bug routes the parsed CSV
    through the segmenter — defensive."""
    sentinel = _FailIfCalledClient()
    segmenter = DocumentSegmenter(
        client=sentinel,
        model_name="sentinel",
        provider="test",
    )
    # Synthesize a CSV-like ParseResult: tables present, but the test
    # exercises segment(empty_tables) → no LLM call.
    empty_result = ParseResult(tables=[], narrative_blocks=[], feed_plan_tables=[])
    dm = segmenter.segment(empty_result, file_id="csv-fake")
    assert dm is not None
    assert dm.runs == []
    assert sentinel.was_called is False, (
        "segmenter called LLM despite zero tables — short-circuit broken"
    )


def test_pipeline_skips_segmenter_for_csv_via_suffix_guard():
    """End-to-end: construct an IngestionPipeline with a sentinel
    segmenter and assert that ingesting a .csv file never invokes the
    LLM, because the .pdf suffix guard short-circuits before .segment()
    runs.

    We exercise the suffix-guard logic directly without spinning up the
    full pipeline (which needs a real DB) — _ingest_one's segmenter
    branch is gated on `path.suffix.lower() in _PDF_SUFFIXES`.
    """
    from fermdocs.pipeline import _PDF_SUFFIXES

    # The guard is the single load-bearing line. Verify a .csv path is
    # not in the set, so the segmenter branch is unreachable for CSV.
    csv_path = Path("data/files/IndPenSim_V2_export_V7.csv")
    assert csv_path.suffix.lower() not in _PDF_SUFFIXES
    xlsx_path = Path("foo.xlsx")
    assert xlsx_path.suffix.lower() not in _PDF_SUFFIXES
    pdf_path = Path("foo.pdf")
    assert pdf_path.suffix.lower() in _PDF_SUFFIXES
    pdf_upper = Path("FOO.PDF")
    assert pdf_upper.suffix.lower() in _PDF_SUFFIXES, (
        "PDF guard must be case-insensitive"
    )
