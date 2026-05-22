"""Live LLM eval: real gemini-3.1-pro-preview on carotenoid PDF.

Plan ref: docs/design/2026-05-03-pdf-document-segmentation.md

This is the only test that:
  - hits the real Gemini API
  - costs tokens (~one structured-output call)
  - requires GEMINI_API_KEY
  - is opt-in via `pytest -m live_llm`

Default test runs DESELECT this (see pyproject.toml addopts). CI doesn't
fire it. To run locally:

    pytest -m live_llm tests/integration/test_segmenter_live_carotenoid.py

This is the eval gate for "does the LLM actually segment a real
multi-batch PDF correctly?". Stub-client tests verify the orchestration
machinery; this verifies the prompt + schema + outline produce
acceptable output from the real model.

Tolerance: the carotenoid PDF has 6 batches. We assert the segmenter
detects 4-8 runs (slack for the LLM splitting/merging boundary cases —
we don't want a flaky test, but we want to catch order-of-magnitude
regressions like "1 run" or "20 runs").
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from fermdocs.parsing.document_segmenter import DocumentSegmenter
from fermdocs.parsing.gemini_segmenter_client import GeminiSegmenterClient


CAROTENOID_PDF = (
    Path(__file__).resolve().parents[2]
    / "data"
    / "files"
    / "Carotenoid_batch_report_MASKED (1).pdf"
)


pytestmark = [
    pytest.mark.live_llm,
    pytest.mark.skipif(
        not os.environ.get("GEMINI_API_KEY"),
        reason="GEMINI_API_KEY not set; skipping live segmenter eval",
    ),
    pytest.mark.skipif(
        not CAROTENOID_PDF.exists(),
        reason=f"carotenoid PDF not present at {CAROTENOID_PDF}",
    ),
]


def test_live_segmenter_detects_six_batches_on_carotenoid():
    """Real LLM call on the carotenoid PDF. The doc has 6 distinct
    fermentation batches (BATCH-01 through BATCH-06). The segmenter
    should detect approximately that many.

    Tolerance: 4-8 runs. Not 6-exact because:
      - BATCH-01/02 headers are masked-overlay-corrupted in Docling output
        — LLM may fail to identify their boundaries cleanly.
      - LLM may split a batch with discontinuous tables or merge two
        adjacent batches if the boundary signal is weak.

    A pass at this tolerance proves the segmenter is producing usable
    output. A failure (1, 2, 20+) signals prompt or model regression.
    """
    from fermdocs.parsing.pdf_parser import DoclingPdfParser

    parsed = DoclingPdfParser().parse(CAROTENOID_PDF)
    client = GeminiSegmenterClient()
    segmenter = DocumentSegmenter(
        client=client,
        model_name=client.model_name,
        provider="gemini",
    )

    dm = segmenter.segment(parsed, file_id="carotenoid-live")

    assert dm is not None, (
        "live segmenter returned None — LLM call failed, schema validation"
        " rejected the response, or table_idx range check rejected. Re-run"
        " with FERMDOCS_DEBUG_SEGMENTER=1 to see raw LLM response."
    )

    # Order-of-magnitude check, not exact-count
    n_runs = len(dm.runs)
    assert 4 <= n_runs <= 8, (
        f"expected segmenter to detect ~6 runs on 6-batch carotenoid PDF;"
        f" got {n_runs}. Run display names: {[r.display_name for r in dm.runs]}."
        f" If consistently wrong, prompt or model needs revisit."
    )

    # Sanity: model should populate display_names with something useful
    for run in dm.runs:
        assert run.display_name, f"run {run.run_id} has empty display_name"
        assert run.run_id.startswith("RUN-"), (
            f"run_id {run.run_id!r} doesn't follow RUN-NNNN convention"
        )

    # Sanity: every assigned table_idx must be valid (commit 3's range
    # check enforces this; if the range check failed, dm would be None
    # and we'd already have asserted out)
    valid_indices = {
        int(t.locator.get("table_idx", -1)) for t in parsed.tables
    }
    for run in dm.runs:
        for idx in run.table_indices:
            assert idx in valid_indices, (
                f"run {run.run_id} cites table_idx={idx} which doesn't exist"
                f" in the parsed PDF; range check should have rejected the map"
            )

    # If the segmenter saw at least 4 batches, BATCH closure sentinels
    # should account for most of them — log for the operator's eyeball
    # check (printed via -s):
    print(
        f"\n[live segmenter] detected {n_runs} runs:"
        f" {[(r.run_id, r.display_name, len(r.table_indices)) for r in dm.runs]}"
    )
    print(
        f"[live segmenter] unassigned tables: {dm.unassigned_table_indices}"
    )
    print(f"[live segmenter] overall_confidence: {dm.overall_confidence}")


def test_live_segmenter_does_not_assign_feed_plan_tables():
    """Defensive: feed-plan tables are filtered out before the segmenter
    sees them (commit 2). The LLM should never receive any table whose
    headers match the feed-plan shape, so it can't accidentally assign
    one to a measurement run.
    """
    from fermdocs.parsing.pdf_parser import DoclingPdfParser, _is_feed_plan_table

    parsed = DoclingPdfParser().parse(CAROTENOID_PDF)
    client = GeminiSegmenterClient()
    segmenter = DocumentSegmenter(
        client=client,
        model_name=client.model_name,
        provider="gemini",
    )

    dm = segmenter.segment(parsed, file_id="carotenoid-live")
    assert dm is not None

    # The segmenter only ever sees tables from parsed.tables (not
    # parsed.feed_plan_tables). Verify no table_idx the LLM assigned
    # corresponds to a table whose headers match the feed-plan pattern.
    table_by_idx = {
        int(t.locator.get("table_idx", -1)): t for t in parsed.tables
    }
    for run in dm.runs:
        for idx in run.table_indices:
            table = table_by_idx.get(idx)
            assert table is not None
            assert not _is_feed_plan_table(table.headers), (
                f"run {run.run_id} assigned table_idx={idx}"
                f" with headers {table.headers!r} which match the feed-plan"
                f" pattern — should have been filtered before segmenter saw it"
            )
