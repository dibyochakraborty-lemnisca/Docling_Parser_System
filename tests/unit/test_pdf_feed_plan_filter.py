"""Tests for _is_feed_plan_table — deterministic feed-plan table detection.

Plan ref: docs/design/2026-05-03-pdf-document-segmentation.md

The carotenoid PDF and similar fermentation reports contain operator-supplied
feeding schedules that look like timecourse tables but aren't measurements.
Letting them flow through pollutes feed_rate_l_per_h with planned setpoints
that diagnose then flags as physically impossible.

The detection rule is intentionally narrow: better to miss a feed-plan table
(falls through to the existing observation path, no regression) than to
misclassify a real measurement table (silently lose data).
"""

from __future__ import annotations

import pytest

from fermdocs.parsing.pdf_parser import _is_feed_plan_table


# ---------- positive cases — should be classified as feed-plan ----------


@pytest.mark.parametrize(
    "headers",
    [
        # Carotenoid BATCH-04 page 10 — exact shape
        ["Segment", "Batch hours (start → end)", "Duration (h)", "Feed rate (mL/h)", "Feed volume (mL)"],
        # Same shape, lowercase + extra whitespace (Docling normalization wobble)
        ["segment", "batch hours start end", "duration h", "feed rate ml h", "feed volume ml"],
        # Carotenoid BATCH-05 page 13 — same shape, slight variation
        ["Segment", "Batch hours (start→end)", "Duration (h)", "Feed rate (mL/h)", "Feed volume (mL)"],
        # Minimal sufficient: just Segment + Batch hours
        ["Segment", "Batch hours"],
        # Minimal sufficient: just Segment + Feed rate
        ["Segment", "Feed rate (mL/h)"],
    ],
)
def test_classifies_feed_plan_tables(headers):
    assert _is_feed_plan_table(headers) is True, f"missed feed-plan table: {headers}"


# ---------- negative cases — must NOT be classified as feed-plan ----------


@pytest.mark.parametrize(
    "headers",
    [
        # Carotenoid timecourse measurement table (BATCH-01 page 3)
        ["Time (h)", "OD", "WCW (mg/3 mL)", "pO₂", "Feed (mL/L/h)"],
        # IndPenSim CSV measurement headers
        ["Time (h)", "Batch_ref", "Batch_ref.1", "Biomass concentration"],
        # Generic timecourse
        ["Time", "OD600", "Glucose"],
        # Composition table
        ["Component", "Concentration (g/L)"],
        # Supplement-additions table (Segment is absent)
        ["Addition", "Batch hour", "Supplement type", "Volume added (mL)"],
        # Empty headers
        [],
        # All None / empty strings
        [None, "", "  "],
    ],
)
def test_does_not_classify_measurement_tables(headers):
    assert _is_feed_plan_table(headers) is False, f"misclassified as feed-plan: {headers}"


# ---------- edge cases ----------


def test_segment_alone_is_not_enough():
    """A 'Segment' header without Batch hours or Feed rate could be many things
    (genome segment, network segment, doc segment). Don't over-classify.
    """
    assert _is_feed_plan_table(["Segment", "Description", "Notes"]) is False


def test_handles_none_headers_gracefully():
    """Docling occasionally emits None as a header cell. Must not crash."""
    assert _is_feed_plan_table([None, "Batch hours", "Feed rate"]) is False


def test_case_insensitive_and_whitespace_tolerant():
    assert _is_feed_plan_table(["  SEGMENT  ", "Batch Hours"]) is True
