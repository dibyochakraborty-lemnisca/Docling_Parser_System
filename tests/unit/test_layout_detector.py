"""LayoutDetector unit tests: structure detection + the safety stack.

The detector turns a raw, header-less grid into a ParsedTable by asking an
LLM only for STRUCTURE (indices), then slicing the grid itself. These tests
use a scripted client so they assert the detector's validation/evidence/
fail-soft behavior, never a live LLM.
"""

from __future__ import annotations

from fermdocs.mapping.layout_detector import LayoutDetector

# A messy sheet: title + metadata rows, then the real table at row 3.
_GRID = [
    ["Praaj Lab Report", None, None],
    ["Batch B471", None, None],
    [None, None, None],
    ["Time (h)", "Biomass", "Temp"],
    ["0", "0.5", "297"],
    ["2", "1.2", "298"],
]
_SOURCE = "praaj.xlsx#B471"


class _ScriptedClient:
    def __init__(self, payload):
        self._payload = payload
        self.calls = 0

    def call(self, system, user):
        self.calls += 1
        return self._payload


class _RaisingClient:
    def call(self, system, user):
        raise RuntimeError("llm down")


def _region(**over):
    base = {
        "header_row": 3,
        "data_start_row": 4,
        "data_end_row": 5,
        "run_grouping": "single_run",
        "header_cells": ["Time (h)", "Biomass", "Temp"],
        "confidence": 0.9,
    }
    base.update(over)
    return base


def test_detects_buried_table():
    det = LayoutDetector(_ScriptedClient({"tables": [_region()]}))
    tables = det.detect({_SOURCE: _GRID})
    assert len(tables) == 1
    t = tables[0]
    assert t.headers == ["Time (h)", "Biomass", "Temp"]
    assert t.rows == [["0", "0.5", "297"], ["2", "1.2", "298"]]
    # confidence capped at 0.85
    assert t.locator["structural_confidence"] == 0.85
    # single_run -> per-sheet run id so multi-sheet workbooks fan out
    assert t.locator["run_id"] == "B471"
    assert t.locator["layout_detected"] is True


def test_wrong_header_row_fails_evidence():
    # LLM points at row 0 (the title) but cites the real header cells; those
    # cells are NOT in row 0, so the evidence gate drops the region.
    region = _region(header_row=0, data_start_row=1, data_end_row=5)
    det = LayoutDetector(_ScriptedClient({"tables": [region]}))
    assert det.detect({_SOURCE: _GRID}) == []


def test_out_of_bounds_region_dropped():
    region = _region(header_row=99, data_start_row=100, data_end_row=101)
    det = LayoutDetector(_ScriptedClient({"tables": [region]}))
    assert det.detect({_SOURCE: _GRID}) == []


def test_data_before_header_dropped():
    region = _region(header_row=3, data_start_row=1, data_end_row=2)
    det = LayoutDetector(_ScriptedClient({"tables": [region]}))
    assert det.detect({_SOURCE: _GRID}) == []


def test_run_id_column_grouping_sets_column_not_run_id():
    region = _region(run_grouping="run_id_column", run_id_column=0)
    det = LayoutDetector(_ScriptedClient({"tables": [region]}))
    t = det.detect({_SOURCE: _GRID})[0]
    assert "run_id" not in t.locator
    assert t.locator["run_id_column"] == 0


def test_digest_keeps_deep_table_and_real_indices():
    # Regression (praaj run b6c116b0): the data table was at row 75 of a
    # 117-row sheet but the old 60-row digest cap hid it -> detector found
    # nothing -> 0 observations. The digest must skip blank rows, keep the
    # ORIGINAL index, and still reach a table buried deep in a sparse sheet.
    grid = [["meta", None]] * 5
    grid += [[None, None]] * 60        # 60 blank rows (would blow the old cap)
    grid += [["Time (h)", "Biomass"]]  # real header at index 65
    grid += [["0", "0.5"], ["2", "1.2"]]
    det = LayoutDetector(_ScriptedClient({"tables": []}))
    digest = det._digest(grid)
    assert "[65] Time (h) | Biomass" in digest  # real index preserved
    assert "[6] " not in digest and "[64] " not in digest  # blank rows skipped

    # And detection at that deep, correctly-indexed row works.
    region = _region(header_row=65, data_start_row=66, data_end_row=67,
                     header_cells=["Time (h)", "Biomass"])
    det2 = LayoutDetector(_ScriptedClient({"tables": [region]}))
    t = det2.detect({"f.xlsx#S": grid})[0]
    assert t.headers == ["Time (h)", "Biomass"]
    assert t.rows == [["0", "0.5"], ["2", "1.2"]]


def test_no_client_returns_empty():
    assert LayoutDetector(None).detect({_SOURCE: _GRID}) == []


def test_llm_error_is_failsoft():
    assert LayoutDetector(_RaisingClient()).detect({_SOURCE: _GRID}) == []


def test_empty_grid_and_missing_tables_key():
    det = LayoutDetector(_ScriptedClient({"tables": [_region()]}))
    assert det.detect({}) == []
    assert det.detect({_SOURCE: []}) == []
    bad = LayoutDetector(_ScriptedClient({"nope": 1}))
    assert bad.detect({_SOURCE: _GRID}) == []


def test_stage_and_time_semantics_ride_in_locator():
    region = _region(stage="MF", time_semantics="reset_per_stage")
    det = LayoutDetector(_ScriptedClient({"tables": [region]}))
    t = det.detect({_SOURCE: _GRID})[0]
    assert t.locator["stage"] == "MF"
    assert t.locator["time_semantics"] == "reset_per_stage"
