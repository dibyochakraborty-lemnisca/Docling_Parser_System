"""B1' — the factor-extraction digest must NOT truncate free-text strategy cells.

The bug: _digest capped every cell at 24 chars (tuned for table-header detection),
which destroyed the strategy prose where the feed window and campaign target live
(~char 95 of a 300+ char cell). Factor mining passes a generous per-cell budget so
that prose reaches the LLM. This test locks the structural fix deterministically
(no LLM): the long cell survives at the factor budget, truncates at the default.
"""
from __future__ import annotations

from fermdocs.mapping.layout_detector import LayoutDetector

# A praaj-shaped strategy cell: the feed window sits ~90 chars in.
_STRATEGY = ("Jaihind Sugars - Raw Sugar; 15g/L Pretreated DBY in MF; "
             "Fed Batch Mode with 7 hrs to 9 hrs feeding pattern; 4 stages PF.")
_GRID = [
    [None, "Key Changes/Strategies", _STRATEGY],
    [None, "Reactor", "100L"],
]


def test_default_digest_truncates_strategy_prose():
    det = LayoutDetector(client=None)               # default 24-char cell cap
    digest = det._digest(_GRID)
    assert "feeding pattern" not in digest          # the bug: prose is lost
    assert "7 hrs to 9 hrs" not in digest


def test_factor_budget_preserves_strategy_prose():
    det = LayoutDetector(client=None)
    digest = det._digest(_GRID, max_cell_chars=800)  # factor-mining budget
    assert "Fed Batch Mode with 7 hrs to 9 hrs feeding pattern" in digest
    assert "feeding pattern" in digest


def test_factor_budget_is_what_extract_uses():
    # Guard against regressing the budget back down: the value extract_design_factors
    # passes must be large enough to clear a typical strategy cell (>= 300 chars).
    import inspect
    from fermdocs.mapping import layout_detector
    src = inspect.getsource(layout_detector.LayoutDetector.extract_design_factors)
    assert "max_cell_chars=" in src
    budget = int(src.split("max_cell_chars=")[1].split(")")[0].split(",")[0])
    assert budget >= 300
