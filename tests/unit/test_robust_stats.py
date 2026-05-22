"""Robust statistics helpers + B10 adapter median/IQR + synth prompt.

Plan ref: plans/2026-05-07-characterize-determinism.md commit 4.

The IndPenSim feedback: mean RQ 1.21 vs median RQ 0.98. Mean alone
over-reports overflow when the time-series has transient spikes
(typical for RQ during feed events).

Covers:
  1. central_tendency on symmetric series → mean recommended
  2. central_tendency on right-skewed series → median recommended
  3. central_tendency raises on < 2 finite values
  4. is_skewed flags IndPenSim-like RQ profile (mean 1.21, median 0.98)
  5. is_skewed does NOT flag clean aerobic RQ (mean ~ median)
  6. **REGRESSION**: synthesizer SYNTHESIZER_INVARIANTS includes the
     'prefer median when divergent' rule + 'cite both when >15% diff'
     guidance.
"""

from __future__ import annotations

import pytest

from fermdocs_characterize.toolkit._stats import (
    central_tendency,
    is_skewed,
)
from fermdocs_hypothesis.agents.synthesizer import SYNTHESIZER_INVARIANTS


# ---------- 1. symmetric → mean ----------


def test_central_tendency_symmetric_recommends_mean() -> None:
    """Clean aerobic RQ profile: mean and median nearly identical."""
    values = [0.95, 0.96, 0.96, 0.96, 0.97, 0.97]
    ct = central_tendency(values)
    assert ct.recommended_summary == "mean"
    assert ct.median == pytest.approx(0.96, abs=0.01)


# ---------- 2. right-skewed → median ----------


def test_central_tendency_skewed_recommends_median() -> None:
    """IndPenSim-like RQ: median ~0.98 with a few transient spikes
    pulling mean up to ~1.21. The skew detector flags this."""
    # 12 typical aerobic points + 3 high-RQ feed-event spikes
    values = [0.95, 0.97, 0.98, 0.99, 1.00, 0.98, 0.97, 0.99, 1.01, 0.98, 0.96, 0.97,
              2.5, 2.8, 2.6]
    ct = central_tendency(values)
    # Median anchored ~0.98, mean pulled up by spikes
    assert ct.median < ct.mean
    assert ct.mean / ct.median > 1.15
    assert ct.recommended_summary == "median"


# ---------- 3. raises on insufficient data ----------


def test_central_tendency_raises_on_too_few_points() -> None:
    with pytest.raises(ValueError, match="need >= 2"):
        central_tendency([0.95])
    # NaN-only inputs collapse to empty after filtering
    nan = float("nan")
    with pytest.raises(ValueError):
        central_tendency([nan, nan])


# ---------- 4. is_skewed on IndPenSim-like ----------


def test_is_skewed_flags_indpensim_like_rq_profile() -> None:
    values = [0.95, 0.97, 0.98, 0.99, 1.00, 0.98, 0.97, 0.99, 1.01, 0.98, 0.96, 0.97,
              2.5, 2.8, 2.6]
    assert is_skewed(values) is True


# ---------- 5. is_skewed does NOT flag clean aerobic ----------


def test_is_skewed_clean_aerobic_returns_false() -> None:
    values = [0.95, 0.96, 0.96, 0.96, 0.97, 0.97, 0.96, 0.95]
    assert is_skewed(values) is False


# ---------- 6. REGRESSION: synth prompt rule present ----------


def test_synthesizer_invariants_includes_robust_stats_rule() -> None:
    """REGRESSION: prompt amendment in commit 4 must include the
    robust-stats guidance. Future prompt rewrites that lose this rule
    would re-open the IndPenSim mean-RQ-misleading bug."""
    flat = " ".join(SYNTHESIZER_INVARIANTS)
    assert "ROBUST STATISTICS" in flat
    assert "recommended_summary" in flat
    assert "median" in flat
    assert "mean" in flat
    # Cite-both rule when they disagree materially
    assert "15%" in flat
    # IndPenSim concrete example for grounding
    assert "1.21" in flat or "0.98" in flat
