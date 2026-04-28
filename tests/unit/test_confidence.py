from __future__ import annotations

from fermdocs.domain.models import ConfidenceBand
from fermdocs.mapping.confidence import band


def test_band_thresholds():
    assert band(0.95) == ConfidenceBand.AUTO
    assert band(0.85) == ConfidenceBand.AUTO
    assert band(0.80) == ConfidenceBand.NEEDS_REVIEW
    assert band(0.60) == ConfidenceBand.NEEDS_REVIEW
    assert band(0.59) == ConfidenceBand.RESIDUAL
    assert band(0.0) == ConfidenceBand.RESIDUAL
