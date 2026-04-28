from __future__ import annotations

from fermdocs.domain.models import ConfidenceBand

AUTO_THRESHOLD = 0.85
REVIEW_THRESHOLD = 0.60


def band(confidence: float) -> ConfidenceBand:
    """Map a self-reported LLM confidence to a band.

    Treat the score as a rank, not a calibrated probability. Thresholds are
    constants here so tuning is one-edit; downstream code never hardcodes them.
    """
    if confidence >= AUTO_THRESHOLD:
        return ConfidenceBand.AUTO
    if confidence >= REVIEW_THRESHOLD:
        return ConfidenceBand.NEEDS_REVIEW
    return ConfidenceBand.RESIDUAL
