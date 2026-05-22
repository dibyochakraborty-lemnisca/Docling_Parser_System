"""Synthesizer + critic prompt rules for metadata anomalies.

Reviewer A1 (metadata-anomaly-prepass): when characterize emits
metadata_anomaly findings (instrument change, header drift, h0 outlier,
scale change, bioreactor change), the synthesizer must acknowledge the
confound in any cross-batch claim, and the critic must reject silent
cross-confound comparisons.
"""

from __future__ import annotations

import re

from fermdocs_hypothesis.agents.critic import CRITIC_INVARIANTS
from fermdocs_hypothesis.agents.synthesizer import SYNTHESIZER_INVARIANTS


def _flat(strs: tuple[str, ...]) -> str:
    return re.sub(r"\s+", " ", " ".join(strs))


# ---------- synthesizer ----------


def test_synthesizer_has_metadata_anomaly_surfacing_invariant() -> None:
    flat = _flat(SYNTHESIZER_INVARIANTS)
    assert "METADATA-ANOMALY SURFACING" in flat
    assert "pattern_kind='metadata_anomaly'" in flat


def test_synthesizer_invariant_lists_anomaly_kinds() -> None:
    flat = _flat(SYNTHESIZER_INVARIANTS)
    for kind in (
        "instrument_change",
        "header_inconsistency",
        "h0_outlier",
        "scale_change",
        "bioreactor_change",
    ):
        assert kind in flat, f"missing anomaly kind: {kind}"


def test_synthesizer_invariant_offers_two_acceptable_responses() -> None:
    """Cite-and-explain OR downgrade-claim."""
    flat = _flat(SYNTHESIZER_INVARIANTS)
    assert "cite the metadata-anomaly finding" in flat
    assert "downgrade the claim" in flat


# ---------- critic ----------


def test_critic_has_metadata_axis_rule() -> None:
    flat = _flat(CRITIC_INVARIANTS)
    assert "[METADATA-AXIS]" in flat
    assert "[metadata-axis]" in flat  # lowercase tag for retry parsing


def test_critic_metadata_axis_anti_overfire() -> None:
    flat = _flat(CRITIC_INVARIANTS)
    assert "over-fire" in flat.lower()
    assert "no metadata-anomaly findings" in flat.lower() or \
           "already scoped within a single" in flat.lower()


# ---------- coexistence ----------


def test_metadata_axis_distinct_from_other_axes() -> None:
    flat = _flat(CRITIC_INVARIANTS).lower()
    for axis in (
        "[metadata-axis]",
        "[trajectory-axis]",
        "[robustness-axis]",
        "[actionability-axis]",
        "[question-axis]",
        "[tool-gap-axis]",
    ):
        assert axis in flat, f"missing axis: {axis}"
