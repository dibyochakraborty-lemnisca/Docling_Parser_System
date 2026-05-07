"""Critic [robustness-axis]: weak-n correlation must carry n/CI caveat.

Plan ref: plans/2026-05-07-rigour-and-actionability.md commit 2.

Review gap #1: r=-0.90 with n=6 shipped without confidence interval.
The critic axis is the runtime defense; the toolkit (test_correlation_ci)
produces the n/CI numbers the synthesizer must surface.
"""

from __future__ import annotations

import re

from fermdocs_hypothesis.agents.critic import CRITIC_INVARIANTS


def _flat(strs: tuple[str, ...]) -> str:
    return re.sub(r"\s+", " ", " ".join(strs))


def test_critic_has_robustness_axis_rule() -> None:
    flat = _flat(CRITIC_INVARIANTS)
    assert "[ROBUSTNESS-AXIS]" in flat
    assert "[robustness-axis]" in flat  # lowercase tag for retry parsing


def test_robustness_axis_names_weak_n_signal() -> None:
    flat = _flat(CRITIC_INVARIANTS)
    assert "weak_n_flag" in flat
    assert "n<8" in flat or "below 8" in flat


def test_robustness_axis_lists_three_acceptable_responses() -> None:
    """Synthesizer can: name n, cite CI, or downgrade language."""
    flat = _flat(CRITIC_INVARIANTS)
    assert "name the n" in flat.lower()
    assert "bootstrap ci" in flat.lower() or "cite the bootstrap ci" in flat.lower()
    assert "downgrade" in flat.lower() or "preliminary association" in flat.lower()


def test_robustness_axis_has_anti_overfire_clause() -> None:
    flat = _flat(CRITIC_INVARIANTS)
    assert "over-fire" in flat.lower()
    assert "n≥8" in flat or "already includes the caveat" in flat.lower()


# ---------- REGRESSION ----------


def test_robustness_axis_distinct_from_other_axes() -> None:
    flat = _flat(CRITIC_INVARIANTS).lower()
    assert "[robustness-axis]" in flat
    assert "[trajectory-axis]" in flat
    assert "[question-axis]" in flat
    assert "[tool-gap-axis]" in flat
