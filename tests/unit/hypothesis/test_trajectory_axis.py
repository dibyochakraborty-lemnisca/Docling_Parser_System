"""Trajectory-axis: synthesizer + critic + judge invariants.

Plan ref: plans/2026-05-07-rigour-and-actionability.md commit 1.

The 7.5/10 review flagged that hypotheses report 0 trajectory citations
despite OD/WCW/pO2 every 6h being available. These tests pin the
prompt-text shape that pushes synthesizer to cite trajectories on
dynamic claims, that lets critic file [trajectory-axis] red, and that
tells judge to weigh those red flags as legitimate.

No LLM calls — prompt text only.
"""

from __future__ import annotations

import re

from fermdocs_hypothesis.agents.critic import CRITIC_INVARIANTS
from fermdocs_hypothesis.agents.judge import JUDGE_INVARIANTS
from fermdocs_hypothesis.agents.synthesizer import SYNTHESIZER_INVARIANTS


def _flat(strs: tuple[str, ...]) -> str:
    return re.sub(r"\s+", " ", " ".join(strs))


# ---------- synthesizer ----------


def test_synthesizer_has_trajectory_citation_invariant() -> None:
    flat = _flat(SYNTHESIZER_INVARIANTS)
    assert "TRAJECTORY CITATION" in flat
    # Must name the time-dependent trigger words so the LLM has concrete cues.
    for cue in ("decline", "peak", "transient", "rate", "kinetic"):
        assert cue in flat, f"missing time-dependent cue: {cue}"


def test_synthesizer_invariant_tells_model_to_weaken_when_no_traj_available() -> None:
    """Anti-fabrication clause: if no matching trajectory, weaken the
    claim rather than invent a citation."""
    flat = _flat(SYNTHESIZER_INVARIANTS)
    assert "don't fabricate" in flat.lower() or "do not fabricate" in flat.lower()
    assert "weaken" in flat.lower() or "point-in-time" in flat.lower()


# ---------- critic ----------


def test_critic_has_trajectory_axis_rule() -> None:
    flat = _flat(CRITIC_INVARIANTS)
    assert "[TRAJECTORY-AXIS]" in flat
    assert "trajectory-axis" in flat  # lowercase tag for retry parsing


def test_critic_trajectory_axis_has_anti_overfire_clause() -> None:
    """Don't fire when trajectories are genuinely absent from the bundle.
    Symmetric to [tool-gap-axis] design — the critic must distinguish
    'agent ignored available evidence' from 'evidence simply isn't there'."""
    flat = _flat(CRITIC_INVARIANTS)
    assert "over-fire" in flat.lower()
    assert "genuinely not be in the bundle" in flat.lower()


def test_critic_trajectory_axis_distinct_from_question_and_tool_gap() -> None:
    """Three axes coexist; each retry needs its own fix."""
    flat = _flat(CRITIC_INVARIANTS)
    assert "[trajectory-axis]" in flat.lower()
    assert "[question-axis]" in flat.lower()
    assert "[tool-gap-axis]" in flat.lower()


# ---------- judge ----------


def test_judge_weighs_trajectory_axis_critiques() -> None:
    flat = _flat(JUDGE_INVARIANTS)
    assert "[TRAJECTORY-AXIS]" in flat
    assert "weigh" in flat.lower() or "uphold" in flat.lower()


def test_judge_does_not_uphold_when_trajectories_genuinely_absent() -> None:
    flat = _flat(JUDGE_INVARIANTS)
    assert "genuinely absent" in flat.lower()


# ---------- REGRESSION ----------


def test_synthesizer_invariants_still_have_user_question_and_robust_stats() -> None:
    """Adding TRAJECTORY CITATION must not displace prior PR invariants."""
    flat = _flat(SYNTHESIZER_INVARIANTS)
    assert "USER QUESTION" in flat
    assert "ROBUST STATISTICS" in flat
    assert "TOOL GAP vs DATA GAP" in flat
    assert "CROSS-METRIC TRIANGULATION" in flat


def test_critic_invariants_still_have_prior_axes() -> None:
    flat = _flat(CRITIC_INVARIANTS)
    assert "USER QUESTION" in flat
    assert "[TOOL-GAP-AXIS]" in flat


def test_judge_invariants_still_have_user_question_rule() -> None:
    flat = _flat(JUDGE_INVARIANTS)
    assert "USER QUESTION" in flat
