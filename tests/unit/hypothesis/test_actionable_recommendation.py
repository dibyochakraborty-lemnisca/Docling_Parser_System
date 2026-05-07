"""Actionable recommendation: schema field + invariants.

Plan ref: plans/2026-05-07-rigour-and-actionability.md commit 4.

Review gap #4: hypotheses are descriptive, not actionable. Fix is a
schema field + synthesizer invariant + critic [actionability-axis] +
judge weighting. Tests pin the prompt-text shape and the schema
back-compat.
"""

from __future__ import annotations

import re

import pytest

from fermdocs_diagnose.schema import ConfidenceBasis
from fermdocs_hypothesis.agents.critic import CRITIC_INVARIANTS
from fermdocs_hypothesis.agents.judge import JUDGE_INVARIANTS
from fermdocs_hypothesis.agents.synthesizer import SYNTHESIZER_INVARIANTS
from fermdocs_hypothesis.schema import FinalHypothesis, HypothesisFull


def _flat(strs: tuple[str, ...]) -> str:
    return re.sub(r"\s+", " ", " ".join(strs))


# ---------- schema ----------


def _hyp_full_kwargs(**overrides):
    base = dict(
        hyp_id="H-0001",
        summary="x",
        facet_ids=["FCT-0001"],
        cited_finding_ids=["F-0001"],
        confidence=0.6,
        confidence_basis=ConfidenceBasis.SCHEMA_ONLY,
    )
    base.update(overrides)
    return base


def _final_kwargs(**overrides):
    base = dict(
        hyp_id="H-0001",
        summary="x",
        facet_ids=["FCT-0001"],
        cited_finding_ids=["F-0001"],
        confidence=0.6,
        confidence_basis=ConfidenceBasis.SCHEMA_ONLY,
        critic_flag="green",
        judge_ruled_criticism_valid=False,
    )
    base.update(overrides)
    return base


def test_hypothesis_full_actionable_recommendation_default_none() -> None:
    h = HypothesisFull(**_hyp_full_kwargs())
    assert h.actionable_recommendation is None


def test_hypothesis_full_actionable_recommendation_populated() -> None:
    h = HypothesisFull(
        **_hyp_full_kwargs(
            actionable_recommendation="Design Batch 7 with DO setpoint ≥40% during fed-batch."
        )
    )
    assert h.actionable_recommendation is not None
    assert "Batch 7" in h.actionable_recommendation


def test_final_hypothesis_actionable_recommendation_default_none() -> None:
    """REGRESSION: existing fixtures load without migration."""
    h = FinalHypothesis(**_final_kwargs())
    assert h.actionable_recommendation is None


def test_final_hypothesis_accepts_insufficient_evidence_string() -> None:
    h = FinalHypothesis(
        **_final_kwargs(
            actionable_recommendation="insufficient evidence to recommend: bundle has only 2 runs"
        )
    )
    assert h.actionable_recommendation is not None
    assert h.actionable_recommendation.startswith("insufficient evidence")


def test_final_hypothesis_recommendation_length_capped() -> None:
    """600-char Pydantic max_length enforced."""
    with pytest.raises(ValueError):
        FinalHypothesis(**_final_kwargs(actionable_recommendation="x" * 601))


# ---------- synthesizer ----------


def test_synthesizer_has_actionable_recommendation_invariant() -> None:
    flat = _flat(SYNTHESIZER_INVARIANTS)
    assert "ACTIONABLE RECOMMENDATION" in flat
    assert "actionable_recommendation" in flat


def test_synthesizer_invariant_names_concrete_format() -> None:
    flat = _flat(SYNTHESIZER_INVARIANTS)
    assert "Design Batch" in flat or "Repeat Batch" in flat


def test_synthesizer_invariant_allows_insufficient_evidence_abstention() -> None:
    flat = _flat(SYNTHESIZER_INVARIANTS)
    assert "insufficient evidence to recommend" in flat


# ---------- critic ----------


def test_critic_has_actionability_axis_rule() -> None:
    flat = _flat(CRITIC_INVARIANTS)
    assert "[ACTIONABILITY-AXIS]" in flat
    assert "[actionability-axis]" in flat


def test_critic_actionability_axis_anti_overfire() -> None:
    """'insufficient evidence' prefix exempts the hypothesis."""
    flat = _flat(CRITIC_INVARIANTS)
    assert "over-fire" in flat.lower()
    assert "insufficient evidence" in flat.lower()


# ---------- judge ----------


def test_judge_weighs_actionability_axis_critiques() -> None:
    flat = _flat(JUDGE_INVARIANTS)
    assert "[ACTIONABILITY-AXIS]" in flat
    assert "insufficient evidence" in flat.lower()


def test_judge_does_not_uphold_actionability_on_red_flagged() -> None:
    flat = _flat(JUDGE_INVARIANTS)
    assert "red-flagged" in flat.lower() or "red flag" in flat.lower()


# ---------- REGRESSION ----------


def test_synthesizer_invariants_still_have_prior_rules() -> None:
    flat = _flat(SYNTHESIZER_INVARIANTS)
    assert "USER QUESTION" in flat
    assert "ROBUST STATISTICS" in flat
    assert "TRAJECTORY CITATION" in flat
    assert "TOOL GAP vs DATA GAP" in flat


def test_critic_invariants_have_all_five_axes() -> None:
    flat = _flat(CRITIC_INVARIANTS).lower()
    for axis in (
        "[question-axis]",
        "[tool-gap-axis]",
        "[trajectory-axis]",
        "[robustness-axis]",
        "[actionability-axis]",
    ):
        assert axis in flat
