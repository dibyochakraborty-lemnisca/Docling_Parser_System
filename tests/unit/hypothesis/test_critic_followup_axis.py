"""make_followup_critic_invariants returns the right [followup-axis] rules.

PR-A2 commit 5. Plan ref: plans/2026-05-05-hitl-followup.md commit 5.

These tests assert the prompt-text shape that fires per follow-up
shape — what triggers a [followup-axis] rejection. They don't run
real LLM calls.
"""

from __future__ import annotations

import re

from fermdocs.domain.user_question import UserQuestion
from fermdocs_diagnose.schema import ConfidenceBasis, TrajectoryRef
from fermdocs_hypothesis.agents.critic import (
    CRITIC_INVARIANTS,
    make_followup_critic_invariants,
)
from fermdocs_hypothesis.schema import CriticView, HypothesisFull


def _flat(strs: tuple[str, ...]) -> str:
    return re.sub(r"\s+", " ", " ".join(strs))


def _hyp(**overrides) -> HypothesisFull:
    base = dict(
        hyp_id="H-0001",
        summary="placeholder",
        facet_ids=["FCT-0001"],
        cited_finding_ids=["F-0001"],
        confidence=0.6,
        confidence_basis=ConfidenceBasis.SCHEMA_ONLY,
    )
    base.update(overrides)
    return HypothesisFull(**base)


def _view(*, user_question: UserQuestion | None) -> CriticView:
    return CriticView(
        hypothesis=_hyp(),
        citation_lookups={},
        user_question=user_question,
    )


# ---------- back-compat: empty rules on bias / legacy ----------


def test_no_user_question_returns_empty() -> None:
    assert make_followup_critic_invariants(_view(user_question=None)) == ()


def test_user_question_raised_by_user_returns_empty() -> None:
    """raised_by='user' is bias posture — drive-mode axis must NOT fire."""
    q = UserQuestion(text="?", raised_by="user", shape="open")
    assert make_followup_critic_invariants(_view(user_question=q)) == ()


# ---------- mechanistic shape ----------


def test_mechanistic_emits_restate_and_disconfirming_axes() -> None:
    q = UserQuestion(
        text="Was oxygen limitation the cause?",
        raised_by="user_followup",
        shape="mechanistic",
    )
    flat = _flat(make_followup_critic_invariants(_view(user_question=q)))
    assert "[FOLLOWUP-AXIS] MECHANISTIC" in flat
    assert "restated the mechanism" in flat
    assert "ignored disconfirming evidence" in flat


# ---------- comparative shape ----------


def test_comparative_emits_one_group_only_axis() -> None:
    q = UserQuestion(
        text="Compare A to B",
        raised_by="user_followup",
        shape="comparative",
    )
    flat = _flat(make_followup_critic_invariants(_view(user_question=q)))
    assert "[FOLLOWUP-AXIS] COMPARATIVE" in flat
    assert "only one group" in flat


# ---------- scoping shape ----------


def test_scoping_emits_extrapolated_beyond_scope_axis() -> None:
    q = UserQuestion(
        text="Focus on RUN-0001 biomass",
        raised_by="user_followup",
        shape="scoping",
        affected_runs=["RUN-0001"],
        affected_variables=["biomass_g_l"],
    )
    flat = _flat(make_followup_critic_invariants(_view(user_question=q)))
    assert "[FOLLOWUP-AXIS] SCOPING" in flat
    assert "extrapolated beyond" in flat


# ---------- open shape: no extra axis ----------


def test_open_shape_does_not_add_a_critic_axis() -> None:
    """Open follow-ups don't get a critic axis — the bias-posture
    USER QUESTION rule from PR-A already covers them. Adding more
    rules would over-constrain a re-framing request."""
    q = UserQuestion(
        text="Different framing please",
        raised_by="user_followup",
        shape="open",
    )
    rules = make_followup_critic_invariants(_view(user_question=q))
    assert rules == ()


# ---------- shape=None: defensive empty ----------


def test_shape_none_returns_empty_axis() -> None:
    """Classifier-failure (shape=None) → no follow-up axis. The
    bias-posture USER QUESTION rule still fires (it's in CRITIC_INVARIANTS)."""
    q = UserQuestion(text="?", raised_by="user_followup")
    assert make_followup_critic_invariants(_view(user_question=q)) == ()


# ---------- composition: CRITIC_INVARIANTS unchanged ----------


def test_critic_invariants_still_has_question_axis_rule() -> None:
    """Bias-posture [question-axis] rule from PR-A is intact."""
    flat = _flat(CRITIC_INVARIANTS)
    assert "USER QUESTION" in flat
    assert "[question-axis]" in flat


def test_followup_axis_tag_distinct_from_question_axis() -> None:
    """[followup-axis] is a different tag from [question-axis] so the
    synthesizer's feedback loop can distinguish them on retry."""
    q = UserQuestion(
        text="?",
        raised_by="user_followup",
        shape="mechanistic",
    )
    flat = _flat(make_followup_critic_invariants(_view(user_question=q)))
    assert "[FOLLOWUP-AXIS]" in flat or "[followup-axis]" in flat
