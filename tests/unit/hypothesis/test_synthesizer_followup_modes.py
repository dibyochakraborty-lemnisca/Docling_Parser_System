"""make_followup_invariants returns the right rules per shape.

PR-A2 commit 4. Plan ref: plans/2026-05-05-hitl-followup.md commit 4.

These tests assert the prompt-text shape that fires per follow-up
shape. They don't run real LLM calls — quality eval lands when we
exercise this in production (per the plan's NOT-in-scope note).
"""

from __future__ import annotations

import re

from fermdocs.domain.user_question import UserQuestion
from fermdocs_diagnose.schema import ConfidenceBasis, TrajectoryRef
from fermdocs_hypothesis.agents.synthesizer import (
    SYNTHESIZER_INVARIANTS,
    make_followup_invariants,
)
from fermdocs_hypothesis.schema import (
    CitationCatalog,
    FacetFull,
    SynthesizerView,
    TopicSourceType,
    TopicSpec,
)


def _flat(strs: tuple[str, ...]) -> str:
    return re.sub(r"\s+", " ", " ".join(strs))


def _facet(facet_id="FCT-0001") -> FacetFull:
    return FacetFull(
        facet_id=facet_id,
        specialist="kinetics",
        summary="kinetic facet",
        cited_finding_ids=["F-0001"],
        confidence=0.6,
        confidence_basis=ConfidenceBasis.SCHEMA_ONLY,
    )


def _topic(*, source_type: TopicSourceType = TopicSourceType.FAILURE) -> TopicSpec:
    return TopicSpec(
        topic_id="T-0001",
        summary="topic",
        source_type=source_type,
        cited_finding_ids=["F-0001"],
    )


def _view(
    *,
    user_question: UserQuestion | None,
    source_type: TopicSourceType = TopicSourceType.FAILURE,
) -> SynthesizerView:
    return SynthesizerView(
        current_topic=_topic(source_type=source_type),
        facets=[_facet()],
        citation_universe=CitationCatalog(finding_ids=["F-0001"]),
        user_question=user_question,
    )


# ---------- back-compat: empty rules on bias / legacy runs ----------


def test_no_user_question_returns_empty_invariants() -> None:
    view = _view(user_question=None)
    assert make_followup_invariants(view) == ()


def test_user_question_raised_by_user_returns_empty() -> None:
    """raised_by='user' is bias posture (PR-A). Drive-mode rules must
    NOT fire — the bias-posture rules in SYNTHESIZER_INVARIANTS already
    handle the user_question field."""
    q = UserQuestion(text="why?", raised_by="user", shape="open")
    view = _view(user_question=q)
    assert make_followup_invariants(view) == ()


# ---------- mechanistic shape ----------


def test_mechanistic_shape_emits_for_against_rule() -> None:
    q = UserQuestion(
        text="Was oxygen limitation the cause?",
        raised_by="user_followup",
        shape="mechanistic",
    )
    view = _view(user_question=q, source_type=TopicSourceType.USER_MECHANISM)
    rules = make_followup_invariants(view)
    flat = _flat(rules)
    assert "MECHANISTIC" in flat
    assert "FOR evidence" in flat
    assert "AGAINST evidence" in flat
    assert "insufficient_data" in flat


def test_mechanistic_shape_via_source_type_only() -> None:
    """Even if question.shape is None, USER_MECHANISM source_type
    triggers the rule (defense-in-depth)."""
    q = UserQuestion(text="?", raised_by="user_followup")
    view = _view(user_question=q, source_type=TopicSourceType.USER_MECHANISM)
    flat = _flat(make_followup_invariants(view))
    assert "MECHANISTIC" in flat


# ---------- comparative shape ----------


def test_comparative_shape_emits_side_by_side_rule() -> None:
    q = UserQuestion(
        text="Compare A to B",
        raised_by="user_followup",
        shape="comparative",
    )
    view = _view(user_question=q, source_type=TopicSourceType.USER_COMPARISON)
    flat = _flat(make_followup_invariants(view))
    assert "COMPARATIVE" in flat
    assert "side-by-side" in flat
    assert "Both named groups" in flat


# ---------- scoping shape: empty match ----------


def test_user_scope_source_emits_insufficient_data_rule() -> None:
    q = UserQuestion(
        text="What about RUN-9999?",
        raised_by="user_followup",
        shape="scoping",
    )
    view = _view(user_question=q, source_type=TopicSourceType.USER_SCOPE)
    flat = _flat(make_followup_invariants(view))
    assert "USER_SCOPE" in flat
    assert "insufficient_data" in flat
    assert "available runs" in flat.lower()


# ---------- scoping shape: in-scope topic ----------


def test_scoping_shape_in_scope_emits_narrow_rule() -> None:
    q = UserQuestion(
        text="Focus on biomass",
        raised_by="user_followup",
        shape="scoping",
    )
    # Real diag-derived topic survived the filter (source_type unchanged)
    view = _view(user_question=q, source_type=TopicSourceType.FAILURE)
    flat = _flat(make_followup_invariants(view))
    assert "SCOPING" in flat
    assert "narrow" in flat.lower() or "STRICTLY" in flat


# ---------- open shape ----------


def test_open_shape_emits_dont_redo_rule() -> None:
    q = UserQuestion(
        text="Different framing please",
        raised_by="user_followup",
        shape="open",
    )
    view = _view(user_question=q, source_type=TopicSourceType.FAILURE)
    flat = _flat(make_followup_invariants(view))
    assert "OPEN-SHAPE" in flat
    assert "parent_hypothesis_ids" in flat


# ---------- common header on every follow-up ----------


def test_all_followups_get_the_drive_header() -> None:
    """Every shape gets the same 'you're driving, not biasing' header."""
    for shape in ("mechanistic", "comparative", "scoping", "open"):
        q = UserQuestion(
            text="?", raised_by="user_followup", shape=shape  # type: ignore[arg-type]
        )
        view = _view(user_question=q)
        flat = _flat(make_followup_invariants(view))
        assert "FOLLOW-UP MODE" in flat, f"shape={shape}"


# ---------- composition: SYNTHESIZER_INVARIANTS unchanged ----------


def test_synthesizer_invariants_still_has_user_question_rule() -> None:
    """Bias-posture user_question rule from PR-A is intact."""
    flat = _flat(SYNTHESIZER_INVARIANTS)
    assert "USER QUESTION" in flat
    assert "question_answered" in flat
