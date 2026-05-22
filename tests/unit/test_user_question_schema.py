"""Schema-level tests for UserQuestion + the FinalHypothesis/HypothesisInput
fields it adds.

PR-A on caisc-hitl. Plan ref: plans/2026-05-04-user-question-and-hitl.md.
"""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from fermdocs.domain.user_question import (
    UserQuestion,
    question_relevance,
)
from fermdocs_characterize.agent_context import AgentContext
from fermdocs_hypothesis.schema import (
    FinalHypothesis,
    HypothesisInput,
    SeedTopic,
)


# ---------- UserQuestion validation ----------


def test_user_question_text_required() -> None:
    with pytest.raises(ValidationError):
        UserQuestion()  # type: ignore[call-arg]


def test_user_question_text_minimum_length() -> None:
    with pytest.raises(ValidationError):
        UserQuestion(text="")


def test_user_question_text_maximum_length() -> None:
    with pytest.raises(ValidationError):
        UserQuestion(text="x" * 2001)


def test_user_question_minimal_construction() -> None:
    q = UserQuestion(text="Why did RUN-0002 plateau early?")
    assert q.text == "Why did RUN-0002 plateau early?"
    assert q.shape is None
    assert q.affected_variables == []
    assert q.affected_runs == []
    assert q.raised_by == "user"


def test_user_question_full_construction() -> None:
    q = UserQuestion(
        text="Compare BATCH-04 and BATCH-05 yields",
        shape="comparative",
        affected_variables=["wcw_g_l"],
        affected_runs=["RUN-0004", "RUN-0005"],
        raised_by="user",
    )
    assert q.shape == "comparative"
    assert "RUN-0004" in q.affected_runs


def test_user_question_is_frozen() -> None:
    q = UserQuestion(text="test")
    with pytest.raises(ValidationError):
        q.text = "mutated"  # type: ignore[misc]


def test_user_question_rejects_unknown_shape() -> None:
    with pytest.raises(ValidationError):
        UserQuestion(text="test", shape="speculative")  # type: ignore[arg-type]


def test_user_question_rejects_unknown_raised_by() -> None:
    with pytest.raises(ValidationError):
        UserQuestion(text="test", raised_by="some_random_actor")  # type: ignore[arg-type]


def test_user_question_caps_affected_variables() -> None:
    with pytest.raises(ValidationError):
        UserQuestion(
            text="test",
            affected_variables=[f"var_{i}" for i in range(11)],
        )


def test_user_question_caps_affected_runs() -> None:
    with pytest.raises(ValidationError):
        UserQuestion(
            text="test",
            affected_runs=[f"RUN-{i:04d}" for i in range(21)],
        )


def test_user_question_accepts_followup_raised_by() -> None:
    q = UserQuestion(text="follow-up", raised_by="user_followup")
    assert q.raised_by == "user_followup"


def test_user_question_accepts_operator_raised_by() -> None:
    q = UserQuestion(text="redirect", raised_by="operator_mid_debate")
    assert q.raised_by == "operator_mid_debate"


# ---------- question_relevance helper ----------


def test_relevance_zero_when_question_none() -> None:
    assert question_relevance(
        affected_variables=["x"], affected_runs=["RUN-0001"], question=None
    ) == 0.0


def test_relevance_zero_when_no_overlap() -> None:
    q = UserQuestion(
        text="?", affected_variables=["paa_mg_l"], affected_runs=["RUN-0099"]
    )
    score = question_relevance(
        affected_variables=["biomass_g_l"],
        affected_runs=["RUN-0001"],
        question=q,
    )
    assert score == 0.0


def test_relevance_variable_overlap_scores_0_4() -> None:
    q = UserQuestion(text="?", affected_variables=["biomass_g_l"])
    score = question_relevance(
        affected_variables=["biomass_g_l"], question=q
    )
    assert score == pytest.approx(0.4)


def test_relevance_run_overlap_scores_0_4() -> None:
    q = UserQuestion(text="?", affected_runs=["RUN-0002"])
    score = question_relevance(affected_runs=["RUN-0002"], question=q)
    assert score == pytest.approx(0.4)


def test_relevance_substring_in_text_scores_0_2() -> None:
    """When the question's text mentions a candidate variable but the
    classifier didn't extract it, substring match still grants 0.2."""
    q = UserQuestion(text="What about wcw values during fed-batch?")
    # NOTE: classifier didn't extract 'wcw_g_l' into affected_variables,
    # but the candidate's wcw_g_l string appears as substring in 'wcw'.
    score = question_relevance(
        affected_variables=["wcw"],  # candidate uses bare 'wcw'
        question=q,
    )
    assert score == pytest.approx(0.2)


def test_relevance_capped_at_1_0() -> None:
    q = UserQuestion(
        text="biomass in RUN-0002",
        affected_variables=["biomass"],
        affected_runs=["RUN-0002"],
    )
    score = question_relevance(
        affected_variables=["biomass"],
        affected_runs=["RUN-0002"],
        question=q,
    )
    # 0.4 + 0.4 + 0.2 = 1.0
    assert score == pytest.approx(1.0)


def test_relevance_case_insensitive_variable_match() -> None:
    q = UserQuestion(text="?", affected_variables=["Biomass_G_L"])
    score = question_relevance(
        affected_variables=["BIOMASS_g_l"], question=q
    )
    assert score == pytest.approx(0.4)


def test_relevance_case_insensitive_run_match() -> None:
    q = UserQuestion(text="?", affected_runs=["run-0002"])
    score = question_relevance(affected_runs=["RUN-0002"], question=q)
    assert score == pytest.approx(0.4)


# ---------- FinalHypothesis fields ----------


def test_final_hypothesis_question_answered_defaults_none() -> None:
    fh = FinalHypothesis(
        hyp_id="H-0001",
        summary="...",
        facet_ids=[],
        confidence=0.5,
        confidence_basis="schema_only",
        critic_flag="green",
        judge_ruled_criticism_valid=False,
        cited_finding_ids=["foo:F-0001"],
    )
    assert fh.question_answered is None
    assert fh.question_response_summary is None


def test_final_hypothesis_question_answered_accepts_yes_partial_insufficient() -> None:
    for value in ("yes", "partial", "insufficient_data"):
        fh = FinalHypothesis(
            hyp_id="H-0001",
            summary="...",
            facet_ids=[],
            confidence=0.5,
            confidence_basis="schema_only",
            critic_flag="green",
            judge_ruled_criticism_valid=False,
            cited_finding_ids=["foo:F-0001"],
            question_answered=value,  # type: ignore[arg-type]
        )
        assert fh.question_answered == value


def test_final_hypothesis_rejects_unknown_answered_value() -> None:
    with pytest.raises(ValidationError):
        FinalHypothesis(
            hyp_id="H-0001",
            summary="...",
            facet_ids=[],
            confidence=0.5,
            confidence_basis="schema_only",
            critic_flag="green",
            judge_ruled_criticism_valid=False,
            cited_finding_ids=["foo:F-0001"],
            question_answered="maybe",  # type: ignore[arg-type]
        )


def test_final_hypothesis_response_summary_caps_at_800() -> None:
    with pytest.raises(ValidationError):
        FinalHypothesis(
            hyp_id="H-0001",
            summary="...",
            facet_ids=[],
            confidence=0.5,
            confidence_basis="schema_only",
            critic_flag="green",
            judge_ruled_criticism_valid=False,
            cited_finding_ids=["foo:F-0001"],
            question_response_summary="x" * 801,
        )


# ---------- HypothesisInput field ----------


def test_hypothesis_input_user_question_defaults_none() -> None:
    hi = HypothesisInput(
        diagnosis={},
        characterization={},
        seed_topics=[],
    )
    assert hi.user_question is None


def test_hypothesis_input_carries_user_question() -> None:
    q = UserQuestion(text="Why?")
    hi = HypothesisInput(
        diagnosis={},
        characterization={},
        seed_topics=[],
        user_question=q,
    )
    assert hi.user_question is q


# ---------- AgentContext field ----------


def test_agent_context_user_question_defaults_none() -> None:
    ctx = AgentContext(process={}, schema_version="1.0")
    assert ctx.user_question is None


def test_agent_context_carries_user_question() -> None:
    q = UserQuestion(text="Why did RUN-0002 plateau?")
    ctx = AgentContext(process={}, schema_version="1.0", user_question=q)
    assert ctx.user_question is q
    assert ctx.user_question.text == "Why did RUN-0002 plateau?"
