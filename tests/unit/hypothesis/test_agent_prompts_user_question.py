"""Hypothesis-stage agent prompts + synthesizer emit-path for user_question.

PR-A on caisc-hitl, commit 6. Plan:
plans/2026-05-04-user-question-and-hitl.md.

Six surfaces touched:
  1. specialist_base.make_user_question_invariant() — shared helper
  2. specialist_kinetics / mass_transfer / metabolic — invariant added
  3. synthesizer prompt — populate question_answered + summary contract
  4. synthesizer emit schema + builder — extract those fields when
     view.user_question is non-null, else null-out (back-compat)
  5. critic invariants — [question-axis] rejection rule
  6. judge invariants — uphold [question-axis] critiques unless
     answer was honestly insufficient_data
  7. runner._build_final — carries hyp.question_answered/summary forward
"""

from __future__ import annotations

from datetime import datetime, timezone

from fermdocs.domain.user_question import UserQuestion
from fermdocs_diagnose.schema import ConfidenceBasis, TrajectoryRef
from fermdocs_hypothesis.agents.critic import CRITIC_INVARIANTS
from fermdocs_hypothesis.agents.judge import JUDGE_INVARIANTS
from fermdocs_hypothesis.agents.specialist_base import make_user_question_invariant
from fermdocs_hypothesis.agents.specialist_kinetics import (
    SPECIALIST_SPEC as KINETICS_SPEC,
)
from fermdocs_hypothesis.agents.specialist_mass_transfer import (
    SPECIALIST_SPEC as MASS_TRANSFER_SPEC,
)
from fermdocs_hypothesis.agents.specialist_metabolic import (
    SPECIALIST_SPEC as METABOLIC_SPEC,
)
from fermdocs_hypothesis.agents.synthesizer import (
    SYNTHESIZER_INVARIANTS,
    SynthesizerAgent,
)
from fermdocs_hypothesis.runner import _build_final
from fermdocs_hypothesis.schema import (
    CitationCatalog,
    CritiqueFull,
    HypothesisFull,
    SynthesizerView,
    TopicSpec,
)


# ---------- specialist factored helper ----------


def test_specialist_invariant_helper_includes_role_name() -> None:
    helper = make_user_question_invariant("kinetics")
    assert "kinetics" in helper
    assert "user_question" in helper


def test_each_specialist_spec_contains_user_question_invariant() -> None:
    for spec, role in (
        (KINETICS_SPEC, "kinetics"),
        (MASS_TRANSFER_SPEC, "mass_transfer"),
        (METABOLIC_SPEC, "metabolic"),
    ):
        joined = " ".join(spec["invariants"])
        assert "user_question" in joined, f"{role} missing user_question invariant"
        # Role name appears (factored helper interpolated it).
        assert role in joined


def test_specialist_invariant_says_explicit_skip_required() -> None:
    """Avoid silent-skip behavior: the rule REQUIRES the persona to either
    address the question or explicitly say so."""
    helper = make_user_question_invariant("kinetics")
    assert "EXPLICITLY" in helper
    assert "silently skip" in helper.lower() or "do not silently" in helper.lower()


# ---------- synthesizer prompt ----------


def test_synthesizer_invariants_include_user_question_rule() -> None:
    joined = " ".join(SYNTHESIZER_INVARIANTS)
    assert "USER QUESTION" in joined
    assert "question_answered" in joined
    assert "question_response_summary" in joined


def test_synthesizer_invariants_cover_all_three_answered_values() -> None:
    joined = " ".join(SYNTHESIZER_INVARIANTS)
    assert "yes" in joined
    assert "partial" in joined
    assert "insufficient_data" in joined


def test_synthesizer_invariants_warn_against_fake_answers() -> None:
    """The 'don't fake an answer' rule: insufficient_data is honest."""
    joined = " ".join(SYNTHESIZER_INVARIANTS)
    assert "honest" in joined.lower() or "fake" in joined.lower()


# ---------- synthesizer emit-path ----------


class _ScriptedClient:
    def __init__(self, parsed: dict) -> None:
        self._parsed = parsed

    def call(self, *, system: str, user_text: str, response_schema: dict, temperature: float = 0.0):
        return self._parsed, 100, 30


def _topic_with_facet():
    from fermdocs_hypothesis.schema import FacetFull, TopicSourceType

    topic = TopicSpec(
        topic_id="T-0001",
        summary="biomass plateau",
        source_type=TopicSourceType.FAILURE,
        affected_variables=["biomass_g_l"],
        cited_finding_ids=["foo:F-0001"],
        cited_narrative_ids=[],
        cited_trajectories=[],
    )
    facet = FacetFull(
        facet_id="FCT-0001",
        specialist="kinetics",
        summary="kinetic facet",
        cited_finding_ids=["foo:F-0001"],
        cited_narrative_ids=[],
        cited_trajectories=[],
        affected_variables=["biomass_g_l"],
        confidence=0.7,
        confidence_basis=ConfidenceBasis.SCHEMA_ONLY,
    )
    return topic, facet


def _view_with_question(q: UserQuestion | None) -> SynthesizerView:
    topic, facet = _topic_with_facet()
    return SynthesizerView(
        current_topic=topic,
        facets=[facet],
        citation_universe=CitationCatalog(
            finding_ids=["foo:F-0001"], narrative_ids=[], trajectories=[]
        ),
        user_question=q,
    )


def test_synthesizer_populates_question_fields_when_question_present() -> None:
    q = UserQuestion(text="Why?", shape="open")
    view = _view_with_question(q)
    parsed = {
        "summary": "answers the question",
        "facet_ids": ["FCT-0001"],
        "cited_finding_ids": ["foo:F-0001"],
        "confidence": 0.8,
        "confidence_basis": "schema_only",
        "question_answered": "yes",
        "question_response_summary": "The bundle showed clear biomass plateau at 24h.",
    }
    agent = SynthesizerAgent(_ScriptedClient(parsed))
    result = agent.synthesize(view, hyp_id="H-0001")
    assert result.hypothesis.question_answered == "yes"
    assert "biomass plateau" in result.hypothesis.question_response_summary


def test_synthesizer_nulls_question_fields_when_view_has_no_question() -> None:
    """Back-compat invariant: if the view's user_question is None, the
    synthesizer must NOT populate question_answered even if the LLM
    response provides it. Avoids fabrication on legacy runs."""
    view = _view_with_question(None)
    parsed = {
        "summary": "no question to answer",
        "facet_ids": ["FCT-0001"],
        "cited_finding_ids": ["foo:F-0001"],
        "confidence": 0.8,
        "confidence_basis": "schema_only",
        # LLM tries to populate these even though view has no question.
        "question_answered": "yes",
        "question_response_summary": "fake answer",
    }
    agent = SynthesizerAgent(_ScriptedClient(parsed))
    result = agent.synthesize(view, hyp_id="H-0001")
    assert result.hypothesis.question_answered is None
    assert result.hypothesis.question_response_summary is None


def test_synthesizer_drops_unknown_question_answered_value() -> None:
    """LLM hallucinates 'maybe' as question_answered → drop to None."""
    q = UserQuestion(text="?")
    view = _view_with_question(q)
    parsed = {
        "summary": "x",
        "facet_ids": ["FCT-0001"],
        "cited_finding_ids": ["foo:F-0001"],
        "confidence": 0.8,
        "confidence_basis": "schema_only",
        "question_answered": "maybe",
    }
    agent = SynthesizerAgent(_ScriptedClient(parsed))
    result = agent.synthesize(view, hyp_id="H-0001")
    assert result.hypothesis.question_answered is None


def test_synthesizer_caps_question_response_summary_at_800() -> None:
    q = UserQuestion(text="?")
    view = _view_with_question(q)
    parsed = {
        "summary": "x",
        "facet_ids": ["FCT-0001"],
        "cited_finding_ids": ["foo:F-0001"],
        "confidence": 0.8,
        "confidence_basis": "schema_only",
        "question_response_summary": "y" * 1500,
    }
    agent = SynthesizerAgent(_ScriptedClient(parsed))
    result = agent.synthesize(view, hyp_id="H-0001")
    assert result.hypothesis.question_response_summary is not None
    assert len(result.hypothesis.question_response_summary) <= 800


# ---------- critic ----------


def test_critic_invariants_include_question_axis_rule() -> None:
    joined = " ".join(CRITIC_INVARIANTS)
    assert "USER QUESTION" in joined
    assert "[question-axis]" in joined


def test_critic_invariants_explain_insufficient_data_is_acceptable() -> None:
    joined = " ".join(CRITIC_INVARIANTS)
    assert "insufficient_data" in joined
    # The rule must say not to reject on this axis when answer is honest insufficient_data.
    assert "do NOT reject" in joined or "do not reject" in joined.lower()


# ---------- judge ----------


def test_judge_invariants_include_question_axis_rule() -> None:
    joined = " ".join(JUDGE_INVARIANTS)
    assert "USER QUESTION" in joined
    assert "[question-axis]" in joined


def test_judge_invariants_uphold_insufficient_data_honesty() -> None:
    joined = " ".join(JUDGE_INVARIANTS)
    assert "insufficient_data" in joined
    assert "honest" in joined.lower() or "do NOT uphold" in joined


# ---------- runner._build_final carries fields forward ----------


def test_build_final_carries_question_fields() -> None:
    hyp = HypothesisFull(
        hyp_id="H-0001",
        summary="x",
        facet_ids=["FCT-0001"],
        cited_finding_ids=["foo:F-0001"],
        confidence=0.8,
        confidence_basis=ConfidenceBasis.SCHEMA_ONLY,
        question_answered="partial",
        question_response_summary="some answer",
    )
    crit = CritiqueFull(
        critique_id="CRIT-0001",
        hyp_id=hyp.hyp_id,
        flag="green",
        reasons=[],
        cited_finding_ids=[],
        cited_narrative_ids=[],
        cited_trajectories=[],
        cited_priors=[],
    )
    final = _build_final(hyp, crit, judge_valid=False)
    assert final.question_answered == "partial"
    assert final.question_response_summary == "some answer"


def test_build_final_passes_through_none_on_legacy_runs() -> None:
    hyp = HypothesisFull(
        hyp_id="H-0001",
        summary="x",
        facet_ids=["FCT-0001"],
        cited_finding_ids=["foo:F-0001"],
        confidence=0.8,
        confidence_basis=ConfidenceBasis.SCHEMA_ONLY,
        # No question fields populated.
    )
    crit = CritiqueFull(
        critique_id="CRIT-0001",
        hyp_id=hyp.hyp_id,
        flag="green",
        reasons=[],
        cited_finding_ids=[],
        cited_narrative_ids=[],
        cited_trajectories=[],
        cited_priors=[],
    )
    final = _build_final(hyp, crit, judge_valid=False)
    assert final.question_answered is None
    assert final.question_response_summary is None
