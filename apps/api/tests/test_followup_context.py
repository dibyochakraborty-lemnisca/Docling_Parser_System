from __future__ import annotations

from datetime import datetime, timezone
from uuid import uuid4

from fermdocs_diagnose.schema import ConfidenceBasis
from fermdocs_hypothesis.schema import (
    BudgetSnapshot,
    FinalHypothesis,
    HypothesisMeta,
    HypothesisOutput,
)
from fermdocs_api.runner_pipeline import _build_followup_context
from fermdocs_api.state import FollowupResult, Run


def _final(hyp_id: str, summary: str) -> FinalHypothesis:
    return FinalHypothesis(
        hyp_id=hyp_id,
        summary=summary,
        facet_ids=["FCT-0001"],
        cited_finding_ids=["char:F-0001"],
        confidence=0.75,
        confidence_basis=ConfidenceBasis.SCHEMA_ONLY,
        supporting_specialists=["kinetics"],
        critic_flag="green",
        judge_ruled_criticism_valid=False,
        question_answered="partial",
        question_response_summary=f"{hyp_id} answered part of the question.",
    )


def _output(*finals: FinalHypothesis) -> HypothesisOutput:
    return HypothesisOutput(
        meta=HypothesisMeta(
            hypothesis_version="test",
            hypothesis_id=uuid4(),
            supersedes_diagnosis_id=uuid4(),
            generation_timestamp=datetime.now(timezone.utc),
            model="stub",
            provider="stub",
            budget_used=BudgetSnapshot(),
        ),
        final_hypotheses=list(finals),
    )


def test_build_followup_context_includes_original_and_prior_followups(tmp_path):
    hyp_dir = tmp_path / "hyp"
    hyp_dir.mkdir()
    original = _output(_final("H-0001", "oxygen limitation likely"))
    (hyp_dir / "hypothesis_output.json").write_text(
        original.model_dump_json()
    )

    run = Run(run_id="R1", upload_id="U1", hypothesis_dir=hyp_dir)
    run.followups.append(
        FollowupResult(
            followup_index=1,
            user_question_text="Why do you think oxygen?",
            output=_output(_final("H-0002", "DO trajectory supports it")),
        )
    )

    ctx = _build_followup_context(run)

    assert ctx is not None
    assert ctx.original_final_hypotheses[0].hyp_id == "H-0001"
    assert "oxygen limitation" in ctx.original_final_hypotheses[0].summary
    assert ctx.previous_followups[0].followup_index == 1
    assert ctx.previous_followups[0].final_hypotheses[0].hyp_id == "H-0002"


def test_build_followup_context_returns_none_when_no_prior_outputs():
    run = Run(run_id="R1", upload_id="U1")
    assert _build_followup_context(run) is None
