"""Critic + Judge agents — offline parser/discipline tests."""

from __future__ import annotations

from fermdocs_hypothesis.agents.critic import CriticAgent
from fermdocs_hypothesis.agents.judge import JudgeAgent, JudgeResult
from fermdocs_hypothesis.schema import CriticView, CritiqueFull, JudgeView
from tests.unit.hypothesis.fixtures import make_hypothesis


def _critic_agent_offline() -> CriticAgent:
    return CriticAgent(client=None, tools=None)  # type: ignore[arg-type]


def test_critic_red_flag_without_reasons_demoted_to_green():
    agent = _critic_agent_offline()
    view = CriticView(hypothesis=make_hypothesis())
    parsed = {"action": "file_critique", "flag": "red", "reasons": []}
    crit = agent._build_critique(parsed, view, tool_calls_used=2)
    # No reasons → demoted to green so CritiqueFull doesn't reject the
    # whole thing
    assert crit.flag == "green"


def test_critic_red_flag_with_reasons_kept():
    agent = _critic_agent_offline()
    view = CriticView(hypothesis=make_hypothesis())
    parsed = {
        "action": "file_critique",
        "flag": "red",
        "reasons": ["citation N-0001 doesn't cover BATCH-05"],
    }
    crit = agent._build_critique(parsed, view, tool_calls_used=3)
    assert crit.flag == "red"
    assert len(crit.reasons) == 1
    assert crit.tool_calls_used == 3


def test_critic_unknown_flag_falls_back_to_green():
    agent = _critic_agent_offline()
    view = CriticView(hypothesis=make_hypothesis())
    parsed = {"action": "file_critique", "flag": "yellow"}
    crit = agent._build_critique(parsed, view, tool_calls_used=0)
    assert crit.flag == "green"


def test_critic_reasons_truncated_to_300_chars():
    agent = _critic_agent_offline()
    view = CriticView(hypothesis=make_hypothesis())
    parsed = {
        "action": "file_critique",
        "flag": "red",
        "reasons": ["x" * 600],
    }
    crit = agent._build_critique(parsed, view, tool_calls_used=1)
    assert len(crit.reasons[0]) == 300


# ---------- Judge ----------


def test_judge_green_critique_short_circuits_to_invalid():
    """Judge should never spend an LLM call when critique is green."""
    agent = JudgeAgent(client=None)  # type: ignore[arg-type]
    hyp = make_hypothesis()
    crit = CritiqueFull(hyp_id=hyp.hyp_id, flag="green", reasons=[])
    view = JudgeView(hypothesis=hyp, critique=crit)
    result = agent.rule(view)
    assert result.criticism_valid is False
    assert result.input_tokens == 0  # short-circuit, no LLM call
    assert "green" in result.rationale.lower() or "no objection" in result.rationale.lower()


def test_judge_view_excludes_debate_history():
    """Schema-level: JudgeView only carries hypothesis + critique + lookups."""
    hyp = make_hypothesis()
    crit = CritiqueFull(hyp_id=hyp.hyp_id, flag="green", reasons=[])
    view = JudgeView(hypothesis=hyp, critique=crit)
    fields = set(view.model_dump().keys())
    assert fields == {"hypothesis", "critique", "citation_lookups"}
