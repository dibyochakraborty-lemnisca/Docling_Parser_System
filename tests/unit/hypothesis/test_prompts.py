"""prompts — 5-layer template builder."""

from __future__ import annotations

from fermdocs_hypothesis.prompts import ToolHint, build_prompt
from fermdocs_hypothesis.schema import BudgetSnapshot, OrchestratorView


def test_build_prompt_emits_all_layers():
    view = OrchestratorView(
        current_turn=1,
        budget_remaining=BudgetSnapshot(),
        top_topics=[],
        open_questions=[],
    )
    parts = build_prompt(
        system_identity="You are the orchestrator.",
        invariants=["Cite at least one finding.", "Stay observational."],
        task_spec="Pick a topic from top_topics.",
        view_obj=view,
        tool_hints=[ToolHint(name="select_topic", purpose="pick a topic by id")],
        recap="Output one tool call: select_topic.",
    )
    assert "INVARIANTS" in parts.system
    assert "TOOLS AVAILABLE" in parts.system
    assert "select_topic" in parts.system
    assert parts.task_spec.startswith("Pick a topic")
    assert "current_turn" in parts.view_json
    assert parts.recap.startswith("Output one tool call")


def test_as_user_message_concatenates_layers():
    view = OrchestratorView(
        current_turn=0,
        budget_remaining=BudgetSnapshot(),
        top_topics=[],
        open_questions=[],
    )
    parts = build_prompt(
        system_identity="x",
        invariants=[],
        task_spec="task here",
        view_obj=view,
        tool_hints=[],
        recap="recap here",
    )
    msg = parts.as_user_message()
    assert "[TASK]" in msg
    assert "[VIEW]" in msg
    assert "[RECAP]" in msg
    assert msg.index("[TASK]") < msg.index("[VIEW]") < msg.index("[RECAP]")
