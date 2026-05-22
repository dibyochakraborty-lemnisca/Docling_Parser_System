"""Budget helpers, NullPastInsightStore, TokenMeter."""

from __future__ import annotations

from fermdocs_hypothesis.budget import add_tokens, add_tool_calls, increment_turn
from fermdocs_hypothesis.instrumentation import TokenMeter
from fermdocs_hypothesis.memory import NullPastInsightStore, PastInsight
from fermdocs_hypothesis.schema import BudgetSnapshot


def test_increment_turn_pure():
    b = BudgetSnapshot()
    b2 = increment_turn(b)
    assert b2.turns_used == 1
    assert b.turns_used == 0  # original untouched


def test_add_tool_calls_pure():
    b = BudgetSnapshot()
    b2 = add_tool_calls(b, 5)
    assert b2.tool_calls_used == 5


def test_add_tokens_accumulates():
    b = BudgetSnapshot()
    b2 = add_tokens(b, input_tokens=1000, output_tokens=200)
    b3 = add_tokens(b2, input_tokens=500, output_tokens=100)
    assert b3.total_input_tokens == 1500
    assert b3.total_output_tokens == 300


def test_budget_exhaustion_detection():
    b = BudgetSnapshot(max_tool_calls_total=10, tool_calls_used=10)
    exhausted, reason = b.is_exhausted()
    assert exhausted
    assert reason == "max_tool_calls_total"


def test_budget_max_turns_detected():
    b = BudgetSnapshot(max_turns=3, turns_used=3)
    exhausted, reason = b.is_exhausted()
    assert exhausted
    assert reason == "max_turns"


def test_null_past_insight_store_returns_empty():
    s = NullPastInsightStore()
    assert s.query("anything") == []
    assert s.query("topic", k=10) == []


def test_token_meter_records_per_agent():
    meter = TokenMeter()
    b = BudgetSnapshot()
    b = meter.record(b, agent="kinetics", input_tokens=100, output_tokens=20)
    b = meter.record(b, agent="kinetics", input_tokens=50, output_tokens=10)
    b = meter.record(b, agent="critic", input_tokens=200, output_tokens=40)
    r = meter.report
    assert r.per_agent_input["kinetics"] == 150
    assert r.per_agent_input["critic"] == 200
    assert r.total_input == 350
    assert r.total_output == 70
    assert b.total_input_tokens == 350
