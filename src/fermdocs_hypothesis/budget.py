"""Budget helpers — pure functions over BudgetSnapshot.

Plan ref: plans/2026-05-03-hypothesis-debate-v0.md §7.

BudgetSnapshot is immutable in spirit; mutations go through these helpers
so the runner stays the only place that updates budget state.
"""

from __future__ import annotations

from fermdocs_hypothesis.schema import BudgetSnapshot


def increment_turn(b: BudgetSnapshot) -> BudgetSnapshot:
    return b.model_copy(update={"turns_used": b.turns_used + 1})


def add_tool_calls(b: BudgetSnapshot, n: int) -> BudgetSnapshot:
    return b.model_copy(update={"tool_calls_used": b.tool_calls_used + n})


def add_tokens(b: BudgetSnapshot, *, input_tokens: int, output_tokens: int) -> BudgetSnapshot:
    return b.model_copy(
        update={
            "total_input_tokens": b.total_input_tokens + input_tokens,
            "total_output_tokens": b.total_output_tokens + output_tokens,
        }
    )
