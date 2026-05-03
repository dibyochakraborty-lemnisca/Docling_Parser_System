"""Token meter — per-agent input/output ledger.

Plan ref: plans/2026-05-03-hypothesis-debate-v0.md §11 Stage 1 deliverable.

Stage 1 stubs feed hardcoded numbers; Stage 2 real agents feed actual usage
from LLM client responses. Same interface either way so stub-built tests
exercise the meter end-to-end.

The meter both updates the BudgetSnapshot (so runner can detect exhaustion)
and accumulates a per-agent TokenReport (surfaced in HypothesisOutput).
"""

from __future__ import annotations

from fermdocs_hypothesis.budget import add_tokens
from fermdocs_hypothesis.schema import BudgetSnapshot, TokenReport


class TokenMeter:
    def __init__(self) -> None:
        self._report = TokenReport()

    @property
    def report(self) -> TokenReport:
        return self._report

    def record(
        self,
        budget: BudgetSnapshot,
        *,
        agent: str,
        input_tokens: int,
        output_tokens: int,
    ) -> BudgetSnapshot:
        """Record an agent call; returns updated BudgetSnapshot.

        Idempotent in interface but stateful: calling twice with the same
        agent name accumulates.
        """
        self._report.per_agent_input[agent] = (
            self._report.per_agent_input.get(agent, 0) + input_tokens
        )
        self._report.per_agent_output[agent] = (
            self._report.per_agent_output.get(agent, 0) + output_tokens
        )
        self._report.total_input += input_tokens
        self._report.total_output += output_tokens
        return add_tokens(
            budget,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
        )
