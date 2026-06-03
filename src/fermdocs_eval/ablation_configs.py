"""Ablation configuration dicts for E3a eval suite.

Each config defines overrides applied to run_stage() for a single ablation.
The full system is the reference column (no overrides).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from fermdocs_hypothesis.schema import BudgetSnapshot


@dataclass(frozen=True)
class AblationConfig:
    name: str
    budget_overrides: dict[str, Any]
    specialist_order: tuple[str, ...] | None = None
    use_memory: bool = True


FULL = AblationConfig(name="full", budget_overrides={})

NO_CRITIC = AblationConfig(
    name="no_critic",
    budget_overrides={"skip_critic": True},
)

SINGLE_SPECIALIST = AblationConfig(
    name="single_spec",
    budget_overrides={},
    specialist_order=("kinetics",),
)

NO_MEMORY = AblationConfig(
    name="no_memory",
    budget_overrides={},
    use_memory=False,
)

BASELINE = AblationConfig(name="baseline", budget_overrides={})

ALL_CONFIGS: tuple[AblationConfig, ...] = (FULL, NO_CRITIC, SINGLE_SPECIALIST, NO_MEMORY, BASELINE)

ABLATION_QUESTIONS = ("q3", "q4", "q5", "q7", "q9")


def make_budget(config: AblationConfig) -> BudgetSnapshot:
    return BudgetSnapshot(
        max_turns=10,
        max_critic_cycles_per_topic=3,
        max_tool_calls_total=80,
        max_total_input_tokens=200_000,
        **config.budget_overrides,
    )
