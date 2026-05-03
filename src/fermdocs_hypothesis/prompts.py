"""Prompt template — 5-layer uniform structure across all roles.

Plan ref: plans/2026-05-03-hypothesis-debate-v0.md §6.

Layer order (all roles use the same shape so prompt-prefix caching works):

  [1] SYSTEM      identity, invariants, output schema reference   (stable)
  [2] TASK SPEC   what you do this turn                           (stable per role)
  [3] VIEW        role-shaped view object as JSON                  (volatile per turn)
  [4] TOOL HINT   one-line purpose for each available tool         (stable per role)
  [5] RECAP       hard rules + how to respond                      (stable; recency anchor)

Cache strategy: layers 1+2+4+5 are stable across turns of one role and form
the cacheable prefix; layer 3 is the only per-turn delta. Anthropic prompt-
caching kicks in on layers 1-2 minimum.

The functions here are pure string builders. LLM client wiring (which model,
tool format) lives in llm_clients.py.
"""

from __future__ import annotations

import json
from collections.abc import Sequence
from dataclasses import dataclass

from pydantic import BaseModel


@dataclass(frozen=True)
class ToolHint:
    name: str
    purpose: str


@dataclass(frozen=True)
class PromptParts:
    """All five layers as strings. Wired into provider-specific shapes
    by llm_clients.py — Gemini takes (system_instruction, contents),
    Anthropic takes (system, messages).
    """

    system: str        # layers 1 + 4 (identity + tool hints — both stable)
    task_spec: str     # layer 2
    view_json: str     # layer 3
    recap: str         # layer 5

    def as_user_message(self) -> str:
        """Single-string render for providers without separate system slots.

        Layout:
          [TASK]
          ...
          [VIEW]
          ...
          [RECAP]
          ...
        """
        return (
            "[TASK]\n"
            + self.task_spec.strip()
            + "\n\n[VIEW]\n"
            + self.view_json.strip()
            + "\n\n[RECAP]\n"
            + self.recap.strip()
        )


def build_prompt(
    *,
    system_identity: str,
    invariants: Sequence[str],
    task_spec: str,
    view_obj: BaseModel,
    tool_hints: Sequence[ToolHint],
    recap: str,
) -> PromptParts:
    """Assemble a PromptParts from typed pieces.

    Notes:
      - view_obj is dumped with Pydantic's mode='json' so timestamps, enums,
        UUIDs serialize cleanly.
      - Tool hints render as a fenced block at the bottom of the system
        message — keeps stable prefix together.
    """
    system = system_identity.strip() + "\n\n"
    if invariants:
        system += "INVARIANTS:\n"
        for inv in invariants:
            system += f"  - {inv}\n"
        system += "\n"
    if tool_hints:
        system += "TOOLS AVAILABLE:\n"
        for t in tool_hints:
            system += f"  - {t.name}: {t.purpose}\n"

    view_json = json.dumps(
        view_obj.model_dump(mode="json"),
        indent=2,
        sort_keys=False,
        default=str,
    )

    return PromptParts(
        system=system.strip(),
        task_spec=task_spec.strip(),
        view_json=view_json,
        recap=recap.strip(),
    )
