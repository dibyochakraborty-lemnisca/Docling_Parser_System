"""Critic agent — adversarially reviews the synthesized hypothesis.

Plan ref: plans/2026-05-03-hypothesis-debate-v0.md §2 (critic role), §5
(critic tools — query_bundle, get_priors, get_narrative_observations,
execute_python, file_critique).

Loop shape (ReAct, max 6 tool calls then forced file_critique):
  - Each turn the agent emits {action: tool_call|file_critique}
  - tool_call routes to HypothesisToolBundle (read tools) OR to
    execute_python (sandboxed pandas/numpy/scipy)
  - file_critique is terminal — emits {flag: red|green, reasons: [...]}
  - Red-flag REQUIRES ≥1 reason (enforced at parse time)

execute_python reuses diagnose's harness directly — same sandbox, same
50KB output cap. Per plan §15 / Stage 3 scope: shared sandbox is fine
for v0; wrap with different limits later if v1 critic needs it.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any

from fermdocs_diagnose.tools_bundle.execute_python import execute_python
from fermdocs_hypothesis.llm_clients import GeminiHypothesisClient
from fermdocs_hypothesis.prompts import ToolHint, build_prompt
from fermdocs_hypothesis.schema import CriticView, CritiqueFull
from fermdocs_hypothesis.tools_bundle.factory import (
    GET_NARRATIVE_OBSERVATIONS,
    GET_PRIORS,
    HypothesisToolBundle,
    QUERY_BUNDLE,
    truncate_result,
)

EXECUTE_PYTHON = "execute_python"
FILE_CRITIQUE = "file_critique"
MAX_CRITIC_TOOL_CALLS = 6


CRITIC_SYSTEM = """\
You are the Critic in a fermentation-hypothesis debate.

Your job: attack the synthesized hypothesis for the current turn. Look
for weak citations, unsupported claims, missed alternative explanations,
arithmetic errors, and overreach beyond the cited evidence. You can
verify numerical claims with execute_python (sandboxed pandas/numpy
against the bundle's observations.csv when available).

Your job is NOT to rewrite the hypothesis. It is to flag concrete
problems if they exist. If the hypothesis is sound, file a green flag
with no reasons. If it is not sound, file a red flag with concrete,
specific reasons.

Be precise. "Citation is weak" is useless. "Cites narrative N-0001 about
BATCH-01 but extends the claim to BATCH-05 which N-0001 does not cover"
is the kind of reason a judge can act on.\
"""

CRITIC_INVARIANTS = (
    "Use tools before judging. A green flag without any tool call is suspicious.",
    "Red flag requires ≥1 concrete, evidence-based reason.",
    "Reasons must be specific (cite IDs, name variables, point at numbers).",
    "Do not rewrite the hypothesis. Do not propose a fix. Just flag problems.",
)

CRITIC_TASK = """\
Read the hypothesis in the view. Optionally call tools to verify
citations / numerics / alternative explanations. Then file a critique.

Tool budget: up to 6 tool calls before you MUST file_critique.

Critique policy:
  - Green flag: hypothesis is well-supported by cited evidence and you
    found no concrete problem.
  - Red flag: at least one concrete problem (citation gap, unsupported
    extrapolation, missed alternative, numerical error, scope drift).
"""

CRITIC_RECAP = """\
Output one JSON action.

When tool_call: {"action":"tool_call","tool":"<name>","args":{...}}
When done: {"action":"file_critique","flag":"red"|"green","reasons":[...]}

Hard rules:
  - flag='red' MUST include ≥1 reason.
  - flag='green' SHOULD include 0 reasons (any reasons must be sub-critical observations).
  - Each reason ≤300 chars.\
"""

CRITIC_TOOL_HINTS = (
    ToolHint(
        name=QUERY_BUNDLE,
        purpose="search findings/narratives/trajectories by id or substring",
    ),
    ToolHint(
        name=GET_PRIORS,
        purpose="organism-aware variable bounds (range, typical, source)",
    ),
    ToolHint(
        name=GET_NARRATIVE_OBSERVATIONS,
        purpose="filter narrative_observations by run_id/tag/variable",
    ),
    ToolHint(
        name=EXECUTE_PYTHON,
        purpose="sandboxed pandas/numpy/scipy; observations.csv is at <bundle>/characterization/observations.csv if present",
    ),
    ToolHint(
        name=FILE_CRITIQUE,
        purpose="TERMINAL: emit critique with flag (red/green) + reasons",
    ),
)


_CRITIC_SCHEMA: dict[str, Any] = {
    "type": "OBJECT",
    "properties": {
        "action": {
            "type": "STRING",
            "enum": ["tool_call", "file_critique"],
        },
        # tool_call branch
        "tool": {
            "type": "STRING",
            "enum": [
                QUERY_BUNDLE,
                GET_PRIORS,
                GET_NARRATIVE_OBSERVATIONS,
                EXECUTE_PYTHON,
            ],
            "nullable": True,
        },
        "args": {
            "type": "OBJECT",
            "nullable": True,
            "properties": {
                "scope": {"type": "STRING", "nullable": True},
                "id_or_query": {"type": "STRING", "nullable": True},
                "organism": {"type": "STRING", "nullable": True},
                "process_family": {"type": "STRING", "nullable": True},
                "variable": {"type": "STRING", "nullable": True},
                "run_id": {"type": "STRING", "nullable": True},
                "tag": {"type": "STRING", "nullable": True},
                "limit": {"type": "INTEGER", "nullable": True},
                "code": {"type": "STRING", "nullable": True},
                "timeout": {"type": "INTEGER", "nullable": True},
            },
        },
        # file_critique branch
        "flag": {"type": "STRING", "enum": ["red", "green"], "nullable": True},
        "reasons": {
            "type": "ARRAY", "items": {"type": "STRING"}, "nullable": True,
        },
    },
    "required": ["action"],
}


@dataclass
class CriticResult:
    critique: CritiqueFull
    input_tokens: int
    output_tokens: int
    tool_calls: int


class CriticAgent:
    def __init__(
        self,
        client: GeminiHypothesisClient,
        tools: HypothesisToolBundle,
    ):
        self._client = client
        self._tools = tools

    def critique(self, view: CriticView) -> CriticResult:
        tool_history: list[dict[str, Any]] = []
        total_in = 0
        total_out = 0

        for call_idx in range(MAX_CRITIC_TOOL_CALLS + 1):
            must_critique = call_idx >= MAX_CRITIC_TOOL_CALLS
            user_text = self._build_user_text(view, tool_history, must_critique)
            parts = build_prompt(
                system_identity=CRITIC_SYSTEM,
                invariants=CRITIC_INVARIANTS,
                task_spec=CRITIC_TASK,
                view_obj=view,
                tool_hints=CRITIC_TOOL_HINTS,
                recap=CRITIC_RECAP,
            )
            parsed, in_tok, out_tok = self._client.call(
                system=parts.system,
                user_text=user_text,
                response_schema=_CRITIC_SCHEMA,
            )
            total_in += in_tok
            total_out += out_tok

            action = parsed.get("action", "")

            if action == "tool_call" and not must_critique:
                tool_name = parsed.get("tool") or ""
                args = dict(parsed.get("args") or {})
                if tool_name == EXECUTE_PYTHON:
                    code = (args.get("code") or "").strip()
                    timeout = int(args.get("timeout") or 60)
                    if not code:
                        result = {"error": "execute_python requires non-empty code"}
                    else:
                        ep_result = execute_python(code, timeout=timeout)
                        result = {
                            "stdout": ep_result.stdout,
                            "stderr": ep_result.stderr,
                            "returncode": ep_result.returncode,
                            "timed_out": ep_result.timed_out,
                            "duration_ms": ep_result.duration_ms,
                        }
                else:
                    result = self._tools.dispatch(tool_name, args)
                tool_history.append(
                    {"call": tool_name, "args": args, "result": result}
                )
                continue

            critique = self._build_critique(parsed, view, len(tool_history))
            return CriticResult(
                critique=critique,
                input_tokens=total_in,
                output_tokens=total_out,
                tool_calls=len(tool_history),
            )

        raise RuntimeError("critic loop exited without filing critique")

    def _build_user_text(
        self,
        view: CriticView,
        tool_history: list[dict[str, Any]],
        must_critique: bool,
    ) -> str:
        parts = build_prompt(
            system_identity=CRITIC_SYSTEM,
            invariants=CRITIC_INVARIANTS,
            task_spec=CRITIC_TASK,
            view_obj=view,
            tool_hints=CRITIC_TOOL_HINTS,
            recap=CRITIC_RECAP,
        )
        body = parts.as_user_message()
        if tool_history:
            body += "\n\n[TOOL HISTORY]\n"
            for entry in tool_history:
                body += f"- {entry['call']}({json.dumps(entry['args'], default=str)}) → "
                body += truncate_result(json.dumps(entry["result"], default=str), cap=2500) + "\n"
        if must_critique:
            body += (
                "\n\n[FORCED]\nTool budget exhausted. You MUST file_critique now."
                " If you found no concrete problems, file flag='green' with empty"
                " reasons."
            )
        return body

    def _build_critique(
        self,
        parsed: dict[str, Any],
        view: CriticView,
        tool_calls_used: int,
    ) -> CritiqueFull:
        flag = parsed.get("flag") or "green"
        if flag not in ("red", "green"):
            flag = "green"
        reasons_raw = parsed.get("reasons") or []
        reasons = [str(r)[:300] for r in reasons_raw if r]
        # Hard guard: red flag without reasons → demote to green (validator
        # would otherwise reject the CritiqueFull entirely).
        if flag == "red" and not reasons:
            flag = "green"
        return CritiqueFull(
            hyp_id=view.hypothesis.hyp_id,
            flag=flag,
            reasons=reasons,
            tool_calls_used=tool_calls_used,
        )


def build_critic(
    client: GeminiHypothesisClient,
    tools: HypothesisToolBundle,
) -> CriticAgent:
    return CriticAgent(client=client, tools=tools)
