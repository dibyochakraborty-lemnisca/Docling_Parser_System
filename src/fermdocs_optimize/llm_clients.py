"""Gemini / Anthropic ReAct clients for the optimizer agent.

Same two-tool (tool_call / emit) shape as the diagnosis and recommendation
stages. The Gemini structured-output schema gives `args` explicit properties
(name, the run_optimization_loop knobs, payload_json) — without them Gemini
cannot emit the loop config or the final narration. Emit carries `payload_json`
(a stringified narration dict) on both providers.
"""

from __future__ import annotations

import json
import os
from typing import Any, Protocol


class OptimizeLLMClient(Protocol):
    def call(self, system: str, messages: list[dict[str, str]]) -> dict[str, Any]: ...


_GEMINI_DEFAULT_MODEL = "gemini-3-pro"
_ANTHROPIC_DEFAULT_MODEL = "claude-opus-4-7"
_DEFAULT_GEMINI_MAX_OUTPUT_TOKENS = 32768
_DEFAULT_ANTHROPIC_MAX_OUTPUT_TOKENS = 16000

_TOOL_ENUM = [
    "get_experiment",
    "get_box",
    "get_levers",
    "get_skill",
    "run_optimization_loop",
    "submit_optimization",
]


class TruncatedResponse(RuntimeError):
    pass


def _max_output_tokens(default: int) -> int:
    raw = os.environ.get("FERMDOCS_OPTIMIZE_MAX_OUTPUT_TOKENS")
    if raw:
        try:
            return int(raw)
        except ValueError:
            pass
    return default


class GeminiOptimizeClient:
    def __init__(self, model: str | None = None, api_key: str | None = None) -> None:
        self._model = (
            model
            or os.environ.get("FERMDOCS_OPTIMIZE_MODEL")
            or os.environ.get("FERMDOCS_GEMINI_MODEL", _GEMINI_DEFAULT_MODEL)
        )
        self._api_key = api_key or os.environ.get("GEMINI_API_KEY")

    def call(self, system: str, messages: list[dict[str, str]]) -> dict[str, Any]:
        from google import genai
        from google.genai import types

        client = genai.Client(api_key=self._api_key)
        contents = [
            {
                "role": "model" if m["role"] == "assistant" else "user",
                "parts": [{"text": m["content"]}],
            }
            for m in messages
        ]
        response = client.models.generate_content(
            model=self._model,
            contents=contents,
            config=types.GenerateContentConfig(
                system_instruction=system,
                response_mime_type="application/json",
                response_schema=_GEMINI_SCHEMA,
                temperature=0.0,
                max_output_tokens=_max_output_tokens(_DEFAULT_GEMINI_MAX_OUTPUT_TOKENS),
            ),
        )
        text = response.text
        finish_reason = None
        candidates = getattr(response, "candidates", None) or []
        if candidates:
            finish_reason = getattr(candidates[0], "finish_reason", None)
        if finish_reason is not None and str(finish_reason).upper().endswith("MAX_TOKENS"):
            raise TruncatedResponse("Gemini optimizer response hit the output token cap")
        if not text:
            raise ValueError("Gemini returned empty optimizer response")
        return json.loads(text)


class AnthropicOptimizeClient:
    def __init__(self, model: str | None = None) -> None:
        self._model = (
            model
            or os.environ.get("FERMDOCS_OPTIMIZE_MODEL")
            or _ANTHROPIC_DEFAULT_MODEL
        )

    def call(self, system: str, messages: list[dict[str, str]]) -> dict[str, Any]:
        from anthropic import Anthropic

        client = Anthropic()
        response = client.messages.create(
            model=self._model,
            max_tokens=_max_output_tokens(_DEFAULT_ANTHROPIC_MAX_OUTPUT_TOKENS),
            system=system,
            messages=messages,  # type: ignore[arg-type]
            tools=[
                {
                    "name": "tool_call",
                    "description": "Read the experiment/box/skill, or run the optimization loop.",
                    "input_schema": _ANTHROPIC_TOOL_CALL_SCHEMA,
                },
                {
                    "name": "emit",
                    "description": "Emit the final optimization narration (JSON string).",
                    "input_schema": _ANTHROPIC_EMIT_SCHEMA,
                },
            ],  # type: ignore[arg-type]
            tool_choice={"type": "any"},
        )
        for block in response.content:
            if getattr(block, "type", None) == "tool_use":
                payload = dict(block.input)
                payload["action"] = block.name
                return payload
        raise ValueError("anthropic optimizer response missing tool_use block")


def build_optimize_client(provider: str | None = None):
    name = (
        provider
        or os.environ.get("FERMDOCS_OPTIMIZE_PROVIDER")
        or os.environ.get("FERMDOCS_RECOMMEND_PROVIDER")
        or os.environ.get("FERMDOCS_MAPPER_PROVIDER", "gemini")
    ).lower()
    if name in ("fake", "none"):
        return None
    if name == "gemini":
        return GeminiOptimizeClient()
    if name == "anthropic":
        return AnthropicOptimizeClient()
    raise ValueError(
        f"unknown optimize provider: {name!r} (expected anthropic/gemini/fake/none)"
    )


# --- Schemas ----------------------------------------------------------------
# Gemini structured output requires explicit properties on OBJECT fields. We
# union every tool's arg keys here; unused keys are simply left null.
_GEMINI_ARGS = {
    "type": "OBJECT",
    "nullable": True,
    "properties": {
        "name": {"type": "STRING", "nullable": True},               # get_skill
        "objective_species": {"type": "STRING", "nullable": True},  # run_optimization_loop
        "model": {"type": "STRING", "nullable": True},
        "proposer": {"type": "STRING", "nullable": True},
        "max_rounds": {"type": "INTEGER", "nullable": True},
        "proposals_per_round": {"type": "INTEGER", "nullable": True},
        "delta_titer_threshold": {"type": "NUMBER", "nullable": True},
        "oracle_search": {"type": "BOOLEAN", "nullable": True},     # run_optimization_loop
        "n_lhs": {"type": "INTEGER", "nullable": True},
        "refine_iters": {"type": "INTEGER", "nullable": True},
        "payload_json": {"type": "STRING", "nullable": True},       # submit_optimization
    },
}

_GEMINI_SCHEMA: dict[str, Any] = {
    "type": "OBJECT",
    "properties": {
        "action": {"type": "STRING", "enum": ["tool_call", "emit"]},
        "tool": {"type": "STRING", "enum": _TOOL_ENUM, "nullable": True},
        "args": _GEMINI_ARGS,
        # emit branch: a stringified narration dict
        "payload_json": {"type": "STRING", "nullable": True},
    },
    "required": ["action"],
}

_ANTHROPIC_TOOL_CALL_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "tool": {"type": "string", "enum": _TOOL_ENUM},
        "args": {"type": "object"},
    },
    "required": ["tool", "args"],
}

_ANTHROPIC_EMIT_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {"payload_json": {"type": "string"}},
    "required": ["payload_json"],
}
