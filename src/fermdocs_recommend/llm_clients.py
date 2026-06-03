"""Gemini / Anthropic ReAct clients for the recommendation agent.

Same two-tool (tool_call / emit) shape as the diagnosis stage. The Gemini
structured-output schema gives `args` explicit properties (code, timeout, name,
payload) — without them Gemini cannot emit the fit code or the final payload,
which was the bug in the first cut. Emit carries `payload_json` (a stringified
RecommendationOutput-shaped dict) on both providers.
"""

from __future__ import annotations

import json
import os
from typing import Any, Protocol


class RecommendLLMClient(Protocol):
    def call(self, system: str, messages: list[dict[str, str]]) -> dict[str, Any]: ...


_GEMINI_DEFAULT_MODEL = "gemini-3-pro"
_ANTHROPIC_DEFAULT_MODEL = "claude-opus-4-7"
_DEFAULT_GEMINI_MAX_OUTPUT_TOKENS = 65536
_DEFAULT_ANTHROPIC_MAX_OUTPUT_TOKENS = 32000

_TOOL_ENUM = [
    "get_hypotheses",
    "get_data_feed",
    "get_skill",
    "execute_python",
    "submit_recommendation",
]


class TruncatedResponse(RuntimeError):
    pass


def _max_output_tokens(default: int) -> int:
    raw = os.environ.get("FERMDOCS_RECOMMEND_MAX_OUTPUT_TOKENS")
    if raw:
        try:
            return int(raw)
        except ValueError:
            pass
    return default


class GeminiRecommendClient:
    def __init__(self, model: str | None = None, api_key: str | None = None) -> None:
        self._model = (
            model
            or os.environ.get("FERMDOCS_RECOMMEND_MODEL")
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
            raise TruncatedResponse("Gemini recommendation response hit the output token cap")
        if not text:
            raise ValueError("Gemini returned empty recommendation response")
        return json.loads(text)


class AnthropicRecommendClient:
    def __init__(self, model: str | None = None) -> None:
        self._model = (
            model
            or os.environ.get("FERMDOCS_RECOMMEND_MODEL")
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
                    "description": "Request data, read a skill, or run brewtwin code in the sandbox.",
                    "input_schema": _ANTHROPIC_TOOL_CALL_SCHEMA,
                },
                {
                    "name": "emit",
                    "description": "Emit the final recommendation payload (JSON string).",
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
        raise ValueError("anthropic recommendation response missing tool_use block")


def build_recommend_client(provider: str | None = None):
    name = (
        provider
        or os.environ.get("FERMDOCS_RECOMMEND_PROVIDER")
        or os.environ.get("FERMDOCS_HYPOTHESIS_PROVIDER")
        or os.environ.get("FERMDOCS_MAPPER_PROVIDER", "gemini")
    ).lower()
    if name in ("fake", "none"):
        return None
    if name == "gemini":
        return GeminiRecommendClient()
    if name == "anthropic":
        return AnthropicRecommendClient()
    raise ValueError(
        f"unknown recommend provider: {name!r} (expected anthropic/gemini/fake/none)"
    )


# --- Schemas ----------------------------------------------------------------
# Gemini structured output requires explicit properties on OBJECT fields. We
# union every tool's arg keys here; unused keys are simply left null.
_GEMINI_ARGS = {
    "type": "OBJECT",
    "nullable": True,
    "properties": {
        "code": {"type": "STRING", "nullable": True},       # execute_python
        "timeout": {"type": "INTEGER", "nullable": True},   # execute_python
        "name": {"type": "STRING", "nullable": True},       # get_skill
        "payload_json": {"type": "STRING", "nullable": True},  # submit_recommendation
    },
}

_GEMINI_SCHEMA: dict[str, Any] = {
    "type": "OBJECT",
    "properties": {
        "action": {"type": "STRING", "enum": ["tool_call", "emit"]},
        "tool": {"type": "STRING", "enum": _TOOL_ENUM, "nullable": True},
        "args": _GEMINI_ARGS,
        # emit branch: a stringified RecommendationOutput-shaped dict
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
