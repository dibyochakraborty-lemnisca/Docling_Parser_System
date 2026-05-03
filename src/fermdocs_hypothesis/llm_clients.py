"""LLM client wrappers for the hypothesis stage.

Plan ref: plans/2026-05-03-hypothesis-debate-v0.md §6, §11 Stage 2.

Mirrors the diagnose-stage pattern: structured-output mode locks the wire
format so agents never parse free text. Each role has its own response
schema. Stage 2 ships the orchestrator schema; specialist + synthesizer
schemas land later in Stage 2c.

Provider resolution:
  FERMDOCS_HYPOTHESIS_PROVIDER > FERMDOCS_DIAGNOSIS_PROVIDER > 'gemini'

Why share with the diagnose env var? Operator-level convenience: if you
already exported FERMDOCS_DIAGNOSIS_PROVIDER=gemini for diagnose, the
hypothesis stage picks the same provider without a new var.
"""

from __future__ import annotations

import json
import os
from typing import Any

_GEMINI_DEFAULT_MODEL = "gemini-3-pro"
_ANTHROPIC_DEFAULT_MODEL = "claude-opus-4-7"


# -----------------------------------------------------------------------------
# Schemas
# -----------------------------------------------------------------------------


_ORCHESTRATOR_SCHEMA: dict[str, Any] = {
    "type": "OBJECT",
    "properties": {
        "action": {
            "type": "STRING",
            "enum": ["select_topic", "add_open_question", "exit_stage"],
        },
        "topic_id": {"type": "STRING", "nullable": True},
        "rationale": {"type": "STRING", "nullable": True},
        "question": {"type": "STRING", "nullable": True},
        "tags": {"type": "ARRAY", "items": {"type": "STRING"}, "nullable": True},
        "exit_reason": {"type": "STRING", "nullable": True},
    },
    "required": ["action"],
}


# -----------------------------------------------------------------------------
# Clients
# -----------------------------------------------------------------------------


class GeminiHypothesisClient:
    """Thin wrapper over google.genai.

    `call(system, user_text, response_schema)` is the one-shot interface;
    each role-call constructs its own user_text from PromptParts.

    Returns the parsed dict (Gemini structured output guarantees JSON).
    """

    def __init__(self, model: str | None = None, api_key: str | None = None) -> None:
        self._model = (
            model
            or os.environ.get("FERMDOCS_HYPOTHESIS_MODEL")
            or os.environ.get("FERMDOCS_DIAGNOSIS_MODEL")
            or os.environ.get("FERMDOCS_GEMINI_MODEL", _GEMINI_DEFAULT_MODEL)
        )
        self._api_key = api_key or os.environ.get("GEMINI_API_KEY")

    @property
    def model_name(self) -> str:
        return self._model

    def call(
        self,
        *,
        system: str,
        user_text: str,
        response_schema: dict[str, Any],
        temperature: float = 0.0,
    ) -> tuple[dict[str, Any], int, int]:
        """Returns (parsed_dict, input_tokens, output_tokens).

        Token counts come from response.usage_metadata when available;
        otherwise we fall back to a rough char-based estimate so the
        token meter never silently records zeros.
        """
        from google import genai
        from google.genai import types

        client = genai.Client(api_key=self._api_key)
        response = client.models.generate_content(
            model=self._model,
            contents=[
                {"role": "user", "parts": [{"text": user_text}]},
            ],
            config=types.GenerateContentConfig(
                system_instruction=system,
                response_mime_type="application/json",
                response_schema=response_schema,
                temperature=temperature,
            ),
        )
        text = response.text
        if os.environ.get("FERMDOCS_DEBUG_HYPOTHESIS"):
            import sys

            print(f"[gemini-hypothesis] raw_response={text!r}", file=sys.stderr)
        if not text:
            raise ValueError("Gemini returned empty hypothesis response")
        parsed = json.loads(text)
        in_tok, out_tok = _extract_usage(response, system, user_text, text)
        return parsed, in_tok, out_tok


def _extract_usage(response, system: str, user_text: str, output_text: str) -> tuple[int, int]:
    """Best-effort token counts. Gemini's usage_metadata may be absent on
    some SDK versions; fall back to ceil(chars / 4) which is the standard
    rough estimate.
    """
    usage = getattr(response, "usage_metadata", None)
    if usage is not None:
        in_tok = getattr(usage, "prompt_token_count", None) or getattr(usage, "input_token_count", None)
        out_tok = getattr(usage, "candidates_token_count", None) or getattr(usage, "output_token_count", None)
        if in_tok is not None and out_tok is not None:
            return int(in_tok), int(out_tok)
    return _approx_tokens(system + "\n" + user_text), _approx_tokens(output_text)


def _approx_tokens(s: str) -> int:
    return max(1, (len(s) + 3) // 4)


def build_hypothesis_client(provider: str | None = None):
    name = (
        provider
        or os.environ.get("FERMDOCS_HYPOTHESIS_PROVIDER")
        or os.environ.get("FERMDOCS_DIAGNOSIS_PROVIDER", "gemini")
    ).lower()
    if name in ("fake", "none"):
        return None
    if name == "gemini":
        return GeminiHypothesisClient()
    raise ValueError(
        f"unknown hypothesis provider: {name!r} "
        "(Stage 2 supports 'gemini'; 'anthropic' lands in Stage 3)"
    )


__all__ = [
    "GeminiHypothesisClient",
    "build_hypothesis_client",
    "_ORCHESTRATOR_SCHEMA",
]
