"""Gemini client for characterize-stage LLM agents.

Mirrors `fermdocs_hypothesis.llm_clients.GeminiHypothesisClient` — same
structured-output interface, different env var namespace so the two
stages can be configured independently.

Provider resolution:
  FERMDOCS_CHARACTERIZE_PROVIDER > FERMDOCS_HYPOTHESIS_PROVIDER >
  FERMDOCS_DIAGNOSIS_PROVIDER > 'gemini'

Why share the namespace ladder? Operator convenience — exporting one
top-level provider env var configures the whole pipeline.
"""

from __future__ import annotations

import json
import os
from typing import Any

_GEMINI_DEFAULT_MODEL = "gemini-3-pro"


class GeminiCharacterizeClient:
    """Thin wrapper over google.genai for characterize agents.

    `call(system, user_text, response_schema)` is the one-shot interface;
    each agent constructs its own user_text from PromptParts.

    Returns (parsed_dict, input_tokens, output_tokens). Token counts use
    response.usage_metadata when available; otherwise fall back to a
    rough char-based estimate so the meter never silently records zeros.
    """

    def __init__(self, model: str | None = None, api_key: str | None = None) -> None:
        self._model = (
            model
            or os.environ.get("FERMDOCS_CHARACTERIZE_MODEL")
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
        from google import genai
        from google.genai import types

        client = genai.Client(api_key=self._api_key)
        response = client.models.generate_content(
            model=self._model,
            contents=[{"role": "user", "parts": [{"text": user_text}]}],
            config=types.GenerateContentConfig(
                system_instruction=system,
                response_mime_type="application/json",
                response_schema=response_schema,
                temperature=temperature,
            ),
        )
        text = response.text
        if os.environ.get("FERMDOCS_DEBUG_CHARACTERIZE"):
            import sys

            print(f"[gemini-characterize] raw_response={text!r}", file=sys.stderr)
        if not text:
            raise ValueError("Gemini returned empty characterize response")
        parsed = json.loads(text)
        in_tok, out_tok = _extract_usage(response, system, user_text, text)
        return parsed, in_tok, out_tok


def _extract_usage(response, system: str, user_text: str, output_text: str) -> tuple[int, int]:
    usage = getattr(response, "usage_metadata", None)
    if usage is not None:
        in_tok = (
            getattr(usage, "prompt_token_count", None)
            or getattr(usage, "input_token_count", None)
        )
        out_tok = (
            getattr(usage, "candidates_token_count", None)
            or getattr(usage, "output_token_count", None)
        )
        if in_tok is not None and out_tok is not None:
            return int(in_tok), int(out_tok)
    return _approx_tokens(system + "\n" + user_text), _approx_tokens(output_text)


def _approx_tokens(s: str) -> int:
    return max(1, (len(s) + 3) // 4)


def build_characterize_client(provider: str | None = None) -> GeminiCharacterizeClient | None:
    """Returns None for 'fake'/'none' providers — callers fall back to
    deterministic-only mode (no analyzer, just spec checks)."""
    name = (
        provider
        or os.environ.get("FERMDOCS_CHARACTERIZE_PROVIDER")
        or os.environ.get("FERMDOCS_HYPOTHESIS_PROVIDER")
        or os.environ.get("FERMDOCS_DIAGNOSIS_PROVIDER", "gemini")
    ).lower()
    if name in ("fake", "none"):
        return None
    if name == "gemini":
        return GeminiCharacterizeClient()
    raise ValueError(
        f"unknown characterize provider: {name!r} "
        "(supported: 'gemini', 'fake', 'none')"
    )
