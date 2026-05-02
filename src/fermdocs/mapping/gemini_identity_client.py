"""Gemini-backed IdentityLLMClient.

Mirrors the GeminiHeaderMapper pattern: structured-output mode (response_schema)
guarantees the LLM emits well-formed JSON. The two-layer schema (observed +
registered) is enforced at the LLM boundary, not just at parse time.

Hallucination guard is upstream in identity_extractor.py: substring evidence
verification, registry whitelist check, fingerprint check, confidence cap.
"""

from __future__ import annotations

import json
import os
from typing import Any

_DEFAULT_MODEL = "gemini-3-flash"


class GeminiIdentityClient:
    """Implements IdentityLLMClient via Google Gemini structured output."""

    def __init__(self, model: str | None = None, api_key: str | None = None) -> None:
        self._model = (
            model
            or os.environ.get("FERMDOCS_IDENTITY_MODEL")
            or os.environ.get("FERMDOCS_GEMINI_MODEL", _DEFAULT_MODEL)
        )
        self._api_key = api_key or os.environ.get("GEMINI_API_KEY")

    def call(self, system: str, user: str) -> dict[str, Any]:
        from google import genai  # lazy import; gemini extra is optional
        from google.genai import types

        client = genai.Client(api_key=self._api_key)
        response = client.models.generate_content(
            model=self._model,
            contents=user,
            config=types.GenerateContentConfig(
                system_instruction=system,
                response_mime_type="application/json",
                response_schema=_GEMINI_IDENTITY_SCHEMA,
                temperature=0.0,
            ),
        )
        text = response.text
        if os.environ.get("FERMDOCS_DEBUG_IDENTITY"):
            import sys

            print(f"[gemini-identity] raw_response={text!r}", file=sys.stderr)
        if not text:
            raise ValueError("Gemini returned empty identity response")
        return json.loads(text)


# Closed schema: matches identity_extractor's _validate_observed / _validate_registered
# expectations. evidence is a list of {paragraph_idx, span_text} pairs that the
# extractor verifies against narrative blocks.
_GEMINI_IDENTITY_SCHEMA: dict[str, Any] = {
    "type": "OBJECT",
    "properties": {
        "observed": {
            "type": "OBJECT",
            "properties": {
                "organism": {"type": "STRING", "nullable": True},
                "product": {"type": "STRING", "nullable": True},
                "process_family_hint": {"type": "STRING", "nullable": True},
                "scale_volume_l": {"type": "NUMBER", "nullable": True},
                "vessel_type": {"type": "STRING", "nullable": True},
                "confidence": {"type": "NUMBER"},
                "evidence": {
                    "type": "ARRAY",
                    "items": {
                        "type": "OBJECT",
                        "properties": {
                            "paragraph_idx": {"type": "INTEGER"},
                            "span_text": {"type": "STRING"},
                        },
                        "required": ["paragraph_idx", "span_text"],
                    },
                },
                "rationale": {"type": "STRING", "nullable": True},
            },
            "required": ["confidence", "evidence"],
        },
        "registered": {
            "type": "OBJECT",
            "properties": {
                "process_id": {"type": "STRING", "nullable": True},
                "confidence": {"type": "NUMBER"},
                "rationale": {"type": "STRING", "nullable": True},
            },
            "required": ["confidence"],
        },
    },
    "required": ["observed", "registered"],
}
