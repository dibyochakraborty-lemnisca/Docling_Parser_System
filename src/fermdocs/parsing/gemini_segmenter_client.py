"""Gemini-backed SegmenterLLMClient for DocumentSegmenter.

Plan ref: docs/design/2026-05-03-pdf-document-segmentation.md

Mirrors the GeminiIdentityClient pattern:
  - lazy `google.genai` import (gemini extra is optional)
  - structured-output mode via response_schema
  - temperature 0.0 for determinism on a structural decision
  - env var overrides for model + key

Closed schema below mirrors RunSegment + DocumentMap field shape so the LLM
emits well-formed JSON. Schema validation in DocumentMap.model_post_init
catches semantic invariants (no duplicate table_idx etc.); this schema only
enforces the structural shape.
"""

from __future__ import annotations

import json
import os
from typing import Any

# Default model per design doc decision (2026-05-03):
# gemini-3.1-pro-preview, override via FERMDOCS_SEGMENTER_MODEL.
_DEFAULT_MODEL = "gemini-3.1-pro-preview"


class GeminiSegmenterClient:
    """Implements SegmenterLLMClient via Google Gemini structured output."""

    def __init__(self, model: str | None = None, api_key: str | None = None) -> None:
        self._model = (
            model
            or os.environ.get("FERMDOCS_SEGMENTER_MODEL")
            or _DEFAULT_MODEL
        )
        self._api_key = api_key or os.environ.get("GEMINI_API_KEY")

    @property
    def model_name(self) -> str:
        return self._model

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
                response_schema=_GEMINI_SEGMENTER_SCHEMA,
                temperature=0.0,
            ),
        )
        text = response.text
        if os.environ.get("FERMDOCS_DEBUG_SEGMENTER"):
            import sys

            print(f"[gemini-segmenter] raw_response={text!r}", file=sys.stderr)
        if not text:
            raise ValueError("Gemini segmenter returned empty response")
        return json.loads(text)


# Closed schema mirrors RunSegment + DocumentMap. Field names match exactly so
# DocumentSegmenter._parse_response can ingest the dict without remapping.
# `file_id`, `llm_model`, `llm_provider`, `schema_version` are NOT in this
# schema — the segmenter sets them from its own state, not from LLM output.
_GEMINI_SEGMENTER_SCHEMA: dict[str, Any] = {
    "type": "OBJECT",
    "properties": {
        "runs": {
            "type": "ARRAY",
            "items": {
                "type": "OBJECT",
                "properties": {
                    "run_id": {"type": "STRING"},
                    "display_name": {"type": "STRING"},
                    "table_indices": {
                        "type": "ARRAY",
                        "items": {"type": "INTEGER"},
                    },
                    "source_signal": {
                        "type": "STRING",
                        "enum": ["section_header", "text_pattern", "inferred"],
                    },
                    "confidence": {"type": "NUMBER"},
                    "rationale": {"type": "STRING"},
                },
                "required": [
                    "run_id",
                    "display_name",
                    "table_indices",
                    "source_signal",
                    "confidence",
                ],
            },
        },
        "unassigned_table_indices": {
            "type": "ARRAY",
            "items": {"type": "INTEGER"},
        },
        "overall_confidence": {"type": "NUMBER"},
    },
    "required": ["runs", "unassigned_table_indices", "overall_confidence"],
}
