from __future__ import annotations

import json
import os
from typing import Any

from fermdocs.domain.models import (
    GoldenSchema,
    MappingEntry,
    MappingResult,
    ParsedTable,
    TableMapping,
)
from fermdocs.mapping.prompt import render_user_prompt, system_prompt

_DEFAULT_MODEL = "gemini-3-flash"


class GeminiHeaderMapper:
    """Google Gemini-backed mapper. Implements HeaderMapper.

    Uses Gemini's structured-output mode (response_schema). Hallucination guard
    is the same as the Anthropic path: the LLM emits only header-to-column
    mappings, never values.
    """

    def __init__(self, model: str | None = None, api_key: str | None = None) -> None:
        self._model = model or os.environ.get("FERMDOCS_GEMINI_MODEL", _DEFAULT_MODEL)
        self._api_key = api_key or os.environ.get("GEMINI_API_KEY")

    def map(self, tables: list[ParsedTable], schema: GoldenSchema) -> MappingResult:
        from google import genai  # imported lazily so the gemini extra is optional
        from google.genai import types

        client = genai.Client(api_key=self._api_key)
        response = client.models.generate_content(
            model=self._model,
            contents=render_user_prompt(tables, schema),
            config=types.GenerateContentConfig(
                system_instruction=system_prompt(),
                response_mime_type="application/json",
                response_schema=_GEMINI_RESPONSE_SCHEMA,
                temperature=0.0,
            ),
        )
        text = response.text
        if os.environ.get("FERMDOCS_DEBUG_MAPPER"):
            import sys

            print(f"[gemini] tables_in={len(tables)}", file=sys.stderr)
            print(f"[gemini] raw_response={text!r}", file=sys.stderr)
        if not text:
            raise ValueError("Gemini returned empty response")
        payload = json.loads(text)
        return _parse_response(payload)


_GEMINI_RESPONSE_SCHEMA: dict[str, Any] = {
    "type": "OBJECT",
    "properties": {
        "tables": {
            "type": "ARRAY",
            "items": {
                "type": "OBJECT",
                "properties": {
                    "table_id": {"type": "STRING"},
                    "entries": {
                        "type": "ARRAY",
                        "items": {
                            "type": "OBJECT",
                            "properties": {
                                "raw_header": {"type": "STRING"},
                                "mapped_to": {"type": "STRING", "nullable": True},
                                "raw_unit": {"type": "STRING", "nullable": True},
                                "confidence": {"type": "NUMBER"},
                                "rationale": {"type": "STRING", "nullable": True},
                            },
                            "required": ["raw_header", "mapped_to", "confidence"],
                        },
                    },
                },
                "required": ["table_id", "entries"],
            },
        }
    },
    "required": ["tables"],
}


def _parse_response(payload: dict[str, Any]) -> MappingResult:
    tables: list[TableMapping] = []
    for t in payload["tables"]:
        entries = [MappingEntry.model_validate(e) for e in t["entries"]]
        tables.append(TableMapping(table_id=t["table_id"], entries=entries))
    return MappingResult(tables=tables)
