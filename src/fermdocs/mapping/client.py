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

_DEFAULT_MODEL = "claude-haiku-4-5-20251001"


class LLMHeaderMapper:
    """Anthropic-backed mapper. Isolates the SDK so a Rust port swaps one file."""

    def __init__(self, model: str | None = None, api_key: str | None = None) -> None:
        self._model = model or os.environ.get("FERMDOCS_MAPPER_MODEL", _DEFAULT_MODEL)
        self._api_key = api_key or os.environ.get("ANTHROPIC_API_KEY")

    def map(self, tables: list[ParsedTable], schema: GoldenSchema) -> MappingResult:
        from anthropic import Anthropic  # imported lazily so tests don't need it

        client = Anthropic(api_key=self._api_key)
        response = client.messages.create(
            model=self._model,
            max_tokens=4096,
            system=system_prompt(),
            messages=[{"role": "user", "content": render_user_prompt(tables, schema)}],
            tools=[
                {
                    "name": "emit_mapping",
                    "description": "Emit the header-to-column mapping for all tables.",
                    "input_schema": _RESPONSE_SCHEMA,
                }
            ],
            tool_choice={"type": "tool", "name": "emit_mapping"},
        )
        payload = _extract_tool_input(response)
        return _parse_response(payload)


_RESPONSE_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "tables": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "table_id": {"type": "string"},
                    "entries": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {
                                "raw_header": {"type": "string"},
                                "mapped_to": {"type": ["string", "null"]},
                                "raw_unit": {"type": ["string", "null"]},
                                "confidence": {"type": "number"},
                                "rationale": {"type": ["string", "null"]},
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


def _extract_tool_input(response: Any) -> dict[str, Any]:
    for block in response.content:
        if getattr(block, "type", None) == "tool_use":
            return block.input  # type: ignore[no-any-return]
    raise ValueError("Mapper response had no tool_use block")


def _parse_response(payload: dict[str, Any]) -> MappingResult:
    tables: list[TableMapping] = []
    for t in payload["tables"]:
        entries = [MappingEntry.model_validate(e) for e in t["entries"]]
        tables.append(TableMapping(table_id=t["table_id"], entries=entries))
    return MappingResult(tables=tables)


def dump_response_schema() -> str:
    return json.dumps(_RESPONSE_SCHEMA, indent=2)
