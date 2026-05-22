"""Anthropic-backed IdentityLLMClient.

Tool-use mode locks the schema at the API boundary. Same hallucination guard
as the Gemini path: validation lives upstream in identity_extractor.py.
"""

from __future__ import annotations

import os
from typing import Any

_DEFAULT_MODEL = "claude-sonnet-4-6"


class AnthropicIdentityClient:
    """Implements IdentityLLMClient via Anthropic tool-use."""

    def __init__(self, model: str | None = None) -> None:
        self._model = (
            model
            or os.environ.get("FERMDOCS_IDENTITY_MODEL")
            or _DEFAULT_MODEL
        )

    def call(self, system: str, user: str) -> dict[str, Any]:
        from anthropic import Anthropic

        client = Anthropic()
        response = client.messages.create(
            model=self._model,
            max_tokens=2048,
            system=system,
            messages=[{"role": "user", "content": user}],
            tools=[
                {
                    "name": "emit_identity",
                    "description": "Emit observed and registered identity layers.",
                    "input_schema": _ANTHROPIC_IDENTITY_SCHEMA,
                }
            ],
            tool_choice={"type": "tool", "name": "emit_identity"},
        )
        for block in response.content:
            if getattr(block, "type", None) == "tool_use":
                return dict(block.input)
        raise ValueError("anthropic identity response missing tool_use block")


def _process_family_enum_anthropic() -> list[str]:
    """Closed-vocab process family names from process_families.yaml.

    Built lazily so the schema stays in sync with the YAML. `unknown` is
    last as the explicit escape hatch. Hardcoded fallback in case the
    YAML is missing or corrupt — prevents the LLM contract from breaking
    on a config issue.
    """
    try:
        from fermdocs.domain.process_families import (
            UNKNOWN_FAMILY_NAME,
            load_process_families,
        )
        names = sorted(load_process_families().keys())
        names = [n for n in names if n != UNKNOWN_FAMILY_NAME]
        names.append(UNKNOWN_FAMILY_NAME)
        return names
    except Exception:
        return [
            "penicillin_fedbatch",
            "yeast_intracellular_product_fedbatch",
            "yeast_aerobic_fedbatch",
            "ecoli_recombinant_protein",
            "unknown",
        ]


_ANTHROPIC_IDENTITY_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "observed": {
            "type": "object",
            "properties": {
                "organism": {"type": ["string", "null"]},
                "product": {"type": ["string", "null"]},
                "process_family_hint": {"type": ["string", "null"]},
                "scale_volume_l": {"type": ["number", "null"]},
                "vessel_type": {"type": ["string", "null"]},
                "confidence": {"type": "number"},
                "evidence": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "paragraph_idx": {"type": "integer"},
                            "span_text": {"type": "string"},
                        },
                        "required": ["paragraph_idx", "span_text"],
                    },
                },
                "rationale": {"type": ["string", "null"]},
            },
            "required": ["confidence", "evidence"],
        },
        "registered": {
            "type": "object",
            "properties": {
                "process_id": {"type": ["string", "null"]},
                # Closed-vocab process family. The LLM picks from the enum
                # (sourced from process_families.yaml) or "unknown". This is
                # the canonical key downstream agents use for routing.
                "process_family": {
                    "type": ["string", "null"],
                    "enum": _process_family_enum_anthropic(),
                },
                "confidence": {"type": "number"},
                "rationale": {"type": ["string", "null"]},
            },
            "required": ["confidence"],
        },
    },
    "required": ["observed", "registered"],
}
