"""Narrative entity extraction (Tier 2).

Deliberately the highest-risk path in the system: this is the ONLY LLM that
emits numeric values. The compensating safety mechanisms are:

  1. Mandatory evidence verification (substring + value-form check + sentence bound).
  2. Schema validation (LLM cannot invent column names).
  3. Dedup against table observations (no phantom corroboration).
  4. Confidence cap of 0.85 (narrative observations never auto-accept).
  5. observation_type fixed to 'reported' (no LLM classification of intent).
  6. Per-call paragraph cap (prevents cross-paragraph entity bleed).
  7. Graceful degradation on any LLM error -> empty list, no crash.

See docs/architecture.md Section 7 for the full design rationale.
"""
from __future__ import annotations

import json
import os
from typing import Any, Iterable, Iterator, Protocol

from fermdocs.domain.models import (
    GoldenSchema,
    NarrativeBlock,
    NarrativeExtraction,
    Observation,
)
from fermdocs.mapping.evidence_gated_llm import (
    LLM_CONFIDENCE_CAP,
    MAX_EVIDENCE_LEN,
    MAX_SENTENCE_BREAKS,
    value_string_forms,
    verify_substring_evidence,
)

# Re-export under the name the rest of the codebase imports.
NARRATIVE_CONFIDENCE_CAP = LLM_CONFIDENCE_CAP
verify_evidence = verify_substring_evidence
_value_string_forms = value_string_forms

MAX_PARAGRAPHS_PER_CALL = 20


class NarrativeExtractor(Protocol):
    def extract(
        self, blocks: list[NarrativeBlock], schema: GoldenSchema
    ) -> list[NarrativeExtraction]: ...


def chunk_blocks(
    blocks: list[NarrativeBlock], size: int = MAX_PARAGRAPHS_PER_CALL
) -> Iterator[list[NarrativeBlock]]:
    """Yield blocks in chunks of up to `size`. Cross-chunk associations are intentionally lost."""
    for i in range(0, len(blocks), size):
        yield blocks[i : i + size]


def is_dup_of_table_observations(
    extraction: NarrativeExtraction,
    table_observations: Iterable[Observation],
    *,
    rel_tolerance: float = 1e-3,
) -> bool:
    """True if (column, value) already exists in observations from the table source.

    Prevents phantom corroboration: prose that summarizes a table value should
    not produce a second observation that the next agent reads as independent.
    """
    for obs in table_observations:
        if obs.column_name != extraction.column:
            continue
        if obs.source_locator.get("section") != "table":
            continue
        canonical = obs.value_canonical or obs.value_raw or {}
        existing_val = canonical.get("value")
        if _values_match(extraction.value, existing_val, rel_tolerance):
            return True
    return False


def _values_match(a: Any, b: Any, rel_tolerance: float) -> bool:
    if a is None or b is None:
        return False
    try:
        fa, fb = float(a), float(b)
        if fa == 0 and fb == 0:
            return True
        denom = max(abs(fa), abs(fb), 1.0)
        return abs(fa - fb) / denom <= rel_tolerance
    except (TypeError, ValueError):
        return str(a).strip().lower() == str(b).strip().lower()


class LLMNarrativeExtractor:
    """Provider-aware (Anthropic / Gemini). Errors degrade to empty list.

    Caches per-block? No: paragraph text rarely repeats. We DO chunk to
    MAX_PARAGRAPHS_PER_CALL to prevent cross-paragraph entity bleed.
    """

    _SYSTEM_PROMPT = (
        "You extract values for canonical golden columns from short fermentation "
        "experiment paragraphs.\n\n"
        "For each value you extract you MUST emit:\n"
        "  - column: a name from the provided golden schema (no other names allowed).\n"
        "  - value: the numeric or string value (NEVER convert units; emit as written).\n"
        "  - unit: the unit string from the prose, or null.\n"
        "  - evidence: a VERBATIM substring of the source paragraph, <= 200 chars, "
        "containing the value's string form.\n"
        "  - source_paragraph_idx: the paragraph_idx of the source block.\n"
        "  - confidence: 0..1, your confidence in the extraction.\n"
        "  - rationale: short reason for the extraction.\n\n"
        "Rules:\n"
        "  - Do NOT emit observation_type; it is fixed to 'reported' downstream.\n"
        "  - Do NOT compute or guess values not present in the prose.\n"
        "  - If a paragraph contains no extractable golden values, emit nothing.\n"
        "  - If the same value is described comparatively (\"ranged from X to Y\"), "
        "do NOT pick one as canonical -- skip the extraction.\n"
        "  - Return an empty list for paragraphs that only describe methodology with no "
        "specific values."
    )

    def __init__(self, provider: str | None = None, model: str | None = None) -> None:
        self._provider = (
            provider
            or os.environ.get("FERMDOCS_NARRATIVE_PROVIDER")
            or os.environ.get("FERMDOCS_MAPPER_PROVIDER")
            or "gemini"
        ).lower()
        self._model = model or os.environ.get("FERMDOCS_NARRATIVE_MODEL")

    def extract(
        self, blocks: list[NarrativeBlock], schema: GoldenSchema
    ) -> list[NarrativeExtraction]:
        if not blocks:
            return []
        try:
            return self._call_llm(blocks, schema)
        except Exception:
            # Best-effort: failure means no narrative observations from this batch.
            # Tier 1 residual capture is unaffected.
            return []

    def _call_llm(
        self, blocks: list[NarrativeBlock], schema: GoldenSchema
    ) -> list[NarrativeExtraction]:
        user_prompt = _render_user_prompt(blocks, schema)
        if self._provider == "anthropic":
            payload = self._call_anthropic(user_prompt)
        elif self._provider == "gemini":
            payload = self._call_gemini(user_prompt)
        else:
            raise ValueError(f"unknown narrative provider: {self._provider}")
        return _parse_extractions(payload)

    def _call_anthropic(self, user_prompt: str) -> dict[str, Any]:
        from anthropic import Anthropic

        client = Anthropic()
        model = self._model or os.environ.get(
            "FERMDOCS_NARRATIVE_MODEL", "claude-sonnet-4-6"
        )
        response = client.messages.create(
            model=model,
            max_tokens=4096,
            system=self._SYSTEM_PROMPT,
            messages=[{"role": "user", "content": user_prompt}],
            tools=[
                {
                    "name": "emit_extractions",
                    "description": "Emit narrative entity extractions for golden columns.",
                    "input_schema": _ANTHROPIC_RESPONSE_SCHEMA,
                }
            ],
            tool_choice={"type": "tool", "name": "emit_extractions"},
        )
        for block in response.content:
            if getattr(block, "type", None) == "tool_use":
                return dict(block.input)
        raise ValueError("narrative extractor response missing tool_use block")

    def _call_gemini(self, user_prompt: str) -> dict[str, Any]:
        from google import genai
        from google.genai import types

        model = self._model or os.environ.get(
            "FERMDOCS_NARRATIVE_MODEL", "gemini-3-pro"
        )
        client = genai.Client(api_key=os.environ.get("GEMINI_API_KEY"))
        response = client.models.generate_content(
            model=model,
            contents=user_prompt,
            config=types.GenerateContentConfig(
                system_instruction=self._SYSTEM_PROMPT,
                response_mime_type="application/json",
                response_schema=_GEMINI_RESPONSE_SCHEMA,
                temperature=0.0,
            ),
        )
        text = response.text
        if not text:
            raise ValueError("empty gemini narrative response")
        return json.loads(text)


def _render_user_prompt(blocks: list[NarrativeBlock], schema: GoldenSchema) -> str:
    schema_lines = []
    for col in schema.columns:
        examples = ", ".join(f"'{e.raw_header}'" for e in col.examples[:3])
        schema_lines.append(
            f"- {col.name} ({col.data_type.value}"
            + (f", {col.canonical_unit}" if col.canonical_unit else "")
            + f"): {col.description}"
            + (f" Synonyms: {', '.join(col.synonyms)}." if col.synonyms else "")
        )
    schema_block = "\n".join(schema_lines)
    paragraphs = [
        {
            "paragraph_idx": b.locator.get("paragraph_idx", -1),
            "page": b.locator.get("page"),
            "type": b.type.value,
            "text": b.text,
        }
        for b in blocks
    ]
    return (
        f"GOLDEN SCHEMA:\n{schema_block}\n\n"
        f"PARAGRAPHS:\n{json.dumps(paragraphs, ensure_ascii=False)}\n\n"
        "Emit extractions as JSON matching the response schema. "
        "Empty list is valid."
    )


def _parse_extractions(payload: dict[str, Any]) -> list[NarrativeExtraction]:
    items = payload.get("extractions", [])
    out: list[NarrativeExtraction] = []
    for item in items:
        try:
            out.append(NarrativeExtraction.model_validate(item))
        except Exception:
            continue
    return out


_ANTHROPIC_RESPONSE_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "extractions": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "column": {"type": "string"},
                    "value": {"type": ["number", "string", "boolean"]},
                    "unit": {"type": ["string", "null"]},
                    "evidence": {"type": "string"},
                    "source_paragraph_idx": {"type": "integer"},
                    "confidence": {"type": "number"},
                    "rationale": {"type": ["string", "null"]},
                },
                "required": [
                    "column", "value", "evidence",
                    "source_paragraph_idx", "confidence",
                ],
            },
        }
    },
    "required": ["extractions"],
}

_GEMINI_RESPONSE_SCHEMA: dict[str, Any] = {
    "type": "OBJECT",
    "properties": {
        "extractions": {
            "type": "ARRAY",
            "items": {
                "type": "OBJECT",
                "properties": {
                    "column": {"type": "STRING"},
                    "value": {"type": "STRING"},
                    "unit": {"type": "STRING", "nullable": True},
                    "evidence": {"type": "STRING"},
                    "source_paragraph_idx": {"type": "INTEGER"},
                    "confidence": {"type": "NUMBER"},
                    "rationale": {"type": "STRING", "nullable": True},
                },
                "required": [
                    "column", "value", "evidence",
                    "source_paragraph_idx", "confidence",
                ],
            },
        }
    },
    "required": ["extractions"],
}
