from __future__ import annotations

import json

from fermdocs.domain.models import GoldenSchema, ParsedTable

_SYSTEM = (
    "You map raw tabular column headers from fermentation experiment reports "
    "to a fixed canonical schema of golden columns. You NEVER invent values; "
    "you only emit a header-to-column mapping with a confidence score.\n\n"
    "For each header you must return:\n"
    "  - mapped_to: the canonical golden column name, or null if no good match.\n"
    "  - raw_unit: the unit string as it appears in the header (e.g., 'g/L'), or null.\n"
    "  - confidence: a number 0..1 indicating how sure you are of the mapping.\n"
    "  - rationale: a short reason.\n"
)


def render_user_prompt(tables: list[ParsedTable], schema: GoldenSchema) -> str:
    schema_lines = []
    for col in schema.columns:
        examples = ", ".join(f"'{e.raw_header}'" for e in col.examples[:3])
        schema_lines.append(
            f"- {col.name} ({col.data_type.value}"
            + (f", {col.canonical_unit}" if col.canonical_unit else "")
            + f"): {col.description}"
            + (f" Synonyms: {', '.join(col.synonyms)}." if col.synonyms else "")
            + (f" Examples: {examples}." if examples else "")
        )
    schema_block = "\n".join(schema_lines)
    tables_block = json.dumps(
        [
            {
                "table_id": t.table_id,
                "headers": t.headers,
                "sample_rows": t.sample_rows(3),
            }
            for t in tables
        ],
        ensure_ascii=False,
        default=str,
    )
    return (
        f"GOLDEN SCHEMA:\n{schema_block}\n\n"
        f"TABLES TO MAP:\n{tables_block}\n\n"
        "Return JSON matching the response schema. Map every header. "
        "Set mapped_to to null when no canonical column fits."
    )


def system_prompt() -> str:
    return _SYSTEM
