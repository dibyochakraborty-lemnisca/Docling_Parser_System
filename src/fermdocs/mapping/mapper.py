from __future__ import annotations

from typing import Protocol

from fermdocs.domain.models import (
    GoldenSchema,
    MappingEntry,
    MappingResult,
    ParsedTable,
    TableMapping,
)


class HeaderMapper(Protocol):
    """Maps raw table headers to canonical golden columns.

    Implementations MUST accept many tables in one call so a multi-table file
    can be mapped with a single LLM round-trip.
    """

    def map(self, tables: list[ParsedTable], schema: GoldenSchema) -> MappingResult: ...


class FakeHeaderMapper:
    """Deterministic mapper for tests.

    Matches by case-insensitive equality on the column name or any synonym.
    Confidence is 1.0 on a synonym hit, 0.0 otherwise.
    """

    def map(self, tables: list[ParsedTable], schema: GoldenSchema) -> MappingResult:
        synonym_index: dict[str, str] = {}
        for col in schema.columns:
            synonym_index[col.name.lower()] = col.name
            for syn in col.synonyms:
                synonym_index[syn.lower()] = col.name
        results: list[TableMapping] = []
        for table in tables:
            entries: list[MappingEntry] = []
            for header in table.headers:
                normalized = _normalize_header(header).lower()
                mapped = synonym_index.get(normalized)
                entries.append(
                    MappingEntry(
                        raw_header=header,
                        mapped_to=mapped,
                        raw_unit=_extract_unit(header),
                        confidence=1.0 if mapped else 0.0,
                    )
                )
            results.append(TableMapping(table_id=table.table_id, entries=entries))
        return MappingResult(tables=results)


def _normalize_header(h: str) -> str:
    """Strip parenthetical/bracketed unit annotations, e.g. 'Titer (g/L)' -> 'Titer'."""
    out = h
    for open_, close_ in [("(", ")"), ("[", "]")]:
        if open_ in out and close_ in out:
            out = out.split(open_)[0]
    return out.strip().rstrip(":").strip()


def _extract_unit(h: str) -> str | None:
    for open_, close_ in [("(", ")"), ("[", "]")]:
        if open_ in h and close_ in h:
            inside = h.split(open_, 1)[1].rsplit(close_, 1)[0].strip()
            if inside:
                return inside
    return None
