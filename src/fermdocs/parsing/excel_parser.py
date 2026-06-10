from __future__ import annotations

from pathlib import Path

import pandas as pd

from fermdocs.domain.models import ParsedTable, ParseResult
from fermdocs.parsing.base import FileParser


class ExcelParser(FileParser):
    def supports(self, path: Path) -> bool:
        return path.suffix.lower() in {".xlsx", ".xls"}

    def parse(self, path: Path) -> ParseResult:
        xls = pd.ExcelFile(path)
        tables: list[ParsedTable] = []
        raw_grids: dict[str, list[list[object]]] = {}
        for sheet_name in xls.sheet_names:
            source_id = f"{path.name}#{sheet_name}"
            # Header-less grid: every cell, no row-0-is-header assumption.
            # The layout detector locates the real header row from this.
            raw = xls.parse(sheet_name, header=None, dtype=str, keep_default_na=False)
            if not raw.empty:
                raw_grids[source_id] = [
                    [_normalize(v) for v in row]
                    for row in raw.itertuples(index=False, name=None)
                ]
            df = xls.parse(sheet_name, dtype=str, keep_default_na=False)
            if df.empty:
                continue
            headers = [str(c).strip() for c in df.columns]
            rows = [
                [_normalize(v) for v in row] for row in df.itertuples(index=False, name=None)
            ]
            tables.append(
                ParsedTable(
                    table_id=source_id,
                    headers=headers,
                    rows=rows,
                    locator={
                        "format": "xlsx", "file": path.name,
                        "sheet": sheet_name, "section": "table",
                    },
                )
            )
        return ParseResult(tables=tables, raw_grids=raw_grids)


def _normalize(v: object) -> object:
    if isinstance(v, str):
        s = v.strip()
        return s if s != "" else None
    return v
