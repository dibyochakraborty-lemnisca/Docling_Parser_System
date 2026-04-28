from __future__ import annotations

from pathlib import Path

import pandas as pd

from fermdocs.domain.models import ParsedTable
from fermdocs.parsing.base import FileParser


class ExcelParser(FileParser):
    def supports(self, path: Path) -> bool:
        return path.suffix.lower() in {".xlsx", ".xls"}

    def parse(self, path: Path) -> list[ParsedTable]:
        xls = pd.ExcelFile(path)
        tables: list[ParsedTable] = []
        for sheet_name in xls.sheet_names:
            df = xls.parse(sheet_name, dtype=str, keep_default_na=False)
            if df.empty:
                continue
            headers = [str(c).strip() for c in df.columns]
            rows = [
                [_normalize(v) for v in row] for row in df.itertuples(index=False, name=None)
            ]
            tables.append(
                ParsedTable(
                    table_id=f"{path.name}#{sheet_name}",
                    headers=headers,
                    rows=rows,
                    locator={"format": "xlsx", "file": path.name, "sheet": sheet_name},
                )
            )
        return tables


def _normalize(v: object) -> object:
    if isinstance(v, str):
        s = v.strip()
        return s if s != "" else None
    return v
