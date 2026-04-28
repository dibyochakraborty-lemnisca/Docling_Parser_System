from __future__ import annotations

from pathlib import Path

import pandas as pd

from fermdocs.domain.models import ParsedTable
from fermdocs.parsing.base import FileParser


class CsvParser(FileParser):
    def supports(self, path: Path) -> bool:
        return path.suffix.lower() in {".csv", ".tsv"}

    def parse(self, path: Path) -> list[ParsedTable]:
        sep = "\t" if path.suffix.lower() == ".tsv" else ","
        df = pd.read_csv(path, sep=sep, dtype=str, keep_default_na=False)
        headers = [str(c).strip() for c in df.columns]
        rows: list[list[object]] = [
            [_normalize(v) for v in row] for row in df.itertuples(index=False, name=None)
        ]
        return [
            ParsedTable(
                table_id=f"{path.name}#0",
                headers=headers,
                rows=rows,
                locator={"format": "csv", "file": path.name},
            )
        ]


def _normalize(v: object) -> object:
    if isinstance(v, str):
        s = v.strip()
        return s if s != "" else None
    return v
