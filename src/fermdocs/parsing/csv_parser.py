from __future__ import annotations

from pathlib import Path

import pandas as pd

from fermdocs.domain.models import ParsedTable, ParseResult
from fermdocs.parsing.base import FileParser


class CsvParser(FileParser):
    def supports(self, path: Path) -> bool:
        return path.suffix.lower() in {".csv", ".tsv"}

    def parse(self, path: Path) -> ParseResult:
        sep = "\t" if path.suffix.lower() == ".tsv" else ","
        source_id = f"{path.name}#0"
        df = pd.read_csv(path, sep=sep, dtype=str, keep_default_na=False)
        headers = [str(c).strip() for c in df.columns]
        rows: list[list[object]] = [
            [_normalize(v) for v in row] for row in df.itertuples(index=False, name=None)
        ]
        # Header-less grid for the layout detector (handles a CSV whose real
        # header isn't the first line, e.g. a title/units preamble).
        raw = pd.read_csv(
            path, sep=sep, dtype=str, header=None, keep_default_na=False
        )
        raw_grids = {
            source_id: [
                [_normalize(v) for v in row]
                for row in raw.itertuples(index=False, name=None)
            ]
        }
        return ParseResult(
            tables=[
                ParsedTable(
                    table_id=source_id,
                    headers=headers,
                    rows=rows,
                    locator={"format": "csv", "file": path.name, "section": "table"},
                )
            ],
            raw_grids=raw_grids,
        )


def _normalize(v: object) -> object:
    if isinstance(v, str):
        s = v.strip()
        return s if s != "" else None
    return v
