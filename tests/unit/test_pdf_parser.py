from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd

from fermdocs.parsing.pdf_parser import DoclingPdfParser


@dataclass
class _FakeProv:
    page_no: int


@dataclass
class _FakeTable:
    df: pd.DataFrame
    page_no: int

    def export_to_dataframe(self) -> pd.DataFrame:
        return self.df

    @property
    def prov(self) -> list[_FakeProv]:
        return [_FakeProv(page_no=self.page_no)]


@dataclass
class _FakeDocument:
    tables: list[_FakeTable]


@dataclass
class _FakeResult:
    document: _FakeDocument


class _FakeConverter:
    def __init__(self, document: _FakeDocument) -> None:
        self._document = document

    def convert(self, _path: str) -> _FakeResult:
        return _FakeResult(document=self._document)


def test_pdf_parser_supports_pdf():
    assert DoclingPdfParser().supports(Path("foo.pdf"))
    assert not DoclingPdfParser().supports(Path("foo.csv"))


def test_pdf_parser_extracts_tables_with_provenance(tmp_path: Path):
    pdf_path = tmp_path / "sample.pdf"
    pdf_path.write_bytes(b"%PDF-1.4 fake")

    df = pd.DataFrame(
        {
            "Strain": ["HEX-12"],
            "Titer (g/L)": ["14.2"],
        }
    )
    document = _FakeDocument(tables=[_FakeTable(df=df, page_no=2)])
    parser = DoclingPdfParser(converter=_FakeConverter(document))

    tables = parser.parse(pdf_path)
    assert len(tables) == 1
    table = tables[0]
    assert table.headers == ["Strain", "Titer (g/L)"]
    assert table.rows == [["HEX-12", "14.2"]]
    assert table.locator["format"] == "pdf"
    assert table.locator["page"] == 2
    assert table.locator["table_idx"] == 0
    assert table.table_id == "sample.pdf#p2#t0"


def test_pdf_parser_skips_empty_tables(tmp_path: Path):
    pdf_path = tmp_path / "empty.pdf"
    pdf_path.write_bytes(b"%PDF-1.4 fake")
    document = _FakeDocument(tables=[_FakeTable(df=pd.DataFrame(), page_no=1)])
    parser = DoclingPdfParser(converter=_FakeConverter(document))
    assert parser.parse(pdf_path) == []
