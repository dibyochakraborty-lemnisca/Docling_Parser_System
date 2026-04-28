from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pandas as pd

from fermdocs.domain.models import NarrativeBlockType
from fermdocs.parsing.pdf_parser import DoclingPdfParser


@dataclass
class _FakeProv:
    page_no: int


@dataclass
class _FakeText:
    text: str
    label: str = "text"
    page_no: int = 1

    @property
    def prov(self) -> list[_FakeProv]:
        return [_FakeProv(page_no=self.page_no)]


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
    tables: list[_FakeTable] = field(default_factory=list)
    texts: list[_FakeText] = field(default_factory=list)


@dataclass
class _FakeResult:
    document: _FakeDocument


class _FakeConverter:
    def __init__(self, doc: _FakeDocument):
        self._doc = doc

    def convert(self, _path: str) -> _FakeResult:
        return _FakeResult(document=self._doc)


def test_pdf_parser_extracts_narrative_blocks(tmp_path: Path):
    pdf = tmp_path / "f.pdf"
    pdf.write_bytes(b"%PDF-1.4 fake")
    doc = _FakeDocument(
        tables=[],
        texts=[
            _FakeText(
                text="The reactor was inoculated with strain HEX-12 at OD600 = 0.4.",
                label="text", page_no=1,
            ),
            _FakeText(text="Methods", label="section_header", page_no=1),
            _FakeText(text="too short", label="text", page_no=1),  # filtered (<20 chars)
        ],
    )
    parser = DoclingPdfParser(converter=_FakeConverter(doc))
    result = parser.parse(pdf)
    assert len(result.tables) == 0
    assert len(result.narrative_blocks) == 1  # short one filtered
    block = result.narrative_blocks[0]
    assert "strain HEX-12" in block.text
    assert block.type == NarrativeBlockType.PARAGRAPH
    assert block.locator["section"] == "narrative"
    assert block.locator["page"] == 1
    assert "paragraph_idx" in block.locator


def test_pdf_parser_classifies_block_types(tmp_path: Path):
    pdf = tmp_path / "f.pdf"
    pdf.write_bytes(b"%PDF-1.4 fake")
    doc = _FakeDocument(
        texts=[
            _FakeText(text="This is a regular paragraph of about thirty chars.", label="text"),
            _FakeText(text="Section Header That Is Long Enough", label="section_header"),
            _FakeText(text="List item bullet point text here.", label="list_item"),
            _FakeText(text="Figure 1: OD600 over time for strain HEX-12.", label="caption"),
        ],
    )
    parser = DoclingPdfParser(converter=_FakeConverter(doc))
    result = parser.parse(pdf)
    types = {b.type for b in result.narrative_blocks}
    assert NarrativeBlockType.PARAGRAPH in types
    assert NarrativeBlockType.HEADING in types
    assert NarrativeBlockType.LIST_ITEM in types
    assert NarrativeBlockType.CAPTION in types


def test_pdf_parser_returns_both_tables_and_narrative(tmp_path: Path):
    pdf = tmp_path / "f.pdf"
    pdf.write_bytes(b"%PDF-1.4 fake")
    df = pd.DataFrame({"Strain": ["HEX-12"], "Titer (g/L)": ["14.2"]})
    doc = _FakeDocument(
        tables=[_FakeTable(df=df, page_no=2)],
        texts=[_FakeText(text="This paragraph is at least twenty characters long.")],
    )
    parser = DoclingPdfParser(converter=_FakeConverter(doc))
    result = parser.parse(pdf)
    assert len(result.tables) == 1
    assert len(result.narrative_blocks) == 1
    assert result.tables[0].locator["section"] == "table"
    assert result.narrative_blocks[0].locator["section"] == "narrative"
