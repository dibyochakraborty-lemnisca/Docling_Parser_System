from __future__ import annotations

from pathlib import Path

from fermdocs.parsing.csv_parser import CsvParser

FIXTURE = Path(__file__).parent.parent / "fixtures" / "sample_run.csv"


def test_csv_parser_supports_csv():
    assert CsvParser().supports(Path("foo.csv"))
    assert not CsvParser().supports(Path("foo.pdf"))


def test_csv_parser_extracts_headers_and_rows():
    result = CsvParser().parse(FIXTURE)
    assert len(result.tables) == 1
    assert result.narrative_blocks == []
    table = result.tables[0]
    assert "Strain" in table.headers
    assert "Titer (g/L)" in table.headers
    assert len(table.rows) == 2
    assert table.rows[0][0] == "HEX-12"
    assert table.locator["format"] == "csv"
    assert table.locator["section"] == "table"
