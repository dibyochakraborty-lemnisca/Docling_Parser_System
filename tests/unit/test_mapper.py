from __future__ import annotations

from fermdocs.domain.models import ParsedTable
from fermdocs.domain.golden_schema import load_schema
from fermdocs.mapping.mapper import FakeHeaderMapper


def test_fake_mapper_matches_synonyms_and_extracts_units():
    schema = load_schema()
    table = ParsedTable(
        table_id="t1",
        headers=["Strain", "Titer (g/L)", "WeirdHeader"],
        rows=[["HEX-12", "14.2", "x"]],
        locator={"format": "csv"},
    )
    result = FakeHeaderMapper().map([table], schema)
    assert len(result.tables) == 1
    by_header = {e.raw_header: e for e in result.tables[0].entries}
    assert by_header["Strain"].mapped_to == "strain_id"
    assert by_header["Titer (g/L)"].mapped_to == "final_titer_g_l"
    assert by_header["Titer (g/L)"].raw_unit == "g/L"
    assert by_header["WeirdHeader"].mapped_to is None


def test_fake_mapper_unit_extraction_handles_brackets():
    schema = load_schema()
    table = ParsedTable(
        table_id="t1",
        headers=["Volume [L]"],
        rows=[["2.0"]],
        locator={"format": "csv"},
    )
    result = FakeHeaderMapper().map([table], schema)
    entry = result.tables[0].entries[0]
    assert entry.mapped_to == "working_volume_l"
    assert entry.raw_unit == "L"
