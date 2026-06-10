"""One source column per canonical variable (regression: praaj phantom
'conflicting product' topic).

Two columns mapping to product_g_l — "Lactic Acid(%w/w)"=10.2 and
"Volume Corrected Lactic Acid (g/L)"=102 — are the same analyte in two units.
Ingesting both into product_g_l created a fake conflict. The reconciler keeps
the canonical-unit column and demotes the rest.
"""

from __future__ import annotations

from fermdocs.domain.models import (
    DataType, GoldenColumn, MappingEntry, MappingResult, TableMapping,
)
from fermdocs.pipeline import _dominant_source_units, _reconcile_duplicate_mappings


def _idx(unit="g/L"):
    return {
        "product_g_l": GoldenColumn(
            name="product_g_l", description="p", data_type=DataType.FLOAT,
            canonical_unit=unit,
        )
    }


def _e(header, unit, conf, mapped="product_g_l"):
    return MappingEntry(raw_header=header, mapped_to=mapped, raw_unit=unit, confidence=conf)


def test_prefers_canonical_unit_over_confidence():
    entries = [
        _e("Lactic Acid(%w/w)", "%w/w", 0.95),
        _e("Volume Corrected Lactic Acid (g/L)", "g/L", 0.70),
    ]
    demoted = _reconcile_duplicate_mappings(entries, _idx())
    # the g/L column wins despite lower confidence; %w/w is demoted to it
    assert demoted == {"Lactic Acid(%w/w)": "Volume Corrected Lactic Acid (g/L)"}


def test_no_unit_match_falls_back_to_confidence():
    entries = [_e("LA_a", "%w/w", 0.6), _e("LA_b", "%w/w", 0.9)]
    demoted = _reconcile_duplicate_mappings(entries, _idx())
    assert demoted == {"LA_a": "LA_b"}  # higher confidence kept


def test_single_mapping_not_demoted():
    entries = [_e("Lactic Acid (g/L)", "g/L", 0.9), MappingEntry(
        raw_header="pH", mapped_to="ph", raw_unit=None, confidence=0.9)]
    assert _reconcile_duplicate_mappings(entries, _idx()) == {}


def test_unmapped_columns_ignored():
    entries = [_e("x", None, 0.0, mapped=None), _e("y", None, 0.0, mapped=None)]
    assert _reconcile_duplicate_mappings(entries, _idx()) == {}


# --- cross-run consistency (regression: B474 phantom anomaly) -------------

def test_dominant_source_unit_is_the_one_most_runs_use():
    # 2 runs report product only in %w/w; one off-template run also has g/L.
    mr = MappingResult(tables=[
        TableMapping(table_id="R1", entries=[_e("LA (%w/w)", "%w/w", 0.9)]),
        TableMapping(table_id="R2", entries=[_e("LA (%w/w)", "%w/w", 0.9)]),
        TableMapping(table_id="B474", entries=[
            _e("LA (%w/w)", "%w/w", 0.9),
            _e("Volume Corrected LA (g/L)", "g/L", 0.9)]),
    ])
    dom = _dominant_source_units(mr)
    assert dom["product_g_l"] == "%w/w"   # 3 runs %w/w vs 1 run g/L


def test_cross_run_dominant_unit_beats_canonical_unit():
    # B474 has both; the cross-run dominant unit is %w/w, so its g/L column is
    # dropped -> B474 stays on the same scale as the other 14 runs (no 10x
    # phantom anomaly), even though g/L is the canonical unit.
    entries = [
        _e("Lactic Acid(%w/w)", "%w/w", 0.9),
        _e("Volume Corrected Lactic Acid (g/L)", "g/L", 0.9),
    ]
    demoted = _reconcile_duplicate_mappings(
        entries, _idx(), dominant_units={"product_g_l": "%w/w"}
    )
    assert demoted == {"Volume Corrected Lactic Acid (g/L)": "Lactic Acid(%w/w)"}


# --- header-dedupe before mapping (regression: praaj 15-sheet JSON truncation) ---

from fermdocs.domain.models import ParsedTable, MappingResult, TableMapping
from fermdocs.pipeline import IngestionPipeline


class _CountingMapper:
    def __init__(self):
        self.tables_seen = []

    def map(self, tables, schema):
        self.tables_seen.append(len(tables))
        return MappingResult(tables=[
            TableMapping(table_id=t.table_id, entries=[
                MappingEntry(raw_header=t.headers[0], mapped_to="product_g_l",
                             raw_unit="g/L", confidence=0.9)])
            for t in tables])


def test_map_tables_dedupes_identical_headers():
    # 15 sheets with identical headers must hit the mapper as ONE table, but
    # the result must still cover all 15 (this is what stopped the huge,
    # truncating mapper response that crashed the praaj run).
    p = IngestionPipeline.__new__(IngestionPipeline)
    p._mapper = _CountingMapper()
    p._schema = None
    tables = [ParsedTable(table_id=f"f#{i}", headers=["Time", "LA (g/L)"],
                          rows=[], locator={}) for i in range(15)]
    result = p._map_tables(tables)
    assert p._mapper.tables_seen == [1]                 # one unique signature
    assert {tm.table_id for tm in result.tables} == {f"f#{i}" for i in range(15)}
