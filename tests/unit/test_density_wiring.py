"""0b — per-run density is threaded into the pipeline's unit conversion.

With density, a %w/w concentration column resolves to the correct g/L (no longer
the ~10x-low raw number, no longer a silent pass-through). Without density for that
run, the converter refuses the channel (legible FAILED), never imputes. This is the
production wiring that makes 0a's converter actually correct the units end to end.
"""
from __future__ import annotations

import uuid

from fermdocs.domain.models import (
    ConversionStatus,
    GoldenColumn,
    GoldenSchema,
    MappingEntry,
    ParsedTable,
    TableMapping,
)
from fermdocs.parsing.run_id_resolver import RunIdResolver
from fermdocs.pipeline import IngestionPipeline
from fermdocs.units.converter import UnitConverter
from fermdocs.units.normalizer import build_default_normalizer


def _schema() -> GoldenSchema:
    return GoldenSchema(version="2.0", columns=[
        GoldenColumn(name="product_g_l", description="Product titer.", data_type="float",
                     canonical_unit="g/L", objective=True, synonyms=["lactic acid", "titer"],
                     examples=[]),
    ])


def _pipeline():
    p = IngestionPipeline.__new__(IngestionPipeline)
    schema = _schema()
    p._schema = schema
    p._schema_index = schema.by_name()
    p._converter = UnitConverter()
    p._normalizer = build_default_normalizer(use_llm=False)
    p._run_id_resolver = RunIdResolver()
    return p


def _table():
    return ParsedTable(
        table_id="praaj#0",
        headers=["Time", "Lactic Acid(%w/w)"],
        rows=[["40", "15.04"]],
        locator={"format": "xlsx", "file": "praaj.xlsx", "section": "table"},
    )


def _mapping():
    return TableMapping(table_id="praaj#0", entries=[
        MappingEntry(raw_header="Lactic Acid(%w/w)", mapped_to="product_g_l",
                     raw_unit="%w/w", confidence=0.95, rationale="lactic acid %w/w"),
    ])


def test_density_converts_pct_ww_to_correct_gl():
    p = _pipeline()
    obs, _ = p._observations_for_table(
        experiment_id="EXP", file_id=uuid.UUID(int=1),
        table=_table(), mapping=_mapping(),
        manifest_run_id="B541", densities={"B541": 1.0947})
    prod = [o for o in obs if o.column_name == "product_g_l"]
    assert len(prod) == 1
    o = prod[0]
    assert o.conversion_status == ConversionStatus.OK
    # 15.04 %w/w x 1.0947 g/mL x 10 = 164.6 g/L  (the sheet's own stated g/L)
    assert abs(o.value_canonical["value"] - 164.6) < 0.5
    assert o.value_canonical["value"] != 15.04          # not the raw pass-through


def test_no_density_for_run_refuses_legibly():
    p = _pipeline()
    obs, _ = p._observations_for_table(
        experiment_id="EXP", file_id=uuid.UUID(int=1),
        table=_table(), mapping=_mapping(),
        manifest_run_id="B999", densities={"B541": 1.0947})  # B999 has no density
    prod = [o for o in obs if o.column_name == "product_g_l"]
    assert len(prod) == 1
    o = prod[0]
    assert o.conversion_status == ConversionStatus.FAILED  # refused, not imputed
    assert o.value_canonical is None
    assert "density" in (o.conversion_error or "")


def test_densities_absent_entirely_is_graceful_refusal():
    # No densities passed at all (e.g. density not extracted) -> %w/w channel
    # refuses rather than silently storing the raw number.
    p = _pipeline()
    obs, _ = p._observations_for_table(
        experiment_id="EXP", file_id=uuid.UUID(int=1),
        table=_table(), mapping=_mapping(), manifest_run_id="B541")
    o = [o for o in obs if o.column_name == "product_g_l"][0]
    assert o.conversion_status == ConversionStatus.FAILED
    assert o.value_canonical is None
