"""Tests for the metadata-column guard in the ingestion pipeline.

Golden columns with canonical_unit=None (experiment_id, strain_id, organism,
product) describe experiment identity, not per-row measurements. Mappers
occasionally route batch-id-style columns to them. The pipeline must refuse
to write per-row observations against metadata columns.

Without this guard, IndPenSim's `Batch_ref` column (one batch number per
timestep) generated 4520 garbage observations against `experiment_id` —
identity belongs in dossier.experiment.process, not as time-series data.
"""

from __future__ import annotations

from fermdocs.domain.models import (
    GoldenColumn,
    GoldenSchema,
    MappingEntry,
    ParsedTable,
    TableMapping,
)
from fermdocs.pipeline import IngestionPipeline


def _schema() -> GoldenSchema:
    """Two columns: one metadata (canonical_unit=None), one measurement."""
    return GoldenSchema(
        version="2.0",
        columns=[
            GoldenColumn(
                name="experiment_id",
                description="Experiment identifier (metadata).",
                data_type="text",
                canonical_unit=None,
                synonyms=["batch_id", "run"],
                examples=[],
            ),
            GoldenColumn(
                name="biomass_g_l",
                description="Biomass concentration.",
                data_type="float",
                canonical_unit="g/L",
                nominal=0.5,
                std_dev=0.05,
                synonyms=["biomass", "X"],
                examples=[],
            ),
        ],
    )


def _table_with_metadata_and_data() -> ParsedTable:
    return ParsedTable(
        table_id="test#0",
        headers=["Batch_ref", "X"],
        rows=[
            ["1", "2.5"],
            ["1", "3.1"],
            ["1", "4.0"],
        ],
        locator={"format": "csv", "file": "test.csv", "section": "table"},
    )


def _mapping(table_id: str = "test#0") -> TableMapping:
    return TableMapping(
        table_id=table_id,
        entries=[
            MappingEntry(
                raw_header="Batch_ref",
                mapped_to="experiment_id",
                raw_unit=None,
                confidence=0.9,
                rationale="batch identifier",
            ),
            MappingEntry(
                raw_header="X",
                mapped_to="biomass_g_l",
                raw_unit=None,
                confidence=0.95,
                rationale="biomass abbreviation",
            ),
        ],
    )


# ---------- The guard ----------


def test_metadata_column_produces_no_observations():
    """experiment_id (canonical_unit=None) must produce 0 observations even
    when the mapper points at it.
    """
    schema = _schema()
    from fermdocs.parsing.run_id_resolver import RunIdResolver

    pipeline = IngestionPipeline.__new__(IngestionPipeline)
    pipeline._schema = schema
    pipeline._schema_index = schema.by_name()
    pipeline._converter = _FakeConverter()
    pipeline._normalizer = None
    pipeline._run_id_resolver = RunIdResolver()

    import uuid

    observations, unmapped = pipeline._observations_for_table(
        experiment_id="EXP-X",
        file_id=uuid.UUID(int=1),
        table=_table_with_metadata_and_data(),
        mapping=_mapping(),
    )

    # Only biomass observations should land
    assert len(observations) == 3
    assert all(o.column_name == "biomass_g_l" for o in observations)


def test_metadata_column_recorded_in_unmapped_with_reason():
    """The skip must be visible in unmapped_columns so we can audit which
    mappings the guard caught.
    """
    schema = _schema()
    from fermdocs.parsing.run_id_resolver import RunIdResolver

    pipeline = IngestionPipeline.__new__(IngestionPipeline)
    pipeline._schema = schema
    pipeline._schema_index = schema.by_name()
    pipeline._converter = _FakeConverter()
    pipeline._normalizer = None
    pipeline._run_id_resolver = RunIdResolver()

    import uuid

    _, unmapped = pipeline._observations_for_table(
        experiment_id="EXP-X",
        file_id=uuid.UUID(int=1),
        table=_table_with_metadata_and_data(),
        mapping=_mapping(),
    )
    metadata_skips = [
        u for u in unmapped if u.get("reason") == "metadata_column_not_observation"
    ]
    assert len(metadata_skips) == 1
    assert metadata_skips[0]["raw_header"] == "Batch_ref"
    assert metadata_skips[0]["mapped_to"] == "experiment_id"


def test_measurement_column_unaffected():
    """Variables with canonical_unit set still produce observations as before."""
    schema = _schema()
    from fermdocs.parsing.run_id_resolver import RunIdResolver

    pipeline = IngestionPipeline.__new__(IngestionPipeline)
    pipeline._schema = schema
    pipeline._schema_index = schema.by_name()
    pipeline._converter = _FakeConverter()
    pipeline._normalizer = None
    pipeline._run_id_resolver = RunIdResolver()

    import uuid

    observations, _ = pipeline._observations_for_table(
        experiment_id="EXP-X",
        file_id=uuid.UUID(int=1),
        table=_table_with_metadata_and_data(),
        mapping=_mapping(),
    )
    biomass_obs = [o for o in observations if o.column_name == "biomass_g_l"]
    assert len(biomass_obs) == 3


# ---------- Test scaffolding ----------


class _FakeConverter:
    """Minimal converter that returns OK status with the value passed through.
    Avoids pulling in pint and unit registry for what's a guard-logic test.
    """

    def convert(self, value, unit_raw, golden_unit, *, normalizer=None):
        from fermdocs.domain.models import ConversionStatus
        from types import SimpleNamespace

        try:
            v = float(value)
        except (TypeError, ValueError):
            v = value
        return SimpleNamespace(
            value_canonical=v,
            unit_canonical=golden_unit,
            status=ConversionStatus.OK,
            via="pint",
            hint=None,
        )
