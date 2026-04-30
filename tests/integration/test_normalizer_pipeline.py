"""Integration test for the normalizer wired through the full pipeline.

Uses fakes for Repository and FileStore so the test runs without Postgres.
Exercises: CSV parser -> FakeHeaderMapper -> UnitConverter -> RuleBasedNormalizer
-> assertion that observations have correct via field and value_canonical shape.
"""
from __future__ import annotations

import uuid
from pathlib import Path
from typing import Any

import pytest

from fermdocs.domain.models import GoldenColumn, GoldenSchema, DataType
from fermdocs.file_store.base import StoredFile
from fermdocs.mapping.mapper import FakeHeaderMapper
from fermdocs.parsing.csv_parser import CsvParser
from fermdocs.parsing.router import FormatRouter
from fermdocs.pipeline import IngestionPipeline
from fermdocs.units.converter import UnitConverter
from fermdocs.units.normalizer import ChainNormalizer, RuleBasedNormalizer


class _FakeRepo:
    def __init__(self):
        self.experiments: list[str] = []
        self.files: dict[uuid.UUID, dict] = {}
        self.observations: list[Any] = []
        self.residuals: list[dict] = []
        self.parsed_marks: list[tuple] = []

    def upsert_experiment(self, experiment_id, name=None, uploaded_by=None):
        self.experiments.append(experiment_id)

    def find_or_create_file(self, record):
        self.files[record.file_id] = {"record": record}
        return record.file_id, True

    def mark_file_parsed(self, file_id, status, error=None, parsed_at=None):
        self.parsed_marks.append((file_id, status, error))

    def write_observations(self, obs_list):
        self.observations.extend(obs_list)
        return len(obs_list)

    def write_residual(self, file_id, experiment_id, payload, extractor_version):
        self.residuals.append({"file_id": file_id, "payload": payload.model_dump()})
        return uuid.uuid4()


class _FakeFileStore:
    def __init__(self):
        self.put_calls = 0

    def put(self, src):
        self.put_calls += 1
        return StoredFile(sha256="deadbeef", storage_path=str(src), size_bytes=src.stat().st_size)

    def open(self, storage_path):
        return Path(storage_path).read_bytes()


def _minimal_schema() -> GoldenSchema:
    return GoldenSchema(
        version="test-1.0",
        columns=[
            GoldenColumn(
                name="biomass_g_l",
                description="biomass",
                data_type=DataType.FLOAT,
                canonical_unit="g/L",
                synonyms=["biomass", "X"],
            ),
            GoldenColumn(
                name="volume_l",
                description="volume",
                data_type=DataType.FLOAT,
                canonical_unit="L",
                synonyms=["volume"],
            ),
        ],
    )


@pytest.fixture
def weird_csv(tmp_path: Path) -> Path:
    p = tmp_path / "weird.csv"
    # Biomass uses normal g/L (pint handles directly).
    # Volume uses 'of broth' annotation (pint fails: 'of' not a unit)
    # -> rule-based normalizer strips 'of broth' -> retry pint succeeds.
    p.write_text(
        "Biomass (g/L),Volume (L of broth)\n"
        "0.5,58000\n"
        "0.6,58100\n"
    )
    return p


def test_pipeline_uses_normalizer_for_unicode_superscripts(weird_csv: Path):
    repo = _FakeRepo()
    store = _FakeFileStore()
    router = FormatRouter([CsvParser()])
    pipeline = IngestionPipeline(
        router=router,
        mapper=FakeHeaderMapper(),
        unit_converter=UnitConverter(),
        repository=repo,
        file_store=store,
        schema=_minimal_schema(),
        normalizer=ChainNormalizer([RuleBasedNormalizer()]),
    )
    result = pipeline.ingest("EXP-INT-1", [weird_csv])
    assert result.all_ok
    assert result.files[0].observations_written >= 4

    biomass_obs = [o for o in repo.observations if o.column_name == "biomass_g_l"]
    volume_obs = [
        o for o in repo.observations if o.column_name == "volume_l"
    ]
    assert len(biomass_obs) == 2
    assert len(volume_obs) == 2

    # Pint handles g/L directly
    for o in biomass_obs:
        assert o.value_canonical["via"] == "pint"
        assert "normalization" not in o.value_canonical

    # Volume needed the rule-based normalizer (stripped 'of broth')
    for o in volume_obs:
        assert o.value_canonical["via"] == "rule_based"
        assert o.value_canonical["normalization"]["action"] == "use_pint_expr"
        assert "of" not in o.value_canonical["normalization"]["pint_expr"]


def test_pipeline_without_normalizer_marks_unparseable_units_failed(weird_csv: Path):
    repo = _FakeRepo()
    store = _FakeFileStore()
    router = FormatRouter([CsvParser()])
    pipeline = IngestionPipeline(
        router=router,
        mapper=FakeHeaderMapper(),
        unit_converter=UnitConverter(),
        repository=repo,
        file_store=store,
        schema=_minimal_schema(),
        normalizer=None,
    )
    result = pipeline.ingest("EXP-INT-2", [weird_csv])
    assert result.all_ok

    volume_obs = [
        o for o in repo.observations if o.column_name == "volume_l"
    ]
    # Without normalizer, volume values exist as raw but value_canonical is None
    for o in volume_obs:
        assert o.value_canonical is None
        assert o.conversion_status == "failed"
