"""Integration test: PDF -> tables + narrative -> mapper + extractor -> observations.

Uses fakes for Docling, repo, file store. Real pipeline + real evidence verification
+ real dedup logic.
"""
from __future__ import annotations

import uuid
from pathlib import Path
from typing import Any

import pandas as pd
import pytest

from fermdocs.domain.models import (
    DataType,
    GoldenColumn,
    GoldenSchema,
    NarrativeExtraction,
)
from fermdocs.file_store.base import StoredFile
from fermdocs.mapping.mapper import FakeHeaderMapper
from fermdocs.parsing.pdf_parser import DoclingPdfParser
from fermdocs.parsing.router import FormatRouter
from fermdocs.pipeline import IngestionPipeline
from fermdocs.units.converter import UnitConverter
from tests.unit.test_narrative_pdf_parser import (
    _FakeConverter,
    _FakeDocument,
    _FakeTable,
    _FakeText,
)


class _FakeRepo:
    def __init__(self):
        self.observations = []
        self.residuals = []
        self.parsed = []

    def upsert_experiment(self, *a, **k): pass
    def find_or_create_file(self, record): return record.file_id, True
    def mark_file_parsed(self, file_id, status, error=None, parsed_at=None):
        self.parsed.append((file_id, status))
    def write_observations(self, obs):
        self.observations.extend(obs); return len(obs)
    def write_residual(self, file_id, exp_id, payload, ver):
        self.residuals.append({"file_id": file_id, "payload": payload.model_dump()})
        return uuid.uuid4()


class _FakeStore:
    def put(self, src):
        return StoredFile(sha256="deadbeef", storage_path=str(src), size_bytes=src.stat().st_size)
    def open(self, p): return Path(p).read_bytes()


class _ScriptedExtractor:
    """Returns canned extractions for a given paragraph."""
    def __init__(self, extractions: list[NarrativeExtraction]):
        self._extractions = extractions
        self.calls = 0
    def extract(self, blocks, schema):
        self.calls += 1
        return self._extractions


def _schema() -> GoldenSchema:
    return GoldenSchema(
        version="test-1.0",
        columns=[
            GoldenColumn(name="final_titer_g_l", description="titer",
                         data_type=DataType.FLOAT, canonical_unit="g/L",
                         synonyms=["titer"]),
            GoldenColumn(name="temperature_c", description="temp",
                         data_type=DataType.FLOAT, canonical_unit="degC",
                         synonyms=["temp", "temperature"]),
            GoldenColumn(name="strain_id", description="strain",
                         data_type=DataType.TEXT, canonical_unit=None,
                         synonyms=["strain"]),
        ],
    )


def _build_pipeline_with_extractor(extractor):
    return IngestionPipeline(
        router=FormatRouter([DoclingPdfParser(converter=_make_doc())]),
        mapper=FakeHeaderMapper(),
        unit_converter=UnitConverter(),
        repository=_FakeRepo(),
        file_store=_FakeStore(),
        schema=_schema(),
        narrative_extractor=extractor,
    )


def _make_doc() -> _FakeConverter:
    df = pd.DataFrame({"Titer (g/L)": ["14.2"]})
    doc = _FakeDocument(
        tables=[_FakeTable(df=df, page_no=1)],
        texts=[
            _FakeText(
                text="The reactor was operated at 30 degrees C with strain HEX-12.",
                label="text", page_no=1,
            ),
        ],
    )
    return _FakeConverter(doc)


@pytest.fixture
def pdf_path(tmp_path: Path) -> Path:
    p = tmp_path / "experiment.pdf"
    p.write_bytes(b"%PDF-1.4 fake")
    return p


def test_narrative_blocks_always_captured_in_residual(pdf_path: Path):
    """Tier 1: even with no extractor, paragraphs land in residual."""
    pipeline = _build_pipeline_with_extractor(extractor=None)
    result = pipeline.ingest("EXP-NARR-1", [pdf_path])

    repo = pipeline._repo
    assert len(repo.residuals) == 1
    narrative = repo.residuals[0]["payload"]["narrative"]
    assert len(narrative) == 1
    assert "30 degrees C" in narrative[0]["text"]


def test_valid_extraction_yields_narrative_observation(pdf_path: Path):
    extractor = _ScriptedExtractor([
        NarrativeExtraction(
            column="temperature_c",
            value=30,
            unit="degC",
            evidence="operated at 30 degrees C",
            source_paragraph_idx=0,
            confidence=0.95,
        ),
    ])
    pipeline = _build_pipeline_with_extractor(extractor)
    pipeline.ingest("EXP-NARR-2", [pdf_path])

    repo = pipeline._repo
    narrative_obs = [
        o for o in repo.observations
        if o.source_locator.get("section") == "narrative"
    ]
    assert len(narrative_obs) == 1
    obs = narrative_obs[0]
    assert obs.column_name == "temperature_c"
    # Confidence cap applied: input 0.95 -> capped at 0.85
    assert obs.mapping_confidence == 0.85
    assert obs.needs_review is True
    assert obs.value_canonical["extracted_via"] == "narrative_llm"
    assert obs.source_locator["evidence_quote"] == "operated at 30 degrees C"


def test_hallucinated_evidence_rejected(pdf_path: Path):
    extractor = _ScriptedExtractor([
        NarrativeExtraction(
            column="temperature_c",
            value=30,
            unit="degC",
            evidence="this string is not in the source paragraph",
            source_paragraph_idx=0,
            confidence=0.95,
        ),
    ])
    pipeline = _build_pipeline_with_extractor(extractor)
    pipeline.ingest("EXP-NARR-3", [pdf_path])

    repo = pipeline._repo
    narrative_obs = [
        o for o in repo.observations
        if o.source_locator.get("section") == "narrative"
    ]
    assert len(narrative_obs) == 0


def test_invalid_column_rejected(pdf_path: Path):
    extractor = _ScriptedExtractor([
        NarrativeExtraction(
            column="nonexistent_column",
            value=30,
            evidence="30 degrees C",
            source_paragraph_idx=0,
            confidence=0.95,
        ),
    ])
    pipeline = _build_pipeline_with_extractor(extractor)
    pipeline.ingest("EXP-NARR-4", [pdf_path])

    repo = pipeline._repo
    narrative_obs = [
        o for o in repo.observations
        if o.source_locator.get("section") == "narrative"
    ]
    assert len(narrative_obs) == 0


def test_dedup_against_table_observation(pdf_path: Path):
    """If table reports titer=14.2 and prose also says 14.2, narrative is dropped."""
    # The fake table has Titer (g/L) = 14.2 in the doc above.
    # Using FakeHeaderMapper: 'Titer (g/L)' would not map (no exact synonym match),
    # so the table won't produce a final_titer_g_l observation in this test.
    # Use 'titer' synonym instead by overriding the schema column in the local doc.
    # For dedup, we need a real table observation; use a column the FakeMapper handles.
    # Use a custom doc where the table header is 'titer' (a synonym).
    df = pd.DataFrame({"titer": ["14.2"]})
    doc = _FakeDocument(
        tables=[_FakeTable(df=df, page_no=1)],
        texts=[
            _FakeText(
                text="The final titer reached 14.2 g/L at end of run.",
                label="text", page_no=1,
            ),
        ],
    )
    extractor = _ScriptedExtractor([
        NarrativeExtraction(
            column="final_titer_g_l",
            value=14.2,
            unit="g/L",
            evidence="final titer reached 14.2 g/L",
            source_paragraph_idx=0,
            confidence=0.92,
        ),
    ])
    pipeline = IngestionPipeline(
        router=FormatRouter([DoclingPdfParser(converter=_FakeConverter(doc))]),
        mapper=FakeHeaderMapper(),
        unit_converter=UnitConverter(),
        repository=_FakeRepo(),
        file_store=_FakeStore(),
        schema=_schema(),
        narrative_extractor=extractor,
    )
    result = pipeline.ingest("EXP-NARR-5", [pdf_path])

    repo = pipeline._repo
    table_obs = [o for o in repo.observations if o.source_locator.get("section") == "table"]
    narr_obs = [o for o in repo.observations if o.source_locator.get("section") == "narrative"]
    assert len(table_obs) == 1, "table should have produced one observation"
    assert len(narr_obs) == 0, "narrative should be deduped"
    assert result.files[0].narrative_extractions_deduped == 1
