"""Plan B Stage 2.2: end-to-end narrative wire-in tests.

Three seams covered, no live API:
  1. build_dossier → narrative_observations field on dossier
  2. CharacterizationPipeline → narrative_observations on output
  3. CLI bundle path → narrative_observations.json in bundle dir

The dossier-side extractor is mocked with a scripted client so all tests
are hermetic.
"""

from __future__ import annotations

import json
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any

import pytest

from dataclasses import dataclass

from fermdocs.bundle import BundleReader, BundleWriter
from fermdocs.domain.models import (
    NarrativeBlock,
    NarrativeBlockType,
)
from fermdocs.dossier import build_dossier
from fermdocs_characterize.pipeline import CharacterizationPipeline
from fermdocs_characterize.schema import (
    CharacterizationOutput,
    NarrativeTag,
)


@dataclass
class _StubExperiment:
    """Duck-typed stand-in for ExperimentRow. build_dossier accesses
    experiment_id, name, uploaded_by, created_at, status as attributes."""

    experiment_id: str
    name: str | None
    uploaded_by: str | None
    created_at: datetime | None
    status: str | None


# ---------------------------------------------------------------------------
# Lightweight fakes
# ---------------------------------------------------------------------------


class _FakeNarrativeClient:
    """Scripted narrative-insight client. Returns a canned list."""

    def __init__(self, response: list[dict[str, Any]]) -> None:
        self._response = response
        self.calls: list[str] = []

    def call(self, rendered_blocks: str) -> list[dict[str, Any]]:
        self.calls.append(rendered_blocks)
        return self._response


class _StubResidual:
    def __init__(self, narrative_payloads: list[dict[str, Any]]) -> None:
        self.residual_id = uuid.uuid4()
        self.file_id = uuid.uuid4()
        self.extractor_version = "v0.1.0"
        self.payload = {"narrative": narrative_payloads}


class _StubFile:
    def __init__(self) -> None:
        self.file_id = uuid.uuid4()
        self.filename = "report.pdf"
        self.sha256 = "deadbeef"
        self.page_count = 18
        self.parse_status = "ok"
        self.parse_error = None


class _StubRepository:
    """Minimal repo stand-in for build_dossier. Only implements the methods
    build_dossier actually calls."""

    def __init__(
        self,
        experiment_id: str,
        residuals: list[_StubResidual],
        files: list[_StubFile] | None = None,
    ) -> None:
        self._experiment_id = experiment_id
        self._residuals = residuals
        self._files = files or [_StubFile()]

    def fetch_experiment(self, experiment_id: str):
        return _StubExperiment(
            experiment_id=experiment_id,
            name="test",
            uploaded_by="tester",
            created_at=datetime(2026, 5, 3),
            status="ok",
        )

    def fetch_files(self, experiment_id: str):
        return self._files

    def fetch_active_observations(self, experiment_id: str):
        return []

    def fetch_residuals(self, experiment_id: str):
        return self._residuals

    def row_to_observation(self, row):  # pragma: no cover - never called (no obs)
        raise NotImplementedError


def _block_payload(text: str, page: int = 1, idx: int = 0) -> dict[str, Any]:
    """Serialized NarrativeBlock as it sits inside residual.payload['narrative']."""
    return NarrativeBlock(
        text=text,
        type=NarrativeBlockType.PARAGRAPH,
        locator={"page": page, "paragraph_idx": idx, "section": "Results", "format": "pdf"},
    ).model_dump(mode="json")


def _canned_observation(
    text: str,
    *,
    tag: str = "closure_event",
    run_id: str | None = "BATCH-01",
    time_h: float | None = 82.0,
    page: int = 3,
) -> dict[str, Any]:
    return {
        "tag": tag,
        "text": text,
        "source_locator": {"page": page, "section": "Results"},
        "run_id": run_id,
        "time_h": time_h,
        "affected_variables": ["viability_pct"],
        "confidence": 0.8,
    }


# ---------------------------------------------------------------------------
# Seam 1: build_dossier
# ---------------------------------------------------------------------------


def test_build_dossier_default_does_not_extract() -> None:
    """Without opt-in, build_dossier must not call the extractor and must
    emit an empty narrative_observations list."""
    repo = _StubRepository(
        "EXP-PB22",
        residuals=[
            _StubResidual(
                [_block_payload("terminated at 82h, white cells observed")]
            )
        ],
    )
    client = _FakeNarrativeClient([_canned_observation("X")])
    dossier = build_dossier(
        "EXP-PB22",
        repo,
        narrative_insight_client=client,  # provided but should not be called
    )
    assert dossier["narrative_observations"] == []
    assert dossier["ingestion_summary"]["narrative_insights_extracted"] == 0
    assert client.calls == []


def test_build_dossier_with_opt_in_calls_extractor() -> None:
    repo = _StubRepository(
        "EXP-PB22",
        residuals=[
            _StubResidual(
                [_block_payload("terminated at 82h, white cells observed")]
            )
        ],
    )
    canned = [_canned_observation("terminated at 82h, white cells observed")]
    client = _FakeNarrativeClient(canned)
    dossier = build_dossier(
        "EXP-PB22",
        repo,
        extract_narrative_insights=True,
        narrative_insight_client=client,
    )
    assert len(client.calls) == 1
    assert len(dossier["narrative_observations"]) == 1
    obs = dossier["narrative_observations"][0]
    assert obs["tag"] == "closure_event"
    assert "white cells" in obs["text"].lower()
    assert dossier["ingestion_summary"]["narrative_insights_extracted"] == 1


def test_build_dossier_env_flag_enables_extraction(monkeypatch) -> None:
    monkeypatch.setenv("FERMDOCS_EXTRACT_NARRATIVE_INSIGHTS", "true")
    repo = _StubRepository(
        "EXP-PB22",
        residuals=[_StubResidual([_block_payload("text body")])],
    )
    client = _FakeNarrativeClient([_canned_observation("text body")])
    dossier = build_dossier(
        "EXP-PB22",
        repo,
        narrative_insight_client=client,
    )
    assert len(client.calls) == 1
    assert len(dossier["narrative_observations"]) == 1


def test_build_dossier_explicit_arg_overrides_env(monkeypatch) -> None:
    """Explicit False overrides FERMDOCS_EXTRACT_NARRATIVE_INSIGHTS=true."""
    monkeypatch.setenv("FERMDOCS_EXTRACT_NARRATIVE_INSIGHTS", "true")
    repo = _StubRepository(
        "EXP-PB22",
        residuals=[_StubResidual([_block_payload("text body")])],
    )
    client = _FakeNarrativeClient([_canned_observation("text body")])
    dossier = build_dossier(
        "EXP-PB22",
        repo,
        extract_narrative_insights=False,
        narrative_insight_client=client,
    )
    assert client.calls == []
    assert dossier["narrative_observations"] == []


def test_build_dossier_extractor_failure_yields_empty_no_block() -> None:
    """Extractor errors must not break the rest of build_dossier."""

    class _BoomClient:
        def call(self, rendered_blocks: str):
            raise RuntimeError("upstream API failure")

    repo = _StubRepository(
        "EXP-PB22",
        residuals=[_StubResidual([_block_payload("x")])],
    )
    dossier = build_dossier(
        "EXP-PB22",
        repo,
        extract_narrative_insights=True,
        narrative_insight_client=_BoomClient(),
    )
    assert dossier["narrative_observations"] == []
    assert dossier["ingestion_summary"]["narrative_insights_extracted"] == 0


# ---------------------------------------------------------------------------
# Seam 2: CharacterizationPipeline materializes narratives from the dossier
# ---------------------------------------------------------------------------


def _minimal_dossier_with_narrative(narratives: list[dict[str, Any]]) -> dict[str, Any]:
    """A dossier with no observations but with narrative observations.
    The characterize pipeline still runs end-to-end (just with empty
    findings/trajectories)."""
    return {
        "experiment": {"experiment_id": "EXP-PB22"},
        "golden_columns": {},
        "narrative_observations": narratives,
        "_specs": {},
    }


def test_characterize_pipeline_renamespaces_narrative_ids() -> None:
    """The dossier-side extractor used a transient UUID; characterize must
    swap the namespace to its own characterization_id."""
    transient_uuid = uuid.uuid4()
    dossier_narrative = {
        "narrative_id": f"{transient_uuid}:N-0001",
        "tag": "closure_event",
        "text": "terminated at 82h",
        "source_locator": {"page": 3, "section": "Results"},
        "run_id": "BATCH-01",
        "time_h": 82.0,
        "affected_variables": ["viability_pct"],
        "confidence": 0.8,
        "extraction_model": "gemini-3.1-pro-preview",
    }
    char_id = uuid.UUID(int=99)
    pipeline = CharacterizationPipeline(validate=False)
    output = pipeline.run(
        _minimal_dossier_with_narrative([dossier_narrative]),
        characterization_id=char_id,
    )
    assert isinstance(output, CharacterizationOutput)
    assert len(output.narrative_observations) == 1
    n = output.narrative_observations[0]
    assert n.narrative_id == f"{char_id}:N-0001"
    assert n.tag == NarrativeTag.CLOSURE_EVENT
    assert n.text == "terminated at 82h"


def test_characterize_pipeline_preserves_position_when_id_malformed() -> None:
    """If a dossier-side narrative_id doesn't match the ':N-NNNN' tail
    pattern, characterize assigns by position rather than dropping."""
    char_id = uuid.UUID(int=100)
    dossier_narrative = {
        "narrative_id": "broken-id-shape",  # no ':N-' tail
        "tag": "observation",
        "text": "white cells observed",
        "confidence": 0.7,
        "extraction_model": "gemini-3.1-pro-preview",
    }
    pipeline = CharacterizationPipeline(validate=False)
    output = pipeline.run(
        _minimal_dossier_with_narrative([dossier_narrative]),
        characterization_id=char_id,
    )
    assert len(output.narrative_observations) == 1
    assert output.narrative_observations[0].narrative_id == f"{char_id}:N-0001"


def test_characterize_pipeline_drops_malformed_entries() -> None:
    """Garbage narrative entries are dropped; valid ones keep flowing."""
    char_id = uuid.UUID(int=101)
    dossier_narratives = [
        "not a dict",  # type: ignore[list-item]
        {"narrative_id": f"{char_id}:N-0001"},  # missing required fields
        {  # this one is valid
            "narrative_id": f"{uuid.uuid4()}:N-0002",
            "tag": "intervention",
            "text": "IPM added at 24h",
            "confidence": 0.8,
            "extraction_model": "gemini-3.1-pro-preview",
        },
    ]
    pipeline = CharacterizationPipeline(validate=False)
    output = pipeline.run(
        _minimal_dossier_with_narrative(dossier_narratives),
        characterization_id=char_id,
    )
    assert len(output.narrative_observations) == 1
    assert output.narrative_observations[0].text == "IPM added at 24h"
    # ID was renamespaced to char_id
    assert output.narrative_observations[0].narrative_id.startswith(str(char_id))


def test_characterize_pipeline_empty_when_dossier_missing_field() -> None:
    """Old dossiers without narrative_observations key produce empty list."""
    char_id = uuid.UUID(int=102)
    dossier = {
        "experiment": {"experiment_id": "EXP"},
        "golden_columns": {},
        "_specs": {},
    }
    pipeline = CharacterizationPipeline(validate=False)
    output = pipeline.run(dossier, characterization_id=char_id)
    assert output.narrative_observations == []


# ---------------------------------------------------------------------------
# Seam 3: characterize CLI writes narrative_observations.json into the bundle
# ---------------------------------------------------------------------------


def test_bundle_writer_persists_narrative_when_present(tmp_path: Path) -> None:
    """Direct test of the writer, mirroring what the CLI does. Confirms the
    file lands in the right path with the right contents."""
    char_id = uuid.UUID(int=200)
    pipeline = CharacterizationPipeline(validate=False)
    output = pipeline.run(
        _minimal_dossier_with_narrative(
            [
                {
                    "narrative_id": f"{uuid.uuid4()}:N-0001",
                    "tag": "closure_event",
                    "text": "terminated 82h",
                    "confidence": 0.8,
                    "extraction_model": "test-model",
                },
                {
                    "narrative_id": f"{uuid.uuid4()}:N-0002",
                    "tag": "intervention",
                    "text": "IPM added 24h",
                    "confidence": 0.8,
                    "extraction_model": "test-model",
                },
            ]
        ),
        characterization_id=char_id,
    )
    writer = BundleWriter.create(
        tmp_path,
        run_ids=["RUN-A"],
        golden_schema_version="2.0",
        pipeline_version="0.1.0",
    )
    writer.write_dossier({"experiment": {"experiment_id": "EXP"}})
    writer.write_characterization(output.model_dump_json())
    if output.narrative_observations:
        writer.write_narrative_observations(
            json.dumps(
                [n.model_dump(mode="json") for n in output.narrative_observations],
                indent=2,
            )
        )
    bundle_path = writer.finalize()

    reader = BundleReader(bundle_path)
    assert reader.has_narrative_observations()
    payload = json.loads(reader.get_narrative_observations_json())
    assert len(payload) == 2
    assert {p["tag"] for p in payload} == {"closure_event", "intervention"}


def test_bundle_skipped_when_no_narrative(tmp_path: Path) -> None:
    """Empty narrative list → no file written. Reader returns empty list."""
    pipeline = CharacterizationPipeline(validate=False)
    output = pipeline.run(
        _minimal_dossier_with_narrative([]), characterization_id=uuid.UUID(int=201)
    )
    writer = BundleWriter.create(
        tmp_path,
        run_ids=["RUN-A"],
        golden_schema_version="2.0",
        pipeline_version="0.1.0",
    )
    writer.write_dossier({"experiment": {"experiment_id": "EXP"}})
    writer.write_characterization(output.model_dump_json())
    if output.narrative_observations:  # this branch should not execute
        writer.write_narrative_observations("[]")
    bundle_path = writer.finalize()

    reader = BundleReader(bundle_path)
    assert reader.has_narrative_observations() is False
    # Backward-compat: missing file reads as empty list
    assert reader.get_narrative_observations_json() == "[]"


# ---------------------------------------------------------------------------
# End-to-end through all three seams (still mocked)
# ---------------------------------------------------------------------------


def test_full_pipeline_dossier_to_bundle(tmp_path: Path) -> None:
    """Build dossier with extraction → run characterize → write bundle.
    Verify a closure_event observation makes it all the way through."""
    repo = _StubRepository(
        "EXP-PB22-E2E",
        residuals=[
            _StubResidual(
                [_block_payload("BATCH-01 REPORT", page=1, idx=0),
                 _block_payload(
                     "Cultivation was terminated at 82 h due to onset of cell death"
                     " and the appearance of white cells during centrifugation.",
                     page=3,
                     idx=4,
                 )]
            )
        ],
    )
    client = _FakeNarrativeClient(
        [
            _canned_observation(
                "Cultivation was terminated at 82 h due to onset of cell death"
                " and the appearance of white cells during centrifugation."
            )
        ]
    )
    dossier = build_dossier(
        "EXP-PB22-E2E",
        repo,
        extract_narrative_insights=True,
        narrative_insight_client=client,
    )
    assert dossier["narrative_observations"]

    char_id = uuid.UUID(int=300)
    pipeline = CharacterizationPipeline(validate=False)
    output = pipeline.run(dossier, characterization_id=char_id)
    assert len(output.narrative_observations) == 1
    n = output.narrative_observations[0]
    assert n.narrative_id == f"{char_id}:N-0001"  # renamespaced
    assert "white cells" in n.text.lower()

    writer = BundleWriter.create(
        tmp_path,
        run_ids=["BATCH-01"],
        golden_schema_version="2.0",
        pipeline_version="0.1.0",
    )
    writer.write_dossier(dossier)
    writer.write_characterization(output.model_dump_json())
    writer.write_narrative_observations(
        json.dumps(
            [n.model_dump(mode="json") for n in output.narrative_observations],
            indent=2,
        )
    )
    bundle_path = writer.finalize()

    reader = BundleReader(bundle_path)
    assert reader.has_narrative_observations()
    payload = json.loads(reader.get_narrative_observations_json())
    assert len(payload) == 1
    assert payload[0]["tag"] == "closure_event"
