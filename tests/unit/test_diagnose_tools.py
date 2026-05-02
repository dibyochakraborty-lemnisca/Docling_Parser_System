"""Tool-surface tests. Three tools, found / not-found per tool."""

from __future__ import annotations

import uuid
from datetime import datetime

import pytest

from fermdocs_characterize.schema import (
    CharacterizationOutput,
    DataQuality,
    EvidenceStrength,
    ExtractedVia,
    Finding,
    FindingType,
    Meta,
    Severity,
    Trajectory,
)
from fermdocs_characterize.specs import DictSpecsProvider, Spec
from fermdocs_diagnose.tools import get_finding, get_spec, get_trajectory


CHAR_ID = uuid.UUID(int=42)


@pytest.fixture
def output() -> CharacterizationOutput:
    return CharacterizationOutput(
        meta=Meta(
            schema_version="2.0",
            characterization_version="v1.0.0",
            characterization_id=CHAR_ID,
            generation_timestamp=datetime(2026, 1, 1),
            source_dossier_ids=["EXP-X"],
        ),
        findings=[
            Finding(
                finding_id=f"{CHAR_ID}:F-0001",
                type=FindingType.RANGE_VIOLATION,
                severity=Severity.MAJOR,
                summary="biomass low",
                confidence=0.8,
                extracted_via=ExtractedVia.DETERMINISTIC,
                evidence_strength=EvidenceStrength(n_observations=1, n_independent_runs=1),
                evidence_observation_ids=["O-1"],
                variables_involved=["biomass_g_l"],
            ),
        ],
        trajectories=[
            Trajectory(
                trajectory_id="T-0001",
                run_id="RUN-1",
                variable="biomass_g_l",
                time_grid=[0.0, 1.0, 2.0],
                values=[1.0, 2.0, 3.0],
                imputation_flags=[False, False, False],
                source_observation_ids=["O-1"],
                unit="g/L",
                quality=1.0,
                data_quality=DataQuality(pct_missing=0.0, pct_imputed=0.0, pct_real=1.0),
            )
        ],
    )


@pytest.fixture
def specs() -> DictSpecsProvider:
    return DictSpecsProvider(
        {
            "biomass_g_l": Spec(
                nominal=10.0, std_dev=1.0, unit="g/L", provenance="schema"
            )
        }
    )


# ---------- get_finding ----------


def test_get_finding_found(output):
    got = get_finding(f"{CHAR_ID}:F-0001", output=output)
    assert isinstance(got, Finding)
    assert got.summary == "biomass low"


def test_get_finding_not_found_returns_error(output):
    got = get_finding(f"{CHAR_ID}:F-9999", output=output)
    assert isinstance(got, dict)
    assert "error" in got
    assert "hint" in got


# ---------- get_trajectory ----------


def test_get_trajectory_found(output):
    got = get_trajectory("RUN-1", "biomass_g_l", output=output)
    assert isinstance(got, Trajectory)
    assert got.unit == "g/L"


def test_get_trajectory_unknown_run(output):
    got = get_trajectory("RUN-X", "biomass_g_l", output=output)
    assert isinstance(got, dict)
    assert "error" in got
    assert "RUN-1" in got["hint"]


def test_get_trajectory_unknown_variable(output):
    got = get_trajectory("RUN-1", "unobtainium", output=output)
    assert isinstance(got, dict)
    assert "biomass_g_l" in got["hint"]


# ---------- get_spec ----------


def test_get_spec_found(specs):
    got = get_spec("biomass_g_l", specs=specs)
    assert isinstance(got, Spec)
    assert got.nominal == 10.0


def test_get_spec_missing_returns_error(specs):
    got = get_spec("missing_var", specs=specs)
    assert isinstance(got, dict)
    assert "error" in got
    assert "missing_var" in got["error"]
