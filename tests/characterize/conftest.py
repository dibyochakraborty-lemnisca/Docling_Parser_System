"""Shared fixtures for characterization tests."""

import json
from datetime import datetime
from pathlib import Path
from uuid import UUID

import pytest

FIXTURES_DIR = Path(__file__).resolve().parents[2] / "evals" / "characterize" / "fixtures"


@pytest.fixture
def fixtures_dir() -> Path:
    return FIXTURES_DIR


@pytest.fixture(params=["01_boundary", "02_missing_data", "03_multi_run"])
def fixture_case(request, fixtures_dir):
    name = request.param
    dossier = json.loads((fixtures_dir / name / "dossier.json").read_text())
    expected = json.loads((fixtures_dir / name / "expected_output.json").read_text())
    return {
        "name": name,
        "dossier": dossier,
        "expected": expected,
        "characterization_id": UUID(expected["meta"]["characterization_id"]),
        "generation_timestamp": datetime.fromisoformat(
            expected["meta"]["generation_timestamp"]
        ),
    }
