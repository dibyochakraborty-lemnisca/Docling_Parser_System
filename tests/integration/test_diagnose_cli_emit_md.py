"""CLI integration tests.

The CLI is a thin shell: read JSONs, run agent (no LLM in tests so it returns
error-output), write JSON, optionally render markdown sidecars. We assert the
plumbing: file IO, exit codes, markdown sidecars exist with expected names.
"""

from __future__ import annotations

import json
import uuid
from datetime import datetime
from pathlib import Path

import pytest
from click.testing import CliRunner

from fermdocs_characterize.schema import (
    CharacterizationOutput,
    Meta,
)
from fermdocs_diagnose.cli import cli


@pytest.fixture
def fixture_paths(tmp_path: Path) -> tuple[Path, Path, Path]:
    """Write a minimal upstream characterization + dossier JSON to disk.

    Returns (dossier_path, characterization_path, output_path).
    """
    char_id = uuid.UUID(int=42)
    upstream = CharacterizationOutput(
        meta=Meta(
            schema_version="2.0",
            characterization_version="v1.0.0",
            characterization_id=char_id,
            generation_timestamp=datetime(2026, 1, 1),
            source_dossier_ids=["EXP-X"],
        ),
    )
    char_path = tmp_path / "characterization.json"
    char_path.write_text(json.dumps(upstream.model_dump(mode="json"), indent=2))

    dossier = {
        "experiment": {
            "experiment_id": "EXP-X",
            "process": {
                "observed": {"organism": "Penicillium chrysogenum", "provenance": "llm_whitelisted"},
                "registered": {"process_id": "penicillin_indpensim", "provenance": "llm_whitelisted"},
            },
        },
        "ingestion_summary": {
            "schema_version": "2.0",
            "stale_schema_versions": [],
            "golden_coverage_percent": 80,
        },
        "golden_columns": {},
    }
    dossier_path = tmp_path / "dossier.json"
    dossier_path.write_text(json.dumps(dossier, indent=2))

    output_path = tmp_path / "diagnosis.json"
    return dossier_path, char_path, output_path


def test_cli_run_writes_diagnosis_json(fixture_paths):
    """Provider=none short-circuits to error-output; verifies plumbing
    without needing API keys. Live-LLM smoke tests are gated separately.
    """
    dossier, char, output = fixture_paths
    runner = CliRunner()
    result = runner.invoke(
        cli,
        [
            "run",
            "--dossier", str(dossier),
            "--characterization", str(char),
            "--output", str(output),
            "--provider", "none",
        ],
    )
    assert result.exit_code == 3
    assert output.exists()
    written = json.loads(output.read_text())
    assert written["meta"]["error"] == "no_llm_client_configured"


def test_cli_emit_markdown_writes_five_files(fixture_paths, tmp_path: Path):
    dossier, char, output = fixture_paths
    md_dir = tmp_path / "md"
    runner = CliRunner()
    result = runner.invoke(
        cli,
        [
            "run",
            "--dossier", str(dossier),
            "--characterization", str(char),
            "--output", str(output),
            "--emit-markdown", str(md_dir),
            "--provider", "none",
        ],
    )
    assert result.exit_code == 3  # error output, but markdown still emitted
    expected = {"Failures.md", "Trends.md", "Analysis.md", "Questions.md", "Diagnosis.md"}
    assert {p.name for p in md_dir.iterdir()} == expected


def test_cli_invalid_dossier_path_errors(tmp_path: Path):
    runner = CliRunner()
    result = runner.invoke(
        cli,
        [
            "run",
            "--dossier", str(tmp_path / "missing.json"),
            "--characterization", str(tmp_path / "missing.json"),
            "--output", str(tmp_path / "out.json"),
            "--provider", "none",
        ],
    )
    # Click handles missing file via path validator, exit_code != 0
    assert result.exit_code != 0


def test_cli_provider_choice_validation(fixture_paths):
    """Unknown provider value rejected by Click before any work runs."""
    dossier, char, output = fixture_paths
    runner = CliRunner()
    result = runner.invoke(
        cli,
        [
            "run",
            "--dossier", str(dossier),
            "--characterization", str(char),
            "--output", str(output),
            "--provider", "openai",
        ],
    )
    assert result.exit_code != 0
