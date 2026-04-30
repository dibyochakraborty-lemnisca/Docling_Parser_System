"""Validator flags outputs whose schema_version is not current."""

import json
from datetime import datetime
from pathlib import Path
from uuid import UUID

from fermdocs_characterize.pipeline import CharacterizationPipeline
from fermdocs_characterize.validators.output_validator import validate_output

FIXTURES_DIR = Path(__file__).resolve().parents[2] / "evals" / "characterize" / "fixtures"


def test_old_schema_version_rejected():
    dossier = json.loads((FIXTURES_DIR / "01_boundary" / "dossier.json").read_text())
    expected = json.loads((FIXTURES_DIR / "01_boundary" / "expected_output.json").read_text())
    pipeline = CharacterizationPipeline(
        current_schema_version="0.5",
        validate=False,  # let it produce "1.0" then we check with current="0.5"
    )
    # Force the meta to use schema_version="0.5" by passing through a custom path
    # Easier: produce the standard output, then validate against current="2.0".
    pipeline2 = CharacterizationPipeline()
    output = pipeline2.run(
        dossier,
        characterization_id=UUID(expected["meta"]["characterization_id"]),
        generation_timestamp=datetime.fromisoformat(expected["meta"]["generation_timestamp"]),
    )
    errors = validate_output(output, current_schema_version="2.0")
    assert any("schema_version" in e for e in errors)


def test_old_priors_version_rejected():
    dossier = json.loads((FIXTURES_DIR / "01_boundary" / "dossier.json").read_text())
    expected = json.loads((FIXTURES_DIR / "01_boundary" / "expected_output.json").read_text())
    pipeline = CharacterizationPipeline()
    output = pipeline.run(
        dossier,
        characterization_id=UUID(expected["meta"]["characterization_id"]),
        generation_timestamp=datetime.fromisoformat(expected["meta"]["generation_timestamp"]),
    )
    # Output has priors_version=None (v1). When validator demands a current priors
    # version, None is acceptable (means "no priors used"). Test the active branch:
    # pretend the output had a priors_version set, that doesn't match current.
    output_with_priors = output.model_copy(
        update={
            "meta": output.meta.model_copy(update={"process_priors_version": "old-1.0"})
        }
    )
    errors = validate_output(
        output_with_priors,
        current_process_priors_version="new-2.0",
    )
    assert any("process_priors_version" in e for e in errors)
