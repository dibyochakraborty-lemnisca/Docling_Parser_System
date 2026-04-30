"""Validator catches findings/deviations/trajectories that cite unknown
observation IDs in the source dossier(s).
"""

import copy
import json
from pathlib import Path

from fermdocs_characterize.pipeline import CharacterizationPipeline
from fermdocs_characterize.validators.output_validator import validate_output

FIXTURES_DIR = Path(__file__).resolve().parents[2] / "evals" / "characterize" / "fixtures"


def test_validator_catches_unknown_obs_id():
    dossier = json.loads((FIXTURES_DIR / "01_boundary" / "dossier.json").read_text())
    expected = json.loads((FIXTURES_DIR / "01_boundary" / "expected_output.json").read_text())
    pipeline = CharacterizationPipeline()
    output = pipeline.run(
        dossier,
        characterization_id=__import__("uuid").UUID(expected["meta"]["characterization_id"]),
        generation_timestamp=__import__("datetime").datetime.fromisoformat(
            expected["meta"]["generation_timestamp"]
        ),
    )

    # Construct a tampered dossier that no longer contains the obs IDs the
    # output cites. Validator must surface every dangling reference.
    tampered_dossier = copy.deepcopy(dossier)
    tampered_dossier["golden_columns"] = {}

    errors = validate_output(
        output,
        dossiers={dossier["experiment"]["experiment_id"]: tampered_dossier},
    )
    assert len(errors) > 0
    assert any("unknown observation_id" in e for e in errors)


def test_validator_passes_with_intact_dossier():
    dossier = json.loads((FIXTURES_DIR / "01_boundary" / "dossier.json").read_text())
    expected = json.loads((FIXTURES_DIR / "01_boundary" / "expected_output.json").read_text())
    pipeline = CharacterizationPipeline()
    output = pipeline.run(
        dossier,
        characterization_id=__import__("uuid").UUID(expected["meta"]["characterization_id"]),
        generation_timestamp=__import__("datetime").datetime.fromisoformat(
            expected["meta"]["generation_timestamp"]
        ),
    )
    errors = validate_output(
        output,
        dossiers={dossier["experiment"]["experiment_id"]: dossier},
    )
    assert errors == []
