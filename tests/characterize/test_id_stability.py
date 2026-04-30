"""Re-running on the same input with the same characterization_id and
generation_timestamp produces identical IDs across all collections.

Protects the contract: stable IDs require sort-stable aggregation everywhere.
If any aggregation step grows non-determinism, this test catches it.
"""

import json

from fermdocs_characterize.pipeline import CharacterizationPipeline


def test_ids_stable_across_runs(fixture_case):
    pipeline = CharacterizationPipeline()
    out1 = pipeline.run(
        fixture_case["dossier"],
        characterization_id=fixture_case["characterization_id"],
        generation_timestamp=fixture_case["generation_timestamp"],
    )
    out2 = pipeline.run(
        fixture_case["dossier"],
        characterization_id=fixture_case["characterization_id"],
        generation_timestamp=fixture_case["generation_timestamp"],
    )
    j1 = json.loads(out1.model_dump_json())
    j2 = json.loads(out2.model_dump_json())
    assert j1 == j2


def test_ids_stable_with_dossier_observation_reorder(fixture_case):
    """Reordering observations within a column should not affect IDs.

    The pipeline must sort observations canonically before assigning IDs.
    """
    import copy

    pipeline = CharacterizationPipeline()
    out_original = pipeline.run(
        fixture_case["dossier"],
        characterization_id=fixture_case["characterization_id"],
        generation_timestamp=fixture_case["generation_timestamp"],
    )

    shuffled = copy.deepcopy(fixture_case["dossier"])
    for col_data in shuffled.get("golden_columns", {}).values():
        if isinstance(col_data, dict):
            col_data["observations"] = list(reversed(col_data.get("observations", [])))

    out_shuffled = pipeline.run(
        shuffled,
        characterization_id=fixture_case["characterization_id"],
        generation_timestamp=fixture_case["generation_timestamp"],
    )
    j1 = json.loads(out_original.model_dump_json())
    j2 = json.loads(out_shuffled.model_dump_json())
    assert j1 == j2
