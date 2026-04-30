"""Integration test: pipeline output matches expected_output.json byte-stable.

For each fixture, run the pipeline with the fixture's pinned characterization_id
and generation_timestamp, dump to JSON, and compare to the expected output.
"""

import json

from fermdocs_characterize.pipeline import CharacterizationPipeline


def test_fixture_matches_expected(fixture_case):
    pipeline = CharacterizationPipeline()
    actual = pipeline.run(
        fixture_case["dossier"],
        characterization_id=fixture_case["characterization_id"],
        generation_timestamp=fixture_case["generation_timestamp"],
    )
    actual_json = json.loads(actual.model_dump_json())
    assert actual_json == fixture_case["expected"], (
        f"Fixture {fixture_case['name']} did not match expected output."
    )
