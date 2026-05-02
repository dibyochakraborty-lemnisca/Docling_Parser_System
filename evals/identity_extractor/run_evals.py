"""Eval runner for the identity extractor.

Each fixture under this directory carries:
  - narrative_blocks.json      : input narrative blocks
  - present_variables.json     : golden columns present in the dossier
  - scripted_llm_response.json : the LLM response to inject (no live call)
  - expected_identity.json     : expected fields in the resulting ProcessIdentity

The runner replays each fixture against the real IdentityExtractor with a
scripted client and asserts the expected fields. No live LLM calls; the
eval set is deterministic and CI-safe.

Usage:
    parsevenv/bin/python evals/identity_extractor/run_evals.py
    parsevenv/bin/python -m pytest evals/identity_extractor/run_evals.py -q
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from fermdocs.domain.models import NarrativeBlock
from fermdocs.mapping.identity_extractor import IdentityExtractor
from fermdocs.mapping.process_registry import load_registry

FIXTURES_DIR = Path(__file__).parent
FIXTURE_NAMES = [
    "01_clear_penicillin",
    "02_off_whitelist",
    "03_fingerprint_mismatch",
    "04_unquoted_evidence",
    "05_empty_narrative",
]


class _Scripted:
    def __init__(self, response: dict[str, Any]) -> None:
        self.response = response
        self.calls = 0

    def call(self, system: str, user: str) -> dict[str, Any]:
        self.calls += 1
        return self.response


def _load_fixture(name: str) -> dict[str, Any]:
    folder = FIXTURES_DIR / name
    return {
        "narrative_blocks": [
            NarrativeBlock.model_validate(b)
            for b in json.loads((folder / "narrative_blocks.json").read_text())
        ],
        "present_variables": set(
            json.loads((folder / "present_variables.json").read_text())
        ),
        "scripted_response": json.loads(
            (folder / "scripted_llm_response.json").read_text()
        ),
        "expected": json.loads((folder / "expected_identity.json").read_text()),
    }


def _check_layer(actual: Any, expected: dict[str, Any], layer_name: str) -> None:
    """Assert each expected key matches the corresponding field on the layer.

    Special expected keys:
      - confidence_max: actual.confidence <= value
      - rationale_contains: substring check (case-insensitive)
      - note: ignored (human-readable annotation)
    """
    for key, want in expected.items():
        if key in {"note", "llm_must_not_be_called"}:
            continue
        if key == "confidence_max":
            assert actual.confidence <= want, (
                f"[{layer_name}] confidence {actual.confidence} > max {want}"
            )
            continue
        if key == "rationale_contains":
            rat = (getattr(actual, "rationale", "") or "").lower()
            assert want.lower() in rat, (
                f"[{layer_name}] rationale {rat!r} missing {want!r}"
            )
            continue
        got = getattr(actual, key, None)
        # Pydantic enums vs string values
        if hasattr(got, "value"):
            got = got.value
        assert got == want, (
            f"[{layer_name}] {key}: got {got!r}, want {want!r}"
        )


@pytest.mark.parametrize("fixture_name", FIXTURE_NAMES)
def test_identity_eval(fixture_name: str) -> None:
    fixture = _load_fixture(fixture_name)
    client = _Scripted(fixture["scripted_response"])
    extractor = IdentityExtractor(load_registry(), client)
    result = extractor.extract(
        fixture["narrative_blocks"], fixture["present_variables"]
    )

    expected = fixture["expected"]
    if "observed" in expected:
        _check_layer(result.observed, expected["observed"], "observed")
    if "registered" in expected:
        _check_layer(result.registered, expected["registered"], "registered")
    if expected.get("llm_must_not_be_called"):
        assert client.calls == 0, (
            f"{fixture_name}: LLM was called {client.calls} time(s); "
            "fixture asserts the extractor must short-circuit."
        )


def _main() -> None:
    """Run all fixtures from the command line, printing a summary table."""
    registry = load_registry()
    print(f"{'fixture':<32} {'observed':<24} {'registered':<24}")
    print("-" * 80)
    for name in FIXTURE_NAMES:
        fixture = _load_fixture(name)
        client = _Scripted(fixture["scripted_response"])
        result = IdentityExtractor(registry, client).extract(
            fixture["narrative_blocks"], fixture["present_variables"]
        )
        obs_summary = (
            f"{result.observed.provenance.value}"
            f"/{result.observed.organism or '-'}"
        )
        reg_summary = (
            f"{result.registered.provenance.value}"
            f"/{result.registered.process_id or '-'}"
        )
        print(f"{name:<32} {obs_summary:<24} {reg_summary:<24}")


if __name__ == "__main__":
    _main()
