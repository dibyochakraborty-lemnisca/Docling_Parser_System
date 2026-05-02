"""Tests for process_registry: load, validation, fingerprint check."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from fermdocs.mapping.process_registry import (
    fingerprint_check,
    load_registry,
    parse_registry_dict,
)


def test_production_registry_loads_against_golden_schema():
    """The shipped registry must validate against the production schema."""
    registry = load_registry()
    assert registry.version == "1.0"
    assert "penicillin_indpensim" in registry.by_id()


def test_duplicate_process_id_rejected():
    data = {
        "version": "1.0",
        "processes": [
            {
                "id": "dup",
                "organism": "X",
                "product": "P",
                "process_family": "fed-batch",
                "variable_fingerprint": {"required": [], "strong": [], "forbidden": []},
            },
            {
                "id": "dup",
                "organism": "Y",
                "product": "Q",
                "process_family": "batch",
                "variable_fingerprint": {"required": [], "strong": [], "forbidden": []},
            },
        ],
    }
    with pytest.raises(ValidationError):
        parse_registry_dict(data)


def test_alias_collision_rejected():
    data = {
        "version": "1.0",
        "processes": [
            {
                "id": "a",
                "organism": "Org A",
                "product": "Product A",
                "process_family": "fed-batch",
                "aliases": {"organism": ["shared-alias"]},
                "variable_fingerprint": {"required": [], "strong": [], "forbidden": []},
            },
            {
                "id": "b",
                "organism": "Org B",
                "product": "Product B",
                "process_family": "batch",
                "aliases": {"organism": ["shared-alias"]},
                "variable_fingerprint": {"required": [], "strong": [], "forbidden": []},
            },
        ],
    }
    with pytest.raises(ValidationError) as excinfo:
        parse_registry_dict(data)
    assert "shared-alias" in str(excinfo.value)


def test_fingerprint_unknown_variable_rejected_when_validated_against_schema(
    tmp_path,
):
    """The loader's variable check only runs when validate_against_schema=True."""
    registry_path = tmp_path / "bad.yaml"
    registry_path.write_text(
        "version: '1.0'\n"
        "processes:\n"
        "  - id: bad\n"
        "    organism: X\n"
        "    product: P\n"
        "    process_family: fed-batch\n"
        "    variable_fingerprint:\n"
        "      required: [does_not_exist_in_golden_schema]\n"
        "      strong: []\n"
        "      forbidden: []\n"
    )
    with pytest.raises(ValueError) as excinfo:
        load_registry(registry_path)
    assert "does_not_exist_in_golden_schema" in str(excinfo.value)


def test_fingerprint_check_passes_when_required_present():
    registry = load_registry()
    entry = registry.by_id()["penicillin_indpensim"]
    ok, reason = fingerprint_check(entry, {"paa_mg_l", "nh3_mg_l", "biomass_g_l"})
    assert ok, reason


def test_fingerprint_check_fails_on_missing_required():
    registry = load_registry()
    entry = registry.by_id()["penicillin_indpensim"]
    ok, reason = fingerprint_check(entry, {"biomass_g_l"})
    assert not ok
    assert "paa_mg_l" in reason
    assert "nh3_mg_l" in reason


def test_fingerprint_check_strong_vars_advisory_only():
    """Strong variables are reinforcing, not required: their absence does not reject."""
    registry = load_registry()
    entry = registry.by_id()["penicillin_indpensim"]
    # required vars present, strong vars absent
    ok, reason = fingerprint_check(entry, {"paa_mg_l", "nh3_mg_l"})
    assert ok, reason


def test_fingerprint_check_forbidden_present_rejects():
    data = {
        "version": "1.0",
        "processes": [
            {
                "id": "test",
                "organism": "X",
                "product": "P",
                "process_family": "fed-batch",
                "variable_fingerprint": {
                    "required": [],
                    "strong": [],
                    "forbidden": ["lactate_g_l"],
                },
            }
        ],
    }
    registry = parse_registry_dict(data)
    entry = registry.by_id()["test"]
    ok, reason = fingerprint_check(entry, {"lactate_g_l"})
    assert not ok
    assert "lactate_g_l" in reason
