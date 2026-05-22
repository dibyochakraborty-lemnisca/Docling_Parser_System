"""MemoryRecord shape, frozenness, provenance immutability.

Plan ref: plans/2026-05-10-memory-layer.md commit 1.
"""

from __future__ import annotations

from types import MappingProxyType

import pytest

from fermdocs_memory.base import MemoryRecord


def _record(**overrides) -> MemoryRecord:
    base = dict(
        memory_id="L-RUN0001-0001",
        kind="lesson",
        summary="placeholder lesson",
        process_family="yeast_intracellular_product_fedbatch",
        organism="S. cerevisiae",
        tenant_id="default",
    )
    base.update(overrides)
    return MemoryRecord(**base)


def test_minimal_record_constructs() -> None:
    r = _record()
    assert r.memory_id == "L-RUN0001-0001"
    assert r.kind == "lesson"
    assert r.affected_variables == ()
    assert r.finding_classes == ()
    assert r.tags == ()
    assert r.confidence is None
    assert r.superseded_by is None


def test_record_is_frozen_attribute_assignment_fails() -> None:
    """frozen dataclass: can't reassign attrs after construction."""
    r = _record()
    with pytest.raises((AttributeError, TypeError)):
        r.memory_id = "L-OTHER"  # type: ignore[misc]


def test_provenance_is_mapping_proxy_after_construction() -> None:
    """Even if caller passes a dict, we wrap it in MappingProxyType."""
    r = _record(provenance={"run_id": "RUN-0001", "hyp_id": "H-0001"})
    assert isinstance(r.provenance, MappingProxyType)
    assert r.provenance["run_id"] == "RUN-0001"


def test_provenance_cannot_be_mutated_after_construction() -> None:
    r = _record(provenance={"run_id": "RUN-0001"})
    with pytest.raises(TypeError):
        r.provenance["run_id"] = "RUN-9999"  # type: ignore[index]


def test_provenance_caller_dict_mutation_does_not_leak() -> None:
    """If the caller mutates their original dict after construction,
    the record's provenance must NOT change. This is the audit
    invariant frozen=True alone doesn't give us."""
    original = {"run_id": "RUN-0001"}
    r = _record(provenance=original)
    original["run_id"] = "RUN-9999"
    assert r.provenance["run_id"] == "RUN-0001"


def test_provenance_default_is_empty_mapping() -> None:
    r = _record()
    assert dict(r.provenance) == {}


def test_kind_is_closed_vocab() -> None:
    """Type checker enforces this; runtime smoke test that Python
    accepts the four documented values."""
    for kind in ("lesson", "ratified_hypothesis", "rejected_hypothesis", "correction"):
        r = _record(kind=kind)
        assert r.kind == kind
