"""NoopBackend: write/supersede no-op; fetch returns []; D7 still enforced.

Plan ref: plans/2026-05-10-memory-layer.md commit 1.
"""

from __future__ import annotations

import pytest

from fermdocs_memory.base import MemoryQuery, MemoryRecord
from fermdocs_memory.noop import NoopBackend, noop_default


def _record(**overrides) -> MemoryRecord:
    base = dict(
        memory_id="L-X-0001",
        kind="lesson",
        summary="x",
        process_family="yeast_intracellular_product_fedbatch",
        organism="yeast",
        tenant_id="default",
    )
    base.update(overrides)
    return MemoryRecord(**base)


def test_write_is_no_op_returns_none() -> None:
    b = NoopBackend()
    assert b.write(_record()) is None


def test_fetch_returns_empty_list_for_valid_query() -> None:
    b = NoopBackend()
    q = MemoryQuery(
        tenant_id="default",
        kind="lesson",
        process_family="yeast_intracellular_product_fedbatch",
    )
    assert b.fetch(q) == []


def test_fetch_still_validates_d7_even_when_backend_is_off() -> None:
    """The whole point of validating in NoopBackend.fetch: a typo at
    a caller (forgot process_family) must surface even when memory is
    disabled, so we don't silently 'pass' on dev and break in prod."""
    b = NoopBackend()
    q = MemoryQuery(tenant_id="default", kind="lesson", process_family=None)
    with pytest.raises(ValueError):
        b.fetch(q)


def test_supersede_is_no_op_returns_none() -> None:
    b = NoopBackend()
    assert b.supersede("L-X-0001", "L-X-0002") is None


def test_module_singleton_is_a_noop_backend() -> None:
    assert isinstance(noop_default, NoopBackend)


def test_noop_default_is_reusable_across_callers() -> None:
    """noop_default exists so callers can pass it as a default kwarg
    without each holding their own instance. Trivial but pinning it
    so a refactor doesn't accidentally make it a factory."""
    a = noop_default
    b = noop_default
    assert a is b
