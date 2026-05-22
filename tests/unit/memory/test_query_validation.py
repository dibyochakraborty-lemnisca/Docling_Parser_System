"""D7: MemoryQuery(kind='lesson', process_family=None) raises.

Plan ref: plans/2026-05-10-memory-layer.md commit 1, decision D7.

Cross-family lesson retrieval is structurally impossible without an
explicit opt-in. The check lives in validate_query() (called by every
backend's fetch) so even NoopBackend surfaces the error — typo at the
caller doesn't get a free pass just because the backend is off.
"""

from __future__ import annotations

import pytest

from fermdocs_memory.base import MemoryQuery, validate_query


def test_lesson_with_process_family_passes() -> None:
    q = MemoryQuery(
        tenant_id="default",
        kind="lesson",
        process_family="yeast_intracellular_product_fedbatch",
    )
    validate_query(q)  # no raise


def test_lesson_without_process_family_raises() -> None:
    q = MemoryQuery(
        tenant_id="default",
        kind="lesson",
        process_family=None,
    )
    with pytest.raises(ValueError, match="process_family"):
        validate_query(q)


def test_non_lesson_kind_without_process_family_passes() -> None:
    """The D7 invariant is specific to lessons. Other kinds (Tier 2/3/5)
    have their own retrieval shapes and may legitimately query without
    process_family."""
    for kind in ("ratified_hypothesis", "rejected_hypothesis", "correction"):
        q = MemoryQuery(tenant_id="default", kind=kind, process_family=None)
        validate_query(q)


def test_kind_none_without_process_family_passes() -> None:
    """kind=None means 'any kind' — primarily for the admin endpoint
    that lists everything. Not the lesson-retrieval path."""
    q = MemoryQuery(tenant_id="default", kind=None, process_family=None)
    validate_query(q)


def test_validation_error_message_names_d7_and_alternative() -> None:
    """The error has to be actionable: name the rule and tell the
    caller what to do instead."""
    q = MemoryQuery(tenant_id="default", kind="lesson", process_family=None)
    with pytest.raises(ValueError) as exc_info:
        validate_query(q)
    msg = str(exc_info.value)
    assert "process_family" in msg
    assert "D7" in msg
