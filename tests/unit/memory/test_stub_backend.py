"""StubBackend: write/fetch/supersede + filters + ranking + top_k.

Plan ref: plans/2026-05-10-memory-layer.md commit 1.

These tests pin the contract that production backends (Synap, future
Postgres) must also satisfy. When a backend ships, run the same
scenarios against it as integration tests.
"""

from __future__ import annotations

import pytest

from fermdocs_memory.base import MemoryQuery, MemoryRecord
from fermdocs_memory.stub import StubBackend


def _record(**overrides) -> MemoryRecord:
    base = dict(
        memory_id="L-X-0001",
        kind="lesson",
        summary="placeholder",
        process_family="yeast_intracellular_product_fedbatch",
        organism="S. cerevisiae",
        tenant_id="default",
    )
    base.update(overrides)
    return MemoryRecord(**base)


# ---------- write ----------


def test_write_then_fetch_roundtrip() -> None:
    b = StubBackend()
    rec = _record()
    b.write(rec)
    out = b.fetch(MemoryQuery(
        tenant_id="default",
        kind="lesson",
        process_family="yeast_intracellular_product_fedbatch",
    ))
    assert out == [rec]


def test_write_duplicate_id_with_same_content_is_idempotent() -> None:
    b = StubBackend()
    rec = _record()
    b.write(rec)
    b.write(rec)  # no raise, no duplication


def test_write_duplicate_id_with_different_content_raises() -> None:
    b = StubBackend()
    b.write(_record(summary="first"))
    with pytest.raises(ValueError, match="already exists"):
        b.write(_record(summary="second"))


# ---------- filtering ----------


def test_tenant_isolation() -> None:
    """Records under tenant=acme must not surface when querying tenant=globex."""
    b = StubBackend()
    b.write(_record(memory_id="L-A", tenant_id="acme"))
    b.write(_record(memory_id="L-G", tenant_id="globex"))
    out = b.fetch(MemoryQuery(
        tenant_id="acme",
        kind="lesson",
        process_family="yeast_intracellular_product_fedbatch",
    ))
    assert [r.memory_id for r in out] == ["L-A"]


def test_process_family_filter_excludes_other_families() -> None:
    b = StubBackend()
    b.write(_record(memory_id="L-Y", process_family="yeast_intracellular_product_fedbatch"))
    b.write(_record(memory_id="L-P", process_family="penicillin_fedbatch"))
    out = b.fetch(MemoryQuery(
        tenant_id="default",
        kind="lesson",
        process_family="yeast_intracellular_product_fedbatch",
    ))
    assert [r.memory_id for r in out] == ["L-Y"]


def test_organism_secondary_filter_when_specified() -> None:
    b = StubBackend()
    b.write(_record(memory_id="L-CER", organism="S. cerevisiae"))
    b.write(_record(memory_id="L-OTH", organism="Y. lipolytica"))
    out = b.fetch(MemoryQuery(
        tenant_id="default",
        kind="lesson",
        process_family="yeast_intracellular_product_fedbatch",
        organism="S. cerevisiae",
    ))
    assert [r.memory_id for r in out] == ["L-CER"]


def test_organism_none_returns_all_within_family() -> None:
    """organism=None means 'no filter on organism' — different from
    process_family=None on lessons (which raises)."""
    b = StubBackend()
    b.write(_record(memory_id="L-CER", organism="S. cerevisiae"))
    b.write(_record(memory_id="L-OTH", organism="Y. lipolytica"))
    out = b.fetch(MemoryQuery(
        tenant_id="default",
        kind="lesson",
        process_family="yeast_intracellular_product_fedbatch",
    ))
    assert {r.memory_id for r in out} == {"L-CER", "L-OTH"}


def test_kind_filter_excludes_other_kinds() -> None:
    b = StubBackend()
    b.write(_record(memory_id="L-LESSON", kind="lesson"))
    b.write(_record(
        memory_id="L-RATIFIED",
        kind="ratified_hypothesis",
        process_family=None,  # ratified can have no family
    ))
    out = b.fetch(MemoryQuery(
        tenant_id="default",
        kind="lesson",
        process_family="yeast_intracellular_product_fedbatch",
    ))
    assert [r.memory_id for r in out] == ["L-LESSON"]


def test_variables_overlap_filter() -> None:
    b = StubBackend()
    b.write(_record(memory_id="L-DO", affected_variables=("dissolved_o2_mg_l",)))
    b.write(_record(memory_id="L-BIO", affected_variables=("biomass_g_l",)))
    out = b.fetch(MemoryQuery(
        tenant_id="default",
        kind="lesson",
        process_family="yeast_intracellular_product_fedbatch",
        variables_overlap=("dissolved_o2_mg_l",),
    ))
    assert [r.memory_id for r in out] == ["L-DO"]


def test_finding_classes_overlap_filter() -> None:
    b = StubBackend()
    b.write(_record(memory_id="L-B10", finding_classes=("B10_overflow",)))
    b.write(_record(memory_id="L-A14", finding_classes=("A14_do_margin",)))
    out = b.fetch(MemoryQuery(
        tenant_id="default",
        kind="lesson",
        process_family="yeast_intracellular_product_fedbatch",
        finding_classes_overlap=("B10_overflow",),
    ))
    assert [r.memory_id for r in out] == ["L-B10"]


# ---------- ranking ----------


def test_semantic_query_substring_ranking() -> None:
    """Stub falls back to substring score when semantic_query is set."""
    b = StubBackend()
    b.write(_record(memory_id="L-1", summary="dissolved oxygen crash post-induction"))
    b.write(_record(memory_id="L-2", summary="biomass plateau at 144h"))
    b.write(_record(memory_id="L-3", summary="oxygen and biomass both decline"))
    out = b.fetch(MemoryQuery(
        tenant_id="default",
        kind="lesson",
        process_family="yeast_intracellular_product_fedbatch",
        semantic_query="oxygen biomass",
    ))
    # L-3 has both tokens; L-1 has 'oxygen'; L-2 has 'biomass'.
    assert out[0].memory_id == "L-3"


def test_no_semantic_query_orders_by_recency_desc() -> None:
    b = StubBackend()
    b.write(_record(memory_id="L-OLD", created_at="2026-01-01T00:00:00Z"))
    b.write(_record(memory_id="L-NEW", created_at="2026-05-01T00:00:00Z"))
    out = b.fetch(MemoryQuery(
        tenant_id="default",
        kind="lesson",
        process_family="yeast_intracellular_product_fedbatch",
    ))
    assert [r.memory_id for r in out] == ["L-NEW", "L-OLD"]


def test_top_k_caps_results() -> None:
    b = StubBackend()
    for i in range(10):
        b.write(_record(memory_id=f"L-{i:02d}"))
    out = b.fetch(MemoryQuery(
        tenant_id="default",
        kind="lesson",
        process_family="yeast_intracellular_product_fedbatch",
        top_k=3,
    ))
    assert len(out) == 3


def test_top_k_default_is_5() -> None:
    b = StubBackend()
    for i in range(10):
        b.write(_record(memory_id=f"L-{i:02d}"))
    out = b.fetch(MemoryQuery(
        tenant_id="default",
        kind="lesson",
        process_family="yeast_intracellular_product_fedbatch",
    ))
    assert len(out) == 5


# ---------- supersession ----------


def test_supersede_existing_record() -> None:
    b = StubBackend()
    b.write(_record(memory_id="L-OLD"))
    b.write(_record(memory_id="L-NEW"))
    b.supersede("L-OLD", by="L-NEW")
    out = b.fetch(MemoryQuery(
        tenant_id="default",
        kind="lesson",
        process_family="yeast_intracellular_product_fedbatch",
    ))
    # L-OLD is filtered out by default; L-NEW is the only result.
    assert [r.memory_id for r in out] == ["L-NEW"]


def test_supersede_then_include_superseded_returns_both() -> None:
    b = StubBackend()
    b.write(_record(memory_id="L-OLD"))
    b.write(_record(memory_id="L-NEW"))
    b.supersede("L-OLD", by="L-NEW")
    out = b.fetch(MemoryQuery(
        tenant_id="default",
        kind="lesson",
        process_family="yeast_intracellular_product_fedbatch",
        include_superseded=True,
    ))
    assert {r.memory_id for r in out} == {"L-OLD", "L-NEW"}


def test_supersede_missing_record_is_no_op() -> None:
    """Idempotency: superseding an absent memory_id doesn't raise."""
    b = StubBackend()
    b.supersede("L-NEVER-EXISTED", by="L-NEW")  # no raise


# ---------- D7 still enforced through Stub.fetch ----------


def test_stub_fetch_enforces_d7() -> None:
    b = StubBackend()
    with pytest.raises(ValueError):
        b.fetch(MemoryQuery(
            tenant_id="default",
            kind="lesson",
            process_family=None,
        ))
