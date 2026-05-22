"""SynapBackend unit tests with the SDK fully mocked.

Plan ref: plans/2026-05-10-memory-layer.md commit 2.

These tests verify the adapter's mapping logic without making any
network calls. The SDK itself is replaced with an in-memory mock so
we can assert: which fields get sent on write, how the response
flattens into MemoryRecords, how filters apply client-side, and
how failures are absorbed.

Live integration tests against the dev instance live in
tests/integration/memory/test_synap_backend_live.py and are skipped
when SYNAP_API_KEY is not set.
"""

from __future__ import annotations

import sys
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest


# ----------------------------------------------------------------------
# Fake SDK module: install before any import of fermdocs_memory.synap
# so that module's _get_sdk_module() resolves to our fake.
# ----------------------------------------------------------------------


class _FakeDocumentType:
    DOCUMENT = "document"
    CONVERSATION = "ai-chat-conversation"


class _FakeIngestMode:
    LONG_RANGE = "long-range"
    FAST = "fast"


class _FakeIngestStatus:
    QUEUED = "queued"
    COMPLETED = "completed"


class _FakeMemoriesInterface:
    def __init__(self) -> None:
        self.creates: list[dict] = []

    async def create(self, *, document, document_type, document_id,
                     user_id, customer_id, mode, metadata):
        self.creates.append(dict(
            document=document, document_type=document_type,
            document_id=document_id, user_id=user_id,
            customer_id=customer_id, mode=mode, metadata=metadata,
        ))
        return SimpleNamespace(
            ingestion_id="11111111-2222-3333-4444-555555555555",
            document_id=document_id,
            status=_FakeIngestStatus.QUEUED,
        )


class _FakeUserContext:
    def __init__(self) -> None:
        self.fetch_calls: list[dict] = []
        self._next_response: SimpleNamespace | None = None
        self._next_exception: Exception | None = None

    def queue_response(self, response: SimpleNamespace) -> None:
        self._next_response = response

    def queue_exception(self, exc: Exception) -> None:
        self._next_exception = exc

    async def fetch(self, *, user_id, search_query=None, max_results=10):
        self.fetch_calls.append(dict(
            user_id=user_id, search_query=search_query, max_results=max_results,
        ))
        if self._next_exception is not None:
            exc = self._next_exception
            self._next_exception = None
            raise exc
        if self._next_response is not None:
            r = self._next_response
            self._next_response = None
            return r
        return SimpleNamespace(
            facts=[], preferences=[], episodes=[],
            emotions=[], temporal_events=[],
        )


class _FakeUserInterface:
    def __init__(self) -> None:
        self.context = _FakeUserContext()


class _FakeSDK:
    instance_id_counter = 0

    def __init__(self, instance_id="", api_key=None):
        _FakeSDK.instance_id_counter += 1
        self.instance_id = instance_id or f"fake-{_FakeSDK.instance_id_counter}"
        self.memories = _FakeMemoriesInterface()
        self.user = _FakeUserInterface()
        self._initialized = False

    async def initialize(self):
        self._initialized = True

    async def shutdown(self):
        self._initialized = False


def _install_fake_sdk(monkeypatch):
    """Install a fake `maximem_synap` module before SynapBackend imports it."""
    fake = SimpleNamespace(
        MaximemSynapSDK=_FakeSDK,
        DocumentType=_FakeDocumentType,
        IngestMode=_FakeIngestMode,
        IngestStatus=_FakeIngestStatus,
    )
    monkeypatch.setitem(sys.modules, "maximem_synap", fake)
    # Reset the lazy-loaded module reference inside fermdocs_memory.synap
    import fermdocs_memory.synap as backend_mod
    monkeypatch.setattr(backend_mod, "_SDK_MODULE", None)
    return fake


# ----------------------------------------------------------------------
# Tests
# ----------------------------------------------------------------------


@pytest.fixture
def fake_sdk(monkeypatch):
    monkeypatch.setenv("SYNAP_API_KEY", "fake-key-for-tests")
    return _install_fake_sdk(monkeypatch)


def _make_record(**overrides):
    from fermdocs_memory.base import MemoryRecord
    base = dict(
        memory_id="L-RUN0001-0001",
        kind="lesson",
        summary="A digest about pigment loss in yeast carotenoid runs.",
        process_family="yeast_intracellular_product_fedbatch",
        organism="S. cerevisiae",
        tenant_id="lemnisca-internal",
        tags=("pigment-loss", "tool-gap-axis"),
        provenance={"run_id": "RUN-0001", "hyp_id": "H-0003"},
    )
    base.update(overrides)
    return MemoryRecord(**base)


# ---- write ----


def test_write_sends_expected_payload_to_sdk(fake_sdk):
    from fermdocs_memory.synap import SynapBackend
    b = SynapBackend()
    rec = _make_record()
    b.write(rec)
    creates = b._sdk.memories.creates
    assert len(creates) == 1
    call = creates[0]
    assert call["document"] == rec.summary
    assert call["document_id"] == rec.memory_id
    assert call["user_id"] == rec.process_family
    assert call["customer_id"] == rec.tenant_id
    assert call["document_type"] == _FakeDocumentType.DOCUMENT
    assert call["mode"] == _FakeIngestMode.LONG_RANGE
    md = call["metadata"]
    assert md["fermdocs_kind"] == "lesson"
    assert md["lesson_id"] == rec.memory_id
    assert md["organism"] == "S. cerevisiae"
    assert md["tags"] == ["pigment-loss", "tool-gap-axis"]
    assert md["provenance"] == {"run_id": "RUN-0001", "hyp_id": "H-0003"}
    b.shutdown()


def test_write_lesson_without_process_family_raises(fake_sdk):
    from fermdocs_memory.synap import SynapBackend
    from fermdocs_memory.base import MemoryRecord
    b = SynapBackend()
    rec = MemoryRecord(
        memory_id="L-X",
        kind="lesson",
        summary="x",
        process_family=None,
        organism=None,
        tenant_id="default",
    )
    with pytest.raises(ValueError, match="process_family"):
        b.write(rec)
    b.shutdown()


def test_write_swallows_sdk_exception_and_logs(fake_sdk, monkeypatch, caplog):
    """SDK outage during write is logged + counter, not re-raised. Memory is opt-in."""
    from fermdocs_memory.synap import SynapBackend
    b = SynapBackend()
    b._ensure_initialized()  # so b._sdk is real before we patch it
    async def boom(*a, **kw):
        raise RuntimeError("simulated outage")
    monkeypatch.setattr(b._sdk.memories, "create", boom)
    # No raise:
    b.write(_make_record())
    b.shutdown()


# ---- fetch ----


def test_fetch_with_no_results_returns_empty_list(fake_sdk):
    from fermdocs_memory.synap import SynapBackend
    from fermdocs_memory.base import MemoryQuery
    b = SynapBackend()
    out = b.fetch(MemoryQuery(
        tenant_id="lemnisca-internal",
        kind="lesson",
        process_family="yeast_intracellular_product_fedbatch",
    ))
    assert out == []
    b.shutdown()


def test_fetch_passes_query_params_to_sdk(fake_sdk):
    from fermdocs_memory.synap import SynapBackend
    from fermdocs_memory.base import MemoryQuery
    b = SynapBackend()
    b.fetch(MemoryQuery(
        tenant_id="lemnisca-internal",
        kind="lesson",
        process_family="yeast_intracellular_product_fedbatch",
        semantic_query="pigment loss",
        top_k=3,
    ))
    calls = b._sdk.user.context.fetch_calls
    assert len(calls) == 1
    assert calls[0]["user_id"] == "yeast_intracellular_product_fedbatch"
    assert calls[0]["search_query"] == ["pigment loss"]
    assert calls[0]["max_results"] == 3
    b.shutdown()


def test_fetch_flattens_facts_and_episodes_into_memory_records(fake_sdk):
    from fermdocs_memory.synap import SynapBackend
    from fermdocs_memory.base import MemoryQuery
    b = SynapBackend()
    # Queue a response with one fact + one episode + one temporal event.
    response = SimpleNamespace(
        facts=[
            SimpleNamespace(
                id="fact-1",
                content="Pigment loss after 144h is documented.",
                confidence=0.92,
                metadata={
                    "lesson_id": "L-RUN0001-0001",
                    "process_family": "yeast_intracellular_product_fedbatch",
                    "organism": "S. cerevisiae",
                    "tags": ["pigment-loss"],
                    "provenance": {"run_id": "RUN-0001"},
                },
                extracted_at="2026-05-10T12:00:00Z",
            ),
        ],
        preferences=[],
        episodes=[
            SimpleNamespace(
                id="episode-1",
                content="In a prior run, white cells appeared post-144h.",
                confidence=0.85,
                metadata={
                    "lesson_id": "L-RUN0001-0001",
                    "process_family": "yeast_intracellular_product_fedbatch",
                },
                extracted_at="2026-05-10T12:00:01Z",
            ),
        ],
        emotions=[],
        temporal_events=[
            SimpleNamespace(
                id="temporal-1",
                content="Event at 144h: pigment loss onset.",
                metadata={
                    "process_family": "yeast_intracellular_product_fedbatch",
                },
            ),
        ],
    )
    b._ensure_initialized()
    b._sdk.user.context.queue_response(response)
    out = b.fetch(MemoryQuery(
        tenant_id="lemnisca-internal",
        kind="lesson",
        process_family="yeast_intracellular_product_fedbatch",
    ))
    assert len(out) == 3
    # All carry our expected lifted metadata + sentinel embedding fields.
    for r in out:
        assert r.kind == "lesson"
        assert r.tenant_id == "lemnisca-internal"
        assert r.process_family == "yeast_intracellular_product_fedbatch"
        assert r.embedding_provider == "synap-managed"
    fact = [r for r in out if r.memory_id == "fact-1"][0]
    assert fact.confidence == 0.92
    assert fact.organism == "S. cerevisiae"
    assert fact.provenance["synap_extraction_kind"] == "facts"
    assert fact.provenance["source_document_id"] == "L-RUN0001-0001"
    b.shutdown()


def test_fetch_applies_client_side_organism_filter(fake_sdk):
    """Synap doesn't filter on metadata; we do client-side."""
    from fermdocs_memory.synap import SynapBackend
    from fermdocs_memory.base import MemoryQuery
    b = SynapBackend()
    b._ensure_initialized()
    b._sdk.user.context.queue_response(SimpleNamespace(
        facts=[
            SimpleNamespace(id="f-cer", content="...", metadata={
                "process_family": "yeast_intracellular_product_fedbatch",
                "organism": "S. cerevisiae",
            }),
            SimpleNamespace(id="f-other", content="...", metadata={
                "process_family": "yeast_intracellular_product_fedbatch",
                "organism": "Y. lipolytica",
            }),
        ],
        preferences=[], episodes=[], emotions=[], temporal_events=[],
    ))
    out = b.fetch(MemoryQuery(
        tenant_id="lemnisca-internal",
        kind="lesson",
        process_family="yeast_intracellular_product_fedbatch",
        organism="S. cerevisiae",
    ))
    assert [r.memory_id for r in out] == ["f-cer"]
    b.shutdown()


def test_fetch_applies_finding_classes_overlap_filter(fake_sdk):
    from fermdocs_memory.synap import SynapBackend
    from fermdocs_memory.base import MemoryQuery
    b = SynapBackend()
    b._ensure_initialized()
    b._sdk.user.context.queue_response(SimpleNamespace(
        facts=[
            SimpleNamespace(id="f-b10", content="...", metadata={
                "process_family": "yeast_intracellular_product_fedbatch",
                "finding_classes": ["B10_overflow"],
            }),
            SimpleNamespace(id="f-a14", content="...", metadata={
                "process_family": "yeast_intracellular_product_fedbatch",
                "finding_classes": ["A14_do_margin"],
            }),
        ],
        preferences=[], episodes=[], emotions=[], temporal_events=[],
    ))
    out = b.fetch(MemoryQuery(
        tenant_id="lemnisca-internal",
        kind="lesson",
        process_family="yeast_intracellular_product_fedbatch",
        finding_classes_overlap=("B10_overflow",),
    ))
    assert [r.memory_id for r in out] == ["f-b10"]
    b.shutdown()


def test_fetch_returns_empty_on_sdk_exception(fake_sdk):
    """ServiceUnavailable etc. -> empty list, run continues without priors."""
    from fermdocs_memory.synap import SynapBackend
    from fermdocs_memory.base import MemoryQuery
    b = SynapBackend()
    b._ensure_initialized()
    b._sdk.user.context.queue_exception(RuntimeError("ServiceUnavailable"))
    out = b.fetch(MemoryQuery(
        tenant_id="lemnisca-internal",
        kind="lesson",
        process_family="yeast_intracellular_product_fedbatch",
    ))
    assert out == []
    b.shutdown()


def test_fetch_enforces_d7_via_validate_query(fake_sdk):
    from fermdocs_memory.synap import SynapBackend
    from fermdocs_memory.base import MemoryQuery
    b = SynapBackend()
    with pytest.raises(ValueError):
        b.fetch(MemoryQuery(
            tenant_id="lemnisca-internal",
            kind="lesson",
            process_family=None,
        ))
    b.shutdown()


def test_fetch_top_k_caps_results(fake_sdk):
    from fermdocs_memory.synap import SynapBackend
    from fermdocs_memory.base import MemoryQuery
    b = SynapBackend()
    b._ensure_initialized()
    b._sdk.user.context.queue_response(SimpleNamespace(
        facts=[
            SimpleNamespace(id=f"f-{i}", content="x", metadata={
                "process_family": "yeast_intracellular_product_fedbatch",
            })
            for i in range(10)
        ],
        preferences=[], episodes=[], emotions=[], temporal_events=[],
    ))
    out = b.fetch(MemoryQuery(
        tenant_id="lemnisca-internal",
        kind="lesson",
        process_family="yeast_intracellular_product_fedbatch",
        top_k=3,
    ))
    assert len(out) == 3
    b.shutdown()


# ---- supersede + lifecycle ----


def test_supersede_is_no_op(fake_sdk):
    from fermdocs_memory.synap import SynapBackend
    b = SynapBackend()
    # Doesn't initialize SDK or raise; logged-only no-op.
    b.supersede("memory-1", "memory-2")
    b.shutdown()


def test_construction_does_not_initialize_sdk(fake_sdk):
    from fermdocs_memory.synap import SynapBackend
    b = SynapBackend()
    assert not b._initialized
    assert b._sdk is None


def test_init_lazy_on_first_call(fake_sdk):
    from fermdocs_memory.synap import SynapBackend
    b = SynapBackend()
    b._ensure_initialized()
    assert b._initialized
    assert b._sdk is not None
    b.shutdown()


def test_missing_api_key_raises_on_init(monkeypatch):
    monkeypatch.delenv("SYNAP_API_KEY", raising=False)
    _install_fake_sdk(monkeypatch)
    from fermdocs_memory.synap import SynapBackend
    b = SynapBackend()
    with pytest.raises(RuntimeError, match="SYNAP_API_KEY"):
        b._ensure_initialized()
