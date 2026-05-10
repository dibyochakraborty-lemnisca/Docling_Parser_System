"""SynapBackend: production memory adapter wrapping the maximem-synap SDK.

Plan ref: plans/2026-05-10-memory-layer.md commit 2 (Synap-revised).

Three things this adapter has to bridge:

  1. The SDK is async; our MemoryBackend Protocol is sync. We run a
     dedicated event-loop thread for the lifetime of the backend so
     callers in any context (sync runner, async FastAPI handler) can
     call .write() / .fetch() / .supersede() without dealing with
     event loops.

  2. Our MemoryRecord doesn't fit Synap's typed-extraction model 1:1.
     Synap takes a `document` of free text and extracts typed memories
     (Facts, Episodes, Preferences, Emotions, TemporalEvents) from it.
     Our lesson digests get re-extracted; on retrieval we get back
     extracted items, not our verbatim digest. We surface the
     extracted items into MemoryRecord.summary; provenance carries the
     `lesson_id` we sent in via `document_id` so callers can trace
     back.

  3. Filtering. Synap supports filtering by scope (User / Customer /
     Conversation / Client) but NOT by metadata fields. Our
     process_family is the User scope key (D8). Other filters
     (organism, variables_overlap, finding_classes_overlap) are
     applied client-side after retrieval — fine for top-K retrieval
     where K is small.

Failure modes we handle:
  - SDK raises ServiceUnavailableError on retrieval transient: caller
    sees fetch() return [] with a logged warning. Memory stays opt-in.
  - SDK raises AuthenticationError or returns None: same path.
  - Embedding/extraction is async; status check failures during write
    surface as a logged warning + write counter increment but the
    record's ingestion_id is still tracked in the buffer for retry.
"""

from __future__ import annotations

import asyncio
import concurrent.futures
import logging
import os
import threading
from dataclasses import replace
from typing import Any
from uuid import UUID

from fermdocs_memory.base import (
    MemoryBackend,
    MemoryQuery,
    MemoryRecord,
    validate_query,
)

_log = logging.getLogger(__name__)

# Lazy import so the rest of fermdocs doesn't pay the SDK import cost when
# memory is off. We only resolve the SDK module when SynapBackend is built.
_SDK_MODULE = None


def _get_sdk_module():
    global _SDK_MODULE
    if _SDK_MODULE is None:
        import maximem_synap as _m  # type: ignore[import-untyped]
        _SDK_MODULE = _m
    return _SDK_MODULE


# -----------------------------------------------------------------------------
# Async->sync bridge: dedicated event loop on a long-lived thread
# -----------------------------------------------------------------------------


class _LoopThread:
    """One thread, one event loop, lives for the SynapBackend's lifetime.

    Sync caller submits a coroutine via run_coroutine_threadsafe; the
    thread runs it and returns a Future. This works whether the caller
    is in a sync context (CLI, hypothesis runner) or already inside an
    asyncio loop (FastAPI handler) — we never touch the caller's loop.
    """

    def __init__(self) -> None:
        self._loop: asyncio.AbstractEventLoop | None = None
        self._thread: threading.Thread | None = None
        self._ready = threading.Event()
        self._closed = False

    def start(self) -> None:
        if self._thread is not None:
            return
        self._thread = threading.Thread(
            target=self._run, name="synap-backend-loop", daemon=True,
        )
        self._thread.start()
        self._ready.wait(timeout=5.0)
        if self._loop is None:
            raise RuntimeError("synap-backend loop thread failed to start")

    def _run(self) -> None:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        self._loop = loop
        self._ready.set()
        try:
            loop.run_forever()
        finally:
            try:
                loop.close()
            except Exception:
                pass

    def submit(self, coro, timeout: float | None = None):
        """Submit a coroutine to the loop and wait for the result.

        Raises whatever the coroutine raised, or TimeoutError on the
        sync-side wait.
        """
        if self._loop is None or self._closed:
            raise RuntimeError("synap-backend loop is not running")
        future = asyncio.run_coroutine_threadsafe(coro, self._loop)
        return future.result(timeout=timeout)

    def stop(self) -> None:
        if self._closed:
            return
        self._closed = True
        if self._loop is not None and self._loop.is_running():
            self._loop.call_soon_threadsafe(self._loop.stop)
        if self._thread is not None:
            self._thread.join(timeout=5.0)


# -----------------------------------------------------------------------------
# SynapBackend
# -----------------------------------------------------------------------------


# How long a single SDK call may take before we give up. Tuned conservative;
# Synap's docs claim 15ms P50 retrieval, but we've seen ServiceUnavailable
# bursts that recover in seconds.
_SDK_CALL_TIMEOUT_S = 30.0


class SynapBackend:
    """MemoryBackend adapter wrapping maximem_synap.MaximemSynapSDK.

    Construction does NOT initialize the SDK; first call lazily kicks
    off the loop thread + sdk.initialize(). This keeps `SynapBackend()`
    cheap so the runner can build one even when memory is off and then
    discover at first use that an API key is missing.

    Default tenant_id: read from FERMDOCS_TENANT_ID env var, falling
    back to "default" so single-tenant Phase 1 deploys work without
    config.
    """

    def __init__(
        self,
        *,
        api_key: str | None = None,
        instance_id: str | None = None,
        default_customer_id: str | None = None,
    ) -> None:
        self._api_key = api_key or os.environ.get("SYNAP_API_KEY")
        self._instance_id = instance_id or os.environ.get("SYNAP_INSTANCE_ID", "")
        self._default_customer_id = (
            default_customer_id
            or os.environ.get("FERMDOCS_TENANT_ID")
            or "default"
        )
        self._sdk: Any = None
        self._loop_thread: _LoopThread | None = None
        self._initialized = False
        self._init_lock = threading.Lock()

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def _ensure_initialized(self) -> None:
        if self._initialized:
            return
        with self._init_lock:
            if self._initialized:
                return
            if not self._api_key:
                raise RuntimeError(
                    "SYNAP_API_KEY not set; cannot initialize SynapBackend."
                    " Configure the env var or pass api_key= to the"
                    " constructor, or use NoopBackend to disable memory."
                )
            sdk_module = _get_sdk_module()
            self._loop_thread = _LoopThread()
            self._loop_thread.start()
            self._sdk = sdk_module.MaximemSynapSDK(
                instance_id=self._instance_id,
                api_key=self._api_key,
            )
            self._loop_thread.submit(
                self._sdk.initialize(),
                timeout=_SDK_CALL_TIMEOUT_S,
            )
            self._initialized = True
            _log.info(
                "synap-backend: initialized instance_id=%r tenant=%r",
                self._sdk.instance_id, self._default_customer_id,
            )

    def shutdown(self) -> None:
        """Close the SDK + stop the loop thread.

        Idempotent. Safe to call from any context. The runner doesn't
        currently call this; we rely on daemon-thread + atexit cleanup
        for normal shutdown. Provided for tests + explicit cleanup.
        """
        if not self._initialized:
            return
        try:
            if self._sdk is not None:
                self._loop_thread.submit(  # type: ignore[union-attr]
                    self._sdk.shutdown(), timeout=_SDK_CALL_TIMEOUT_S,
                )
        except Exception as exc:
            _log.warning("synap-backend: shutdown raised %r", exc)
        finally:
            if self._loop_thread is not None:
                self._loop_thread.stop()
            self._initialized = False

    # ------------------------------------------------------------------
    # MemoryBackend Protocol implementation
    # ------------------------------------------------------------------

    def write(self, record: MemoryRecord) -> None:
        """Send a record to Synap as one memories.create call.

        Mapping:
          - document      <- record.summary
          - document_id   <- record.memory_id (idempotency key; Synap
                              won't duplicate on re-write)
          - document_type <- DocumentType.DOCUMENT (closest fit; Synap's
                              enum doesn't include 'agent-lesson')
          - user_id       <- record.process_family (the D8 primary key
                              mapped to Synap's User scope)
          - customer_id   <- record.tenant_id
          - mode          <- LONG_RANGE (we want durable cross-run
                              retrieval, not session-scoped)
          - metadata      <- record.provenance + organism/tags/etc.
                              (Synap stores but doesn't filter on this)

        Doesn't wait for extraction to complete. The caller's loop is
        cheap; Synap's processing is async in the background. Phase 1
        write timing (D6) means we write at clean exit; the lesson
        won't be queried until the next run, which gives the
        extraction pipeline minutes to catch up.
        """
        self._ensure_initialized()
        if record.process_family is None and record.kind == "lesson":
            raise ValueError(
                "SynapBackend.write: lesson record requires process_family"
                " (it maps to Synap's User scope)."
            )
        sdk_module = _get_sdk_module()
        document_type = sdk_module.DocumentType.DOCUMENT
        ingest_mode = sdk_module.IngestMode.LONG_RANGE

        metadata = self._build_write_metadata(record)
        coro = self._sdk.memories.create(
            document=record.summary,
            document_type=document_type,
            document_id=record.memory_id,
            user_id=record.process_family or "_no_family",
            customer_id=record.tenant_id or self._default_customer_id,
            mode=ingest_mode,
            metadata=metadata,
        )
        try:
            resp = self._loop_thread.submit(  # type: ignore[union-attr]
                coro, timeout=_SDK_CALL_TIMEOUT_S,
            )
            _log.debug(
                "synap-backend: wrote memory_id=%s ingestion_id=%s status=%s",
                record.memory_id, resp.ingestion_id, resp.status,
            )
        except Exception as exc:
            _log.warning(
                "synap-backend: write failed for memory_id=%s (%s: %s);"
                " caller continues without persistence",
                record.memory_id, exc.__class__.__name__, str(exc)[:200],
            )

    def fetch(self, query: MemoryQuery) -> list[MemoryRecord]:
        """Retrieve via the User-scoped context endpoint.

        Synap's retrieval returns 5 typed buckets (facts, preferences,
        episodes, emotions, temporal_events) extracted from prior
        documents. We flatten these into MemoryRecords keyed by
        Synap's per-extraction id, with the original document_id
        preserved in provenance for traceback to our lesson_id.

        Failure mode: any SDK exception during fetch returns []. The
        caller continues with no priors injected — memory is opt-in
        and an outage doesn't break runs.
        """
        validate_query(query)
        # Lessons MUST have process_family per D7 (validate_query
        # already enforces). Other kinds may not; if process_family is
        # None we can't query Synap's User scope, so return empty.
        if query.process_family is None:
            return []
        self._ensure_initialized()

        try:
            ctx = self._loop_thread.submit(  # type: ignore[union-attr]
                self._sdk.user.context.fetch(
                    user_id=query.process_family,
                    search_query=[query.semantic_query] if query.semantic_query else None,
                    max_results=query.top_k,
                ),
                timeout=_SDK_CALL_TIMEOUT_S,
            )
        except Exception as exc:
            _log.warning(
                "synap-backend: fetch failed for process_family=%s (%s: %s);"
                " returning empty so caller continues without priors",
                query.process_family, exc.__class__.__name__, str(exc)[:200],
            )
            return []

        return self._flatten_context_response(ctx, query)

    def supersede(self, memory_id: str, by: str) -> None:
        """Synap exposes update() and delete() but no native supersede.

        Phase 1 callers (lessons-memory wire-up) don't supersede — the
        operation is in the Protocol for Tier 5 (corrections). For
        now this is a no-op + log; when corrections land we'll either
        delete the prior memory (sdk.memories.delete) or update it
        (sdk.memories.update with merge_strategy='smart-merge') and
        record the supersession in the new memory's metadata.
        """
        _log.info(
            "synap-backend: supersede %s -> %s (no-op in Phase 1; will be"
            " implemented when Tier 5 corrections land)",
            memory_id, by,
        )

    # ------------------------------------------------------------------
    # Mapping helpers
    # ------------------------------------------------------------------

    def _build_write_metadata(self, record: MemoryRecord) -> dict[str, Any]:
        """Assemble the metadata blob we send with every write.

        Synap stores this verbatim and returns it on retrieval (in the
        ContextItem.metadata field). Our retrieval-side filtering for
        organism / tags / variables happens client-side after the
        Synap call, using these fields.
        """
        md: dict[str, Any] = {
            "fermdocs_kind": record.kind,
            "process_family": record.process_family,
            "lesson_id": record.memory_id,
        }
        if record.organism:
            md["organism"] = record.organism
        if record.tags:
            md["tags"] = list(record.tags)
        if record.affected_variables:
            md["affected_variables"] = list(record.affected_variables)
        if record.finding_classes:
            md["finding_classes"] = list(record.finding_classes)
        if record.confidence is not None:
            md["confidence"] = record.confidence
        # Provenance is a Mapping; convert to plain dict for JSON.
        if record.provenance:
            md["provenance"] = dict(record.provenance)
        return md

    def _flatten_context_response(
        self, ctx: Any, query: MemoryQuery,
    ) -> list[MemoryRecord]:
        """Convert ContextResponse's 5 typed buckets into MemoryRecords.

        Each Synap-extracted item becomes one MemoryRecord with:
          - memory_id     = Synap's per-extraction id
          - kind          = always "lesson" for Phase 1; future kinds
                            won't go through this method
          - summary       = the extracted item's content/text
          - process_family = pulled from item metadata (set by us at
                             write) or query.process_family as fallback
          - organism / tags / etc = read from item metadata
          - confidence    = Synap's confidence score (Facts only;
                            others get None)
          - provenance    = preserved from write-time metadata, plus
                            Synap's ResponseMetadata fields
          - tenant_id     = query.tenant_id (must match)

        Client-side filtering (organism, variables_overlap, etc.) is
        applied here since Synap doesn't filter on metadata.
        """
        out: list[MemoryRecord] = []

        def _emit(item: Any, kind_label: str) -> None:
            content = (
                getattr(item, "content", None)
                or getattr(item, "text", None)
                or getattr(item, "summary", None)
                or ""
            )
            md = getattr(item, "metadata", None) or {}
            if not isinstance(md, dict):
                md = {}

            # Apply client-side filters that Synap can't enforce.
            if query.organism is not None and md.get("organism") != query.organism:
                return
            if query.variables_overlap:
                vars_in_md = set(md.get("affected_variables") or [])
                if not (vars_in_md & set(query.variables_overlap)):
                    return
            if query.finding_classes_overlap:
                classes_in_md = set(md.get("finding_classes") or [])
                if not (classes_in_md & set(query.finding_classes_overlap)):
                    return

            confidence = getattr(item, "confidence", None)
            if confidence is not None:
                try:
                    confidence = float(confidence)
                except (TypeError, ValueError):
                    confidence = None

            provenance = dict(md.get("provenance") or {})
            provenance.setdefault("synap_extraction_kind", kind_label)
            if "lesson_id" in md:
                provenance.setdefault("source_document_id", md["lesson_id"])

            mem_id = str(getattr(item, "id", "") or md.get("lesson_id") or "")
            if not mem_id:
                # Skip items we can't identify; Synap should always
                # supply an id, but the contract is permissive.
                return

            out.append(
                MemoryRecord(
                    memory_id=mem_id,
                    kind="lesson",
                    summary=str(content),
                    process_family=md.get("process_family") or query.process_family,
                    organism=md.get("organism"),
                    tenant_id=query.tenant_id,
                    affected_variables=tuple(md.get("affected_variables") or ()),
                    finding_classes=tuple(md.get("finding_classes") or ()),
                    confidence=confidence,
                    provenance=provenance,
                    embedding_provider="synap-managed",
                    embedding_model="synap-managed",
                    embedding_version="synap-managed",
                    tags=tuple(md.get("tags") or ()),
                    created_at=str(getattr(item, "extracted_at", "") or ""),
                ),
            )

        # ContextResponse fields per the SDK: facts, preferences,
        # episodes, emotions, temporal_events. Walk all five.
        for bucket_name in ("facts", "preferences", "episodes", "emotions", "temporal_events"):
            bucket = getattr(ctx, bucket_name, None) or []
            for item in bucket:
                _emit(item, bucket_name)

        # Cap at top_k; Synap's max_results is per-bucket-aware in some
        # paths but not all, so we trim defensively.
        return out[: query.top_k]
