"""StubBackend: in-memory dict implementation for unit tests.

Lets tests assert behavior of the wire-up without needing the cloud
Synap backend or a real Postgres container. Contract-equivalent to
production backends within the limits of an in-memory store:

  - write/fetch/supersede behave correctly
  - fetch ranking falls back to substring match when semantic_query is
    set (no real embeddings in tests)
  - filters: kind, tenant_id, process_family, organism,
    variables_overlap, finding_classes_overlap, top_k, include_superseded

What StubBackend deliberately does NOT do:
  - persist across process boundaries (fixture scope)
  - implement embedding-based ranking (any unit test that needs that
    is using the wrong abstraction; mock a SynapBackend instead)
"""

from __future__ import annotations

from typing import Iterable

from fermdocs_memory.base import (
    MemoryQuery,
    MemoryRecord,
    validate_query,
)


def _any_overlap(a: Iterable[str], b: Iterable[str]) -> bool:
    """True when sets share at least one element. Empty filter on b
    means 'no filter' and matches everything."""
    bs = set(b)
    if not bs:
        return True
    return any(x in bs for x in a)


class StubBackend:
    """Process-local memory store.

    State is one dict keyed by memory_id. Reads scan the dict applying
    filters; this is O(N) per fetch which is fine for tests with
    dozens of records.
    """

    def __init__(self) -> None:
        self._store: dict[str, MemoryRecord] = {}
        self._supersession: dict[str, str] = {}

    # ------------------------------------------------------------------
    # MemoryBackend Protocol implementation
    # ------------------------------------------------------------------

    def write(self, record: MemoryRecord) -> None:
        # Phase 1 callers don't re-write; treat duplicate memory_id with
        # different content as a programming error so tests catch it.
        existing = self._store.get(record.memory_id)
        if existing is not None and existing != record:
            raise ValueError(
                f"StubBackend: memory_id={record.memory_id!r} already"
                f" exists with different content; refusing silent"
                f" overwrite."
            )
        self._store[record.memory_id] = record

    def fetch(self, query: MemoryQuery) -> list[MemoryRecord]:
        validate_query(query)
        candidates = [
            r for r in self._store.values()
            if self._matches_filters(r, query)
        ]
        ranked = self._rank(candidates, query)
        return ranked[: query.top_k]

    def supersede(self, memory_id: str, by: str) -> None:
        if memory_id not in self._store:
            # Idempotent: superseding a missing record is a no-op so
            # callers can run cleanup without checking existence.
            return
        existing = self._store[memory_id]
        # Update in place via dataclass.replace pattern; MemoryRecord is
        # frozen so we have to build a new one.
        from dataclasses import replace
        self._store[memory_id] = replace(existing, superseded_by=by)
        self._supersession[memory_id] = by

    # ------------------------------------------------------------------
    # filtering + ranking helpers
    # ------------------------------------------------------------------

    def _matches_filters(self, r: MemoryRecord, q: MemoryQuery) -> bool:
        if r.tenant_id != q.tenant_id:
            return False
        if q.kind is not None and r.kind != q.kind:
            return False
        if q.process_family is not None and r.process_family != q.process_family:
            return False
        if q.organism is not None and r.organism != q.organism:
            return False
        if q.variables_overlap and not _any_overlap(
            r.affected_variables, q.variables_overlap
        ):
            return False
        if q.finding_classes_overlap and not _any_overlap(
            r.finding_classes, q.finding_classes_overlap
        ):
            return False
        if r.superseded_by is not None and not q.include_superseded:
            return False
        return True

    def _rank(
        self,
        candidates: list[MemoryRecord],
        q: MemoryQuery,
    ) -> list[MemoryRecord]:
        """Rank by semantic_query substring score, else by recency.

        Substring score: number of query tokens (whitespace-split) found
        in the summary. Crude but deterministic; production backends
        use embeddings.
        """
        if q.semantic_query:
            tokens = [t.lower() for t in q.semantic_query.split() if t]

            def score(r: MemoryRecord) -> int:
                summary = r.summary.lower()
                return sum(1 for t in tokens if t in summary)

            # Stable sort: by score desc, then created_at desc as tie-breaker.
            return sorted(
                candidates,
                key=lambda r: (-score(r), -_created_at_sort_key(r)),
            )
        # No semantic query → newest first.
        return sorted(
            candidates,
            key=lambda r: -_created_at_sort_key(r),
        )


def _created_at_sort_key(r: MemoryRecord) -> float:
    """Coerce ISO8601 created_at to a sortable float.

    Empty string sorts as 0 (oldest). Garbage timestamps also sort as 0
    rather than raising — backends fill created_at server-side, so by
    the time records are queried this is well-formed.
    """
    if not r.created_at:
        return 0.0
    from datetime import datetime
    try:
        return datetime.fromisoformat(r.created_at.replace("Z", "+00:00")).timestamp()
    except ValueError:
        return 0.0
