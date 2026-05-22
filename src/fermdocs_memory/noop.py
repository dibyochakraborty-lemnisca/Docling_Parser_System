"""NoopBackend: the default, off-by-default memory backend.

When no MemoryBackend is configured, the runner instantiates this. Every
write is a no-op; every fetch returns []. The system runs identically
to pre-memory-layer fermdocs.

The hard regression invariant on Phase 1: with NoopBackend wired in,
prompts and test outcomes are byte-identical to today.
"""

from __future__ import annotations

from fermdocs_memory.base import (
    MemoryBackend,
    MemoryQuery,
    MemoryRecord,
    validate_query,
)


class NoopBackend:
    """Off switch for the memory layer.

    Implements MemoryBackend so the runner doesn't need a None check
    before every memory.fetch / memory.write call. The fetch still
    validates the query so D7 violations surface even when memory is
    disabled — protects against typos in callers.
    """

    def write(self, record: MemoryRecord) -> None:
        # Intentionally drop. The record's existence is captured in
        # global.md / event log; we just don't persist for cross-run.
        return None

    def fetch(self, query: MemoryQuery) -> list[MemoryRecord]:
        # Validate first so a buggy caller (e.g. forgot process_family
        # on a lesson fetch) gets the same error as on a real backend.
        validate_query(query)
        return []

    def supersede(self, memory_id: str, by: str) -> None:
        return None


# Module-level singleton convenience. Most callers don't need their own
# instance; this lets them write `from fermdocs_memory import noop_default`
# and pass it as the default kwarg without thinking about lifetimes.
noop_default: MemoryBackend = NoopBackend()
