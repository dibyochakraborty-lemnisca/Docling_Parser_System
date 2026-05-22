"""Memory layer for fermdocs.

Plan ref: plans/2026-05-10-memory-layer.md.

Across-run memory for the hypothesis stage. Bundle stays the per-run
state boundary; this layer holds priors that compound across bundles.

Phase 1 ships three backends:

  - NoopBackend (default, off): no writes, empty fetches. Existing tests
    use this; behavior is byte-identical to pre-memory-layer fermdocs.
  - StubBackend (in-memory dict): for unit tests that need to assert on
    write/fetch behavior without a real backend.
  - SynapBackend (managed): the Phase 1 production backend. Wraps the
    Synap Python SDK.

Adding a new backend (Postgres-self-hosted, Mem0, Letta, etc.) is one
new module that implements the MemoryBackend Protocol.
"""

from fermdocs_memory.base import (
    MemoryBackend,
    MemoryKind,
    MemoryQuery,
    MemoryRecord,
)
from fermdocs_memory.noop import NoopBackend
from fermdocs_memory.stub import StubBackend


def _build_synap_backend(*args, **kwargs):
    """Lazy import: SynapBackend pulls maximem-synap which is only needed
    when the Synap backend is actually wired in. Most callers (tests,
    NoopBackend default) never touch this."""
    from fermdocs_memory.synap import SynapBackend
    return SynapBackend(*args, **kwargs)


__all__ = [
    "MemoryBackend",
    "MemoryKind",
    "MemoryQuery",
    "MemoryRecord",
    "NoopBackend",
    "StubBackend",
    "_build_synap_backend",
]
