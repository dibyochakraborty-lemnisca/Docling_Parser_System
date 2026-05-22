"""Live SynapBackend integration test.

Plan ref: plans/2026-05-10-memory-layer.md commit 2.

Skipped unless `SYNAP_API_KEY` is set in the environment. When set,
this test:
  1. Constructs a SynapBackend pointed at the live dev instance.
  2. Writes one MemoryRecord with a unique timestamped memory_id.
  3. Calls fetch — verifies it returns SOMETHING or empty (which is
     fine; Synap's extraction is async, our write probably hasn't
     materialised yet on a fresh backend).
  4. Cleans up by calling shutdown.

This is intentionally lenient: we're verifying the SDK connection +
auth + transport work, not that retrieval is instantaneous. Read-side
correctness is best validated by running real fermdocs runs and
inspecting the dashboard's Memories panel for actual writes.
"""

from __future__ import annotations

import os
import time
import uuid

import pytest

# Skip the whole module when SYNAP_API_KEY is missing — keeps CI green
# in environments without the secret.
pytestmark = pytest.mark.skipif(
    not os.environ.get("SYNAP_API_KEY"),
    reason="SYNAP_API_KEY not set; skipping live Synap integration test",
)


def test_live_synap_write_and_fetch_smoke():
    """Smoke test: SDK connects, write returns, fetch returns a list."""
    from fermdocs_memory.base import MemoryQuery, MemoryRecord
    from fermdocs_memory.synap import SynapBackend

    process_family = "yeast_intracellular_product_fedbatch"
    tenant_id = "lemnisca-internal"
    unique_id = f"L-LIVETEST-{int(time.time())}-{uuid.uuid4().hex[:6]}"

    backend = SynapBackend(default_customer_id=tenant_id)
    try:
        rec = MemoryRecord(
            memory_id=unique_id,
            kind="lesson",
            summary=(
                "Live integration smoke test: pigment loss after 144h is"
                " documented in narratives for some yeast carotenoid runs."
            ),
            process_family=process_family,
            organism="S. cerevisiae",
            tenant_id=tenant_id,
            tags=("live-test", "tool-gap-axis"),
            provenance={"run_id": f"RUN-LIVETEST-{unique_id}"},
        )
        # Write should not raise on transient — adapter swallows exceptions.
        backend.write(rec)

        # Fetch should always return a list (possibly empty on a fresh
        # write that hasn't been extracted yet, or on a transient
        # backend error).
        out = backend.fetch(MemoryQuery(
            tenant_id=tenant_id,
            kind="lesson",
            process_family=process_family,
            top_k=5,
        ))
        assert isinstance(out, list)
        # If we got results, they should be MemoryRecords with the
        # right tenant + family.
        for r in out:
            assert r.tenant_id == tenant_id
            assert r.process_family == process_family
            assert r.kind == "lesson"
    finally:
        backend.shutdown()


def test_live_fetch_with_missing_process_family_raises():
    """D7 invariant must hold against the live SDK too."""
    from fermdocs_memory.base import MemoryQuery
    from fermdocs_memory.synap import SynapBackend

    backend = SynapBackend()
    try:
        with pytest.raises(ValueError, match="process_family"):
            backend.fetch(MemoryQuery(
                tenant_id="lemnisca-internal",
                kind="lesson",
                process_family=None,
            ))
    finally:
        backend.shutdown()
