"""Minimal spike: write one document to Synap dev, wait, stop.

Purpose: settle whether the Memories panel is empty because our earlier
deletes cleaned up, or because indexing is slow.

Action plan:
  1. Send one ADD with a fresh document_id (timestamped so it can't
     hit any prior idempotency key).
  2. wait_for_completion (acknowledged + queued).
  3. Sleep 60 more seconds to let extraction/indexing finish.
  4. Print the document_id and memory_ids; stop.

Then check the dashboard Memories panel: if the document is there, we
know writes work end-to-end. If still empty, the deeper issue is
something else (indexing pipeline, filtering, scope mismatch).
"""

from __future__ import annotations

import asyncio
import os
import time
from uuid import UUID

from dotenv import load_dotenv

load_dotenv()

from maximem_synap import DocumentType, IngestMode, MaximemSynapSDK  # noqa: E402

CUSTOMER_ID = "lemnisca-internal"
PROCESS_FAMILY = "yeast_intracellular_product_fedbatch"

DIGEST_ID = f"L-PROBE-{int(time.time())}"
DIGEST_TEXT = (
    "Spike probe: pigment loss after 144h is documented in narratives"
    " for some yeast carotenoid runs but lacks coincident DO crash."
    " Synthesizer should not infer mass-transfer cause without"
    " positive evidence."
)


async def main() -> None:
    if not os.environ.get("SYNAP_API_KEY"):
        raise SystemExit("SYNAP_API_KEY not set")

    sdk = MaximemSynapSDK()
    await sdk.initialize()
    print(f"[init] instance_id={sdk.instance_id!r}")

    print(f"\n[write] document_id={DIGEST_ID}")
    resp = await sdk.memories.create(
        document=DIGEST_TEXT,
        document_type=DocumentType.DOCUMENT,
        document_id=DIGEST_ID,
        user_id=PROCESS_FAMILY,
        customer_id=CUSTOMER_ID,
        mode=IngestMode.LONG_RANGE,
        metadata={
            "lesson_id": DIGEST_ID,
            "process_family": PROCESS_FAMILY,
            "tags": ["probe", "tool-gap-axis"],
            "fermdocs_kind": "lesson",
            "spike": True,
        },
    )
    print(f"  ingestion_id={resp.ingestion_id} status={resp.status}")
    print(f"  document_id={resp.document_id}")

    print("\n[wait] wait_for_completion (timeout 90s)...")
    final = await sdk.memories.wait_for_completion(
        ingestion_id=UUID(str(resp.ingestion_id)),
        timeout_seconds=90,
        poll_interval_seconds=2,
    )
    print(f"  status={final.status}")
    print(f"  memories_created={final.memories_created}")
    print(f"  memory_ids={list(final.memory_ids or [])}")

    print("\n[soak] sleeping 60s to let indexing settle...")
    await asyncio.sleep(60)

    print(
        f"\n[done] check the dashboard Memories panel for document_id"
        f" {DIGEST_ID!r} or any memory whose source_document_id matches it."
    )
    print(f"  customer_id={CUSTOMER_ID}, user_id={PROCESS_FAMILY}")
    print(
        "  NOT cleaning up. If you want to delete after, run:\n"
        f"    parsevenv/bin/python -c \""
        f"import asyncio,os; from dotenv import load_dotenv; load_dotenv();"
        f" from maximem_synap import MaximemSynapSDK; from uuid import UUID;\n"
        f"async def go():\n"
        f"  s = MaximemSynapSDK(); await s.initialize()\n"
        f"  for mid in {list(final.memory_ids or [])}:\n"
        f"    await s.memories.delete(memory_id=UUID(mid))\n"
        f"  await s.shutdown()\n"
        f"asyncio.run(go())\""
    )

    await sdk.shutdown()


if __name__ == "__main__":
    asyncio.run(main())
