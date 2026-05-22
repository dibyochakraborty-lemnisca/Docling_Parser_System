"""Spike: send three real lesson-digest-shaped documents to Synap and
observe what comes back from retrieval.

Goal: answer the open question from the plan review — does Synap's
extraction model preserve our digest shape (good for surfacing in
synthesizer prompts) or transform it into something else (Facts /
Episodes / etc) that requires us to rewire the prompt block?

What this script does:
  1. Ingests three lesson digests under user_id="yeast_intracellular_product_fedbatch"
     and customer_id="lemnisca-internal", each with a stable document_id.
  2. Polls each ingestion until done.
  3. Fetches via sdk.user.context.fetch with a search_query mimicking
     what the projector would send.
  4. Prints the full response shape so we can decide:
       - Does extraction surface our digest as Episodes? Facts? Both?
       - Is the original digest text preserved or paraphrased?
       - What identifiers come back — our document_id or Synap-minted?
       - What metadata flows through?

Run:
    parsevenv/bin/python spike/synap_extraction_test.py

Cleans up after itself: deletes the three test memories at the end.
Cost is on the dev instance's $12.45 starter credit.
"""

from __future__ import annotations

import asyncio
import json
import os
from datetime import datetime, timezone
from pathlib import Path

from dotenv import load_dotenv

# Load .env so SYNAP_API_KEY is available without sourcing.
load_dotenv()

from maximem_synap import (  # noqa: E402  (import after load_dotenv)
    DocumentType,
    IngestMode,
    MaximemSynapSDK,
)


CUSTOMER_ID = "lemnisca-internal"
PROCESS_FAMILY = "yeast_intracellular_product_fedbatch"  # used as user_id

# Three digests modelled on what the lessons_summarizer agent actually
# produces today. Verbatim shapes from prior runs in global.md.
# Use a per-run timestamp suffix so re-runs don't hit the idempotency
# key from a prior spike run.
import time as _time
SPIKE_RUN_ID = f"R{int(_time.time())}"

LESSON_DIGESTS = [
    {
        "lesson_id": f"L-{SPIKE_RUN_ID}-0001",
        "text": (
            "Recurring rejection axis: hypotheses extending single-batch"
            " evidence to multi-batch claims get knocked down by the"
            " critic [trajectory-axis] when only RUN-0001 has the cited"
            " trajectory. Synthesizer should scope to within-run claims"
            " when only one run has the relevant data."
        ),
        "tags": ["trajectory-axis", "scope-narrowing", "single-batch-evidence"],
        "run_id": "RUN-SPIKE-0001",
        "hyp_id": "H-0003",
    },
    {
        "lesson_id": "L-SPIKE-0002",
        "text": (
            "Pattern across yeast carotenoid runs: pigment loss after 144h"
            " is documented in narratives (white-cell phenotype) but lacks"
            " a coincident DO crash or kLa excursion. Critic [tool-gap-axis]"
            " correctly fired when synthesizer used 'insufficient_data' to"
            " hide behind a tool gap; the bundle had the data, the"
            " toolchain failed to compute on RUN-0004."
        ),
        "tags": ["tool-gap-axis", "pigment-loss", "yeast-carotenoid"],
        "run_id": "RUN-SPIKE-0002",
        "hyp_id": "H-0007",
    },
    {
        "lesson_id": "L-SPIKE-0003",
        "text": (
            "Robustness reminder: r=-0.99 between min DO and final WCW on"
            " n=6 was rejected with [robustness-axis] until synthesizer"
            " surfaced the bootstrap CI [-0.99, -0.40] and downgraded the"
            " language to 'preliminary association'. Small-n correlations"
            " on this family always need the n+CI caveat."
        ),
        "tags": ["robustness-axis", "weak-n", "correlation-CI"],
        "run_id": "RUN-SPIKE-0003",
        "hyp_id": "H-0011",
    },
]

OUTPUT_PATH = Path(__file__).parent / "synap_spike_output.json"


async def main() -> None:
    if not os.environ.get("SYNAP_API_KEY"):
        raise SystemExit("SYNAP_API_KEY not set in environment / .env")

    sdk = MaximemSynapSDK()
    await sdk.initialize()
    print(f"[init] instance_id={sdk.instance_id!r}")

    output: dict = {
        "spike_ts": datetime.now(timezone.utc).isoformat(),
        "customer_id": CUSTOMER_ID,
        "process_family_as_user_id": PROCESS_FAMILY,
        "ingestions": [],
        "fetch_response": None,
        "fetch_response_raw": None,
        "decision_inputs": {},
    }

    try:
        # ---------- WRITE ----------
        for lesson in LESSON_DIGESTS:
            print(f"\n[write] document_id={lesson['lesson_id']}")
            metadata = {
                "lesson_id": lesson["lesson_id"],
                "tags": lesson["tags"],
                "run_id": lesson["run_id"],
                "hyp_id": lesson["hyp_id"],
                "process_family": PROCESS_FAMILY,
                "fermdocs_kind": "lesson",
                "spike": True,
            }
            resp = await sdk.memories.create(
                document=lesson["text"],
                document_type=DocumentType.DOCUMENT,
                document_id=lesson["lesson_id"],
                user_id=PROCESS_FAMILY,
                customer_id=CUSTOMER_ID,
                mode=IngestMode.LONG_RANGE,
                metadata=metadata,
            )
            print(f"  -> ingestion_id={resp.ingestion_id} status={resp.status}")
            output["ingestions"].append({
                "lesson_id": lesson["lesson_id"],
                "ingestion_id": str(resp.ingestion_id),
                "document_id": resp.document_id,
                "status": str(resp.status),
            })

        # ---------- WAIT FOR PROCESSING ----------
        # SDK has a built-in wait_for_completion; use it.
        print("\n[wait] waiting for completions (timeout 90s each)...")
        from uuid import UUID
        for ing in output["ingestions"]:
            try:
                final = await sdk.memories.wait_for_completion(
                    ingestion_id=UUID(ing["ingestion_id"]),
                    timeout_seconds=90,
                    poll_interval_seconds=2,
                )
                ing["final_status"] = str(final.status)
                ing["completed_at"] = (
                    final.completed_at.isoformat() if final.completed_at else None
                )
                ing["memories_created"] = final.memories_created
                ing["memory_ids"] = list(final.memory_ids or [])
                ing["error_message"] = final.error_message
                print(
                    f"  {ing['lesson_id']} -> {final.status}"
                    f" (memories_created={final.memories_created})"
                )
            except Exception as e:
                ing["final_status"] = "timeout_or_error"
                ing["error_message"] = str(e)
                print(f"  {ing['lesson_id']} -> ERROR: {e}")

        # ---------- FETCH ----------
        # Mimic what the projector will send: a search_query close to
        # the topic.summary it's working on this turn.
        topic_summary = "pigment loss yeast carotenoid"
        print(f"\n[fetch] sdk.user.context.fetch(user_id={PROCESS_FAMILY!r})")
        print(f"        search_query={topic_summary!r}")
        # Synap had a transient ServiceUnavailable on first run; retry up to 3x.
        ctx = None
        last_err: Exception | None = None
        for attempt in range(3):
            try:
                ctx = await sdk.user.context.fetch(
                    user_id=PROCESS_FAMILY,
                    search_query=[topic_summary],
                    max_results=10,
                )
                break
            except Exception as e:
                last_err = e
                print(f"  fetch attempt {attempt+1}/3 failed: {e!r}")
                await asyncio.sleep(2 ** attempt)
        if ctx is None:
            raise last_err  # type: ignore[misc]
        print(f"  facts: {len(ctx.facts)}")
        print(f"  preferences: {len(ctx.preferences)}")
        print(f"  episodes: {len(ctx.episodes)}")
        print(f"  emotions: {len(ctx.emotions)}")
        print(f"  temporal_events: {len(ctx.temporal_events)}")

        # Serialize the response so we can inspect it carefully after.
        def _to_dict(item) -> dict:
            if hasattr(item, "model_dump"):
                return item.model_dump(mode="json")
            return {k: v for k, v in item.__dict__.items() if not k.startswith("_")}

        output["fetch_response"] = {
            "facts": [_to_dict(f) for f in ctx.facts],
            "preferences": [_to_dict(p) for p in ctx.preferences],
            "episodes": [_to_dict(e) for e in ctx.episodes],
            "emotions": [_to_dict(e) for e in ctx.emotions],
            "temporal_events": [_to_dict(t) for t in ctx.temporal_events],
            "metadata": _to_dict(ctx.metadata) if ctx.metadata else None,
        }

        # ---------- DECISION INPUTS ----------
        # The questions the spike must answer, with answers from the response.
        all_items = (
            output["fetch_response"]["facts"]
            + output["fetch_response"]["preferences"]
            + output["fetch_response"]["episodes"]
            + output["fetch_response"]["emotions"]
            + output["fetch_response"]["temporal_events"]
        )
        # Q1: Does verbatim digest text survive in any returned item?
        verbatim_hits = []
        for digest in LESSON_DIGESTS:
            chars = digest["text"][:80]  # first 80 chars as a fingerprint
            for item in all_items:
                content = (
                    item.get("content")
                    or item.get("text")
                    or item.get("summary")
                    or ""
                )
                if chars[:50] in content:
                    verbatim_hits.append({
                        "digest_id": digest["lesson_id"],
                        "matched_in": item,
                    })
        # Q2: Do our document_ids round-trip in retrieved metadata?
        doc_id_roundtrip = []
        for item in all_items:
            md = item.get("metadata") or {}
            if isinstance(md, dict) and md.get("lesson_id"):
                doc_id_roundtrip.append({
                    "lesson_id_in_metadata": md.get("lesson_id"),
                    "item_id": item.get("id"),
                })

        output["decision_inputs"] = {
            "n_returned_total": len(all_items),
            "n_facts": len(ctx.facts),
            "n_episodes": len(ctx.episodes),
            "verbatim_digest_survives": bool(verbatim_hits),
            "verbatim_hits": verbatim_hits[:3],
            "our_document_id_in_metadata": bool(doc_id_roundtrip),
            "document_id_roundtrip_examples": doc_id_roundtrip[:3],
        }

        print("\n[decision-inputs]")
        for k, v in output["decision_inputs"].items():
            if isinstance(v, list):
                print(f"  {k}: {len(v)} items")
            else:
                print(f"  {k}: {v}")

        # ---------- WRITE OUTPUT FILE ----------
        OUTPUT_PATH.write_text(json.dumps(output, indent=2, default=str))
        print(f"\n[done] full response saved to {OUTPUT_PATH}")

    finally:
        # ---------- CLEANUP ----------
        # Delete by memory_id (UUID), which we got from wait_for_completion.
        print("\n[cleanup] deleting spike memories by memory_id...")
        from uuid import UUID
        deleted = 0
        for ing in output["ingestions"]:
            for mid_str in ing.get("memory_ids", []) or []:
                try:
                    await sdk.memories.delete(memory_id=UUID(mid_str))
                    deleted += 1
                except Exception as e:
                    print(f"  failed to delete {mid_str}: {e}")
        print(f"  deleted {deleted} memories")
        await sdk.shutdown()


if __name__ == "__main__":
    asyncio.run(main())
