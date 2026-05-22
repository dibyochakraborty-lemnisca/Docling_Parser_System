"""Minimal fetch probe: try every variant we can think of.

Goal: find ANY fetch call that doesn't return ServiceUnavailable.
If all fail, retrieval is gated on paid plan or there's a backend
issue we can't work around without Synap support.

Tries in order, stopping at the first success:
  1. user.context.fetch(user_id=...) — no search_query, no max_results
  2. user.context.fetch(user_id=..., max_results=1)
  3. user.context.fetch(user_id=..., search_query=["yeast"])
  4. customer.context.fetch(customer_id=...)
  5. client.context.fetch()
"""

from __future__ import annotations

import asyncio
import os

from dotenv import load_dotenv

load_dotenv()

from maximem_synap import MaximemSynapSDK  # noqa: E402

CUSTOMER_ID = "lemnisca-internal"
PROCESS_FAMILY = "yeast_intracellular_product_fedbatch"


async def try_call(label: str, coro_factory):
    print(f"\n[try] {label}")
    try:
        ctx = await coro_factory()
        print("  SUCCESS")
        print(f"    facts: {len(ctx.facts)}")
        print(f"    preferences: {len(ctx.preferences)}")
        print(f"    episodes: {len(ctx.episodes)}")
        print(f"    emotions: {len(ctx.emotions)}")
        print(f"    temporal_events: {len(ctx.temporal_events)}")
        # Print first item content if any
        for label, items in [
            ("facts", ctx.facts),
            ("episodes", ctx.episodes),
            ("temporal_events", ctx.temporal_events),
        ]:
            for item in items[:2]:
                content = (
                    getattr(item, "content", None)
                    or getattr(item, "text", None)
                    or getattr(item, "summary", None)
                    or "?"
                )
                print(f"    {label}[{getattr(item, 'id', '?')}]: {content[:200]}")
        return ctx
    except Exception as e:
        print(f"  FAILED: {type(e).__name__}: {e}")
        return None


async def main() -> None:
    sdk = MaximemSynapSDK()
    await sdk.initialize()
    print(f"[init] instance_id={sdk.instance_id!r}")

    attempts = [
        (
            "user.context.fetch(user_id) — bare",
            lambda: sdk.user.context.fetch(user_id=PROCESS_FAMILY),
        ),
        (
            "user.context.fetch(user_id, max_results=1)",
            lambda: sdk.user.context.fetch(user_id=PROCESS_FAMILY, max_results=1),
        ),
        (
            "user.context.fetch(user_id, search_query=['yeast'])",
            lambda: sdk.user.context.fetch(
                user_id=PROCESS_FAMILY, search_query=["yeast"]
            ),
        ),
        (
            "customer.context.fetch(customer_id) — bare",
            lambda: sdk.customer.context.fetch(customer_id=CUSTOMER_ID),
        ),
        (
            "client.context.fetch() — bare",
            lambda: sdk.client.context.fetch(),
        ),
    ]

    success_count = 0
    for label, factory in attempts:
        ctx = await try_call(label, factory)
        if ctx is not None:
            success_count += 1

    print(f"\n[result] {success_count}/{len(attempts)} succeeded")
    await sdk.shutdown()


if __name__ == "__main__":
    asyncio.run(main())
