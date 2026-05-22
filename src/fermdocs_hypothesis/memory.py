"""PastInsightStore protocol + null implementation.

Plan ref: plans/2026-05-03-hypothesis-debate-v0.md §16 (forward-compat seam).

v0 wires NullPastInsightStore everywhere. v1 swaps in a SuperMemory client
without prompt or schema changes.
"""

from __future__ import annotations

from typing import Protocol

from pydantic import BaseModel, ConfigDict, Field


class PastInsight(BaseModel):
    model_config = ConfigDict(frozen=True)

    insight_id: str
    summary: str
    source_hypothesis_id: str | None = None
    confidence: float = Field(default=0.0, ge=0.0, le=1.0)
    tags: list[str] = Field(default_factory=list)


class PastInsightStore(Protocol):
    def query(self, topic: str, k: int = 5) -> list[PastInsight]: ...


class NullPastInsightStore:
    """Returns no past insights. Default for v0."""

    def query(self, topic: str, k: int = 5) -> list[PastInsight]:
        return []
