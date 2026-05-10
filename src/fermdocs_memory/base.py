"""MemoryBackend Protocol + record/query schemas.

Plan ref: plans/2026-05-10-memory-layer.md, eng-review decisions D2-D8.

The Protocol is the contract every backend (Noop, Stub, Synap, future
Postgres) must satisfy. The contract is intentionally small:

  - write(record): persist one record
  - fetch(query):  retrieve top-k matching records
  - supersede(id, by): mark a record as replaced by another

Three invariants the contract enforces by construction:

  1. (D7) MemoryQuery(kind="lesson", process_family=None) raises
     ValueError. There is no "all strains, all families" lesson
     retrieval. Cross-family is opt-in via a future explicit method.
  2. (D8) For kind="lesson", process_family is the primary retrieval
     key. organism is a secondary re-ranker that may vary across runs.
  3. MemoryRecord is frozen; provenance is stored as MappingProxyType
     so callers can't mutate it after construction (matches the audit
     intent of frozen dataclasses).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Any, Literal, Mapping, Protocol


# Closed vocabulary of memory kinds. Phase 1 only writes "lesson"; the
# others are reserved for Tier 2/3/5 (see plan doc).
MemoryKind = Literal[
    "lesson",
    "ratified_hypothesis",
    "rejected_hypothesis",
    "correction",
]


def _freeze_provenance(p: Mapping[str, Any] | dict | None) -> Mapping[str, Any]:
    """Return a read-only mapping. Empty dict when input is None.

    We accept dict/None at construction so callers don't have to wrap
    every literal, but store as MappingProxyType so the resulting
    MemoryRecord can't be mutated after the fact.
    """
    if p is None:
        return MappingProxyType({})
    if isinstance(p, MappingProxyType):
        return p
    return MappingProxyType(dict(p))


@dataclass(frozen=True)
class MemoryRecord:
    """One persisted memory.

    Field invariants:
      - memory_id: stable across re-reads. For lessons, this is the
        lesson_id minted at emission (D2).
      - process_family: closed-vocab string from the family registry
        (e.g. "yeast_intracellular_product_fedbatch"). Primary
        retrieval key for kind="lesson" (D8).
      - organism: free-form string from the dossier; varies across runs.
        Secondary re-ranker / filter only.
      - tenant_id: required. Routes to the per-tenant scope on the
        backend (Synap Customer / future Postgres schema). Default
        "default" for single-tenant Phase 1 deploys.
      - provenance: immutable Mapping. For lessons:
        {run_id, hyp_id, generation_timestamp, source_event_offset,
         lesson_id}.
      - embedding_provider/model/version: even when the backend
        manages embeddings (Synap), we still carry these so a future
        backend swap can audit/migrate. Synap adapter sets
        embedding_provider="synap-managed".
    """

    memory_id: str
    kind: MemoryKind
    summary: str
    process_family: str | None
    organism: str | None
    tenant_id: str
    affected_variables: tuple[str, ...] = ()
    finding_classes: tuple[str, ...] = ()
    confidence: float | None = None
    provenance: Mapping[str, Any] = field(default_factory=dict)
    embedding_provider: str = "unset"
    embedding_model: str = "unset"
    embedding_version: str = "unset"
    tags: tuple[str, ...] = ()
    created_at: str = ""  # ISO8601 UTC; backend fills if blank
    superseded_by: str | None = None

    def __post_init__(self) -> None:
        # Lock provenance after construction. Frozen dataclass freezes
        # attribute assignment but not inner dict mutation; this stops
        # the loophole.
        object.__setattr__(self, "provenance", _freeze_provenance(self.provenance))


@dataclass(frozen=True)
class MemoryQuery:
    """One retrieval request.

    Hard rules (enforced by backends, not by the dataclass):
      - kind="lesson" + process_family=None must raise ValueError (D7).
      - tenant_id is required and routes to the backend's tenant scope.

    Filtering shape:
      - process_family + organism: scope filter (applied first)
      - kind: type filter
      - variables_overlap / finding_classes_overlap: any-overlap match
      - semantic_query: ranks within the filtered subset; backends with
        no embedding capability (Stub) fall back to substring match.
      - top_k: cap; backends return at most this many records.
      - include_superseded: default False; Tier 2+ may flip this.
    """

    tenant_id: str
    kind: MemoryKind | None = None
    process_family: str | None = None
    organism: str | None = None
    variables_overlap: tuple[str, ...] = ()
    finding_classes_overlap: tuple[str, ...] = ()
    semantic_query: str | None = None
    top_k: int = 5
    include_superseded: bool = False


def validate_query(q: MemoryQuery) -> None:
    """Enforce the D7 invariant: no silent cross-strain lesson retrieval.

    Backends MUST call this before executing a fetch. We keep the check
    here (rather than in __post_init__) so a query can be constructed
    for inspection without raising; the raise happens at fetch time.
    """
    if q.kind == "lesson" and q.process_family is None:
        raise ValueError(
            "MemoryQuery(kind='lesson', process_family=None) is not"
            " allowed: cross-family lesson retrieval would silently leak"
            " priors across strains. Pass an explicit process_family,"
            " or use a backend-specific cross-family method when one"
            " exists. (D7)"
        )


class MemoryBackend(Protocol):
    """The contract every memory adapter implements.

    Three operations; that's the whole API. Backend-specific config
    (DB connection strings, API keys, tenant routing) lives in the
    adapter's __init__, not here.
    """

    def write(self, record: MemoryRecord) -> None:
        """Persist one record. Idempotent on memory_id.

        Backends should treat re-write of an existing memory_id as
        either a no-op (if content matches) or as an explicit error
        (if content differs). Phase 1 callers don't re-write.
        """
        ...

    def fetch(self, query: MemoryQuery) -> list[MemoryRecord]:
        """Return up to query.top_k records matching the filters.

        MUST call validate_query(query) first; will raise on D7
        violations.

        Ordering: when semantic_query is set, results are ranked by
        similarity (cosine, substring score, or backend-native).
        Otherwise, ordered by created_at DESC (newest first).
        """
        ...

    def supersede(self, memory_id: str, by: str) -> None:
        """Mark `memory_id` as replaced by `by`.

        Future retrievals filter superseded records by default unless
        MemoryQuery.include_superseded is True. Idempotent: re-applying
        with the same `by` is a no-op.

        Phase 1 callers don't supersede; the operation exists so Tier 5
        (corrections) can land without Protocol changes.
        """
        ...
