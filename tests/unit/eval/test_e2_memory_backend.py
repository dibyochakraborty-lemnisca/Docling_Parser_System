from __future__ import annotations

from fermdocs_eval.fixture_builder import DefectSpec
from fermdocs_eval.suites.e2 import EVAL_TENANT_ID, _make_memory_backend


def _spec(*, memory_seed: tuple = ()) -> DefectSpec:
    return DefectSpec(
        fixture_id="t-mem",
        labeled_axis="memory-axis" if memory_seed else "clean",
        difficulty="clear" if memory_seed else "clean",
        leading_question="anything",
        mutation_kind="noop",
        memory_seed=memory_seed,
    )


def test_empty_seed_returns_empty_backend() -> None:
    backend = _make_memory_backend(_spec(memory_seed=()))
    # StubBackend stores in self._store dict.
    assert len(backend._store) == 0


def test_single_seed_written() -> None:
    spec = _spec(memory_seed=(("penicillin_fedbatch", "lesson text"),))
    backend = _make_memory_backend(spec)
    assert len(backend._store) == 1
    rec = next(iter(backend._store.values()))
    assert rec.process_family == "penicillin_fedbatch"
    assert rec.summary == "lesson text"
    assert rec.tenant_id == EVAL_TENANT_ID
    assert rec.kind == "lesson"


def test_multiple_seeds_get_distinct_ids() -> None:
    spec = _spec(memory_seed=(
        ("penicillin_fedbatch", "lesson A"),
        ("penicillin_fedbatch", "lesson B"),
    ))
    backend = _make_memory_backend(spec)
    assert len(backend._store) == 2
    ids = list(backend._store.keys())
    assert ids[0] != ids[1]


def test_fetch_finds_seeded_lesson() -> None:
    from fermdocs_memory.base import MemoryQuery

    spec = _spec(memory_seed=(
        ("penicillin_fedbatch", "yeast cooling-jacket valve fix"),
    ))
    backend = _make_memory_backend(spec)
    q = MemoryQuery(
        tenant_id=EVAL_TENANT_ID,
        kind="lesson",
        process_family="penicillin_fedbatch",
        semantic_query="cooling",
        top_k=5,
    )
    results = backend.fetch(q)
    assert len(results) == 1
    assert "cooling-jacket" in results[0].summary
