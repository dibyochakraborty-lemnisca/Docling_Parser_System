# Memory Layer — Plan

**Date:** 2026-05-10
**Branch:** `memory-layer-plan` (off `followup-context`)
**Status:** Plan only — no code yet. Phase 1 is the first implementation step; everything past it requires explicit go-ahead.

## Why this exists

fermdocs is currently a **single-shot reasoner over one bundle**. The bundle is a strong unit of state — typed, versioned, inspectable — but everything outside the bundle gets thrown away:

- The `lessons_summarizer` agent already produces 50–60 token distilled lessons per run. They land in `global.md` and disappear.
- Every accepted `FinalHypothesis` from every prior run — the substantive "what did we conclude on this strain" record — is invisible to the next run.
- Every rejected hypothesis (with critic reasons) is invisible. **Negative examples are gold and we throw them out.**
- Human follow-up corrections ("no, this wasn't oxygen limitation, the impeller had degraded") affect one run only.
- A new bundle from the same strain enters the pipeline cold. "Anomalous" only means "weird vs physical bounds" — it never means "weird vs this strain's history."

The framing: **a world model without memory is a single-shot reasoner with a fancy state representation.** Memory is the substrate that lets specialists become *learned priors* over a deployment, not just static prompts.

This plan ships memory in measured stages, with deterministic Postgres-native primitives first, behind an adapter so a managed cloud product can swap in later if the relevance ceiling becomes the bottleneck.

## Non-goals

- **Replacing the bundle.** Bundle is the per-run boundary; nothing in this plan changes that.
- **Replacing characterize/diagnose state.** Within-run state stays in `global.md` and bundle JSON. Memory is across-run only.
- **Vendor lock-in to any specific cloud memory product** (Synap, Mem0, Letta, etc.). Phase 1 backend is plain Postgres; cloud is one swappable adapter.
- **Trajectory-shape similarity / dynamic time warping.** Powerful but real engineering; deferred to later.
- **Any change that lets memory silently override bundle evidence.** Memory is *prior context*, never *ground truth* over the current bundle.

## Architecture in one diagram

```
     +---------- characterize ---------+
     |  per-run KPIs, findings,        |
     |  narratives                     |
     +-----------+---------------------+
                 |
                 v
     +---------- diagnose -------------+
                 |
                 v
     +---------- hypothesis -----------+
     |  specialists / synthesizer /    |
     |  critic / judge / lessons       |
     +-----+----------+----------+-----+
           ^          ^          |
           |          |          v
         (read)    (read)    (write)
           |          |          |
           |          |          |
     +-----+----------+----------+-----+
     |        MemoryBackend            |  Phase 1: Postgres adapter
     |  (Protocol; pluggable)          |  Phase 2: cloud adapter (later)
     +---------------------------------+
           ^          ^          ^
           |          |          |
     [lessons]  [ratified +    [strain-conditional
                 rejected      KPI priors]
                 hypotheses]
```

Key property: **agents call `memory.fetch_context(…)` and don't know what's underneath.** Lock-in stays at the adapter file.

## Five tiers, in priority order

Each tier is independently shippable and testable. **Phase 1 is Tier 1 only.**

### Tier 1 — Lessons memory (Phase 1)

**What:** Persist every `lessons_summarizer` output keyed by `(organism, process_family, embedding)`. At synthesizer-view-build time, retrieve top-K relevant prior lessons and inject as `view.prior_lessons`.

**Why this first:** highest impact-per-hour. We're already producing the data; we're just throwing it away. The synthesizer prompt already has a `cross_topic_lessons` slot — same shape, sourced from history instead of in-run state. **One change in input, immediate change in output.**

**What gets written:** `LessonRecord(lesson_text, organism, process_family, hyp_id, run_id, embedding, created_at)`.

**What gets read:** at the start of every hypothesis run, fetch top 5 lessons by cosine similarity over the run's seed-topic summaries, filtered to the run's organism. Inject into specialist + synthesizer + critic views.

**Effort:** ~1 day.

### Tier 2 — Ratified-hypothesis store

**What:** Index every accepted `FinalHypothesis`. Fields: `summary`, `organism`, `process_family`, `affected_variables`, `cited_finding_classes` (e.g. `B10_overflow`, `A14_do_margin`), `critic_flag`, `judge_ruled_criticism_valid`, `confidence`, `confidence_basis`, embedding over `summary`.

**Why:** specialists become learned priors. The kinetics specialist sees "your last 12 ratified claims about μ post-induction in *S. cerevisiae*; here's what was rejected and why."

**What gets written:** at run finalization (the same projector point where `_render_charts_into_finals` runs).

**What gets read:** at specialist-view-build time, retrieve top-K hypotheses where (organism matches) AND (`affected_variables` overlap) OR (`cited_finding_classes` overlap). Each specialist gets a cap of 4–6 retrieved priors.

**Effort:** ~1 day on top of Tier 1.

### Tier 3 — Rejected-hypothesis store

**What:** Same shape as Tier 2 but for `RejectedHypothesis`. Carries `rejection_reason` and `critic_reasons`.

**Why:** negative examples. The critic's reasons-for-rejection are some of the most information-dense data we produce. "We rejected 8 hypotheses in this organism/variable space because they extended single-batch evidence to multi-batch claims." Saves the synthesizer from re-learning each time.

**Effort:** ~0.5 day on top of Tier 2 (separate index, same adapter primitives).

### Tier 4 — Strain-conditional KPI prior table

**What:** Per `(organism, process_family, variable)`, store the empirical distribution of catalog metrics (mu_max, peak_titer, doubling_time, q_s, B10 rq, B16 closure, etc.) from every prior run. New table: `kpi_priors(organism, process_family, variable, n_runs, p2_5, p25, median, p75, p97_5, mean, std, last_updated)`.

**Why:** characterization's `finding_validator` currently checks against organism-agnostic `_PHYSICAL_BOUNDS`. With this tier, it can additionally check against strain-conditional 2-sigma bands and emit `strain_anomaly` findings: *"this run's μ_max sits 2.4 σ below the 47-run cohort for this strain in this reactor."* Deterministic, no LLM. The reviewer's *"anomalous relative to this strain's history"* lift, in code.

**What gets written:** every catalog-runner emission appends to a per-strain rolling stats record. Update is incremental (Welford / parallel-mean).

**What gets read:** `finding_validator` queries `kpi_priors` per metric and emits an additional `strain_anomaly` finding alongside its existing physicality check.

**Effort:** ~2 days. Bigger because it's a new finding-type and adds a write step to characterize.

### Tier 5 — Human-correction memory

**What:** When a user submits a follow-up answer or HITL correction, persist it with high weight + provenance. Schema: `correction_text`, `correcting_user_id`, `original_hypothesis_id`, `original_run_id`, `verdict` (`refute` / `confirm` / `refine`), embedding.

**Why:** these are the closest thing we have to *labelled training data*. A senior scientist's correction on RUN-2024-091 ("impeller had degraded") becomes a durable prior. Next time a similar signature appears, surface the correction as a strong hint.

**What gets written:** at follow-up answer submission and at the existing HITL answer endpoint.

**What gets read:** alongside Tier 2 retrieval; corrections rank above ratified hypotheses by default.

**Effort:** ~1 day on top of Tier 2.

---

## The MemoryBackend interface (frozen contract)

This is what the rest of the system depends on. Keeping it small is the whole point.

```python
# src/fermdocs_memory/base.py
from dataclasses import dataclass
from typing import Protocol, Literal

MemoryKind = Literal[
    "lesson",
    "ratified_hypothesis",
    "rejected_hypothesis",
    "correction",
]

@dataclass(frozen=True)
class MemoryRecord:
    memory_id: str
    kind: MemoryKind
    summary: str          # the human-readable blob; embedding source
    organism: str | None
    process_family: str | None
    affected_variables: tuple[str, ...]
    finding_classes: tuple[str, ...]   # e.g. ("B10_overflow", "A14_do_margin")
    confidence: float | None
    provenance: dict       # {run_id, hyp_id, generation_timestamp, ...}
    created_at: str        # ISO8601 UTC
    superseded_by: str | None = None

@dataclass(frozen=True)
class MemoryQuery:
    kind: MemoryKind | None  # None = all kinds
    organism: str | None
    process_family: str | None
    variables_overlap: tuple[str, ...] = ()
    finding_classes_overlap: tuple[str, ...] = ()
    semantic_query: str | None = None  # embedding-based retrieval
    top_k: int = 5
    include_superseded: bool = False

class MemoryBackend(Protocol):
    def write(self, record: MemoryRecord) -> None: ...
    def fetch(self, query: MemoryQuery) -> list[MemoryRecord]: ...
    def supersede(self, memory_id: str, by: str) -> None: ...
```

The Protocol is the contract. Adapters in Phase 1:

- `NoopBackend` — default. `fetch` returns `[]`. Lets the system run with memory disabled. **Tests use this by default.**
- `StubBackend` — in-memory dict. Lets tests assert behavior.
- `PostgresBackend` — the real one in Phase 1. Uses `pgvector` for embeddings.

Later adapters (not Phase 1):
- `SynapBackend` — cloud. Maps `MemoryKind` to their typed memory categories.
- `Mem0Backend` / `LettaBackend` — if we ever need them.

---

## Phase 1 PR plan (the only thing this plan commits to)

Four commits on a child branch when this gets approved.

### Commit 1 — Protocol + Noop + Stub backends + tests

- New package `src/fermdocs_memory/`
- `base.py`: MemoryRecord, MemoryQuery, MemoryBackend Protocol, MemoryKind enum
- `noop.py`: returns empty
- `stub.py`: in-memory dict, supports write/fetch/supersede
- `tests/unit/memory/test_backends.py`: contract tests both backends pass
- Wire `NoopBackend` as default into runner + characterize entry points behind `memory: MemoryBackend = NoopBackend()` parameter

**Hard rule:** every existing test must pass unchanged. Default `NoopBackend` means zero behavior change.

### Commit 2 — Postgres adapter

- `postgres.py`: SQLAlchemy model `MemoryRow` + `PostgresBackend` impl
- New table: `memory_records(memory_id, kind, summary, organism, process_family, affected_variables_jsonb, finding_classes_jsonb, confidence, provenance_jsonb, embedding_vector pgvector(768), created_at, superseded_by)`
- Indexes: btree on `(kind, organism, process_family)`, GIN on `affected_variables_jsonb`, ivfflat on `embedding_vector`
- `tests/integration/memory/test_postgres_backend.py` against a real Postgres + pgvector container (the same fixture pattern we use for ingest)
- `EMBEDDING_PROVIDER` plumbing: small helper that calls Gemini text-embedding via existing `_client` so we don't add a new vendor dep

**Hard rule:** Postgres adapter is opt-in. `DATABASE_URL` empty → we instantiate `NoopBackend` and the system runs identically to today.

### Commit 3 — Tier 1 wire-up (lessons memory)

- `lessons_summarizer` agent's existing output gets a write hook: at end of each run, write one `MemoryRecord(kind="lesson", summary=lesson_text, organism=hyp_input.organism, …)` per distinct lesson surfaced
- New `view.prior_lessons` field on `SpecialistView` and `SynthesizerView`
- View-builder fetches top-5 by `MemoryQuery(kind="lesson", organism=hyp_input.organism, semantic_query=topic.summary)`
- Synthesizer + critic prompts get a `[PRIOR LESSONS]` block (mirror of the existing `[CROSS-TOPIC LESSONS]` block)
- New invariant on synthesizer: "prior lessons are *priors*, not ground truth. Bundle evidence overrides them. If a prior contradicts the bundle, surface the contradiction."
- `tests/unit/memory/test_lessons_wireup.py`: with StubBackend, the synthesizer prompt contains the seeded lesson; with NoopBackend, prompt is byte-identical to today (REGRESSION).

### Commit 4 — Provenance + supersession + observability

- Every memory write carries `provenance.{run_id, generation_timestamp, lesson_id_in_global_md}` so we can trace any retrieved lesson back to its source run
- A `/api/memory/records` GET endpoint (admin / dev only) so we can inspect what's in the store without psql
- Counters in token-report-style summary at end of every run: `memory_writes`, `memory_fetches`, `memory_records_returned`
- `tests/unit/memory/test_provenance.py`: every retrieved record carries traceable provenance

**End of Phase 1:** Tier 1 fully wired, swappable, observable. **No** Tier 2/3/4/5 yet — those are separate plans.

---

## Risk register

| Risk | Mitigation |
|---|---|
| Memory contradicts bundle evidence and the synthesizer trusts memory | Add a `[memory-axis]` critic rule that fires when a hypothesis cites a prior lesson but the cited finding is from a *different* run / strain. Memory is prior, never ground truth. |
| Strain drift makes old memories actively misleading | Tier 4 (KPI priors) handles this by maintaining rolling stats; Tiers 1–3 add `superseded_by` chain so a corrected lesson can override the original. |
| Embedding similarity returns superficially-related but irrelevant lessons | Phase 1 retrieval filters by `organism` + `process_family` *first*, embedding only ranks within that subset. Cross-strain retrieval is opt-in via explicit `MemoryQuery` flag. |
| Postgres + pgvector adds infra burden | Already a dependency for ingest. pgvector is a one-line extension on the same instance. |
| Provenance breaks if `global.md` location changes | Provenance stores `run_id` and `generation_timestamp` (immutable IDs), not file paths. |
| LLM-emitted lessons hallucinate facts that get persisted | Phase 1 only persists `lessons_summarizer` output, which is constrained by its prompt to summarize *retries within the run* — a narrow, bounded scope. Tier 2 (ratified hypotheses) is structurally constrained by judge approval. |

## Success criteria for Phase 1

- A run on a strain we've previously run shows different `[PRIOR LESSONS]` content in the synthesizer prompt vs. a never-seen-before strain. Verifiable by reading `global.md`.
- The memory store contains ≥1 lesson per completed run after a week of usage.
- Token-report counters show `memory_fetches > 0` on every run with a known organism, `= 0` on unknown-organism runs.
- Full unit suite green; integration test for Postgres backend passes against a real container.
- **Hard regression invariant:** with `NoopBackend` (default), every byte of prompt content + every test outcome is identical to current behavior. Memory is opt-in.

## What gets deferred (and the explicit triggers)

| Tier | Trigger to start |
|---|---|
| Tier 2 (ratified hypotheses) | Phase 1 has run for ≥2 weeks and we observe lessons-only memory missing the *substantive* prior conclusions; OR a process scientist explicitly says "the system should know we already concluded X about this strain." |
| Tier 3 (rejected hypotheses) | Bundled with Tier 2 — same adapter primitives, same write point. |
| Tier 4 (KPI priors) | We have ≥15 runs on a single (organism, process_family) pair. Below that, statistics aren't meaningful. |
| Tier 5 (corrections) | First time a user submits a HITL correction that the next run on the same strain visibly fails to apply. |
| Cloud backend (Synap or other) | Postgres relevance ceiling becomes the bottleneck, OR we need cross-organism entity resolution that pgvector + organism-filter can't deliver. |
| Trajectory-shape similarity (DTW / PAA) | Specialists need to retrieve *runs with similarly-shaped μ(t) curves*, not just runs with similar metadata. Real engineering; not Phase 2. |

## Open questions to resolve before Phase 1 commits

1. **Embedding provider.** Use Gemini's text-embedding API (already have the key + client) or pull in a small local model (`sentence-transformers/all-MiniLM-L6-v2`)? Cost says local; latency + quality say Gemini. Default proposal: Gemini, with the embedding call abstracted so a swap is one file.
2. **Where the Postgres table lives.** Same database as ingest, or separate `memory_db`? Default proposal: same DB, separate schema (`memory.records`).
3. **Per-deployment isolation.** When Lemnisca runs this for multiple customers, do we want a `tenant_id` partition on `memory_records`? Default proposal: yes, add `tenant_id` from day one, default `"default"` for now.
4. **What goes into `lesson_text`.** The summarizer currently produces free-form 50–60 token blobs. For embedding quality, do we want it to also emit a short `tags: [list]` field for filtering? Default proposal: yes — add a `tags` field to lessons-summarizer output schema, persist tags in the memory record.

---

## What the user gets out of this

After Phase 1 only:
- Re-running a bundle on a strain we've run before → specialists see prior lessons in their prompt context → fewer repeat mistakes, faster convergence.
- Operator corrections start to compound into durable knowledge in Tier 5.
- The system stops being "a single-shot reasoner" and starts being "a reasoner that remembers what its peers concluded."

Not delivered yet (deferred to later phases):
- Strain-conditional anomaly detection (Tier 4).
- Cross-bundle hypothesis retrieval (Tier 2).
- Entity resolution ("yeast" = "S. cerevisiae" = "Sacc") — Postgres exact-match only in Phase 1; cloud adapter or string normalisation handles the long tail later.

The order is deliberate: ship the smallest verifiable change in system behavior first, then earn each subsequent tier with usage data.
