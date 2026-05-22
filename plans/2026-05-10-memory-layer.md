# Memory Layer — Plan

**Date:** 2026-05-10
**Branch:** `memory-layer-plan` (off `followup-context`)
**Status:** Plan only — no code yet. Phase 1 is the first implementation step; everything past it requires explicit go-ahead.
**Eng review:** 2026-05-10. 10 decisions resolved (D1–D10). See "Eng-review decisions" at the end of this doc.
**Backend revision (2026-05-10, post-review):** Phase 1 backend is **Synap (managed)**, not Postgres+pgvector. The `MemoryBackend` Protocol is unchanged; `SynapBackend` replaces `PostgresBackend` as the Phase 1 production adapter. See "Backend revision" section near the end of this doc for what changed and why.

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

**What:** Persist every `lessons_summarizer` output keyed by `(process_family, organism, embedding)`. **Primary retrieval key is `process_family`** (D8 — closed-vocab, registry-validated; avoids string variation on free-form `organism`). Organism is a secondary re-ranker. At view-build time, retrieve top-K relevant prior lessons and inject as `view.cross_run_lessons` (D5 — renamed from `cross_topic_lessons`, which becomes `view.in_run_lessons`).

**Why this first:** highest impact-per-hour. We're already producing the data; we're just throwing it away. The hypothesis stage already has an in-run `cross_topic_lessons` slot — same prompt shape, sourced from history instead of in-run state. **One change in input, immediate change in output.**

**What gets written:** `LessonRecord(lesson_id, lesson_text, process_family, organism, hyp_id, run_id, embedding, embedding_provider, embedding_model, embedding_version, tenant_id, created_at)`. Each lesson carries a stable `lesson_id` minted at emission time (D2).

**Write timing (D6 + D10):** lessons are buffered in-memory during the run. On HITL pause, the buffer is serialized to `<bundle>/lesson_buffer.json` and reloaded at `resume_stage`. Persistence to the memory store happens **only when the run reaches a clean `exit_reason` (`consensus_reached` or `no_topics_left`)**. `budget_exhausted`, `max_turns_reached`, and exception paths skip the write — failed-run lessons never pollute the store.

**What gets read:** at the start of every hypothesis run, fetch top 5 lessons by cosine similarity over the run's seed-topic summaries, **filtered first by `process_family` exact match** (D8). Topic embeddings are cached per run so each unique `topic.summary` is embedded once, not per turn (4.1). Inject into specialist + synthesizer + critic views as `view.cross_run_lessons`.

**Effort:** ~1.5 days (was ~1 day; +0.5 day for lesson_id refactor + buffer file plumbing + eval harness).

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
from types import MappingProxyType
from typing import Mapping, Protocol, Literal, Any

MemoryKind = Literal[
    "lesson",
    "ratified_hypothesis",
    "rejected_hypothesis",
    "correction",
]

@dataclass(frozen=True)
class MemoryRecord:
    memory_id: str             # for lessons: stable lesson_id minted at emission (D2)
    kind: MemoryKind
    summary: str               # the human-readable blob; embedding source
    process_family: str | None # primary retrieval key for kind="lesson" (D8)
    organism: str | None       # secondary re-ranker; may vary across runs
    tenant_id: str             # required; routes to memory_tenant_<id>.records schema (D3)
    affected_variables: tuple[str, ...]
    finding_classes: tuple[str, ...]   # e.g. ("B10_overflow", "A14_do_margin")
    confidence: float | None
    provenance: Mapping[str, Any]      # MappingProxyType at construction; immutable
    embedding_provider: str    # "gemini" | "openai" | "local" (D4)
    embedding_model: str       # "text-embedding-004" etc.
    embedding_version: str     # provider-specific version tag
    created_at: str            # ISO8601 UTC
    superseded_by: str | None = None

@dataclass(frozen=True)
class MemoryQuery:
    kind: MemoryKind | None  # None = all kinds
    tenant_id: str           # required; cross-tenant retrieval is impossible by design
    process_family: str | None  # PRIMARY KEY for kind="lesson"; raises ValueError if None (D7)
    organism: str | None     # secondary filter / re-ranker
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

**Invariant (D7):** `fetch(MemoryQuery(kind="lesson", process_family=None, ...))` raises `ValueError`. Cross-strain retrieval is opt-in via a separate explicit method on the backend (added later, not Phase 1).

**Tenant isolation (D3):** every write/fetch routes to a per-tenant Postgres schema (`memory_tenant_<id>.records`). The Postgres adapter sets `search_path` per request. Cross-tenant retrieval is structurally impossible — a query without the tenant prefix targets a non-existent schema and fails noisily.

The Protocol is the contract. Adapters in Phase 1:

- `NoopBackend` — default. `fetch` returns `[]`. Lets the system run with memory disabled. **Tests use this by default.**
- `StubBackend` — in-memory dict. Lets tests assert behavior.
- `PostgresBackend` — the real one in Phase 1. Uses `pgvector` for embeddings.

Later adapters (not Phase 1):
- `SynapBackend` — cloud. Maps `MemoryKind` to their typed memory categories.
- `Mem0Backend` / `LettaBackend` — if we ever need them.

---

## Phase 1 PR plan (the only thing this plan commits to)

Four commits on a child branch when this gets approved. The eval harness (D9) is the gate before merge.

**Parallelization:** Commit 1 blocks both downstream lanes. Then Commits 2 and 3 can land in parallel worktrees (Commit 3 dev-tests against `StubBackend`, doesn't need Postgres). Commit 4 is sequential after both. Both Commit 2 and Commit 3 touch `src/fermdocs_memory/__init__.py` re-exports — coordinate to avoid trivial conflicts.

### Commit 1 — Protocol + Noop + Stub backends + tests

- New package `src/fermdocs_memory/`
- `base.py`: `MemoryRecord`, `MemoryQuery`, `MemoryBackend` Protocol, `MemoryKind` enum. Provenance stored as `MappingProxyType` at construction (immutable by convention).
- `noop.py`: `write` no-op; `fetch` returns `[]`; `supersede` no-op
- `stub.py`: in-memory dict; supports `write`/`fetch`/`supersede` with the same filtering logic as the Postgres adapter (process_family + organism + kind + top_k); semantic_query falls back to substring match when no embeddings are wired
- **Tests:**
  - `tests/unit/memory/test_record_schema.py` — frozen, validation, provenance immutability
  - `tests/unit/memory/test_query_validation.py` — D7: `fetch(kind="lesson", process_family=None)` raises `ValueError`
  - `tests/unit/memory/test_noop_backend.py` — Noop returns empty; supersede no-op
  - `tests/unit/memory/test_stub_backend.py` — write/fetch/supersede + filters + top_k boundary
- Wire `NoopBackend` as default into runner + characterize entry points behind `memory: MemoryBackend = NoopBackend()` parameter

**Hard rule:** every existing test must pass unchanged. Default `NoopBackend` means zero behavior change.

### Commit 2 — Postgres adapter

- `postgres.py`: SQLAlchemy model `MemoryRow` + `PostgresBackend` impl
- **Schema-per-tenant (D3):** writes/fetches set `search_path = memory_tenant_<id>` per request. New tenants require an explicit migration step (out of scope for Phase 1; manual `CREATE SCHEMA` for the default tenant is enough to ship).
- New table per schema: `memory_records(memory_id, kind, summary, process_family, organism, tenant_id, affected_variables_jsonb, finding_classes_jsonb, confidence, provenance_jsonb, embedding_vector pgvector(768), embedding_provider varchar(32), embedding_model varchar(64), embedding_version varchar(16), created_at, superseded_by)`
- **Indexes:**
  - btree on `(kind, process_family, organism)`
  - GIN on `affected_variables_jsonb`
  - **HNSW** on `embedding_vector` (`m=16, ef_construction=64`) for Phase 1 — correct for small tables. Switch to ivfflat when `n_rows > 10_000` (4.2).
- `tests/integration/memory/test_postgres_backend.py` against a real Postgres + pgvector container (same fixture pattern as ingest). Tests cover write happy path, embedding-API error path, duplicate `memory_id`, fewer-than-top_k results, embedding metadata triple round-trip.
- `tests/integration/memory/test_postgres_tenant_schema.py` — D3 enforcement: query against a non-existent schema fails noisily; queries with the wrong tenant return 0 rows.
- `EMBEDDING_PROVIDER` plumbing: helper that calls Gemini text-embedding via existing `_client` so we don't add a new vendor dep. **Topic embeddings cached per run** so each unique `topic.summary` is embedded once (4.1).

**Hard rule:** Postgres adapter is opt-in. `DATABASE_URL` empty → we instantiate `NoopBackend` and the system runs identically to today.

### Commit 3 — Tier 1 wire-up (lessons memory)

- **`LessonsSummarizedEvent` shape change (D2):** event now carries `lessons: list[Lesson]` where each `Lesson` has `lesson_id`, `text`, `tags`. Backwards-compatible read of legacy single-string `digest` field for existing `global.md` replay.
- **`lessons_summarizer` agent:** mints `lesson_id` (`L-<run_id_short>-<NNNN>`) at emission time and emits structured list. Adds `tags: list[str]` per lesson for filtering (resolves Open Question #4).
- **Buffer + write timing (D6 + D10):** lessons are buffered in `RunnerState`. Serialized to `<bundle>/lesson_buffer.json` at HITL pause; reloaded at `resume_stage`. Persisted to memory store only when `exit_reason in {consensus_reached, no_topics_left}`.
- **View schema rename (D5):** `cross_topic_lessons` → `in_run_lessons` on `SpecialistView`, `SynthesizerView`, `CriticView`. New field `cross_run_lessons: LessonsDigest | None` on the same three views.
- **Projector (3 sites):** `_build_view` populates `in_run_lessons` from existing `latest_lessons_digest` (rename only, no logic change). New `cross_run_lessons` populated via `memory.fetch(MemoryQuery(kind="lesson", process_family=hyp_input.process_family, semantic_query=topic.summary, top_k=5, tenant_id=...))`.
- **Synthesizer prompt:** new `[CROSS-RUN LESSONS]` block + new invariant: "cross-run lessons are *priors*, not ground truth. Bundle evidence overrides them. If a prior contradicts the bundle, surface the contradiction."
- **Critic prompt:** new `[memory-axis]` rejection rule for hypotheses that cite a prior lesson but the cited evidence is from a different run/strain.
- **Tests (REGRESSION-CRITICAL marked ⚠️):**
  - ⚠️ `tests/unit/hypothesis/memory/test_lessons_event_shape.py` — old + new `LessonsSummarizedEvent` round-trip; `read_events` from existing global.md files still parses
  - ⚠️ `tests/unit/hypothesis/memory/test_runner_write_timing.py` — `NoopBackend` produces byte-identical prompts to today; buffer + write only on clean exit; skip on `budget_exhausted`/failure
  - ⚠️ `tests/unit/hypothesis/memory/test_projector_field_rename.py` — `in_run_lessons` feedback loop unchanged from prior `cross_topic_lessons` behavior
  - `tests/unit/hypothesis/memory/test_projector_cross_run_lessons.py` — populates correctly via StubBackend
  - `tests/unit/hypothesis/memory/test_synthesizer_prior_lessons_block.py` — prompt rendering, empty-handling
  - `tests/unit/hypothesis/memory/test_critic_memory_axis.py` — fires when hypothesis cites missing/wrong-run prior

### Commit 4 — Provenance + supersession + observability + failure-mode hardening

- Every memory write carries `provenance.{run_id, hyp_id, generation_timestamp, lesson_id, source_event_offset}` so any retrieved record traces back to a specific event in `global.md`.
- `/api/memory/records` GET endpoint (admin only — auth check) for inspection without psql. Filters: `kind`, `process_family`, `organism`, `limit`. Sort: `created_at DESC`.
- Counters in token-report-style summary at end of every run: `memory_writes`, `memory_fetches`, `memory_records_returned`, `memory_embedding_calls`, `memory_embedding_failures`.
- **Failure-mode handling (review section 4):**
  - Embedding-API failure during write → log warning, increment `memory_embedding_failures`, retry once, then skip the lesson with a clear log line (don't fail the whole run finalization)
  - Lesson-buffer JSON corruption at resume → log warning, drop the buffered lessons, continue resume (degraded mode, not a failure)
  - Missing tenant schema on first write → return a clear error message naming the missing schema and the migration command, not a 500
- **Tests:**
  - `tests/unit/memory/test_provenance.py` — every retrieved record carries traceable provenance
  - `tests/integration/api/test_memory_records_endpoint.py` — auth, filters, pagination
  - `tests/unit/memory/test_failure_modes.py` — embedding API failure, buffer corruption, missing schema

### Eval harness (gate before merge — D9)

- `tests/eval/eval_synthesizer_priors.py` — 2–3 fixed fixture bundles (carotenoid + penicillin synthetic) run twice: with `NoopBackend` (no priors) vs with `StubBackend` seeded with realistic priors. Compare accept rate, critic rejection rate, mean confidence. Many-run averaging to handle LLM nondeterminism.
- `tests/eval/eval_critic_memory_axis.py` — does the `[memory-axis]` rule fire when a synthesizer is induced (via planted prior) to cite a prior lesson that doesn't exist in retrieval results?
- **Gate:** Phase 1 only ships if eval shows **no regression** on existing fixtures (NoopBackend path is byte-identical) AND the critic memory-axis rule fires on planted bad-prior citations.

**End of Phase 1:** Tier 1 fully wired, swappable, observable, eval-gated. **No** Tier 2/3/4/5 yet — those are separate plans.

---

## Risk register

| Risk | Mitigation |
|---|---|
| Memory contradicts bundle evidence and the synthesizer trusts memory | `[memory-axis]` critic rule fires when a hypothesis cites a prior lesson but the cited finding is from a different run/strain. Memory is prior, never ground truth. |
| Strain drift makes old memories actively misleading | Tier 4 (KPI priors) handles this with rolling stats; Tiers 1–3 use the `superseded_by` chain so a corrected lesson can override. |
| Embedding similarity returns superficially-related but irrelevant lessons | Phase 1 filters by `process_family` first (D8 — closed-vocab), embedding ranks only within that subset. Cross-family retrieval is structurally impossible without an explicit opt-in. |
| Cross-tenant data leak (regulated-biotech concern) | Schema-per-tenant isolation (D3). Database-enforced; queries without a tenant prefix target a non-existent schema and fail noisily. |
| Postgres + pgvector adds infra burden | Postgres is already a dependency for ingest. pgvector is a one-line extension on the same instance. |
| Embedding-provider lock-in | `embedding_provider/model/version` columns (D4) let us migrate by dual-writing during cutover and filtering retrieval to the active triple. |
| Bad lessons from failed runs pollute the store | Buffer + write-on-clean-exit (D6). Failed runs (`budget_exhausted`, `max_turns_reached`, exceptions) don't write. Buffer survives HITL pause via `<bundle>/lesson_buffer.json` (D10). |
| Silent cross-strain retrieval bug | `MemoryQuery.fetch(kind="lesson", process_family=None)` raises `ValueError` (D7). Cross-strain retrieval requires an explicit separate method. |
| Provenance breaks if `global.md` location changes | Provenance stores immutable IDs (`run_id`, `lesson_id`, `generation_timestamp`, `source_event_offset`), not file paths. |
| Embedding API outage during write | Retry once, then skip the lesson with a logged warning + counter increment. Run finalization continues. |
| LLM-emitted lessons hallucinate facts that get persisted | Phase 1 persists only `lessons_summarizer` output, constrained by its prompt to summarize retries within the run — narrow, bounded. Tier 2 records are constrained by judge approval. |
| LLM hallucinates a citation to a prior lesson that wasn't in the retrieved set | `[memory-axis]` critic rule catches; eval harness (D9) verifies the rule fires on planted bad citations. |

## Success criteria for Phase 1

- A run on a `process_family` we've previously seen shows different `[CROSS-RUN LESSONS]` content in the synthesizer prompt vs. a never-seen-before family. Verifiable by reading `global.md`. (D8: keying on family, not free-form organism string, is what makes this criterion actually achievable in Phase 1.)
- The memory store contains ≥1 lesson per cleanly-finished run after a week of usage.
- Token-report counters show `memory_fetches > 0` on every run with a known `process_family`, `= 0` on unknown-family runs.
- Full unit suite green; integration test for Postgres backend passes against a real container; tenant-schema isolation test passes.
- Eval harness (D9) shows: (a) `NoopBackend` path is byte-identical to today on existing fixture bundles; (b) `[memory-axis]` critic rule fires on planted bad-citation cases.
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

## Open questions resolved (eng review 2026-05-10)

1. **Embedding provider** — Gemini text-embedding-004 (768-dim) via existing client. The embedding call is abstracted so a swap is one adapter file. **Provider/model/version stored as columns (D4)** so a future swap doesn't require a full re-embed.
2. **Where the Postgres table lives** — same DB as ingest, separate **per-tenant schemas** (D3): `memory_tenant_<id>.records`. Schema isolation, not column partition.
3. **Per-deployment isolation** — schema-per-tenant from day one (D3). Default tenant `"default"` for the single-customer Phase 1 deploy.
4. **What goes into `lesson_text`** — `lessons_summarizer` emits structured `Lesson(lesson_id, text, tags)` entries. `tags` field is persisted on the memory record for filtering. (D2 + Open Q4 resolution rolled together.)

## TODOs (filed for later, not blocking Phase 1)

- **Cross-strain retrieval method.** When Phase 2 needs cross-family lessons (e.g. for general fed-batch wisdom that applies across organisms), add an explicit `fetch_cross_family(query)` to `MemoryBackend`. Until then, omitting it prevents silent cross-family leaks.
- **Memory record TTL / decay.** No automatic expiry today. If retrieval starts surfacing 2-year-old irrelevant lessons after deployment matures, add a `decay_after_days` field and apply a recency penalty in ranking.
- **Tenant schema provisioning automation.** Phase 1 ships with manual `CREATE SCHEMA` for the default tenant. Real multi-tenant ops needs a provisioning script.

---

## Eng-review decisions (D1–D10, 2026-05-10)

| # | Decision | Choice |
|---|---|---|
| D1 | Phase 1 scope | Plan as-written: full Protocol + Postgres + pgvector |
| D2 | Lesson identification | `lesson_id` minted at emission; `LessonsSummarizedEvent` carries structured list |
| D3 | Tenant isolation | Schema-per-tenant (`memory_tenant_<id>.records`) |
| D4 | Embedding-provider lock-in | 768-dim column + `embedding_provider/model/version` metadata |
| D5 | Field naming | `cross_topic_lessons` → `in_run_lessons`; new `cross_run_lessons` field |
| D6 | Write timing | Buffer in `RunnerState`; persist only on clean `exit_reason` |
| D7 | Cross-strain guard | `fetch(kind="lesson", process_family=None)` raises `ValueError` |
| D8 | Retrieval primary key | `process_family` (closed vocab), not free-form `organism` |
| D9 | Eval harness | Build small harness as Phase 1 deliverable; gates merge |
| D10 | HITL resume composition | Buffer serialized to `<bundle>/lesson_buffer.json` |

Plus three "must specify, no decision needed" clarifications:
- **4.1** Topic embeddings cached per run (compute once per unique `topic.summary`)
- **4.2** HNSW index for Phase 1 (`m=16, ef_construction=64`); switch to ivfflat at ~10K rows
- **Failure modes (3 critical gaps)** Embedding API failure during write, lesson-buffer corruption at resume, missing tenant schema — all explicit error paths in Commit 4

---

## Backend revision (2026-05-10, post-review)

The original plan committed to Postgres + pgvector as Phase 1's production backend. After reviewing Synap's docs and capabilities, **Phase 1 ships against Synap (managed)** instead. The `MemoryBackend` Protocol is unchanged; the swap is a single adapter file.

### What changed

| Plan item | Before (Postgres+pgvector) | After (Synap) |
|---|---|---|
| Production adapter (Commit 2) | `PostgresBackend` with pgvector | `SynapBackend` wrapping the Synap Python SDK |
| Embedding strategy | Self-managed Gemini text-embedding-004, 768-dim column, ivfflat/HNSW index | Synap manages embedding internally; we send `document` text, they handle vectors |
| Tenant isolation (D3) | Schema-per-tenant in Postgres | Synap scope chain (Customer = tenant, User = process_family) |
| Embedding-provider lock-in (D4) | `embedding_provider/model/version` columns | Synap's choice; opaque to us. Tradeoff accepted: we trust Synap's embedding quality. |
| Infra burden | New pgvector extension on existing DB | New cloud dependency; need `SYNAP_API_KEY_DEV` (and later `_PROD`) |
| Cost model | Self-hosted (DB compute already paid) | Synap usage-based ($12.45 starter credit on dev account) |
| Data residency | Self-hosted, on our infrastructure | Synap-hosted (US). Customers requiring residency get the alternate Postgres adapter when that path opens. |

### What stayed the same

- `MemoryBackend` Protocol contract (D7 raise, top_k, kind, etc.)
- `NoopBackend` and `StubBackend` (default-off + tests)
- Lesson lifecycle: `lesson_id` at emission (D2), buffer-then-persist on clean exit (D6), HITL pause via `<bundle>/lesson_buffer.json` (D10)
- View renaming: `cross_topic_lessons` → `in_run_lessons`; new `cross_run_lessons` (D5)
- Synthesizer + critic prompt changes: `[CROSS-RUN LESSONS]` block, `[memory-axis]` rule
- Eval harness as the merge gate (D9)
- Process-family-first retrieval (D8) — implemented as Synap's `User` scope key

### How fermdocs concepts map to Synap

| fermdocs | Synap |
|---|---|
| Lemnisca (deployment owner) | `Client` |
| Tenant (per-customer) | `Customer` (e.g. `customer_id="lemnisca-internal"`) |
| `process_family` (closed-vocab key) | `User` (e.g. `user_id="yeast_intracellular_product_fedbatch"`) |
| Per-lesson record | one `memories.create` call with `document_type="agent-lesson"` |
| Lesson metadata (run_id, hyp_id, lesson_id, organism, tags) | `metadata={...}` on the create call |
| Retrieval at view-build | `sdk.user.context.fetch(user_id=process_family, search_query=[topic.summary], types=[…])` |

The unusual choice: **`User = process_family`**, not a human user. Justification in the use-case markdown (`plans/synap_setup/fermdocs-dev-usecase.md`). If Synap's billing/analytics gets weird because of low cardinality (~6 process families), we revisit and put process_family in metadata with a single `system` user instead. Spike-testable.

### Synap instance setup

- **`fermdocs-dev`** — provisioned 2026-05-10. API key in `SYNAP_API_KEY_DEV`. Used by the eval harness, dev runs, debugging.
- **`fermdocs-prod`** — not yet provisioned. Will be created when Phase 1 graduates from dev. API key will be `SYNAP_API_KEY_PROD`.
- Use-case markdown uploaded: `plans/synap_setup/fermdocs-dev-usecase.md`.

### Risks specific to managed backend

| Risk | Mitigation |
|---|---|
| Synap outage blocks runs | `SynapBackend.fetch` failures fall back to empty result + log a warning; runs continue with no priors. Matches "memory is opt-in" invariant. |
| Synap latency on the hot path (15ms P50 claimed; what's P95?) | Topic embedding cache (4.1) means one `fetch` per unique topic per run. Worst case ~3 calls per run, not per turn. |
| Vendor lock at adapter boundary | Protocol is the contract. Swap to Postgres is one file's worth of work; we have the spec ready. |
| Data residency constraint hits a customer | `PostgresBackend` is filed as the alternate adapter. Same Protocol, alternate runtime. |
| Synap's typed memory categories don't fit our shape | Phase 1 sends everything as a single document type with our metadata. We don't rely on their facts/preferences/episodes/emotions split. |

### Phase 1 PR plan, Synap-revised

Same four-commit shape; Commit 2 is the one that changes:

- **Commit 1** — Protocol + Noop + Stub. **Unchanged from original plan.**
- **Commit 2 — `SynapBackend` adapter.** Wraps the Synap Python SDK. Instance + API key from env. Maps `MemoryRecord` → `sdk.memories.create(document, document_type, customer_id, user_id, metadata)`. Maps `MemoryQuery` → `sdk.user.context.fetch(...)`. Tests against the dev instance via integration test (gated on `SYNAP_API_KEY_DEV` env var; skipped in CI without it).
- **Commit 3 — Tier 1 wire-up.** **Unchanged from original plan.**
- **Commit 4 — Provenance + observability.** **Unchanged from original plan.**
- **Eval harness.** **Unchanged from original plan.**

---

## What the user gets out of this

After Phase 1 only:
- Re-running a bundle on a process family we've already seen → specialists see prior lessons in their prompt context → fewer repeat mistakes, faster convergence.
- The eval harness is reusable for every future prompt change in this stage.
- The system stops being "a single-shot reasoner" and starts being "a reasoner that remembers what its peers concluded."

Not delivered yet (deferred to later phases):
- Strain-conditional anomaly detection (Tier 4).
- Cross-bundle hypothesis retrieval (Tier 2).
- Human-correction memory (Tier 5).
- Entity resolution beyond `process_family` keying — closed-vocab matching only in Phase 1; cloud adapter or alias normalization handles the long tail later.

The order is deliberate: ship the smallest verifiable change in system behavior first, then earn each subsequent tier with usage data.
