# Synap Memory Integration

## What it does

Synap is a cloud memory service. When fermdocs analyzes a fermentation batch and
learns something useful, it saves that lesson to Synap. Next time it sees a batch
from the same strain, it reads past lessons first so the agents don't start from
scratch.

Memory is **append-only** in Phase 1. No updates, no deletes, no corrections.
If Synap goes down, nothing breaks — agents proceed without priors.

---

## File map

### The contract (what gets stored, how you ask for it)

| File | What it does |
|------|-------------|
| `src/fermdocs_memory/base.py` | Defines `MemoryRecord` (one stored lesson), `MemoryQuery` (one retrieval request), and the `MemoryBackend` Protocol (write / fetch / supersede). All backends implement this. |
| `src/fermdocs_memory/__init__.py` | Package surface. Exports the Protocol + schemas. Has a lazy `_build_synap_backend()` factory so nobody imports `synap.py` directly. |

### The backends (who actually stores it)

| File | When it's used |
|------|---------------|
| `src/fermdocs_memory/noop.py` | Default. All writes silently dropped, fetches return nothing. Memory is off. |
| `src/fermdocs_memory/stub.py` | Unit tests. In-memory dict with substring matching. |
| `src/fermdocs_memory/synap.py` | Production. Wraps the `maximem_synap` SDK. Calls `sdk.memories.create()` to write, `sdk.user.context.fetch()` to read. |

### Where lessons come from

| File | What it does |
|------|-------------|
| `src/fermdocs_hypothesis/schema.py` | `Lesson` dataclass (lesson_id + text + tags) and `LessonsDigest` (list of lessons + prompt-ready digest string). |
| `src/fermdocs_hypothesis/agents/lessons_summarizer.py` | Takes the last 20 critic complaints, calls Gemini, returns 3-5 distilled lessons with stable IDs like `L-abc123-0001`. |
| `src/fermdocs_hypothesis/events.py` | `LessonsSummarizedEvent` — carries lessons in the event log until run exit triggers persistence. |

### Where memory gets written and read

| File | What it does |
|------|-------------|
| `src/fermdocs_hypothesis/runner.py` | `_persist_lessons_to_memory()` (line 796) — THE write site. `_fetch_cross_run_lessons()` (line 861) — THE read site. |
| `src/fermdocs_hypothesis/live_hooks.py` | `LiveHooks` wires memory into every agent's view. `_fetch_cross_run()` (line 104) caches the fetch per topic so 4 agents don't each trigger a separate Synap call. |

### API entry point

| File | What it does |
|------|-------------|
| `apps/api/fermdocs_api/runner_pipeline.py` | `_build_memory_backend()` (line 732) reads `FERMDOCS_MEMORY` env var, constructs the backend, passes it to `LiveHooks` and `run_stage()`. |

### Tests

| File | What it tests |
|------|-------------|
| `tests/unit/memory/test_synap_backend_unit.py` | SynapBackend with SDK fully mocked. Write mapping, fetch flattening, client-side filtering, lazy init, failure resilience. |
| `tests/unit/memory/test_record_schema.py` | MemoryRecord is frozen, provenance is immutable. |
| `tests/unit/memory/test_query_validation.py` | D7 invariant: lesson queries without process_family raise ValueError. |
| `tests/unit/memory/test_noop_backend.py` | Writes drop, fetches return empty. |
| `tests/unit/memory/test_stub_backend.py` | In-memory round-trip, filtering. |
| `tests/unit/hypothesis/memory/test_runner_memory_wireup.py` | Persist on clean exit, skip on failure, skip without process_family, absorb backend errors. |
| `tests/unit/hypothesis/memory/test_lesson_id_minting.py` | Lesson IDs are stable (L-RUN-NNNN). |
| `tests/integration/memory/test_synap_backend_live.py` | Hits real Synap. Skipped unless `SYNAP_API_KEY` is set. |

### Spikes (exploration before building)

| File | What it tested |
|------|---------------|
| `spike/synap_one_write.py` | "Can we write one document to Synap?" |
| `spike/synap_minimal_fetch.py` | "Which fetch endpoint works? user.context? customer.context?" |
| `spike/synap_extraction_test.py` | "If we write 3 lesson digests, what does Synap's extraction return?" |

---

## Scenario 1: Writing lessons (run completes successfully)

**When:** A hypothesis run finishes with `consensus_reached` or `no_topics_left`.

**What happens:**

1. During the debate, the critic keeps rejecting hypotheses with reasons like
   "you didn't account for the pigment loss" or "the O2 model is wrong."

2. After enough rejections, the **lessons summarizer** agent is called. It reads
   the last 20 critic reasons and asks Gemini: "what are the recurring patterns?"
   Gemini returns 3-5 distilled lessons. Each gets a stable ID like `L-abc123-0001`.

3. These lessons are stored in the event log as `LessonsSummarizedEvent`.

4. **At run exit**, `_persist_lessons_to_memory()` in `runner.py:796` fires. It
   checks three gates:

   ```
   Is this a clean exit?           → NO  → stop, write nothing
   Does this run have a process_family? → NO  → stop, write nothing
   Are there lessons in the event log?  → NO  → stop, write nothing
   ```

   If all three pass, it loops through each lesson and calls `memory.write()`.

5. `SynapBackend.write()` calls `sdk.memories.create()` with:
   - `document` = the lesson text
   - `document_id` = the lesson ID (idempotency key)
   - `user_id` = the process_family (e.g. `"yeast_carotenoid_fedbatch"`)
   - `customer_id` = the tenant ID (company isolation)
   - `metadata` = provenance, organism, tags, variables

6. Synap receives this, embeds it, and indexes it asynchronously. We don't wait.

**If the run failed or was budget-exhausted:** nothing gets written. Bad runs
don't pollute the memory store.

**If Synap is down or the write fails:** a warning is logged. The run still
completes successfully. Memory is opt-in.

---

## Scenario 2: Fetching prior lessons (new run on the same strain)

**When:** A new run starts on the same `process_family` as a previous run that
wrote lessons.

**What happens:**

1. The orchestrator picks a topic, e.g. "anomalous pigment loss after 75h."

2. `LiveHooks` needs to build views for the specialist, synthesizer, critic, and
   judge. Before building any view, it calls `_fetch_cross_run()` in
   `live_hooks.py:104` with the topic summary.

3. This calls `_fetch_cross_run_lessons()` in `runner.py:861`, which calls
   `memory.fetch()` with:
   - `kind` = `"lesson"`
   - `process_family` = the current strain's family
   - `semantic_query` = the topic summary
   - `top_k` = 5

4. `SynapBackend.fetch()` calls `sdk.user.context.fetch()` with:
   - `user_id` = the process_family (scoping — only this strain's lessons)
   - `search_query` = the topic summary (Synap ranks by embedding similarity)
   - `max_results` = 5

5. Synap returns extracted items from 5 buckets (facts, episodes, preferences,
   emotions, temporal_events). Our adapter flattens these into `MemoryRecord`
   objects, applies any client-side filters (organism, variables), and caps at
   `top_k`.

6. The result is converted into a `LessonsDigest` and injected into **all four
   agent views** as `cross_run_lessons`. The fetch is cached per topic — one
   Synap round-trip serves all four agents.

7. Agents see something like:
   ```
   Prior lessons from previous runs:
     - Pigment loss after 80h tracks with nitrogen depletion, not oxygen limitation
     - Substrate consumption rate doubles after 60h — Monod kinetics underpredict
   ```

**If this is the first run for this strain:** Synap returns nothing. Agents
proceed without priors. Normal behavior.

**If Synap is down:** `fetch()` returns an empty list. Agents proceed without
priors. A warning is logged.

---

## Scenario 3: Conflicting lessons (known limitation)

**When:** Run 1 wrote a lesson that turns out to be wrong, and Run 3 wrote a
better one.

**What happens:** Both lessons exist in Synap. There is no update or delete.
When Run 4 fetches, Synap ranks by semantic similarity to the query. The more
relevant lesson usually wins, but there's no recency bias — an older wrong
lesson could outrank a newer correct one if the embeddings score it higher.

**What will fix it:** Tier 5 (corrections) — `supersede(old_id, new_id)` will
mark the old lesson as replaced. The Protocol method and schema fields
(`MemoryRecord.superseded_by`, `MemoryQuery.include_superseded`) already exist
but aren't implemented yet.

---

## Scoping model

Two isolation layers prevent data leaks:

| Layer | Maps to in Synap | What it isolates |
|-------|-----------------|-----------------|
| `tenant_id` | `customer_id` | Company A vs Company B. Hard infrastructure boundary. |
| `process_family` | `user_id` | Strain A vs Strain B within the same company. |

A third filter — `organism` — is applied **client-side** after Synap returns
results. Synap stores it in metadata but can't query on it.

---

## SDK endpoints called

| Endpoint | When | Where in code |
|----------|------|--------------|
| `sdk.initialize()` | Once, on first write or fetch | `synap.py:202` |
| `sdk.memories.create()` | On write (run exit) | `synap.py:271` |
| `sdk.user.context.fetch()` | On read (topic view build) | `synap.py:317` |
| `sdk.shutdown()` | Tests / explicit cleanup only | `synap.py:223` |

---

## Env vars

| Variable | What it does | Default |
|----------|-------------|---------|
| `FERMDOCS_MEMORY` | Backend selector. `"synap"` or `"noop"`. | `"noop"` (off) |
| `SYNAP_API_KEY` | SDK authentication. Required if `synap`. | — |
| `SYNAP_INSTANCE_ID` | Synap instance. | auto-resolved |
| `FERMDOCS_TENANT_ID` | Company isolation key. | `"default"` |
