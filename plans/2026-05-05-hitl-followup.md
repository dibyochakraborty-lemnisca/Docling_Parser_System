# PR-A2: Follow-up drive posture (HITL follow-up)

**Branch:** `hitl-followup` (off `caisc-hitl`)
**Status:** Plan locked — about to start commit 1.
**Companion plans:**
- `plans/2026-05-04-user-question-and-hitl.md` — the parent plan that PR-A
  built bias posture on. PR-A2 layers drive posture on top.
- `plans/2026-05-04-metric-catalog-and-toolkit.md` — finished; the catalog
  is the substrate every shape branch consumes.

**Estimated cost:** ~1000 LOC across 7 commits, mirroring PR-A's cadence.

---

## Context

PR-A on `caisc-hitl` (already merged-ready, pushed) ships **bias posture**:
the user types a question at upload, the question biases scoring + prompts,
the system reports `question_answered ∈ {yes, partial, insufficient_data}`.
First production runs (PDF carotenoid + IndPenSim) confirmed the wiring
works: catalog metric_ids surface in hypothesis summaries, classifier
extracts run/variable hints, synthesizer populates the badge field, critic
gets the `[question-axis]` axis.

The user explicitly chose "bias for v1, drive on follow-up" (D3 = A in
the PR-A planning conversation). PR-A2 builds the drive half.

User flow target:

```
1. User uploads bundle + types question — runs as PR-A bias posture
2. Run completes (status=done), UI shows H-0001/H-0003/H-0005 cards
3. User sees a "Follow up" textarea below the cards
4. Types: "Focus on RUN-0002's RQ peak — was it a sensor artifact?"
5. Clicks Submit. Status returns to HYPOTHESIZING (not INGESTING)
6. NEW UserQuestion classified, written to bundle as user_question.json
   (raised_by="user_followup")
7. Pipeline does NOT re-run ingest/characterize/diagnose — bundle frozen
8. Hypothesis stage runs again with shape-aware branching
9. New hypothesis cards append BELOW the originals, labeled "follow-up #1"
10. User can follow up again — appends as "follow-up #2"
```

Key: **the bundle is frozen**. The follow-up is a new question against
the same evidence, not a re-ingest. If user wants different data, they
upload again.

---

## Architecture in one diagram

```
                    ┌─────────────────────────────────┐
                    │ FRONTEND                        │
                    │  status=done → show "Follow up" │
                    │  textarea + Submit button       │
                    └────────┬────────────────────────┘
                             │ POST /runs/{run_id}/followup {question}
                             ▼
            ┌────────────────────────────────────────┐
            │ apps/api/fermdocs_api/runner_pipeline  │
            │  execute_followup_run()                │
            │   - load existing bundle (frozen)      │
            │   - classify question                  │
            │   - write user_question.json (overwrite│
            │     or append-versioned — see below)   │
            │   - dispatch by shape                  │
            └────────┬───────────────────────────────┘
                     │
       ┌─────────────┼─────────────┬──────────────┐
       ▼             ▼             ▼              ▼
   ┌────────┐   ┌──────────┐  ┌──────────┐   ┌────────────┐
   │scoping │   │mechanistic│  │comparative│   │open        │
   ├────────┤   ├──────────┤  ├──────────┤   ├────────────┤
   │filter  │   │bypass    │  │bypass    │   │PR-A bias   │
   │seed    │   │seeds;    │  │seeds;    │   │posture +   │
   │topics  │   │ONE topic │  │ONE topic │   │prior       │
   │by run/ │   │= user's  │  │= named   │   │hypotheses  │
   │var hint│   │mechanism │  │comparison│   │as context  │
   └───┬────┘   └────┬─────┘  └────┬─────┘   └────┬───────┘
       │             │             │              │
       ▼             ▼             ▼              ▼
   if filter    debate tests   debate          regular bias
   empties:     mechanism;     contrasts       debate cycle
   one Final-   FOR vs         the named       (existing
   Hyp w/       AGAINST        groups          PR-A path)
   answered=    evidence       across runs
   "insuff"
       │             │             │              │
       └──────┬──────┴─────────────┴──────────────┘
              ▼
       ┌──────────────────────────────────────────┐
       │ runner.run_stage produces HypothesisOutput│
       │ APPEND to existing run, not replace       │
       │ frontend renders chained cards w/ index   │
       └──────────────────────────────────────────┘
```

---

## Locked decisions (4)

These were discussed before plan writing; pinning here so post-compact
resume doesn't re-litigate.

### D1: Drive doesn't re-run characterize
Bundle is frozen. User can't introduce new data via follow-up — only
new questions. If they want to add data, they upload again. **Implication:**
follow-up is fast (~30-90s) because it skips ~5 minutes of pipeline work.

### D2: `mechanistic` branch produces ONE topic, not many
The user's mechanism becomes the topic. Specialists try to support OR
refute it — not survey the bundle. This is the biggest behavioral
departure from bias posture. **Implication:** synthesizer prompt has a
new sub-branch for mechanistic mode that frames the hypothesis as
"the user proposes X; the data is/isn't consistent with X because…"

### D3: `scoping` empty-filter case
If the user's scope ("RUN-0099 in the late phase") doesn't match any
topic (RUN-0099 doesn't exist OR no topic touches it), the system
emits ONE FinalHypothesis with `question_answered="insufficient_data"`
and `question_response_summary="Your scope didn't match any data in
this bundle. Available runs are RUN-0001, RUN-0002. Available variables
that could match the scope: ..."` Don't run a debate on nothing.

### D4: `open` shape on follow-up = bias posture + history context
Effectively a re-run. Cheap, useful for "actually I want a different
framing." Inject `accepted_hypotheses_so_far` (from previous run's
output) into the orchestrator/synthesizer view so the new pass doesn't
redo work the user already saw.

---

## Schema additions (commit 1)

### New: `Run.followup_index`
Tracks how many follow-ups this run has seen. UI shows
`H-0007 (follow-up #2)` when `followup_index=2`. Persists in
`Run` dataclass on the API server.

### New: `FollowupResult` in `apps/api/fermdocs_api/state.py`
```python
@dataclass
class FollowupResult:
    followup_index: int  # 1-indexed
    user_question_text: str
    output: HypothesisOutput  # produced by execute_followup_run
    created_at: datetime
```

`Run.followups: list[FollowupResult] = []` accumulates them.

### Modified: `FinalHypothesis.parent_hypothesis_ids: list[str] = []`
On follow-up runs, the synthesizer can reference prior hypotheses by
hyp_id. UI uses this to draw a "↑ refines H-0003" link when present.
Optional; mostly populated when `mechanistic` shape evolves a prior
hypothesis.

### Modified: `UserQuestion.raised_by`
Already supports `"user_followup"` — set by `execute_followup_run` when
classifying the follow-up question. Synthesizer prompt branches on this
field (commit 4).

---

## File-by-file plan

### Commit 1 — Schema (~80 LOC)
- `src/fermdocs_hypothesis/schema.py`: add `parent_hypothesis_ids: list[str] = []` to `FinalHypothesis`. **Add to `HypothesisFull` only if** the synthesizer needs it during the debate cycle (i.e. one synthesizer pass references hypotheses from a prior pass). If `FinalHypothesis` is the only consumer (the more likely case — parents are populated at projection-to-final time), leave `HypothesisFull` alone. Verify before writing the field.
- `apps/api/fermdocs_api/state.py`: add `FollowupResult` + `Run.followups` + `Run.followup_index`
- `tests/unit/test_followup_schema.py`: ~10 tests covering field defaults, list append, FollowupResult validation

### Commit 2 — Runner (`execute_followup_run`) (~210 LOC)
- `apps/api/fermdocs_api/runner_pipeline.py`:
  - **First** extract `_run_with_status_lifecycle(run, target_status, body_fn)` that captures the shared try/except/event-publish skeleton from `execute_run` (line 53) and `execute_resume` (line 131). Refactor those two onto it (no behavior change, pure structural).
  - **Then** add `execute_followup_run(store, run, question_text)` on top of it:
    1. Asserts `run.status == DONE` and `run.bundle_dir is not None`
    2. Sets `run.status = HYPOTHESIZING` via the lifecycle helper
    3. Reads bundle's run_ids/variables, classifies the question with `raised_by="user_followup"`
    4. Writes `<bundle>/user_question.json` (overwrite — only the most-recent question is in the file; full history lives in `Run.followups`)
    5. Calls `_run_hypothesis_blocking(bundle_dir, global_md, followup_mode=True)`
    6. Appends result to `run.followups`, increments `run.followup_index`
    7. Status back to DONE; publishes a follow-up-result event
- `_run_hypothesis_blocking` gains optional `followup_mode: bool = False` flag — passed through to `LiveHooks`
- **Persistence note:** `Run.followups` lives in the in-memory `RunStore` only (same as the rest of `Run`). API process restart wipes follow-up history along with the rest. Callers see the bundle's `user_question.json` (most-recent) on disk but no history. This matches today's `Run` lifetime — it's not a regression, just inherited. Listed in "NOT in scope" below.
- `tests/unit/test_followup_runner.py`: ~14 tests covering status transitions, **frozen-bundle invariant via `mock.patch` on `_run_ingest_blocking` / `_run_characterize_blocking` / `_run_diagnose_blocking` asserting `call_count == 0`** after `execute_followup_run`, question classification, append-not-replace, lifecycle-helper refactor doesn't change `execute_run`/`execute_resume` semantics

### Commit 3 — Shape-aware seed topic logic (~150 LOC)
- `src/fermdocs_hypothesis/seed_topic_extractor.py`: new func `extract_seed_topics_for_followup(diag, question, prior_output)` that branches on `question.shape`:
  - `scoping`: filter normal seed topics to ones overlapping `question.affected_runs/variables`. If empty, return single placeholder topic flagged `is_empty_scope=True`
  - `mechanistic`: ignore diag entirely; emit single synthetic SeedTopic with `summary=question.text`, `affected_variables=question.affected_variables`, `priority=1.0`, `source_type=USER_MECHANISM` (new enum value)
  - `comparative`: similar — single synthetic topic, `source_type=USER_COMPARISON`
  - `open`: full PR-A bias path with priority bumps
- `src/fermdocs_hypothesis/schema.py`: extend `TopicSourceType` enum with `USER_MECHANISM`, `USER_COMPARISON`
- `tests/unit/test_seed_topic_followup.py`: ~15 tests, one per shape × edge case (empty scope, all-runs scope, multi-variable mechanism, etc.)

### Commit 4 — Synthesizer drive-mode prompt (~100 LOC)
- `src/fermdocs_hypothesis/agents/synthesizer.py`: SYNTHESIZER_INVARIANTS gets a new conditional rule that fires when `view.user_question.raised_by == "user_followup"`. Different framing per shape:
  - `mechanistic`: "the user proposes mechanism X; structure your summary as 'X is/isn't consistent with the evidence because...'"
  - `comparative`: "structure your summary as a side-by-side: 'In Group A, X. In Group B, Y. The difference is...'"
  - `scoping` (when scope matched): "narrow your hypothesis strictly to the cited scope; do not extrapolate"
  - `open` + prior_output present: "don't re-derive what was already accepted; build on or contradict the prior hypotheses"
- Synthesizer reads from `view.user_question.raised_by` to pick the rule
- `tests/unit/hypothesis/test_synthesizer_followup_modes.py`: ~10 tests, one per shape, mock LLM verifying the right rule is in the prompt

### Commit 5 — Critic followup-axis (~50 LOC)
- `src/fermdocs_hypothesis/agents/critic.py`: CRITIC_INVARIANTS gets a `[followup-axis]` rule. When `view.user_question.raised_by == "user_followup"` AND the user_question.shape == "mechanistic":
  - If hypothesis just restates the user's mechanism without citing FOR/AGAINST evidence → reject `[followup-axis]: hypothesis restated the mechanism without testing it`
  - If hypothesis cites only confirming evidence and never considered AGAINST → reject `[followup-axis]: hypothesis ignored disconfirming evidence`
- For `comparative`: reject if hypothesis doesn't actually contrast the named groups
- `tests/unit/hypothesis/test_critic_followup_axis.py`: ~8 tests covering each rejection trigger + the no-rejection-when-honest invariant

### Commit 6 — API endpoint + state (~140 LOC)
- `apps/api/fermdocs_api/main.py`: new endpoint
  ```
  @app.post("/api/runs/{run_id}/followup")
  async def followup_run(run_id, body: FollowupRequest, background)
  ```
  with `FollowupRequest{question: str}`. Validates run exists, status is DONE, bundle_dir present and exists on disk. Returns 409 on bad status, 410 Gone on missing bundle (so frontend can disable the textarea). Spawns `execute_followup_run` as a background task.
- `apps/api/fermdocs_api/state.py`: `RunStore.add_followup(run_id, result)` accumulates results. `Run.bundle_followup_eligible` computed property returns `False` when `bundle_dir is None or not bundle_dir.exists()` — surfaced via GET `/api/runs/{run_id}`.
- API GET `/api/runs/{run_id}` returns `followups: list` and `bundle_followup_eligible: bool` so frontend can render conditionally.
- `tests/integration/test_followup_api.py`: ~12 tests covering happy path, status guard (rejects non-DONE → 409), 404 on missing run, 410 on GC'd bundle, frozen-bundle invariant, multiple follow-ups in sequence, `bundle_followup_eligible` flips correctly

### Commit 7 — Frontend + e2e (~150 LOC)
- `apps/web/src/app/runs/[id]/page.tsx`:
  - Show `Follow up` textarea ONLY when `status === "done" && bundle_followup_eligible === true`
  - When `status === "hypothesizing" && followup_index > 0`, badge reads "Running follow-up #{followup_index}" so user can distinguish original-running from follow-up-running
  - Submit button calls `POST /runs/{id}/followup`, immediately polls run detail after the response
  - Render each `run.followups[i]` block below the originals: header `"Follow-up #{i+1}: {question_text}"` + render its hypotheses with the same card layout (badge, summary, citations) as the original
  - Add visual divider between original cards and follow-up sections
- `apps/web/src/lib/api.ts`: new `submitFollowup(runId, question)` client + extended `RunDetails` type with `followups[]` and `bundle_followup_eligible`
- `tests/integration/test_followup_e2e_backcompat.py`: ~5 e2e tests including the back-compat invariant (a run with no follow-ups looks identical to today), bundle-GC'd hides textarea, status badge differentiates original vs follow-up runs

---

## Tests at every step

Mirror PR-A's discipline:

- Schema layer: field defaults, validation, frozen-ness
- Runner: status transitions, frozen-bundle (assert no characterize subprocess fired), append-not-replace, multiple follow-ups in sequence, empty-question rejection
- Shape branches: each shape has its own test file with happy-path + 2-3 edge cases
- Prompts: factored helpers + invariants strings present, mock-LLM tests verify the right framing fires
- API: status guard, 404s, malformed body, multiple follow-ups
- Frontend: type checks + back-compat (existing legacy runs still render)
- E2E back-compat: a run with no follow-ups looks identical to today's `caisc-hitl` HEAD output

Target: ~80 new tests across the 7 commits. Full suite finishing >1180
total. (Currently at 1101 on `caisc-hitl`.)

---

## What this PR-A2 explicitly does NOT do

- **Multi-step follow-up chains**. Follow-up of a follow-up is out of
  scope for v1. The data model supports it (`followups: list`) but the
  synthesizer's prompt logic only handles one level deep. If the user
  follows up twice, the second one ignores the first follow-up's output
  and just sees the original run's output.
- **Edit/retract previous hypotheses**. They stay frozen; new cards
  append. Hypothesis state is append-only.
- **Mid-debate human input during the follow-up itself**. That's PR-B
  HITL — operator-note injection mid-debate. Out of scope here.
- **Cross-bundle question reuse**. Each follow-up is scoped to its
  parent run/bundle.
- **Question editing post-submit**. Once a follow-up is submitted, you
  can't edit it. You can submit another one.
- **Rerun characterize/diagnose with new question context**. The user
  can't say "rerun diagnose with this question in mind" — the bundle is
  frozen by design (D1).
- **Persistent follow-up history across API restarts.** `Run.followups`
  lives in the in-memory `RunStore`. API process restart wipes it. The
  bundle's `user_question.json` retains the most-recent question only.
  Acceptable because today's `Run` lifetime is also in-memory. Persisting
  the full `Run` graph is a larger separate effort.
- **Real-LLM eval baseline for the new prompts.** Commits 4/5 use
  mock-LLM tests to verify the right prompt shape fires. Quality eval
  on production runs is the gate, not a CI eval suite. If the
  shape-aware prompts produce worse hypotheses than bias posture,
  we'll see it in the first follow-up production run and iterate.

---

## Risks & mitigations

1. **Follow-up runs concurrent with the original.** The original run
   might still be HYPOTHESIZING when the user clicks follow-up. Mitigation:
   API guards on `status == DONE` and returns 409 Conflict otherwise.

2. **Bundle dir gets garbage-collected between original and follow-up.**
   API retention policy might delete `bundle_dir` after some time;
   follow-up would 404. Mitigation: when the user retires a run, the
   server explicitly marks it non-followup-able and the UI hides the
   textarea.

3. **`mechanistic` shape produces vacuous hypotheses.** If the user's
   mechanism is unfalsifiable from the bundle (e.g. "was the bioreactor
   secretly contaminated?"), the synthesizer should set
   `question_answered="insufficient_data"` rather than fake a yes.
   Tested by commit 4's prompt rule and commit 5's critic axis.

4. **Comparative shape with no overlap between named groups.** User
   asks "compare RUN-0099 to RUN-9999" — neither exists. Same path as
   D3: emit single FinalHypothesis with `insufficient_data` listing
   available runs.

5. **Status transitions race during background task.** The async
   `execute_followup_run` flips status; the GET endpoint races. Same
   pattern as the existing `execute_run` / `execute_resume` pair, well-
   tested today.

6. **Frontend UI stale after follow-up submission.** UI needs to refresh
   the run details to see the new follow-up card. Mitigation: same
   3-second polling that drives the runs list, plus an immediate
   refresh after the POST returns.

7. **Follow-up #2 doesn't see follow-up #1's output.** Synthesizer
   prompt logic for `open` shape only injects the *original* run's
   `accepted_hypotheses_so_far`, not earlier follow-ups. User asking
   refining questions in series may be surprised when refinement
   doesn't compound. Mitigation: documented in "NOT in scope"; revisit
   if production usage shows people chaining follow-ups.

---

## Cost summary

| Component | LOC |
|---|---|
| Schema additions | ~80 |
| Runner (execute_followup_run + dispatch) | ~180 |
| Shape-aware seed topic logic | ~150 |
| Synthesizer drive-mode prompt | ~100 |
| Critic followup-axis | ~50 |
| API endpoint + state | ~120 |
| Frontend + e2e | ~150 |
| Tests at every layer | ~250 |
| **Total** | **~1080 LOC, 7 commits** |

---

## Resume checklist post-compact

If we hit `/compact` mid-build, the resume agent should:

1. `cd /Users/dibyochakraborty/Docling_Parse && git branch --show-current` — confirm we're on `hitl-followup`
2. `git log --oneline hitl-followup ^caisc-hitl | head` — see how many of the 7 commits are done
3. Read this file `plans/2026-05-05-hitl-followup.md` for the per-commit scope
4. Check the most recent commit's diff to see what's mid-flight
5. Continue from the next commit in sequence; tests + full-suite gate
   between every commit

Branch state at plan-writing time: `hitl-followup` is at `d5666bb`
(same as `caisc-hitl`). Zero PR-A2 commits yet. Full suite passing at
1101 tests.

---

## Companion: post-PR-A test signal

Verified this plan is the right shape based on the live PR-A run
`772ecf9f-085a-4037-a0d8-a5ae4fbdb4a4`. Output had:
- H-0003 with `question_answered="partial"` correctly framing RUN-0001's
  RQ peak
- H-0005 with `question_answered="insufficient_data"` correctly noting
  reactor geometry was missing
- Both cited catalog metric_ids (B10 F-0113, F-0117, F-0119)
- `confidence_basis: cross_run` auto-upgraded on H-0003

The bias posture works. The two-cards-on-one-question redundancy is
exactly what drive posture would collapse: the user follows up with
"focus on RUN-0001 only" and gets ONE refined hypothesis instead of two
overlapping ones. That's the user-visible value PR-A2 unlocks.
