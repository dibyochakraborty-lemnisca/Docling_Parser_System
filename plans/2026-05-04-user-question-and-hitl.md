# User-question-driven debate + HITL integration

**Status:** Deferred — bundle with HITL PR.
**Trigger:** Whenever HITL implementation starts.
**Estimated cost:** ~900-1100 LOC for question-driven debate alone, ~1500-2000 LOC combined with HITL infrastructure.

---

## Problem

Today the pipeline is **bottom-up**: parse → characterize → diagnose →
seed_topics → ranker → debate. Topics emerge from the data; whatever
the data surfaces is what the system debates.

The user wants a **top-down overlay**: arrive with a specific question
("Why did RUN-0034 plateau early?"), have the agentic system run as it
does today but biased toward addressing that question at every layer.

The question doesn't *replace* bottom-up flow — the data still has to
surface real findings. The question becomes a **lens** that biases every
downstream decision toward addressing it.

---

## Three shapes a user question can take

Naming the patterns up front so prompts and tests have concrete targets:

1. **Scoping** — "Why did RUN-0034 plateau early?" Narrows to a
   specific run / time / variable.
2. **Mechanistic** — "Was DO limitation responsible for the biomass drop?"
   Proposes a hypothesis the user wants tested.
3. **Comparative** — "What's different between the high-yield batches
   and the rest?" Points at a comparison to explore.

Each shape biases the pipeline differently but shares the same
schema slot and threading infrastructure.

---

## Where the question lands in each pipeline stage

```
parse / characterize ──▶ unaffected (data extraction is question-agnostic)
                          (future: aggressive version could selectively
                          re-process if user only cares about one run)

diagnose ──────────────▶ prompt addition: "USER QUESTION: <text>. After
                          your normal analysis, ensure your output
                          includes ≥1 claim or open_question that bears
                          directly on this question."
                          (no schema change — agent's existing output
                          already accepts this shape)

seed_topic_extractor ──▶ new helper _question_relevance_score(claim, q)
                          (tag-overlap heuristic min, LLM-summarized
                          embedding match ideal). Question-relevant
                          topics get priority bump.

ranker ────────────────▶ existing formula gets a new term:
                            score += W_user * question_relevance(topic)
                          W_user weighted high enough to dominate but
                          not so high that critical findings unrelated
                          to the question get starved.

orchestrator ──────────▶ sees question in OrchestratorView; uses for
                          tie-breaking when ranker scores cluster.

specialists ───────────▶ see question in SpecialistView; facets framed
                          in service of it. Prompt: "user is asking
                          about X — your facet should explicitly address
                          whether X is consistent with what you see."

synthesizer ───────────▶ frames hypothesis as answer to question.
                          ("The data is consistent with DO limitation
                          because...")

critic ────────────────▶ checks whether hypothesis actually answers
                          the question (additional rejection axis).

judge ─────────────────▶ weighs question-responsiveness alongside
                          existing critique-validity criteria.

output ────────────────▶ FinalHypothesis gains question_answered field:
                          bool | "partial" | "insufficient_data".
                          UI renders "we did / didn't / partially
                          answered your question."
```

---

## Why HITL is the right moment to ship this

The user question is **the simplest, most natural HITL input**. It's
already aligned with how the codebase thinks about human input:

- `HumanInputRecord` schema slot exists (added in feedback-loop PR)
- `HumanInputReceivedEvent` already in the event log discriminated union
- `resume_stage` already accepts human input via `answers` parameter
- `pending_question_seeds` already drains into events at `init` phase

What's missing is the *plumbing* that surfaces the question to each
agent's view + prompt and biases ranking.

### Three HITL surfaces for the question

**(1) Initial question at stage start.** User uploads bundle + types a
question. CLI/API surfaces it as a `pending_question_seeds` entry that
lands as `QuestionAddedEvent` with `raised_by="user"` and tags like
`["user_question"]`. **The runner already handles this today**. The
gap is letting agents know that a question with `raised_by="user"` is
special priority.

**(2) Mid-debate refinement.** User watches the debate live, says
"actually focus on temperature, not DO." `HumanInputRecord` slot on
`AttemptRecord` was built for this exact case — operator note attaches
to current topic, agents see it on retry, debate redirects.

**(3) Question-driven resume.** Run completes with
`question_answered="insufficient_data"`. User sees in UI, provides new
info ("the temp probe in RUN-0002 was miscalibrated") via existing
`resume_stage` machinery. Debate restarts with new context.

All three have the event-log + schema substrate today.

---

## Concrete schema additions

```python
class UserQuestion(BaseModel):
    """A directive from the human user that biases the debate.

    `shape` is optional but useful for prompt routing — different shapes
    take slightly different agent treatments.
    """
    text: str = Field(min_length=1)
    shape: Literal["scoping", "mechanistic", "comparative", "open"] | None = None
    affected_variables: list[str] = []  # optional scoping hints
    affected_runs: list[str] = []
    raised_by: str = "user"


class HypothesisInput(BaseModel):
    diagnosis: ...
    characterization: ...
    seed_topics: ...
    user_question: UserQuestion | None = None  # NEW


class FinalHypothesis(BaseModel):
    ...existing fields...
    question_answered: Literal["yes", "partial", "insufficient_data"] | None = None
    question_response_summary: str | None = None  # one-paragraph
```

Then thread `user_question` into:
- `OrchestratorView`
- `SpecialistView`
- `SynthesizerView`
- `CriticView`
- `JudgeView`
- Diagnose's bundle context (already has `AgentContext` — add a field)

---

## Cost breakdown

| Component | LOC |
|---|---|
| Schema additions (`UserQuestion`, view fields, output fields) | ~50 |
| Projector wiring (thread `user_question` into all views) | ~80 |
| Ranker (relevance scoring + new term) | ~30 |
| Seed topic extractor (relevance helper + priority bump) | ~60 |
| Five agent prompt updates | ~200 |
| Diagnose prompt update | ~50 |
| CLI/API surfaces (`--question` flag, frontend input field) | ~50 |
| Tests at every layer | ~400-500 |
| **Total — question-driven debate alone** | **~900-1100** |
| HITL infra additions (mid-debate input handler, event types, projector for HumanInputRecord) | ~500-700 |
| **Total — combined HITL + user-question PR** | **~1500-2000** |

---

## Build order (when HITL kicks off)

Steps 1-6 are the "user question" side. Step 7 is the "mid-debate human
steer" side. They share schema + event-log infrastructure, which is why
bundling them is the right call.

1. Define `UserQuestion` schema; add optional field to `HypothesisInput`
2. Thread `user_question` into all five agent views + diagnose's
   `AgentContext`
3. Update all agent prompts with question-aware rules (scoping /
   mechanistic / comparative branches as needed)
4. Add ranker `W_user * question_relevance(topic)` term + seed topic
   priority bump
5. Add `question_answered` + `question_response_summary` fields to
   `FinalHypothesis`; synthesizer/critic populate them
6. Add `--question` CLI flag + API endpoint accepts `user_question` body
   field + frontend question-input UI
7. Add mid-debate `HumanInputReceivedEvent` handling — the HITL-specific
   piece. Projector populates `AttemptRecord.human_input` on retry.
   Synthesizer/critic prompts get a "if `human_input` is present, weight
   it heavily" rule.

---

## Three ship options

**A — Bundle with HITL (recommended).** Single focused PR ~1500-2000 LOC.
Shares infrastructure cleanly. Full directive-driven debate capability
in one ship.

**B — Question-at-start before HITL.** ~500-700 LOC. Uses existing
`pending_question_seeds` plumbing; just adds threading + agent prompts.
No new event types needed. Mid-debate refinement waits for HITL.
Cheaper to ship but two PRs.

**C — HITL first, question as special case.** HITL ships with general
operator-note infrastructure. User question becomes a special case of
operator-note attached at run start. Cleanest separation of concerns
but ships HITL without the headline use case.

**Recommendation:** A. The shared infrastructure makes bundling cheaper
than separating; the user-facing capability lands as one coherent ship.

---

## Tests to write when this lands

### Schema + threading
- `UserQuestion` validates required text field, accepts optional shape
- `HypothesisInput.user_question` defaults to None (back-compat)
- Each view (Orchestrator/Specialist/Synthesizer/Critic/Judge) carries
  the user_question through projection
- `FinalHypothesis.question_answered` accepts `yes/partial/insufficient_data/None`

### Ranking
- Ranker with no user_question matches today's output exactly (back-compat)
- Topic that overlaps user_question's affected_variables ranks higher
  than equivalent topic that doesn't
- Critical finding unrelated to question still ranks above minor
  question-relevant topic (W_user doesn't fully starve real anomalies)

### Prompt content
- Each agent prompt mentions user_question handling
- Synthesizer prompt requires question_answered field population when
  user_question is non-None
- Critic prompt includes "does this hypothesis actually answer the
  user's question?" as additional rejection axis

### End-to-end (mocked LLM)
- Full debate with scoping question on RUN-0034 produces hypothesis
  that explicitly references RUN-0034
- Question with no supporting evidence in bundle → final hypothesis has
  question_answered="insufficient_data" + open_questions populated
- Mechanistic question "is X responsible for Y?" → hypothesis explicitly
  evaluates X-Y relationship

### HITL integration
- Mid-debate `HumanInputReceivedEvent` populates `AttemptRecord.human_input`
  on next retry's projector pass
- Synthesizer prompt with non-None `human_input` produces hypothesis that
  references the operator note
- `resume_stage` with new operator note continues debate without
  regenerating prior accepted hypotheses

---

## Cross-references

Related plans:
- `plans/2026-05-03-hypothesis-debate-v0.md` — the v0 debate spec this
  layers on top of.
- `plans/2026-05-04-specialist-routing.md` — companion deferred plan
  for scaling specialist count. User question and routing are
  independent but compose well: routing decides which specialists run,
  the question biases what they look for.

Memory:
- `~/.claude/projects/.../memory/project_specialist_routing_plan.md`
- `~/.claude/projects/.../memory/project_user_question_hitl_plan.md` (if
  saved as a memory after this conversation)
