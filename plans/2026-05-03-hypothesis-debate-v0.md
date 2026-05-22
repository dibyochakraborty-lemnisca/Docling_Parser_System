# Plan — Hypothesis Stage v0

**Date:** 2026-05-03
**Status:** draft (revised — agentic-AI-engineer pass)
**Branch:** `hypothesis-stage-v0`
**Owner:** diagnosis pipeline → hypothesis stage
**Wave:** 3 (multi-agent)

---

## 0. What changed from the first draft

The first draft proposed a 3-specialist **debate** with voting + critic + judge. After a context-engineering review:

- **Pattern shift: facet-contribution + synthesis + adversarial review**, not inter-specialist debate. Specialists in fermentation contribute different angles on the same failure (kinetic, transport, metabolic) — they don't usually disagree, they triangulate. Voting is overhead until evals show real disagreement that synthesis is hiding.
- **Role-shaped views, not raw `global.md` reads.** Every agent sees a projection tailored to its job. The Observer becomes an Observer + Projector.
- **Deterministic where possible.** Topic ranking, vote tallying (when reintroduced), termination, citation integrity — all deterministic. LLMs do synthesis, facet generation, criticism, judgment.
- **Trim tools.** Specialists 4 tools (was 7). Critic 4 (was 7). Judge 0 — structured output only.
- **Token instrumentation in Stage 1**, not Stage 3. We measure before we optimize.
- **Eval cases written before Stage 2**, not Stage 4.

The first-draft architecture is preserved as a v1 fallback if Stage 3 evals show specialists are silently disagreeing under the synthesizer.

---

## 1. Goal

Take a frozen `DiagnosisOutput` + its upstream `CharacterizationOutput` bundle and run a bounded multi-agent process producing:

1. A small set of **synthesized, criticized hypotheses** explaining the failures/analyses/trends.
2. A list of **unresolved open questions** the stage could not settle.
3. A **structured event log** (`global.md`) — the canonical artifact, written by Observer, projected per-role for agent consumption.

End-to-end success: run on the carotenoid bundle and produce a hypothesis a fermentation engineer wants to read, in under a fixed budget, with valid citations and reproducible across reruns.

Non-goals for v0:
- LangGraph / framework migration
- `simulate()` tool (kinetic model invocation)
- Parallel facet generation
- SuperMemory filter
- Per-agent persistent YAMLs (`history.yaml` / `voting-logs.yaml`) — derived from `global.md`
- Inter-specialist voting (deferred until evidence of disagreement)
- Human-in-the-loop modes (answer/criticize/accept-and-ask)

---

## 2. Architecture (v0)

```
                 HypothesisStage(input: DiagnosisOutput, CharBundle)
                                    │
                                    ▼
                       ┌─────────────────────────┐
                       │  Topic Ranker (det.)    │
                       │  scores seed topics +   │
                       │  open questions         │
                       └─────────────────────────┘
                                    │ top topic
                                    ▼
                       ┌─────────────────────────┐
                       │  Orchestrator (LLM)     │
                       │  picks topic from top-K │
                       │  (det. ranker resolves  │
                       │  ties; LLM only for     │
                       │  ambiguous heads)       │
                       └─────────────────────────┘
                                    │
                ┌───────────────────┼────────────────────┐
                ▼                   ▼                    ▼
          Specialist:         Specialist:          Specialist:
          kinetics            mass-transfer        metabolic
          (facet)             (facet)              (facet)
                │                   │                    │
                └───────────────────┼────────────────────┘
                                    ▼
                       ┌─────────────────────────┐
                       │  Synthesizer (LLM)      │
                       │  merges facets → 1 hyp  │
                       └─────────────────────────┘
                                    │
                                    ▼
                       ┌─────────────────────────┐
                       │  Critic (LLM, tools)    │
                       │  attacks the hypothesis │
                       └─────────────────────────┘
                                    │
                                    ▼
                       ┌─────────────────────────┐
                       │  Judge (LLM, no tools)  │
                       │  rules on critique      │
                       └─────────────────────────┘
                                    │
                       ┌────────────┴───────────┐
                       │                        │
                  rejected: revise         accepted: emit
                  (≤ max_critic_cycles)    final hypothesis
                       │                        │
                       └────────► next turn ◄───┘

  Observer (deterministic): writes structured events to global.md.
  Projector (deterministic): renders role-shaped views from events for each agent.
```

Roles:
- **Topic Ranker** (deterministic) — scores seed topics + open questions; LLM only resolves top-K ties.
- **Orchestrator** (LLM) — picks current topic from ranker's top-K, decides whether to terminate when ranker is empty/exhausted, manages open-questions ledger.
- **Specialists ×3** (LLM) — kinetics, mass-transfer/physical, metabolic/biology. Each contributes a *facet* on the current topic, not a competing hypothesis.
- **Synthesizer** (LLM) — merges facets into one structured hypothesis with citations, confidence, basis.
- **Critic** (LLM + tools) — attacks the synthesized hypothesis; can run code, query bundles, check priors.
- **Judge** (LLM, no tools) — structured ruling on whether critic's flag is valid.
- **Observer + Projector** (deterministic) — writes events; computes per-role views.

---

## 3. State: `global.md` as event log

**Canonical artifact.** Single writer (Observer). No agent reads it raw — they read role-shaped views.

Format: markdown with embedded fenced JSONL blocks. Human-readable AND machine-parseable. Atomic temp+rename per write. Lives in a sibling `hypothesis/` dir next to the diagnosis bundle.

Event types:
```jsonc
{"type": "stage_started", "ts": "...", "input_diagnosis_id": "...", "budget": {...}}
{"type": "topic_selected", "turn": 1, "topic_id": "T-0001", "summary": "...", "rationale": "..."}
{"type": "facet_contributed", "turn": 1, "facet_id": "FCT-0001", "specialist": "kinetics", "summary": "...", "cited_finding_ids": [...], "cited_narrative_ids": [...], "cited_trajectories": [...], "confidence": 0.7, "confidence_basis": "process_priors"}
{"type": "hypothesis_synthesized", "turn": 1, "hyp_id": "H-0001", "summary": "...", "facet_ids": ["FCT-0001","FCT-0002","FCT-0003"], "cited_finding_ids": [...], "cited_narrative_ids": [...], "cited_trajectories": [...], "affected_variables": [...], "confidence": 0.65, "confidence_basis": "process_priors"}
{"type": "critique_filed", "turn": 1, "hyp_id": "H-0001", "flag": "red|green", "reasons": [...], "tool_calls": [...]}
{"type": "judge_ruling", "turn": 1, "hyp_id": "H-0001", "criticism_valid": true, "rationale": "..."}
{"type": "hypothesis_accepted", "turn": 1, "hyp_id": "H-0001"}
{"type": "hypothesis_rejected", "turn": 1, "hyp_id": "H-0001", "reason": "..."}
{"type": "question_added", "qid": "Q-0001", "question": "...", "raised_by": "kinetics", "tags": ["DO","kLa"]}
{"type": "question_resolved", "qid": "Q-0001", "resolution": "...", "resolved_in_turn": 2}
{"type": "tokens_used", "turn": 1, "agent": "synthesizer", "input": 2400, "output": 380}
{"type": "stage_exited", "reason": "budget_exhausted|consensus|max_turns|no_topics", "final_hyp_ids": [...]}
```

Why event log not prose: deterministic projection per role, cheap parsing, derived views (per-specialist contributions, vote tallies if reintroduced) computed on demand, humans render to prose for reading.

---

## 4. Role-shaped views (Projector spec)

Every agent receives a **typed view object**, not raw events. Each view is small, role-relevant, and refreshed each turn.

### `OrchestratorView`
```python
class OrchestratorView(BaseModel):
    current_turn: int
    budget_remaining: BudgetSnapshot
    top_topics: list[RankedTopic]          # top-K from deterministic ranker
    open_questions: list[OpenQuestionRef]  # unresolved only
    last_turn_outcome: TurnOutcome | None  # 1-line summary of last turn
    accepted_hypotheses_so_far: list[HypothesisRef]  # id + summary only
```

### `SpecialistView`
```python
class SpecialistView(BaseModel):
    specialist_role: Literal["kinetics","mass_transfer","metabolic"]
    current_topic: TopicSpec               # what to contribute on
    relevant_findings: list[FindingRef]    # filtered to specialist's domain
    relevant_narratives: list[NarrativeRef]
    relevant_trajectories: list[TrajectoryRef]
    relevant_priors: list[ResolvedPrior]   # auto-resolved by domain
    open_questions_in_domain: list[OpenQuestionRef]
    prior_facets_this_topic: list[FacetSummary]  # from this turn's other specialists, summary only
```

### `SynthesizerView`
```python
class SynthesizerView(BaseModel):
    current_topic: TopicSpec
    facets: list[FacetFull]                # full text from all 3 specialists
    citation_universe: CitationCatalog     # union of all cited IDs across facets
```

### `CriticView`
```python
class CriticView(BaseModel):
    hypothesis: HypothesisFull             # synthesized hypothesis as structured object
    citation_lookups: dict[str, Any]       # pre-resolved from bundle for each cited ID
    relevant_priors: list[ResolvedPrior]
    debate_summary_one_line: str           # not full debate
```

### `JudgeView`
```python
class JudgeView(BaseModel):
    hypothesis: HypothesisFull
    critique: CritiqueFull
    citation_lookups: dict[str, Any]
    # explicitly NO debate history
```

Each view has a hard token budget enforced by the projector (cap, then truncate-with-marker oldest-first).

---

## 5. Tools per agent (trimmed)

Tools are an API the LLM consumes. Fewer high-leverage tools.

**Orchestrator (3):**
- `select_topic(topic_id, rationale)` — pick from view's `top_topics`
- `add_open_question(question, tags, raised_by)` / `resolve_open_question(qid, resolution)`
- `exit_stage(reason)`

(Topic ranking, budget tracking, view assembly all happen in the runner — not orchestrator tools.)

**Specialist (4):**
- `query_bundle(scope: "diagnosis"|"characterization", id_or_query)` — unified, scope-parameterized
- `get_priors(organism, process_family, variable)` — same as diagnose
- `get_narrative_observations(run_id, tag, variable, limit)` — same as diagnose
- `contribute_facet(summary, cited_finding_ids, cited_narrative_ids, cited_trajectories, affected_variables, confidence, confidence_basis)`

**Synthesizer (1):**
- `emit_hypothesis(summary, cited_finding_ids, cited_narrative_ids, cited_trajectories, affected_variables, confidence, confidence_basis)`

(Synthesizer needs no read tools — its view contains all facets pre-loaded.)

**Critic (4):**
- `query_bundle(scope, id_or_query)`
- `get_priors(...)`, `get_narrative_observations(...)`
- `execute_python(code, timeout)` — sandboxed, reuses diagnose harness
- `file_critique(flag, reasons)`

**Judge (0):**
- Structured-output call only. Returns `{criticism_valid: bool, rationale: str}`.

**Observer:**
- Not an agent in v0. Deterministic Python. Writes events on behalf of runner.

---

## 6. Prompt template (uniform across roles)

Every LLM call in v0 uses this layered prompt structure. Stage 1 builds the template; subsequent stages only fill slots.

```
[1] SYSTEM
    Identity, invariants, tool-use rules, output schema reference.
    STABLE across all turns of a role → cache-friendly prefix.

[2] TASK SPEC
    What you are doing this turn. One paragraph.
    STABLE per role.

[3] VIEW
    Role-shaped view object, JSON, with section headers.
    VOLATILE — changes every turn.

[4] TOOL SCHEMAS
    Available tools with one-line purpose each.
    STABLE per role.

[5] RECAP
    "This turn: [task]. Output via [tool|structured response].
     Hard rule: cite at least one finding_id OR narrative_id."
    STABLE — recency anchor.
```

Cache strategy: [1] + [2] + [4] + [5] form the cacheable prefix. [3] is the per-turn diff. Anthropic prompt caching saves the prefix; only [3] is uncached cost per turn.

---

## 7. Termination (deterministic policy)

Hard budgets enforced by the runner. Orchestrator-LLM cannot override.

- `max_turns: 5`
- `max_critic_cycles_per_topic: 2` — after 2 critic-rejections on same topic, runner forces topic change
- `max_tool_calls_total: 80`
- `max_tokens_per_agent_call: 4000` (input cap; output via model defaults)
- `max_open_questions: 15` — beyond this, orchestrator must resolve before adding
- `max_total_input_tokens: 200_000` — runaway guard

Exit reasons (in priority order): `budget_exhausted`, `max_turns_reached`, `consensus_reached` (≥2 accepted hypotheses), `no_topics_left`.

---

## 8. Deterministic topic ranker

Replaces LLM-driven topic picking. Score:

```
score(topic) =
    severity_weight(topic.source) * topic.priority
  + 0.3 * citation_density(topic)
  + 0.2 * unresolved_question_overlap(topic)
  - 0.5 * times_attempted(topic)         # prevents loops
  - 1.0 * times_rejected_by_judge(topic) # prevents thrashing
```

Returns top-K (K=3). Orchestrator-LLM picks from top-K; if scores are within ε=0.05, LLM resolves; else runner auto-picks #1 and orchestrator just confirms.

Open questions enter the ranker as synthetic topics with their own `priority`.

---

## 9. Hypothesis input contract

```python
class HypothesisInput(BaseModel):
    diagnosis: DiagnosisOutput
    characterization: CharacterizationOutput
    bundle_path: Path
    seed_topics: list[SeedTopic]    # derived from diagnosis
    organism: str | None            # for prior resolution
    process_family: str | None
```

`SeedTopic`:
```python
class SeedTopic(BaseModel):
    topic_id: str
    summary: str
    source_type: Literal["failure","analysis","trend","open_question"]
    source_id: str
    cited_finding_ids: list[str]
    cited_narrative_ids: list[str]
    cited_trajectories: list[TrajectoryRef]
    affected_variables: list[str]
    severity: Severity
    priority: float                  # ranker input
```

---

## 10. Output contract

```python
class HypothesisOutput(BaseModel):
    meta: HypothesisMeta             # schema_version, hypothesis_id, supersedes_diagnosis_id, model, ts, budget_used
    final_hypotheses: list[FinalHypothesis]
    rejected_hypotheses: list[RejectedHypothesis]   # with critic+judge rationale
    open_questions: list[OpenQuestion]              # unresolved at exit
    debate_summary: str                             # short prose render
    global_md_path: Path
    token_report: TokenReport                       # per-agent input/output totals
```

Same provenance model as DiagnosisOutput. Citation integrity hard, provenance downgrade soft. Reuses the diagnose validators where possible (`validate_hypothesis(out, upstream=...)`).

---

## 11. Stages

Land in 4 stages, ship clean at each. Each stage gates on the next.

### Stage 1 — Skeleton, state, projector, instrumentation (no LLM)
- New package `src/fermdocs_hypothesis/`
- `schema.py` — HypothesisInput, HypothesisOutput, SeedTopic, FinalHypothesis, all event models, all view models
- `event_log.py` — Observer (deterministic writer), atomic write, JSONL parser
- `projector.py` — view assemblers per role (OrchestratorView, SpecialistView, SynthesizerView, CriticView, JudgeView), each with budget enforcement
- `state.py` — open-questions ledger, derived tallies
- `ranker.py` — deterministic topic ranker
- `runner.py` — full orchestration loop, budget enforcement, stub agents emit canned events
- `instrumentation.py` — token counter, per-agent ledger
- Tests: event roundtrip, projector view sizing, ranker determinism, budget exhaustion, ledger
- **Gate:** stage runs end-to-end with stubs producing valid HypothesisOutput; token meter reports realistic budget usage estimate

### Stage 2 — Orchestrator + 1 specialist + synthesizer (real LLM)
- `agents/orchestrator.py` (Anthropic + Gemini)
- `agents/specialist_kinetics.py` — first specialist with persona spec
- `agents/synthesizer.py`
- `tools_bundle/` — query_bundle, get_priors, get_narrative_observations, contribute_facet, emit_hypothesis, select_topic, add/resolve_open_question
- Prompt template implemented per §6
- Run on carotenoid bundle: orchestrator picks topic, kinetics contributes one facet, synthesizer emits hypothesis. No critic yet.
- **Gate:** produces ≥1 hypothesis with valid citations on carotenoid bundle; total tokens within budget; cache hit rate ≥40% on prompt prefix

### Stage 3 — All specialists + critic + judge + validators
- Add `specialist_mass_transfer.py` + `specialist_metabolic.py`
- `agents/critic.py` (with execute_python via shared diagnose sandbox)
- `agents/judge.py` (structured output, no tools)
- `validators.py` — citation integrity, provenance downgrade (reuse diagnose validators where possible)
- Run on carotenoid + IndPenSim bundles
- **Gate:** end-to-end on both bundles; final hypothesis cites ≥1 narrative AND ≥1 finding; critic+judge cycle exercised at least once; token report within `max_total_input_tokens`

### Stage 4 — Eval, CLI, docs
- `cli.py` — `fermdocs-hypothesize <bundle_path>`
- Authoring guide for adding specialists (specialist persona spec format)
- CI grep guard: no agent reads other agents' state directly (only via Projector views)
- N=5 reliability rerun on carotenoid + IndPenSim, document variance
- Adversarial eval: inject hypothesis with known weak citation, confirm critic catches it
- **Gate:** eval matrix passes; reliability variance documented; PR ready

---

## 12. Eval matrix (write before Stage 2)

`tests/eval/hypothesis_v0/*.yaml`:

**Capability (3 cases):**
- `carotenoid_pdf.yaml` — full PDF bundle; expected hypothesis class: "premature termination + cell death across batches"; required citation pattern: ≥1 narrative + ≥1 finding
- `indpensim_csv.yaml` — IndPenSim CSV bundle; expected: PAA-cessation hypothesis cited as `process_priors`
- `mixed_bundle.yaml` — synthetic mix of CSV trajectories + 3 narrative observations; expected: hypothesis covering both surfaces

**Reliability:** N=5 rerun of each capability case. Score: cluster of root-cause variables across reruns (Jaccard ≥0.6 = pass).

**Adversarial:** `weak_citation.yaml` — inject a hypothesis where one cited_finding_id is real but the citation is irrelevant to the claim. Critic must red-flag.

**Token/cost:** p50 and p95 input tokens per run. Regression check: <20% increase between stages.

---

## 13. What's deferred to v1 (with trigger conditions)

- **Inter-specialist voting + debate** — re-introduce if Stage 3 evals show specialists silently disagree (e.g., facets contradict and synthesizer picks arbitrarily)
- **`simulate()` tool** — when hypothesis quality plateaus and execute_python isn't enough
- **`web_search()`** — when open questions consistently need external lookup
- **SuperMemory filter** — when running across many bundles and past insights would help
- **LLM-powered Observer** (narrative rendering) — when humans complain `debate_summary` is too terse
- **Parallel facet generation** — when latency bites
- **LangGraph** — when state passing in `runner.py` exceeds ~300 lines or branches get gnarly
- **Human-in-the-loop modes** — separate Wave 4 work
- **Per-agent persistent YAMLs** — confirmed redundant; only revisit if Projector views can't compute a needed derived view efficiently

---

## 14. Risks + mitigations

- **Synthesizer averages facets into mush.** Mitigation: synthesizer prompt requires preserving each facet's distinguishing claim; Stage 2 eval reads synthesized hypothesis vs facet inputs.
- **Token blowout despite caps.** Mitigation: instrumentation in Stage 1; per-view token caps in projector; cache-friendly prompt structure.
- **Topic ranker degenerates** — same topic re-picked. Mitigation: `times_attempted` and `times_rejected_by_judge` penalties already in formula.
- **Critic rubber-stamps** (always green-flags). Mitigation: adversarial eval case in Stage 4; if critic accuracy <80% on injection, force critic to use ≥1 tool before filing.
- **Judge collusion** with critic. Mitigation: judge sees no debate history (per JudgeView spec); if eval shows collusion, switch judge to different model in v1.
- **Specialist domain bleed** (kinetics specialist comments on metabolic). Mitigation: SpecialistView filters relevant_* to domain; soft prompt rule, not hard validator (v0).

---

## 15. Open questions before Stage 1 starts

1. `global.md` lives in `<diagnosis_bundle>/hypothesis/` (sibling) — confirmed by §3.
2. Specialists vote? **No in v0** — facet-contribution instead. Revisit after Stage 3.
3. Critic execute_python sandbox — **shared with diagnose**, reuse harness.
4. Judge model — **same as critic, different system prompt**. Multi-model judge deferred to v1.
5. Specialist persona format — **`SPECIALIST_SPEC` dict per file** in `agents/specialists/`. Stage 4 documents the format.

All five answered with leans. No open blockers for Stage 1.

---

## 16. First commits on `hypothesis-stage-v0`

1. This plan (current commit)
2. Stage 1 skeleton: empty `src/fermdocs_hypothesis/` package + schema.py + event_log.py
3. Stage 1 projector + ranker + tests
4. Stage 1 runner + stub agents + token meter + end-to-end stub run
5. (Stage 1 gate) PR-ready commit with passing tests

Then Stage 2 begins on the same branch.

---

## 17. Deferred to v0.6 — free-form chat

v0.5 ships **structured-answer HITL only**: the system asks specific
open questions, the user answers tied to a `qid`, the system resumes.
Free-form chat ("redo with X consideration", "explain hypothesis H-3",
"what about temperature drift in BATCH-04?") is **not** supported.

The orchestrator agent today has tools `select_topic`, `add_open_question`,
`exit_stage`. It has no "respond to user message" surface, and the runner
has no event type for "user said X out of band."

Adding chat is real work, not a tweak. v0.6 backlog:

1. **New event type** `user_message_received` (carries free-form text + ts)
2. **New run state** `awaiting_user` (vs `paused` which is structured-only)
3. **New runner branch** that, on receiving a user message, projects a new
   `OrchestratorView` with recent user messages and asks the orchestrator-
   LLM to either: open a new topic for it, add it as an open question, or
   answer it directly from `global.md` history.
4. **A "Q&A agent"** for accept-and-ask mode — different agent shape,
   retrieval over `global.md` events, no debate.
5. **Frontend chat panel** with input box, message history, threading
   by topic.
6. **WebSocket bidirectional** (today it's server→client only).

Estimated 2-3 days of focused work. Trigger: structured-answer HITL
proves insufficient for real research use after a week or two of
real-bundle runs.
