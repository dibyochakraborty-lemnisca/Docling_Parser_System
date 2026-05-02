# Diagnosis Agent — Execution Plan

**Date:** 2026-05-02
**Author:** Pushkar + Claude (pair)
**Status:** Approved (post plan-eng-review revisions, 2026-05-02)
**Supersedes:** N/A (new agent)
**Depends on:** `plans/2026-05-01-identity-and-agent-context.md` (characterization agent + AgentContext, shipped)

---

## Revision log

- **2026-05-02 v1.0** — Initial draft, approved by Pushkar.
- **2026-05-02 v1.1** — plan-eng-review pass. Changes:
  - Issue 1: `cited_trajectories` typed as `list[TrajectoryRef]` (no tuples).
  - Issue 2: `re_run_from` removed from `OpenQuestion` for this PR. Diagnosis
    only emits data-gap questions. Hypothesis-stage routing deferred to the
    hypothesis-agent plan.
  - Issue 3: `dossier.experiment.user_clarifications` deferred to the orchestrator
    PR. Dossier schema unchanged in this PR.
  - Issue 4: `AgentContext._finding_summary` expanded to include `statistics` and
    one evidence observation. Byte-stability test added.
  - Issue 5: `confidence_basis` validator soft-downgrades rather than rejects.
    Hard rejection reserved for unknown finding IDs.
  - Issue 6: `FailureClaim`, `TrendClaim`, `AnalysisClaim` inherit from
    `BaseClaim`.
  - Issue 7: `domain_tags` stays `list[str]` (free-form) in v1. Closed enum
    deferred to the hypothesis-agent plan when orchestrator filtering needs it.
  - Critical gap: malformed-JSON retry policy added to §8.
  - Step 0 deferrals: markdown renderers + `narrative` field stay in scope
    (Pushkar accepted full scope).

---

## 1. Purpose

The diagnosis agent is the first **LLM-authored** stage in the multi-agent
pipeline. It reads the deterministic `CharacterizationOutput` + `AgentContext`
and produces an **observational** account of what happened in the run:
failures, trends, analysis, and open questions — each claim cited back to
specific finding IDs.

It does **not** speculate on causes. Causal reasoning belongs to the
hypothesis stage that comes after.

## 2. Pipeline context

```
Characterization (deterministic, shipped)
   ├── CharacterizationOutput (findings + meta)
   └── AgentContext (compact prompt prefix, ~1500 tokens)
        │
        ▼
Diagnosis (THIS PR — observational LLM agent)
   └── DiagnosisOutput (failures, trends, analysis, open_questions)
        │
        ▼
Hypothesis (later — orchestrator + domain experts: Dr.Strain, Dr.Data, ...)
   └── HypothesisOutput (causal hypotheses, ranked + open_questions w/ re_run_from)
        │
        ▼
Simulation (later — tests hypotheses against data/models)
        │
        ▼
Critic (later — adversarial review)
        │
        ▼
Refinement loop (later — user answers open_questions, re-run from
                 diagnosis or hypothesis based on hypothesis-stage routing)
```

Everything below the Diagnosis box is **out of scope for this PR**. It is
documented here so the diagnosis output schema supports it without rework.
LangGraph will be the orchestrator runtime when the hypothesis pipeline lands.

## 3. Output schema

New artifact, persisted alongside characterization. Stable, typed, validator-
enforced.

```python
class DiagnosisMeta(BaseModel):
    schema_version: str                        # "1.0"
    diagnosis_version: str                     # "v1.0.0"
    diagnosis_id: UUID
    supersedes_characterization_id: UUID       # link to upstream artifact
    generation_timestamp: datetime
    model: str                                 # e.g. "claude-opus-4-7"
    provider: Literal["anthropic", "gemini"]


class ConfidenceBasis(str, Enum):
    SCHEMA_ONLY = "schema_only"        # only nominal/std_dev from golden schema
    PROCESS_PRIORS = "process_priors"  # registered process priors invoked
    CROSS_RUN = "cross_run"            # multi-run pattern


class TrajectoryRef(BaseModel):
    """Typed reference to a trajectory. Replaces tuple[str, str] for stability."""
    run_id: str
    variable: str


class BaseClaim(BaseModel):
    """Shared spine for all diagnosis claims. Subclasses add kind-specific fields."""
    claim_id: str                       # "D-{F|T|A}-NNNN"
    summary: str                        # one sentence, observational only
    cited_finding_ids: list[str]        # validator rejects unknown IDs
    affected_variables: list[str]
    confidence: float                   # ≤ 0.85 LLM cap
    confidence_basis: ConfidenceBasis
    domain_tags: list[str]              # free-form in v1; closed vocab deferred
                                        # suggested values in prompt: growth,
                                        # metabolism, environmental, data_quality,
                                        # process_control, yield


class FailureClaim(BaseClaim):
    severity: Severity
    time_window: TimeWindow | None = None


class TrendClaim(BaseClaim):
    direction: Literal["increasing", "decreasing", "plateau", "oscillating"]
    cited_trajectories: list[TrajectoryRef] = Field(default_factory=list)
    time_window: TimeWindow | None = None
    # cited_finding_ids may be empty when cited_trajectories is non-empty;
    # validator enforces ≥1 citation across both fields.


class AnalysisClaim(BaseClaim):
    kind: Literal[
        "cross_run_observation",
        "data_quality_caveat",
        "spec_alignment",
        "phase_characterization",
    ]


class OpenQuestion(BaseModel):
    """Diagnosis-stage open questions are always data-gap questions.

    The `re_run_from` routing field is deliberately omitted in this PR. Diagnosis
    cannot judge whether a user answer would change reasoning vs interpretation —
    that judgement requires hypothesis-stage context. The hypothesis-agent plan
    will reintroduce open-question support with routing.
    """
    question_id: str                    # "D-Q-0001"
    question: str                       # natural language, ≤30s to answer
    why_it_matters: str                 # one sentence on what changes if answered
    cited_finding_ids: list[str]        # what data prompted this question
    answer_format_hint: Literal["yes_no", "free_text", "numeric", "categorical"]
    domain_tags: list[str]


class DiagnosisOutput(BaseModel):
    meta: DiagnosisMeta
    failures: list[FailureClaim]
    trends: list[TrendClaim]
    analysis: list[AnalysisClaim]
    open_questions: list[OpenQuestion]
    narrative: str | None = None        # optional rollup, <500 words
```

### Validators

**Hard rejection (whole output fails to construct):**

- Any `cited_finding_ids` entry must exist in the source `CharacterizationOutput`.
  Citation integrity is the contract that lets downstream agents trust IDs.
- Every `claim_id` is unique within the output and follows `D-{F|T|A}-NNNN` pattern.
- Every `question_id` follows `D-Q-NNNN` pattern.
- All `confidence` values are in [0.0, 0.85] (LLM cap).
- Each `TrendClaim` has `len(cited_finding_ids) + len(cited_trajectories) >= 1`.
- Each `FailureClaim` and `AnalysisClaim` has `len(cited_finding_ids) >= 1`.
- Any `cited_trajectories` entry must reference a real `(run_id, variable)` pair
  in the upstream `Summary`.

**Soft enforcement (warn + auto-fix, output remains valid):**

- Under `UNKNOWN_PROCESS` or `UNKNOWN_ORGANISM` flag in AgentContext: any claim
  with `confidence_basis = "process_priors"` is downgraded to `"schema_only"`,
  a warning is logged, and a per-claim flag `provenance_downgraded: bool` is set.
- `summary` strings are checked against a forbidden-phrases list to enforce
  observational-only invariant: "because", "due to", "caused by", "resulted in",
  "leading to". Soft check (warn + log), not modified — diagnosis output remains
  usable, hypothesis stage will catch lingering causal language.

**Rationale for soft enforcement:** A reject-and-fail loop on a multi-claim
output is expensive (whole batch thrown away for one bad claim). Hard rejection
is reserved for issues that break the data contract (unknown citations); soft
enforcement covers issues the prompt should already prevent and the validator
exists only to catch drift.

## 4. Reasoning pattern

**ReAct, single loop, max 6 steps.** Per characterization output the agent:

1. Reads AgentContext (the cacheable prefix, now richer per Issue 4)
2. For each finding in `top_findings`, optionally fetches more detail via tools
3. Emits claims as it reasons; stops when no new claims would add information
4. Renders optional narrative as a final pass

Direct answer is too thin (skips evidence fetching). Plan-and-execute is overkill
(no replanning needed for per-finding reasoning). Reflection deferred until first-
pass quality is measurable.

## 5. Tool surface

Three thin read-only tools. Deliberately small.

```python
def get_finding(finding_id: str) -> Finding:
    """Full finding record (more detail than AgentContext.top_findings)."""

def get_trajectory(run_id: str, variable: str) -> Trajectory:
    """Time grid + values + imputation flags + quality score."""

def get_spec(variable: str) -> Spec:
    """Nominal, std_dev, unit, source, provenance."""
```

**Explicitly NOT included:**

- `compare_to_nominal(...)` — comparison logic stays in the model. That's the
  reasoning we're paying for.
- `get_dossier()` — invites context bloat. AgentContext already projects what
  matters.
- `search_findings(...)` — findings are enumerated in AgentContext. No search
  needed.
- `get_process_registry(...)` — the agent must not invoke registry priors
  beyond what the AgentContext already surfaces.

Tool outputs are bounded (<2KB each) and structured. Errors return
`{error: str, hint: str}` rather than raising.

## 6. Prompt invariants

These are enforced by both prompt wording and eval checks.

1. **Observational, not causal.** State what happened. Do not state why.
   Hypothesis stage handles causes.
2. **Citation discipline.** Every claim cites ≥1 finding ID (or trajectory
   for trends). Uncited claims rejected by validator.
3. **Honesty under UNKNOWN flags.** When `UNKNOWN_PROCESS` or
   `UNKNOWN_ORGANISM` is in `ctx.flags`, the agent must explicitly state in
   its reasoning that recipe-specific priors are unavailable, and use
   `confidence_basis: "schema_only"` for all affected claims.
4. **Confidence cap.** All confidences ≤ 0.85 (matches identity_extractor
   convention).
5. **Question quality.** Open questions must reference a specific finding,
   carry a clear `why_it_matters`, and be answerable in under 30 seconds.

## 7. AgentContext extension (Issue 4)

`AgentContext` itself is unchanged in shape. The change is in
`agent_context._finding_summary` (currently at `agent_context.py:235`):

**Today:** drops `statistics` and `evidence_observation_ids`, returns ~5 fields
per finding.

**Change:** include `statistics` (small dict, sigma + observed_value + nominal +
std_dev) and the first entry of `evidence_observation_ids` (one ID, not the full
list). Adds ~50 tokens per top finding, ~500 tokens for the max-10 case. Stays
under the 1500-token budget on all 3 fixtures (verified by existing
`test_existing_fixtures_serialize_under_budget` parametrized test).

**Why:** Without this, the diagnosis agent cannot reason about magnitudes from
the prefix alone and burns tool calls on every finding to see numbers. With this,
the model sees enough numeric context in the prefix to triage which findings
warrant a deeper `get_finding` lookup.

**New regression test:** assert byte-stable serialization for a fixed
`(dossier, output)` input — protects prompt-cache shape.

## 8. Markdown renderers

Generated **on demand**, not persisted as source of truth. The structured
`DiagnosisOutput` is the contract; MD is the human-readable view.

```python
def render_failures_md(output: DiagnosisOutput) -> str:    # "Failures.md"
def render_trends_md(output: DiagnosisOutput) -> str:      # "Trends.md"
def render_analysis_md(output: DiagnosisOutput) -> str:    # "Analysis.md"
def render_questions_md(output: DiagnosisOutput) -> str:   # "Questions.md"
def render_diagnosis_md(output: DiagnosisOutput) -> str:   # combined report
```

CLI flag `--emit-markdown DIR` writes all five files. Default off.

## 9. LLM provider + error handling

Reuse the existing provider abstraction:

- Default: Anthropic (`claude-opus-4-7` per current convention)
- Fallback: Gemini
- Reuse `evidence_gated_llm` primitives if any quoted evidence appears in
  claim summaries (substring verification + 0.85 cap)
- Same scripted-client pattern for tests as identity_extractor

**Malformed-output handling (critical gap from review):**

- LLM returns invalid JSON → 1 retry with `system: "your previous response was
  not valid JSON. return only valid JSON matching the schema."`
- Retry also fails → return `DiagnosisOutput` with empty claim lists and
  `meta.error: "llm_output_unparseable"`. Persist for audit; downstream agents
  see an empty diagnosis and can route the run to manual review.
- LLM returns valid JSON but validators reject (unknown citation) → log per
  rejected claim, drop the claim, keep the rest. If all claims rejected, treat
  same as malformed output.

Zero new provider work beyond the retry policy.

## 10. Human-in-the-loop refinement (forward-looking note)

**Out of scope for this PR.** Documented so the future hypothesis-agent plan
can pick up the thread without re-deriving design.

The full pipeline will support a refinement loop where users answer open
questions and the system re-runs. Two question sources, two re-run targets:

| Source     | Question kind     | `re_run_from` | What changes        |
|------------|-------------------|---------------|---------------------|
| Diagnosis  | Data-gap          | (implicit)    | Re-interpret data   |
| Hypothesis | Reasoning-gap     | `hypothesis`  | Re-rank hypotheses  |
| Critic     | Adversarial probe | `hypothesis`  | Force re-reasoning  |

**Why diagnosis-stage `re_run_from` is implicit, not a field:** Diagnosis cannot
judge whether a user answer would change reasoning vs interpretation. Letting
the LLM pick `re_run_from` would route on a guess. Hypothesis-stage agents have
the context to make that call for their own questions.

**Hypothesis-stage HITL specifics (for the future hypothesis-agent plan):**

- `HypothesisOutput.open_questions: list[OpenQuestion]` reintroduces the
  `re_run_from: Literal["diagnosis", "hypothesis"]` field.
- Each Dr.X domain expert may emit child questions tagged with their
  `domain_tags` so the UI can group questions by domain. Domain vocabulary
  closes at that stage based on real filtering needs.
- Critic agent may *upgrade* a Diagnosis open question to `re_run_from =
  "hypothesis"` if the answer would change reasoning rather than data
  interpretation.
- The orchestrator (LangGraph) decides routing: same `OpenQuestion` schema
  shape across stages, dispatched by `re_run_from`.
- User clarifications land in the dossier as a durable layer (added at the
  orchestrator PR, not here):

```python
# dossier.experiment.user_clarifications: list[UserClarification] = []
class UserClarification(BaseModel):
    clarification_id: str
    question_id: str                    # links to D-Q-NNNN or H-Q-NNNN
    answer: str
    answered_at: datetime
    answered_by: str | None
    source_stage: Literal["diagnosis", "hypothesis", "critic"]
```

**Loop termination rules (for future orchestrator):**

- Max 2 refinement passes total.
- Each pass must produce ≥1 new claim OR upgrade an existing claim's
  confidence by ≥0.1, or the loop stops with "no new signal from clarifications".
- User can always force-stop.
- Each clarification stamps the resulting claim's `cited_clarification_ids`
  for auditability (added to claim schemas in the hypothesis PR).

## 11. Eval set

Mirrors characterization fixtures so we test on the same dossiers.

```
evals/diagnosis/
  01_boundary/expected_claims.yaml
  02_missing_data/expected_claims.yaml
  03_multi_run/expected_claims.yaml
  04_unknown_everything/                # adversarial: organism + process unknown
    expected_claims.yaml                # asserts no process_priors used,
                                        # asserts heavy open_questions
  run_evals.py                          # pytest-parametrized runner
```

`expected_claims.yaml` shape:

```yaml
expected_failures:
  - summary_keywords: ["biomass", "plateau"]
    must_cite_findings: ["F-0003"]
    min_confidence: 0.5
    must_have_domain_tags: ["growth"]

expected_open_questions:
  - keywords: ["sampling", "imputation"]
    must_cite_findings: ["F-0004"]

forbidden_phrases:
  - "because"
  - "due to"
  - "caused by"

citation_integrity:
  all_cited_findings_must_exist: true

honesty_under_unknown_flags:
  if_flag: "unknown_process"
  no_claim_with_basis: "process_priors"
```

Scoring:

- **Claim recall:** % of expected claims matched (keyword + cited finding)
- **Citation integrity:** 100% of cited finding IDs must resolve (hard pass/fail)
- **Forbidden-phrase rate:** soft warning, not pass/fail
- **Honesty under UNKNOWN flags:** hard pass/fail
- **Question quality (04_unknown_everything):** must produce ≥3 open questions,
  each citing a specific finding

## 12. Files to create

```
src/fermdocs_diagnose/
  __init__.py
  schema.py              # DiagnosisOutput, BaseClaim + subclasses, TrajectoryRef
  agent.py               # ReAct loop, prompt assembly, retry policy
  tools.py               # get_finding, get_trajectory, get_spec
  validators.py          # citation integrity, soft downgrade, confidence cap
  renderers.py           # markdown renderers
  cli.py                 # --emit-markdown, --provider, etc.

src/fermdocs_characterize/agent_context.py
  (extend _finding_summary per Issue 4 — statistics + 1 evidence id)

tests/unit/test_diagnose_schema.py
tests/unit/test_diagnose_validators.py
tests/unit/test_diagnose_tools.py
tests/unit/test_diagnose_agent.py            # ReAct loop, scripted LLM
tests/unit/test_diagnose_renderers.py
tests/unit/test_agent_context.py             # add byte-stability regression test
tests/integration/test_diagnose_pipeline.py
tests/integration/test_diagnose_cli_emit_md.py

evals/diagnosis/
  01_boundary/expected_claims.yaml
  02_missing_data/expected_claims.yaml
  03_multi_run/expected_claims.yaml
  04_unknown_everything/dossier.json
  04_unknown_everything/expected_claims.yaml
  run_evals.py
```

**Reuse audit (no rebuilds):**

| Component | Source | How used |
|-----------|--------|----------|
| `Severity`, `TimeWindow` | `fermdocs_characterize.schema` | imported, not redefined |
| `Finding` | `fermdocs_characterize.schema` | tool returns it directly |
| `Trajectory`, `Spec` | existing | tools return as-is |
| `evidence_gated_llm` | `fermdocs.mapping.evidence_gated_llm` | substring + cap |
| LLM provider abstraction | `fermdocs.mapping.identity_extractor` pattern | scripted client for tests |
| `AgentContext` builder | `fermdocs_characterize.agent_context` | extended via Issue 4 |
| `CharacterizationOutput` JSON storage | existing pipeline persistence | sibling artifact |
| `ProcessFlag` enum | `fermdocs_characterize.flags` | read from AgentContext.flags |

## 13. Migration / DB impact

None. Diagnosis output is persisted as JSON alongside characterization output;
existing storage handles it. `user_clarifications` is deferred to the
orchestrator PR (not added in this PR).

If/when we persist clarifications to DB (later), it becomes its own table with
FK to experiment_id.

## 14. Out of scope (explicit)

- Hypothesis agents (Dr.Strain, Dr.Data, etc.)
- Simulation agent
- Critic agent
- LangGraph orchestrator
- Refinement loop runtime
- `re_run_from` routing field on `OpenQuestion` (added at hypothesis-agent PR)
- `dossier.experiment.user_clarifications` layer (added at orchestrator PR)
- `domain_tags` closed-vocabulary enum (deferred until orchestrator filtering
  proves the right vocabulary)
- DB persistence for clarifications
- A UI for answering open questions
- Causal reasoning of any kind in DiagnosisOutput

## 15. Worktree parallelization

```
Wave 1 (parallel):
  Lane A — schema + validators
    src/fermdocs_diagnose/schema.py + validators.py
    tests/unit/test_diagnose_schema.py + test_diagnose_validators.py
  Lane B — tools
    src/fermdocs_diagnose/tools.py
    tests/unit/test_diagnose_tools.py
  Lane C — AgentContext finding summary expansion (Issue 4)
    edit src/fermdocs_characterize/agent_context.py
    extend tests/unit/test_agent_context.py with byte-stability regression test

Wave 2 (after Wave 1 merges):
  Lane D — agent + ReAct loop
    src/fermdocs_diagnose/agent.py
    tests/unit/test_diagnose_agent.py + tests/integration/test_diagnose_pipeline.py
  Lane E — renderers + CLI (parallel with D, no shared modules)
    src/fermdocs_diagnose/renderers.py + cli.py
    tests/unit/test_diagnose_renderers.py + tests/integration/test_diagnose_cli_emit_md.py

Wave 3 (after Wave 2 merges):
  Lane F — evals
    evals/diagnosis/* with all 4 fixtures
    eval gates wired into pytest run
```

**Conflict flag:** Lanes A, B, D, E all touch `src/fermdocs_diagnose/`. Wave 1
finishes before Wave 2 starts to avoid module-internal merge churn.

## 16. Failure modes

| Codepath | Failure mode | Test? | Error handling | Silent? |
|----------|--------------|-------|----------------|---------|
| `agent.py` ReAct loop | LLM returns malformed JSON | ✅ | 1 retry → empty output w/ meta.error | not silent |
| `agent.py` ReAct loop | LLM cites non-existent finding ID | ✅ | validator drops claim, logs | not silent |
| `agent.py` ReAct loop | LLM exceeds step budget | ✅ | terminate, emit what's produced | not silent |
| `tools.get_finding` | finding_id not in output | ✅ | return error dict | not silent |
| `tools.get_trajectory` | (run_id, var) not in summary | ✅ | return error dict | not silent |
| `validators.py` | UNKNOWN_PROCESS + process_priors | ✅ | soft downgrade + warn | not silent |
| `validators.py` | forbidden phrase in summary | ✅ | warn only | not silent |
| `renderers.py` | claim missing optional field | ✅ snapshot | render placeholder | not silent |

**No critical gaps.** Malformed-JSON handling closed by §9 retry policy.

## 17. Execution log

_Filled in as work progresses. Same convention as the identity-and-agent-
context plan: one entry per commit._

| Date | Commit | Lane | Notes |
|------|--------|------|-------|
| 2026-05-02 | 9559639 | Wave 1 (A+B+C) | schema, validators, tools, AgentContext finding-summary expansion. 36 new tests, 230 total pass. |
| 2026-05-02 | a1c97c0 | Wave 2 (D+E) | ReAct agent.py with scripted-client testability + retry/error paths, renderers.py (5 markdown sidecars), cli.py. 29 new tests, 259 total pass. |
| 2026-05-02 | 82c95ee | Wave 3 (F) | 4 eval fixtures (01_boundary, 02_missing_data, 03_multi_run, 04_unknown_everything) with scripted LLM responses + expected_claims.yaml, pytest-parametrized run_evals.py scoring citation integrity, honesty under UNKNOWN flags, claim recall, provenance downgrade, forbidden phrases. 29 eval pass + 7 skipped (by-design optional checks). 293 total tests pass. |

## 18. Working agreement

- This file is the source of truth for diagnosis-agent scope.
- I draft, you approve, I execute. No file writes outside `/plans/` and
  the listed paths above without explicit approval.
- Tests pass before each commit. Eval scores logged in commit messages.
- No amending committed history.
