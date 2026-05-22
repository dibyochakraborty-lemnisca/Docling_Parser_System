# Things to steal from `fermentation-debate-langgraph`

**Status:** Reference notes — to be consulted when sharpening agents, scaling specialists, or building the audit UI.
**Reference repo:** `~/fermentation-debate-langgraph`
**Audit date:** 2026-05-04

After reading the reference repo's prompts, tool catalog, panel
orchestration, and shared-state machinery, here's what's worth stealing
into our system, ranked by ROI. Each item names the source file, what
to copy, what NOT to copy, and where it lands in our codebase.

---

## Tier S+ — steal first, before anything else

### 0. The metric catalog (the actual reason their analysis is so good)

**Source:** `src/tools/analysis_catalog.py` — 60 typed metrics across A/B/C tiers, each with a stable metric_id, required inputs (with CSV column hints), required parameters (with literature defaults), output shape, and applicability rule.

**This is the single highest-leverage steal in the repo.** Reading the catalog explains why their analyst on the IndPenSim run computed μ_max, doubling time, phase-resolved Qp, RQ, Yx/s, all in a few minutes — they're not asking the LLM to invent an analysis plan, they're handing it a structured menu of 60 well-defined metrics with default parameters and forcing it to pick from there.

**The taxonomy:**

```
Tier A (24 metrics) — core logistics + kinetics, almost always computable
  A1   inoculum % check                A13  Qp phasewise
  A2   OD dilution correction          A14  DO margin vs critical
  A3   biomass mass conversion         A15  controller excursions
  A4   working-volume timecourse       A16  operational profile per phase
  A5   cumulative feed                 A17  tip speed
  A6   cumulative substrate fed        A18  P/V (power per volume)
  A7   cumulative base/acid            A19  cross-run KPI table (winner/loser)
  A8   μ(t) via Savitzky-Golay         A20  pairwise timecourse deviation
  A9   doubling time                   A21  variance decomposition + corr matrix
  A10  phase segmentation (lag/exp/    A22  % deviation vs reference + tolerance
        linear/stat/decline)           A23  productivity reduction (Qp_a vs Qp_b)
  A11  phasewise mean μ                A24  data completeness audit
  A12  Qp endpoint

Tier B (20 metrics) — direct analytical, measured inputs required
  B1   substrate consumed (measured)    B11  qO2 = OUR/X
  B2   qs = (dS/dt)/X (measured)        B12  qCO2 = CER/X
  B3   qp = (dP/dt)/X                   B13  Yx/O2
  B4   Yx/s (measured ΔS)               B14  OTR with measured kLa
  B5   Yp/s (measured ΔS)               B15  kLa from dynamic gassing-out
  B6   Yby/s (byproduct yield —         B16  carbon balance closure
        ethanol/acetate overflow)      B17  nitrogen balance closure
  B7   yield vs theoretical %           B18  degree-of-reduction balance
  B8   OUR from offgas                  B19  RQ profile pairwise comparison
  B9   CER from offgas                  B20  endpoint apparent Yx/s (fed-batch
  B10  RQ = CER/OUR + overflow flag           fallback when residual S unmeasured)

Tier C (16 metrics) — literature-assisted estimates with citations
  C1   OUR estimated (μ + Y_XO2_max + m_O2)
  C2   CER estimated (OUR × assumed RQ)
  C3   kLa back-calculated (OUR + DO steady state)
  C4   C* O2 saturation (Henry's law from T,P)
  C5   qs estimated (μ/Yx/s_max + m_S)
  C6   substrate consumed estimated (∫qs × X dt)
  C7   residual substrate back-calculated (S_fed − S_consumed_est)
  C8   metabolic heat (ΔH_combustion × substrate rate)
  C9   kLa from Van't Riet correlation (P/V + vvm)
  C10  biomass soft-sensed from CER
  C11  biomass soft-sensed from base addition
  C12  product from qp × ∫X dt
  C13  C* hydrostatic-head corrected
  C14  shear rate from tip speed
  C15  mixing time (Nienow correlation)
  C16  OTR/OUR consistency check
```

**Why this matters more than persona prose or anti-cascade rules:**

The reason their tier-A analyst is so strong is **not** because the prompt is brilliant — the prompt is fine, ours is comparable. The strength comes from the LLM operating against a 60-metric catalog rather than against an open prompt. When the agent is told "compute every applicable metric on this checklist or explicitly mark as data_gap," it produces depth. When ours is told "find patterns in the trajectories," it produces 2-3 high-confidence patterns and stops.

This is exactly the deferred Layer A from the trajectory_analyzer plan, surfaced as a real artifact rather than as a future intent.

**Each catalog entry carries:**

```python
@dataclass(frozen=True)
class CatalogEntry:
    metric_id: str             # "A8_specific_growth_rate_timecourse"
    tier: Tier                 # "A" | "B" | "C"
    short_description: str     # "µ(t) from Savitzky-Golay on ln(X)"
    long_description: str      # 2-4 sentences with formula + caveats
    applies_to: str            # "any run with ≥ savgol_window biomass timepoints"
    required_inputs: tuple     # InputSpec with CSV column hints
    required_parameters: tuple # ParamSpec with literature defaults + source_tag
    output_shape: OutputShape  # scalar | timecourse_csv | table_csv | ...
    output_columns: tuple      # column names for CSV outputs
```

**Discipline encoded in the catalog itself:**
- Every Tier C metric has a literature source baked in (`source_tag="S. cerevisiae, Verduyn 1990"`, `"Nienow 1998"`, `"Van't Riet 1979"`, etc.). Citations aren't a prompt rule — they're schema-enforced.
- `is_precomputable()` returns True iff every parameter has a default. Catalog can pre-compute everything without specialist input; specialists override params on demand.
- CSV column hints (`csv_column_hints=("OD", "OD600", "WCW", "DCW", "VCD")`) let the analyst auto-locate inputs in real-world messy data. This is what their `CHART_SYSTEM_PROMPT` warns about ("column names are NEVER what you expect") — handled in the catalog rather than left to the LLM.
- Cross-references (`source_tier="derived"`) declare which inputs come from earlier metrics — the catalog is a DAG.

**Steal:**

1. **The full taxonomy structure** (`CatalogEntry`, `InputSpec`, `ParamSpec`, tier system). Drop `metric_id` into our `Finding.statistics["metric_id"]` field — already stash-eligible since `statistics` is `dict[str, Any]`. No schema changes needed.

2. **The metric IDs themselves as the `pattern_kind` vocabulary.** Today our trajectory_analyzer emits `pattern_kind: "phase_boundary"` etc. — strings the LLM invents. Replace with `pattern_kind ∈ {A1..A24, B1..B20, C1..C16}`. Consequence: stable cross-bundle comparison ("did the analyzer detect A10 phase segmentation on this bundle? what about A14 DO margin alerts?").

3. **The literature-citation discipline for Tier C.** Every Tier C estimate carries a paper reference. We have `confidence_basis="process_priors"` but not a citation field. Adding `Finding.statistics["literature_source"]` populated by the catalog gets us 90% of the audit value with zero schema change.

4. **The "everything on the checklist gets a verdict — no silent skipping" prompt rule** from `ANALYZE_SYSTEM_PROMPT` line 162: "Every metric on the checklist gets a verdict: a computed value, OR a Data Gap entry with the specific reason. Silent skipping is forbidden." Prevents the trajectory_analyzer from quietly producing 2 findings when 24 were possible.

5. **The CSV column hints.** When characterize encounters real PDFs/CSVs with messy headers, having `csv_column_hints=("OD", "OD600", "WCW", "DCW")` lets the agent map raw column names to canonical metric inputs without guessing.

**Don't steal:** the metric catalog wholesale into our schema. Their `CatalogEntry` is a dataclass; our `Finding` is Pydantic. Keep them separate — the catalog is a **registry the analyzer consults**, not a schema the rest of the pipeline depends on.

**Land in:**

- New file: `src/fermdocs_characterize/agents/metric_catalog.py` — port the 60 entries (or a subset relevant to your initial use cases). Even half of Tier A alone (12 metrics) would 4× the depth of our trajectory_analyzer's output.
- `src/fermdocs_characterize/agents/trajectory_analyzer.py` prompt — replace open-ended "find patterns" with "for each metric in the catalog, either compute or data-gap. Silent skipping forbidden."
- `Finding.statistics["metric_id"]` — populate from catalog when emitting trajectory_pattern findings. Enables cross-bundle queries like "show me every A8 finding across all bundles."

**Cost estimate:** ~400 LOC for porting the catalog + ~50 LOC for prompt changes + ~30 LOC for population in coercer. Bigger than the prompt-only steals but **the single change most likely to make the system materially better**. This is what the user was reacting to in "tell me what we can steal" — the depth comes from this catalog.

**Trigger:** Now, if we want a real depth jump on the next IndPenSim/carotenoid run. Otherwise next time we touch the trajectory_analyzer.

---

## Tier S — steal soon (high value, low risk)

### 1. Specialist persona library (config-driven)

**Source:** `config/agents.yaml` + `src/agents/panel_prompts.py:_SPECIALIST_EXPERTISE`

What they have: **7 specialists** declared in YAML, each with:
- `id`, `name`, `role`
- `expertise_tags` — tag list for routing
- `personality` — one-line voice anchor
- `speak_threshold` — when this specialist should chime in (used by the moderator)

Plus a parallel `_SPECIALIST_EXPERTISE` dict with rich domain prose
("Red flags you own: high late-phase cell death, plasmid loss…") that
gets pasted into the system prompt.

**What we have:** 3 specialists hardcoded as `Literal["kinetics", "mass_transfer", "metabolic"]` in `src/fermdocs_hypothesis/schema.py`.

**Steal:**
- The **YAML registry pattern** — `config/agents.yaml`. Drives the
  specialist-routing plan we already drafted (`plans/2026-05-04-specialist-routing.md`).
  When we lift `SpecialistRole` from Literal to open-string registry,
  this is the format to use.
- The **personality + expertise prose** for our matching domains.
  Specifically the "Red flags you own" wording is genuinely useful — it
  sharpens specialist focus. Their `fermentation_physiologist` and
  `scaleup_specialist` blocks map almost 1:1 to our `kinetics` and
  `mass_transfer` specialists.
- The **`speak_threshold`** field — this is the tag-overlap routing
  signal we'd implement in Layer A1 of the routing plan.

**Don't steal:** their 7-specialist taxonomy verbatim. Some of their
specialists (contamination_expert, downstream_expert) don't fit our
characterize/diagnose/hypothesis pipeline yet. Pick the ones that do
when the time comes.

**Land in:** `config/agents.yaml` (new) when we build the routing layer.
Map their persona blocks into our existing 3 specialists' system prompts
today as a free upgrade.

---

### 2. Tier A/B/C analysis vocabulary

**Source:** `src/agents/analyze_prompt.py:ANALYZE_SYSTEM_PROMPT` (lines 134-247) + `src/tools/analysis_catalog.py`

What they have: a structured vocabulary for analysis depth:

- **Tier A** = direct/measured (always cheap, almost always computable)
- **Tier B** = derived from measurements (substrate timecourse, off-gas, kLa) — often missing, honest data-gaps expected
- **Tier C** = literature-assisted estimates with citations — for organism reference values

Each metric in the catalog has:
- `metric_id` (stable snake-case)
- `tier` (A/B/C)
- `short_description` + `long_description`
- `required_inputs` (columns)
- `required_parameters` (with defaults — Tier C always has defaults)
- `output_shape`

**What we have:** our `Finding.tier` enum already has A/B/C. We don't have the per-metric catalog or the prompt-level tier discipline.

**Steal:**
- The **prompt's tier-aware reasoning rules** verbatim:
  - "Tier hints: Tier A = should usually be computable. Tier B = needs
    measured inputs. Often missing; honest data-gap entries are
    expected. Tier C = literature-assisted estimates."
  - "Always check `get_tier_assumptions` before citing a Tier C number"
  - "Prefer Tier A/B numbers when both exist for the same quantity"
- The **language test** in their prompt (line 226-232) — "✅" / "❌"
  examples of acceptable vs forbidden phrasing. Concrete, hard to misread.
- The **forbidden sections** rule (line 214-224) — "Hypotheses /
  Recommendations / Interpretations" are explicitly disallowed in the
  analyst tier so the output stays observational. Our characterize agent
  could use this discipline when it's tempted to interpret.

**Don't steal:** the full metric catalog if it doesn't match our schema.
Their catalog is built around their specific finding types
(A14_max_biomass, A19_cross_run_KPI). We'd want to either adopt their
metric_id convention into our `Finding.statistics["metric_id"]` field, or
build our own catalog driven by our finding types.

**Land in:** `src/fermdocs_characterize/agents/trajectory_analyzer.py`
prompt — specifically the rules section. The "language test" pattern
should also land in `src/fermdocs_diagnose/agent.py` where we already have
similar forbidden-words discipline but their phrasing is sharper.

---

### 3. Cross-examination structure (anti-cascade rules)

**Source:** `src/agents/panel_prompts.py:CROSS_EXAM_TURN_PROMPT` (lines 353-422)

What they have: when specialists speak in sequence, each new specialist
must **explicitly engage with prior turns**. The prompt has hard rules:

> "**Engage with prior turns.** If ANY prior turns exist, you MUST
> explicitly reference at least one by name (`Dr. Scale claimed X — here's
> why I think that holds / fails / needs qualification…`). Silent
> concurrence is noise."
>
> "**Anti-cascade.** If you AGREE with a prior turn, you MUST add either
> (a) new evidence, (b) new reasoning, or (c) a specific constraint on
> when that agreement holds. Endorsement alone is rejected."
>
> "**Disagreement is welcome.** Silence on a wrong prior claim is
> complicity."

**What we have:** our specialists see `prior_facets_this_topic` but don't
have anti-cascade rules. They can rubber-stamp each other.

**Steal:** the **anti-cascade + must-engage-by-name** rules verbatim.
This is one of the highest-leverage prompt edits we could ship — it
directly attacks the "all specialists agree because the first one set the
frame" failure mode.

**Don't steal:** the panel/round-types machinery (cross_exam vs
generative vs reconciliation vs challenge — see Tier A below). That's a
bigger lift.

**Land in:** `src/fermdocs_hypothesis/agents/specialist_base.py` (or
wherever the per-role specialist prompts live) — add the engage-by-name
+ anti-cascade rules to the specialist invariants. ~30 LOC change, big
behavioral upgrade.

---

### 4. Devil's advocate (challenge round)

**Source:** `src/agents/panel_prompts.py:CHALLENGE_TURN_PROMPT` (lines 425-458)

What they have: when a hypothesis approaches consensus (≥3 supporters,
0 refuters), the moderator can call a **challenge round** — exactly ONE
specialist is cast as devil's advocate against it. Their job is NOT
balanced: it's to find the strongest case AGAINST. Specifically:

- Identify the WEAKEST supporting evidence on the board
- Propose at least one alternative explanation that fits the same evidence
- Name specific data that, if present, would refute the hypothesis
- If there's a fatal flaw, propose `argue_for_rejection`

**What we have:** our critic plays this role for every hypothesis, but
not in the "you might be wrong, prove me wrong" framing. The critic checks
for citation discipline + scope drift; it doesn't aggressively try to
construct counterfactuals.

**Steal:** the **devil's advocate framing as a separate critic mode**.
When a hypothesis on retry ALSO gets a green flag from the critic
unanimously — that's exactly the consensus-cascade failure mode their
challenge round catches. Adding a "challenge mode" trigger in our critic
that fires when a hypothesis seems "too easy" (high confidence + no real
red flags + retry depth = 0) would be a strong addition.

**Don't steal:** their full round-type state machine. Their moderator
picks between cross_exam / generative / reconciliation / challenge /
call_vote — too much for our current architecture.

**Land in:** `src/fermdocs_hypothesis/agents/critic.py` — add a
`challenge_mode` flag to `CriticView` populated by the runner when
heuristics suggest the hypothesis is sailing through unchallenged.
Critic prompt branches on it.

---

## Tier A — steal next (medium value, medium effort)

### 5. HypothesisBoard pattern (shared mutable state across rounds)

**Source:** `src/tools/board_tools.py` + `src/graph/panel_state.py:HypothesisBoard`

What they have: a **HypothesisBoard** that lives in graph state.
Every specialist action — propose, support, refute, update_stance,
propose_merge — is a mutation against the board. Board carries:

- List of `Hypothesis` objects with `{id, title, role, status, lead,
  proposer, stance_tally, evidence, what_we_need}`
- Status auto-computed from stance_tally (`compute_status`)
- Mutations are pure functions (board → new_board), no side effects

This makes it possible for specialists to **see and modify each other's
hypotheses across rounds**, not just contribute facets that get merged
once.

**What we have:** our hypothesis is a single output of synthesizer per
topic per turn. Specialists contribute facets, synthesizer merges, that's
it. There's no shared state where multiple hypotheses coexist and get
voted on.

**Steal:** the **board-as-shared-state pattern** when we eventually want
multi-hypothesis-per-topic debate. Today our model is "one topic → one
hypothesis (with retries)." Their model is "one topic → N hypotheses,
specialists propose/refute/merge, vote at end." Their model is more
like real scientific debate.

**Don't steal:** today. This is a foundational architecture shift, not a
prompt edit. It would replace our current synthesizer-emits-one-hypothesis
flow with a propose-debate-vote flow. Significant PR. **Defer until** we
hit the limits of single-hypothesis-per-topic — most likely when users
say "the system should consider multiple competing explanations of the
same phenomenon."

**Land in:** future architectural plan — bundle with the LangGraph
migration we already deferred. The HypothesisBoard maps directly to a
LangGraph state.

---

### 6. Tool catalog rendered into prompts (single source of truth)

**Source:** `src/tools/specialist_tool_bundle.py:render_tool_catalog`

What they have: tool docstrings ARE the tool catalog. Their
`render_tool_catalog(tools)` reflects on the actual tool list and
generates a markdown bullet list — `- \`tool_name\` — first line of
docstring.` — that gets injected into every specialist prompt.

This means: rename a tool or reword its docstring, and **every prompt
that mentions that tool updates automatically**. No stale hand-typed
copies.

**What we have:** prompts hand-list tool names (`SYNTHESIZER_TOOL_HINTS`,
`CRITIC_TOOL_HINTS`). When we rename a tool, we have to update every
prompt that mentions it.

**Steal:** the **render-tool-catalog-from-tools** pattern. Replace our
`ToolHint` tuples with reflection over the actual tool list.

**Don't steal:** their tool surface itself. Their tool naming
conventions don't all match ours.

**Land in:** `src/fermdocs_hypothesis/prompts.py` — replace
`ToolHint`-based catalog with reflection. ~50 LOC. Pays off every time
we add or rename a tool.

---

### 7. Bundle-curried tool factory

**Source:** `src/tools/specialist_tool_bundle.py:make_specialist_tools`

What they have: tools take `bundle_dir` once at construction time, not
per-call. The LLM sees clean signatures like `get_batch_metric(batch_id,
metric)` instead of `get_batch_metric(bundle_dir, batch_id, metric)`.
Hides plumbing from the model.

Bonus: every tool result runs through `_sanitize_json` to replace
`NaN`/`Infinity` with `None` because Gemini rejects raw NaN in tool
outputs. This is a real bug they hit and we will too.

**What we have:** `HypothesisToolBundle` already does this for read
tools. The pattern is consistent.

**Steal:** the `_sanitize_json` helper — explicit guard against
Gemini-rejecting-NaN. Not yet a problem in our system but **will be** the
moment a tool returns a pandas DataFrame with missing cells. Cheap
preventive fix.

**Land in:** `src/fermdocs_hypothesis/tools_bundle/factory.py` — wrap
every tool result in `_sanitize_json` before returning. ~10 LOC.

---

### 8. Lineage at extraction time (per-cell provenance JSON)

**Source:** `src/agents/analyze_prompt.py:EXTRACT_SYSTEM_PROMPT` (lines 30-56) + `src/tools/cell_lineage.py`

What they have: when extraction produces a CSV from a PDF, it ALSO
produces a `<csv_stem>.lineage.json` with per-cell provenance:

```json
{
  "csv_filename": "biomass_data.csv",
  "cells": [
    {
      "row": 0,
      "column": "biomass_g_l",
      "value": "24.7",
      "source": {
        "file": "report.pdf",
        "page": 12,
        "table": 3,
        "row_in_source": 5,
        "col_in_source": 2,
        "raw_text": "24.7"
      }
    }
  ]
}
```

Every cell traces back to a PDF page + table + cell. Wet-lab scientists
can audit any number in any chart back to the source.

**What we have:** narrative_observations have `source_locator` (page,
section, paragraph_index, char_offset) but our table-derived findings
don't carry per-cell provenance. We have observation_ids that link to
the dossier, but the dossier→PDF link is implicit.

**Steal:** the **per-cell lineage JSON as a first-class artifact**. This
is the audit trail wet-lab consumers will demand the moment they spot a
suspicious number. Bonus: it makes regression tests on extraction
quality trivial (just diff lineage JSONs).

**Don't steal:** their exact JSON schema verbatim — we'd want to align
it with our existing `observation_id` namespace.

**Land in:** new module `src/fermdocs/parsing/cell_lineage.py`. Hook
into the existing PDF extraction path to emit lineage alongside the
parsed tables. Bigger lift than other items here (~200 LOC + tests +
bundle writer changes), but the audit story is genuinely valuable.

---

## Tier B — read for inspiration, don't directly steal

### 9. Moderator action types (round selection)

**Source:** `src/agents/panel_prompts.py:MODERATOR_ACTION_PROMPT` (lines 275-345)

Their moderator picks between five round types each iteration:
- `cross_exam_round` (default)
- `generative_round` (when stuck — propose new hypotheses)
- `reconciliation_round` (when 2+ hypotheses overlap)
- `challenge_round` (when consensus is too easy)
- `call_vote` (terminate)

Each has explicit rules: panel size, panel selection criteria
(supporter+non-supporter for cross-exam, EXACTLY one non-supporter for
challenge, etc.), and termination conditions.

**Why don't directly steal:** assumes the HypothesisBoard architecture
which we don't have. But the **structured round-type vocabulary** is the
right shape for our future debate flow when we do have multiple-hypotheses
per topic.

**Use:** as the canonical reference when we eventually rebuild the debate
loop with multiple coexisting hypotheses. Save these prompts.

---

### 10. Stance vote (final ballot) + stance tally → status

**Source:** `src/agents/panel_prompts.py:STANCE_VOTE_PROMPT` (lines 539-565) + `src/graph/report_models.py:compute_status`

Their final phase: every specialist votes on every hypothesis with
`{stance: support|refute|insufficient, confidence: float, rationale:
str}`. Status computed from stance_tally:

- ≥3 supports + 0 refutes → consensus_leaning
- ≥3 refutes → rejected
- mixed → actively_contested
- mostly insufficient → unresolved

**Why don't directly steal:** voting is structurally weaker than our
critic+judge adversarial check. **But** the per-specialist
stance+confidence+rationale schema is useful for the audit trail — when
hypothesis X is accepted, knowing which specialists supported it with
what confidence is valuable downstream.

**Use:** if we ever expand to "every specialist sees the synthesizer's
output and votes individually" as an additional check beyond the critic,
this is the schema to use.

---

### 11. Phase 1 → Phase 2 case file structure

**Source:** `src/agents/panel_prompts.py:INTAKE_PROMPT` + `_INITIAL_ANALYSIS_SHARED`

Their structured handoff between extraction/analysis (Phase 1) and panel
debate (Phase 2):

> "A structured `CaseFile` in the user message:
> - The user's question
> - Experiment overview (organism, product, process, scale, equipment)
> - Cross-run KPI table (Phase 1 Tier A output)
> - Per-tier data gaps flagged by Phase 1
> - Pointers to derived CSVs"

Plus the framing note: "**The CaseFile is framing, not evidence. You
MUST fetch real numbers via tools before committing to any claim.**"

**Why don't directly steal:** maps onto our characterize → diagnose
boundary, which already does similar work via `AgentContext`. But the
explicit "framing not evidence" framing is sharp.

**Use:** add the "framing not evidence — fetch real numbers" line to our
diagnose agent's prompt. Subtle but reinforces the existing
`execute_python`-default policy.

---

## Tier C — explicitly DO NOT steal

### Voting model
Their consensus-by-vote is structurally weaker than our critic+judge
adversarial structure. Voting is bad at catching "the synthesizer
paraphrased its way around a real problem." Keep our model.

### Loose schemas
Their schemas (`InitialAnalysis`, `Hypothesis`, `Stance`) are pydantic
but looser than ours — fewer namespace validators, looser citation
discipline. Our Pydantic strictness is a moat. Keep it.

### LLM-first characterize
They run LLM analysis from the start. We run deterministic spec checks
first, LLM trajectory_analyzer second. Our reproducibility story is
better. Keep the order.

### Single-graph LangGraph orchestration
They run everything as one big LangGraph. Ours is split into
characterize → diagnose → hypothesis with file-based bundle handoff.
Our split is more debuggable and more independently testable. Don't
unify.

---

## Concrete steal-now PR options

### Option α — afternoon-sized, prompts + tooling only (~150 LOC)

If we want to ship gains from this audit **without architectural changes**,
the highest-leverage prompt+tooling PR is:

1. Add **anti-cascade + engage-by-name** rules to specialist prompts (steal #3)
2. Add **devil's advocate trigger** to critic when hypothesis sails through (steal #4)
3. Add **`_sanitize_json`** wrapper to all tool results (steal #7)
4. Add **"framing not evidence"** line to diagnose prompt (steal #11)
5. Add the **language test** examples to characterize trajectory_analyzer prompt (steal #2)
6. Pull the relevant **persona prose** from their `_SPECIALIST_EXPERTISE` into our 3 specialists' prompts as expertise-block expansions (steal #1, partial)

Total: ~150 LOC. Could ship in a single afternoon and the specialists
would be visibly sharper. **Does NOT close the depth gap** that the
reference repo has on initial analysis — it just sharpens what we
already do.

### Option β — depth-jump PR (~600-800 LOC)

If we want our system to match the reference's analytical depth on
unknown_process bundles like IndPenSim, the headline addition is:

1. Port (a useful subset of) the **metric catalog** into
   `src/fermdocs_characterize/agents/metric_catalog.py` (steal #0).
   Even just Tier A (24 metrics) is enough — Tier B/C can come later.
2. Rewrite trajectory_analyzer prompt to **operate against the catalog**
   instead of open-ended pattern discovery: "for each metric_id in the
   catalog, either compute or data-gap. Silent skipping forbidden."
3. Populate `Finding.statistics["metric_id"]` and
   `Finding.statistics["literature_source"]` from the catalog.
4. Plus all of Option α (the prompt+tooling work compounds).

Total: ~600-800 LOC. **This is what would actually close the depth gap**
on the reference repo. The 2-finding IndPenSim output we got today
becomes a 12-20 metric Tier A audit with phase segmentation, μ(t),
Qp_phasewise, DO margin alerts, cross-run KPI table — all the things
their analyst produced in the log we watched.

Recommendation: ship Option β. The catalog is the moat their system
has and we don't. Everything else in the steal-list is a sharpening of
what we already do.

---

## When to revisit

- **When we add specialist #4** → bring in steals 1 (full YAML registry),
  3 (anti-cascade), 4 (challenge mode). Bundle with the existing
  specialist-routing plan.
- **When we ship HITL with user-question support** → use steal 11
  (CaseFile structure) as the framing for our user-question integration.
- **When we hit "users want multiple competing hypotheses"** → bring in
  steals 5 (HypothesisBoard), 9 (moderator round types), 10 (stance vote).
  This is the LangGraph migration moment.
- **When wet-lab consumers ask "where did this number come from?"** →
  bring in steal 8 (per-cell lineage JSON).

---

## Cross-references

- `plans/2026-05-04-specialist-routing.md` — companion plan; steals 1, 3, 4 align with it.
- `plans/2026-05-04-user-question-and-hitl.md` — steal 11 aligns with the CaseFile structure.
- Reference repo: `~/fermentation-debate-langgraph/`
- Specific file paths in this doc are relative to that clone.
