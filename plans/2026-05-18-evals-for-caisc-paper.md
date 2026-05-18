# Evals for CAISc 2026 paper

**Branch**: `evals-for-paper` (off `frontend-styling`)
**Created**: 2026-05-18
**Status**: planning — no code yet

## Why this plan exists

The CAISc 2026 paper has a Reproducibility & Responsibility checklist and an
§4 Evaluation slot. fermdocs is a working system with opinionated architecture
but no quantitative numbers on record. Without evals the paper is "we built
this thing"; with evals it's "we built this thing and here's what it buys you."

User decisions (2026-05-18):
- **Scope**: all three eval families, but reshaped for small-N reality (see below)
- **Ground truth**: 2 real bundles total — 1 yeast, 1 indpensim. No within-family pairs.
- **Rater**: LLM-as-judge v1, disclosed in AI Involvement checklist
- **Bundle state**: both need re-ingest in Phase 0

## Reshape for N=2 reality

The original plan assumed 6+ bundles. With 2, we keep all three eval families
but change what each one claims:

- **E2 becomes the quantitative headline** — synthetic hypotheses don't need
  real bundles, so the 40-hypothesis P/R + confusion matrix is unaffected and
  is now the paper's strongest number.
- **E1 downgrades to mechanism demonstration** — run the same bundle twice,
  cold then warm, and show that injected lessons get cited and change critic
  outcomes. This is not a generalization claim; it's "the memory loop closes
  end-to-end."
- **E3 becomes 2 case studies** — yeast and indpensim, each compared against
  single-shot Gemini with the same raw bundle, LLM-judged for specificity and
  grounding. Per-bundle qualitative narrative, no aggregate preference rate.
  Honest framing: "we have 2 bundles; here is what each shows."

## What we are NOT doing

- No human-rater protocol (too slow for paper timeline)
- No claim of generalization beyond fermentation process bundles
- No comparison against other agentic frameworks (LangGraph, AutoGen, etc.) — too much surface area; we compare against the single-shot LLM baseline only
- No statistical significance claims at N<10. Report effect sizes + ranges, not p-values.

## Three eval families

### E1 — Memory mechanism demonstration (downgraded from A/B)

**Claim** (revised): the memory loop closes end-to-end — lessons emitted on
run 1 are persisted, retrieved on run 2, cited by the synthesizer, and change
critic outcomes. This is a mechanism claim, not a generalization claim.

**Setup**:
- Per bundle (yeast, indpensim), run the pipeline **twice**:
  - **Run 1 (cold)**: `FERMDOCS_MEMORY=synap`, no priors yet. System emits
    lessons on clean exit.
  - **Run 2 (warm)**: same bundle, same memory backend, lessons from run 1
    now retrievable.
- Repeat per bundle with 2 seeds for run-2 variance.

**Metrics** (per bundle):
1. **Lesson emission count** (run 1) — did the system actually write anything
2. **Lesson activation rate** (run 2) — fraction of retrieved lessons cited
   in synthesizer output (visible in the `cross_run_lessons` block)
3. **Critic axis fire delta** — which critic axes that fired in run 1 do
   *not* fire in run 2 (priors prevented their own re-trigger)
4. **Specificity score** (LLM-judge) on final hypothesis, 1-5

**What this does NOT claim**: that memory generalizes across runs of *different*
bundles in the same family. We don't have the data for that.

**Output for paper**: small table, 2 bundles × 4 metrics × (cold, warm), with
a worked-example callout showing one retrieved lesson being cited.

### E2 — Critic-axes precision/recall (validates framework)

**Claim**: the multi-axis critic catches axis-specific defects at axis-specific
rates, justifying the per-axis design over a single-pass critic.

**Path B (full-pipeline) chosen** over Path A (standalone critic) because the
paper's strongest defense is "we tested the system as deployed."

**Mitigations for Path B's known problem**:
- **Synthesizer-fixes-the-defect confound** — fixtures use *force-commit
  mini-bundles*: leading user question, sparse evidence, dossier hints that
  make the planted defect the path of least resistance. Engineering work
  per fixture, but preserves per-axis P/R as the headline metric.

**Model strategy**: all-pro (`gemini-3-pro`) across orchestrator, specialists,
synthesizer, critic, judge. Mixed-model was planned but LiveHooks shares one
client across the hypothesis stage today, and a per-agent override refactor
is out of scope for the paper. Budget revised to $60-120 (approved).

**Setup**:
- 40 fixture mini-bundles. Each is a runnable bundle directory (dossier +
  characterization + diagnosis + meta) carrying:
  - a labeled defect axis (one of 7) or "clean"
  - a leading user question that forces the synthesizer toward the defect
  - sparse-but-pointed evidence so the synthesizer commits early
- 5× clean (control), 5× per axis = 5 × 8 = 40 total
- Within each defect axis: 3 clear-cut + 2 borderline
- Run full hypothesis pipeline on each. Record critic axis fires from the
  ratified hypothesis output (and from the rejection chain leading to it).

**Metrics** (revised 2026-05-18 after dry-run finding):

The 2026-05-18 validation dry run on e2-traj-clear-01 surfaced an
important finding: the critic correctly *detected* the trajectory defect
(reasoning chain talked about ignoring trajectories and growth rate), but
*tagged* the rejection with [question-axis] because the leading question
also demanded growth rate. This is a real finding about taxonomy
crispness, not a bug — and reframes the eval as **two-metric**:

1. **Catch rate** (headline, per-axis): of N defect fixtures on axis A,
   how many got any axis fired? Plus false-positive rate over clean
   fixtures. This is the "did the critic see the problem" metric.
2. **Tag accuracy** (secondary): of caught defects on axis A, how many
   had axis A in the fired set? This is the "did the taxonomy match
   the critic's reasoning" metric.
3. **7×8 confusion matrix** still reported for color, plus
4. **Pipeline iteration count** per fixture.

The paper claim becomes "critic catches axis-A defects at rate X; the
taxonomy assigns the labeled axis at rate Y where Y<=X." Both numbers
honest.

**Authoring**: I draft all 40, user spot-checks 5 clean + 5 borderline (the
trickiest ones) before any pipeline runs.

**Output for paper**: confusion matrix figure + P/R table + appendix
example bundle showing the planted defect.

**Limitations disclosed in paper**:
- Synthetic defects may not match the distribution of real failures
- Mixed-model pipeline is not the production default
- Mini-bundles are smaller than real bundles (one-batch trajectories), so
  characterization-stage signals are weaker than typical deployment

### E3 — End-to-end vs single-shot baseline (strongest claim, most expensive)

**Claim**: the multi-agent pipeline produces hypotheses preferred over a
single-shot LLM given the same bundle.

**Setup**:
- Take all available bundles (real + synthesized) — target N=8-12
- **Treatment**: full fermdocs pipeline → final ratified hypothesis
- **Baseline**: single Gemini call with the raw bundle files + the same user
  question dumped into one prompt. Same model (`gemini-3-pro`), same temperature.
- **Judge**: separate LLM (`claude-opus-4-7` or `gemini-3-pro` in different
  configuration) given both outputs **blind, randomized order**. Asked:
  "which hypothesis is more specific, evidence-grounded, and actionable?"
- Each pair judged 3× with different judge seeds to estimate judge variance.

**Metrics**:
- Preference rate (treatment wins / total)
- Judge agreement (how often do the 3 seeds agree)
- Per-bundle breakdown (does treatment win more on hard bundles?)

**Mitigation for LLM-judge bias**:
- Disclose judge model in paper
- Counterbalance order
- Report a known issue: LLM judges tend to prefer longer outputs. Include a
  length-controlled secondary metric (preference at matched output length).

**Output for paper**: preference rate with CIs (bootstrap), example pair in
appendix.

## Plan structure

### Phase 0 — eval harness scaffolding (1 day CC)

New package: `src/fermdocs_eval/`
- `harness.py` — `EvalRun` dataclass, run-and-record loop
- `judges.py` — `llm_judge_specificity()`, `llm_judge_preference()`, with
  prompt templates checked into repo
- `metrics.py` — preference rate, P/R, confusion matrix
- `synthetic_bundles.py` — generators for the planted-defect bundles used
  in E2 (and where needed in E1/E3)
- CLI: `python -m fermdocs_eval run --suite e1|e2|e3 --out results/`

Output format: one `results.jsonl` per suite, one row per trial, all raw
records so we can re-compute metrics without re-running.

### Phase 1 — E1 memory A/B (2 days)

- Pick 2 families
- Author or identify 6 bundles (3 per family — 2 for A/B test, 1 for prior-seeding)
- Run k=3 rotations × 2 conditions × 2 families = 12 pipeline runs
- LLM-judge specificity on the final hypotheses
- Write `eval/e1_memory_ab.md` reporting numbers

### Phase 2 — E2 critic axes (2 days)

- Author 40-hypothesis test set (this is real labor — most of the eval cost)
- Run critic on each, record axis fires
- Compute matrix + P/R
- Write `eval/e2_critic_axes.md`

### Phase 3 — E3 end-to-end vs baseline (2 days)

- 8-12 bundles total (pad with synthetic if real count is low)
- Run full pipeline + single-shot baseline on each
- 3× judge seeds per pair
- Bootstrap CIs
- Write `eval/e3_end_to_end.md`

### Phase 4 — paper integration (1 day, on the paper branch not this one)

- Pull the 3 eval markdowns into the paper §4
- Generate final figures (matplotlib for paper style — not the editorial
  Plotly used in the app)
- Update AI Involvement checklist with judge-model disclosure

## Risks and honest limitations

- **LLM-judge bias**: known and disclosed. Mitigated by counterbalancing and
  length-control secondary metric.
- **Synthetic ground truth**: the planted defects in E2 may not match the
  distribution of real defects. Stated as a limitation; future work = label
  real critic flags from production runs.
- **N is small**: 8-12 bundles is a case study, not a benchmark. We'll frame
  it as such in the paper and not claim generalization.
- **Self-evaluation**: we are both the system authors and the eval designers.
  Mitigation: prompts and harness are open-source in the paper supplement;
  judge model is different from any model in the pipeline where possible.

## Files this plan will produce

```
src/fermdocs_eval/
  __init__.py
  harness.py
  judges.py
  metrics.py
  synthetic_bundles.py
  cli.py
tests/unit/eval/
  test_judges.py
  test_metrics.py
eval/
  e1_memory_ab.md
  e2_critic_axes.md
  e3_end_to_end.md
  results/  (gitignored, regenerable)
  prompts/  (judge prompt templates, checked in)
```

## Decisions locked (2026-05-18)

1. **N=2 reality**: 1 yeast + 1 indpensim bundle. E1 reshaped to mechanism
   demo, E3 reshaped to 2 case studies. E2 unaffected.
2. **E3 baseline model**: `gemini-3.1-pro-preview` — strong baseline so any
   pipeline win is meaningful, not an artifact of model mismatch.
3. **Budget**: approved (~$10-30 in `GEMINI_API_KEY` spend).
4. **Bundle state**: both need re-ingest in Phase 0 from
   `data/files/yeast_batch_run_3.csv` and
   `data/files/IndPenSim_V2_export_V7.csv`.
5. **User question**: per-bundle tailored. To be filled in before E1/E3 runs:
   - yeast: TBD
   - indpensim: TBD
