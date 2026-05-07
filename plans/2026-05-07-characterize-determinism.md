# Characterize stage: deterministic per-run runner + product KPIs

**Branch:** `characterize-determinism` (off `frontend-redesign`)
**Status:** Plan locked — ready for commit 1.
**Companion plans:**
- `plans/2026-05-04-metric-catalog-and-toolkit.md` — the catalog this builds on.
- `plans/2026-05-05-hitl-followup.md` — PR-A2 hypothesis-stage drive posture.
- `plans/2026-05-07-multi-file-upload-and-submit.md` — frontend redesign
  predecessor.

**Estimated cost:** ~740 LOC across 6 commits + 28 tests.

---

## Why

Real-world feedback on run `406c524f-61fc-4da2-b75a-2ff290407a6d` (IndPenSim,
2 runs, penicillin fed-batch) revealed five distinct failure modes that
combine to make the agent miss the obvious answer. The user manually
computed metrics on both runs and concluded "RUN-2 performed ~2× better
(final titer 30.4 vs 14.3 g/L, mean RQ 0.90 vs 1.21, PAA consumed vs
wasted)." Our pipeline said: *"a definitive conclusion cannot be reached—
equivalent kinetic and metabolic metrics are absent for RUN-0002."*

The data was there. The agent did not compute it. Five reasons:

1. **Asymmetric metric extraction.** The trajectory analyzer is an LLM
   that's TOLD to "emit ONE pattern per applicable run" but the loop is
   inside the model's reasoning. Under token pressure it computes A8/A9/
   A10/A11/B6/B10 once for RUN-1, marks the catalog row "covered," and
   moves on. RUN-2 silently goes uncomputed.
2. **Missing product-KPI registry.** The catalog has no
   `final_product_titer`, no `peak_product_titer`, no
   `titer_decline_after_peak`. For penicillin the *primary*
   performance metric is final P titer, and the catalog can't surface it.
3. **Unit / physicality validator gap.** A finding `"PAA yield 204.5 g/g"`
   passed citation discipline, prompt invariants, critic, and judge.
   Yields > 1 g/g are non-physical. Nothing in the stack rejects it.
4. **Statistic choice is wrong for skewed time-series.** Mean RQ 1.21 vs
   median 0.98 tells different stories. Catalog hardcodes mean. For
   spike-prone signals (RQ, growth rate transients), median + IQR is the
   honest summary.
5. **False conservatism.** "Insufficient data" reads as rigor but is
   *prompt failure*: the synthesizer interpreted catalog-side absence as
   data-side absence. The data was there; the catalog runner didn't
   produce a finding for it. Without a tool-vs-data distinction, the
   synthesizer always falls back to "insufficient_data" — the safest
   option — even when the bundle could answer the question.

Fix all five together. Each is independently testable; they're shipped on
one branch because fix 5 reads outputs only fix 1 produces, and fix 6
(eval) gates all five.

---

## Eval gate (the boss test)

The IndPenSim 2-run bundle is the regression fixture. After this branch:

```python
def test_indpensim_two_runs_winner():
    output = run_full_pipeline(
        indpensim_2run_bundle,
        question="which run performed better?",
    )
    h = output.final_hypotheses[0]
    # Must mention RUN-2 as winner.
    assert "RUN-0002" in h.summary or "RUN-2" in h.summary
    # Must cite final titer for both runs.
    assert "30.4" in (h.plain_language_summary or h.summary)
    assert "14.3" in (h.plain_language_summary or h.summary)
    # Must NOT use 'insufficient_data' — the data IS sufficient.
    assert h.question_answered != "insufficient_data"
```

If this test does not pass at the end of the branch, the work is
incomplete.

---

## Architecture in one diagram

```
  BEFORE                                AFTER
  =======                               =====
  trajectory_analyzer (LLM)             1. catalog_runner.py (NEW, deterministic)
    - reads bundle                         for run_id in bundle.run_ids:
    - reads catalog checklist                for entry in ready_entries():
    - decides what to compute                  if applies(entry, run_id):
    - emits N findings                            try: result = entry.toolkit_fn(traj)
      (silently asymmetric)                       except: result = None
                                                  emit Finding(metric_id=..., run_id=...)
                                              -> N×M findings, fully populated

                                         2. validate_findings.py (NEW)
                                            for finding in findings:
                                              if violates_physicality(finding):
                                                convert to data_gap

                                         3. trajectory_analyzer (LLM, slimmed)
                                            - sees catalog runner's findings
                                            - emits ONLY open-ended findings
                                              for things outside the catalog
                                              ("unusual oscillation in O2",
                                              "narrative says X happened at Y")

                                         4. symmetry_check.py (NEW)
                                            assert metric coverage matches across runs;
                                            emit explicit data_gap for any (metric, run)
                                            asymmetry so synthesizer can flag tool-gap
```

---

## Locked decisions (3)

From the design discussion 2026-05-07:

### D1: One branch, six commits in dependency order
Not five separate branches — the fixes are tightly coupled. Fix 5
(symmetry validator) reads outputs only fix 1 (catalog runner) emits;
fix 6 (eval) only meaningfully passes after fixes 1–5 land. Splitting
would create artificial coupling between PRs.

### D2: Process-family routing in `process_priors.yaml`
Hardcoded in code = release-bound. YAML = operator-editable. The priors
layer already loads YAML for organism-specific ranges; product/precursor
routing fits the same shape:

```yaml
process_families:
  penicillin_fedbatch:
    product_variable: penicillin_g_l
    precursor_variables: [paa_mg_l]
    overflow_byproducts: []  # PAA is precursor here, not byproduct
  ecoli_acetate_overflow:
    product_variable: recombinant_protein_g_l
    precursor_variables: []
    overflow_byproducts: [acetate_g_l]
```

### D3: False-conservatism fix on BOTH synthesizer and critic
Synthesizer self-corrects first pass: "if metric_id appears for some runs
but not all, the absence is a TOOL gap not a DATA gap; draw conclusions
from what was computed." Critic is the safety net: "do NOT accept
question_answered='insufficient_data' when symmetry-validator findings
show tool-side gaps; flag with [tool-gap-axis] reason."

---

## File-by-file plan

### Commit 1 — Deterministic per-run catalog runner (~280 LOC)

- `src/fermdocs_characterize/agents/catalog_runner.py` (NEW): `MetricCatalogRunner`
  class with one entry point `compute_all(bundle) -> list[Finding]`. For
  each `run_id × ready_entry`, calls the toolkit_fn, wraps result in a
  `computed_metric` Finding or a `data_gap` Finding on exception.
  Deterministic: same bundle in → same findings out. Logs every (run, metric,
  outcome) tuple at DEBUG.
- **A2 fix — pre-flight import check**: `MetricCatalogRunner.__init__`
  imports every ready entry's `toolkit_fn` once at construction. If any
  required-tier module fails to import, raises `RuntimeError` with the
  failing module name, aborting characterize loud rather than silently
  filling the bundle with `"computation failed due to tool error"`
  data_gaps. Catches the same regression class as the carotenoid
  `f659194e` run.
- **Q1 fix — shared iteration helper**: `applicable_metric_run_pairs(bundle)
  -> Iterator[(metric_id, run_id)]`. Used by `compute_all` (commit 1)
  AND by `check_symmetry` (commit 5) so iteration shape is single-source.
- `src/fermdocs_characterize/agents/trajectory_analyzer.py`: prompt rewrite —
  remove the catalog-checklist responsibility. **A1 fix**: prepend an
  explicit `[ALREADY COMPUTED]` block to the prompt enumerating every
  metric_id the catalog runner produced findings for. Add hard rule:
  *"Do NOT re-emit any pattern with a metric_id from the [ALREADY
  COMPUTED] block. Your job here is open-ended findings only —
  trajectory patterns the catalog doesn't cover."* Replace
  `_build_metric_checklist` with `_build_catalog_summary`.
- `src/fermdocs_characterize/pipeline.py`: insert `catalog_runner.compute_all`
  call BEFORE `trajectory_analyzer.analyze`. Findings from both layers
  merge into the bundle. Pipeline ALWAYS overwrites bundle artifacts on
  re-run; never merges with prior findings (Q2 — documented invariant).
- `tests/unit/test_catalog_runner.py` (~210 LOC, 11 tests):
  1. single-run bundle emits all applicable metrics
  2. multi-run bundle emits N×M
  3. missing inputs emit data_gap
  4. toolkit exception becomes data_gap (not silent drop)
  5. **deterministic re-run produces identical findings (Q2 idempotency)**
  6. cross-run metrics (A19/A20/A21) emit once not N times
  7. organism-required Tier C metrics data_gap when organism is None
  8. run_ids list on emitted Finding is correctly scoped
  9. **A2: pre-flight import failure aborts loud with module name**
  10. empty `bundle.run_ids` returns empty findings cleanly
  11. `applicable_metric_run_pairs` helper used by symmetry check (Q1)
      returns same pairs as runner enumerated
- `tests/unit/test_trajectory_analyzer_prompt.py` (NEW, ~60 LOC, 2 tests):
  - **REGRESSION**: `[ALREADY COMPUTED]` block contains every catalog-runner
    metric_id when prompt is built (mock LLM, inspect prompt text).
  - **REGRESSION**: prompt no longer contains the per-bundle `_build_metric_checklist`
    enumeration that existed before commit 1 (anti-rebuild guard).

### Commit 2 — Product KPI tier + process-family routing (~230 LOC)

- `src/fermdocs/domain/process_families.py` (NEW): `ProcessFamilyConfig`
  dataclass + loader for `process_families.yaml`. Config carries
  `product_variable`, `precursor_variables`, `overflow_byproducts`.
- `data/process_families.yaml` (NEW): seed entries for `penicillin_fedbatch`,
  `ecoli_acetate_overflow`, `melanin_batch`, `unknown` (catch-all that
  routes to no product-KPI metrics).
- **A3 fix — bundle-time config-mismatch detection**: when a P-tier metric
  runs and the process family says `product_variable=penicillin_g_l` but
  the bundle has no such trajectory, emit ONE `[CONFIG_MISMATCH]` data_gap
  with reason `"process_families.yaml says
  penicillin_fedbatch.product_variable=penicillin_g_l, but this bundle
  has no such trajectory. Available: <list>"`. NOT per-metric data_gaps
  (avoids burying the user under five identical data_gaps when one
  config typo is the cause).
- `src/fermdocs_characterize/agents/metric_catalog.py`: new Tier P entries.
  `P1_FINAL_TITER`, `P2_PEAK_TITER`, `P3_TITER_DECLINE`,
  `P4_INTEGRAL_PRODUCTIVITY`, `P5_PRECURSOR_UTILIZATION`. Each declares
  `required_inputs` resolved through the process-family routing layer.
- `src/fermdocs_characterize/toolkit/products.py` (NEW): five toolkit_fn
  implementations as small pandas functions. Each handles trajectories
  with NaNs, sparse sampling, and missing precursor variables.
- `tests/unit/test_product_kpi.py` (~140 LOC, 8 tests):
  1. each toolkit_fn with a synthetic trajectory matching expected output ± 5%
  2. routing yaml load roundtrip
  3. missing-product-variable returns None cleanly
  4. monotonic-product (no decline) emits P3=0
  5. **declining-product (RUN-1 case 21.6 → 14.3): P3 fractional
     decline computed correctly** — boss-eval-relevant
  6. **precursor-as-input (PAA case): P5 utilization fraction = (input
     - residual) / input, NOT yield** — catches the polarity bug from
     IndPenSim feedback
  7. **A3 config-mismatch: missing `product_variable` emits ONE
     `[CONFIG_MISMATCH]` data_gap, not five per-metric data_gaps**
  8. unknown process family falls through cleanly to no-P-tier

### Commit 3 — Unit / physicality validator (~100 LOC)

- `src/fermdocs_characterize/agents/finding_validator.py` (NEW): function
  `validate_finding(f) -> Finding` that runs unit/range checks based on
  the catalog entry's `output_columns` semantics. Yields (g/g, mol/mol,
  fractional %) must be in [0, 1] (or [0, 100] for %). Out-of-bound
  values are converted to data_gap with reason
  `"computed value violated physical bounds: {field}={value}"`. Original
  computed value is preserved in `statistics["raw_invalid"]` for audit.
- `src/fermdocs_characterize/pipeline.py`: pipe every finding from
  `catalog_runner` and `trajectory_analyzer` through `validate_finding`
  before adding to the bundle.
- `tests/unit/test_finding_validator.py` (~80 LOC, 6 tests): valid
  yield passes through unchanged, yield > 1 converts to data_gap with
  raw_invalid set, percentage > 100 fails, ratio of demand/supply > 10x
  flagged, missing field tolerated (passes through), data_gap input
  passed through unchanged.

### Commit 4 — Robust statistics (~60 LOC)

- `src/fermdocs_characterize/toolkit/_stats.py` (NEW): helpers
  `central_tendency(arr) -> {mean, median, p25, p75, iqr}` and
  `is_skewed(arr) -> bool` (mean/median ratio > 1.15 or |skew| > 1.0).
- `src/fermdocs_characterize/toolkit/kinetics.py`: B10 (RQ) toolkit fn
  upgraded to emit `mean`, `median`, `p25`, `p75`, `frac_above_1.1`,
  and `recommended_summary` field that is `"median"` when skewed else
  `"mean"`.
- `src/fermdocs_characterize/agents/metric_catalog.py`: B10's
  `output_columns` includes the new fields. `applies_to` and
  `output_shape` unchanged.
- `src/fermdocs_hypothesis/agents/synthesizer.py`: prompt invariant
  amendment — *"When a finding's statistics include both mean and
  median, prefer the value indicated by `recommended_summary`. Cite both
  when they tell different stories."*
- `tests/unit/test_robust_stats.py` (~40 LOC, 3 tests): symmetric
  series → mean recommended; right-skewed series → median recommended;
  output_columns surface as expected.

### Commit 5 — Symmetry post-condition + prompt rebalance (~110 LOC)

- `src/fermdocs_characterize/agents/symmetry_check.py` (NEW): function
  `check_symmetry(findings, run_ids) -> list[Finding]`. Group findings
  by metric_id. For each metric_id present in some runs but not all,
  emit a `data_gap` Finding for each absent (metric_id, run_id) pair
  with reason `"asymmetric coverage — metric computed for some runs
  but not this one; investigate tool path"`. **Q1 reuse**: iterates
  `applicable_metric_run_pairs(bundle)` from commit 1, not its own
  enumeration.
- `src/fermdocs_characterize/pipeline.py`: call `check_symmetry` after
  validator, append its outputs to the bundle's findings.
- `src/fermdocs_hypothesis/seed_topic_extractor.py`: extend
  `_SUPPRESSED_ANALYSIS_KINDS`-equivalent suppression to filter symmetry
  data_gaps OUT of seed-topic generation. Otherwise every multi-run
  bundle would seed N×M debate topics for asymmetric metrics, which is
  the wrong direction — those gaps are tooling notes for the
  synthesizer, not topics worth debating.
- `src/fermdocs_hypothesis/agents/synthesizer.py`: prompt invariant —
  *"If a metric_id appears in findings for some runs and not others,
  the absence is a TOOL gap, not a DATA gap. Report it as 'metric X
  could not be computed for run Y due to a known toolchain limitation,'
  do not let it block a comparative conclusion supported by metrics that
  DID compute on all runs."*
- `src/fermdocs_hypothesis/agents/critic.py`: new `[tool-gap-axis]`
  rejection rule — *"If question_answered='insufficient_data' and the
  hypothesis cites symmetry-violation findings as the reason, file red
  with reason '[tool-gap-axis]: hypothesis treated tool gaps as data
  gaps; the bundle has the data, the toolchain failed to compute it'.
  Allow 'insufficient_data' only when the data itself is missing
  from the bundle (not just from the findings)."*
- `tests/unit/test_symmetry_check.py` (~80 LOC, 6 tests):
  1. symmetric coverage → no extra findings
  2. B10 only on RUN-1 → emits one data_gap for B10×RUN-2
  3. B10 fully missing → no extra findings (both runs equally missing is fine)
  4. cross-run metric (A19) ignored from symmetry check (intentionally no run_ids)
  5. **REGRESSION**: synthesizer prompt USER QUESTION rule + new
     tool-gap rule both fire when both apply; neither breaks the other
  6. **CRITICAL `[tool-gap-axis]` does NOT over-fire**: bundle with
     legitimately missing run data (only RUN-1 was uploaded; no RUN-2
     trajectories) → critic accepts insufficient_data, does NOT file
     red. Test pins the difference between TOOL gap (symmetry-validator
     emitted data_gap) and DATA gap (bundle has no run-2 data at all).

### Commit 6 — IndPenSim regression eval fixture (~50 LOC)

- `tests/integration/fixtures/indpensim_2run/` (NEW directory): pinned
  bundle from the user's working IndPenSim 2-run example. Stored as
  small CSV + dossier + manifest, NOT as a 20MB characterization.json
  (test re-runs characterize end-to-end so the fix is what's tested).
- `tests/integration/test_indpensim_eval.py` (NEW, ~50 LOC, 1 boss test
  + 3 supporting): full pipeline from dossier through hypothesis
  must produce a hypothesis that names RUN-2 as winner with cited
  numerics (30.4 vs 14.3, RQ 0.90 vs 1.21). Marked as a slow test
  (~5 min, real LLM calls). **A4 — manual merge ritual**: not run on
  every test invocation. Required before merging this branch:
  `pytest -m eval tests/integration/test_indpensim_eval.py` must
  pass locally before `git push`. Document this in the resume
  checklist below.

---

## Tests at every step

| Layer | Coverage |
|---|---|
| Catalog runner | per-(metric, run) loop, exception → data_gap, deterministic |
| Product KPIs | each toolkit_fn against synthetic trajectories |
| Validator | each physicality check |
| Robust stats | skew detection + recommended_summary |
| Symmetry | one-sided coverage emits data_gaps |
| Eval | full IndPenSim run produces correct verdict |

Target: 28 new tests, full suite finishing > 1215 (today: 1186 with
plain-language-summaries branch).

---

## What this branch explicitly does NOT do

- **Re-characterize old bundles.** Forward-only. Re-upload to get the
  new findings. Old bundles in `out/api/uploads/` keep their old
  characterizations.
- **Operator verdict as input** (your manual analysis fed back as a
  ground-truth note). That's a separate feature — needs Run.verdict
  field, UI, synthesizer prompt support. Bigger PR.
- **Catalog entries beyond the five product-KPI metrics.** B-tier
  expansion (substrate qs, biomass yields, multi-element balances) can
  land later; this PR ships the minimum to answer "which run was
  better."
- **Migration of existing bundles' characterizations.** Anyone
  re-running the same upload will get a new bundle; existing bundles
  on disk are untouched.
- **LLM eval suite for the new prompts.** First production runs are
  the eval. If shape-aware prompts produce regressions, we'll see and
  iterate.
- **Cross-bundle ground-truth tracking.** The IndPenSim eval fixture is
  one bundle; we don't yet have a corpus.

---

## Risks & mitigations

1. **Catalog runner takes longer than the LLM.** Toolkit functions are
   pandas operations on small dataframes; they should run in <1s per
   metric per run. Worst case 60 metrics × 5 runs × 1s = 5 min added to
   characterize (vs current ~3 min LLM). Mitigation: profile during
   commit 1; if any toolkit_fn is slow, mark it `status="slow"` and
   gate behind a flag.
2. **YAML routing is one more file users can break.** Mitigation:
   pydantic-validate the YAML on load; missing process-family entries
   fall through to `unknown` family (no product-KPI computed) rather
   than crashing.
3. **Trajectory analyzer LLM may regress on open-ended findings now
   that catalog work is gone from its prompt.** Mitigation: keep the
   prompt firm on "emit findings for things the catalog doesn't
   cover"; first production run will reveal whether open-ended quality
   dropped.
4. **Validator over-rejects (false positives turning real findings
   into data_gaps).** Mitigation: only validate well-known unit
   semantics (yields, percentages, ratios). Anything ambiguous passes
   through. Tests pin the explicit reject cases.
5. **Symmetry check on bundles with intentionally per-run-different
   experiments** (e.g., RUN-1 has biomass measurements every 4h, RUN-2
   every 24h). Mitigation: symmetry only fires when the toolkit_fn
   could have run — if RUN-2 fails the input precondition, it already
   emits its own data_gap, no extra symmetry data_gap needed.
6. **IndPenSim eval test is slow (~5 min, real Gemini calls).**
   Mitigation: marked `@pytest.mark.eval`; not run on every test
   invocation; CI runs nightly or pre-merge only.

---

## Cost summary

| Component | LOC |
|---|---|
| Catalog runner + pre-flight + iteration helper (commit 1) | ~280 |
| Product KPI tier + routing + config-mismatch (commit 2) | ~230 |
| Validator (commit 3) | ~100 |
| Robust stats (commit 4) | ~60 |
| Symmetry + prompt + seed-topic suppression (commit 5) | ~110 |
| Eval fixture (commit 6) | ~50 |
| **Total** | **~830 LOC, 6 commits, ~35 new tests** |

LOC and test counts grew slightly after eng-review revisions:
- A1 (`[ALREADY COMPUTED]` block + 2 regression tests on prompt)
- A2 (pre-flight import check + test)
- A3 (config-mismatch single data_gap + test)
- A4 (manual merge ritual documented — no LOC cost)
- Q1 (shared `applicable_metric_run_pairs` helper + test)
- Q2 (idempotent re-run test)
- Plus 3 critical regression tests for the prompt changes in commits
  1, 4, 5 — most importantly the `[tool-gap-axis]` over-fire test in
  commit 5 (must NOT reject legitimate insufficient-data answers).

---

## Resume checklist post-compact

If we hit `/compact` mid-build:

1. `git branch --show-current` — confirm `characterize-determinism`.
2. `git log --oneline characterize-determinism ^frontend-redesign` — see
   how many of the 6 commits done.
3. Read this file `plans/2026-05-07-characterize-determinism.md`.
4. Check the most recent commit's diff to see what's mid-flight.
5. Continue from the next commit; full-suite gate between commits.
6. The boss eval test (commit 6) gates everything; if it fails, no
   single commit of 1–5 is the answer — reread the plan.

## Merge ritual (A4)

Before `git push origin characterize-determinism`:

1. Full Python suite green: `python -m pytest -x -q`.
2. Frontend type-check green: `cd apps/web && npx tsc --noEmit`.
3. **Boss eval test green**: `pytest -m eval
   tests/integration/test_indpensim_eval.py`. Real LLM call, ~5 min, ~$0.20.
   This test is the merge gate, not the CI gate. CI doesn't run real LLM
   tests; the developer running this branch is responsible for the eval.

If the boss eval fails, do NOT push. Read its output, identify which
layer (catalog runner / product KPI / synthesizer prompt) needs work,
and iterate on that commit. Re-run boss eval. Push only after green.

Branch state at plan-writing time: `characterize-determinism` is at
`a0ce0c3` (same as `frontend-redesign`). Zero commits yet.

---

## Eng-review revisions (2026-05-07)

The plan was reviewed by `/plan-eng-review` after first draft and
amended with 7 fixes:

1. **A1**: trajectory analyzer prompt gets explicit `[ALREADY COMPUTED]`
   block, hard rule against re-emitting catalog metric_ids. Two
   regression tests gate the prompt change.
2. **A2**: catalog runner pre-flight imports every toolkit_fn at
   construction; loud abort on import failure prevents silent
   degradation that today produces "computation failed due to tool
   error" data_gaps.
3. **A3**: process-family / bundle-variable mismatch produces ONE
   `[CONFIG_MISMATCH]` data_gap with helpful message, not five
   per-metric data_gaps.
4. **A4**: boss eval test (commit 6) is documented as the manual merge
   ritual; running it is required before `git push`.
5. **Q1**: `applicable_metric_run_pairs(bundle)` helper is shared
   between catalog runner (commit 1) and symmetry check (commit 5),
   eliminating the drift risk between two iteration shapes.
6. **Q2**: idempotency test (re-run produces identical findings)
   pins pipeline determinism as a foundation invariant.
7. **Three critical regression tests**: trajectory analyzer prompt
   `[ALREADY COMPUTED]` block, `[tool-gap-axis]` does NOT over-fire on
   legitimately data-missing bundles, synthesizer prompt composability
   between USER QUESTION and tool-gap rules.

Revisions did not change architecture, branch shape, or boss-eval
definition. Confirmed before commit 1.

---

## User feedback that drove this plan

The user manually validated the IndPenSim 2-run output against the CSV.
Key verbatim findings:

- *"Asymmetric metric extraction — the central bug. The agent computed
  mu_max, RQ, PAA yield, doubling time for RUN-1 but kept saying
  'equivalent kinetic and metabolic metrics are absent for RUN-2'. This
  is factually wrong. The same OUR, CER, X_offline, P columns exist for
  Batch 2 in the same CSV."*
- *"Missed the primary performance metric: penicillin titer. RUN-2
  produced ~2.1× more penicillin. Your agent never reported this number."*
- *"PAA yield '204.5 g/g' is impossible. Yields >1 g/g are non-physical."*
- *"Mean RQ = 1.21 but median = 0.98. Same metric should be applied
  symmetrically with a robust statistic."*
- *"False conservatism. The agent's stance — 'insufficient data to
  definitively conclude' — sounds rigorous but is actually wrong here."*

Ground-truth verdict from the user: **RUN-2 performed ~2× better.**
Final titer 30.4 vs 14.3 g/L. RUN-1 product *declined* from 21.6 → 14.3
after 168h (likely β-lactamase or hydrolysis). RUN-2 used PAA precursor
efficiently (consumed to 634 mg/L) while RUN-1 wasted it (5203 mg/L
residual).
