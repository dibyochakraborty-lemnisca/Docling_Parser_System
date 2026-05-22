# Rigour & Actionability — Plan

**Date:** 2026-05-07
**Branch:** `rigour-and-actionability` (off `agentic-system-architecture-plan`)
**Trigger:** External 7.5/10 review identified 4 concrete gaps that block 8.5+:

1. No statistical rigour visible (correlations cited without CI / n-aware caveats)
2. Anomaly flagging missing (instrument changes, h0 outliers, sensor step-changes)
3. Trajectory data exists but agent reports 0 trajectory citations
4. Hypotheses are descriptive, not actionable

Reviewer also flagged intervention-pattern detection as a gap. That belongs in a
separate plan (bigger scope, requires per-run protocol diffing). Tracked in
[plans/2026-05-07-intervention-deltas.md] (TBD, not in this plan).

## Non-goals

- Adding new specialists. The deferred specialist-routing plan covers that and
  needs to land **after** anomaly flagging exists (anomalies become evidence the
  process-control specialist would consume).
- Memory layer (Synap). Phase 2.
- Carotenoid-specific logic. Every change here must work generically; REGRESSION
  tests on penicillin enforce that.

## Why these four, why now

Reviewer's failure modes all come from the carotenoid bundle but the *primitives*
are organism-agnostic:

| Gap | Failure surface | Generic primitive |
|---|---|---|
| (1) | r=-0.90 with n=6, no CI | bootstrap CI on any cross-run correlation |
| (2) | Hitachi→LABMAN swap, pO2 26.8→95.1 in 6h, WCW 8× at h0 | sensor/protocol-metadata anomaly detection |
| (3) | 0 trajectory citations despite OD/WCW/pO2 every 6h | trajectories surfaced as first-class citables |
| (4) | "data shows X" instead of "design Batch 7 with Y" | `actionable_recommendation` field on FinalHypothesis |

Fix order is by impact-to-effort ratio; (3) is highest leverage.

## Commit plan

### Commit 1 — Trajectories as citable evidence (Item 3)

**Files:**
- `src/fermdocs_hypothesis/bundle_loader.py` — surface trajectory summaries (variable, run_id, n_points, time_range, value_range) into the synthesizer-visible context, not just raw arrays.
- `src/fermdocs_hypothesis/agents/synthesizer.py` — SYNTHESIZER_INVARIANTS adds [TRAJECTORY-CITATION] axis: when a hypothesis references a time-dependent claim (kinetics, decline, peak, transient), it MUST cite ≥1 entry in `cited_trajectories: [{run_id, variable}]`. Falls back to data_gap if no trajectory observed for the claim.
- `src/fermdocs_hypothesis/agents/judge.py` — reject hypothesis when it asserts time-dependent behavior with empty `cited_trajectories`.
- `src/fermdocs_hypothesis/agents/critic.py` — add [trajectory-axis] critic rule: "claim asserts dynamics but cites no trajectory".

**Tests:** `tests/unit/test_trajectory_citation.py`
- Synthesizer prompt rendering includes trajectory list
- Judge rejects time-dependent hypothesis with empty cited_trajectories
- Critic [trajectory-axis] fires correctly
- REGRESSION: penicillin run still produces non-empty cited_trajectories on dynamic claims

**Acceptance:** Re-run carotenoid bundle, expect `cited_trajectories` count > 0 for at least one final hypothesis.

---

### Commit 2 — N-aware bootstrap CI on correlations (Item 1)

**Files:**
- `src/fermdocs_characterize/toolkit/cross_run.py` — new `compute_correlation(x, y, *, n_bootstrap=1000, seed=42)` returning `CorrelationResult(r, n, ci_low, ci_high, weak_n_flag)`. Pure deterministic, fixed seed.
- `src/fermdocs_characterize/agents/metric_catalog.py` — new entry A22 PEARSON_CORRELATION (Tier A, status=ready) operating on cross-run KPI table pairs.
- `src/fermdocs_characterize/agents/catalog_runner_adapters.py` — adapter for A22.
- `src/fermdocs_hypothesis/agents/critic.py` — [robustness-axis] rule: any cited correlation finding with `weak_n_flag=true` (n<8) MUST carry an explicit n-caveat or be downgraded to data_gap.

**Tests:** `tests/unit/test_correlation_ci.py`
- Strong correlation (r≈0.95, n=20) returns tight CI
- Weak n (n=6) sets `weak_n_flag=true` and widens CI
- Deterministic across runs (fixed seed)
- REGRESSION: existing cross_run tests still pass byte-identical

---

### Commit 3 — Anomaly flagging at characterize layer (Item 2)

Three sub-detectors, each emitting Findings with `metric_id` prefix `ANOMALY_*`:

**3a — Instrument-change detector**
- Scans narrative + dossier for instrument-name strings per run; emits Finding when same measurement_kind has different instrument across runs.
- Generic: works for spectrophotometers, DO probes, pH probes, any named instrument.

**3b — H0 outlier detector**
- For each variable, compute median(t≈0) across runs; flag runs where |run_h0 − cohort_median| > 3·MAD.

**3c — Step-change detector**
- Per trajectory: flag intervals where |Δvalue / Δtime| exceeds physical-bounds-derived rate (e.g. pO2 jump of 70% in <1h).

**Files:**
- `src/fermdocs_characterize/agents/anomaly_detectors.py` (NEW)
- `src/fermdocs_characterize/agents/metric_catalog.py` — three new ANOMALY_* entries.
- `src/fermdocs_characterize/agents/catalog_runner_adapters.py` — adapters.
- `src/fermdocs_hypothesis/agents/synthesizer.py` — promote ANOMALY_* findings to first-class context (don't bury in fact list).

**Tests:** `tests/unit/test_anomaly_detectors.py`
- Synthetic 2-instrument bundle → instrument-change fires
- H0 outlier with WCW 8× cohort median → fires
- pO2 step 26.8→95.1 in 6h → fires
- REGRESSION: penicillin clean bundle emits zero anomalies

---

### Commit 4 — Actionable recommendation on FinalHypothesis (Item 4)

**Files:**
- `src/fermdocs_hypothesis/schema.py` — `FinalHypothesis.actionable_recommendation: str | None = None`. Optional during transition; judge enforces presence on green-flag hypotheses in commit 4b.
- `src/fermdocs_hypothesis/agents/synthesizer.py` — invariant: green hypothesis must propose a concrete next-batch parameter change OR explicit "insufficient evidence to recommend" with reason.
- `src/fermdocs_hypothesis/agents/judge.py` — reject green flag with null + non-insufficient-evidence recommendation.
- `apps/web/src/lib/api.ts` + UI surface — render recommendation under hypothesis card.
- `apps/api/fermdocs_api/runner_pipeline.py` — pass through.

**Tests:** `tests/unit/test_actionable_recommendation.py`
- Schema field default null
- Judge rejects green hypothesis with null recommendation
- Judge accepts red hypothesis with null recommendation
- Judge accepts "insufficient evidence" string as valid null-substitute
- REGRESSION: existing fixtures load without migration (default None)

---

## Risk register

| Risk | Mitigation |
|---|---|
| Trajectory citation requirement breaks penicillin runs (no dynamic claims) | REGRESSION test on penicillin synthetic bundle; only fire on time-dependent claims |
| Bootstrap CI is slow on large n | n_bootstrap=1000 caps wall time; deterministic seed |
| Anomaly detectors over-fire on noisy data | Each detector emits as Finding, not as judge-rejection; synthesizer decides salience |
| Schema change breaks loaded fixtures | Default None; no migration needed |

## Success criteria

Re-run carotenoid bundle end-to-end:
- ≥1 final hypothesis cites ≥1 trajectory
- Cross-run correlation findings carry CI + weak_n flag when n<8
- Hitachi→LABMAN anomaly appears in fact list
- Every green-flag hypothesis has either an actionable recommendation or explicit insufficient-evidence reason

Re-run penicillin synthetic bundle:
- Byte-identical hypothesis output vs. pre-change baseline (modulo new optional fields defaulting None)
- Zero anomalies fired
- All existing tests pass
