# Characterization Agent — Execution Plan

Durable plan for building the **Characterization Agent**, the second agent in the fermdocs multi-agent pipeline. This file is the source of truth for scope, ordering, decisions, and triggers across versions. Update it when reality forces a change; don't update it to match a passing whim.

---

## 1. Position in the system

```
Ingestion (built)  →  Characterization (this plan)  →  Diagnosis  →  Orchestrator(loop)
                                  ↑                                       ↓
                                  │                                       ↓
                                  └──────── stable artifact ────────── →  Simulation (TS-JEPA, future)
                                                                          ↓
                                                                          →  Critic
```

Characterization produces **one append-only, queryable artifact** every downstream agent reads. Stable IDs, calibrated uncertainty, trajectory-clean data, closed vocabularies. Built once, read many times under different lenses.

## 2. Operating principles (do not break)

1. **Plan-and-execute with deterministic gates.** No ReAct, no multi-agent, no reflection inside characterization. The plan is fixed; only candidate judgments (v2+) use an LLM.
2. **LLM never sees the dossier.** It sees pre-fetched evidence packs per candidate. Stable cacheable prefix; volatile suffix.
3. **Closed vocabularies.** Finding types, edge types, and `OpenQuestion.decision_type` are closed enums. Novelty routes through `open_questions`, never through new vocabulary at runtime.
4. **Provenance is non-negotiable.** Every finding cites real ingestion observation IDs; the validator rejects orphans and follows supersession chains.
5. **Determinism is a contract.** Sort-stable aggregations, temperature-0 judge (v2+), idempotency cache. Re-running on the same input produces the same finding IDs.
6. **Output is immutable.** Re-characterizing produces a new output with `supersedes` pointing at the old. Old outputs kept for audit.
7. **Globally unique finding IDs.** Namespaced by characterization output ID: `<characterization_id>:F-NNNN`.

## 3. Decisions, locked

| Decision | Resolution |
|---|---|
| Findings shape | Flat list. `facts_graph` is structural only (Sample/Measurement/Condition). |
| Vocabulary policy | Closed. Extension requires code change + eval coverage. |
| Confidence calibration | Split. Statistical (p-value, effect size, n) for stat-grounded; ≤0.85 cap for LLM-judged. |
| ID stability | Sort-stable aggregations always; temperature-0 judge in v2+; idempotency cache keyed on `hash(candidate + evidence_obs_hashes + judge_prompt_version)`. |
| Auto-accept threshold | `auto_accept_policy.yaml` in v2b. v1 has nothing to auto-accept. |
| Evidence ID supersession | Validator follows chain; rewrites or marks finding stale. |
| Schema versioning | `meta.schema_version` and `meta.process_priors_version`. Outputs older than current versions regenerate, never silently patch. |
| Trajectory shape | Generic regular-grid + masks + provenance. Happens to be JEPA-friendly. JEPA conforms to data, not vice versa. |
| Hot vs cold path | `mode: full \| fast` parameter introduced only in v2. v1 has no LLM, so doesn't need it. |
| `open_questions` schema | `question_text`, `relevant_finding_ids`, `relevant_run_ids`, `decision_type` (closed enum), `would_resolve_with`. |
| Output retention | Immutable, append-only. `supersedes` chain. |
| Finding ID scope | Globally unique via `<characterization_id>:F-NNNN`. |
| Data quality | Field on `Trajectory`, not a finding type. |
| Multi-run vs single-run | Schema handles both. Cohort-style paths produce no findings when N=1. Test both. |
| Reasoning pattern | v1: deterministic plan-and-execute. v2+: same with constrained LLM judge per candidate. |

## 4. Version plan

### v1 — minimal viable artifact (build now)

**Goal:** produce a useful artifact for the future Diagnosis Agent with zero LLM calls and zero external priors.

**Finding types (closed):** `range_violation` only.

**Edge vocabulary (closed):** `measured`, `under_condition`, `at_time`, `derived_from`. Structural only. No relationship types.

**`OpenQuestion.decision_type` (closed):** `causal_attribution`, `anomaly_classification`, `evidence_request`.

**Output shape (full schema, partially populated):**

```
CharacterizationOutput
├─ meta
│  ├─ schema_version
│  ├─ characterization_version
│  ├─ characterization_id            # UUID, namespaces all finding IDs
│  ├─ generation_timestamp
│  ├─ supersedes                     # characterization_id | null
│  └─ source_dossier_ids
├─ findings                         # v1: range_violation only
├─ timeline                         # ordered events with lags
├─ expected_vs_observed             # setpoint vs actual table
├─ trajectories                     # regular-grid time series, JEPA-shape-friendly
│  └─ each carries data_quality{pct_missing, pct_imputed, pct_real}
├─ facts_graph                      # Sample / Measurement / Condition nodes only
├─ kinetic_estimates: []            # empty in v1, populated in v3
└─ open_questions                   # structured
```

**Pipeline (deterministic, zero LLM):**

```
build_summary_view  →  build_trajectories  →  range_violation_generator
                                                  ↓
                                          build_expected_vs_observed
                                                  ↓
                                          build_timeline
                                                  ↓
                                          build_facts_graph
                                                  ↓
                                          validate_output
                                                  ↓
                                          emit CharacterizationOutput
```

**File layout:**

```
src/fermdocs_characterize/
  schema.py                  # Pydantic contract
  vocabularies/
    finding_types.md         # closed list
    edge_vocabulary.md       # closed list
    decision_types.md        # closed list (open_questions)
  pipeline.py                # plan-and-execute orchestrator
  views/
    summary.py               # (run, time, variable) DataFrame, sort-stable
    trajectories.py          # regular-grid resampling + imputation + data_quality
  candidates/
    range_violation.py       # nominal ± k·σ, k from per-finding-type config
  builders/
    timeline.py
    expected_vs_observed.py
    facts_graph.py
  validators/
    output_validator.py      # ID resolution, supersession, vocab compliance, schema version
  cli.py                     # `fermdocs-characterize <dossier.json>` → output.json
  langgraph_node.py          # typed I/O for LangGraph integration
evals/
  fixtures/                  # 3 synthetic dossiers + expected outputs (no real data yet)
  test_range_violation_unit.py
  test_fixtures_integration.py
  test_id_stability.py
  test_provenance_integrity.py
  test_schema_version_invalidation.py
```

**Fixtures (synthetic, since no real data exists yet):**

1. **Boundary fixture** — single-run dossier with values at exactly ±2σ, ±3σ, ±5σ to test threshold sharpness.
2. **Missing-data fixture** — single-run dossier with sparse measurements, large imputation gaps, partial coverage. Tests `data_quality` field population.
3. **Multi-run fixture** — dossier with 4 runs at the same condition, deliberately introducing range violations in 1 of them. Tests multi-run handling without invoking cohort logic.

Each fixture ships with a hand-written expected `CharacterizationOutput.json`. The integration test asserts byte-stable equality on a sorted/normalized form.

**Tests (5):**

- `test_range_violation_unit` — generator flags values at ±2σ (info), ±3σ (major), ±5σ (critical); ignores values within ±2σ.
- `test_fixtures_integration` — each of 3 fixtures produces the expected output.
- `test_id_stability` — same input twice → identical finding IDs.
- `test_provenance_integrity` — every finding's evidence IDs resolve through ingestion's supersession chain.
- `test_schema_version_invalidation` — outputs with older `schema_version` are flagged invalid by the validator.

**Success criterion for v1:** running on each of the 3 fixtures produces output where a future Diagnosis Agent prompt, citing only the JSON, could identify the introduced failures. If yes, v1 is done. If no, the schema is wrong; fix before adding any candidate types.

**Explicitly NOT in v1:**

- `mass_balance` (needs density prior)
- `data_quality_issue` finding type (data_quality is a field on `Trajectory`, not a finding)
- `mode: full \| fast` parameter (no LLM, so no need)
- `process_priors.yaml`
- `auto_accept_policy.yaml`
- LLM judge
- Cohort logic
- Kinetic estimator
- Causal cascade
- TS-JEPA-explicit trajectory export

---

### v2 — LLM judge and process priors

**Trigger:** Diagnosis Agent has run on ≥3 real dossiers and produced `diagnosis_gaps.md` listing failures it wishes characterization had surfaced. Each gap names: what was missed, what evidence was available, what finding type would have caught it. Pick v2a or v2b first based on which gaps are most frequent.

v2 is split into v2a and v2b so a single failure landing doesn't take down the other.

#### v2a — LLM judge for cohort_outlier

**Adds:**

- `cohort_outlier` finding type (requires cohort store — see v3, may co-arrive)
- LLM judge module (`judge/llm_judge.py`)
  - temperature 0
  - structured output (Pydantic-validated)
  - evidence pack assembly per candidate (≤20 observations)
  - cacheable prefix: system prompt + vocabularies + judging instructions (~1.5K tokens)
  - volatile suffix: candidate dict + evidence + cohort stats (~2-3K tokens)
  - confidence ≤ 0.85 (LLM-judged)
  - idempotency cache: `hash(candidate + evidence_obs_hashes + judge_prompt_version)` → skip on hit
- `mode: full | fast` parameter on the pipeline
  - `full`: runs LLM judge
  - `fast`: deterministic candidates only (matches v1 behavior, used by Critic loop on simulated trajectories)
- LLM call cap: `N_max` candidates per finding type per dossier, prioritized by deterministic score; remainder emitted as low-confidence findings without judgment.

**Open question (name now, decide at v2a time):** judge call concurrency. Serial, batched, or async-with-semaphore? Affects Critic-loop latency.

**Eval gate to ship v2a:** 30 hand-labeled candidates, ≥85% judge agreement on `surface | demote | reject`.

#### v2b — process relationship checks with priors

**Adds:**

- `process_priors.yaml` containing density, Monod constants, Arrhenius constants, plausible causal pairs. Versioned via `meta.process_priors_version`.
- `auto_accept_policy.yaml` containing per-finding-type thresholds. Above threshold → finding without LLM. Below → judge.
- New finding types (one at a time, each with eval coverage):
  - `mass_balance_error` (requires density)
  - `process_relationship_violation` (OUR/OTR balance, pH-NH3 coupling, Monod-substrate, T-mu, PAA depletion)
- Validator extension: outputs older than current `process_priors_version` flagged for regeneration.

---

### v3 — cohort store and kinetic estimation (sub-project)

**Triggers (all must be true):**

1. Real golden schema is locked (biotech colleague's design landed).
2. ≥10 dossiers exist in storage.
3. An `open_questions` pattern recurs across debate loops requiring cross-dossier reasoning.

**When triggered:** open a separate design conversation. v3 cohort store is a sub-project, not a feature. It involves indexing, schema-evolution handling, possibly a separate database, plus a query API. Do not implement without dedicated design.

**Adds (deferred to v3 design):**

- Cohort store (read-side query layer over past characterization outputs)
- `kinetic_estimator` (windowed regression for `mu_x`, `mu_p`, `kla` with confidence intervals; populates `kinetic_estimates`)
- `precedes_with_lag` causal cascade with closed adjacency list of plausible causal pairs
- Optional: TS-JEPA-explicit trajectory export (only if the simulation architecture is committed to TS-JEPA at that point)

## 5. LangGraph integration

Characterization is a LangGraph node with typed I/O:

```python
class CharacterizationState(TypedDict):
    dossier: IngestionResult            # input
    output: CharacterizationOutput      # populated by this node
    errors: list[str]
```

State uses file-pointer + in-memory hybrid: `dossier` and `output` can be `{path: ...}` references for large artifacts so LangGraph state stays small. Diagnosis reads `output` directly. Debate-loop nodes read specific fields by ID. Future Simulation reads `output.trajectories` and `output.kinetic_estimates`.

## 6. Context engineering invariants (v2+)

Per LLM judge call:

- **Cacheable prefix (~1.5K tokens, ~80% hit rate):** system prompt + finding_types + edge_vocabulary + judging instructions.
- **Volatile suffix (~2-3K tokens):** one candidate dict + ≤20 evidence observations + cohort/process stats relevant to *this* candidate.
- **No tool use inside the judge.** Orchestrator pre-fetches everything.
- **Total per call: ~3-4K tokens.** Cost is bounded by `N_max`, not by data.

## 7. Failure mode → mitigation table

| Failure mode | Mitigation |
|---|---|
| Empty findings (clean run) | Output is still valid: empty `findings`, populated `trajectories` and `facts_graph`. Tests assert this. |
| Ingestion supersedes an obs after characterization ran | Validator follows supersession chain; rewrites finding evidence IDs to latest non-superseded obs OR marks finding stale. |
| Same dossier, different schema version | `meta.schema_version` mismatch → invalid; downstream regenerates rather than reads stale output. |
| Two characterization runs on overlapping dossiers, same finding number | Globally unique IDs (`<characterization_id>:F-NNNN`) prevent conflation. |
| LLM judge non-determinism (v2+) | Temperature 0 + idempotency cache. |
| Pathological dossier produces 1000s of candidates (v2+) | `N_max` cap per finding type, prioritized by deterministic score. Remainder emitted as low-confidence findings without judgment. |
| Critic loop wants to re-run characterization on simulated trajectory | `mode: fast` skips the LLM judge for cheap re-runs. |
| Residual narrative blocks ignored | Out of scope for v1–v3. Separate design conversation: should narrative-tier observations be promoted to findable surface, or routed independently to Diagnosis? |

## 8. Build order, locked

**v1 (do this first, in order):**

1. `schema.py` — the contract.
2. `vocabularies/finding_types.md`, `edge_vocabulary.md`, `decision_types.md` — closed lists.
3. Three synthetic fixtures + expected outputs.
4. **Gate:** can a future Diagnosis Agent prompt, citing only each fixture's expected output JSON, identify the introduced failures? If no, schema is wrong; iterate before writing any pipeline code.
5. `views/summary.py`, `views/trajectories.py` — deterministic data prep.
6. `candidates/range_violation.py` — the only v1 candidate generator.
7. `builders/timeline.py`, `builders/expected_vs_observed.py`, `builders/facts_graph.py`.
8. `validators/output_validator.py`.
9. `pipeline.py` — wires the above together.
10. The 5 tests.
11. `cli.py` and `langgraph_node.py`.

**v2a (do when triggered):**

1. Cohort store decision (may bring v3 forward).
2. `judge/llm_judge.py` with structured output and idempotency cache.
3. `candidates/cohort_outlier.py`.
4. `mode: full | fast` parameter on pipeline.
5. 30 hand-labeled candidates as eval set.

**v2b (do when triggered):**

1. `process_priors.yaml` (versioned).
2. `auto_accept_policy.yaml`.
3. `candidates/mass_balance.py`.
4. `candidates/process_relations.py` (one check at a time, each with eval coverage).
5. Validator extension for `process_priors_version` invalidation.

**v3 (sub-project, separate design when triggered):**

- Cohort store design conversation.
- Kinetic estimator.
- Causal cascade.
- Optional TS-JEPA-explicit export.

## 9. What this plan does not do (and why)

- **No vector store, no RAG, no semantic search.** Data is structured; retrieval is by ID.
- **No multi-agent debate inside characterization.** Characterization is single-pass; debate is the orchestrator's job downstream.
- **No graph database.** JSON-as-graph fits in LangGraph state. Migrate only when query patterns demand it.
- **No reflection/self-critique inside the judge.** Add only if v2a evals show first pass is unreliable.
- **No `cluster` or `condition_correlate` candidates.** Those serve optimization, not diagnosis. Add when an optimization agent asks.
- **No TS-JEPA-explicit shaping in v1.** `Trajectory` is generic. JEPA conforms when committed.
- **No real-data fixtures in v1.** No real data exists yet. Synthetic fixtures cover edge cases real data wouldn't anyway. First real-data run becomes the first eval after v1 ships.

## 10. Compatibility with downstream

The artifact this plan produces is compatible with the planned downstream architecture:

- **Stable, globally-unique finding IDs** → debate/voting works; finding-level citations are unambiguous.
- **Calibrated uncertainty + caveats + competing_explanations** → Critic has real flaws to attack; debate doesn't degenerate to vibes.
- **Flat queryable lists with rich metadata** → Orchestrator can re-query under reframed problems without re-running characterization.
- **Trajectories and kinetic_estimates as first-class** → future Simulation (TS-JEPA or other) drops in cleanly; Critic can cross-reference embedding-space anomalies with finding-level explanations.
- **`open_questions` field with closed `decision_type`** → explicit handoff for things only the debate loop should resolve. Characterization doesn't fake answers it doesn't have.
- **Immutable + supersedes** → reproducibility for science workloads.

## 11. Update protocol

Update this file when:

- A v1→v2 or v2→v3 trigger fires.
- A locked decision is overturned by reality (document why).
- A new finding type, edge type, or decision_type is added (note which version added it).
- A failure mode is discovered that isn't in section 7.

Do NOT update this file when:

- An implementation detail changes that doesn't affect the contract.
- A new test is added (just add the test).
- A bug is fixed (just fix it).

---

*Last reviewed against feedback: two rounds of plan review (architecture + scope challenge). v1 scope locked at `range_violation` only. No real data; v1 ships with 3 synthetic fixtures.*
