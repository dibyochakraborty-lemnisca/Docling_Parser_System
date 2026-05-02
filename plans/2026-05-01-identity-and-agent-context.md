# Plan: Process Identity, Flags, and Agent Context

**Date:** 2026-05-01
**Status:** v4.1 — refactored mid-implementation per user pushback (see Revision log)
**Scope:** characterization-agent foundation work — adds identity extraction, deterministic flags, and an on-demand agent-prompt context builder
**Splits into:** PR1 (cross-cutting identity) + PR2 (characterize-only flags + agent context)

---

## Revision log

### v4.1 (2026-05-01) — split identity into observed facts vs registered process

**What changed.** v4 conflated two different signals into a single `ProcessIdentity`: surface facts (organism, product, scale — usually quoted verbatim from the source paper) and process classification (registry match — requires generalization). Treating them as one meant non-registered processes (e.g. yeast when only penicillin is in the registry) lost their organism too — the dossier carried `provenance: UNKNOWN` and nothing else. Downstream agents were left without information that was readily available in the prose.

**The fix.** Split into two layers under `experiment.process`:

- `experiment.process.observed`: organism, product, scale, process_family_hint. Always populated when the LLM finds them in prose. Substring evidence verification still applies. Confidence still capped at 0.85.
- `experiment.process.registered`: process_id, confidence, provenance. Only populated on registry hit + fingerprint pass. Stays `null/UNKNOWN` for processes outside the registry.

**Single LLM call** returns both layers in one JSON object (`{"observed": {...}, "registered": {...}}`). No cost increase vs v4.

**Why now, not later.** The user requirement that prompted this — "Diagnosis agent should see organism no matter what" — is load-bearing for PR2's agent context. Building PR2 against a shape that doesn't satisfy it would mean reworking PR2 too. Cost was concentrated in uncommitted tests + fixtures, which were being written anyway.

**Naming convention.** `experiment.process.observed.organism`, `experiment.process.registered.process_id`. Keeps both under one umbrella so downstream code reads cleanly.

**What didn't change.** Registry, fingerprint check, manifest path, evidence_locators, evidence_gated_llm primitive, IdentityProvenance enum. Manifest still pins both layers and forces provenance=MANIFEST.

---

---

## Why this plan exists

The characterize agent today produces a `CharacterizationOutput` artifact, but it has no first-class layer that downstream LLM agents (Diagnosis, Critic, Orchestrator) can use as a stable, cacheable prompt prefix. Three things are missing:

1. **Process identity** — what organism, product, scale, recipe family this experiment represents. Without it, every future agent prompt re-derives this from raw text and gets it slightly different each time.
2. **Deterministic flags** — closed-vocabulary signals like `STALE_SCHEMA_OBSERVATIONS`, `SPARSE_DATA`, `UNKNOWN_PROCESS` that summarize the data posture. Today this lives implicitly in scattered fields; agents have to infer it.
3. **Agent context builder** — a small, on-demand serializer that projects (dossier + characterize output) into a ~1500-token prompt prefix. The shape every downstream agent reads first.

Locked design decisions (from review v3):

- **A1** ✅ shared `evidence_gated_llm_call` primitive; refactor narrative_extractor first, then build identity extractor on top
- **A2** ✅ `variable_fingerprint` (required + strong + forbidden) in registry schema; post-LLM check downgrades to `UNKNOWN` on mismatch
- **A3** ✅ `EvidenceLocator` (file_id, paragraph_idx, span_text, span_start) instead of raw strings; KG-ready
- **Q1** ✅ emission rules + thresholds + rationale in each `ProcessFlag` enum docstring
- **Q2** ✅ `AgentContext` holds raw inputs only; rollups computed at serialize time
- **Q3** ✅ runtime token-budget truncation with logged warning, in addition to test assertion
- **Eval set** ✅ `evals/identity_extractor/` with ≥5 scripted dossier fixtures
- **PR split** ✅ two PRs: PR1 cross-cutting identity, PR2 characterize-only flags+context

---

## Architecture

```
                ┌─────────────────────────────────────────────┐
                │  Ingestion (PDF/CSV/Excel)                  │
                └────────────────────┬────────────────────────┘
                                     │
                          ┌──────────▼─────────┐
                          │  build_dossier()   │  (PR1)
                          │  - manifest path?  │
                          │  - else extractor  │
                          └──────────┬─────────┘
                                     │
                  ┌──────────────────▼──────────────────┐
                  │  Dossier dict                       │
                  │  ├── experiment.process: identity   │  ← new in PR1
                  │  ├── golden_columns                 │
                  │  ├── ingestion_summary              │
                  │  └── residual                       │
                  └──────────────────┬──────────────────┘
                                     │
                          ┌──────────▼──────────┐
                          │ CharacterizationP.  │
                          └──────────┬──────────┘
                                     │
                       ┌─────────────▼─────────────┐
                       │ CharacterizationOutput    │
                       │ + findings, trajectories  │
                       └─────────────┬─────────────┘
                                     │
                          ┌──────────▼──────────┐
                          │ build_agent_context │  (PR2)
                          │ + compute_flags     │
                          │ + serialize w/ cap  │
                          └──────────┬──────────┘
                                     │
                          ▼ (downstream LLM agents)
```

---

## PR 1 — Identity (cross-cutting)

### 1.1 `src/fermdocs/mapping/evidence_gated_llm.py` (new)

Refactor target. Lift these from `src/fermdocs/mapping/narrative_extractor.py` into a shared primitive both extractors call:

- `verify_evidence` → `verify_substring_evidence(evidence, source_text, value)` (public)
- `_value_string_forms` → `value_string_forms` (public)
- `MAX_EVIDENCE_LEN`, `MAX_SENTENCE_BREAKS` constants → module-level
- `NARRATIVE_CONFIDENCE_CAP` → `LLM_CONFIDENCE_CAP`

Add a thin orchestrator:

```python
def evidence_gated_call(
    *, prompt: str, source_text: str, llm_client: LLMClient,
    confidence_cap: float = LLM_CONFIDENCE_CAP,
    value_for_verification: Any = None,
    timeout_s: float = 30.0,
) -> tuple[dict | None, str | None]:
    """Returns (parsed_response, error_reason).

    Hard contracts:
      - Truncate confidence to confidence_cap
      - Reject responses whose evidence span isn't a substring of source_text
      - Catch all LLM errors → (None, reason); never raise
    """
```

`narrative_extractor.py` and `identity_extractor.py` both call this. Single seam for safety bugs.

### 1.2 `src/fermdocs/schema/process_registry.yaml` (new)

```yaml
version: "1.0"
processes:
  - id: penicillin_indpensim
    organism: "Penicillium chrysogenum"
    product: "penicillin"
    process_family: "fed-batch"
    typical_scale_l: 58000
    aliases:
      organism: ["P. chrysogenum", "P chrysogenum"]
      product: ["pen-G", "penicillin G", "6-APA precursor"]
    variable_fingerprint:
      required: [paa_mg_l, nh3_mg_l]
      strong:   [alpha_kla, mu_p_max_per_h, biomass_g_l]
      forbidden: []
```

Loader (`src/fermdocs/mapping/process_registry.py`) validates:

- Unique `id` across the file
- No alias collisions across processes
- Every fingerprint variable exists in the loaded golden schema (uses `cached_schema()`)

### 1.3 `src/fermdocs/domain/models.py` — extensions (v4.1)

```python
class IdentityProvenance(str, Enum):
    MANIFEST = "manifest"
    LLM_WHITELISTED = "llm_whitelisted"
    UNKNOWN = "unknown"

class EvidenceLocator(BaseModel):
    file_id: str
    paragraph_idx: int
    span_text: str = Field(max_length=200)
    span_start: int | None = None  # char offset within paragraph; optional

class ScaleInfo(BaseModel):
    volume_l: float | None = None
    vessel_type: str | None = None

class ObservedFacts(BaseModel):
    """Surface facts extracted from prose. Populated whenever the LLM finds
    them in the source documents, regardless of whether the process matches
    the registry.

    Substring-evidence verification applies to every populated field.
    Confidence is capped at 0.85.
    """
    organism: str | None = None
    product: str | None = None
    process_family_hint: str | None = None  # "fed-batch" / "batch" / "perfusion"
    scale: ScaleInfo | None = None
    confidence: float = Field(ge=0.0, le=1.0, default=0.0)
    provenance: IdentityProvenance = IdentityProvenance.UNKNOWN
    evidence_locators: list[EvidenceLocator] = Field(default_factory=list, max_length=5)
    rationale: str | None = None

class RegisteredProcess(BaseModel):
    """Registry classification. Only populated on registry hit + fingerprint
    pass. Stays UNKNOWN for processes outside the registry.
    """
    process_id: str | None = None
    confidence: float = Field(ge=0.0, le=1.0, default=0.0)
    provenance: IdentityProvenance = IdentityProvenance.UNKNOWN
    rationale: str | None = None

class ProcessIdentity(BaseModel):
    """Wrapper carrying both layers. Downstream agents read both:
      - process.observed.organism: usually present even on non-registered runs
      - process.registered.process_id: only present on registry hit
    """
    observed: ObservedFacts = Field(default_factory=ObservedFacts)
    registered: RegisteredProcess = Field(default_factory=RegisteredProcess)
```

### 1.4 `src/fermdocs/mapping/identity_extractor.py` (v4.1, single-pass two-layer)

Decision flow (single LLM call returns both layers):

```
manifest provided?
   YES → ProcessIdentity from manifest, both layers provenance=MANIFEST. Skip LLM.
   NO ↓
narrative blocks empty?
   YES → ProcessIdentity(observed=UNKNOWN, registered=UNKNOWN). No LLM call.
   NO ↓
single LLM call → {observed: {...}, registered: {...}}
   error/timeout → both layers UNKNOWN
   else split into two layers, validate each independently:

   OBSERVED layer:
     for each populated field, evidence span MUST be substring of source
     any field whose evidence fails → that field nulled, kept in rationale
     surviving fields → observed.provenance=LLM_WHITELISTED
     all fields fail evidence → observed.provenance=UNKNOWN

   REGISTERED layer:
     process_id null or off-whitelist → registered.provenance=UNKNOWN
     on-whitelist ↓
     fingerprint check (required/forbidden vars in dossier)
       pass → registered.provenance=LLM_WHITELISTED
       fail → registered.provenance=UNKNOWN with rationale
     confidence capped at LLM_CONFIDENCE_CAP (0.85) regardless
```

Key property: failure of the registered layer does NOT nullify the observed layer. Yeast experiments produce `observed.organism = "Saccharomyces cerevisiae"` + `registered = UNKNOWN`.

Prompt explicitly constrains:
- For OBSERVED fields: "quote the exact substring of one of the input paragraphs"
- For REGISTERED.process_id: "must be one of the listed registry IDs or null"
- Both: "do not invent values not present in the prose"

Output validated against the registry post-call.

### 1.5 `src/fermdocs/dossier.py` — extension

`build_dossier(experiment_id, repository, *, manifest_path=None)` reads optional manifest, runs the priority chain above, attaches `experiment.process: ProcessIdentity` to the returned dossier dict. Bumps `DOSSIER_SCHEMA_VERSION` to `"1.1"`.

### 1.6 `src/fermdocs/cli.py` — `--process-manifest` flag

YAML manifest validates against `ProcessIdentity` directly. Malformed YAML → loud Pydantic error. Missing file → exit 2 with clear message.

### 1.7 `evals/identity_extractor/` (new)

Five scripted-LLM fixtures:

- `01_clear_penicillin/` — paper mentioning P. chrysogenum + penicillin → expect `penicillin_indpensim`
- `02_off_whitelist/` — paper mentions E. coli (not in registry) → expect `UNKNOWN`
- `03_fingerprint_mismatch/` — paper says penicillin but dossier lacks `paa_mg_l` → expect `UNKNOWN`
- `04_unquoted_evidence/` — scripted LLM emits hallucinated evidence → expect rejection
- `05_empty_narrative/` — Excel-only ingestion → expect `UNKNOWN`, no LLM call

Each fixture: `narrative_blocks.json`, `golden_columns.json`, `scripted_llm_response.json`, `expected_identity.json`, plus a runner.

### 1.8 Tests for PR1 (v4.1)

```
tests/unit/test_evidence_gated_llm.py
  - verify_substring_evidence happy + 4 rejection paths
  - confidence cap constant
  - value_string_forms variants

tests/unit/test_process_registry.py
  - load valid registry
  - duplicate process_id rejected
  - alias collision across processes rejected
  - fingerprint variable not in golden schema rejected
  - fingerprint_check happy/required-missing/forbidden-present/strong-advisory

tests/unit/test_identity_extractor.py (covers both layers separately)
  - OBSERVED: happy path emits LLM_WHITELISTED + populated organism/product
  - OBSERVED: unquoted evidence on a single field nulls only that field
  - OBSERVED: confidence capped at 0.85
  - OBSERVED: empty narrative → both layers UNKNOWN, no LLM call
  - OBSERVED: LLM timeout/error → both layers UNKNOWN
  - REGISTERED: off-whitelist process_id → registered=UNKNOWN, observed unaffected
  - REGISTERED: fingerprint mismatch → registered=UNKNOWN, observed retains organism
  - REGISTERED: null process_id but observed present → observed populated, registered UNKNOWN
  - WRAPPER: yeast-style case (organism in prose, no registry match)
    → observed.organism populated, registered.process_id null

tests/integration/test_dossier_with_identity.py
  - manifest path → both layers provenance=MANIFEST, LLM not called
  - extractor: registry hit + fingerprint pass → both layers LLM_WHITELISTED
  - extractor: organism in prose but off-registry → observed populated, registered UNKNOWN
  - extractor: no client configured → both layers UNKNOWN
  - existing 01_boundary, 02_missing_data, 03_multi_run still validate

evals/identity_extractor/run_evals.py
  - runs all 5 fixtures, asserts expected observed + registered fields
```

---

## PR 2 — Flags + agent context (characterize-only, after PR1 merges)

### 2.1 `src/fermdocs_characterize/flags.py` (new)

```python
class ProcessFlag(str, Enum):
    STALE_SCHEMA_OBSERVATIONS = "stale_schema_observations"
    """Fires when dossier.ingestion_summary.stale_schema_versions is non-empty.

    Rationale: observations were extracted under an older schema; column
    names or unit semantics may have shifted. Downstream agents should
    flag rather than reason.
    """

    LOW_QUALITY_TRAJECTORY = "low_quality_trajectory"
    """Fires when any trajectory has quality < 0.8.

    Rationale: a trajectory with >20% imputed/missing values is unreliable
    for derivative-based reasoning (rates, slopes, breaks).
    """

    SPARSE_DATA = "sparse_data"
    """Fires when summary.rows < 20 OR golden_coverage_percent < 50.

    Rationale: <20 observations is below the noise floor for most patterns;
    <50% coverage means most variables have no signal at all.
    """

    MIXED_RUNS = "mixed_runs"
    """Fires when len(summary.run_ids) > 1.

    Rationale: cohort-level reasoning requires the agent to know it's
    looking at multiple runs, not one.
    """

    UNKNOWN_PROCESS = "unknown_process"
    """Fires when dossier.experiment.process.provenance == UNKNOWN.

    Rationale: identity is the strongest prior on what's normal. Without
    it, agents should not invoke domain-specific reasoning.
    """

    SPECS_MOSTLY_MISSING = "specs_mostly_missing"
    """Fires when >50% of summary.variables have no spec in the loaded schema.

    Rationale: range-violation reasoning needs specs. If most are missing,
    expect the output to be heavy on open_questions and light on findings.
    """
```

`compute_flags(dossier, summary, trajectories) -> list[ProcessFlag]`. Pure function, deterministic, sorted output.

### 2.2 `src/fermdocs_characterize/agent_context.py` (new)

```python
class AgentContext(BaseModel):
    process: ProcessIdentity
    schema_version: str
    extractor_version: str
    n_runs: int
    n_observations: int
    time_range_h: tuple[float, float] | None
    variables_with_specs: list[str]
    variables_without_specs: list[str]
    flags: list[ProcessFlag]
    findings_ref: list[str]   # raw finding_ids; severity rollup computed at serialize


def build_agent_context(
    dossier: dict, output: CharacterizationOutput,
    *, max_tokens: int = 1500,
) -> AgentContext: ...


def serialize_for_agent(ctx: AgentContext, output: CharacterizationOutput) -> str:
    """Compute rollups, format JSON, assert/truncate to max_tokens.

    Truncation order: drop lowest-severity findings_ref first, log a warning
    naming the dossier and the dropped count. Never silently exceed budget.
    """
```

### 2.3 Tests for PR2

```
tests/unit/test_flags.py (one test per ProcessFlag, with boundary conditions)
  - STALE: empty list vs non-empty
  - LOW_QUALITY: 0.79 vs 0.80
  - SPARSE: rows=19 vs 20; coverage=49 vs 50
  - MIXED: 1 run vs 2
  - UNKNOWN_PROCESS: each provenance value
  - SPECS_MOSTLY_MISSING: 49% vs 51%
  - multiple flags fire together

tests/unit/test_agent_context.py
  - build returns correct fields from fixture
  - empty CharacterizationOutput round-trips
  - token budget ≤ 1500 on all 3 existing fixtures
  - runtime truncation: oversized findings_ref → warning logged, output ≤ budget
  - rollups computed correctly at serialize time
```

---

## Order of operations + parallelization

```
PR 1 — Identity
  Lane A (independent): registry + domain models + EvidenceLocator
  Lane B (independent): evidence_gated_llm refactor (lifts from narrative_extractor)
  Lane C: identity_extractor                (depends on A + B)
  Lane D: dossier extension + CLI manifest  (depends on C)
  Lane E: tests + eval set                  (depends on D)

PR 2 — Flags + context (after PR 1 merges)
  Lane F (independent): ProcessFlag enum + compute_flags
  Lane G: agent_context builder             (depends on F + PR1)
  Lane H: tests                             (depends on G)
```

A and B run in parallel inside PR1. F can start in parallel with PR1's Lane B (no conflict).

---

## Risk register

| Risk | Mitigation |
|---|---|
| LLM picks plausible-on-list wrong process | variable_fingerprint check downgrades to UNKNOWN |
| LLM hangs/errors | evidence_gated_call returns (None, reason); never raises |
| Empty narrative wastes LLM call | pre-check; short-circuit before calling |
| Token budget exceeded | runtime truncation with warning; test assertion on fixtures |
| Existing fixtures break | regression test on all 3; experiment.process defaults to UNKNOWN |
| narrative_extractor refactor breaks behavior | refactor first as no-op (lift then re-import); existing 116 tests must pass before adding new code |
| Adding to registry forgets fingerprint | registry loader validates fingerprint required field |

---

## NOT in scope

- Spec-free anomaly generator (`find_intra_run_anomalies`)
- Diagnosis / Critic / Orchestrator agents
- Wiring `from_schema_with_overrides` into the pipeline
- KG scenario (b) — insight/causal edges (separate design)
- LLM result caching (TODO with trigger: "when ingest p50 > 10s")
- Multi-process-per-experiment
- `domain_priors` dict
- Identity extraction without a registry

---

## Acceptance gates

- All 116 existing tests still pass
- New tests pass (target ~30 new across PR1 + PR2)
- All 5 identity-extractor evals pass
- `parsevenv/bin/fermdocs ingest <dir>` produces a dossier with `experiment.process` populated (or `UNKNOWN`, both valid)
- `build_agent_context` on every existing fixture serializes ≤1500 tokens
- Adding a new process to the registry is one YAML edit; loader validates fingerprint
- `git log --oneline` reads as a clean trace from refactor → models → extractor → dossier → tests → flags → context

---

## Commit-by-commit shape

Mirroring the trace style used for the schema v2 work:

1. `refactor(mapping): extract evidence_gated_llm primitive from narrative_extractor` ✅
2. `feat(domain): add IdentityProvenance, EvidenceLocator, ScaleInfo, ProcessIdentity` ✅ (v4 shape)
3. `schema: add process_registry.yaml with penicillin_indpensim` ✅
4. `feat(mapping): add identity_extractor with whitelist + fingerprint gate` ✅ (v4 shape)
5. `feat(dossier,cli): wire identity into dossier with manifest override` ✅ (v4 shape)
6. **`refactor(domain,mapping,dossier): split ProcessIdentity into observed + registered`** ← v4.1 follow-up
7. `eval+test(identity): scripted-LLM fixtures and full test coverage` ← Lane E

*PR1 ships, then:*

8. `feat(characterize): add ProcessFlag enum with documented emission rules`
9. `feat(characterize): add agent_context builder with token-budget truncation`
10. `test(characterize): per-flag boundary tests and agent-context coverage`

---

## Execution log

This section is updated as work is implemented. Each entry records date, commit hash, what shipped, and any deviations from the plan.

### 2026-05-01 — PR1 commits 1-5 landed (v4 shape)

- `fc2a87b refactor(mapping): extract evidence_gated_llm primitive from narrative_extractor`
- `c08ff39 feat(domain): add IdentityProvenance, EvidenceLocator, ScaleInfo, ProcessIdentity`
- `975e8d0 schema: add process_registry.yaml with penicillin_indpensim`
- `400a458 feat(mapping): add identity_extractor with whitelist + fingerprint gate`
- `6685060 feat(dossier,cli): wire identity into dossier with manifest override`

All 116 existing tests still pass. The 5 committed commits use the v4 atomic-`ProcessIdentity` shape. The v4.1 split is being applied as a follow-up commit before Lane E (tests + eval set) lands, so committed test coverage matches shipped code shape.

### 2026-05-01 — v4.1 design pivot

User pushback: yeast (and any non-registered process) was losing organism info because v4's atomic `ProcessIdentity` collapsed surface facts and registry classification into one. v4.1 splits them; see Revision log at top. Refactor and Lane E commit together as commits 6-7. No history rewrites — the v4 commits stay as-is, the split lands as `refactor(domain,mapping,dossier): split ProcessIdentity into observed + registered`.

### 2026-05-01 — PR1 commits 6-7 landed (v4.1 split + Lane E)

- `79b1d35 refactor(domain,mapping,dossier): split ProcessIdentity into observed + registered`
- `f1d7b45 eval+test(identity): scripted-LLM fixtures and full test coverage`

160 tests pass. The yeast guard is verified by `evals/identity_extractor/02_off_whitelist`: an *E. coli* paper produces `observed.organism = "Escherichia coli"` + `registered.process_id = null`. PR1 complete.

### 2026-05-01 — PR2 Lanes F + G + H landed

- `feat(characterize): add ProcessFlag enum with documented emission rules`
- `feat(characterize): add agent_context builder with token-budget truncation`
- `test(characterize): per-flag boundary tests and agent-context coverage`

194 tests pass total (160 prior + 21 flag tests + 13 agent_context tests). All 3 existing characterize fixtures (01_boundary, 02_missing_data, 03_multi_run) serialize the AgentContext under the 1500-token budget (~160-300 tokens each). Yeast-style cases verified: `observed.organism` surfaces in `AgentContext.process` even when `registered.process_id` is null; `UNKNOWN_PROCESS` and `UNKNOWN_ORGANISM` are independent flags so cases that have one without the other are preserved.
