# Tier-wise metric catalog + verified toolkit

**Status:** Deferred plan, ready to build when prioritized.
**Estimated cost:** ~1000-1300 LOC across catalog + toolkit + integration + tests.
**Triggers this plan:** when we want IndPenSim-shape bundles to produce rich debate instead of 0-2 topics, OR when wet-lab consumers ask for literature-cited audit trails.
**Companion plan:** `plans/2026-05-04-steal-from-langgraph-reference.md` (this is the "Option β" deep-dive).

---

## The architectural play in one diagram

```
┌──────────────────────────────────────────────────────────────────────────┐
│ CHARACTERIZE STAGE                                                       │
│                                                                          │
│  ┌────────────────────┐   declarative menu of 60 metrics                 │
│  │ metric_catalog.py  │   (tier A/B/C, applies_to, default params,       │
│  │                    │    literature sources, output shape)             │
│  └─────────┬──────────┘                                                  │
│            │ reads                                                       │
│            ▼                                                             │
│  ┌────────────────────────┐                                              │
│  │ TrajectoryAnalyzerAgent│   LLM picks WHAT to compute from catalog     │
│  │  (Gemini)              │   for each applicable metric, writes         │
│  │                        │   `execute_python` code that imports         │
│  │                        │   verified toolkit functions                 │
│  └─────────┬──────────────┘                                              │
│            │ calls via execute_python                                    │
│            ▼                                                             │
│  ┌──────────────────────────────────────────────────────────────────┐    │
│  │ toolkit/fermentation.py    verified Python (numpy + pandas + scipy)│  │
│  │  compute_mu()                  segment_growth_phases()             │  │
│  │  doubling_time()               compute_phasewise_qp()              │  │
│  │  reconstruct_volume()          compute_carbon_balance()            │  │
│  │  compute_RQ()                  compute_kla_van_t_riet()            │  │
│  │  ... (no LLM in math hot path; deterministic + tested)             │  │
│  └─────────┬────────────────────────────────────────────────────────┘    │
│            │ returns numbers                                             │
│            ▼                                                             │
│  Findings emitted with:                                                  │
│   - type=trajectory_pattern                                              │
│   - statistics["metric_id"]    e.g. "A8", "B10", "C5"                    │
│   - statistics["literature_source"] (when Tier C)                        │
│   - evidence_observation_ids (real CSV cells)                            │
└─────────────────────┬────────────────────────────────────────────────────┘
                      ▼
┌──────────────────────────────────────────────────────────────────────────┐
│ DIAGNOSE STAGE  (the interpreter — unchanged architecture)               │
│                                                                          │
│  Reads: 30 findings (mix of range_violation + trajectory_pattern)        │
│         narrative_observations                                           │
│         trajectories (raw access still available via execute_python)     │
│                                                                          │
│  Job: INTERPRETATION — combine multiple findings into causal stories.    │
│   Diagnose does NOT recompute. It uses the numbers characterize already  │
│   calculated and trusted.                                                │
│                                                                          │
│  Example:                                                                │
│   "I see F-0108 with metric_id=B10 (mean RQ=1.42),                       │
│    F-0115 with metric_id=B16 (carbon balance=0.78),                      │
│    F-0103 with metric_id=A10 (phase segmentation showing                 │
│    decline phase starting at 144h).                                      │
│    → emit FailureClaim: 'B1 carbon balance fails because overflow        │
│       metabolism dumps carbon into ethanol/acetate post-36h'             │
│       cited_finding_ids=[F-0108, F-0115, F-0103]"                        │
└─────────────────────┬────────────────────────────────────────────────────┘
                      ▼
┌──────────────────────────────────────────────────────────────────────────┐
│ HYPOTHESIS STAGE  (the debate — unchanged architecture)                  │
│                                                                          │
│  Diagnose claims become seed_topics. Specialists are filtered by         │
│  metric_id family (kinetics gets A8/A9/A10/A11/A13/B10; mass_transfer    │
│  gets A14/A15/A17/A18/B8/B14/C9; metabolic gets B6/B10/B16/B17/B18).     │
│                                                                          │
│  Each specialist now has 5-8 grounded findings to argue from instead     │
│  of 1-2. Synthesizer gets richer facets. Critic + judge can verify       │
│  any number by metric_id without recomputing.                            │
│                                                                          │
│  Feedback loop (already shipped) handles retries with cross-attempt      │
│  visibility.                                                             │
└──────────────────────────────────────────────────────────────────────────┘
```

---

## Why this architecture is the right shape

The reference repo got the separation right and we'd be silly not to copy it:

- **Catalog** = declarative menu (what's computable, with what defaults, citing what literature). No code execution.
- **Toolkit** = imperative implementation (verified pandas/numpy math, no LLM in the hot path). Deterministic, tested, fast.
- **LLM as planner only** = picks which catalog entries apply to the current bundle, writes `execute_python` code that imports the right toolkit function, parses the result. Math is never invented.

The properties this gives us:
- **Reproducibility** — same inputs, same numbers (LLM can't fudge math)
- **Auditability** — every number traces to a specific function call with logged code + inputs
- **Speed** — pure pandas, no per-cell LLM cost
- **Domain rigor** — literature citations baked into parameter defaults, not asked of the LLM at runtime
- **Cross-bundle queries** — `metric_id` is stable across bundles, so "show every B10 finding where mean RQ > 1.3" is just SQL

---

## Stage-by-stage spec

### Stage 1 — Characterize gains a verified toolkit + structured catalog

**Files to create:**

```
src/fermdocs_characterize/
├── agents/
│   ├── metric_catalog.py        (NEW — port of analysis_catalog.py)
│   └── trajectory_analyzer.py   (MODIFIED — operates against catalog)
├── toolkit/                     (NEW directory)
│   ├── __init__.py
│   ├── fermentation.py          (NEW — port of fermentation_toolkit.py,
│   │                              trimmed for our column conventions)
│   ├── kinetics.py              (NEW — A1..A13: μ, doubling, phases, Qp)
│   ├── operational.py           (NEW — A14..A18: DO margin, excursions,
│   │                              tip speed, P/V)
│   ├── cross_run.py             (NEW — A19..A24: KPI tables, deviation,
│   │                              variance decomposition, completeness)
│   ├── balances.py              (NEW — B6..B18: yields, RQ, OUR, CER,
│   │                              C/N/γ balances)
│   └── literature.py            (NEW — C1..C16: literature constants
│                                  loader + Tier C estimators)
```

**Catalog structure** (`metric_catalog.py`) — ported with our naming conventions but identical shape:

```python
@dataclass(frozen=True)
class CatalogEntry:
    metric_id: str             # "A8" | "B10" | "C5" — stable across runs
    tier: Literal["A", "B", "C"]
    short_description: str
    long_description: str      # 2-4 sentences with formula sketch
    applies_to: str            # eligibility rule in plain English
    required_inputs: tuple[InputSpec, ...]
    required_parameters: tuple[ParamSpec, ...]
    output_shape: Literal[
        "scalar", "scalar_with_metadata", "timecourse_csv",
        "table_csv", "multi_run_summary", "per_run_scalar",
    ]
    output_columns: tuple[str, ...]
    toolkit_fn: str            # "fermdocs_characterize.toolkit.kinetics:compute_mu"

    def is_precomputable(self) -> bool: ...
```

The `toolkit_fn` field is our addition vs. the reference — points the LLM at the
exact import path so the prompt can say "for metric A8, call
`from fermdocs_characterize.toolkit.kinetics import compute_mu`". Removes one layer
of LLM creativity that doesn't help us.

**Toolkit functions** (`toolkit/*.py`) — ported and trimmed:

The reference toolkit is ~600 LOC but ~30% of it is organism-specific
(carotenoid columns, B01-B03 vs B04-B06 batch convention, IndPenSim-shaped
feed segments). What we port:

| Reference function | Our home | Generalize? |
|---|---|---|
| `load_batch_data` | drop | their column convention only |
| `reconstruct_volume` | `kinetics.py` | yes — generalize against golden_schema variables |
| `compute_dcw` | drop | OD↔DCW conversion is organism-specific; let LLM compute or skip |
| `compute_mu` | `kinetics.py` | yes — Savitzky-Golay over `(time, biomass)`, organism-agnostic |
| `savgol_smooth` | `kinetics.py` (helper) | yes — pure numerical |
| `segment_growth_phases` | `kinetics.py` | yes — μ-threshold-based, parameterized |
| `compute_carbon_balance` | `balances.py` | yes — needs C-content per substrate; literature lookup |
| `estimate_glucose_concentration` | `balances.py` | yes — when residual S unmeasured |
| `estimate_our` | `balances.py` | yes — Verduyn yields parameterized |
| `estimate_kla` | `literature.py` | yes — Van't Riet correlation, organism-agnostic |

Plus we ADD functions for catalog entries the reference doesn't have a
toolkit fn for (their tier C lookups did literature delegation; we hard-code
the constants):

- `henry_constant_O2(T_K, pressure_atm)` → C* via Schumpe 1982
- `verduyn_yields(organism: Literal["s_cerevisiae", "e_coli", "p_chrysogenum"])` → returns Y_XO2_max, m_O2, Y_XS_max, m_S
- `van_t_riet_kla(P_per_V, vvm, alpha=0.026, beta=0.4, gamma=0.5)` → kLa
- `nienow_mixing_time(P_per_V, V_L, D_m)` → t_mix
- `mu_max_from_organism(organism)` → reference doubling time

These constants live in `toolkit/literature.py` as a module-level dict so the
characterize stage NEVER does a network lookup at runtime. Reproducibility +
no Tier C timeout bug the reference hit on IndPenSim.

**Trajectory analyzer prompt update** (`trajectory_analyzer.py`):

Replace the open-ended "find patterns" framing with a checklist contract.
Steal verbatim from `ANALYZE_SYSTEM_PROMPT` in the reference:

> "For each metric in the catalog applicable to this bundle, EITHER compute it
> via the toolkit function OR add a Data Gap entry naming the metric_id and the
> specific missing input. Silent skipping is forbidden."

Plus the language test (✅/❌ examples) and forbidden sections (no causal
language, no recommendations) — both already in the steal-list, this is where
they land.

The prompt tells the LLM: catalog at `metric_catalog.CATALOG`, toolkit at
`fermdocs_characterize.toolkit.*`. Each catalog entry's `toolkit_fn` field
gives the import path. LLM writes:

```python
from fermdocs_characterize.toolkit.kinetics import compute_mu, segment_growth_phases
df = pd.read_csv(obs_path)
mu_per_run = {}
for run_id, g in df.groupby('run_id'):
    bio = g[g['variable'] == 'biomass_g_l'].dropna(subset=['value'])
    if len(bio) >= 5:
        mu_per_run[run_id] = compute_mu(bio['time_h'], bio['value'])
print(mu_per_run)
```

LLM decides which metrics apply, runs them, reads stdout, builds Findings.

**Finding population** (`_coerce_pattern_to_finding` in trajectory_analyzer.py):

Today our coercer puts `pattern_kind` (string) into `statistics`. After this:

```python
finding = Finding(
    finding_id=f"{char_id}:F-{idx:04d}",
    type=FindingType.TRAJECTORY_PATTERN,
    summary=f"{catalog_entry.short_description}: {computed_summary}",
    confidence=0.85,                          # toolkit math is deterministic
    extracted_via=ExtractedVia.STATISTICAL,   # NOT LLM_JUDGED — math came from toolkit
    tier=Tier.B,                              # mirrors catalog tier mapping
    statistics={
        "metric_id": catalog_entry.metric_id,
        "tier": catalog_entry.tier,
        "literature_source": "Verduyn 1990" if tier_c else None,
        "params_used": {...},                  # which defaults vs overrides
        # plus the actual numbers
    },
    ...
)
```

The shift from `LLM_JUDGED` to `STATISTICAL` is meaningful — the existing schema
caps LLM_JUDGED at 0.85 confidence. Toolkit-computed findings come out of a
verified function with deterministic math, so the cap shouldn't apply.
`STATISTICAL` is the right `ExtractedVia`.

---

### Stage 2 — Diagnose adapts to the richer Finding stream

**Files to modify:**
- `src/fermdocs_diagnose/agent.py` (prompt only)

**No schema changes needed.** Diagnose already accepts `Finding` objects with
arbitrary `type`. The only update is in the system prompt:

Add a section explaining the catalog → finding → claim flow:

> **HOW TO READ THESE FINDINGS:**
>
> Each `trajectory_pattern` finding carries a `metric_id` in its statistics
> dict — e.g. "A8" (specific growth rate), "B10" (RQ overflow), "C5" (qs
> estimated from Verduyn yields). These are stable identifiers from the
> characterize-stage metric catalog. **The numbers are pre-computed and
> verified — you do NOT need to recompute them via execute_python.** Cite
> the finding_id; the metric_id is part of the audit trail.
>
> When emitting claims, prefer to cite multiple metric_ids that triangulate
> the same conclusion. Example:
>
> > FailureClaim: "B1 carbon balance fails post-36h. Cited:
> > F-0108 (B10 RQ=1.42 sustained = overflow flag),
> > F-0112 (B16 carbon balance closure 0.78 vs B2's 0.94),
> > F-0110 (B6 ethanol yield 0.31 g/g, well above respiratory baseline 0.05)."
>
> Tier hierarchy:
> - **Tier A** (always cheap, almost always present): A1..A24
> - **Tier B** (measured inputs required, often missing): B1..B20
> - **Tier C** (literature-assisted estimates with citations): C1..C16
>
> When citing a Tier C finding, mention "via Verduyn 1990" or whatever
> literature source is in `statistics["literature_source"]` so the user
> can audit.

Plus a recap of the existing rules — diagnose still does interpretation only,
no math, no causal language without grounding.

**Why this is a small change:** the heavy lift was characterize. Diagnose's
existing prompt already says "interpret the findings, emit claims that cite
them." We're just adding context about *what kind* of findings it'll see.

---

### Stage 3 — Hypothesis stage gets richer specialist views

**Files to modify:**
- `src/fermdocs_hypothesis/state.py` (`specialist_domain_tags` extension)
- `src/fermdocs_hypothesis/projector.py` (filtering by metric_id family)
- `src/fermdocs_hypothesis/agents/specialist_*.py` (prompt updates)

**Step 3a — Specialist filtering by metric_id family**

Today specialists filter findings by tag/variable overlap. After the catalog
exists, we add metric_id family filtering:

```python
# state.py — extend the existing function

KINETICS_METRICS = frozenset({"A8", "A9", "A10", "A11", "A13", "A23", "B10"})
MASS_TRANSFER_METRICS = frozenset({"A14", "A15", "A17", "A18", "B8", "B11",
                                   "B14", "B15", "C3", "C4", "C9", "C13"})
METABOLIC_METRICS = frozenset({"B6", "B10", "B16", "B17", "B18", "C5",
                               "C6", "C7", "C10"})
DATA_QUALITY_METRICS = frozenset({"A24"})  # data scientist domain (future)


def specialist_metric_ids(role: SpecialistRole) -> frozenset[str]:
    if role == "kinetics":
        return KINETICS_METRICS
    if role == "mass_transfer":
        return MASS_TRANSFER_METRICS
    if role == "metabolic":
        return METABOLIC_METRICS
    return frozenset()
```

**Step 3b — Projector filters findings using both signals**

Today's projector (`project_specialist`) filters by tags + variables. After:

```python
def _finding_relevant(f: FindingRef) -> bool:
    # Existing tag-based logic stays
    if f.finding_id in cited_finding_ids:
        return True
    # ... existing variable + tag matches ...

    # NEW: metric_id-based match for trajectory_pattern findings
    metric_id = f.statistics.get("metric_id") if f.statistics else None
    if metric_id and metric_id in specialist_metric_ids(role):
        return True

    return False
```

This means kinetics specialist sees every A8/A9/A10/A11/A13/A23/B10 finding
on the topic AUTOMATICALLY, regardless of whether the topic's tags happen
to match. Specialists become naturally domain-grounded.

**Step 3c — Specialist prompts mention the catalog**

Specialist system prompts get a small addition:

> The findings in your view carry `statistics.metric_id` from the
> characterize-stage catalog. Your domain covers: A8 (μ(t)), A9 (doubling
> time), A10 (phase segmentation), A11 (phasewise μ), A13 (phasewise Qp),
> A23 (productivity reduction), B10 (RQ + overflow flag). Cite these
> metric_ids in your facet summary so the synthesizer can triangulate.

Three ~10-line prompt additions, one per specialist.

**Step 3d — Synthesizer prompt mentions cross-metric triangulation**

Synthesizer already integrates facets. New rule:

> When facets cite different metric_ids that point at the same conclusion
> (e.g. kinetics cites A8 + A10 for B1 declining post-144h, metabolic cites
> B10 + B16 for B1 overflow post-36h), surface the cross-metric pattern
> explicitly: "The metabolic and kinetic evidence converge — overflow
> metabolism (B10) coincides with carbon balance failure (B16) and growth
> arrest (A10) in the same time window."

This is what produces the qualitative shift from "shallow topic + 1
finding" to "deep topic + 5 findings cross-referenced."

---

## Build order (ship this in 2-3 PRs)

### PR 1 — Catalog + skeleton toolkit (~500 LOC)
- `metric_catalog.py` with all 60 entries (port + our naming)
- `toolkit/__init__.py` + skeleton modules
- `toolkit/kinetics.py` with A8 (`compute_mu`), A9 (`doubling_time`), A10 (`segment_growth_phases`), A11 (phasewise μ helper)
- Tests for each toolkit function with synthetic data
- Trajectory analyzer prompt updated to mention catalog (but still works in old open-ended mode if catalog is empty)

This PR alone moves IndPenSim from 2 patterns to ~12 grounded Tier A findings.
Visibly richer immediately.

### PR 2 — Tier B + balances (~400 LOC)
- `toolkit/balances.py` with B6 (`compute_byproduct_yield`), B10 (`compute_rq`), B16 (`compute_carbon_balance_closure`)
- `toolkit/operational.py` with A14 (`compute_do_margin`), A15 (`controller_excursions`), A17 (`tip_speed`), A18 (`p_per_v`)
- `toolkit/cross_run.py` with A19 (`cross_run_kpi_table`), A20 (`pairwise_deviation`), A21 (`variance_decomposition`)
- Tier B inputs check enforced in catalog (graceful data-gap when measurement missing)
- Diagnose prompt updated to read trajectory_pattern findings as primary evidence

### PR 3 — Tier C literature + specialist filtering (~300 LOC)
- `toolkit/literature.py` with hardcoded constants for s_cerevisiae,
  e_coli, p_chrysogenum (avoids the network-lookup timeout the reference
  hit on IndPenSim)
- C1, C5, C9, C10, C11, C16 implementations
- Specialist projector update with metric_id family filtering
- Specialist + synthesizer prompts updated to use metric_ids in citations

### Tests at every step
- Catalog roundtrip (every `metric_id` resolves to a real toolkit function)
- Each toolkit function: synthetic CSV in, expected number out, +/- tolerance
- Integration: full pipeline on a fixture IndPenSim-shape CSV produces ≥10
  trajectory_pattern findings with valid metric_ids
- Specialist filter: kinetics finding count > 0 when bundle has biomass
  trajectory; mass_transfer finding count > 0 when bundle has DO/RPM/airflow
- E2E: run characterize + diagnose + hypothesis on a fixture, assert ≥3 seed
  topics survive the spec-only filter, assert hypothesis output cites ≥2
  metric_ids per claim

---

## What this change DOES NOT do

Worth being explicit on the limits:

- **Doesn't make hypotheses correct.** Richer evidence ≠ better conclusions.
  LLM still has to reason well. What it buys is more catchable contradictions
  + denser audit trail.
- **Doesn't fix CSV-only narrative gap.** No PDF prose = no operator-witnessed
  events; debates will be quantitatively rich but causally guess-y on
  CSV-only bundles.
- **Doesn't replace the feedback loop.** Catalog adds ammunition; feedback
  loop adds discipline. Both matter; this plan is purely the ammunition side.
- **Tier C estimates need organism-specific yield/maintenance constants — but
  the system still generalizes.** Tier A (24 metrics: μ, doubling time, phase
  segmentation, cross-batch variance, DO margin, controller excursions, etc.)
  and Tier B (20 metrics: yields, RQ, OUR/CER, carbon balance) are fully
  organism-agnostic — pure trajectory math and mass balance against measured
  columns. Only the ~16 Tier C metrics ("qs estimated from Verduyn yields",
  "kLa back-calculated") need organism-specific reference values, and those
  consume our existing `fermdocs.domain.process_priors` registry rather than
  hardcoded dicts. When the bundle's organism isn't in the priors registry,
  Tier C metrics emit data-gap entries citing the missing prior; Tiers A + B
  still produce their full output (~44 grounded findings). Adding a new
  organism is a YAML edit to `process_priors.yaml`, not a code change. Users
  can also supply per-run constants via the dossier, which override priors
  for that run only.
- **Doesn't replace specialist routing.** Specialist routing plan
  (`plans/2026-05-04-specialist-routing.md`) is separate. Catalog makes
  routing easier (specialists declare which metric_ids they consume) but
  doesn't BUILD the routing layer.

---

## Cost vs. benefit summary

| Dimension | Today | After PR 1 | After PR 2 | After PR 3 |
|---|---|---|---|---|
| Findings on IndPenSim | ~5 | ~12 | ~22 | ~28 |
| Seed topics surviving filter | 0 | 3-5 | 5-8 | 5-10 |
| Topics actually debated | 0 | 3 | 3 | 3 (top-K) |
| Findings cited per topic | — | 2-3 | 3-5 | 5-8 |
| Specialists with substantive input | 0/3 | 1-2/3 | 2-3/3 | 3/3 |
| Wall-clock | ~2 min | ~5 min | ~8 min | ~12 min |
| Cumulative LOC | 0 | 500 | 900 | 1200 |

After PR 3, IndPenSim should produce output comparable in depth to what
the reference repo produces in ~50 minutes — but in ~12 minutes because
we skip their network-lookup timeouts and don't do the multi-round
voting.

---

## Cross-references

- `plans/2026-05-04-steal-from-langgraph-reference.md` — companion steal-list. This plan is the "Option β" deep-dive. Items 0 (catalog), 2 (tier vocabulary), 11 (case file framing) all land here.
- `plans/2026-05-04-specialist-routing.md` — specialist routing plan. Catalog makes the routing layer cleaner (metric_id families instead of opaque tag lists).
- `plans/2026-05-04-user-question-and-hitl.md` — user-question plan. Catalog gives the user_question feature a richer substrate to reason against (user can ask "which Tier B metrics couldn't be computed?" and get a structured answer).
- Reference repo: `~/fermentation-debate-langgraph/`
  - `src/tools/analysis_catalog.py` — source of the metric catalog
  - `src/tools/fermentation_toolkit.py` — source of the verified toolkit
  - `src/agents/analyze_prompt.py` — source of the tier-aware prompt rules
