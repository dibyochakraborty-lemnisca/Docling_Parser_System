# Plan — process_priors.yaml + organism-aware diagnosis

**Date:** 2026-05-03
**Author:** Pushkar + Claude
**Status:** Draft v1
**Predecessors:** `2026-05-02-execute-python-default.md` (Stage 1+2+3 shipped on main, commit 61491a0)

## 0. Why this plan exists

Stage 3 of the spine-flip plan landed organism *vocabulary* (yeast and E. coli
columns added to golden_schema.yaml) but not organism-aware *bounds*. Today
the diagnosis agent has two grounding paths:

1. **schema_only** — schema nominal/std_dev. Useless for fed-batch trajectory
   questions because schema specs are setpoints, not bounds (the IndPenSim
   9320-flood lesson).
2. **cross_run** — compare batches in the same dossier. Strong when N≥2 runs;
   collapses to nothing on single-run dossiers.

There's a missing third path: **process_priors** — organism × process-family
expected ranges (mu_x_max, qs, Yx/s, expected DO trajectory shape, etc.)
sourced from literature and operator knowledge. With priors loaded, the agent
can reason about a single batch in isolation: "biomass plateaued at 80 g/L —
expected mature S. cerevisiae fed-batch reaches ~120-180 g/L per literature,
this is a yield miss." Without priors, that single-batch case is undiagnosable.

This plan adds the priors layer cleanly without touching the spine flip.

## 1. Scope split — what's NEW vs UNCHANGED

**New:**
- `process_priors.yaml` — registry of organism × process_family → variable bounds
- `ProcessPriors` Pydantic model + loader (parallel to GoldenSchema)
- `get_priors(organism?, process_family?, variable?)` tool on the diagnosis agent
- Diagnosis prompt update — bound-grounding hierarchy: priors > cross_run > schema_only
- Validator update — `confidence_basis = process_priors` allowed only when matching priors exist; auto-downgrade to schema_only when absent
- 3 organism prior sets at launch: *Penicillium chrysogenum*, *S. cerevisiae*, *E. coli*

**Unchanged:**
- `golden_schema.yaml` — vocabulary stays where it is
- Bundle layout, BundleWriter, BundleReader
- Wave 1 schema_only / cross_run reasoning paths
- range_violation detector, aggregation cap
- DiagnosisOutput schema, FailureClaim citation rules
- All ingestion / mapping pipeline
- Stage 1+2+3 infrastructure

The agent's only new capability: ask `get_priors()` and use the answer to
ground a claim. Everything else is the prompt teaching it to reach for that
tool first.

## 2. The priors data model

`process_priors.yaml`:

```yaml
version: "1.0"

organisms:
  - name: "Saccharomyces cerevisiae"
    aliases: ["S. cerevisiae", "Saccharomyces", "yeast", "baker's yeast"]
    process_families:
      - name: "aerobic_fed_batch_glucose"
        description: "Crabtree-controlled glucose-limited fed-batch"
        priors:
          mu_x_max_per_h:
            range: [0.10, 0.30]
            typical: 0.20
            source: "Verduyn 1991; Hensing 1995"
          qs_glucose_g_per_g_per_h:
            range: [0.10, 0.40]
            typical: 0.25
            source: "Sonnleitner 1986"
          yxs_g_per_g:
            range: [0.45, 0.55]
            typical: 0.50
            source: "Verduyn 1991"
          ethanol_g_l:
            range: [0.0, 2.0]
            typical: 0.5
            note: "Ethanol > 5 g/L indicates Crabtree overflow / overfeeding"
            source: "Operator heuristic; van Hoek 1998"
          biomass_endpoint_g_l:
            range: [80.0, 180.0]
            typical: 120.0
            source: "Industrial fed-batch range; Lemnisca operator survey"

  - name: "Escherichia coli"
    aliases: ["E. coli", "BL21", "DH5a"]
    process_families:
      - name: "aerobic_fed_batch_iptg_induction"
        description: "Glucose-limited fed-batch with IPTG-induced protein production"
        priors:
          mu_x_max_per_h:
            range: [0.40, 0.70]
            typical: 0.55
            source: "Korz 1995; Shiloach 2005"
          qs_glucose_g_per_g_per_h:
            range: [0.30, 0.70]
            typical: 0.50
          yxs_g_per_g:
            range: [0.40, 0.50]
            typical: 0.45
          acetate_g_l:
            range: [0.0, 2.0]
            typical: 0.5
            note: "Acetate > 5 g/L typically inhibits growth"
            source: "Eiteman 2006"
          biomass_endpoint_g_l:
            range: [60.0, 100.0]
            typical: 80.0

  - name: "Penicillium chrysogenum"
    aliases: ["P. chrysogenum", "Penicillium"]
    process_families:
      - name: "submerged_fed_batch_paa"
        description: "Submerged fed-batch with PAA precursor feed for penicillin"
        priors:
          mu_x_max_per_h:
            range: [0.05, 0.15]
            typical: 0.10
          mu_p_max_per_h:
            range: [0.020, 0.050]
            typical: 0.035
          paa_mg_l:
            range: [800.0, 2000.0]
            typical: 1400.0
            note: "PAA > 5000 mg/L indicates cessation of biosynthesis (precursor pooling)"
          biomass_endpoint_g_l:
            range: [30.0, 50.0]
            typical: 40.0
```

**Schema rules:**
- `range: [low, high]` — soft bounds. A value outside is a candidate finding,
  not a hard failure.
- `typical: <value>` — point estimate for residual / sigma calculations.
- `source` — required on every prior. No prior ships without a citation.
- `note` — optional operator wisdom. Surfaces in the prompt context.
- Alias matching is case-insensitive substring on organism name + identity
  flags from the dossier.

**What's deliberately NOT in the priors:**
- Setpoint values (those live in dossier `_specs` per-run)
- Equipment constants (kLa, working volume — those depend on bioreactor)
- Anything that varies more by recipe than by organism (feed schedule)

## 3. Loader + lookup API

```python
# src/fermdocs/domain/process_priors.py
from fermdocs.domain.process_priors import (
    ProcessPriors,         # full Pydantic model
    load_priors,           # path → ProcessPriors
    cached_priors,         # lru_cached version
    resolve_priors,        # (organism, process_family) → list[Prior]
    Prior,                 # single variable's bounds + source
)
```

`resolve_priors` is the one the agent's tool calls. Resolution rules:
1. Match organism by name or alias (case-insensitive substring)
2. If `process_family` given, filter to it; else return all matching families
3. Return flat `list[Prior]` with `(variable, range, typical, source, note,
   organism, process_family)` per row
4. Empty list when no match — agent must handle this and downgrade to
   schema_only

## 4. Tool surface addition

One new method on `DiagnosisToolBundle`:

```python
def get_priors(
    self,
    organism: str | None = None,
    process_family: str | None = None,
    variable: str | None = None,
) -> dict:
    """Return process priors (organism + process-family expected ranges).

    Cost: Low. Returns {"priors": [...], "n": int, "matched_organism": str|None,
    "matched_process_family": str|None}.

    Use this BEFORE claiming a value is anomalous on a single-run dossier.
    Without priors, single-run reasoning falls back to schema_only and your
    confidence is capped lower. The bundle's organism/process come from
    get_meta() — pass them through.

    Each prior carries a source citation. Cite the source in your claim's
    summary when you use the prior to ground a finding.
    """
```

Routing wires through `_dispatch_tool_bundle` exactly like the other tools.

The factory's `make_diagnosis_tools` gets a new optional `priors:
ProcessPriors | None` kwarg; when omitted it loads the default
`process_priors.yaml`.

## 5. Confidence basis enforcement

Today `ConfidenceBasis` is an enum with `schema_only`, `process_priors`,
`cross_run`. The validator currently lets the agent claim
`confidence_basis: process_priors` regardless of what's actually loaded —
that's a soundness bug.

**New rule** (in `validators.validate_diagnosis`): for each claim where
`confidence_basis == process_priors`:
1. The claim must reference a variable that has a matching prior in the
   loaded priors set
2. If no matching prior, auto-downgrade to `schema_only` AND set
   `provenance_downgraded=True` AND drop confidence by 0.15 (capped at 0)

The downgrade isn't punitive — it's keeping the agent's audit trail honest.
The user sees `provenance_downgraded=true` and knows the claim's basis
weakened.

## 6. Bundle-mode prompt update

Insert a new GROUNDING section after the existing TOOLS list:

```
GROUNDING HIERARCHY:
  Use this order when claiming a value is anomalous:

  1. process_priors (preferred). If get_priors() returns matching ranges
     for the organism+variable, the claim's confidence_basis is
     "process_priors". Cite the source in your summary.

  2. cross_run (secondary). If priors are absent but >=2 runs exist,
     compare across runs and use confidence_basis="cross_run". Quote
     the magnitude delta (e.g. "RUN-A=43 g/L vs RUN-B=2 g/L").

  3. schema_only (last resort). If neither priors nor multiple runs
     exist, use confidence_basis="schema_only" and cap your confidence
     at 0.6. Note explicitly that recipe-specific priors are unavailable.

  On single-run dossiers, you MUST call get_priors() before emitting
  any failure claim. Skipping this is the leading cause of weak,
  ungrounded diagnoses.
```

## 7. Migration / rollout

**Stage 1 — schema + loader (no behavior change):**
- Add `src/fermdocs/schema/process_priors.yaml` with the 3 organism sets above
- Implement `ProcessPriors` model + `resolve_priors`
- Add `get_priors_version()` to surface in `BundleMeta.process_priors_version`
- Tests: model roundtrip, alias matching, empty-on-mismatch

**Stage 2 — agent tool wiring (additive):**
- Add `get_priors` to `DiagnosisToolBundle`
- Dispatcher route + Gemini/Anthropic schema enum entries
- Tests: tool returns expected shape, organism-aware filtering

**Stage 3 — prompt + validator enforcement:**
- Update `_BUNDLE_SYSTEM_PROMPT` with the GROUNDING HIERARCHY section
- Add validator downgrade: process_priors basis without matching priors → schema_only + downgrade flag + confidence drop
- Re-run IndPenSim eval. Expect: agent reaches for `get_priors` early; basis distribution shifts from cross_run-heavy to a mix of process_priors + cross_run; same or better recall on critical findings.

**Stage 4 — cleanup:**
- Document priors authoring guide in CONTRIBUTING.md
- Add a CI check: every prior in process_priors.yaml has a non-empty `source`

## 8. Eval changes

**New eval cases (3 minimum):**

1. **Single-run yeast diagnosability** — synthetic 1-batch yeast dossier with
   biomass plateau at 60 g/L. Without priors → weak schema_only claim.
   With priors → "biomass endpoint 60 g/L below typical 120 g/L per
   Verduyn 1991 — yield miss".

2. **Single-run E. coli acetate failure** — IPTG induction dossier with
   acetate climbing to 7 g/L. Without priors → no failure (schema has no
   acetate nominal). With priors → "acetate 7 g/L exceeds inhibitory range
   per Eiteman 2006 — likely overfeeding during induction".

3. **Penicillin priors regression** — IndPenSim run with priors loaded.
   The PAA pooling failure that already surfaces should now cite the
   priors-derived "> 5000 mg/L cessation threshold" rather than only
   cross-run delta.

**Reliability eval extension:** N=5 reruns on each capability eval, ≥80%
set-overlap on critical findings, with priors providing more stable
grounding than cross_run alone.

## 9. Open questions

1. **Where do priors come from after launch?** The 3 organism sets are
   operator-curated. Future organisms need a contribution path. Punt to a
   `priors-authoring.md` guide post-launch.
2. **Process-family granularity.** "aerobic_fed_batch_glucose" might be too
   coarse. We may need sub-families (induction vs constitutive, glucose vs
   glycerol). Start coarse; subdivide when an eval failure forces it.
3. **Should priors be versioned per organism?** A new mu_x range should
   bump only that organism's version, not the whole file. Defer; current
   single-version-per-file is enough for first ship.
4. **Operator-asserted priors override.** A specific lab's strain might
   genuinely have mu_max=0.35 even though Verduyn says 0.30. Should the
   dossier carry per-run prior overrides (`_priors_override` block)?
   Yes, but defer to a follow-up. For first ship, the literature priors are
   the floor.

## 10. Non-goals (explicit)

- Strain-specific priors (Pichia X33 vs GS115). Organism-level only.
- Mammalian / CHO / Pichia. Tier A is penicillin, yeast, E. coli only.
- Auto-learning priors from past runs (cross-debate memory). Wave 3+.
- Prior conflict resolution between literature and operator data. Defer.
- Replacing schema nominal/std_dev — schema stays for vocabulary only.

## 11. Stage gating

| Stage | PR title | Gate |
|---|---|---|
| 1 | `feat(priors): add process_priors.yaml + loader for 3 Tier A organisms` | Loader tests pass; existing evals unchanged |
| 2 | `feat(diagnose): get_priors tool + dispatcher routing` | New tool eval cases pass |
| 3 | `feat(diagnose): GROUNDING HIERARCHY prompt + validator basis enforcement` | Single-run yeast + E. coli evals pass; IndPenSim regression green |
| 4 | `chore(priors): authoring guide + CI source-required check` | All evals green |

Stage 3 is the load-bearing change. Stages 1-2 are infra.

## 12. Effort estimate

Total: ~2-3 focused days.

- Stage 1: half a day (YAML + loader + tests)
- Stage 2: half a day (tool + schema enum updates + tests)
- Stage 3: 1 day (prompt iteration + validator + 3 eval cases + IndPenSim re-validation)
- Stage 4: half a day (docs + CI guard)

## 13. Followups recorded but not in this plan

- Per-dossier priors override
- Strain-specific priors layer
- Cross-organism prior sharing (mammalian shares some core but diverges)
- Priors quality dashboard (which priors fired in which diagnoses)
