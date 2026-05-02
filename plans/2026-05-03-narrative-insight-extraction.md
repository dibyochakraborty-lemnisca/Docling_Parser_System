# Plan — narrative insight extraction from PDF prose

**Date:** 2026-05-03
**Author:** Pushkar + Claude
**Status:** Draft v1
**Predecessors:**
  - `2026-05-02-execute-python-default.md` (Stage 1+2+3 shipped, commit 61491a0)
  - `2026-05-03-process-priors.md` (priors plan — should land first)

## 0. Why this plan exists

Real industrial fermentation reports — the carotenoid PDFs are the canonical
example — carry the *actual diagnosis* in prose, not tables:

- "terminated at 82h due to onset of cell death and the appearance of white
  cells during centrifugation"
- "200 mL of Isopropyl Myristate (IPM) was added directly to the reactor"
- "the gradual increase in feed rate ensured that glucose concentrations
  remained below inhibitory levels"

Today the ingestion pipeline parses tables well and drops everything else.
For the carotenoid case that means the agent sees biomass numbers but cannot
see "white cells = pigment loss = carotenoid yield collapse" — which is the
*entire scientific point* of the campaign. Without prose extraction the
system has a structural blind spot: it diagnoses what was measured, never
what was observed.

This plan adds a narrative extraction layer: ingestion identifies prose
insights, characterization carries them as a typed collection alongside
findings, and the diagnosis agent can cite them.

## 1. Scope split — what's NEW vs UNCHANGED

**New:**
- `NarrativeObservation` Pydantic model with closed `tag` vocabulary
- LLM-driven prose-extraction pass during ingestion (Gemini, structured output)
- `narrative_observations` field on `CharacterizationOutput`
- Bundle persists `characterization/narrative_observations.json` alongside
  the existing `characterization.json`
- `get_narrative_observations(run_id?, tag?)` tool on the diagnosis agent
- Prompt update — agent must check narrative observations before claiming
  what happened in a run

**Unchanged:**
- Table parsing logic, units, mapping pipeline
- range_violation detector
- Bundle schema version — narrative is additive (defaults to empty list)
- DiagnosisOutput shape — narrative claims slot into existing FailureClaim /
  TrendClaim / AnalysisClaim with `cited_narrative_ids`
- All execute_python infrastructure
- Process priors layer (Plan A)

## 2. The narrative observation model

```python
class NarrativeTag(str, Enum):
    CLOSURE_EVENT = "closure_event"          # "terminated at 82h, cell death"
    DEVIATION = "deviation"                  # "DO dropped to 20% at 78h"
    INTERVENTION = "intervention"            # "200 mL IPM added at 24h"
    OBSERVATION = "observation"              # "white cells during centrifugation"
    CONCLUSION = "conclusion"                # "yield was 30% below target"
    PROTOCOL_NOTE = "protocol_note"          # method-section detail


class NarrativeObservation(BaseModel):
    """A prose-derived insight from the source document.

    These bypass the deterministic finding pipeline — they are pre-structured
    statements from operators, scientists, or report authors. The agent
    treats them as direct evidence rather than something to be re-inferred.
    """
    narrative_id: str = Field(pattern=r"^N-\d{4,}$")
    tag: NarrativeTag
    text: str = Field(min_length=1, description="Verbatim or near-verbatim quote.")
    source_locator: SourceLocator = Field(
        description="Page, section, paragraph index from the source PDF."
    )
    run_id: str | None = Field(default=None, description="When attributable to a single run.")
    time_h: float | None = Field(default=None, description="When attributable to a time point.")
    affected_variables: list[str] = Field(default_factory=list)
    confidence: float = Field(ge=0.0, le=0.85, description="LLM extractor's confidence.")
    extraction_model: str = Field(description="Model+version that emitted this.")
```

**Tag taxonomy rules:**
- Closed enum. Adding a tag is a code change + eval.
- One tag per observation. If a sentence carries multiple tags, the
  extractor splits it into multiple observations.
- `text` is verbatim from the source where possible. The extractor MAY
  normalize whitespace; it MUST NOT paraphrase.

**What's NOT a narrative observation:**
- Numerical values from tables (those are observations / trajectories)
- Variable names from headers (those are mapping inputs)
- Process priors (those live in process_priors.yaml)
- Causal claims ("ethanol overflow caused cell death") — observational only

## 3. Extraction pipeline

A new ingestion stage between table parsing and dossier assembly:

```
PDF text → existing table extractor → table_observations (unchanged)
        ↓
        prose blocks  →  NarrativeExtractor (Gemini)  →  narrative_observations
                                                      ↓
                  table_observations + narrative_observations → dossier
```

`NarrativeExtractor` runs ONE Gemini call per document (not per page —
context window allows the whole-doc pass for typical reports under ~50KB).
Structured output schema mirrors the `NarrativeObservation` model with
the closed tag enum.

**Cost envelope:**
- Per-PDF: 1 LLM call, structured output, ~10-30K input tokens, ~2-5K output
- At Gemini 3.1 Pro pricing: ~$0.10-0.30 per document
- Cap: refuse extraction if PDF text > 200KB (split into chunks, defer
  cross-chunk dedup to a follow-up)

**Failure modes:**
- Extractor returns empty list → no narrative observations, run continues
- Extractor returns malformed JSON → caught, ingestion logs warning,
  empty list falls through. Never blocks the rest of the pipeline.
- Extractor times out (60s) → empty list

The narrative layer is **additive**. A document with no extractable prose
insights produces an empty `narrative_observations` list, and downstream
behaves identically to today.

## 4. Characterization integration

Add to `CharacterizationOutput`:

```python
narrative_observations: list[NarrativeObservation] = Field(default_factory=list)
```

Validator: `narrative_id` shape `N-NNNN`, namespaced to characterization_id
like findings (`<characterization_id>:N-NNNN`).

Also add to `OpenQuestion` and the claim types:

```python
cited_narrative_ids: list[str] = Field(default_factory=list)
```

`FailureClaim._has_citation` validator now accepts narrative-only citations:

```python
if (
    not self.cited_finding_ids
    and not self.cited_trajectories
    and not self.cited_narrative_ids
):
    raise ValueError(...)
```

## 5. Bundle persistence

`BundleWriter`:
```python
def write_narrative_observations(self, observations_json: str) -> Path:
    """Persist NarrativeObservation list at
    characterization/narrative_observations.json."""
```

Optional file. `BundleReader.get_narrative_observations_json()` returns the
text or empty list when missing. Bundles produced before this plan stay
backward-compatible — readers just see no narrative.

`bundle_schema_version` stays "1.0" (this is additive). The first time a
truly breaking layout change ships, we bump.

## 6. Diagnosis agent — new tool

```python
def get_narrative_observations(
    self,
    *,
    run_id: str | None = None,
    tag: str | None = None,
    variable: str | None = None,
    limit: int = 50,
) -> dict:
    """Return prose insights extracted from the source document.

    Cost: Low. Returns {"observations": [...], "n": int, "tags_present": [...]}.

    These are direct statements from the report author / operator —
    closure events ("white cells observed"), interventions ("IPM added"),
    deviations, conclusions. Treat them as primary evidence: if the report
    says cells died, they died, regardless of what biomass numbers show.

    You MUST call this once before submit_diagnosis on any bundle that
    has narrative observations. Skipping it is the leading cause of
    diagnoses that miss the actual story.
    """
```

Routing: `_dispatch_tool_bundle` registers it; Gemini/Anthropic schema enum
gets `get_narrative_observations` added.

## 7. Bundle-mode prompt update

After the GROUNDING HIERARCHY section (added in Plan A), insert:

```
NARRATIVE EVIDENCE (highest priority):
  Reports often carry the actual diagnosis in prose, not numbers. Before
  emitting any failure or conclusion, call get_narrative_observations()
  and read every observation tagged closure_event, deviation, or
  observation. These are direct statements from the people who ran the
  experiment.

  When a narrative observation contradicts your numerical analysis,
  the narrative wins for the OBSERVATION (e.g. cells died) but you
  should still report the numerical context (e.g. biomass plateaued
  at 80 g/L when this happened).

  Cite narrative IDs in cited_narrative_ids, just like finding_ids.
  Closure events and operator interventions almost always belong in
  failures or analysis. Pure protocol notes belong in nothing — skip them.
```

## 8. Migration / rollout

**Stage 1 — schema additions only (no behavior change):**
- Add `NarrativeObservation` + `NarrativeTag` to characterize schema
- Add `narrative_observations` field to `CharacterizationOutput` (defaults
  empty)
- Add `cited_narrative_ids` to all claim types and OpenQuestion
- Update FailureClaim validator to accept narrative-only citations
- Existing fixtures backfill with empty list — zero regressions
- Tests: schema roundtrip, validator covers narrative-only failure citation

**Stage 2 — extractor implementation:**
- `NarrativeExtractor` Gemini client + structured output schema
- Wire into ingestion pipeline (post-table-parse, pre-dossier-assemble)
- Per-document size cap, timeout, error swallowing
- Unit tests with scripted Gemini client returning canned narrative observations
- Integration test on the carotenoid PDF — must extract ≥1 closure_event
  ("white cells") and ≥1 intervention ("IPM added")

**Stage 3 — bundle + tool wiring:**
- `BundleWriter.write_narrative_observations` + `BundleReader.get_*`
- characterize CLI writes the file when bundle is enabled
- `get_narrative_observations` tool on `DiagnosisToolBundle`
- Dispatcher route + Gemini/Anthropic enum entries
- Tests: tool returns expected shape, filters by tag/run_id

**Stage 4 — prompt + eval:**
- `_BUNDLE_SYSTEM_PROMPT` gets the NARRATIVE EVIDENCE section
- Re-run carotenoid PDF case end-to-end. Gate: at least one failure cites
  the white-cells closure event and identifies pigment loss as the actual
  story.
- Re-run IndPenSim regression. Should be unchanged (no narrative present).

## 9. Eval changes

**New eval cases:**

1. **Carotenoid PDF capture (the primary target)** — full 6-batch report
   ingested. Gates:
   - ≥1 closure_event extracted per batch ("white cells", "cell death")
   - ≥1 intervention extracted ("IPM added at B05/B06")
   - Diagnosis emits a failure citing the closure events that explicitly
     names pigment loss / yield failure
   - Open question references the prose ("white cells were observed; was
     this checked against carotenoid yield assays?")

2. **Single-run narrative-only test** — synthetic dossier with one
   batch and a "process upset at 48h, foaming, antifoam added" prose
   block but normal-looking numerics. Without narrative extraction the
   agent finds nothing. With it, the agent emits an intervention-tagged
   trend or analysis claim.

3. **No-narrative regression** — IndPenSim CSV (no prose). Output should
   be identical (within reliability tolerance) to the pre-narrative
   version. Catches accidental narrative-required prompts breaking
   numerical cases.

**Reliability eval extension:** the existing N=5 reliability eval (Plan A)
gets a "narrative present" axis. Track whether narrative-grounded claims
are more or less stable across reruns than purely numerical ones.

## 10. Open questions

1. **Multi-document dossiers.** A campaign report covering 6 batches in one
   PDF vs 6 separate PDFs. The extractor needs to attribute each
   observation to a specific run_id. Heuristic: extract per-document, then
   align via section headers ("BATCH-04 REPORT"). If alignment fails,
   `run_id=None` and the agent treats the observation as campaign-level.
2. **OCR'd vs text-extracted PDFs.** Bad OCR produces noise that the
   extractor will hallucinate against. Add a confidence gate: extractor
   confidence < 0.5 → drop. Defer.
3. **Privacy and confidentiality.** Reports often have redacted fields
   (`COMPANY_ABC`, `STRAIN_X`). Extractor must preserve redactions
   verbatim, never expand them. Test for this.
4. **Section detection.** "Procedure" sections describe what was *planned*;
   "Results" sections describe what *happened*. The extractor should weight
   Results-section observations higher. Open: do we tell the LLM, or
   pre-segment in code?
5. **Cross-document narrative dedup.** "Method identical to Batch-01"
   shouldn't produce duplicate narrative observations across batches.
   Defer; first version produces some redundancy and we measure.

## 11. Non-goals (explicit)

- LLM-driven anomaly detection in narrative ("the report sounds worried —
  flag a failure"). Extraction only emits what's explicitly stated.
- Cross-document fact merging (one campaign → unified knowledge graph).
  Wave 3+.
- Narrative observation editing post-extraction. Immutable per run.
- Multi-language support. English-language reports only at first.
- Translation of organism-specific jargon. The agent reads narrative as-is.

## 12. Stage gating

| Stage | PR title | Gate |
|---|---|---|
| 1 | `feat(characterize): NarrativeObservation schema + claim citation support` | Existing evals unchanged; backfill empty |
| 2 | `feat(ingest): Gemini narrative extractor with closed-tag schema` | Carotenoid PDF extraction unit test passes |
| 3 | `feat(diagnose,bundle): persist narrative observations + get_narrative_observations tool` | Tool returns expected shape on real bundle |
| 4 | `feat(diagnose): NARRATIVE EVIDENCE prompt + carotenoid eval target` | Carotenoid eval passes; IndPenSim regression green |

Stage 4 is the load-bearing behavior change. Stages 1-3 are infrastructure
that ship without changing diagnosis output.

## 13. Effort estimate

Total: ~1-2 weeks of focused work.

- Stage 1: 1 day (schema + validators + tests)
- Stage 2: 3-4 days (extractor + Gemini structured output + ingestion
  wiring + carotenoid integration test)
- Stage 3: 1-2 days (bundle persistence + tool + dispatcher)
- Stage 4: 2-3 days (prompt iteration + carotenoid eval validation +
  reliability eval)

The 3-4 day Stage 2 is the bulk. Most of it is iterating the extraction
prompt against real reports until the closure_event / intervention /
observation tagging is consistent.

## 14. Followups recorded but not in this plan

- Multi-language extraction
- Per-section weighting (Results > Procedure)
- Cross-document narrative graph
- Operator-asserted narrative override (user adds observations manually)
- Narrative-only diagnosis mode (when no numerics are available — pure
  text reports)
- Narrative confidence scoring per tag — closure_events should probably
  weight higher than protocol_notes when the agent reasons over them

## 15. Why this comes after Plan A (process priors)

Plan A unlocks single-run diagnosability with priors. Plan B unlocks
prose-grounded diagnosis with narrative. The two are independent in code —
they touch different pipeline stages — but they couple in agent behavior:

- With priors AND narrative: the agent has "expected ranges per organism"
  AND "what the operator actually observed." It can produce diagnoses on
  par with a competent fermentation engineer reading the report.
- With priors only (post-Plan-A, pre-Plan-B): the agent is good on CSV
  data, weaker on PDFs.
- With narrative only (Plan B without Plan A): the agent has the right
  observations but can't ground their magnitude against expected.

Shipping A first means the next time you point the system at a PDF, the
numerical reasoning is solid even if the prose blind spot remains. That's
the right risk profile — it makes the system robustly half-good before
making it potentially-fully-good.
