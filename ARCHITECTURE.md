# fermdocs Architecture

This document describes the current architecture of `fermdocs`. It is the
repo-root design reference. Historical notes in `plans/` are useful for
intent, but the current contracts are the code, schemas, tests, and this
document.

## System Shape

`fermdocs` is a staged analysis system for fermentation reports:

```text
source files
  -> ingest
  -> bundle
  -> characterize
  -> diagnose
  -> hypothesize
  -> local API/web app
```

The stages are intentionally separated by typed JSON artifacts. This keeps
each agent from needing direct access to every upstream implementation detail
and makes the output inspectable after every stage.

## Core Packages

```text
src/fermdocs
  Parsing, header mapping, unit normalization, storage, dossier creation,
  process identity, PDF segmentation, narrative extraction, and bundle I/O.

src/fermdocs_characterize
  Deterministic trajectory construction, metric catalog execution, anomaly
  detection, narrative observation materialization, optional LLM trajectory
  analysis, and validation.

src/fermdocs_diagnose
  Observational diagnosis agent. It uses a bounded ReAct loop over bundle
  tools and emits failures, trends, analyses, and open questions.

src/fermdocs_hypothesis
  Multi-agent causal hypothesis stage. It contains the state machine,
  specialist agents, typed projector views, synthesis/critique/judgment,
  HITL resume, follow-up, chart specs, and Plotly rendering.

apps/api
  Local FastAPI wrapper around the full pipeline.

apps/web
  Next.js UI for upload, run progress, websocket events, hypotheses, charts,
  follow-up, and print-to-PDF.
```

## Artifact Flow

### 1. Raw Inputs

Supported user-facing inputs:

```text
.csv
.xlsx
.pdf
.zip bundle
```

CSV/XLSX/PDF inputs run the full pipeline. Zip inputs are treated as existing
bundles and skip directly to hypothesis.

### 2. Dossier

The ingest stage emits a dossier JSON. It is the first downstream artifact and
contains observations, provenance, residual material, process identity, and
document mapping information.

Important ingest principles:

- The LLM maps headers and gives unit hints; it should not invent numeric
  values.
- Source material that does not map cleanly is preserved as residual data.
- Provenance is part of the data model, not a debug feature.
- Process identity separates observed identity from registered identity.
- Operator manifests can override LLM identity extraction.

### 3. Bundle

The bundle is the central artifact boundary.

Typical structure:

```text
bundle_<id>/
|-- meta.json
|-- dossier.json
|-- characterization/
|   |-- characterization.json
|   |-- observations.csv
|   `-- narrative_observations.json
|-- diagnosis/
|   `-- diagnosis.json
|-- audit/
`-- user_question.json
```

`meta.json` is written last and is the readiness signal. Readers reject
missing or incompatible metadata. This prevents later stages from consuming
half-written bundles.

Runtime code should not use `audit/` as evidence. Audit files are for traces,
debugging, and post-run inspection.

### 4. Characterization Output

Characterization converts dossier observations into:

```text
trajectories
findings
narrative observations
timeline events
expected-vs-observed summaries
facts graph
open questions
kinetic estimates
```

The stage is deliberately deterministic-first. Metric catalog execution,
toolkit functions, robust statistics, metadata anomaly detectors, product
KPIs, and physicality validators run before optional LLM analysis.

### 5. Diagnosis Output

Diagnosis is observational. It should answer:

```text
What failed?
What trended?
What was observed?
What is uncertain?
```

It should not speculate causal mechanisms beyond what the evidence supports.
The causal debate belongs to hypothesis.

Diagnosis claims cite upstream evidence by IDs: finding IDs, trajectory IDs,
or narrative observation IDs. Bundle-backed diagnosis uses tools such as
`get_findings`, `get_trajectory`, `get_narrative_observations`, and
`execute_python`.

The ReAct loop has hard tool-use enforcement. A model that tries to emit a
diagnosis without first using tools gets retried once; if it still does not
use tools, the result goes down an error path.

### 6. Hypothesis Output

Hypothesis is causal and argumentative. It proposes mechanisms, compares
alternatives, exposes uncertainty, and produces actionable recommendations.

The stage emits:

```text
final hypotheses
rejected hypotheses
open questions
debate summary
token report
global.md event log path
Plotly chart JSON
```

`global.md` is the canonical human-readable event log. The JSON output is the
machine contract.

## Runtime State Machine

The hypothesis runner is an explicit state machine, not LangGraph today.

```text
init
  -> select_topic
  -> retry_topic
  -> contribute_facet
  -> synthesize
  -> critique
  -> judge
  -> finalize_turn
  -> exit
  -> done
```

The live agent order is:

```text
orchestrator
  -> kinetics specialist
  -> mass_transfer specialist
  -> metabolic specialist
  -> synthesizer
  -> critic
  -> judge
  -> lessons summarizer, when retry patterns need compression
```

The current specialist set is fixed at three. Routing is intentionally static.
The planned routing rule is: add deterministic specialist routing only when a
fourth specialist exists.

## User Question and HITL Model

User questions are a lens over the evidence. They should bias topic selection,
ranking, views, and summaries, but they should not suppress bottom-up analysis.

The question classifier resolves:

```text
question shape
affected runs
affected variables
```

This is written to `user_question.json` in the bundle when applicable.

There are two interactive modes:

```text
answers/resume
  The run pauses on open questions. User answers are attached and the same
  debate resumes.

follow-up
  The bundle is frozen. A new user question overwrites user_question.json and
  only the hypothesis stage runs again.
```

Diagnosis open questions intentionally do not include `re_run_from`.
Hypothesis open questions may include `re_run_from` because only hypothesis
knows whether the right next action is diagnosis or hypothesis.

## Chart Model

Hypothesis agents emit declarative `chart_specs`. Local deterministic code
turns those specs into Plotly JSON.

Supported chart families include:

```text
time_series_overlay
scatter_correlation
faceted_time_series
```

This split matters: the LLM can request the chart it needs, but rendering,
data lookup, regression details, and JSON shape stay deterministic and
testable.

## API and Web Architecture

The API is a local FastAPI service:

```text
POST /api/uploads
POST /api/runs
GET  /api/runs
GET  /api/runs/{id}
WS   /api/runs/{id}/events
POST /api/runs/{id}/answers
POST /api/runs/{id}/followup
```

Raw file runs execute:

```text
fermdocs ingest
  -> fermdocs-characterize --bundle
  -> fermdocs-diagnose --bundle
  -> in-process hypothesis run
```

Bundle zip runs execute:

```text
unzip bundle
  -> load_bundle
  -> in-process hypothesis run
```

The API publishes status messages and hypothesis events to websocket
subscribers. It stores uploads and run outputs under `FERMDOCS_API_ROOT`,
defaulting to `out/api`.

The web app is a development UI. It is not a production SaaS shell. It has no
authentication, no tenant isolation, and no durable background worker.

## Data and Schema Boundaries

The most important contracts are:

```text
fermdocs.domain.models
  Golden schema, parsed tables, observations, residual payloads, dossier,
  document maps, process identity.

fermdocs_characterize.schema
  CharacterizationOutput, Finding, Trajectory, NarrativeObservation, facts,
  expected-vs-observed, timeline, open questions.

fermdocs_diagnose.schema
  DiagnosisOutput, failures, trends, analyses, diagnosis open questions.

fermdocs_hypothesis.schema
  HypothesisInput, topics, facets, hypotheses, critiques, judgments,
  HITL records, open questions, chart specs, final output.
```

Prefer extending these schemas deliberately over passing loose dictionaries
between stages.

## Validation and Invariants

Important invariants:

- Bundle metadata gates bundle readability.
- IDs are namespaced and cross-references are validated.
- Findings must cite real observations when possible.
- LLM-judged evidence has confidence caps.
- Characterization validates physical plausibility and data coverage.
- Diagnosis claims must cite upstream evidence.
- Hypothesis validators check citation integrity and provenance downgrades.
- Production code should not import test stubs.
- Hypothesis agents should not import each other directly except through
  approved shared base abstractions.
- Runtime modules should not read from `audit/`.

Useful scripts:

```bash
python scripts/check_audit_invariant.py
python scripts/check_hypothesis_invariants.py
```

## LLM Provider Boundaries

Current provider reality:

```text
ingest mapper:        Gemini, Anthropic, fake
unit normalizer:      rule-based plus optional Gemini/Anthropic fallback
narrative extraction: Gemini/Anthropic paths exist in ingest code
characterization:     deterministic plus optional Gemini trajectory analyzer
diagnosis:            Gemini or Anthropic clients, fake/none error path
hypothesis:           Gemini live path
```

Prompt composition in the hypothesis package is cache-friendly: stable policy
and contract layers are kept separate from volatile view payloads. That does
not mean every provider path uses explicit prompt caching.

## Budgets

Default hypothesis CLI budget:

```text
max_turns = 10
max_critic_cycles_per_topic = 3
max_tool_calls_total = 80
max_total_input_tokens = 200000
```

The API uses a larger budget for interactive runs:

```text
max_turns = 20
max_critic_cycles_per_topic = 6
max_tool_calls_total = 160
max_total_input_tokens = 400000
max_open_questions = 30
```

Raising one budget usually requires raising the related budgets. For example,
more turns generally means more tool calls and more token budget.

## Testing Strategy

The project uses layered testing:

```text
unit tests
  Fast deterministic checks for parsing, schema validation, metric toolkit,
  bundle behavior, agent contracts, chart generation, and API offline paths.

integration tests
  Pipeline-level tests with fakes or fixtures.

evals
  Scripted characterization, diagnosis, and hypothesis reliability fixtures.

live_llm tests
  Opt-in tests that require API keys and cost tokens.
```

The default pytest config deselects `live_llm`.

Common commands:

```bash
pytest tests/unit -v
pytest tests -v
cd apps/web && npm run typecheck && npm run build
```

## Operational Limitations

Current known limitations:

- The API run store is not durable across backend restarts.
- No auth, tenants, quotas, secrets management, or production deployment
  story is implemented.
- Full raw ingest requires Postgres.
- PDF quality depends on Docling extraction and source layout.
- Browser print is the current PDF export path.
- Hypothesis live execution is Gemini-only today.
- Some historical docs and package docstrings still reflect earlier stages of
  the project.

## Extension Guidance

### Adding a Golden Column

Edit:

```text
src/fermdocs/schema/golden_schema.yaml
```

Add real observed header examples. Examples are usually more useful to the
mapper than long prose descriptions.

### Adding a Characterization Metric

Add a deterministic toolkit function, register it in the metric catalog, and
ensure it emits stable evidence and `metric_id` metadata. Downstream routing
and hypothesis views rely on those metric IDs.

### Adding a Specialist

Add the specialist, projector view rules, prompts, schema/test coverage, and
then introduce deterministic top-K specialist routing. Do not add routing
while the specialist set remains the current fixed three.

### Adding a Provider

Keep provider-specific SDK code behind client factories. The rest of the
stage should consume typed request/response objects or small protocols, not
raw provider payloads.

### Changing Bundle Schema

Treat bundle schema changes as compatibility work:

1. Update schema/version metadata.
2. Update writer and reader together.
3. Add tests for old/new behavior.
4. Decide whether older bundles hard-fail or warn.

## Mental Model for Future Work

When changing the system, ask which boundary owns the behavior:

```text
raw extraction or provenance?          src/fermdocs
deterministic observations/findings?   src/fermdocs_characterize
observational summary?                 src/fermdocs_diagnose
causal mechanism debate?               src/fermdocs_hypothesis
upload/run orchestration?              apps/api
human workflow and display?            apps/web
```

Keeping that separation is what lets the system evolve without every agent
and UI surface becoming coupled to every upstream implementation detail.

