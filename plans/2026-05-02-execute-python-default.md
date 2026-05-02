# Plan — execute_python-default diagnosis (Wave 2)

**Date:** 2026-05-02
**Author:** Pushkar + Claude
**Status:** Draft v1
**Predecessors:** `2026-05-02-diagnosis-agent.md` (Wave 1 shipped on main)

## 0. Why this plan exists

Wave 1 shipped a diagnosis agent that reads pre-computed findings from
`range_violation` detectors and writes a structured `DiagnosisOutput`. The
IndPenSim end-to-end run exposed the deeper problem: **deterministic detectors
can't distinguish "violated a setpoint" from "is a problem"**. They produced
9320 findings on a clean batch (specs were setpoints, not bounds) and only 1
finding on a real overfeed event. The aggregation cap mitigated symptoms; it
didn't fix the frame.

The reference repo (`fermentation-debate-langgraph`) takes the opposite
approach: specialists fetch raw data via tools and reason over it. There are no
pre-computed "violations" — there's a tier of analysis reports + raw timecourse
access + an `execute_python` sandbox.

This plan adopts that frame *selectively*. We keep `golden_schema` for what
it's good at (variable identification, units, canonical names) and stop
asking it to define what counts as a problem. Diagnosis becomes
**execute_python-default**: the agent's spine is dynamic pandas analysis on
the raw trajectory; deterministic findings are one available artifact, not the
forced input.

## 1. Scope split — what schema owns vs what the LLM owns

**Schema continues to own:**
- Variable identification (which CSV column is biomass vs glucose)
- Unit canonicalization (g/L vs mg/mL)
- Canonical naming + nominal/std_dev for known process variables
- The mapping pipeline (Anthropic/Gemini identity → schema → observations)

**Schema stops owning:**
- What counts as a problem (range_violation no longer drives diagnosis input)
- The shape of the agent's reasoning (no more "rank the findings list")

**LLM (via execute_python) starts owning:**
- Anomaly detection on the trajectory (is this curve weird?)
- Cross-batch comparison (does RUN-0001 differ from RUN-0002?)
- Derivation (compute µ, qs, Yx/s on the fly)
- Open-question generation grounded in what it actually computed

`range_violation` detectors keep running upstream and write findings to the
bundle, but as **one artifact among several**, exposed via a `get_findings()`
tool the agent can choose to call.

## 2. Bundle directory — the canonical handoff

Replace in-memory chaining with file-based artifacts under
`out/bundle_<run_id>_<utc_iso>_<short_uuid>/`:

```
out/bundle_RUN-0001_20260502T143012Z_a1b2c3/
  meta.json                       # see §2.1 — the cross-stage handshake
  characterization/
    observations.parquet          # canonical timecourse (existing pipeline output)
    findings.json                 # deterministic range_violation findings (existing)
    specs.json                    # the resolved spec set used (provenance)
  diagnosis/
    diagnosis.json                # DiagnosisOutput (existing schema)
    summary.md                    # human-readable sidecar
  audit/
    diagnosis_trace.json          # per-call ReAct trace (NEVER read at runtime)
    python_calls/                 # spillover for large stdouts
      py_000001.json
      ...
    python_trace.jsonl            # one line per execute_python call (or pointer)
```

**Invariants:**
- `audit/` is write-only at runtime. Any node reading from `audit/` is a bug.
  Enforced by CI guard from Stage 1 (see §9). Not honor-system.
- `meta.json` is the only cross-stage handshake. Schema version mismatches fail loudly.
- `characterization/` is independently re-runnable; `diagnosis/` reads only `characterization/` + raw input.

**Bundle naming + collision handling:**
- Format: `bundle_<run_id>_<utc_iso_compact>_<short_uuid>`. Multi-run bundles use `bundle_multi_<utc_iso>_<short_uuid>` and list run_ids inside `meta.json`.
- Created via `mkdir(exist_ok=False)` on a temp suffix `out/.bundle_<...>.tmp/`, populated, then renamed atomically to `out/bundle_<...>/`. `meta.json` is the **last** file written before rename — its presence is the bundle's "ready" signal.
- BundleReader rejects any path that doesn't have a `meta.json`. Half-written bundles are invisible.
- Disk-full mid-write: temp dir remains as `.bundle_*.tmp/`, never gets a final name, never read. Manual cleanup (or a `gc` CLI) sweeps `.bundle_*.tmp` older than N hours.

## 2.1 `meta.json` schema

```python
class BundleMeta(BaseModel):
    bundle_schema_version: Literal["1.0"]        # exact match required, else fail
    golden_schema_version: str                   # e.g. "2.0"; warn on minor diff, fail on major
    pipeline_version: str                        # git short sha or pkg version
    created_at: datetime                         # UTC, ISO 8601
    bundle_id: str                               # equals dirname suffix
    run_ids: list[str]                           # at least one
    model_labels: dict[str, str]                 # {"characterization": "anthropic/claude-...", "diagnosis": "gemini/gemini-3.1-pro-preview"}
    flags: dict[str, bool]                       # e.g. {"budget_exhausted": false}
```

Mismatch policy on read:
- `bundle_schema_version` not equal → raise `BundleSchemaMismatch`. Hard fail.
- `golden_schema_version` major differs → raise `GoldenSchemaMajorMismatch`. Hard fail.
- `golden_schema_version` minor differs → log warning, proceed. Findings may carry stale spec references.

## 3. The diagnosis agent's new tool surface

Tools the agent sees, in order of expected use:

| Tool | Purpose | Cost |
|---|---|---|
| `execute_python(code, timeout=120)` | Sandboxed pandas/numpy/scipy on the trajectory | High |
| `get_timecourse(run_id, columns?, time_range_h?)` | Fetch trajectory rows (capped 100, NaN→null) | Low |
| `list_runs()` | Enumerate run_ids in the bundle | Low |
| `get_findings(run_id?, variable?, severity?, tier?)` | Filtered access to deterministic findings | Low |
| `get_specs(variable)` | The resolved spec (nominal, std_dev, source) | Low |
| `get_meta()` | bundle metadata (organism, product, schema_version) | Low |
| `submit_diagnosis(DiagnosisOutput)` | Synthetic terminator. Validates + writes diagnosis/diagnosis.json. | n/a |

**No compute tools** beyond execute_python. No statistical-test tools, no ML
tools, no anomaly-detector tools. The reference repo validated this — pure
data-fetch + sandboxed python is enough. Comparison/correlation/changepoint
logic stays in the LLM (or in code the LLM writes inside execute_python).

**Implementation split:**
- `BundleReader` (data layer): loads observations.parquet **once** at init, holds the DataFrame in memory. All `get_timecourse` / `list_runs` / `get_findings` calls slice from memory. No re-reads. (P3)
- `BundleWriter` (data layer): atomic temp+rename, schema-version enforcement, used by characterization stage and end-of-diagnosis.
- Tool surface is closure-curried over `bundle_dir` via `make_diagnosis_tools(bundle_reader, bundle_writer)` (reference repo pattern in `specialist_tool_bundle.py`) so the LLM sees clean signatures without bundle path arguments.

**Tool docstring contract.** Every tool ships with an agent-facing docstring containing:
1. *When to call it* (one sentence linking it to a reasoning need)
2. *Cost hint* (Low / Medium / High, matching the table)
3. *Output shape* (concrete keys + types, not "a dict")
4. *Failure modes* the agent should expect (e.g. "returns `{error: 'unknown_run'}` if run_id missing")

The catalog injected into the system prompt is rendered from these docstrings (single source of truth, no copy-paste prompt drift).

## 3.1 Agent state machine

The diagnosis agent has three states:

```
RUNNING ──submit_diagnosis()──▶ SUBMITTED ──finalize()──▶ DONE
   │                                  │
   └──budget exhausted────────────────┘
                                      │
   any tool call after SUBMITTED ─▶ returns {error: "already_finalized"}
```

- `submit_diagnosis` is **idempotent**: calling it twice with identical payload is a no-op; calling it twice with different payloads returns `{error: "diagnosis_already_submitted"}` and does not overwrite.
- Any tool call (including `execute_python`) after `SUBMITTED` returns the same error and does not execute.
- Budget exhaustion (20 tool calls or 420s wall-clock) auto-transitions `RUNNING → SUBMITTED` with whatever DiagnosisOutput is partial in the trace, plus `meta.flags.budget_exhausted = true`. If no partial diagnosis exists, emit an error DiagnosisOutput.

## 4. execute_python sandbox

Lift the structure from `/tmp/fermentation-debate-langgraph/src/tools/execute_python.py`:

- Subprocess via `asyncio.create_subprocess_exec(sys.executable, "-c", code)`
- 120s default timeout, configurable up to 420s wall-clock for the whole agent
- 50KB stdout cap with truncation marker (never silent)
- 1MB cap on stdout/stderr captured into the trace record itself; anything larger spills to `audit/python_calls/py_NNNNNN.json` with a pointer in the jsonl. (Spill threshold also applies to *records* >100KB as in the reference repo, but the per-stream cap protects against pandas dumping a giant frame.)
- Available libs: pandas, numpy, scipy, scikit-learn, plotly (skip pdfplumber/openpyxl/bioservices for diagnosis)
- **`cwd` pinned to project root** so `from fermdocs...` / `from fermdocs_diagnose...` imports resolve. Sandboxed code can call into the codebase (reuse parsing helpers, schema lookups, etc.). Matches the reference repo's pattern.
- `PYTHON_TRACE_FILE` env var → per-call JSON trace appended to `audit/python_trace.jsonl`
- **Full fidelity in traces, no truncation in audit/** — only the agent-facing return value is capped at 50KB

Pandas, numpy, etc. are pip-installed in the project venv and available via
`sys.executable`. The agent loads the bundle's parquet via
`pd.read_parquet(<path from get_meta()>)`.

**Trust posture.** With cwd=project root, sandboxed code has read access to repo
files (source, schema YAMLs, `.env`). This is acceptable because:
- The code being executed is generated by *our* LLM under *our* prompt — not user input
- Inputs to the LLM are scientific timecourse data, not adversarial text
- The reference repo runs the same posture in production
If we ever expose execute_python to untrusted user prompts (e.g., a public API),
this decision needs to be revisited (see §12 open question 1).

**OOM containment.** Pandas can blow past memory on large frames. Subprocess gets
`resource.setrlimit(RLIMIT_AS, ...)` set to a configurable cap (default 2GB). Exceeding
the cap kills the subprocess with the OS's standard signal; trace records the kill,
agent sees a clear error.

## 4.1 Shared TraceWriter

Both `execute_python` per-call traces and the ReAct loop's per-LLM-call traces (§6)
share the same write semantics: append-only jsonl, spill records >100KB to
sibling files with pointers, never truncate in audit/. Implement once as
`fermdocs_diagnose.audit.TraceWriter` with two callers:
- `execute_python` writes `kind: "python_call"` records to `audit/python_trace.jsonl`
- ReAct loop writes `kind: "tool_call" | "llm_response" | "tool_result"` records to `audit/diagnosis_trace.json`

One implementation, two callers. No duplicate spill logic.

## 5. Budget + enforcement

Per-diagnosis budget (matches reference repo's specialist_react.py):
- **Max tool calls: 20**
- **Wall-clock: 420s**
- **execute_python timeout: 120s default per call**

Hard tool-use enforcement (the retry-once-if-zero pattern):
- If the agent emits zero tool calls in its first response → retry once with a
  stern addendum: *"You MUST fetch evidence via tools before claiming numbers.
  Start with `list_runs()` and `get_meta()`."*
- If still zero on retry → emit error DiagnosisOutput with `analysis` describing
  the enforcement failure. Never fabricate.

Termination:
- Agent calls a synthetic `submit_diagnosis(DiagnosisOutput)` tool to finish
- Or budget exhaustion → auto-finalize with whatever's in the trace + a
  "budget_exhausted" flag in `meta.json`

## 6. Per-call trajectory persistence

Every LLM call (not just execute_python) writes a JSON record to
`audit/diagnosis_trace.json`:

```json
{
  "seq": 3,
  "ts": 1746201234.567,
  "kind": "tool_call" | "llm_response" | "tool_result",
  "model": "gemini-3.1-pro-preview",
  "input_tokens": 1234,
  "output_tokens": 567,
  "tool_name": "execute_python",
  "args": {...},
  "result_bytes": 1024,
  "spilled_to": null
}
```

Aggregated atomically at end-of-run from per-call writes. **Never read by any
runtime node.** This is the discipline that makes the `audit/` invariant
load-bearing — if we violate it once, context poisoning is guaranteed.

## 7. Tier labels on findings (preserved from earlier discussion)

Add `tier: Literal['A', 'B', 'C']` to the `Finding` model in characterization:
- **Tier A** — direct measurement violations (range_violation against measured nominal+std_dev)
- **Tier B** — derived (rates, yields, ratios computed from raw measurements)
- **Tier C** — modeled / process-priors-derived (back-calculated from priors)

Today everything is Tier A. The field exists so:
1. The diagnosis agent can weight evidence (`get_findings(severity='critical', tier='A')`)
2. Future B/C detectors plug in without schema migration
3. Hypothesis-stage agents inherit the trust signal

This is a 1-field schema change + a default-A backfill. Cheap.

## 8. What doesn't change in this wave

- `golden_schema.yaml` — unchanged
- The mapping pipeline (`fermdocs/pipeline.py`) — unchanged
- `range_violation` detectors — unchanged, just gain a `tier='A'` label
- `RunIdResolver` — unchanged
- `DiagnosisOutput` Pydantic schema — unchanged (the *production* of it changes, not the shape)
- Storage layer — unchanged; bundle is a parallel file-system artifact, DB still gets observations

## 9. Migration path

**Stage 1 — bundle infrastructure + audit guard (no behavior change):**
- Add `BundleWriter` (atomic temp+rename, meta.json schema enforcement) that captures existing characterization output to `out/bundle_<...>/characterization/`
- Add `BundleReader` (cached parquet load, schema-version check)
- Add `tier='A'` field to Finding, default-backfill
- Diagnosis CLI accepts `--bundle <path>` and reads from there if present, falls back to in-memory
- **Add CI guard** (grep): any source file under `src/fermdocs_diagnose/` (excluding `audit/` writers and tests) that contains a string match for `audit/` or `python_trace` paths fails the build. Move from Stage 4 to Stage 1 because the invariant is load-bearing from day one.
- Tests: existing diagnosis evals keep passing reading from bundle (regression test)

**Stage 2 — execute_python tool + fetch tools (additive):**
- Implement `TraceWriter` (shared between execute_python and ReAct trace)
- Lift `execute_python.py` into `src/fermdocs_diagnose/tools/execute_python.py` with the §4 divergences (cwd=bundle_dir, RLIMIT_AS, 1MB stream cap)
- Add bundle-fetch tools (`get_timecourse`, `list_runs`, `get_findings`, `get_specs`, `get_meta`) wrapping `BundleReader`
- Add `submit_diagnosis` synthetic terminator with the §3.1 state machine
- Wire tools into the ReAct loop. Agent is *allowed* but not *forced* to use them — system prompt unchanged from Wave 1
- Add 5-10 eval cases that require execute_python (cross-batch comparison, derivation, anomaly-shape that range_violation misses)
- Measure: how often does the agent reach for execute_python organically? This data decides Stage 3a.

**Stage 3a — prompt flip (gated on Stage 2 measurement):**
- *Only ship if Stage 2 evals show the agent under-uses execute_python.* If it already reaches for it organically on the new eval cases, skip 3a entirely.
- Update diagnosis system prompt: lead with execute_python, demote findings to "available evidence, may be incomplete"
- Re-run all evals. Gate: ≥80% recall on golden findings, no regression on Wave 1 eval set.

**Stage 3b — enforcement + trajectory persistence:**
- Add hard tool-use enforcement (retry-once-if-zero, error-output-on-second-zero)
- Wire ReAct trace persistence via the shared TraceWriter from Stage 2
- Add `meta.flags.budget_exhausted` auto-flagging
- This is the load-bearing reliability commit — gate on eval delta

**Stage 4 — cleanup:**
- Remove the in-memory diagnosis path (bundle becomes mandatory)
- Document the audit/never-read invariant in CONTRIBUTING.md
- Strengthen the Stage 1 CI guard if needed (ast-grep instead of plain grep, etc.)

## 10. Evals + replay

The reference repo has zero evals on its extraction layer; we won't repeat
that mistake. Evals split into:

**Capability evals** (does it produce the right diagnosis):
- Existing IndPenSim B1/B2 cases
- Yeast adversarial fixture
- ≥5 new cases requiring execute_python (cross-batch, derived rates, anomaly shapes that range_violation misses)

**Reliability evals** (does it produce the same diagnosis):
- Re-run each eval N=5 times, measure DiagnosisOutput diff (set-similarity on findings/trends, presence of key open_questions)
- Target: <20% variance on critical-severity findings, >80% recall on golden findings

**Replay (new):**
- `python_trace.jsonl` contains every code+stdout. A `replay <bundle>` CLI re-executes the python calls deterministically and diffs results
- Catches: model drift, sandbox dependency drift, dataset drift

**Determinism the eval suite must protect:**
- Same input + same seed + same model snapshot → DiagnosisOutput is *similar*, not identical (acknowledged tradeoff). Findings set must overlap ≥80%.

## 11. What this buys upcoming hypothesis agents

The hypothesis stage (deferred, but designed-for here):
- Reads from `out/bundle_<id>/diagnosis/` via tools, not in-memory state
- Sees tier labels on findings → can downweight Tier C in early debate rounds
- Inherits the same execute_python pattern → Dr. Strain / Dr. Data don't need new infra
- The audit/never-read invariant means hypothesis agents can't accidentally
  read diagnosis trajectories and poison their context with stale reasoning

Concretely: a `HypothesisBoard` (future) sits at `out/bundle_<id>/hypothesis/board.json`,
specialists mutate it via typed BoardActions, the orchestrator reads only the
board (not specialist trajectories). Same pattern as diagnosis at one level up.

## 12. Open questions to resolve before Stage 3

1. **Sandbox isolation level.** Subprocess is the reference repo's choice. Do
   we need stronger (Docker, firejail) for production? Decision: subprocess for
   now, revisit if we expose to untrusted inputs.
2. **Cache strategy for execute_python.** Same code + same trajectory → same
   stdout. Worth caching by hash? Decision: defer; measure cost first.
3. **Failure mode when execute_python repeatedly fails.** Currently: errors
   surface to agent, agent retries. Cap at 3 consecutive failures → emit error
   diagnosis. (Add to Stage 3.)
4. **Determinism mode for evals.** Should evals pin a seed + temperature=0 +
   model snapshot? Decision: yes for capability evals, no for reliability evals.

## 13. Non-goals (explicit)

- HypothesisBoard / debate panel — Wave 3
- Cross-debate SQLite memory — Wave 3+
- Replacing `golden_schema` with dynamic schema inference — never
- Adding ML/statistical tools beyond execute_python — never (reference repo
  validated that pure data-fetch + sandboxed python is enough)
- execute_python in characterization — out of scope; characterization stays
  deterministic (it's the cacheable fast path)

## 14. Stage gating

Each stage lands as its own PR with its own eval delta:

| Stage | PR title | Gate |
|---|---|---|
| 1 | `feat(bundle): file-based stage handoff + tier labels + audit CI guard` | Existing evals pass; CI guard rejects audit/-reads |
| 2 | `feat(diagnose): execute_python tool + bundle fetch tools + TraceWriter` | New execute_python evals pass; tool-use rate measured |
| 3a | `feat(diagnose): flip spine to execute_python-default (prompt)` | ≥80% recall on golden findings; **conditional — skip if Stage 2 measurement shows organic adoption** |
| 3b | `feat(diagnose): hard tool-use enforcement + trajectory persistence` | Reliability eval ≥80% set-overlap on critical findings; latency <420s p95 |
| 4 | `chore(diagnose): drop in-memory path + harden CI guard` | All evals green |

Stages 1-2 are infrastructure that ship without behavior change so we can roll
back cheaply. Stage 3a may be unnecessary depending on Stage 2 data. Stage 3b
is the load-bearing reliability commit.

## 15. Test gaps + failure modes (from eng review)

Concrete tests required by the plan; implementation must include these:

**Bundle layer (Stage 1):**
- Atomic write: kill process between `populate(temp)` and `rename(temp, final)`. Assert no readable bundle exists.
- Schema-version mismatch: bundle with `bundle_schema_version="0.9"` raises `BundleSchemaMismatch` on read.
- Golden-schema major mismatch fails; minor mismatch warns and proceeds.
- Tier field default-backfill: load a pre-tier `findings.json`, confirm all entries get `tier='A'`.
- Disk-full simulation: temp dir survives, no final bundle, no partial read.
- CI guard test: a deliberately bad import in a test fixture (a runtime module reading from `audit/`) fails the guard; whitelisted writer modules pass.

**execute_python (Stage 2):**
- Success path: simple `print("hi")` returns "hi\n".
- Timeout: 130s sleep with 120s timeout → subprocess killed, agent sees timeout error string, trace records `timed_out: true`.
- Non-zero exit: `raise SystemExit(2)` → agent sees stderr, returncode in trace.
- Stdout overflow: print 100KB → agent sees 50KB + truncation marker, full 100KB in spillover file.
- Trace spillover: code+stdout combined >100KB → record in jsonl is a pointer, full record in `python_calls/`.
- OOM containment: allocate 4GB on RLIMIT_AS=2GB → killed, trace records the kill.
- Codebase import: `from fermdocs_diagnose.schema import DiagnosisOutput` succeeds (cwd=project root).

**Agent state machine (Stage 2 + 3b):**
- `submit_diagnosis` called twice with same payload: idempotent, no error.
- `submit_diagnosis` called twice with different payloads: second call returns `{error: "diagnosis_already_submitted"}`, file unchanged.
- Tool call after submit: returns `{error: "already_finalized"}`, no execution.
- Hard-tool-use (Stage 3b): mock LLM that emits zero tools twice → final DiagnosisOutput is the error template, never fabricated.
- Budget exhaustion: mock LLM that loops 21 tool calls → finalize at 20, `meta.flags.budget_exhausted=true`.

**Reliability (Stage 3b):**
- N=5 reruns on each capability eval. Critical-finding set-overlap ≥80%.
- Replay smoke: run a known bundle, capture `python_trace.jsonl`, replay deterministically, diff stdouts. Allowed diff: timestamps. Disallowed: numeric values, error messages.

**Failure mode summary:**

| Codepath | Failure | Test location | Error path | User sees |
|---|---|---|---|---|
| BundleWriter | Disk full mid-rename | test_bundle_atomic | temp dir survives, no final | clear error from CLI |
| execute_python | Subprocess hang | test_exec_timeout | subprocess kill at 120s | clear "timed out" |
| execute_python | OOM (RLIMIT_AS) | test_exec_oom | subprocess killed by signal | clear "memory limit" |
| BundleReader | meta.json missing | test_bundle_meta_missing | raise on read | clear error |
| Agent enforcement | Two zero-tool responses | test_zero_tools | error DiagnosisOutput | clear "enforcement_failed" |
| Audit invariant | Module reads from audit/ | CI grep guard | build fails | dev sees CI error |
