"""Diagnosis ReAct agent: turns CharacterizationOutput + AgentContext into a
DiagnosisOutput.

Plan ref: plans/2026-05-02-diagnosis-agent.md §4 §5 §9.

Loop shape (max 6 steps):

    AgentContext (prefix) ─┐
                           ├──▶ LLM ──▶ {"action": "tool_call"|"emit"}
    findings/trajectories ─┘            │
                                        ├─ tool_call → run tool, append result
                                        └─ emit     → parse claims, finish

The agent never raises. Failure paths return a DiagnosisOutput with empty
claim lists and `meta.error` set, so downstream agents see an explicit
"diagnosis unavailable" rather than a missing artifact. Cross-output
validation (citation integrity, soft enforcement) is delegated to
validators.validate_diagnosis after the LLM emits.
"""

from __future__ import annotations

import json
import logging
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any, Literal, Protocol

from fermdocs.bundle import BundleReader
from fermdocs_characterize.agent_context import (
    AgentContext,
    build_agent_context,
    serialize_for_agent,
)
from fermdocs_characterize.schema import CharacterizationOutput
from fermdocs_characterize.specs import DictSpecsProvider, SpecsProvider
from fermdocs_diagnose.audit.trace_writer import TraceWriter
from fermdocs_diagnose.schema import (
    AnalysisClaim,
    ConfidenceBasis,
    DiagnosisMeta,
    DiagnosisOutput,
    FailureClaim,
    OpenQuestion,
    TrajectoryRef,
    TrendClaim,
)
from fermdocs_diagnose.tools import get_finding, get_spec, get_trajectory
from fermdocs_diagnose.tools_bundle import DiagnosisToolBundle, make_diagnosis_tools
from fermdocs_diagnose.validators import validate_diagnosis

_log = logging.getLogger(__name__)

DEFAULT_MAX_STEPS = 6
BUNDLE_MAX_STEPS = 20
"""Step budget when running with --bundle (Stage 3 spine flip).

Matches the reference repo's specialist_react.py cap of 20 tool calls.
Enough room for execute_python iteration; capped to keep cost bounded.
"""

DIAGNOSIS_SCHEMA_VERSION = "1.0"
DIAGNOSIS_AGENT_VERSION = "v1.0.0"

_SYSTEM_PROMPT = (
    "You are the diagnosis stage of a fermentation analysis pipeline.\n\n"
    "You read a deterministic characterization (findings, trajectories,"
    " process flags, identity) and produce an OBSERVATIONAL account of what"
    " happened in the run.\n\n"
    "HARD RULES:\n"
    "  1. Observational only. State what happened. Never state why.\n"
    "     Causal reasoning belongs to the next stage.\n"
    "     Forbidden words in claim summaries: because, due to, caused by,\n"
    "     resulted in, leading to.\n"
    "  2. Cite real IDs. Every claim must reference at least one finding_id\n"
    "     from the provided list (or, for trends, a (run_id, variable)\n"
    "     trajectory pair). Do not invent IDs.\n"
    "  3. Confidence ≤ 0.85 on every claim.\n"
    "  4. Under UNKNOWN_PROCESS or UNKNOWN_ORGANISM flags, you must use\n"
    "     confidence_basis='schema_only' and explicitly note in your\n"
    "     reasoning that recipe-specific priors are unavailable.\n"
    "  5. Open questions: data-gap only. Each question must reference a\n"
    "     specific finding_id, carry a why_it_matters one-liner, and be\n"
    "     answerable in under 30 seconds.\n\n"
    "TOOLS:\n"
    "  - get_finding(finding_id)\n"
    "  - get_trajectory(run_id, variable)\n"
    "  - get_spec(variable)\n\n"
    "Tool budget: 6 calls max. Use the prefix's top_findings.statistics for\n"
    "magnitude triage; only call tools when you need detail the prefix\n"
    "doesn't carry.\n\n"
    "RESPONSE FORMAT:\n"
    "Each turn, emit a JSON object with one of two shapes:\n\n"
    "  Tool call:\n"
    "    {\"action\": \"tool_call\",\n"
    "     \"tool\": \"get_finding|get_trajectory|get_spec\",\n"
    "     \"args\": {...}}\n\n"
    "  Final emit:\n"
    "    {\"action\": \"emit\",\n"
    "     \"failures\": [{summary, cited_finding_ids, affected_variables,\n"
    "                    confidence, confidence_basis, domain_tags,\n"
    "                    severity, time_window?}, ...],\n"
    "     \"trends\": [{summary, cited_finding_ids?, cited_trajectories?,\n"
    "                  affected_variables, confidence, confidence_basis,\n"
    "                  domain_tags, direction, time_window?}, ...],\n"
    "     \"analysis\": [{summary, cited_finding_ids, affected_variables,\n"
    "                    confidence, confidence_basis, domain_tags,\n"
    "                    kind}, ...],\n"
    "     \"open_questions\": [{question, why_it_matters,\n"
    "                          cited_finding_ids, answer_format_hint,\n"
    "                          domain_tags}, ...],\n"
    "     \"narrative\": \"<optional <500-word rollup>\"}\n\n"
    "Do not include claim_id or question_id; the runtime assigns those\n"
    "deterministically. Suggested domain_tags: growth, metabolism,\n"
    "environmental, data_quality, process_control, yield."
)

_BUNDLE_SYSTEM_PROMPT = (
    "You are the diagnosis stage of a fermentation analysis pipeline.\n\n"
    "Your job: find the REAL ANOMALIES in this run. Look at the trajectories\n"
    "yourself; do not anchor on the deterministic findings list (it includes\n"
    "thousands of fake \"violations\" when schema specs are setpoints, not\n"
    "trajectory bounds). Be the eyes the rest of the pipeline trusts.\n\n"
    "REASONING APPROACH (execute_python-default):\n"
    "  Lead with execute_python. Load the bundle's observations CSV (path\n"
    "  from get_meta() → observations_csv_path; columns: run_id, variable,\n"
    "  time_h, value, imputed, unit) with pd.read_csv(). Compute what you\n"
    "  actually need: derivatives, cross-batch deltas, residuals against\n"
    "  expected curves, changepoints, μ/qs/Yx/s, whatever fits the question.\n"
    "  The fetch tools (list_runs, get_findings, get_timecourse) are for\n"
    "  narrow lookups; execute_python is for thinking.\n\n"
    "  Example first call:\n"
    "    import pandas as pd\n"
    "    meta = {...}  # paste from get_meta()\n"
    "    df = pd.read_csv(meta['observations_csv_path'])\n"
    "    print(df.groupby(['run_id', 'variable'])['value'].agg(['count','mean','min','max']))\n\n"
    "WHAT COUNTS AS A FAILURE vs A TREND vs ANALYSIS:\n"
    "  - failure: a real anomaly. Something deviated from a credible\n"
    "    reference (process priors, cross-run baseline, your own derived\n"
    "    expectation, mass balance, control-loop tracking). You MUST name\n"
    "    the magnitude (e.g. '43.8 g/L vs 2.79 g/L in cohort'), the time\n"
    "    window when known, and which run(s). \"Variable X exceeded its\n"
    "    schema nominal\" by itself is NOT a failure — schema nominals are\n"
    "    setpoints. Convert to failure only when you've checked the shape\n"
    "    against process priors or the other run.\n"
    "  - trend: time-shape observation that is part of normal process\n"
    "    operation (fed-batch volume rise, biomass growth phase, etc.).\n"
    "    Use this for things that are real but not problems.\n"
    "  - analysis: meta-observations about the data itself (spec misalignment,\n"
    "    measurement gaps, cross-run pattern, data quality caveat).\n\n"
    "Default heuristic: if the trajectory shape disagrees with what a\n"
    "competent fermentation engineer would expect for this organism and\n"
    "process at this point in the run — that's a failure. If it matches\n"
    "expectation — that's a trend or no-claim.\n\n"
    "HARD RULES:\n"
    "  1. Observational, not causal. State WHAT deviated and BY HOW MUCH.\n"
    "     Do not claim WHY (\"because of feed pump fault\", \"due to oxygen\n"
    "     limitation\"). Causal reasoning belongs to the next stage.\n"
    "     Forbidden words in claim summaries: because, due to, caused by,\n"
    "     resulted in, leading to.\n"
    "  2. Cite real IDs. Every claim must reference a real finding_id or\n"
    "     a (run_id, variable) trajectory pair. Do not invent IDs.\n"
    "  3. Confidence ≤ 0.85 on every claim.\n"
    "  4. Under UNKNOWN_PROCESS or UNKNOWN_ORGANISM flags, use\n"
    "     confidence_basis='schema_only' and note that recipe-specific priors\n"
    "     are unavailable.\n"
    "  5. Open questions: data-gap only. Each must reference a finding_id,\n"
    "     carry a why_it_matters one-liner, and be answerable in <30s.\n"
    "  6. You MUST call execute_python (or another data-fetch tool) before\n"
    "     submit_diagnosis. Claims based purely on the AgentContext prefix\n"
    "     without trajectory-grounded evidence will be rejected.\n"
    "  7. If you found a real anomaly via execute_python, EMIT IT as a\n"
    "     failure. An empty failures list is correct only when, after\n"
    "     looking at the data, you genuinely see no anomalies — which is\n"
    "     rare on real industrial data and unlikely here.\n\n"
    "TOOLS (cost: H = high, L = low):\n"
    "  - execute_python(code, timeout=120) [H] — sandboxed pandas/numpy/scipy.\n"
    "    cwd is project root; `from fermdocs...` imports work. The bundle's\n"
    "    parquet path comes from get_meta() → bundle_dir.\n"
    "  - list_runs() [L] — run_ids in this bundle.\n"
    "  - get_meta() [L] — bundle metadata (organism, schema version, paths).\n"
    "  - get_findings(finding_id?, run_id?, variable?, severity?, tier?) [L]\n"
    "  - get_timecourse(run_id, variable, time_range_h?, max_points?) [L]\n"
    "  - get_specs(variable) [L] — schema spec (nominal/std_dev/unit).\n"
    "  - submit_diagnosis(payload) — terminator. Idempotent; second call\n"
    "    with a different payload is rejected.\n\n"
    "Tool budget: 20 calls / 7 min wall-clock. Spend them.\n\n"
    "RESPONSE FORMAT:\n"
    "Each turn, emit a single JSON object:\n\n"
    "  Tool call:\n"
    "    {\"action\": \"tool_call\", \"tool\": \"<name>\", \"args\": {...}}\n\n"
    "  Final emit (preferred terminator — auto-submitted):\n"
    "    {\"action\": \"emit\",\n"
    "     \"failures\": [{summary, cited_finding_ids, affected_variables,\n"
    "                    confidence, confidence_basis, domain_tags,\n"
    "                    severity, time_window?}, ...],\n"
    "     \"trends\": [{summary, cited_finding_ids?, cited_trajectories?,\n"
    "                  affected_variables, confidence, confidence_basis,\n"
    "                  domain_tags, direction, time_window?}, ...],\n"
    "     \"analysis\": [{summary, cited_finding_ids, affected_variables,\n"
    "                    confidence, confidence_basis, domain_tags, kind}, ...],\n"
    "     \"open_questions\": [{question, why_it_matters, cited_finding_ids,\n"
    "                          answer_format_hint, domain_tags}, ...],\n"
    "     \"narrative\": \"<optional <500-word rollup>\"}\n\n"
    "Do not include claim_id or question_id; the runtime assigns those.\n"
    "Suggested domain_tags: growth, metabolism, environmental,\n"
    "data_quality, process_control, yield."
)


class DiagnosisLLMClient(Protocol):
    """Minimal protocol so tests can supply a scripted client.

    `messages` is the running conversation; the client returns a single JSON
    object (the next assistant turn). The agent appends the response to
    messages itself.
    """

    def call(self, system: str, messages: list[dict[str, str]]) -> dict[str, Any]: ...


class DiagnosisAgent:
    """ReAct loop with bounded steps and a single retry on parse failure."""

    def __init__(
        self,
        client: DiagnosisLLMClient | None = None,
        *,
        max_steps: int = DEFAULT_MAX_STEPS,
        model: str = "claude-opus-4-7",
        provider: Literal["anthropic", "gemini"] = "anthropic",
    ) -> None:
        self._client = client
        self._max_steps = max_steps
        self._model = model
        self._provider = provider

    def diagnose(
        self,
        dossier: dict[str, Any],
        output: CharacterizationOutput,
        *,
        specs_provider: SpecsProvider | None = None,
        diagnosis_id: uuid.UUID | None = None,
        generation_timestamp: datetime | None = None,
        bundle: BundleReader | None = None,
    ) -> DiagnosisOutput:
        """Run the diagnosis ReAct loop.

        Args:
            dossier: ingestion dossier (used for specs + AgentContext).
            output: upstream CharacterizationOutput.
            specs_provider: override; defaults to DictSpecsProvider.from_dossier.
            diagnosis_id / generation_timestamp: pinned for ID-stable replays.
            bundle: when set, the agent gets the Stage 2 tool surface
                (execute_python + bundle fetches). Trace is persisted under
                <bundle>/audit/. Without `bundle`, the legacy 3-tool surface
                is used; behavior matches Wave 1.
        """
        diagnosis_id = diagnosis_id or uuid.uuid4()
        generation_timestamp = generation_timestamp or datetime.utcnow()
        specs = specs_provider or DictSpecsProvider.from_dossier(dossier)
        ctx = build_agent_context(dossier, output)

        if self._client is None:
            return self._error_output(
                diagnosis_id,
                generation_timestamp,
                output,
                error="no_llm_client_configured",
            )

        # Stage 3: when running with a bundle, flip the spine.
        # - Use the execute_python-default system prompt
        # - Bump the step budget to BUNDLE_MAX_STEPS (20)
        # - Persist every LLM call + tool result to audit/diagnosis_trace.jsonl
        # - Enforce hard tool-use (retry-once-if-zero on the first response)
        tool_bundle: DiagnosisToolBundle | None = None
        trace_writer: TraceWriter | None = None
        system_prompt = _SYSTEM_PROMPT
        max_steps = self._max_steps
        if bundle is not None:
            trace_writer = TraceWriter(bundle.dir / "audit" / "diagnosis_trace.jsonl")
            tool_bundle = make_diagnosis_tools(
                bundle,
                output,
                specs=specs,
                dossier=dossier,
                trace_writer=trace_writer,
            )
            system_prompt = _BUNDLE_SYSTEM_PROMPT
            # Only ramp the budget when the caller didn't override max_steps.
            if self._max_steps == DEFAULT_MAX_STEPS:
                max_steps = BUNDLE_MAX_STEPS

        prefix = serialize_for_agent(ctx, output)
        messages: list[dict[str, str]] = [
            {"role": "user", "content": _initial_user_prompt(prefix)},
        ]

        emit_payload: dict[str, Any] | None = None
        budget_exhausted = False
        enforcement_retried = False
        for step in range(max_steps):
            try:
                response = self._client.call(system_prompt, messages)
            except Exception as exc:
                _log.warning(
                    "diagnosis: LLM call failed at step %d: %s: %s",
                    step,
                    exc.__class__.__name__,
                    str(exc)[:300],
                )
                _record_trace(
                    trace_writer,
                    kind="llm_error",
                    step=step,
                    model=self._model,
                    provider=self._provider,
                    error=exc.__class__.__name__,
                    message=str(exc)[:1000],
                )
                return self._error_output(
                    diagnosis_id,
                    generation_timestamp,
                    output,
                    error=f"llm_call_failed:{exc.__class__.__name__}",
                )

            _record_trace(
                trace_writer,
                kind="llm_response",
                step=step,
                model=self._model,
                provider=self._provider,
                response=response,
            )

            action = response.get("action")

            # Hard tool-use enforcement (Stage 3b): when running with a bundle,
            # the first response must call a tool. If the agent emits without
            # calling any tools, retry once with a stern addendum. If still
            # zero on retry, return an error DiagnosisOutput — never fabricate.
            if (
                tool_bundle is not None
                and step == 0
                and action == "emit"
                and tool_bundle.state.tool_calls == 0
            ):
                _log.info("diagnosis: enforcement triggered — agent emitted with zero tools")
                _record_trace(
                    trace_writer,
                    kind="enforcement_retry",
                    step=step,
                    reason="zero_tools_on_first_response",
                )
                messages.append({"role": "assistant", "content": json.dumps(response)})
                messages.append(
                    {
                        "role": "user",
                        "content": (
                            "You emitted a diagnosis without calling any tools."
                            " You MUST fetch evidence via tools before claiming"
                            " numbers. Start with `get_meta()` and `list_runs()`,"
                            " then use `execute_python` to inspect the trajectories."
                            " Return a tool_call now."
                        ),
                    }
                )
                enforcement_retried = True
                continue

            if action == "emit":
                if (
                    tool_bundle is not None
                    and enforcement_retried
                    and tool_bundle.state.tool_calls == 0
                ):
                    _log.warning(
                        "diagnosis: enforcement retry failed — agent still emitted"
                        " without tools. Returning error."
                    )
                    _record_trace(
                        trace_writer,
                        kind="enforcement_failed",
                        step=step,
                    )
                    return self._error_output(
                        diagnosis_id,
                        generation_timestamp,
                        output,
                        error="enforcement_failed:zero_tool_calls",
                    )
                emit_payload = response
                break
            if action == "tool_call":
                if tool_bundle is not None:
                    tool_result = _dispatch_tool_bundle(response, tools=tool_bundle)
                else:
                    tool_result = _dispatch_tool(response, output=output, specs=specs)
                _record_trace(
                    trace_writer,
                    kind="tool_result",
                    step=step,
                    tool=response.get("tool"),
                    result=tool_result,
                )
                messages.append(
                    {"role": "assistant", "content": json.dumps(response)}
                )
                messages.append(
                    {"role": "user", "content": json.dumps(tool_result)}
                )
                # If the agent invoked submit_diagnosis successfully and we
                # have a payload, treat that as termination.
                if (
                    tool_bundle is not None
                    and response.get("tool") == "submit_diagnosis"
                    and tool_bundle.state.submitted
                    and tool_bundle.state.diagnosis_payload is not None
                ):
                    emit_payload = {
                        "action": "emit",
                        **tool_bundle.state.diagnosis_payload,
                    }
                    break
                continue

            _log.warning(
                "diagnosis: malformed step %d response (no valid action)", step
            )
            messages.append(
                {"role": "assistant", "content": json.dumps(response)}
            )
            messages.append(
                {
                    "role": "user",
                    "content": (
                        "Your previous response was not valid JSON with"
                        " action=tool_call|emit. Return a single valid JSON"
                        " object matching the schema."
                    ),
                }
            )

        if emit_payload is None:
            _log.warning("diagnosis: agent exhausted max_steps without emit")
            budget_exhausted = True
            _record_trace(trace_writer, kind="budget_exhausted", steps=max_steps)
            if bundle is not None:
                _flag_budget_exhausted(bundle.dir)
            return self._error_output(
                diagnosis_id,
                generation_timestamp,
                output,
                error="step_budget_exhausted",
            )

        try:
            built = _build_output(
                emit_payload,
                output=output,
                meta=self._meta(diagnosis_id, generation_timestamp, output),
            )
        except (ValueError, KeyError) as exc:
            _log.warning("diagnosis: emit payload failed schema: %s", exc)
            return self._error_output(
                diagnosis_id,
                generation_timestamp,
                output,
                error=f"emit_invalid:{exc.__class__.__name__}",
            )

        return validate_diagnosis(
            built,
            upstream=output,
            flags=ctx.flags,
        )

    def _meta(
        self,
        diagnosis_id: uuid.UUID,
        generation_timestamp: datetime,
        output: CharacterizationOutput,
        *,
        error: str | None = None,
    ) -> DiagnosisMeta:
        return DiagnosisMeta(
            schema_version=DIAGNOSIS_SCHEMA_VERSION,
            diagnosis_version=DIAGNOSIS_AGENT_VERSION,
            diagnosis_id=diagnosis_id,
            supersedes_characterization_id=output.meta.characterization_id,
            generation_timestamp=generation_timestamp,
            model=self._model,
            provider=self._provider,
            error=error,
        )

    def _error_output(
        self,
        diagnosis_id: uuid.UUID,
        generation_timestamp: datetime,
        output: CharacterizationOutput,
        *,
        error: str,
    ) -> DiagnosisOutput:
        return DiagnosisOutput(
            meta=self._meta(
                diagnosis_id, generation_timestamp, output, error=error
            )
        )


# -----------------------------------------------------------------------------
# Internal helpers
# -----------------------------------------------------------------------------


def _initial_user_prompt(prefix: str) -> str:
    return (
        "Here is the AgentContext for this run. Reason over the findings and"
        " emit your diagnosis. Use tool_calls only when the prefix lacks"
        " detail you need.\n\n"
        f"AGENT_CONTEXT:\n{prefix}"
    )


def _record_trace(trace_writer: TraceWriter | None, **fields: Any) -> None:
    """Best-effort write to the diagnosis trace. Never raises."""
    if trace_writer is None:
        return
    try:
        trace_writer.write(fields)
    except Exception:
        # Trace writing must not break the user's run.
        pass


def _flag_budget_exhausted(bundle_dir: Path) -> None:
    """Update meta.json to set flags.budget_exhausted=True after budget hit.

    Direct file mutation — meta.json is normally immutable post-finalize, but
    the budget flag is a runtime signal we want surfaced to consumers without
    rewriting the bundle. This is the one allowed mutation, scoped to flags only.
    """
    meta_path = bundle_dir / "meta.json"
    if not meta_path.exists():
        return
    try:
        data = json.loads(meta_path.read_text())
        flags = data.get("flags") or {}
        flags["budget_exhausted"] = True
        data["flags"] = flags
        meta_path.write_text(json.dumps(data, indent=2))
    except (OSError, ValueError):
        pass


def _dispatch_tool_bundle(
    response: dict[str, Any],
    *,
    tools: DiagnosisToolBundle,
) -> dict[str, Any]:
    """Dispatch a tool_call against the Stage 2 bundle tool surface.

    Tools never raise across the agent boundary; unknown tools come back as
    a hint payload so the agent can correct itself.
    """
    tool = response.get("tool")
    args = response.get("args") or {}
    handler = {
        # Stage 2 surface
        "list_runs": tools.list_runs,
        "get_meta": tools.get_meta,
        "get_findings": tools.get_findings,
        "get_specs": tools.get_specs,
        "get_timecourse": tools.get_timecourse,
        "execute_python": tools.execute_python,
        "submit_diagnosis": tools.submit_diagnosis,
        # Wave 1 aliases (agent prompt still names these). Best-effort: route
        # to bundle-aware equivalents so an unchanged system prompt keeps
        # working when --bundle is used.
        "get_finding": lambda finding_id="": tools.get_findings(finding_id=finding_id),
        "get_trajectory": lambda run_id="", variable="": tools.get_timecourse(run_id=run_id, variable=variable),
        "get_spec": lambda variable="": tools.get_specs(variable=variable),
    }.get(tool or "")
    if handler is None:
        return {
            "tool": tool,
            "error": f"unknown tool {tool!r}",
            "hint": (
                "valid tools: list_runs, get_meta, get_findings, get_specs,"
                " get_timecourse, execute_python, submit_diagnosis"
            ),
        }
    # Schema escape hatch: if the model sent payload_json (string), parse it
    # back into the dict the tool expects. Currently used by submit_diagnosis.
    if isinstance(args, dict) and "payload_json" in args and "payload" not in args:
        try:
            args = {**args, "payload": json.loads(args["payload_json"])}
            args.pop("payload_json", None)
        except (json.JSONDecodeError, TypeError) as exc:
            return {"tool": tool, "error": "bad_args", "detail": f"payload_json invalid: {exc}"}
    # Drop schema-only None placeholders so they don't override real defaults.
    if isinstance(args, dict):
        args = {k: v for k, v in args.items() if v is not None}
    try:
        result = handler(**args) if isinstance(args, dict) else handler(args)
    except TypeError as exc:
        return {"tool": tool, "error": "bad_args", "detail": str(exc)}
    return {"tool": tool, "result": result}


def _dispatch_tool(
    response: dict[str, Any],
    *,
    output: CharacterizationOutput,
    specs: SpecsProvider,
) -> dict[str, Any]:
    tool = response.get("tool")
    args = response.get("args") or {}
    if tool == "get_finding":
        result = get_finding(args.get("finding_id", ""), output=output)
    elif tool == "get_trajectory":
        result = get_trajectory(
            args.get("run_id", ""), args.get("variable", ""), output=output
        )
    elif tool == "get_spec":
        result = get_spec(args.get("variable", ""), specs=specs)
    else:
        return {
            "tool": tool,
            "error": f"unknown tool {tool!r}",
            "hint": "valid tools: get_finding, get_trajectory, get_spec",
        }
    if hasattr(result, "model_dump"):
        return {"tool": tool, "result": result.model_dump(mode="json")}
    if hasattr(result, "__dict__") and not isinstance(result, dict):
        # dataclass like Spec
        from dataclasses import asdict, is_dataclass

        if is_dataclass(result):
            return {"tool": tool, "result": asdict(result)}
    return {"tool": tool, "result": result}


def _build_output(
    payload: dict[str, Any],
    *,
    output: CharacterizationOutput,
    meta: DiagnosisMeta,
) -> DiagnosisOutput:
    """Convert the LLM emit dict into a typed DiagnosisOutput.

    Assigns claim_ids and question_ids deterministically by position. The
    LLM never sees IDs; the runtime owns them so reruns are stable across
    model output churn.
    """
    failures = []
    for i, raw in enumerate(payload.get("failures") or []):
        f_trajs = [
            TrajectoryRef(run_id=t["run_id"], variable=t["variable"])
            for t in (raw.get("cited_trajectories") or [])
            if isinstance(t, dict) and "run_id" in t and "variable" in t
        ]
        failures.append(
            FailureClaim(
                claim_id=f"D-F-{i+1:04d}",
                summary=str(raw.get("summary", "")),
                cited_finding_ids=list(raw.get("cited_finding_ids") or []),
                cited_trajectories=f_trajs,
                affected_variables=list(raw.get("affected_variables") or []),
                confidence=_clamp_conf(raw.get("confidence", 0.0)),
                confidence_basis=_basis(raw.get("confidence_basis")),
                domain_tags=list(raw.get("domain_tags") or []),
                severity=raw.get("severity", "minor"),
                time_window=raw.get("time_window"),
            )
        )

    trends = []
    for i, raw in enumerate(payload.get("trends") or []):
        trajs = [
            TrajectoryRef(run_id=t["run_id"], variable=t["variable"])
            for t in (raw.get("cited_trajectories") or [])
            if isinstance(t, dict) and "run_id" in t and "variable" in t
        ]
        trends.append(
            TrendClaim(
                claim_id=f"D-T-{i+1:04d}",
                summary=str(raw.get("summary", "")),
                cited_finding_ids=list(raw.get("cited_finding_ids") or []),
                cited_trajectories=trajs,
                affected_variables=list(raw.get("affected_variables") or []),
                confidence=_clamp_conf(raw.get("confidence", 0.0)),
                confidence_basis=_basis(raw.get("confidence_basis")),
                domain_tags=list(raw.get("domain_tags") or []),
                direction=raw.get("direction", "plateau"),
                time_window=raw.get("time_window"),
            )
        )

    analysis = []
    for i, raw in enumerate(payload.get("analysis") or []):
        analysis.append(
            AnalysisClaim(
                claim_id=f"D-A-{i+1:04d}",
                summary=str(raw.get("summary", "")),
                cited_finding_ids=list(raw.get("cited_finding_ids") or []),
                affected_variables=list(raw.get("affected_variables") or []),
                confidence=_clamp_conf(raw.get("confidence", 0.0)),
                confidence_basis=_basis(raw.get("confidence_basis")),
                domain_tags=list(raw.get("domain_tags") or []),
                kind=raw.get("kind", "phase_characterization"),
            )
        )

    questions = []
    for i, raw in enumerate(payload.get("open_questions") or []):
        questions.append(
            OpenQuestion(
                question_id=f"D-Q-{i+1:04d}",
                question=str(raw.get("question", "")),
                why_it_matters=str(raw.get("why_it_matters", "")),
                cited_finding_ids=list(raw.get("cited_finding_ids") or []),
                answer_format_hint=raw.get("answer_format_hint", "free_text"),
                domain_tags=list(raw.get("domain_tags") or []),
            )
        )

    return DiagnosisOutput(
        meta=meta,
        failures=failures,
        trends=trends,
        analysis=analysis,
        open_questions=questions,
        narrative=payload.get("narrative"),
    )


def _clamp_conf(raw: Any) -> float:
    try:
        c = float(raw)
    except (TypeError, ValueError):
        return 0.0
    return max(0.0, min(0.85, c))


def _basis(raw: Any) -> ConfidenceBasis:
    if isinstance(raw, ConfidenceBasis):
        return raw
    try:
        return ConfidenceBasis(raw)
    except (ValueError, TypeError):
        return ConfidenceBasis.SCHEMA_ONLY


__all__ = [
    "DiagnosisAgent",
    "DiagnosisLLMClient",
    "DEFAULT_MAX_STEPS",
    "DIAGNOSIS_SCHEMA_VERSION",
    "DIAGNOSIS_AGENT_VERSION",
]
