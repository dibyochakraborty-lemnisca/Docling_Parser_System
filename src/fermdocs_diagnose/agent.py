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
from typing import Any, Literal, Protocol

from fermdocs_characterize.agent_context import (
    AgentContext,
    build_agent_context,
    serialize_for_agent,
)
from fermdocs_characterize.schema import CharacterizationOutput
from fermdocs_characterize.specs import DictSpecsProvider, SpecsProvider
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
from fermdocs_diagnose.validators import validate_diagnosis

_log = logging.getLogger(__name__)

DEFAULT_MAX_STEPS = 6
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
    ) -> DiagnosisOutput:
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

        prefix = serialize_for_agent(ctx, output)
        messages: list[dict[str, str]] = [
            {"role": "user", "content": _initial_user_prompt(prefix)},
        ]

        emit_payload: dict[str, Any] | None = None
        for step in range(self._max_steps):
            try:
                response = self._client.call(_SYSTEM_PROMPT, messages)
            except Exception as exc:
                _log.warning(
                    "diagnosis: LLM call failed at step %d: %s",
                    step,
                    exc.__class__.__name__,
                )
                return self._error_output(
                    diagnosis_id,
                    generation_timestamp,
                    output,
                    error=f"llm_call_failed:{exc.__class__.__name__}",
                )

            action = response.get("action")
            if action == "emit":
                emit_payload = response
                break
            if action == "tool_call":
                tool_result = _dispatch_tool(response, output=output, specs=specs)
                messages.append(
                    {"role": "assistant", "content": json.dumps(response)}
                )
                messages.append(
                    {"role": "user", "content": json.dumps(tool_result)}
                )
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
        failures.append(
            FailureClaim(
                claim_id=f"D-F-{i+1:04d}",
                summary=str(raw.get("summary", "")),
                cited_finding_ids=list(raw.get("cited_finding_ids") or []),
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
