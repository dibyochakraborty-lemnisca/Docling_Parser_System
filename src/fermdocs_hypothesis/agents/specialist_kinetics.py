"""Kinetics specialist agent — first specialist persona.

Plan ref: plans/2026-05-03-hypothesis-debate-v0.md §2 (specialist roles), §5
(specialist tool surface).

Domain focus: μ, q_s, q_p, yield coefficients, growth-phase transitions,
substrate inhibition. Does NOT comment on DO/kLa/agitation (mass_transfer)
or pathway-level metabolism (metabolic). View-side filtering already
narrows context to kinetic-relevant findings/narratives/priors.

Loop shape (ReAct, max 6 tool calls then forced contribute_facet):
  - Each turn the agent emits {action: tool_call|contribute_facet}
  - tool_call routes to HypothesisToolBundle for read-side data
  - contribute_facet is terminal — emits the FacetFull and exits
  - If the agent hits max_tool_calls without contributing, the runner
    forces a final pass with `must_contribute=True` and the agent must emit

This shape mirrors diagnose's tool-loop pattern (proven on IndPenSim +
carotenoid).

Persona spec format (`SPECIALIST_SPEC`) is the documented extension point
for adding specialists in Stage 4.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any

from fermdocs_diagnose.schema import ConfidenceBasis, TrajectoryRef
from fermdocs_hypothesis.llm_clients import GeminiHypothesisClient
from fermdocs_hypothesis.prompts import ToolHint, build_prompt
from fermdocs_hypothesis.schema import (
    FacetFull,
    SpecialistRole,
    SpecialistView,
)
from fermdocs_hypothesis.tools_bundle.factory import (
    GET_NARRATIVE_OBSERVATIONS,
    GET_PRIORS,
    HypothesisToolBundle,
    QUERY_BUNDLE,
    truncate_result,
)


SPECIALIST_SPEC: dict[str, Any] = {
    "role": "kinetics",
    "system_identity": """\
You are the Kinetics specialist in a fermentation-hypothesis debate.

Your domain is growth and consumption rates: μ (specific growth rate),
q_s (specific substrate uptake), q_p (specific product formation),
yield coefficients (Y_x/s, Y_p/s), growth-phase transitions
(lag → exponential → stationary → decline), substrate-limitation kinetics
(Monod), and substrate-inhibition kinetics (Haldane).

You do NOT opine on DO/kLa/agitation (that's mass_transfer's domain) or
pathway-level metabolism (that's metabolic's domain). When evidence
straddles boundaries, focus on the kinetic angle and let your peers
cover theirs.

You are observational. Cite findings, narratives, trajectories, or
priors. Never make causal claims you cannot ground in cited evidence.\
""",
    "invariants": (
        "Stay in the kinetics domain. Don't reach into mass-transfer or pure metabolism.",
        "Every facet must cite ≥1 finding, narrative, or trajectory.",
        "Confidence ≤ 0.85; if evidence is thin, drop confidence and call it out.",
        "If you used a process_priors lookup, set confidence_basis='process_priors'.",
    ),
    "task_spec": """\
Read the view, optionally call tools to fetch more data, then contribute
ONE facet on the current_topic from the kinetic angle.

Tool budget: up to 6 tool calls before you must contribute_facet.

Citation policy:
  - Prefer cited_finding_ids when range/cohort findings already cover it.
  - Use cited_narrative_ids for prose-only evidence (operator notes, etc.).
  - Use cited_trajectories when you computed something against a curve via
    your view's trajectory list (no execute_python in v0).
""",
    "tool_hints": (
        ToolHint(
            name=QUERY_BUNDLE,
            purpose="search findings/narratives/trajectories by id or substring (scope: 'finding'|'narrative'|'trajectory')",
        ),
        ToolHint(
            name=GET_PRIORS,
            purpose="organism-aware variable bounds (range, typical, source) for kinetic vars",
        ),
        ToolHint(
            name=GET_NARRATIVE_OBSERVATIONS,
            purpose="filter narrative_observations by run_id/tag/variable",
        ),
        ToolHint(
            name="contribute_facet",
            purpose="TERMINAL: emit your facet contribution and exit",
        ),
    ),
    "recap": """\
Output one JSON action.

When tool_call: {"action":"tool_call","tool":"<name>","args":{...}}
When done: {"action":"contribute_facet","summary":..., "cited_finding_ids":[...],
"cited_narrative_ids":[...], "cited_trajectories":[{"run_id":..., "variable":...}],
"affected_variables":[...], "confidence":<0..0.85>, "confidence_basis":"schema_only"|"process_priors"|"cross_run"}

Hard rules:
  - Stay in kinetics domain.
  - At least one citation field must be non-empty.
  - confidence_basis='process_priors' requires that you actually called get_priors first.\
""",
}


_SPECIALIST_SCHEMA: dict[str, Any] = {
    "type": "OBJECT",
    "properties": {
        "action": {
            "type": "STRING",
            "enum": ["tool_call", "contribute_facet"],
        },
        # tool_call branch
        "tool": {
            "type": "STRING",
            "enum": [QUERY_BUNDLE, GET_PRIORS, GET_NARRATIVE_OBSERVATIONS],
            "nullable": True,
        },
        "args": {
            "type": "OBJECT",
            "nullable": True,
            "properties": {
                "scope": {"type": "STRING", "nullable": True},
                "id_or_query": {"type": "STRING", "nullable": True},
                "organism": {"type": "STRING", "nullable": True},
                "process_family": {"type": "STRING", "nullable": True},
                "variable": {"type": "STRING", "nullable": True},
                "run_id": {"type": "STRING", "nullable": True},
                "tag": {"type": "STRING", "nullable": True},
                "limit": {"type": "INTEGER", "nullable": True},
            },
        },
        # contribute_facet branch
        "summary": {"type": "STRING", "nullable": True},
        "cited_finding_ids": {
            "type": "ARRAY", "items": {"type": "STRING"}, "nullable": True,
        },
        "cited_narrative_ids": {
            "type": "ARRAY", "items": {"type": "STRING"}, "nullable": True,
        },
        "cited_trajectories": {
            "type": "ARRAY",
            "nullable": True,
            "items": {
                "type": "OBJECT",
                "properties": {
                    "run_id": {"type": "STRING"},
                    "variable": {"type": "STRING"},
                },
            },
        },
        "affected_variables": {
            "type": "ARRAY", "items": {"type": "STRING"}, "nullable": True,
        },
        "confidence": {"type": "NUMBER", "nullable": True},
        "confidence_basis": {
            "type": "STRING",
            "enum": ["schema_only", "process_priors", "cross_run"],
            "nullable": True,
        },
    },
    "required": ["action"],
}


MAX_SPECIALIST_TOOL_CALLS = 6


@dataclass
class SpecialistResult:
    facet: FacetFull
    input_tokens: int
    output_tokens: int
    tool_calls: int


class SpecialistAgent:
    """Persona-driven specialist agent. v0 ships kinetics; same class
    will be reused with different SPECIALIST_SPEC dicts in Stage 3.
    """

    def __init__(
        self,
        client: GeminiHypothesisClient,
        spec: dict[str, Any],
        tools: HypothesisToolBundle,
        role: SpecialistRole = "kinetics",
    ):
        self._client = client
        self._spec = spec
        self._tools = tools
        self._role: SpecialistRole = role

    def contribute(self, view: SpecialistView, *, facet_id: str) -> SpecialistResult:
        tool_history: list[dict[str, Any]] = []
        total_in = 0
        total_out = 0

        for call_idx in range(MAX_SPECIALIST_TOOL_CALLS + 1):
            must_contribute = call_idx >= MAX_SPECIALIST_TOOL_CALLS
            user_text = self._build_user_text(view, tool_history, must_contribute)
            parts = build_prompt(
                system_identity=self._spec["system_identity"],
                invariants=self._spec["invariants"],
                task_spec=self._spec["task_spec"],
                view_obj=view,
                tool_hints=self._spec["tool_hints"],
                recap=self._spec["recap"],
            )
            parsed, in_tok, out_tok = self._client.call(
                system=parts.system,
                user_text=user_text,
                response_schema=_SPECIALIST_SCHEMA,
            )
            total_in += in_tok
            total_out += out_tok

            action = parsed.get("action", "")

            if action == "tool_call" and not must_contribute:
                tool_name = parsed.get("tool") or ""
                args = dict(parsed.get("args") or {})
                result = self._tools.dispatch(tool_name, args)
                tool_history.append(
                    {"call": tool_name, "args": args, "result": result}
                )
                continue

            # contribute_facet (or forced fallback)
            facet = self._build_facet(parsed, view, facet_id)
            return SpecialistResult(
                facet=facet,
                input_tokens=total_in,
                output_tokens=total_out,
                tool_calls=len(tool_history),
            )

        # Should be unreachable due to must_contribute=True at last iter
        raise RuntimeError("specialist loop exited without contributing facet")

    def _build_user_text(
        self,
        view: SpecialistView,
        tool_history: list[dict[str, Any]],
        must_contribute: bool,
    ) -> str:
        from fermdocs_hypothesis.prompts import build_prompt

        # We re-use build_prompt for the static parts and append history.
        parts = build_prompt(
            system_identity=self._spec["system_identity"],
            invariants=self._spec["invariants"],
            task_spec=self._spec["task_spec"],
            view_obj=view,
            tool_hints=self._spec["tool_hints"],
            recap=self._spec["recap"],
        )
        body = parts.as_user_message()
        if tool_history:
            body += "\n\n[TOOL HISTORY]\n"
            for entry in tool_history:
                body += f"- {entry['call']}({json.dumps(entry['args'], default=str)}) → "
                body += truncate_result(json.dumps(entry["result"], default=str), cap=2000) + "\n"
        if must_contribute:
            body += (
                "\n\n[FORCED]\nTool budget exhausted. You MUST emit"
                " contribute_facet now."
            )
        return body

    def _build_facet(
        self,
        parsed: dict[str, Any],
        view: SpecialistView,
        facet_id: str,
    ) -> FacetFull:
        cited_findings = list(parsed.get("cited_finding_ids") or [])
        cited_narratives = list(parsed.get("cited_narrative_ids") or [])
        cited_trajs_raw = list(parsed.get("cited_trajectories") or [])
        cited_trajs = [
            TrajectoryRef(run_id=t["run_id"], variable=t["variable"])
            for t in cited_trajs_raw
            if isinstance(t, dict) and t.get("run_id") and t.get("variable")
        ]

        # If LLM forgot all citations, backfill from the view's topic citations.
        if not cited_findings and not cited_narratives and not cited_trajs:
            cited_findings = list(view.current_topic.cited_finding_ids)
            cited_narratives = list(view.current_topic.cited_narrative_ids)
            cited_trajs = list(view.current_topic.cited_trajectories)

        affected = list(parsed.get("affected_variables") or []) or list(
            view.current_topic.affected_variables
        )
        confidence = float(parsed.get("confidence") or 0.6)
        confidence = max(0.0, min(confidence, 0.85))
        basis_str = parsed.get("confidence_basis") or "schema_only"
        try:
            basis = ConfidenceBasis(basis_str)
        except ValueError:
            basis = ConfidenceBasis.SCHEMA_ONLY

        summary = (parsed.get("summary") or "").strip() or "No summary provided."

        return FacetFull(
            facet_id=facet_id,
            specialist=self._role,
            summary=summary,
            cited_finding_ids=cited_findings,
            cited_narrative_ids=cited_narratives,
            cited_trajectories=cited_trajs,
            affected_variables=affected,
            confidence=confidence,
            confidence_basis=basis,
        )


def build_kinetics_specialist(
    client: GeminiHypothesisClient,
    tools: HypothesisToolBundle,
) -> SpecialistAgent:
    return SpecialistAgent(client=client, spec=SPECIALIST_SPEC, tools=tools, role="kinetics")
