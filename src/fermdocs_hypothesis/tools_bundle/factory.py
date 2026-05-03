"""Tool dispatcher for hypothesis-stage agents.

Plan ref: plans/2026-05-03-hypothesis-debate-v0.md §5.

Provider-agnostic: every tool returns a JSON-safe dict. LLM clients
(llm_clients.py) translate between provider tool schemas and these
canonical names.

Stage 2 surface (subset of plan §5):

  Read tools (every agent):
    - query_bundle(scope, id_or_query)
    - get_priors(organism, process_family, variable)
    - get_narrative_observations(run_id, tag, variable, limit)

  Specialist terminal tool:
    - contribute_facet(...)

  Synthesizer terminal tool:
    - emit_hypothesis(...)

  Orchestrator tools:
    - select_topic(topic_id, rationale)
    - add_open_question(question, tags, raised_by)
    - resolve_open_question(qid, resolution)
    - exit_stage(reason)

Critic + judge tools land in Stage 3.

The factory builds a dispatcher around a LoadedBundle so tool calls have
read access to findings/narratives/trajectories/priors. Stateless across
calls except for the open-questions ledger (which the runner re-derives
from events anyway).
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, Callable

from fermdocs.domain.process_priors import resolve_priors
from fermdocs_hypothesis.bundle_loader import LoadedBundle


# Tool name constants (also used in agent prompts).
QUERY_BUNDLE = "query_bundle"
GET_PRIORS = "get_priors"
GET_NARRATIVE_OBSERVATIONS = "get_narrative_observations"
CONTRIBUTE_FACET = "contribute_facet"
EMIT_HYPOTHESIS = "emit_hypothesis"
SELECT_TOPIC = "select_topic"
ADD_OPEN_QUESTION = "add_open_question"
RESOLVE_OPEN_QUESTION = "resolve_open_question"
EXIT_STAGE = "exit_stage"


# Output cap to keep tool results bounded — over-cap returns truncated with marker.
MAX_TOOL_RESULT_BYTES = 8000


@dataclass
class HypothesisToolBundle:
    """Read-only tool surface for read-side tools. Terminal tools
    (contribute_facet, emit_hypothesis, select_topic, etc.) are
    intercepted by the agent layer and not dispatched here — the agent
    parses them out of the LLM response and hands them to the runner.

    This object is passed to LLM clients and to tests.
    """

    bundle: LoadedBundle

    # --- query_bundle ---

    def query_bundle(self, scope: str, id_or_query: str) -> dict[str, Any]:
        """scope ∈ {'finding', 'narrative', 'trajectory', 'open_question_diag'}.

        id_or_query: exact ID OR a substring/keyword (case-insensitive) for
        scopes that support search. Returns up to 5 hits.
        """
        scope = (scope or "").strip().lower()
        q = (id_or_query or "").strip()
        if scope == "finding":
            return self._query_findings(q)
        if scope == "narrative":
            return self._query_narratives(q)
        if scope == "trajectory":
            return self._query_trajectories(q)
        if scope == "open_question_diag":
            return self._query_diag_questions(q)
        return {
            "error": f"unknown scope: {scope!r}",
            "valid_scopes": [
                "finding",
                "narrative",
                "trajectory",
                "open_question_diag",
            ],
        }

    def _query_findings(self, q: str) -> dict[str, Any]:
        char = self.bundle.characterization
        # Exact ID first
        for f in char.findings:
            if f.finding_id == q:
                return {
                    "scope": "finding",
                    "exact": True,
                    "finding": _finding_to_dict(f),
                }
        # Substring on summary or variables
        ql = q.lower()
        hits = []
        for f in char.findings:
            if ql in f.summary.lower() or any(ql in v.lower() for v in f.variables_involved):
                hits.append(_finding_to_dict(f))
            if len(hits) >= 5:
                break
        return {"scope": "finding", "exact": False, "hits": hits}

    def _query_narratives(self, q: str) -> dict[str, Any]:
        char = self.bundle.characterization
        for n in char.narrative_observations:
            if n.narrative_id == q:
                return {
                    "scope": "narrative",
                    "exact": True,
                    "narrative": _narrative_to_dict(n),
                }
        ql = q.lower()
        hits = []
        for n in char.narrative_observations:
            blob = (n.text + " " + (n.tag.value if hasattr(n.tag, "value") else str(n.tag))).lower()
            if ql in blob:
                hits.append(_narrative_to_dict(n))
            if len(hits) >= 5:
                break
        return {"scope": "narrative", "exact": False, "hits": hits}

    def _query_trajectories(self, q: str) -> dict[str, Any]:
        char = self.bundle.characterization
        ql = q.lower()
        hits = []
        for t in char.trajectories:
            if ql in t.run_id.lower() or ql in t.variable.lower():
                hits.append(
                    {
                        "trajectory_id": t.trajectory_id,
                        "run_id": t.run_id,
                        "variable": t.variable,
                        "unit": t.unit,
                        "quality": t.quality,
                        "n_points": len(t.time_grid),
                    }
                )
            if len(hits) >= 5:
                break
        return {"scope": "trajectory", "hits": hits}

    def _query_diag_questions(self, q: str) -> dict[str, Any]:
        diag = self.bundle.diagnosis
        for oq in diag.open_questions:
            if oq.question_id == q:
                return {
                    "scope": "open_question_diag",
                    "exact": True,
                    "question": _diag_question_to_dict(oq),
                }
        ql = q.lower()
        hits = [_diag_question_to_dict(oq) for oq in diag.open_questions if ql in oq.question.lower()]
        return {"scope": "open_question_diag", "hits": hits[:5]}

    # --- get_priors (mirrors diagnose tool exactly) ---

    def get_priors(
        self,
        *,
        organism: str | None = None,
        process_family: str | None = None,
        variable: str | None = None,
    ) -> dict[str, Any]:
        organism = organism or self.bundle.hyp_input.organism
        process_family = process_family or self.bundle.hyp_input.process_family
        try:
            from fermdocs.domain.process_priors import cached_priors
            priors = cached_priors()
        except Exception as e:
            return {"error": f"priors not loadable: {e}", "results": []}
        if not organism:
            return {"results": [], "note": "no organism resolved"}
        resolved = resolve_priors(
            priors,
            organism=organism,
            process_family=process_family,
            variable=variable,
        )
        return {"results": [r.to_dict() for r in resolved]}

    # --- get_narrative_observations ---

    def get_narrative_observations(
        self,
        *,
        run_id: str | None = None,
        tag: str | None = None,
        variable: str | None = None,
        limit: int = 50,
    ) -> dict[str, Any]:
        char = self.bundle.characterization
        out = []
        for n in char.narrative_observations:
            if run_id and (n.run_id or "") != run_id:
                continue
            t = n.tag.value if hasattr(n.tag, "value") else str(n.tag)
            if tag and t != tag:
                continue
            if variable and variable.lower() not in (n.text.lower() + " ".join(n.affected_variables or []).lower()):
                continue
            out.append(_narrative_to_dict(n))
            if len(out) >= limit:
                break
        return {"results": out, "count": len(out)}

    # --- dispatch (provider-neutral) ---

    def dispatch(self, tool_name: str, args: dict[str, Any]) -> dict[str, Any]:
        if tool_name == QUERY_BUNDLE:
            return self.query_bundle(args.get("scope", ""), args.get("id_or_query", ""))
        if tool_name == GET_PRIORS:
            return self.get_priors(
                organism=args.get("organism"),
                process_family=args.get("process_family"),
                variable=args.get("variable"),
            )
        if tool_name == GET_NARRATIVE_OBSERVATIONS:
            return self.get_narrative_observations(
                run_id=args.get("run_id"),
                tag=args.get("tag"),
                variable=args.get("variable"),
                limit=int(args.get("limit", 50)),
            )
        return {"error": f"unknown read-tool: {tool_name!r}"}


# ---------- helpers ----------


def _finding_to_dict(f) -> dict[str, Any]:
    return {
        "finding_id": f.finding_id,
        "type": f.type.value if hasattr(f.type, "value") else str(f.type),
        "severity": f.severity.value if hasattr(f.severity, "value") else str(f.severity),
        "summary": f.summary,
        "confidence": f.confidence,
        "variables_involved": list(f.variables_involved),
    }


def _narrative_to_dict(n) -> dict[str, Any]:
    return {
        "narrative_id": n.narrative_id,
        "tag": n.tag.value if hasattr(n.tag, "value") else str(n.tag),
        "text": n.text,
        "run_id": n.run_id,
        "time_h": n.time_h,
        "affected_variables": list(n.affected_variables or []),
    }


def _diag_question_to_dict(oq) -> dict[str, Any]:
    return {
        "question_id": oq.question_id,
        "question": oq.question,
        "why_it_matters": oq.why_it_matters,
        "answer_format_hint": oq.answer_format_hint,
        "cited_finding_ids": list(oq.cited_finding_ids),
        "cited_narrative_ids": list(oq.cited_narrative_ids),
    }


def truncate_result(result_json: str, cap: int = MAX_TOOL_RESULT_BYTES) -> str:
    """Bound tool result size; mark truncation explicitly."""
    if len(result_json) <= cap:
        return result_json
    return result_json[:cap] + f"\n...[TRUNCATED {len(result_json) - cap} bytes]"


def make_tool_bundle(bundle: LoadedBundle) -> HypothesisToolBundle:
    return HypothesisToolBundle(bundle=bundle)
