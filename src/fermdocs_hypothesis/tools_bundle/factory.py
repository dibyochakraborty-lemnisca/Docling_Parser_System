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
QUERY_RELATIONSHIP = "query_relationship"
EXECUTE_PYTHON = "execute_python"
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

    # --- query_relationship (opportunity debate: test a lever's titer effect) ---

    def query_relationship(self, lever: str, objective: str | None = None) -> dict[str, Any]:
        """Return how a controllable design factor tracked the objective across
        runs (the within-run association). Looks up the precomputed pool, so it's
        cheap and deterministic. Lets a specialist TEST 'does nitrogen source
        actually move titer?' instead of asserting. Reports ALL design factors,
        including ones beyond the top-N shown in the view."""
        pool = getattr(self.bundle, "within_run_pool", None) or []
        want = (lever or "").strip()
        for a in pool:
            if want and (a.lever == want or a.assoc_id == want):
                return {
                    "assoc_id": a.assoc_id, "lever": a.lever, "delta": a.delta,
                    "direction": a.direction, "n": a.n, "norm_effect": a.norm_effect,
                    "best_setting": a.best_setting, "objective": a.objective,
                    "summary": a.summary,
                }
        known = [a.lever for a in pool]
        return {
            "error": f"no within-run association for {want!r}",
            "known_levers": known,
            "hint": ("Cite a lever from known_levers, or use execute_python to "
                     "compute a custom relationship from observations.csv."),
        }

    # --- execute_python (constrained, read-only analysis over observations) ---

    def execute_python(self, code: str) -> dict[str, Any]:
        """Run short analysis code in the shared sandbox (same isolation the
        critic uses: subprocess + rlimit + timeout). `obs` (a pandas DataFrame of
        observations.csv: run_id, variable, time_h, value) and `OBS_CSV` (its
        path) are pre-loaded so a specialist can CHECK a claim against the data
        instead of asserting. Tighter than the diagnosis sandbox: 60s, 10KB out."""
        from fermdocs_diagnose.tools_bundle.execute_python import execute_python as _run

        bundle_dir = getattr(self.bundle, "bundle_dir", None)
        obs_path = None
        if bundle_dir is not None:
            cand = f"{bundle_dir}/characterization/observations.csv"
            obs_path = cand
        preamble = (
            "import pandas as pd, numpy as np\n"
            f"OBS_CSV = {obs_path!r}\n"
            "obs = pd.read_csv(OBS_CSV) if OBS_CSV else None\n"
        )
        result = _run(preamble + (code or ""), timeout=60)
        text = result.to_agent_text()
        if len(text) > 10_000:
            text = text[:10_000] + "\n... (truncated at 10KB)"
        return {"output": text, "timed_out": result.timed_out,
                "returncode": result.returncode}

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
                limit=_safe_limit(args.get("limit"), default=50),
            )
        if tool_name == QUERY_RELATIONSHIP:
            return self.query_relationship(args.get("lever", ""), args.get("objective"))
        if tool_name == EXECUTE_PYTHON:
            return self.execute_python(args.get("code", ""))
        return {"error": f"unknown read-tool: {tool_name!r}"}


def _safe_limit(raw: object, *, default: int, cap: int = 10_000) -> int:
    """Coerce an LLM-emitted limit value to a sane int.

    Handles three failure modes seen in production:
      - explicit None (LLM passed `limit: null`)
      - non-numeric strings ('all', 'max')
      - absurdly long numeric strings (>4300 digits triggers Python 3.11+
        `ValueError: Exceeds the limit for integer string conversion`)
    """
    if raw is None:
        return default
    if isinstance(raw, str):
        # Truncate before int() so we don't trip the digit limit.
        raw = raw.strip()[:8]
        if not raw or not raw.lstrip("-").isdigit():
            return default
    try:
        n = int(raw)
    except (ValueError, TypeError):
        return default
    if n <= 0:
        return default
    return min(n, cap)


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
