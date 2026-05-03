"""Synthesizer agent — merges facets into one HypothesisFull.

Plan ref: plans/2026-05-03-hypothesis-debate-v0.md §2 (synthesizer role),
§5 (synthesizer tool surface — emit_hypothesis only, no read tools needed).

The view is pre-loaded with all facets and a citation_universe.
Synthesizer's job: write one summary that preserves each facet's
distinguishing claim, and emit a HypothesisFull citing the union of
relevant evidence.

No tool loop — single LLM call, terminal emit.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from fermdocs_diagnose.schema import ConfidenceBasis, TrajectoryRef
from fermdocs_hypothesis.llm_clients import GeminiHypothesisClient
from fermdocs_hypothesis.prompts import ToolHint, build_prompt
from fermdocs_hypothesis.schema import HypothesisFull, SynthesizerView


SYNTHESIZER_SYSTEM = """\
You are the Synthesizer in a fermentation-hypothesis debate.

Each turn you receive 1-3 facet contributions from domain specialists
(kinetics, mass-transfer, metabolic). Your job: write ONE synthesized
hypothesis that integrates the angles into a coherent observational
claim — without losing any specialist's distinguishing point.

Do not average, do not paper over disagreements. If two facets push in
different directions, surface that tension in the summary; let the
critic + judge resolve it later.

You are observational. Cite evidence. Never claim causality you cannot
ground in the cited evidence.\
"""

SYNTHESIZER_INVARIANTS = (
    "Preserve each facet's distinguishing claim in your summary.",
    "Cite the union of facet citations (you can drop irrelevant ones).",
    "Confidence ≤ 0.85 and ≤ max(facet confidence).",
    "If any facet used confidence_basis='process_priors', use that; else use the strongest basis present.",
)

SYNTHESIZER_TASK = """\
Read the facets and citation_universe. Emit one HypothesisFull that
integrates all facets.

Drafting checklist:
  - Lead with the integrated claim, then mention the angles.
  - Pull through every facet_id into the facet_ids field.
  - Drop a citation only if it's clearly irrelevant to the integrated claim.
"""

SYNTHESIZER_RECAP = """\
Output one JSON object: emit_hypothesis with all required fields.

Hard rules:
  - facet_ids must include EVERY facet_id from the view.
  - At least one of cited_finding_ids / cited_narrative_ids /
    cited_trajectories must be non-empty.
  - confidence ≤ min(0.85, max facet confidence).
"""

SYNTHESIZER_TOOL_HINTS = (
    ToolHint(
        name="emit_hypothesis",
        purpose="TERMINAL: emit the synthesized hypothesis (only available action)",
    ),
)


_SYNTHESIZER_SCHEMA: dict[str, Any] = {
    "type": "OBJECT",
    "properties": {
        "summary": {"type": "STRING"},
        "facet_ids": {"type": "ARRAY", "items": {"type": "STRING"}},
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
        "confidence": {"type": "NUMBER"},
        "confidence_basis": {
            "type": "STRING",
            "enum": ["schema_only", "process_priors", "cross_run"],
        },
    },
    "required": ["summary", "facet_ids", "confidence", "confidence_basis"],
}


@dataclass
class SynthesizerResult:
    hypothesis: HypothesisFull
    input_tokens: int
    output_tokens: int


class SynthesizerAgent:
    def __init__(self, client: GeminiHypothesisClient):
        self._client = client

    def synthesize(self, view: SynthesizerView, *, hyp_id: str) -> SynthesizerResult:
        parts = build_prompt(
            system_identity=SYNTHESIZER_SYSTEM,
            invariants=SYNTHESIZER_INVARIANTS,
            task_spec=SYNTHESIZER_TASK,
            view_obj=view,
            tool_hints=SYNTHESIZER_TOOL_HINTS,
            recap=SYNTHESIZER_RECAP,
        )
        parsed, in_tok, out_tok = self._client.call(
            system=parts.system,
            user_text=parts.as_user_message(),
            response_schema=_SYNTHESIZER_SCHEMA,
        )
        hyp = self._build_hypothesis(parsed, view, hyp_id)
        return SynthesizerResult(hypothesis=hyp, input_tokens=in_tok, output_tokens=out_tok)

    def _build_hypothesis(
        self,
        parsed: dict[str, Any],
        view: SynthesizerView,
        hyp_id: str,
    ) -> HypothesisFull:
        facet_ids = list(parsed.get("facet_ids") or [])
        # Hard guard: must include every facet
        view_facet_ids = [f.facet_id for f in view.facets]
        if not facet_ids or set(facet_ids) != set(view_facet_ids):
            facet_ids = view_facet_ids

        cited_findings = list(parsed.get("cited_finding_ids") or [])
        cited_narratives = list(parsed.get("cited_narrative_ids") or [])
        cited_trajs_raw = list(parsed.get("cited_trajectories") or [])
        cited_trajs = [
            TrajectoryRef(run_id=t["run_id"], variable=t["variable"])
            for t in cited_trajs_raw
            if isinstance(t, dict) and t.get("run_id") and t.get("variable")
        ]

        # Backfill from citation_universe if LLM dropped everything
        if not cited_findings and not cited_narratives and not cited_trajs:
            cited_findings = list(view.citation_universe.finding_ids)
            cited_narratives = list(view.citation_universe.narrative_ids)
            cited_trajs = list(view.citation_universe.trajectories)

        affected = list(parsed.get("affected_variables") or []) or list(
            view.current_topic.affected_variables
        )
        confidence = float(parsed.get("confidence") or 0.6)
        confidence = max(0.0, min(confidence, 0.85))
        # Cap at max facet confidence
        if view.facets:
            facet_max = max(f.confidence for f in view.facets)
            confidence = min(confidence, facet_max)

        basis_str = parsed.get("confidence_basis") or "schema_only"
        try:
            basis = ConfidenceBasis(basis_str)
        except ValueError:
            basis = ConfidenceBasis.SCHEMA_ONLY

        summary = (parsed.get("summary") or "").strip() or "No synthesis provided."

        return HypothesisFull(
            hyp_id=hyp_id,
            summary=summary,
            facet_ids=facet_ids,
            cited_finding_ids=cited_findings,
            cited_narrative_ids=cited_narratives,
            cited_trajectories=cited_trajs,
            affected_variables=affected,
            confidence=confidence,
            confidence_basis=basis,
        )


def build_synthesizer(client: GeminiHypothesisClient) -> SynthesizerAgent:
    return SynthesizerAgent(client=client)
