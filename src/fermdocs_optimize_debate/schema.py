"""Contracts for the optimization debate.

The debate REUSES the hypothesis engine's output shape (`HypothesisOutput` with
`FinalHypothesis` entries) — a confirmed hypothesis IS a debated optimization
lever. This module adds the thin view the optimizer consumes through the
inform-only seam: `OptimizationLever` maps a debated hypothesis onto the knob(s)
the optimizer can actually move, without re-deriving the debate.
"""
from __future__ import annotations

import json
from pathlib import Path

from pydantic import BaseModel, Field



def _get(obj, attr, default=None):
    """Read `attr` from either an object (getattr) or a dict (key)."""
    if isinstance(obj, dict):
        return obj.get(attr, default)
    return getattr(obj, attr, default)


class OptimizationLever(BaseModel):
    """A debated, evidence-grounded lever — the optimizer's prior over the box.

    Built from a green-flagged FinalHypothesis. The optimizer reads these to set
    the narrative and reconcile its oracle-verified optimum against the debate; it
    does NOT use them to constrain the search (inform-only)."""

    lever_id: str                              # the hypothesis hyp_id
    summary: str
    # The controllable lever(s) this hypothesis is about. For the data path these
    # are the REAL discovered design factors (from the hypothesis's cited
    # within-run associations, e.g. main_fermentation_nitrogen_source); only when
    # a hypothesis cites no association (LABS synthetic path) do they fall back to
    # the static LABS knob mapping (biomass/total_sub/malt_frac/dilution).
    knobs: list[str] = Field(default_factory=list)
    affected_variables: list[str] = Field(default_factory=list)
    actionable_recommendation: str | None = None
    confidence: float = 0.0
    supporting_specialists: list[str] = Field(default_factory=list)


def levers_from_output(output) -> list[OptimizationLever]:
    """Project a HypothesisOutput's final hypotheses into optimization levers,
    highest-confidence first. Levers that map to no knob are kept (knobs=[]) so
    the narrative can still cite them, but they sort below knob-bearing levers.

    `output` is duck-typed (a real HypothesisOutput, a stub, OR a parsed JSON
    dict) so this works without importing the engine here."""
    levers: list[OptimizationLever] = []
    for h in _get(output, "final_hypotheses", []) or []:
        affected = list(_get(h, "affected_variables", []) or [])
        specialists = [s.value if hasattr(s, "value") else str(s)
                       for s in _get(h, "supporting_specialists", []) or []]
        # The REAL design factors the hypothesis cited (WRA-<lever> -> <lever>).
        # No LABS knob fallback (de-LABS): a hypothesis that cites no association
        # carries no knob and sorts below knob-bearing levers.
        assoc_ids = list(_get(h, "cited_association_ids", []) or [])
        knobs = [a[4:] if a.startswith("WRA-") else a for a in assoc_ids]
        levers.append(OptimizationLever(
            lever_id=_get(h, "hyp_id", "H-0000"),
            summary=_get(h, "summary", ""),
            knobs=knobs,
            affected_variables=affected,
            actionable_recommendation=_get(h, "actionable_recommendation", None),
            confidence=float(_get(h, "confidence", 0.0) or 0.0),
            supporting_specialists=specialists,
        ))
    levers.sort(key=lambda v: (bool(v.knobs), v.confidence), reverse=True)
    return levers


def levers_from_debate_json(path: str | Path) -> list[OptimizationLever]:
    """Read an optimization_debate.json (HypothesisOutput shape) into levers.

    This is the lightweight seam the optimizer uses — it pulls in no engine code,
    just parses the file the debate stage wrote."""
    p = Path(path)
    if not p.exists():
        return []
    try:
        return levers_from_output(json.loads(p.read_text()))
    except (json.JSONDecodeError, OSError, ValueError):
        return []
