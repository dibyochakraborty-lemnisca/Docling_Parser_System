"""Tool surface the ReAct loop calls. Three thin read-only tools.

Plan ref: plans/2026-05-02-diagnosis-agent.md §5.

Why thin: comparison logic (`compare_to_nominal`) stays in the model — that's
the reasoning we're paying the LLM for. Whole-dossier access invites context
bloat. AgentContext already enumerates findings, so search is unneeded.

Errors return structured `{error, hint}` dicts rather than raising. The LLM
sees the hint and can pivot; raising would break the ReAct loop.
"""

from __future__ import annotations

from typing import TypedDict

from fermdocs_characterize.schema import CharacterizationOutput, Finding, Trajectory
from fermdocs_characterize.specs import Spec, SpecsProvider


class ToolError(TypedDict):
    error: str
    hint: str


def get_finding(
    finding_id: str, *, output: CharacterizationOutput
) -> Finding | ToolError:
    """Full Finding record. Use when AgentContext.top_findings summary is too
    sparse for a magnitude judgement.
    """
    for f in output.findings:
        if f.finding_id == finding_id:
            return f
    known = [f.finding_id for f in output.findings[:5]]
    return {
        "error": f"finding_id {finding_id!r} not in output",
        "hint": f"known finding_ids include {known}; did you mean one of these?",
    }


def get_trajectory(
    run_id: str, variable: str, *, output: CharacterizationOutput
) -> Trajectory | ToolError:
    """Full Trajectory: time grid + values + imputation flags + quality."""
    for t in output.trajectories:
        if t.run_id == run_id and t.variable == variable:
            return t
    known_runs = sorted({t.run_id for t in output.trajectories})
    known_vars = sorted({t.variable for t in output.trajectories})
    return {
        "error": f"no trajectory for run_id={run_id!r} variable={variable!r}",
        "hint": (
            f"known run_ids={known_runs[:5]} variables={known_vars[:10]}"
        ),
    }


def get_spec(
    variable: str, *, specs: SpecsProvider
) -> Spec | ToolError:
    """Spec carrying nominal, std_dev, unit, provenance, source_ref."""
    spec = specs.get(variable)
    if spec is None:
        return {
            "error": f"no spec for variable={variable!r}",
            "hint": (
                "spec may be missing because the schema lacks nominal/std_dev"
                " for this variable, or the dossier did not override it"
            ),
        }
    return spec
