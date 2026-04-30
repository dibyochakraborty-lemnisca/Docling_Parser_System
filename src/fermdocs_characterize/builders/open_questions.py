"""OpenQuestion builder: emits evidence_request questions for findings on
low-quality trajectories. v1 only emits this single category.

The Critic Agent uses these to attack findings whose evidence is sparse.
"""

from __future__ import annotations

from collections import defaultdict

from fermdocs_characterize.candidates.range_violation import LOW_QUALITY_THRESHOLD
from fermdocs_characterize.schema import (
    DecisionType,
    Finding,
    OpenQuestion,
    Trajectory,
)
from fermdocs_characterize.views.trajectories import get_trajectory

# Per-variable hint of what evidence would resolve the question. Generic
# fallbacks for variables not in this map.
_RESOLUTION_HINTS: dict[str, list[str]] = {
    "Biomass (X)": ["finer_biomass_sampling", "online_OD_trace"],
    "Substrate (S)": ["finer_substrate_sampling", "online_HPLC"],
    "PAA": ["finer_PAA_sampling"],
    "NH3": ["finer_NH3_sampling"],
}


def _resolution_hints_for(variable: str) -> list[str]:
    if variable in _RESOLUTION_HINTS:
        return _RESOLUTION_HINTS[variable]
    return [f"finer_{variable.lower().replace(' ', '_')}_sampling"]


def build_open_questions(
    findings: list[Finding],
    trajectories: list[Trajectory],
    *,
    dt_hours: float | None = None,
) -> list[OpenQuestion]:
    # Group findings by (run_id, variable) so one question covers all findings
    # on the same low-quality trajectory.
    grouped: dict[tuple[str, str], list[Finding]] = defaultdict(list)
    for f in findings:
        if not f.run_ids or not f.variables_involved:
            continue
        grouped[(f.run_ids[0], f.variables_involved[0])].append(f)

    questions: list[OpenQuestion] = []
    for key in sorted(grouped.keys()):
        run_id, variable = key
        traj = get_trajectory(trajectories, run_id, variable)
        if traj is None or traj.quality >= LOW_QUALITY_THRESHOLD:
            continue

        traj_findings = sorted(grouped[key], key=lambda f: f.finding_id)
        # Build a brief trajectory pattern string from real (non-imputed) values.
        real_values = [
            f"{v:g}"
            for v, flag in zip(traj.values, traj.imputation_flags, strict=True)
            if v is not None and not flag
        ]
        pattern = " → ".join(real_values) + f" {traj.unit}"

        pct_imputed_int = int(round(traj.data_quality.pct_imputed * 100))
        grid_phrase = (
            f"on a {int(dt_hours)}h grid" if dt_hours is not None else "on the imputed grid"
        )
        question_text = (
            f"{variable} trajectory in {run_id} has {pct_imputed_int}% imputed data"
            f" {grid_phrase}. Is the apparent growth pattern ({pattern})"
            f" genuinely monotonic, or does sparse sampling hide a non-monotonic"
            f" trajectory?"
        )

        questions.append(
            OpenQuestion(
                question_id=f"Q-{len(questions) + 1:04d}",
                question_text=question_text,
                decision_type=DecisionType.EVIDENCE_REQUEST,
                relevant_finding_ids=[f.finding_id for f in traj_findings],
                relevant_run_ids=[run_id],
                would_resolve_with=_resolution_hints_for(variable),
            )
        )
    return questions
