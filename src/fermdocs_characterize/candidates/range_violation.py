"""range_violation candidate generator (v1, deterministic).

For each summary row that has nominal+std_dev specs, compute sigmas. If
|sigmas| >= 2.0, emit a CandidateFinding. Severity rubric per
vocabularies/finding_types.md.

v1 trajectory caveat: if the row's trajectory has quality < 0.8, attach a
caveat naming the imputation level. Surfaces uncertainty for the Critic.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from fermdocs_characterize.schema import (
    EvidenceStrength,
    ExtractedVia,
    FindingType,
    Severity,
    TimeWindow,
    Trajectory,
)
from fermdocs_characterize.views.summary import Summary, SummaryRow
from fermdocs_characterize.views.trajectories import get_trajectory

LOW_QUALITY_THRESHOLD = 0.8


@dataclass(frozen=True)
class CandidateFinding:
    """A finding-shaped record without an ID. The pipeline assigns IDs after
    sorting all candidates from all generators. Decoupling generation from ID
    assignment is what gives stable IDs across runs.
    """

    type: FindingType
    severity: Severity
    summary: str
    confidence: float
    extracted_via: ExtractedVia
    caveats: list[str]
    competing_explanations: list[str]
    evidence_strength: EvidenceStrength
    evidence_observation_ids: list[str]
    variables_involved: list[str]
    time_window: TimeWindow | None
    run_ids: list[str]
    statistics: dict[str, Any] = field(default_factory=dict)

    @property
    def sort_key(self) -> tuple:
        """Sort: severity desc, |sigmas| desc, run_id, time, variable."""
        severity_rank = {
            Severity.CRITICAL: 4,
            Severity.MAJOR: 3,
            Severity.MINOR: 2,
            Severity.INFO: 1,
        }[self.severity]
        sigmas = self.statistics.get("sigmas", 0.0)
        run_id = self.run_ids[0] if self.run_ids else ""
        time_start = self.time_window.start if self.time_window and self.time_window.start is not None else 0.0
        variable = self.variables_involved[0] if self.variables_involved else ""
        return (-severity_rank, -abs(sigmas), run_id, time_start, variable)


def _severity_for_sigmas(sigmas: float) -> Severity | None:
    a = abs(sigmas)
    if a < 2.0:
        return None
    if a < 3.0:
        return Severity.MINOR
    if a < 5.0:
        return Severity.MAJOR
    return Severity.CRITICAL


_CONFIDENCE_BY_SEVERITY = {
    Severity.MINOR: 0.85,
    Severity.MAJOR: 0.95,
    Severity.CRITICAL: 0.99,
    Severity.INFO: 0.6,
}


def _trajectory_caveat(trajectory: Trajectory | None, dt_hours: float | None) -> str | None:
    if trajectory is None or trajectory.quality >= LOW_QUALITY_THRESHOLD:
        return None
    pct_bad = trajectory.data_quality.pct_imputed + trajectory.data_quality.pct_missing
    pct_int = int(round(pct_bad * 100))
    if dt_hours is not None:
        return f"trajectory has {pct_int}% imputed/missing data on a {int(dt_hours)}h grid"
    return f"trajectory has {pct_int}% imputed/missing data"


def find_range_violations(
    summary: Summary,
    trajectories: list[Trajectory],
    *,
    dt_hours: float | None = None,
) -> list[CandidateFinding]:
    """Emit one CandidateFinding per row whose |sigmas| >= 2."""
    candidates: list[CandidateFinding] = []
    for row in summary.rows:
        if row.expected is None or row.expected_std_dev is None or row.expected_std_dev == 0:
            continue
        # Round before threshold check so FP boundary cases (e.g. 2.9999… from
        # 1.15 / 0.05) land in the right severity tier deterministically.
        sigmas = round((row.value - row.expected) / row.expected_std_dev, 6)
        severity = _severity_for_sigmas(sigmas)
        if severity is None:
            continue

        traj = get_trajectory(trajectories, row.run_id, row.variable)
        caveat = _trajectory_caveat(traj, dt_hours)
        caveats = [caveat] if caveat else []

        direction = "above" if sigmas > 0 else "below"
        residual = round(row.value - row.expected, 6)
        sigmas_rounded = sigmas  # already rounded above
        # Sigmas and time use .1f for readability ("5.0σ", "24.0h"); other
        # numbers use :g to preserve natural precision (0.05 stays 0.05).
        summary_text = (
            f"{row.variable} {row.value:g} is {abs(sigmas_rounded):.1f}σ {direction}"
            f" nominal {row.expected:.1f} ± {row.expected_std_dev:g}"
            f" in run {row.run_id} at {row.time:.1f}h"
        )
        candidates.append(
            CandidateFinding(
                type=FindingType.RANGE_VIOLATION,
                severity=severity,
                summary=summary_text,
                confidence=_CONFIDENCE_BY_SEVERITY[severity],
                extracted_via=ExtractedVia.DETERMINISTIC,
                caveats=caveats,
                competing_explanations=[],
                evidence_strength=EvidenceStrength(
                    n_observations=1, n_independent_runs=1, statistical_power=None
                ),
                evidence_observation_ids=[row.observation_id],
                variables_involved=[row.variable],
                time_window=TimeWindow(start=row.time, end=row.time),
                run_ids=[row.run_id],
                statistics={
                    "nominal": row.expected,
                    "std_dev": row.expected_std_dev,
                    "observed": row.value,
                    "sigmas": sigmas_rounded,
                    "residual": residual,
                },
            )
        )
    return candidates
