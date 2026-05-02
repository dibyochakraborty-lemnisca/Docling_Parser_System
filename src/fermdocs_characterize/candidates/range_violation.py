"""range_violation candidate generator (v1, deterministic).

For each summary row that has nominal+std_dev specs, compute sigmas. If
|sigmas| >= 2.0, emit a CandidateFinding. Severity rubric per
vocabularies/finding_types.md.

v1 trajectory caveat: if the row's trajectory has quality < 0.8, attach a
caveat naming the imputation level. Surfaces uncertainty for the Critic.

Aggregation circuit breaker: if a single (variable, run_id) pair produces
more than AGGREGATE_THRESHOLD per-row violations, the per-row findings are
collapsed into one rollup finding. This protects downstream agents from
prompt-budget explosions when schema specs have setpoint semantics but the
real signal is a trajectory (e.g. biomass growing 50x over a fed-batch
fermentation).
"""

from __future__ import annotations

from collections import defaultdict
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
AGGREGATE_THRESHOLD = 100  # per (variable, run_id) — above this we roll up


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
    return _aggregate_runaways(candidates)


def _aggregate_runaways(
    candidates: list[CandidateFinding],
) -> list[CandidateFinding]:
    """Collapse runaway (variable, run_id) groups into single rollup findings.

    Why: when schema specs have setpoint semantics (one nominal at inoculation)
    but the real signal is a trajectory (biomass grows 50x over 230h), every
    timestep flags a violation. 9000+ findings explode the diagnosis-agent
    prompt budget. One rollup carries the same information density and frees
    the budget for other signals.

    Aggregation preserves: max |sigmas|, observed range, time window, all
    contributing observation_ids. Severity is the highest seen in the group.
    """
    by_group: dict[tuple[str, str], list[CandidateFinding]] = defaultdict(list)
    others: list[CandidateFinding] = []
    for c in candidates:
        var = c.variables_involved[0] if c.variables_involved else None
        run = c.run_ids[0] if c.run_ids else None
        if var is None or run is None:
            others.append(c)
            continue
        by_group[(var, run)].append(c)

    out: list[CandidateFinding] = list(others)
    for (var, run), group in by_group.items():
        if len(group) <= AGGREGATE_THRESHOLD:
            out.extend(group)
            continue
        out.append(_make_rollup(var, run, group))
    return out


_SEVERITY_RANK = {
    Severity.INFO: 1,
    Severity.MINOR: 2,
    Severity.MAJOR: 3,
    Severity.CRITICAL: 4,
}


def _make_rollup(
    variable: str, run_id: str, group: list[CandidateFinding]
) -> CandidateFinding:
    n = len(group)
    max_sev = max(group, key=lambda c: _SEVERITY_RANK[c.severity]).severity
    sigmas_list = [
        c.statistics.get("sigmas", 0.0) for c in group if "sigmas" in c.statistics
    ]
    observed_list = [
        c.statistics.get("observed") for c in group if c.statistics.get("observed") is not None
    ]
    times = [
        c.time_window.start
        for c in group
        if c.time_window and c.time_window.start is not None
    ]
    nominal = group[0].statistics.get("nominal")
    std_dev = group[0].statistics.get("std_dev")

    max_abs_sigmas = max((abs(s) for s in sigmas_list), default=0.0)
    obs_min = min(observed_list, default=None)
    obs_max = max(observed_list, default=None)
    t_start = min(times, default=None)
    t_end = max(times, default=None)

    obs_range = (
        f"{obs_min:g}-{obs_max:g}"
        if obs_min is not None and obs_max is not None and obs_min != obs_max
        else (f"{obs_min:g}" if obs_min is not None else "?")
    )
    summary_text = (
        f"{variable} violates spec across {n} observations in run {run_id}"
        f" (range {obs_range}, max {max_abs_sigmas:.1f}σ"
        f" vs nominal {nominal:g} ± {std_dev:g})"
        if nominal is not None and std_dev is not None
        else (
            f"{variable} violates spec across {n} observations in run {run_id}"
            f" (range {obs_range}, max {max_abs_sigmas:.1f}σ)"
        )
    )

    # Carry every contributing observation_id so the diagnosis agent can drill
    # down via get_finding / get_trajectory if it wants to. Bounded by group
    # size; the AgentContext's top_findings cap keeps prompt budget safe.
    evidence_ids = [
        oid for c in group for oid in c.evidence_observation_ids
    ]

    return CandidateFinding(
        type=FindingType.RANGE_VIOLATION,
        severity=max_sev,
        summary=summary_text,
        confidence=_CONFIDENCE_BY_SEVERITY[max_sev],
        extracted_via=ExtractedVia.DETERMINISTIC,
        caveats=[
            f"aggregated from {n} per-row range_violation candidates;"
            " schema spec likely carries setpoint semantics rather than"
            " trajectory bounds"
        ],
        competing_explanations=[],
        evidence_strength=EvidenceStrength(
            n_observations=n, n_independent_runs=1, statistical_power=None
        ),
        evidence_observation_ids=evidence_ids,
        variables_involved=[variable],
        time_window=TimeWindow(start=t_start, end=t_end),
        run_ids=[run_id],
        statistics={
            "nominal": nominal,
            "std_dev": std_dev,
            "observed_min": obs_min,
            "observed_max": obs_max,
            "max_abs_sigmas": max_abs_sigmas,
            "n_violations": n,
            "aggregated": True,
        },
    )
