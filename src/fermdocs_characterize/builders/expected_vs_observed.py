"""Deviation builder: one Deviation per summary row that has expected/std_dev.

Unlike findings, deviations include in-spec measurements (sigmas < 2). They
give the Diagnosis Agent the full setpoint-vs-observed context, not just the
violations.

Sort: (run_id, time, variable). IDs assigned in sort order: D-0001, D-0002, …
"""

from __future__ import annotations

from fermdocs_characterize.schema import Deviation
from fermdocs_characterize.views.summary import Summary


def build_deviations(summary: Summary) -> list[Deviation]:
    rows = [r for r in summary.rows if r.expected is not None and r.expected_std_dev is not None]
    rows = sorted(rows, key=lambda r: (r.run_id, r.time, r.variable))
    out: list[Deviation] = []
    for i, r in enumerate(rows, start=1):
        residual = round(r.value - r.expected, 6) if r.expected is not None else 0.0
        sigmas: float | None
        if r.expected_std_dev and r.expected_std_dev != 0 and r.expected is not None:
            sigmas = round((r.value - r.expected) / r.expected_std_dev, 6)
        else:
            sigmas = None
        out.append(
            Deviation(
                deviation_id=f"D-{i:04d}",
                run_id=r.run_id,
                variable=r.variable,
                time=r.time,
                expected=r.expected if r.expected is not None else 0.0,
                expected_std_dev=r.expected_std_dev,
                observed=r.value,
                residual=residual,
                sigmas=sigmas,
                source_observation_ids=[r.observation_id],
            )
        )
    return out
