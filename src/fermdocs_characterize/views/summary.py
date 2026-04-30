"""Summary view: dossier observations → sorted list of SummaryRow.

Forward-compatible: reads `run_id` and `timestamp_h` from each observation's
`source.locator`. Real fermdocs ingestion does not yet emit these fields;
fixtures carry them so v1 can be tested end-to-end. Observations missing either
are logged and skipped (returned in `dropped`).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from fermdocs_characterize.specs import SpecsProvider


@dataclass(frozen=True)
class SummaryRow:
    observation_id: str
    run_id: str
    time: float
    variable: str
    value: float
    unit: str
    expected: float | None
    expected_std_dev: float | None


@dataclass(frozen=True)
class DroppedObservation:
    observation_id: str
    variable: str
    reason: str


@dataclass(frozen=True)
class Summary:
    rows: list[SummaryRow]
    dropped: list[DroppedObservation]
    run_ids: list[str]
    variables: list[str]


def build_summary(
    dossier: dict[str, Any], specs: SpecsProvider
) -> Summary:
    """Flatten dossier.golden_columns into a sorted list of SummaryRow.

    Sort key: (run_id, time, variable, observation_id) for ID stability.
    """
    rows: list[SummaryRow] = []
    dropped: list[DroppedObservation] = []
    run_ids: set[str] = set()
    variables: set[str] = set()

    golden_columns = dossier.get("golden_columns", {})
    for variable, col_data in golden_columns.items():
        observations = col_data.get("observations", []) if isinstance(col_data, dict) else []
        for obs in observations:
            obs_id = obs.get("observation_id")
            value = obs.get("value")
            unit = obs.get("unit") or ""
            locator = (obs.get("source") or {}).get("locator") or {}
            run_id = locator.get("run_id")
            time = locator.get("timestamp_h")

            if obs_id is None or value is None:
                dropped.append(
                    DroppedObservation(
                        observation_id=str(obs_id),
                        variable=variable,
                        reason="missing observation_id or value",
                    )
                )
                continue
            if run_id is None or time is None:
                dropped.append(
                    DroppedObservation(
                        observation_id=str(obs_id),
                        variable=variable,
                        reason="locator missing run_id or timestamp_h",
                    )
                )
                continue

            spec = specs.get(variable)
            rows.append(
                SummaryRow(
                    observation_id=str(obs_id),
                    run_id=str(run_id),
                    time=float(time),
                    variable=variable,
                    value=float(value),
                    unit=str(unit),
                    expected=spec.nominal if spec else None,
                    expected_std_dev=spec.std_dev if spec else None,
                )
            )
            run_ids.add(str(run_id))
            variables.add(variable)

    rows.sort(key=lambda r: (r.run_id, r.time, r.variable, r.observation_id))
    return Summary(
        rows=rows,
        dropped=dropped,
        run_ids=sorted(run_ids),
        variables=sorted(variables),
    )
