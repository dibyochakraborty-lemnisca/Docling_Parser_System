"""Trajectory builder: per-(run_id, variable) regular-grid time series.

If the dossier carries a `_trajectory_grid` hint with `dt_hours`, `start`, `end`,
build a regular grid and impute missing values via carry_forward. Otherwise use
observed timestamps directly (no imputation, single-point series for v1).
"""

from __future__ import annotations

from collections import defaultdict
from typing import Any

from fermdocs_characterize.schema import DataQuality, Trajectory
from fermdocs_characterize.views.summary import Summary, SummaryRow


def build_trajectories(
    summary: Summary, dossier: dict[str, Any]
) -> list[Trajectory]:
    """Group rows by (run_id, variable), produce one Trajectory per group.

    Trajectory IDs assigned in sorted order: (run_id, variable). T-0001, T-0002, …
    """
    grid_hint = dossier.get("_trajectory_grid") if isinstance(dossier, dict) else None

    grouped: dict[tuple[str, str], list[SummaryRow]] = defaultdict(list)
    for row in summary.rows:
        grouped[(row.run_id, row.variable)].append(row)

    trajectories: list[Trajectory] = []
    next_id = 1
    for key in sorted(grouped.keys()):
        run_id, variable = key
        rows = sorted(grouped[key], key=lambda r: r.time)
        unit = rows[0].unit
        traj_id = f"T-{next_id:04d}"
        next_id += 1

        if grid_hint and "dt_hours" in grid_hint:
            dt = float(grid_hint["dt_hours"])
            start = float(grid_hint.get("start", rows[0].time))
            end = float(grid_hint.get("end", rows[-1].time))
            n_steps = int(round((end - start) / dt)) + 1
            time_grid = [round(start + i * dt, 6) for i in range(n_steps)]

            obs_by_time: dict[float, SummaryRow] = {}
            for r in rows:
                obs_by_time[round(r.time, 6)] = r

            values: list[float | None] = []
            imputation_flags: list[bool] = []
            obs_ids_used: list[str] = []
            last_real_value: float | None = None
            for t in time_grid:
                if t in obs_by_time:
                    r = obs_by_time[t]
                    values.append(r.value)
                    imputation_flags.append(False)
                    obs_ids_used.append(r.observation_id)
                    last_real_value = r.value
                elif last_real_value is not None:
                    values.append(last_real_value)
                    imputation_flags.append(True)
                else:
                    values.append(None)
                    imputation_flags.append(False)

            n = len(time_grid)
            n_real = sum(
                1
                for v, flag in zip(values, imputation_flags, strict=True)
                if v is not None and not flag
            )
            n_imputed = sum(1 for flag in imputation_flags if flag)
            n_missing = sum(
                1
                for v, flag in zip(values, imputation_flags, strict=True)
                if v is None and not flag
            )
            pct_real = n_real / n
            pct_imputed = n_imputed / n
            pct_missing = n_missing / n
            imputation_method: str | None = "carry_forward" if n_imputed else None
            # Source observation IDs: only the real ones, in time order.
            source_obs_ids = obs_ids_used
        else:
            time_grid = [r.time for r in rows]
            values = [r.value for r in rows]
            imputation_flags = [False] * len(rows)
            imputation_method = None
            source_obs_ids = [r.observation_id for r in rows]
            pct_real = 1.0
            pct_imputed = 0.0
            pct_missing = 0.0

        trajectories.append(
            Trajectory(
                trajectory_id=traj_id,
                run_id=run_id,
                variable=variable,
                time_grid=time_grid,
                values=values,
                imputation_flags=imputation_flags,
                imputation_method=imputation_method,
                source_observation_ids=source_obs_ids,
                unit=unit,
                quality=round(pct_real, 6),
                data_quality=DataQuality(
                    pct_missing=round(pct_missing, 6),
                    pct_imputed=round(pct_imputed, 6),
                    pct_real=round(pct_real, 6),
                ),
            )
        )

    return trajectories


def get_trajectory(
    trajectories: list[Trajectory], run_id: str, variable: str
) -> Trajectory | None:
    for t in trajectories:
        if t.run_id == run_id and t.variable == variable:
            return t
    return None
