"""Deterministic Plotly figure builder from synthesizer chart_specs.

Plan ref: charts-and-pdf-export branch.

The LLM emits intent (kind, runs, variables, story) via ChartSpec; this
module renders the actual Plotly JSON from real bundle data. The split
lets the LLM be creative without being able to fabricate values.

Supported kinds:
  - time_series_overlay: one trace per run, shared axes
  - scatter_correlation: cross-run scatter with regression + bootstrap CI
  - faceted_time_series: one subplot per run

The builder returns plain dicts shaped like Plotly figure JSON:
    {"data": [...], "layout": {...}, "spec": <original ChartSpec dump>}

The frontend feeds the {data, layout} pair into react-plotly.js. Spec is
included so the rendered card can show the rationale alongside the plot.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np

from fermdocs_characterize.toolkit.cross_run import (
    WEAK_N_THRESHOLD,
    compute_correlation,
)
from fermdocs_hypothesis.schema import ChartSpec

_log = logging.getLogger(__name__)


# Plotly's default 10-color qualitative palette (D3.schemeCategory10).
# We rotate through it so each run gets a stable color across the page.
_PALETTE = (
    "#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd",
    "#8c564b", "#e377c2", "#7f7f7f", "#bcbd22", "#17becf",
)

_HIGHLIGHT_COLOR = "#000000"  # solid black for highlighted runs


@dataclass(frozen=True)
class TrajectoryData:
    """Minimal carrier shape the builder needs.

    Decoupled from CharacterizationOutput.Trajectory so tests can
    construct synthetic charts without instantiating the full schema.
    The runner adapts real Trajectory objects into this in one line.
    """

    run_id: str
    variable: str
    time_h: list[float]
    values: list[float | None]


def build_chart(
    spec: ChartSpec,
    trajectories_by_run_var: dict[tuple[str, str], TrajectoryData],
) -> dict | None:
    """Render one ChartSpec to a Plotly figure dict.

    Returns None when the spec can't be satisfied (missing trajectories,
    too few points, etc). Caller skips None results rather than rendering
    an empty plot. Logging surfaces the reason for tracing.
    """
    if spec.kind == "time_series_overlay":
        return _build_time_series_overlay(spec, trajectories_by_run_var)
    if spec.kind == "scatter_correlation":
        return _build_scatter_correlation(spec, trajectories_by_run_var)
    if spec.kind == "faceted_time_series":
        return _build_faceted_time_series(spec, trajectories_by_run_var)
    _log.warning("unknown chart kind: %s", spec.kind)
    return None


def _color_for_run(run_id: str, idx: int, highlights: set[str]) -> str:
    if run_id in highlights:
        return _HIGHLIGHT_COLOR
    return _PALETTE[idx % len(_PALETTE)]


def _line_width(run_id: str, highlights: set[str]) -> int:
    return 3 if run_id in highlights else 2


def _annotations_for_layout(
    spec: ChartSpec,
    *,
    is_scatter: bool = False,
    scatter_xy_by_run: dict[str, tuple[float, float]] | None = None,
) -> list[dict]:
    """Convert ChartSpec.annotations into Plotly layout.annotations.

    For time-series: x = annotation.time_h, y = paper-relative.
    For scatter: x/y = the run's scatter point.
    """
    out: list[dict] = []
    for ann in spec.annotations:
        if is_scatter:
            if ann.run_id is None or scatter_xy_by_run is None:
                continue
            xy = scatter_xy_by_run.get(ann.run_id)
            if xy is None:
                continue
            x, y = xy
            out.append({
                "x": float(x), "y": float(y),
                "text": ann.text,
                "showarrow": True, "arrowhead": 2,
                "ax": 0, "ay": -30,
            })
        else:
            if ann.time_h is None:
                continue
            out.append({
                "x": float(ann.time_h),
                "y": 1.0, "yref": "paper",
                "text": ann.text,
                "showarrow": True, "arrowhead": 2,
                "ax": 0, "ay": -40,
            })
    return out


# -----------------------------------------------------------------------------
# time_series_overlay
# -----------------------------------------------------------------------------


def _build_time_series_overlay(
    spec: ChartSpec,
    trajectories_by_run_var: dict[tuple[str, str], TrajectoryData],
) -> dict | None:
    if len(spec.variables) != 1:
        _log.warning(
            "time_series_overlay needs exactly 1 variable, got %d",
            len(spec.variables),
        )
        return None
    var = spec.variables[0]
    runs = list(spec.runs) if spec.runs else sorted(
        {r for r, v in trajectories_by_run_var if v == var}
    )
    if not runs:
        return None
    highlights = set(spec.highlight_runs)
    traces: list[dict] = []
    for idx, run_id in enumerate(runs):
        td = trajectories_by_run_var.get((run_id, var))
        if td is None or not td.time_h:
            continue
        # Drop None values for clean line; preserve order.
        x_vals: list[float] = []
        y_vals: list[float] = []
        for t, v in zip(td.time_h, td.values):
            if v is None or not np.isfinite(v):
                continue
            x_vals.append(float(t))
            y_vals.append(float(v))
        if not x_vals:
            continue
        traces.append({
            "type": "scatter",
            "mode": "lines+markers",
            "name": run_id,
            "x": x_vals, "y": y_vals,
            "line": {
                "color": _color_for_run(run_id, idx, highlights),
                "width": _line_width(run_id, highlights),
            },
            "marker": {"size": 6},
        })
    if not traces:
        return None
    layout = {
        "title": {"text": spec.title},
        "xaxis": {"title": {"text": "time (h)"}},
        "yaxis": {"title": {"text": var}},
        "hovermode": "x unified",
        "annotations": _annotations_for_layout(spec),
        "margin": {"l": 60, "r": 30, "t": 50, "b": 50},
    }
    return {
        "data": traces,
        "layout": layout,
        "spec": spec.model_dump(mode="json"),
    }


# -----------------------------------------------------------------------------
# scatter_correlation
# -----------------------------------------------------------------------------


def _representative_value(td: TrajectoryData) -> float | None:
    """Final non-None value as the run's scalar for cross-run scatter."""
    for v in reversed(td.values):
        if v is not None and np.isfinite(v):
            return float(v)
    return None


def _build_scatter_correlation(
    spec: ChartSpec,
    trajectories_by_run_var: dict[tuple[str, str], TrajectoryData],
) -> dict | None:
    if len(spec.variables) != 2:
        _log.warning(
            "scatter_correlation needs exactly 2 variables (x, y), got %d",
            len(spec.variables),
        )
        return None
    x_var, y_var = spec.variables
    runs = list(spec.runs) if spec.runs else sorted(
        {r for r, v in trajectories_by_run_var if v in (x_var, y_var)}
    )
    if not runs:
        return None
    highlights = set(spec.highlight_runs)

    points: list[tuple[str, float, float]] = []
    for run_id in runs:
        x_td = trajectories_by_run_var.get((run_id, x_var))
        y_td = trajectories_by_run_var.get((run_id, y_var))
        if x_td is None or y_td is None:
            continue
        x_val = _representative_value(x_td)
        y_val = _representative_value(y_td)
        if x_val is None or y_val is None:
            continue
        points.append((run_id, x_val, y_val))

    if len(points) < 2:
        return None

    xs = [p[1] for p in points]
    ys = [p[2] for p in points]

    # Per-point trace so highlights get distinct color.
    traces: list[dict] = []
    for idx, (run_id, x, y) in enumerate(points):
        traces.append({
            "type": "scatter",
            "mode": "markers+text",
            "name": run_id,
            "x": [x], "y": [y],
            "text": [run_id],
            "textposition": "top right",
            "marker": {
                "size": 12 if run_id in highlights else 9,
                "color": _color_for_run(run_id, idx, highlights),
                "line": {"width": 1, "color": "#333"},
            },
        })

    # Correlation overlay when n >= 2; CI band only when n >= 4.
    n = len(xs)
    try:
        corr = compute_correlation(xs, ys)
    except ValueError:
        corr = None

    annotations = _annotations_for_layout(
        spec,
        is_scatter=True,
        scatter_xy_by_run={p[0]: (p[1], p[2]) for p in points},
    )

    if corr is not None and n >= 2:
        # OLS line
        slope, intercept = np.polyfit(xs, ys, 1)
        x_min, x_max = min(xs), max(xs)
        line_x = [x_min, x_max]
        line_y = [slope * x_min + intercept, slope * x_max + intercept]
        traces.append({
            "type": "scatter",
            "mode": "lines",
            "name": "OLS fit",
            "x": line_x, "y": line_y,
            "line": {"color": "#666", "width": 1, "dash": "dash"},
            "showlegend": False,
            "hoverinfo": "skip",
        })
        weak_tag = " ⚠️ weak n" if corr.weak_n_flag else ""
        annotations.append({
            "x": 0.02, "y": 0.98,
            "xref": "paper", "yref": "paper",
            "text": (
                f"r = {corr.r:.2f}, n = {corr.n}<br>"
                f"95% CI [{corr.ci_low:.2f}, {corr.ci_high:.2f}]{weak_tag}"
            ),
            "showarrow": False,
            "align": "left",
            "bgcolor": "rgba(255,255,255,0.85)",
            "bordercolor": "#999",
            "borderwidth": 1,
        })

    layout = {
        "title": {"text": spec.title},
        "xaxis": {"title": {"text": x_var}},
        "yaxis": {"title": {"text": y_var}},
        "hovermode": "closest",
        "annotations": annotations,
        "showlegend": False,
        "margin": {"l": 60, "r": 30, "t": 50, "b": 50},
    }
    return {
        "data": traces,
        "layout": layout,
        "spec": spec.model_dump(mode="json"),
    }


# -----------------------------------------------------------------------------
# faceted_time_series
# -----------------------------------------------------------------------------


def _build_faceted_time_series(
    spec: ChartSpec,
    trajectories_by_run_var: dict[tuple[str, str], TrajectoryData],
) -> dict | None:
    if len(spec.variables) != 1:
        _log.warning(
            "faceted_time_series needs exactly 1 variable, got %d",
            len(spec.variables),
        )
        return None
    var = spec.variables[0]
    runs = list(spec.runs) if spec.runs else sorted(
        {r for r, v in trajectories_by_run_var if v == var}
    )
    if not runs:
        return None
    highlights = set(spec.highlight_runs)

    # Plotly subplots via shared y-domains. Build one trace per run and
    # assign axis indices xN/yN; layout declares each axis range.
    n = len(runs)
    traces: list[dict] = []
    layout: dict = {
        "title": {"text": spec.title},
        "showlegend": False,
        "margin": {"l": 60, "r": 30, "t": 60, "b": 50},
        "annotations": [],
    }
    # Horizontal grid: each subplot gets 1/n of the x-axis.
    for idx, run_id in enumerate(runs):
        td = trajectories_by_run_var.get((run_id, var))
        if td is None or not td.time_h:
            continue
        x_vals: list[float] = []
        y_vals: list[float] = []
        for t, v in zip(td.time_h, td.values):
            if v is None or not np.isfinite(v):
                continue
            x_vals.append(float(t))
            y_vals.append(float(v))
        if not x_vals:
            continue
        x_axis = "x" if idx == 0 else f"x{idx + 1}"
        y_axis = "y" if idx == 0 else f"y{idx + 1}"
        traces.append({
            "type": "scatter",
            "mode": "lines+markers",
            "name": run_id,
            "x": x_vals, "y": y_vals,
            "xaxis": x_axis, "yaxis": y_axis,
            "line": {
                "color": _color_for_run(run_id, idx, highlights),
                "width": _line_width(run_id, highlights),
            },
            "marker": {"size": 5},
        })
        # Per-subplot axis layout. Domain divides [0, 1] into n columns.
        col_width = 1.0 / n
        x_start = idx * col_width
        x_end = x_start + col_width - 0.02 if idx < n - 1 else x_start + col_width
        layout[f"xaxis{idx + 1 if idx > 0 else ''}"] = {
            "domain": [x_start, x_end],
            "title": {"text": f"{run_id} — time (h)"},
        }
        layout[f"yaxis{idx + 1 if idx > 0 else ''}"] = {
            "anchor": x_axis,
            "title": {"text": var} if idx == 0 else {"text": ""},
        }
    if not traces:
        return None
    layout["annotations"].extend(_annotations_for_layout(spec))
    return {
        "data": traces,
        "layout": layout,
        "spec": spec.model_dump(mode="json"),
    }
