"""Deterministic Plotly figure builder tests.

Branch: charts-and-pdf-export.

The LLM emits ChartSpec; chart_builder renders the actual Plotly JSON
from real bundle data. These tests pin output shape per kind, missing-
data robustness, and the n/CI annotation on scatter charts (closes the
gap between commit-2's bootstrap CI and what users actually see).
"""

from __future__ import annotations

from fermdocs_hypothesis.chart_builder import (
    TrajectoryData,
    build_chart,
)
from fermdocs_hypothesis.schema import ChartAnnotation, ChartSpec


def _td(run_id: str, variable: str, n: int = 5) -> TrajectoryData:
    return TrajectoryData(
        run_id=run_id,
        variable=variable,
        time_h=[float(i * 6) for i in range(n)],
        values=[1.0 + 0.5 * i for i in range(n)],
    )


# ---------- time_series_overlay ----------


def test_time_series_overlay_basic_shape() -> None:
    spec = ChartSpec(
        kind="time_series_overlay",
        title="Biomass over time",
        rationale="all batches show the same growth phase ordering",
        runs=["RUN-1", "RUN-2", "RUN-3"],
        variables=["biomass_g_l"],
    )
    data = {
        ("RUN-1", "biomass_g_l"): _td("RUN-1", "biomass_g_l"),
        ("RUN-2", "biomass_g_l"): _td("RUN-2", "biomass_g_l"),
        ("RUN-3", "biomass_g_l"): _td("RUN-3", "biomass_g_l"),
    }
    fig = build_chart(spec, data)
    assert fig is not None
    assert len(fig["data"]) == 3
    assert all(t["type"] == "scatter" for t in fig["data"])
    assert fig["layout"]["title"]["text"] == "Biomass over time"
    assert fig["layout"]["xaxis"]["title"]["text"] == "time (h)"
    assert fig["layout"]["yaxis"]["title"]["text"] == "biomass_g_l"
    # Original spec preserved for the frontend rationale block.
    assert fig["spec"]["rationale"]


def test_time_series_overlay_highlight_run_gets_distinct_color() -> None:
    spec = ChartSpec(
        kind="time_series_overlay",
        title="x", rationale="x",
        runs=["RUN-1", "RUN-2"],
        variables=["biomass_g_l"],
        highlight_runs=["RUN-2"],
    )
    data = {
        ("RUN-1", "biomass_g_l"): _td("RUN-1", "biomass_g_l"),
        ("RUN-2", "biomass_g_l"): _td("RUN-2", "biomass_g_l"),
    }
    fig = build_chart(spec, data)
    by_run = {t["name"]: t for t in fig["data"]}
    assert by_run["RUN-2"]["line"]["color"] == "#000000"
    assert by_run["RUN-1"]["line"]["color"] != "#000000"
    assert by_run["RUN-2"]["line"]["width"] > by_run["RUN-1"]["line"]["width"]


def test_time_series_overlay_drops_none_values() -> None:
    """Imputation gaps mustn't appear as zero or NaN points."""
    td = TrajectoryData(
        run_id="RUN-1", variable="biomass_g_l",
        time_h=[0.0, 6.0, 12.0, 18.0, 24.0],
        values=[1.0, None, 3.0, None, 5.0],
    )
    spec = ChartSpec(
        kind="time_series_overlay",
        title="x", rationale="x",
        runs=["RUN-1"], variables=["biomass_g_l"],
    )
    fig = build_chart(spec, {("RUN-1", "biomass_g_l"): td})
    trace = fig["data"][0]
    assert len(trace["x"]) == 3
    assert trace["y"] == [1.0, 3.0, 5.0]


def test_time_series_overlay_returns_none_when_no_data() -> None:
    spec = ChartSpec(
        kind="time_series_overlay",
        title="x", rationale="x",
        runs=["RUN-1"], variables=["biomass_g_l"],
    )
    assert build_chart(spec, {}) is None


def test_time_series_overlay_renders_annotations() -> None:
    spec = ChartSpec(
        kind="time_series_overlay",
        title="x", rationale="x",
        runs=["RUN-1"], variables=["biomass_g_l"],
        annotations=[ChartAnnotation(text="feed start", time_h=12.0)],
    )
    fig = build_chart(
        spec, {("RUN-1", "biomass_g_l"): _td("RUN-1", "biomass_g_l")},
    )
    assert fig["layout"]["annotations"][0]["text"] == "feed start"
    assert fig["layout"]["annotations"][0]["x"] == 12.0


# ---------- scatter_correlation ----------


def test_scatter_correlation_basic_shape() -> None:
    spec = ChartSpec(
        kind="scatter_correlation",
        title="Final biomass vs min DO",
        rationale="r=-0.99 with n=6, robustness flagged",
        runs=[f"RUN-{i}" for i in range(1, 7)],
        variables=["min_do", "final_biomass"],
    )
    # Strong negative correlation across 6 runs.
    data: dict = {}
    for i in range(1, 7):
        run_id = f"RUN-{i}"
        # min_do trajectory — last value used as scalar
        data[(run_id, "min_do")] = TrajectoryData(
            run_id=run_id, variable="min_do",
            time_h=[0.0], values=[float(50 - 5 * i)],
        )
        data[(run_id, "final_biomass")] = TrajectoryData(
            run_id=run_id, variable="final_biomass",
            time_h=[0.0], values=[float(2 + 0.4 * i)],
        )
    fig = build_chart(spec, data)
    assert fig is not None
    # 6 point traces + 1 OLS line trace
    assert len(fig["data"]) == 7
    point_traces = [t for t in fig["data"] if t["mode"] == "markers+text"]
    assert len(point_traces) == 6
    ols = [t for t in fig["data"] if t["name"] == "OLS fit"]
    assert len(ols) == 1


def test_scatter_correlation_includes_n_and_ci_annotation() -> None:
    """The reviewer's gap: r=-0.99 must ship with n + CI on the chart."""
    spec = ChartSpec(
        kind="scatter_correlation",
        title="x", rationale="x",
        runs=[f"RUN-{i}" for i in range(1, 7)],
        variables=["a", "b"],
    )
    data: dict = {}
    for i in range(1, 7):
        run_id = f"RUN-{i}"
        data[(run_id, "a")] = TrajectoryData(
            run_id=run_id, variable="a", time_h=[0.0], values=[float(i)],
        )
        data[(run_id, "b")] = TrajectoryData(
            run_id=run_id, variable="b", time_h=[0.0], values=[float(2 * i)],
        )
    fig = build_chart(spec, data)
    ann_texts = " ".join(a["text"] for a in fig["layout"]["annotations"])
    assert "r = " in ann_texts
    assert "n = 6" in ann_texts
    assert "95% CI" in ann_texts
    # n=6 < WEAK_N_THRESHOLD=8, weak-n badge required.
    assert "weak n" in ann_texts.lower()


def test_scatter_correlation_no_weak_n_badge_when_n_above_threshold() -> None:
    spec = ChartSpec(
        kind="scatter_correlation",
        title="x", rationale="x",
        runs=[f"RUN-{i}" for i in range(1, 11)],
        variables=["a", "b"],
    )
    data: dict = {}
    for i in range(1, 11):
        run_id = f"RUN-{i}"
        data[(run_id, "a")] = TrajectoryData(
            run_id=run_id, variable="a", time_h=[0.0], values=[float(i)],
        )
        data[(run_id, "b")] = TrajectoryData(
            run_id=run_id, variable="b", time_h=[0.0], values=[float(2 * i)],
        )
    fig = build_chart(spec, data)
    ann_texts = " ".join(a["text"] for a in fig["layout"]["annotations"])
    assert "n = 10" in ann_texts
    assert "weak n" not in ann_texts.lower()


def test_scatter_correlation_returns_none_when_too_few_points() -> None:
    spec = ChartSpec(
        kind="scatter_correlation",
        title="x", rationale="x",
        runs=["RUN-1"], variables=["a", "b"],
    )
    data = {
        ("RUN-1", "a"): TrajectoryData(
            run_id="RUN-1", variable="a", time_h=[0.0], values=[1.0],
        ),
        ("RUN-1", "b"): TrajectoryData(
            run_id="RUN-1", variable="b", time_h=[0.0], values=[2.0],
        ),
    }
    assert build_chart(spec, data) is None


# ---------- faceted_time_series ----------


def test_faceted_time_series_one_subplot_per_run() -> None:
    spec = ChartSpec(
        kind="faceted_time_series",
        title="Biomass per batch",
        rationale="y-scales differ; faceting clarifies",
        runs=["RUN-1", "RUN-2", "RUN-3"],
        variables=["biomass_g_l"],
    )
    data = {
        (f"RUN-{i}", "biomass_g_l"): _td(f"RUN-{i}", "biomass_g_l")
        for i in (1, 2, 3)
    }
    fig = build_chart(spec, data)
    assert fig is not None
    assert len(fig["data"]) == 3
    # Each trace pinned to a different x-axis.
    axes = {t["xaxis"] for t in fig["data"]}
    assert axes == {"x", "x2", "x3"}


# ---------- robustness ----------


def test_unknown_kind_returns_none() -> None:
    """Pydantic validates kind, but defensive check in builder doesn't crash."""
    # ChartSpec rejects unknown kinds at construction; this test pins
    # the builder layer for safety. We bypass via model_construct.
    spec = ChartSpec.model_construct(
        kind="unsupported_kind",  # type: ignore[arg-type]
        title="x", rationale="x",
        runs=["RUN-1"], variables=["v"],
        highlight_runs=[],
        annotations=[],
    )
    assert build_chart(spec, {}) is None


def test_chart_specs_field_default_empty_on_hypothesis_full() -> None:
    """REGRESSION: legacy fixtures load without migration."""
    from fermdocs_diagnose.schema import ConfidenceBasis
    from fermdocs_hypothesis.schema import HypothesisFull

    h = HypothesisFull(
        hyp_id="H-0001",
        summary="x",
        facet_ids=["FCT-0001"],
        cited_finding_ids=["F-0001"],
        confidence=0.6,
        confidence_basis=ConfidenceBasis.SCHEMA_ONLY,
    )
    assert h.chart_specs == []


def test_chart_specs_field_default_empty_on_final_hypothesis() -> None:
    """REGRESSION: existing fixtures load without migration."""
    from fermdocs_diagnose.schema import ConfidenceBasis
    from fermdocs_hypothesis.schema import FinalHypothesis

    h = FinalHypothesis(
        hyp_id="H-0001",
        summary="x",
        facet_ids=["FCT-0001"],
        cited_finding_ids=["F-0001"],
        confidence=0.6,
        confidence_basis=ConfidenceBasis.SCHEMA_ONLY,
        critic_flag="green",
        judge_ruled_criticism_valid=False,
    )
    assert h.chart_specs == []
    assert h.plotly_charts == []
