"""MetricCatalogRunner: deterministic per-run catalog dispatch.

Plan ref: plans/2026-05-07-characterize-determinism.md commit 1.

Covers:
  1. single-run bundle emits all applicable per-run metrics
  2. multi-run bundle emits N×M per-run findings (the IndPenSim fix)
  3. missing input variable → data_gap
  4. toolkit raise → data_gap with tool-error reason
  5. idempotent re-run produces identical findings (Q2)
  6. cross-run metric (A19) iterates ONCE not N times
  7. organism-required entries gracefully gap when adapter missing
  8. run_ids on emitted Finding correctly scoped
  9. **A2** pre-flight import failure aborts loud with module name
 10. empty bundle.run_ids → empty findings list
 11. **Q1** applicable_metric_run_pairs helper used by both runner and
     symmetry check (commit 5) returns same pairs as runner emits
"""

from __future__ import annotations

from unittest.mock import patch

import pytest

from fermdocs_characterize.agents.catalog_runner import (
    CROSS_RUN_METRIC_IDS,
    MetricCatalogRunner,
    _BundleView,
    applicable_metric_run_pairs,
)
from fermdocs_characterize.schema import (
    DataQuality,
    ExtractedVia,
    FindingType,
    Trajectory,
)


CHAR_ID = "00000000-0000-0000-0000-000000000001"


def _trajectory(
    run_id: str,
    variable: str,
    *,
    times: list[float],
    values: list[float | None],
    traj_id: str | None = None,
) -> Trajectory:
    """Tight Trajectory factory for tests — trajectory_id matches the
    schema's `T-NNNN` shape."""
    if traj_id is None:
        traj_id = f"T-{abs(hash((run_id, variable))) % 10000:04d}"
    return Trajectory(
        trajectory_id=traj_id,
        run_id=run_id,
        variable=variable,
        time_grid=times,
        values=values,
        imputation_flags=[False] * len(times),
        source_observation_ids=[f"obs-{run_id}-{variable}-{i}" for i in range(len(times))],
        unit="g/L",  # not enforced for tests; toolkit reads values numerically
        quality=1.0,
        data_quality=DataQuality(pct_missing=0.0, pct_imputed=0.0, pct_real=1.0),
    )


def _exponential_biomass(run_id: str, n: int = 12, mu: float = 0.05) -> Trajectory:
    """A clean exponential biomass trajectory good enough for A8/A9/A10/A11."""
    times = [float(i * 4) for i in range(n)]  # every 4h
    x0 = 0.5
    values: list[float | None] = []
    for t in times:
        values.append(x0 * (2.71828 ** (mu * t)))
    return _trajectory(run_id, "biomass_g_l", times=times, values=values)


def _aerobic_rq_pair(run_id: str) -> tuple[Trajectory, Trajectory]:
    """OUR + CER trajectories with mean RQ ~0.95 (clean aerobic)."""
    times = [float(i * 4) for i in range(8)]
    our = [10.0, 12.0, 15.0, 18.0, 17.0, 14.0, 11.0, 9.0]
    cer = [9.5, 11.4, 14.3, 17.1, 16.2, 13.3, 10.5, 8.6]
    return (
        _trajectory(run_id, "our_mmol_per_l_per_h", times=times, values=our),
        _trajectory(run_id, "cer_mmol_per_l_per_h", times=times, values=cer),
    )


def _do_trajectory(run_id: str, low: bool = False) -> Trajectory:
    times = [float(i * 4) for i in range(8)]
    values = [10.0, 12.0, 15.0, 18.0, 14.0, 12.0, 10.0, 9.0] if low else (
        [60.0, 65.0, 70.0, 72.0, 68.0, 65.0, 62.0, 58.0]
    )
    return _trajectory(run_id, "dissolved_o2_mg_l", times=times, values=values)


def _bundle(
    run_ids: list[str], trajectories: list[Trajectory], organism: str | None = None
) -> _BundleView:
    return _BundleView(
        characterization_id=CHAR_ID,
        run_ids=run_ids,
        trajectories=trajectories,
        organism=organism,
        process_family=None,
    )


# ---------- 1. single-run bundle: applicable metrics emit ----------


def test_single_run_emits_applicable_metrics() -> None:
    runner = MetricCatalogRunner()
    bundle = _bundle(
        ["RUN-1"],
        [
            _exponential_biomass("RUN-1"),
            *_aerobic_rq_pair("RUN-1"),
            _do_trajectory("RUN-1"),
        ],
    )
    findings = runner.compute_all(bundle)
    metric_ids = {
        f.statistics["metric_id"] for f in findings
        if f.statistics.get("pattern_kind") == "computed_metric"
    }
    # A8/A9/A10/A11 from biomass; A14 from DO; B10 from OUR/CER. B6 needs
    # a byproduct trajectory we didn't supply → data_gap. B16 needs substrate
    # → data_gap.
    assert {"A8", "A9", "A10", "A11", "A14", "B10"}.issubset(metric_ids)


# ---------- 2. multi-run: N×M coverage (the IndPenSim fix) ----------


def test_multi_run_emits_per_run_findings() -> None:
    """This is the regression that motivates the whole branch:
    every per-run metric must produce a finding for EVERY run."""
    runner = MetricCatalogRunner()
    bundle = _bundle(
        ["RUN-1", "RUN-2"],
        [
            _exponential_biomass("RUN-1", mu=0.05),
            _exponential_biomass("RUN-2", mu=0.04),
            *_aerobic_rq_pair("RUN-1"),
            *_aerobic_rq_pair("RUN-2"),
            _do_trajectory("RUN-1"),
            _do_trajectory("RUN-2"),
        ],
    )
    findings = runner.compute_all(bundle)
    by_metric_run = {
        (f.statistics["metric_id"], tuple(f.run_ids))
        for f in findings
        if f.statistics.get("pattern_kind") == "computed_metric"
    }
    # The bug we're fixing: A8 must appear for BOTH runs, not just RUN-1.
    assert ("A8", ("RUN-1",)) in by_metric_run
    assert ("A8", ("RUN-2",)) in by_metric_run
    assert ("B10", ("RUN-1",)) in by_metric_run
    assert ("B10", ("RUN-2",)) in by_metric_run
    assert ("A14", ("RUN-1",)) in by_metric_run
    assert ("A14", ("RUN-2",)) in by_metric_run


# ---------- 3. missing input → data_gap ----------


def test_missing_variable_emits_data_gap() -> None:
    """With at least one trajectory present (so data_gaps can anchor
    to a real observation_id), missing-variable adapters emit data_gap
    findings. Bundles with NO trajectories at all produce zero findings
    — there's nothing to anchor to, and that's the bundle's problem,
    not the runner's. Tested separately in test_empty_run_ids."""
    runner = MetricCatalogRunner()
    # Provide a biomass trajectory so data_gaps from OTHER metrics can
    # anchor to its observation_ids; the OUR/CER/DO adapters still
    # return None and emit data_gap.
    bundle = _bundle(["RUN-1"], [_exponential_biomass("RUN-1")])
    findings = runner.compute_all(bundle)
    # B10 (needs OUR+CER) and A14 (needs DO) emit data_gap.
    data_gaps = [
        f for f in findings
        if f.statistics.get("pattern_kind") == "data_gap"
    ]
    assert data_gaps, "expected at least one data_gap (B10 / A14 missing inputs)"
    # Reason mentions precondition
    reasons = {f.statistics.get("reason", "") for f in data_gaps}
    assert any("precondition not met" in r for r in reasons)
    # Anchored observation_ids must come from the biomass trajectory.
    biomass_obs = bundle.trajectory("RUN-1", "biomass_g_l").source_observation_ids
    for f in data_gaps:
        assert all(oid in biomass_obs for oid in f.evidence_observation_ids), (
            "data_gap evidence_observation_ids must resolve through bundle"
        )


# ---------- 4. toolkit raise → data_gap with tool-error reason ----------


def test_toolkit_exception_becomes_tool_error_data_gap() -> None:
    runner = MetricCatalogRunner()
    bundle = _bundle(
        ["RUN-1"],
        [_exponential_biomass("RUN-1")],
    )
    # Patch compute_mu (used by A8) to raise.
    with patch(
        "fermdocs_characterize.agents.catalog_runner_adapters.compute_mu",
        side_effect=RuntimeError("simulated toolkit failure"),
    ):
        findings = runner.compute_all(bundle)

    a8_findings = [
        f for f in findings if f.statistics.get("metric_id") == "A8"
    ]
    assert len(a8_findings) == 1
    a8 = a8_findings[0]
    assert a8.statistics["pattern_kind"] == "data_gap"
    reason = a8.statistics.get("reason", "")
    assert "tool error" in reason
    assert "simulated toolkit failure" in reason


# ---------- 5. idempotent re-run (Q2) ----------


def test_runner_is_idempotent() -> None:
    """Running compute_all twice on the same bundle produces findings
    with identical statistics. Pins determinism as a foundation."""
    runner = MetricCatalogRunner()
    bundle = _bundle(
        ["RUN-1", "RUN-2"],
        [
            _exponential_biomass("RUN-1"),
            _exponential_biomass("RUN-2"),
            *_aerobic_rq_pair("RUN-1"),
            *_aerobic_rq_pair("RUN-2"),
        ],
    )
    a = runner.compute_all(bundle)
    b = runner.compute_all(bundle)
    # Same number of findings, same metric_ids and run_ids in same order.
    assert len(a) == len(b)
    for fa, fb in zip(a, b):
        assert fa.statistics["metric_id"] == fb.statistics["metric_id"]
        assert fa.run_ids == fb.run_ids
        assert fa.statistics["pattern_kind"] == fb.statistics["pattern_kind"]
        # Computed values match exactly (deterministic floating-point).
        for key in ("mu_max", "mean_rq", "frac_below_threshold"):
            if key in fa.statistics:
                assert fa.statistics[key] == fb.statistics[key], key


# ---------- 6. cross-run metric iterated once ----------


def test_cross_run_metric_iterates_once_not_per_run() -> None:
    """applicable_metric_run_pairs yields cross-run metrics ONCE per
    metric, not once per run. Even though A19 has no adapter today,
    the pair enumeration is what symmetry check will read; commit 5
    relies on this scoping."""
    bundle = _bundle(["RUN-1", "RUN-2", "RUN-3"], [])
    pairs = list(applicable_metric_run_pairs(bundle))
    cross_pairs = [(m, r) for m, r in pairs if m in CROSS_RUN_METRIC_IDS]
    # Each cross-run metric_id appears exactly once with run_id=None.
    by_metric: dict[str, int] = {}
    for m, r in cross_pairs:
        by_metric[m] = by_metric.get(m, 0) + 1
        assert r is None, f"cross-run pair {m} should have run_id=None, got {r}"
    for m, count in by_metric.items():
        assert count == 1, f"{m} iterated {count} times; expected 1"


# ---------- 7. organism-required entries fall through ----------


def test_organism_required_metrics_fall_through_when_no_adapter() -> None:
    """C-tier metrics need organism priors. They have no adapter in
    commit 1; the runner soft-skips them and they fall through to the
    LLM analyzer (existing path). No exception, no spurious findings."""
    runner = MetricCatalogRunner()
    bundle = _bundle(
        ["RUN-1"],
        [_exponential_biomass("RUN-1")],
        organism=None,
    )
    findings = runner.compute_all(bundle)
    metric_ids = {f.statistics["metric_id"] for f in findings}
    # No C-tier ID emitted from the runner.
    assert not any(m.startswith("C") for m in metric_ids)


# ---------- 8. run_ids on Finding scoped correctly ----------


def test_per_run_finding_run_ids_scoped_to_one_run() -> None:
    runner = MetricCatalogRunner()
    bundle = _bundle(
        ["RUN-1", "RUN-2"],
        [
            _exponential_biomass("RUN-1"),
            _exponential_biomass("RUN-2"),
        ],
    )
    findings = runner.compute_all(bundle)
    for f in findings:
        if f.statistics.get("pattern_kind") != "computed_metric":
            continue
        # Per-run metric: one run_id only.
        if f.statistics["metric_id"] not in CROSS_RUN_METRIC_IDS:
            assert len(f.run_ids) == 1, (
                f"{f.statistics['metric_id']} per-run finding has"
                f" run_ids={f.run_ids}; expected length 1"
            )


# ---------- 9. A2: pre-flight import failure aborts loud ----------


def test_preflight_import_failure_aborts_loud() -> None:
    """If a ready entry's toolkit_fn fails to import, runner construction
    raises with the failing module name, not a silent data_gap."""
    from fermdocs_characterize.agents import metric_catalog

    target_entry = next(metric_catalog.ready_entries().__iter__())
    target_id = target_entry.metric_id

    class _ExplodingEntry:
        metric_id = target_id
        toolkit_fn = target_entry.toolkit_fn
        tier = target_entry.tier
        required_inputs = target_entry.required_inputs
        status = "ready"

        def is_ready(self):
            return True

        def resolve_toolkit_fn(self):
            raise ImportError("simulated dep regression")

    fake_entries = [_ExplodingEntry()]

    # Patch the symbol the runner module imported, not the source module.
    with patch(
        "fermdocs_characterize.agents.catalog_runner.ready_entries",
        return_value=fake_entries,
    ):
        with pytest.raises(RuntimeError) as ei:
            MetricCatalogRunner()
    msg = str(ei.value)
    assert "pre-flight import failed" in msg
    assert target_id in msg
    assert "simulated dep regression" in msg


# ---------- 10. empty run_ids → empty findings ----------


def test_empty_run_ids_returns_empty() -> None:
    runner = MetricCatalogRunner()
    bundle = _bundle([], [])
    findings = runner.compute_all(bundle)
    # Cross-run metrics still iterate, but they have no adapters today —
    # they're soft-skipped. Per-run metrics produce zero findings
    # because the inner loop is over an empty list. Net: zero.
    assert findings == []


def test_data_gap_dropped_when_no_observation_anchor(tmp_path) -> None:
    """REGRESSION: bundle with run_ids but NO trajectories on those runs
    cannot anchor data_gap findings to a real observation_id. Runner
    must DROP the would-be-data_gap rather than emit a finding the
    validator will reject (the 'cites unknown observation_id
    deterministic-runner' bug)."""
    runner = MetricCatalogRunner()
    # run_id 'RUN-1' is declared but no Trajectory rows exist for it.
    bundle = _bundle(["RUN-1"], [])
    findings = runner.compute_all(bundle)
    # Should be EMPTY: every adapter returns None, every data_gap path
    # drops because there's no observation to anchor to. The dev sees
    # WARNING logs about the drops; bundle stays valid.
    assert findings == []


# ---------- 11. Q1: applicable_metric_run_pairs == runner emits ----------


def test_applicable_pairs_match_runner_findings() -> None:
    """The same enumeration used by the runner is what symmetry check
    (commit 5) will use. Pin: every (metric_id, run_id) the runner
    emits a finding for is also yielded by applicable_metric_run_pairs,
    AND vice versa for adapter-backed metrics."""
    runner = MetricCatalogRunner()
    bundle = _bundle(
        ["RUN-1", "RUN-2"],
        [
            _exponential_biomass("RUN-1"),
            _exponential_biomass("RUN-2"),
        ],
    )
    findings = runner.compute_all(bundle)
    emitted_pairs = {
        (f.statistics["metric_id"], f.run_ids[0] if f.run_ids else None)
        for f in findings
    }
    enumerated_pairs = set(applicable_metric_run_pairs(bundle))

    # Every emission corresponds to an enumerated pair.
    for pair in emitted_pairs:
        assert pair in enumerated_pairs, (
            f"emission {pair} not in enumeration; iteration shapes drift"
        )
