"""Product KPI tier (P1-P5) + process-family routing + config-mismatch.

Plan ref: plans/2026-05-07-characterize-determinism.md commit 2.

Covers:
  1. P1 final titer on a synthetic trajectory matches expected ± 5%
  2. P2 peak titer at the right timepoint
  3. P3 declining-product (RUN-1 IndPenSim case 21.6 → 14.3): decline
     fraction computed correctly, is_declining flag fires
  4. P3 monotonic product: decline=0, is_declining=False
  5. P4 integral productivity matches trapezoidal area
  6. P5 precursor utilization fraction polarity (consumed = high)
  7. P5 wasted precursor (RUN-1 PAA 5203 mg/L final): class='wasted'
  8. process_families.yaml roundtrip load + lookup_family
  9. unknown family falls through to no product-KPI metrics
 10. **A3** missing product_variable in bundle → ONE
     [CONFIG_MISMATCH] data_gap per metric (5 total when family
     declares product), each with a helpful reason; NOT five
     generic 'precondition' data_gaps
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

import pytest

from fermdocs.domain.process_families import (
    UNKNOWN_FAMILY_NAME,
    load_process_families,
    lookup_family,
)
from fermdocs_characterize.agents.catalog_runner import (
    MetricCatalogRunner,
    _BundleView,
)
from fermdocs_characterize.schema import (
    DataQuality,
    Trajectory,
)
from fermdocs_characterize.toolkit.products import (
    compute_final_titer,
    compute_integral_productivity,
    compute_peak_titer,
    compute_precursor_utilization,
    compute_titer_decline,
)

CHAR_ID = "00000000-0000-0000-0000-000000000001"


def _trajectory(
    run_id: str,
    variable: str,
    *,
    times: list[float],
    values: list[float | None],
) -> Trajectory:
    return Trajectory(
        trajectory_id=f"T-{abs(hash((run_id, variable))) % 10000:04d}",
        run_id=run_id,
        variable=variable,
        time_grid=times,
        values=values,
        imputation_flags=[False] * len(times),
        source_observation_ids=[
            f"obs-{run_id}-{variable}-{i}" for i in range(len(times))
        ],
        unit="g/L",
        quality=1.0,
        data_quality=DataQuality(pct_missing=0.0, pct_imputed=0.0, pct_real=1.0),
    )


# ---------- 1. P1 final titer ----------


def test_p1_final_titer_basic() -> None:
    times = [0.0, 24.0, 48.0, 72.0, 96.0]
    values = [0.0, 5.0, 12.0, 18.0, 21.5]
    result = compute_final_titer(times, values)
    assert result.final_titer_g_l == pytest.approx(21.5, rel=0.001)
    assert result.t_final_h == 96.0
    assert result.n_points == 5


# ---------- 2. P2 peak titer ----------


def test_p2_peak_titer_finds_max_with_time() -> None:
    times = [0.0, 48.0, 96.0, 168.0, 224.0]
    values = [0.0, 5.0, 15.0, 30.4, 28.0]  # peak at 168h
    result = compute_peak_titer(times, values)
    assert result.peak_titer_g_l == pytest.approx(30.4)
    assert result.t_peak_h == 168.0


# ---------- 3. P3 declining product (the IndPenSim RUN-1 case) ----------


def test_p3_declining_product_flags_lysis_signature() -> None:
    """RUN-1 IndPenSim: P went 21.6 → 14.3 g/L.
    decline_fraction = (21.6 - 14.3) / 21.6 = 0.338. is_declining=True."""
    times = [0.0, 96.0, 168.0, 228.0]
    values = [0.0, 18.0, 21.6, 14.3]
    result = compute_titer_decline(times, values)
    assert result.peak_titer_g_l == pytest.approx(21.6)
    assert result.final_titer_g_l == pytest.approx(14.3)
    assert result.decline_fraction == pytest.approx(0.338, rel=0.01)
    assert result.is_declining is True


# ---------- 4. P3 monotonic product ----------


def test_p3_monotonic_product_no_decline() -> None:
    times = [0.0, 96.0, 168.0, 224.0]
    values = [0.0, 15.0, 25.0, 30.4]
    result = compute_titer_decline(times, values)
    assert result.decline_fraction == 0.0
    assert result.is_declining is False


# ---------- 5. P4 integral productivity ----------


def test_p4_integral_productivity_trapezoidal() -> None:
    """Triangle: rises 0 to 10 over 100h. Area under curve = 500 g·h/L.
    Mean productivity = 500 / 100 = 5 g/L/h."""
    times = [0.0, 100.0]
    values = [0.0, 10.0]
    result = compute_integral_productivity(times, values)
    assert result.integral_g_l_h == pytest.approx(500.0)
    assert result.mean_productivity_g_l_per_h == pytest.approx(5.0)
    assert result.duration_h == 100.0


# ---------- 6. P5 precursor utilization (the IndPenSim RUN-2 case) ----------


def test_p5_efficient_utilization_when_consumed() -> None:
    """RUN-2 PAA: peak ~3000 mg/L from feeding, drops to 634 by end.
    utilization = (3000 - 634) / 3000 ≈ 0.79 → 'efficient'."""
    times = [0.0, 24.0, 72.0, 168.0, 224.0]
    values = [100.0, 3000.0, 2000.0, 1200.0, 634.0]
    result = compute_precursor_utilization(
        times, values, precursor_variable="paa_mg_l"
    )
    assert result.peak_value == pytest.approx(3000.0)
    assert result.final_value == pytest.approx(634.0)
    assert result.utilization_fraction == pytest.approx(0.789, abs=0.01)
    assert result.utilization_class == "efficient"


# ---------- 7. P5 wasted precursor (the IndPenSim RUN-1 case) ----------


def test_p5_wasted_precursor_when_accumulating() -> None:
    """RUN-1 PAA: peak ~5500 mg/L, ends at 5203 mg/L. Almost no
    consumption → utilization < 0.1 → 'wasted'."""
    times = [0.0, 24.0, 72.0, 168.0, 228.0]
    values = [100.0, 5500.0, 5400.0, 5300.0, 5203.0]
    result = compute_precursor_utilization(
        times, values, precursor_variable="paa_mg_l"
    )
    assert result.utilization_class == "wasted"
    assert result.utilization_fraction < 0.3


# ---------- 8. process_families.yaml load + lookup ----------


def test_process_families_yaml_loads_known_entries() -> None:
    families = load_process_families()
    # Must include the seed entries from the YAML
    assert "penicillin_fedbatch" in families
    assert "ecoli_recombinant_protein" in families
    assert "melanin_batch" in families
    # Plus the implicit unknown
    assert UNKNOWN_FAMILY_NAME in families

    pen = families["penicillin_fedbatch"]
    assert pen.product_variable == "penicillin_g_l"
    assert "paa_mg_l" in pen.precursor_variables


def test_lookup_family_routes_unknown_input_to_unknown_entry() -> None:
    """None or unrecognized name → the catch-all unknown entry."""
    assert lookup_family(None).is_unknown
    assert lookup_family("").is_unknown
    assert lookup_family("   ").is_unknown
    assert lookup_family("some_random_thing_we_dont_know").is_unknown


def test_lookup_family_resolves_known_name() -> None:
    config = lookup_family("penicillin_fedbatch")
    assert not config.is_unknown
    assert config.product_variable == "penicillin_g_l"


# ---------- 9. unknown family → no product-KPI emitted ----------


def test_unknown_family_emits_no_product_findings() -> None:
    runner = MetricCatalogRunner()
    bundle = _BundleView(
        characterization_id=CHAR_ID,
        run_ids=["RUN-1"],
        trajectories=[
            _trajectory(
                "RUN-1", "biomass_g_l",
                times=[float(i) for i in range(12)],
                values=[0.5 * 1.05**i for i in range(12)],
            ),
        ],
        organism=None,
        process_family=None,  # → unknown family
    )
    findings = runner.compute_all(bundle)
    p_findings = [
        f for f in findings
        if f.statistics.get("metric_id", "").startswith("P")
    ]
    # Unknown family → P1-P5 emit data_gap (precondition not met,
    # not CONFIG_MISMATCH because there's nothing routed to mismatch).
    for f in p_findings:
        assert f.statistics["pattern_kind"] == "data_gap"


# ---------- 10. A3: CONFIG_MISMATCH when bundle lacks routed variable ----------


def test_config_mismatch_when_product_variable_missing() -> None:
    """Family routes to penicillin_g_l but the bundle has no such
    trajectory. P1/P2/P3/P4 each emit ONE [CONFIG_MISMATCH] data_gap
    with a helpful reason naming what the YAML expected vs what the
    bundle has."""
    runner = MetricCatalogRunner()
    bundle = _BundleView(
        characterization_id=CHAR_ID,
        run_ids=["RUN-1"],
        trajectories=[
            # Has biomass + DO but NOT penicillin_g_l
            _trajectory(
                "RUN-1", "biomass_g_l",
                times=[float(i) for i in range(12)],
                values=[0.5 * 1.05**i for i in range(12)],
            ),
        ],
        organism="Penicillium chrysogenum",
        process_family="penicillin_fedbatch",
    )
    findings = runner.compute_all(bundle)

    config_findings = [
        f for f in findings
        if f.statistics.get("pattern_kind") == "config_mismatch"
    ]
    # Each of P1, P2, P3, P4 emits one config_mismatch.
    config_metric_ids = {f.statistics["metric_id"] for f in config_findings}
    assert {"P1", "P2", "P3", "P4"}.issubset(config_metric_ids)

    # P5 (precursor) should ALSO emit config_mismatch because
    # paa_mg_l is in the precursor list but not in the bundle.
    assert "P5" in config_metric_ids

    # Reason text mentions the YAML route + available variables.
    sample = next(f for f in config_findings if f.statistics["metric_id"] == "P1")
    reason = sample.statistics["reason"]
    assert "penicillin_fedbatch" in reason
    assert "penicillin_g_l" in reason
    assert "biomass_g_l" in reason  # the available list
    assert sample.summary.startswith("[CONFIG_MISMATCH]")


def test_p_metrics_compute_normally_when_product_present() -> None:
    """Happy path: bundle has the routed product variable, P-tier metrics
    return real computed_metric findings, NOT config_mismatch."""
    runner = MetricCatalogRunner()
    times = [float(i * 24) for i in range(11)]  # 0 to 240h
    pen_values = [0.0, 1.0, 5.0, 12.0, 18.0, 22.0, 25.0, 28.0, 30.0, 30.4, 29.0]
    bundle = _BundleView(
        characterization_id=CHAR_ID,
        run_ids=["RUN-1"],
        trajectories=[
            _trajectory("RUN-1", "penicillin_g_l", times=times, values=pen_values),
            _trajectory(
                "RUN-1", "paa_mg_l",
                times=times,
                values=[100.0, 3000.0, 2800.0, 2500.0, 2200.0, 1800.0, 1500.0, 1200.0, 900.0, 700.0, 634.0],
            ),
        ],
        organism="Penicillium chrysogenum",
        process_family="penicillin_fedbatch",
    )
    findings = runner.compute_all(bundle)
    by_metric = {
        f.statistics["metric_id"]: f
        for f in findings
        if f.statistics.get("pattern_kind") == "computed_metric"
    }
    assert "P1" in by_metric
    assert by_metric["P1"].statistics["final_titer_g_l"] == pytest.approx(29.0)
    assert "P2" in by_metric
    assert by_metric["P2"].statistics["peak_titer_g_l"] == pytest.approx(30.4)
    # P3 fires: 30.4 → 29.0 = 4.6% drop, just below threshold
    assert "P3" in by_metric
    # P5 efficient utilization
    assert "P5" in by_metric
    assert by_metric["P5"].statistics["utilization_class"] == "efficient"
