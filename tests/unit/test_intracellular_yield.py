"""P_INTRACELLULAR_YIELD: catalog metric for intracellular products.

Plan ref: yeast-intracellular-product branch.

Generic metric covering carotenoids, lipids, terpenoids, intracellular
recombinant protein, vitamin K2, polyhydroxyalkanoates. Routed via
ProcessFamilyConfig.intracellular_product_variable so the operator names
the column once in YAML; the toolkit + adapter are organism-agnostic.

Covers:
  1. compute_intracellular_yield happy path — monotonic accumulation
  2. compute_intracellular_yield with peak-then-decline (carotenoid
     white-cells signature)
  3. monotonic yield → decline=0, is_declining=False
  4. < 2 finite points → raises ValueError
  5. paired biomass → final_volumetric_yield computed
  6. no biomass → final_volumetric_yield is None
  7. ProcessFamilyConfig.intracellular_product_variable defaults to
     None on every existing family entry (back-compat invariant)
  8. yeast_intracellular_product_fedbatch family loads from YAML
  9. Adapter routes via family.intracellular_product_variable; missing
     variable in bundle → [CONFIG_MISMATCH] data_gap
 10. Adapter on family WITHOUT the field (penicillin, SCP) → None
     (silent skip, no config_mismatch)
 11. End-to-end through the catalog runner on a synthetic carotenoid
     bundle → metric fires with correct values for both runs
 12. Validator bound: yield > 500 mg/g DCW → data_gap with violation
 13. **REGRESSION**: penicillin bundle still produces zero
     intracellular_yield findings (back-compat — we don't accidentally
     fire P_INTRACELLULAR_YIELD on penicillin runs)
"""

from __future__ import annotations

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
from fermdocs_characterize.agents.finding_validator import validate_finding
from fermdocs_characterize.schema import (
    DataQuality,
    EvidenceStrength,
    ExtractedVia,
    Finding,
    FindingType,
    Severity,
    Tier,
    Trajectory,
)
from fermdocs_characterize.toolkit.products import (
    compute_intracellular_yield,
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
        unit="mg/g",
        quality=1.0,
        data_quality=DataQuality(pct_missing=0.0, pct_imputed=0.0, pct_real=1.0),
    )


# ---------- 1. toolkit happy path: monotonic accumulation ----------


def test_compute_intracellular_yield_monotonic() -> None:
    """Astaxanthin-style accumulation: 0 → 12 mg/g over the run."""
    times = [0.0, 24.0, 48.0, 72.0, 96.0, 120.0]
    yields = [0.0, 1.5, 4.0, 7.0, 10.0, 12.0]
    result = compute_intracellular_yield(times, yields)

    assert result.final_yield_mg_per_g_dcw == pytest.approx(12.0)
    assert result.peak_yield_mg_per_g_dcw == pytest.approx(12.0)
    assert result.t_peak_h == 120.0
    assert result.yield_decline_after_peak == 0.0
    assert result.is_yield_declining is False
    assert result.final_volumetric_yield_mg_per_l is None


# ---------- 2. peak-then-decline (the carotenoid white-cells case) ----------


def test_compute_intracellular_yield_decline_after_peak() -> None:
    """Carotenoid that peaks at 96h then drops by 30% — the
    'white cells' signature observed visually in carotenoid runs."""
    times = [0.0, 24.0, 48.0, 72.0, 96.0, 120.0]
    yields = [0.0, 2.0, 6.0, 9.0, 12.0, 8.4]  # peak 12 at 96, drop to 8.4
    result = compute_intracellular_yield(times, yields)

    assert result.peak_yield_mg_per_g_dcw == pytest.approx(12.0)
    assert result.final_yield_mg_per_g_dcw == pytest.approx(8.4)
    assert result.t_peak_h == 96.0
    assert result.yield_decline_after_peak == pytest.approx(0.3, rel=0.05)
    assert result.is_yield_declining is True


# ---------- 3. monotonic = no decline flag ----------


def test_compute_intracellular_yield_zero_decline_when_held() -> None:
    times = [0.0, 24.0, 48.0]
    yields = [1.0, 5.0, 5.0]  # monotonic non-decreasing
    result = compute_intracellular_yield(times, yields)
    assert result.yield_decline_after_peak == 0.0
    assert result.is_yield_declining is False


# ---------- 4. < 2 finite points raises ----------


def test_compute_intracellular_yield_raises_too_few() -> None:
    with pytest.raises(ValueError):
        compute_intracellular_yield([0.0], [5.0])


# ---------- 5. paired biomass → volumetric yield computed ----------


def test_compute_intracellular_yield_with_biomass_volumetric() -> None:
    """With biomass paired on the same time grid, volumetric yield
    = specific yield × biomass at final timepoint."""
    times = [0.0, 48.0, 96.0]
    yields = [0.0, 5.0, 10.0]      # mg/g DCW
    biomass = [1.0, 12.0, 25.0]    # g/L
    result = compute_intracellular_yield(times, yields, biomass_g_l=biomass)
    # 10 mg/g × 25 g/L = 250 mg/L
    assert result.final_volumetric_yield_mg_per_l == pytest.approx(250.0)


# ---------- 6. no biomass → volumetric is None ----------


def test_compute_intracellular_yield_no_biomass() -> None:
    times = [0.0, 24.0, 48.0]
    yields = [0.0, 3.0, 7.0]
    result = compute_intracellular_yield(times, yields)
    assert result.final_volumetric_yield_mg_per_l is None


# ---------- 7. existing families have intracellular field None (back-compat) ----------


def test_existing_families_have_no_intracellular_field() -> None:
    """REGRESSION: every family entry that existed before this branch
    must still have intracellular_product_variable=None. Otherwise we'd
    accidentally fire P_INTRACELLULAR_YIELD on penicillin / SCP / etc."""
    families = load_process_families()
    for name in (
        "penicillin_fedbatch",
        "ecoli_recombinant_protein",
        "yeast_aerobic_fedbatch",
        "melanin_batch",
        UNKNOWN_FAMILY_NAME,
    ):
        assert families[name].intracellular_product_variable is None, (
            f"{name} unexpectedly declares intracellular_product_variable;"
            " this would route P_INTRACELLULAR_YIELD to a family that"
            " historically didn't compute it."
        )


# ---------- 8. new family loads correctly ----------


def test_yeast_intracellular_product_family_loads() -> None:
    config = lookup_family("yeast_intracellular_product_fedbatch")
    assert not config.is_unknown
    assert config.product_variable is None
    assert config.intracellular_product_variable == "product_mg_per_g_dcw"
    # Tier P excreted-product metrics will route to None and skip;
    # P_INTRACELLULAR_YIELD will route to product_mg_per_g_dcw and fire.


# ---------- 9. adapter config-mismatch when bundle lacks declared variable ----------


def test_adapter_config_mismatch_when_variable_missing() -> None:
    """Family declares intracellular_product_variable but the bundle
    has no such trajectory → [CONFIG_MISMATCH] data_gap with helpful
    message naming the YAML route + available variables."""
    runner = MetricCatalogRunner()
    bundle = _BundleView(
        characterization_id=CHAR_ID,
        run_ids=["RUN-1"],
        trajectories=[
            # Has biomass but NOT product_mg_per_g_dcw
            _trajectory(
                "RUN-1", "biomass_g_l",
                times=[0.0, 24.0, 48.0],
                values=[0.5, 5.0, 12.0],
            ),
        ],
        organism="S. cerevisiae",
        process_family="yeast_intracellular_product_fedbatch",
    )
    findings = runner.compute_all(bundle)
    intracellular = [
        f for f in findings
        if f.statistics.get("metric_id") == "P_INTRACELLULAR_YIELD"
    ]
    assert len(intracellular) == 1
    assert intracellular[0].statistics["pattern_kind"] == "config_mismatch"
    reason = intracellular[0].statistics["reason"]
    assert "yeast_intracellular_product_fedbatch" in reason
    assert "product_mg_per_g_dcw" in reason
    assert "biomass_g_l" in reason  # available list


# ---------- 10. families WITHOUT field skip silently ----------


def test_adapter_silent_skip_on_family_without_field() -> None:
    """REGRESSION: penicillin family doesn't declare
    intracellular_product_variable. Adapter returns None (precondition
    not met), which becomes a generic data_gap, NOT config_mismatch.
    Distinct outcomes for distinct causes."""
    runner = MetricCatalogRunner()
    bundle = _BundleView(
        characterization_id=CHAR_ID,
        run_ids=["RUN-1"],
        trajectories=[
            _trajectory(
                "RUN-1", "biomass_g_l",
                times=[0.0, 24.0, 48.0],
                values=[0.5, 5.0, 12.0],
            ),
        ],
        organism="Penicillium chrysogenum",
        process_family="penicillin_fedbatch",  # no intracellular field
    )
    findings = runner.compute_all(bundle)
    intracellular = [
        f for f in findings
        if f.statistics.get("metric_id") == "P_INTRACELLULAR_YIELD"
    ]
    assert len(intracellular) == 1
    # Not config_mismatch — nothing's mismatched, family just doesn't have it.
    assert intracellular[0].statistics["pattern_kind"] == "data_gap"
    assert "precondition not met" in intracellular[0].statistics["reason"]


# ---------- 11. end-to-end carotenoid bundle: metric fires correctly ----------


def test_carotenoid_bundle_metric_fires_with_correct_values() -> None:
    """Synthetic 2-run carotenoid bundle. Adapter routes via
    yeast_intracellular_product_fedbatch → product_mg_per_g_dcw.
    Verify both runs get a computed_metric finding with the right shape."""
    runner = MetricCatalogRunner()
    times = [0.0, 24.0, 48.0, 72.0, 96.0, 120.0]

    # RUN-1: peaks at 96h then declines (white-cells signature)
    bundle = _BundleView(
        characterization_id=CHAR_ID,
        run_ids=["RUN-1", "RUN-2"],
        trajectories=[
            _trajectory(
                "RUN-1", "product_mg_per_g_dcw",
                times=times, values=[0.0, 2.0, 6.0, 9.0, 12.0, 8.4],
            ),
            _trajectory(
                "RUN-1", "biomass_g_l",
                times=times, values=[1.0, 8.0, 15.0, 22.0, 25.0, 23.0],
            ),
            # RUN-2: monotonic accumulation, holds at peak
            _trajectory(
                "RUN-2", "product_mg_per_g_dcw",
                times=times, values=[0.0, 2.5, 7.0, 11.0, 14.0, 14.0],
            ),
            _trajectory(
                "RUN-2", "biomass_g_l",
                times=times, values=[1.0, 9.0, 17.0, 24.0, 27.0, 28.0],
            ),
        ],
        organism="S. cerevisiae",
        process_family="yeast_intracellular_product_fedbatch",
    )
    findings = runner.compute_all(bundle)
    by_run = {
        f.run_ids[0]: f.statistics
        for f in findings
        if f.statistics.get("metric_id") == "P_INTRACELLULAR_YIELD"
        and f.statistics.get("pattern_kind") == "computed_metric"
    }
    # Both runs computed (the asymmetric-extraction regression invariant)
    assert "RUN-1" in by_run
    assert "RUN-2" in by_run

    # RUN-1 lysis signature
    r1 = by_run["RUN-1"]
    assert r1["peak_yield_mg_per_g_dcw"] == pytest.approx(12.0)
    assert r1["final_yield_mg_per_g_dcw"] == pytest.approx(8.4)
    assert r1["is_yield_declining"] is True

    # RUN-2 holds
    r2 = by_run["RUN-2"]
    assert r2["peak_yield_mg_per_g_dcw"] == pytest.approx(14.0)
    assert r2["final_yield_mg_per_g_dcw"] == pytest.approx(14.0)
    assert r2["is_yield_declining"] is False

    # Volumetric yield computed when biomass is paired
    assert "final_volumetric_yield_mg_per_l" in r1
    assert r1["final_volumetric_yield_mg_per_l"] == pytest.approx(
        8.4 * 23.0  # final_yield × biomass at final
    )


# ---------- 12. validator catches non-physical yield ----------


def test_validator_rejects_yield_above_500_mg_per_g() -> None:
    """Non-physical yield (units mismatch, fabricated data) → data_gap
    with reason naming the violation. 500 mg/g is the ceiling; carotenoid
    runs hit at most 50 mg/g, lipids 200 mg/g, intracellular protein
    might approach 300 mg/g. 1500 mg/g is units-bug territory."""
    bad_finding = Finding(
        finding_id=f"{CHAR_ID}:F-0001",
        type=FindingType.KINETIC_ANOMALY,
        severity=Severity.MINOR,
        tier=Tier.P,
        summary="intracellular yield computed",
        confidence=0.85,
        extracted_via=ExtractedVia.DETERMINISTIC,
        evidence_strength=EvidenceStrength(
            n_observations=10, n_independent_runs=1
        ),
        evidence_observation_ids=["obs-1"],
        variables_involved=["product_mg_per_g_dcw"],
        run_ids=["RUN-1"],
        statistics={
            "pattern_kind": "computed_metric",
            "metric_id": "P_INTRACELLULAR_YIELD",
            "tier": "P",
            "final_yield_mg_per_g_dcw": 1500.0,  # garbage
            "peak_yield_mg_per_g_dcw": 1500.0,
        },
    )
    validated = validate_finding(bad_finding)
    assert validated.statistics["pattern_kind"] == "data_gap"
    assert "final_yield_mg_per_g_dcw" in validated.statistics["reason"]
    assert validated.statistics["raw_invalid"]["final_yield_mg_per_g_dcw"] == 1500.0


# ---------- 13. REGRESSION: penicillin still emits no intracellular findings ----------


def test_penicillin_runs_emit_no_computed_intracellular_findings() -> None:
    """REGRESSION: a penicillin bundle (the canonical excreted-product
    case) must NEVER produce a `computed_metric` P_INTRACELLULAR_YIELD
    finding. The family doesn't declare intracellular_product_variable;
    the adapter returns None; runner emits a precondition data_gap.
    Pinning so we'd notice if a future YAML edit accidentally enables it."""
    runner = MetricCatalogRunner()
    times = [0.0, 24.0, 48.0, 96.0, 168.0, 224.0]
    bundle = _BundleView(
        characterization_id=CHAR_ID,
        run_ids=["RUN-1"],
        trajectories=[
            _trajectory(
                "RUN-1", "penicillin_g_l",
                times=times, values=[0.0, 1.0, 5.0, 18.0, 28.0, 30.4],
            ),
            _trajectory(
                "RUN-1", "biomass_g_l",
                times=times, values=[1.0, 5.0, 12.0, 22.0, 24.0, 24.0],
            ),
        ],
        organism="Penicillium chrysogenum",
        process_family="penicillin_fedbatch",
    )
    findings = runner.compute_all(bundle)
    intracellular = [
        f for f in findings
        if f.statistics.get("metric_id") == "P_INTRACELLULAR_YIELD"
        and f.statistics.get("pattern_kind") == "computed_metric"
    ]
    assert intracellular == [], (
        "Penicillium runs must not produce computed_metric"
        " P_INTRACELLULAR_YIELD findings; family doesn't declare"
        " intracellular_product_variable."
    )
