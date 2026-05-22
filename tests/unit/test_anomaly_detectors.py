"""Generic anomaly detectors: instrument-change + h0-outlier.

Plan ref: plans/2026-05-07-rigour-and-actionability.md commit 3.

Review gap #2: Hitachi→LABMAN spectrophotometer swap, WCW 8× cohort
at h0. Both belong as deterministic findings, not LLM judgement.
"""

from __future__ import annotations

import pytest

from fermdocs_characterize.toolkit.anomalies import (
    H0Outlier,
    InstrumentChange,
    detect_h0_outliers,
    detect_instrument_changes,
)


# ---------- instrument change ----------


def test_instrument_change_fires_on_hitachi_to_labman_spectrophotometer() -> None:
    """The actual carotenoid case from the 7.5/10 review."""
    narratives = {
        "RUN-1": ["WCW measured on Hitachi spectrophotometer at 600nm."],
        "RUN-2": ["WCW measured on Hitachi spectrophotometer at 600nm."],
        "RUN-3": ["WCW measured on Hitachi spectrophotometer at 600nm."],
        "RUN-4": ["WCW measured on LABMAN spectrophotometer at 600nm."],
    }
    found = detect_instrument_changes(narratives)
    assert len(found) == 1
    change = found[0]
    assert change.instrument_kind == "spectrophotometer"
    assert change.instruments_by_run["RUN-1"] == "Hitachi"
    assert change.instruments_by_run["RUN-4"] == "LABMAN"


def test_no_instrument_change_when_all_runs_use_same_brand() -> None:
    narratives = {
        "RUN-1": ["DO probe: Mettler-Toledo InPro 6800."],
        "RUN-2": ["DO probe: Mettler-Toledo InPro 6800."],
    }
    assert detect_instrument_changes(narratives) == []


def test_no_change_when_only_one_run_present() -> None:
    """Single-run bundles can't have cross-run instrument differences."""
    assert detect_instrument_changes({"RUN-1": ["Hitachi spectrophotometer"]}) == []


def test_no_false_positive_when_keyword_absent() -> None:
    narratives = {
        "RUN-1": ["Run completed normally."],
        "RUN-2": ["Foam events recorded at 12h, 18h."],
    }
    assert detect_instrument_changes(narratives) == []


def test_detects_change_across_multiple_instrument_kinds() -> None:
    narratives = {
        "RUN-1": [
            "Hitachi spectrophotometer for WCW.",
            "DO probe: Hamilton VisiFerm.",
        ],
        "RUN-2": [
            "LABMAN spectrophotometer for WCW.",
            "DO probe: Hamilton VisiFerm.",
        ],
    }
    found = detect_instrument_changes(narratives)
    kinds = {c.instrument_kind for c in found}
    assert "spectrophotometer" in kinds
    # DO probe is unchanged — must not fire.
    assert "do probe" not in kinds


# ---------- h0 outlier ----------


def test_h0_outlier_fires_on_8x_wcw_anomaly() -> None:
    """Carotenoid case: Batch 4 WCW 8× the cohort median at h=0."""
    h0 = {
        "RUN-1": {"wcw_g_l": 1.0},
        "RUN-2": {"wcw_g_l": 1.1},
        "RUN-3": {"wcw_g_l": 0.9},
        "RUN-4": {"wcw_g_l": 8.0},
        "RUN-5": {"wcw_g_l": 1.05},
    }
    found = detect_h0_outliers(h0)
    assert len(found) == 1
    out = found[0]
    assert out.variable == "wcw_g_l"
    assert out.run_id == "RUN-4"
    assert out.mad_score > 3.0


def test_h0_outlier_no_fire_on_clean_cohort() -> None:
    """REGRESSION: penicillin-shaped clean bundle emits zero outliers."""
    h0 = {
        f"RUN-{i}": {"biomass_g_l": 1.0 + 0.05 * i, "substrate_g_l": 50.0 + i}
        for i in range(5)
    }
    assert detect_h0_outliers(h0) == []


def test_h0_outlier_skips_when_fewer_than_three_runs() -> None:
    """Need 3+ runs for a meaningful cohort median."""
    h0 = {
        "RUN-1": {"v": 1.0},
        "RUN-2": {"v": 100.0},
    }
    assert detect_h0_outliers(h0) == []


def test_h0_outlier_skips_constant_cohort() -> None:
    """Zero MAD → any deviation is infinite. Skip rather than fire on every diff."""
    h0 = {
        "RUN-1": {"v": 1.0},
        "RUN-2": {"v": 1.0},
        "RUN-3": {"v": 1.0},
        "RUN-4": {"v": 1.0},
    }
    assert detect_h0_outliers(h0) == []


def test_h0_outlier_threshold_is_configurable() -> None:
    h0 = {f"RUN-{i}": {"v": float(i)} for i in range(5)}
    h0["RUN-OUT"] = {"v": 12.0}
    # Default threshold (3.0) should fire.
    assert detect_h0_outliers(h0) != []
    # Very loose threshold should not.
    assert detect_h0_outliers(h0, mad_threshold=100.0) == []


def test_h0_outlier_drops_nan() -> None:
    h0 = {
        "RUN-1": {"v": 1.0},
        "RUN-2": {"v": 1.0},
        "RUN-3": {"v": 1.0},
        "RUN-4": {"v": float("nan")},
    }
    # NaN dropped; cohort is constant; no outlier emitted.
    assert detect_h0_outliers(h0) == []


def test_returns_dataclass_with_full_provenance() -> None:
    h0 = {f"RUN-{i}": {"v": 1.0 + 0.01 * i} for i in range(6)}
    h0["RUN-OUT"] = {"v": 20.0}
    found = detect_h0_outliers(h0)
    assert len(found) == 1
    out = found[0]
    assert isinstance(out, H0Outlier)
    assert out.run_value == 20.0
    assert out.cohort_median > 0
    assert out.mad > 0


# ---------- empty / edge ----------


def test_empty_inputs() -> None:
    assert detect_instrument_changes({}) == []
    assert detect_h0_outliers({}) == []


# ---------- scale change ----------


def test_scale_change_fires_on_carotenoid_volume_drift() -> None:
    """Reviewer A1: 2.5-3 L → 3.5-4 L → 1 L drift."""
    from fermdocs_characterize.toolkit.anomalies import detect_scale_changes

    volumes = {
        "RUN-1": 2.5,
        "RUN-2": 2.7,
        "RUN-3": 3.0,
        "RUN-4": 3.5,
        "RUN-5": 1.0,
        "RUN-6": 1.0,
    }
    found = detect_scale_changes(volumes)
    assert len(found) == 1
    sc = found[0]
    assert sc.min_l == 1.0
    assert sc.max_l == 3.5
    assert sc.relative_spread > 1.0  # 250% spread


def test_scale_change_no_fire_within_threshold() -> None:
    """REGRESSION: 2.5 L → 2.6 L (4%) is normal variation, not a scale change."""
    from fermdocs_characterize.toolkit.anomalies import detect_scale_changes

    volumes = {f"RUN-{i}": 2.5 + 0.02 * i for i in range(5)}
    assert detect_scale_changes(volumes) == []


def test_scale_change_drops_none_volumes() -> None:
    from fermdocs_characterize.toolkit.anomalies import detect_scale_changes

    volumes = {"RUN-1": 2.5, "RUN-2": None, "RUN-3": 3.5}
    found = detect_scale_changes(volumes)
    assert len(found) == 1
    assert "RUN-2" not in found[0].volumes_by_run


def test_scale_change_skips_when_fewer_than_two_valid_runs() -> None:
    from fermdocs_characterize.toolkit.anomalies import detect_scale_changes

    assert detect_scale_changes({"RUN-1": 2.5}) == []
    assert detect_scale_changes({"RUN-1": 2.5, "RUN-2": None}) == []
    assert detect_scale_changes({}) == []


def test_scale_change_threshold_is_configurable() -> None:
    from fermdocs_characterize.toolkit.anomalies import detect_scale_changes

    volumes = {"RUN-1": 2.5, "RUN-2": 2.7}  # 8% spread
    assert detect_scale_changes(volumes) == []  # 10% default
    found = detect_scale_changes(volumes, relative_threshold=0.05)  # 5% loose
    assert len(found) == 1


# ---------- bioreactor change ----------


def test_bioreactor_change_fires_on_named_swap() -> None:
    """Reviewer A1: BIOREACTOR_A on RUN-1..3, BIOREACTOR_B on RUN-4..6."""
    from fermdocs_characterize.toolkit.anomalies import detect_bioreactor_changes

    reactors = {
        "RUN-1": "BIOREACTOR_A",
        "RUN-2": "BIOREACTOR_A",
        "RUN-3": "BIOREACTOR_A",
        "RUN-4": "BIOREACTOR_B",
        "RUN-5": "BIOREACTOR_B",
        "RUN-6": "BIOREACTOR_B",
    }
    found = detect_bioreactor_changes(reactors)
    assert len(found) == 1
    assert set(found[0].reactors_by_run.values()) == {"BIOREACTOR_A", "BIOREACTOR_B"}


def test_bioreactor_change_no_fire_when_all_same() -> None:
    from fermdocs_characterize.toolkit.anomalies import detect_bioreactor_changes

    reactors = {f"RUN-{i}": "BIOREACTOR_A" for i in range(5)}
    assert detect_bioreactor_changes(reactors) == []


def test_bioreactor_change_normalises_case_and_whitespace() -> None:
    """'BIOREACTOR_A' and 'Bioreactor A' are the same vessel — don't fire."""
    from fermdocs_characterize.toolkit.anomalies import detect_bioreactor_changes

    reactors = {
        "RUN-1": "BIOREACTOR_A",
        "RUN-2": "  Bioreactor A ",
        "RUN-3": "bioreactor a",
    }
    assert detect_bioreactor_changes(reactors) == []


def test_bioreactor_change_drops_none_and_empty() -> None:
    from fermdocs_characterize.toolkit.anomalies import detect_bioreactor_changes

    reactors = {
        "RUN-1": "BIOREACTOR_A",
        "RUN-2": None,
        "RUN-3": "",
        "RUN-4": "BIOREACTOR_B",
    }
    found = detect_bioreactor_changes(reactors)
    assert len(found) == 1
    assert set(found[0].reactors_by_run.keys()) == {"RUN-1", "RUN-4"}


# ---------- header inconsistency ----------


def test_header_inconsistency_fires_on_wcw_unit_drift() -> None:
    """Reviewer A1: 'WCW (mg/3 mL)' on 5 runs, 'Wet cell weight (mg)' on 1."""
    from fermdocs_characterize.toolkit.anomalies import detect_header_inconsistencies

    headers = {
        "RUN-1": {"wcw_g_l": "WCW (mg/3 mL)"},
        "RUN-2": {"wcw_g_l": "WCW (mg/3 mL)"},
        "RUN-3": {"wcw_g_l": "WCW (mg/3 mL)"},
        "RUN-4": {"wcw_g_l": "WCW (mg/3 mL)"},
        "RUN-5": {"wcw_g_l": "WCW (mg/3 mL)"},
        "RUN-6": {"wcw_g_l": "Wet cell weight (mg)"},
    }
    found = detect_header_inconsistencies(headers)
    assert len(found) == 1
    assert found[0].variable == "wcw_g_l"
    assert found[0].raw_headers_by_run["RUN-6"] == "Wet cell weight (mg)"


def test_header_inconsistency_ignores_whitespace_and_case() -> None:
    """OCR-style noise mustn't fire."""
    from fermdocs_characterize.toolkit.anomalies import detect_header_inconsistencies

    headers = {
        "RUN-1": {"wcw_g_l": "WCW (mg/3 mL)"},
        "RUN-2": {"wcw_g_l": " wcw  (mg/3 mL) "},
        "RUN-3": {"wcw_g_l": "WCW  (mg/3 mL)"},
    }
    assert detect_header_inconsistencies(headers) == []


def test_header_inconsistency_no_fire_when_uniform() -> None:
    """REGRESSION: clean penicillin bundle."""
    from fermdocs_characterize.toolkit.anomalies import detect_header_inconsistencies

    headers = {
        f"RUN-{i}": {"biomass_g_l": "Biomass (g/L)", "substrate_g_l": "Substrate (g/L)"}
        for i in range(5)
    }
    assert detect_header_inconsistencies(headers) == []


def test_header_inconsistency_handles_partial_coverage() -> None:
    """Variable present in some runs, absent in others — only compare presents."""
    from fermdocs_characterize.toolkit.anomalies import detect_header_inconsistencies

    headers = {
        "RUN-1": {"wcw_g_l": "WCW (mg/3 mL)"},
        "RUN-2": {},  # this run never reported WCW
        "RUN-3": {"wcw_g_l": "Wet cell weight (mg)"},
    }
    found = detect_header_inconsistencies(headers)
    assert len(found) == 1
    assert set(found[0].raw_headers_by_run.keys()) == {"RUN-1", "RUN-3"}


def test_header_inconsistency_per_variable_independent() -> None:
    """Drift on one variable doesn't taint findings on another."""
    from fermdocs_characterize.toolkit.anomalies import detect_header_inconsistencies

    headers = {
        "RUN-1": {"wcw_g_l": "WCW (mg/3 mL)", "biomass_g_l": "Biomass (g/L)"},
        "RUN-2": {"wcw_g_l": "Wet cell weight (mg)", "biomass_g_l": "Biomass (g/L)"},
    }
    found = detect_header_inconsistencies(headers)
    variables = {h.variable for h in found}
    assert "wcw_g_l" in variables
    assert "biomass_g_l" not in variables


# ---------- regression: penicillin clean bundle emits zero across all detectors ----------


def test_clean_bundle_emits_zero_anomalies_across_all_detectors() -> None:
    from fermdocs_characterize.toolkit.anomalies import (
        detect_bioreactor_changes,
        detect_header_inconsistencies,
        detect_scale_changes,
    )

    runs = [f"RUN-{i}" for i in range(5)]
    assert detect_scale_changes({r: 2.5 for r in runs}) == []
    assert detect_bioreactor_changes({r: "BIOREACTOR_A" for r in runs}) == []
    assert detect_header_inconsistencies(
        {r: {"biomass_g_l": "Biomass (g/L)"} for r in runs}
    ) == []
    # Existing detectors stay green too.
    assert detect_instrument_changes(
        {r: ["DO probe: Hamilton VisiFerm"] for r in runs}
    ) == []
