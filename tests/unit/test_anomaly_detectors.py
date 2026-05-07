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
