"""Bundles reporting OD600 / WCW (no direct biomass_g_l DCW) should
still light up the kinetics catalog.

Production bug from carotenoid run 099b40f4: bundle had wcw_g_l +
od600_au + 6 runs, expected behaviour was A8/A9/A10/A11 to fire on
WCW or OD as biomass proxies. Actual behaviour was 11 of 14 ready
metrics emitted as data_gap because the catalog only accepted
biomass_g_l literal.

Fix: InputSpec.accepted_proxies declares scale-invariant alternatives;
checklist resolves any of {variable, *accepted_proxies}.
"""

from __future__ import annotations

from fermdocs_characterize.agents.metric_catalog import (
    BIOMASS_PROXIES,
    DO_PROXIES,
    get_entry,
)
from fermdocs_characterize.agents.trajectory_analyzer import TrajectoryAnalyzerAgent


def test_biomass_proxies_constant_includes_wcw_and_od() -> None:
    assert "wcw_g_l" in BIOMASS_PROXIES
    assert "od600_au" in BIOMASS_PROXIES


def test_do_proxies_includes_pct_saturation() -> None:
    assert "do_pct_saturation" in DO_PROXIES


def test_a8_input_spec_carries_proxies() -> None:
    spec = get_entry("A8").required_inputs[0]
    assert spec.variable == "biomass_g_l"
    assert "wcw_g_l" in spec.accepted_proxies
    assert "od600_au" in spec.accepted_proxies


def test_a9_a10_a11_all_carry_biomass_proxies() -> None:
    for mid in ("A9", "A10", "A11"):
        spec = get_entry(mid).required_inputs[0]
        assert spec.variable == "biomass_g_l"
        assert "wcw_g_l" in spec.accepted_proxies, f"{mid} missing wcw proxy"


def test_a14_input_spec_carries_do_proxies() -> None:
    spec = get_entry("A14").required_inputs[0]
    assert spec.variable == "dissolved_o2_mg_l"
    assert "do_pct_saturation" in spec.accepted_proxies


# ---------- checklist behavior with proxies ----------


def test_checklist_marks_a8_applicable_with_only_wcw() -> None:
    out = TrajectoryAnalyzerAgent._build_metric_checklist(
        variables={"wcw_g_l"}, n_runs=1
    )
    assert "[APPLICABLE] A8" in out
    assert "via proxy 'wcw_g_l'" in out


def test_checklist_marks_a8_applicable_with_only_od600() -> None:
    out = TrajectoryAnalyzerAgent._build_metric_checklist(
        variables={"od600_au"}, n_runs=1
    )
    assert "[APPLICABLE] A8" in out
    assert "via proxy 'od600_au'" in out


def test_checklist_marks_a14_applicable_with_only_do_pct() -> None:
    out = TrajectoryAnalyzerAgent._build_metric_checklist(
        variables={"do_pct_saturation"}, n_runs=1
    )
    assert "[APPLICABLE] A14" in out
    assert "via proxy 'do_pct_saturation'" in out


def test_checklist_full_carotenoid_shape() -> None:
    """Reproducer for run 099b40f4 — 6 runs, OD/WCW/DO%/feed only."""
    out = TrajectoryAnalyzerAgent._build_metric_checklist(
        variables={"od600_au", "wcw_g_l", "do_pct_saturation", "feed_rate_l_per_h"},
        n_runs=6,
    )
    # Kinetics light up via biomass proxies
    for mid in ("A8", "A9", "A10", "A11"):
        assert f"[APPLICABLE] {mid}" in out, f"{mid} should fire via proxy"
    # DO margin lights up via do_pct_saturation
    assert "[APPLICABLE] A14" in out
    # Cross-run cohort metrics applicable at 6 runs
    for mid in ("A19", "A20", "A21"):
        assert f"[APPLICABLE] {mid}" in out
    # B10 RQ still data_gap (no OUR/CER)
    assert "[DATA_GAP] B10" in out
    # B16 carbon balance still data_gap (no substrate/CO2)
    assert "[DATA_GAP] B16" in out


def test_checklist_prefers_primary_variable_when_present() -> None:
    """When biomass_g_l (DCW) IS present, the checklist should match it
    directly without dressing it as 'via proxy'."""
    out = TrajectoryAnalyzerAgent._build_metric_checklist(
        variables={"biomass_g_l"}, n_runs=1
    )
    assert "[APPLICABLE] A8" in out
    # Primary match — no proxy annotation
    a8_line = [ln for ln in out.splitlines() if "A8" in ln][0]
    assert "via proxy" not in a8_line
