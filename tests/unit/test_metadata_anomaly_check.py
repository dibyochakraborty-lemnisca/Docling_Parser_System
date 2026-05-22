"""Pipeline-level wire-up of metadata anomaly detectors.

Reviewer feedback A1. The toolkit detectors are tested in
test_anomaly_detectors.py; this module pins the integration: detectors
actually fire on the dossier+trajectory shape that lands at
characterize-time, and the emitted Findings are well-formed (validator
will accept them).
"""

from __future__ import annotations

from uuid import UUID, uuid4

import pytest

from fermdocs_characterize.agents.catalog_runner import _BundleView
from fermdocs_characterize.agents.metadata_anomaly_check import (
    check_metadata_anomalies,
)
from fermdocs_characterize.schema import (
    DataQuality,
    Trajectory,
)


def _make_traj(
    *,
    traj_id: str,
    run_id: str,
    variable: str,
    time_grid: list[float],
    values: list[float],
    obs_id: str,
) -> Trajectory:
    return Trajectory(
        trajectory_id=traj_id,
        run_id=run_id,
        variable=variable,
        time_grid=time_grid,
        values=values,
        imputation_flags=[False] * len(time_grid),
        imputation_method=None,
        source_observation_ids=[obs_id],
        unit="g/L",
        quality=1.0,
        data_quality=DataQuality(
            pct_missing=0.0, pct_imputed=0.0, pct_real=1.0
        ),
    )


def _bundle_with_runs(*run_ids: str) -> tuple[_BundleView, list[Trajectory]]:
    """Build a bundle with one biomass trajectory per run (anchor obs ids)."""
    trajs = [
        _make_traj(
            traj_id=f"T-{i+1:04d}",
            run_id=run_id,
            variable="biomass_g_l",
            time_grid=[0.0, 6.0, 12.0],
            values=[1.0, 2.0, 3.0],
            obs_id=f"OBS-{run_id}-1",
        )
        for i, run_id in enumerate(run_ids)
    ]
    bundle = _BundleView(
        characterization_id="BUNDLE-1",
        run_ids=list(run_ids),
        trajectories=trajs,
        organism=None,
        process_family=None,
    )
    return bundle, trajs


# ---------- instrument-change wire-up ----------


def test_instrument_change_fires_via_pipeline_wireup() -> None:
    char_id = uuid4()
    bundle, _ = _bundle_with_runs("RUN-1", "RUN-2", "RUN-3", "RUN-4")
    narratives = [
        {"run_id": "RUN-1", "text": "WCW measured on Hitachi spectrophotometer."},
        {"run_id": "RUN-2", "text": "WCW measured on Hitachi spectrophotometer."},
        {"run_id": "RUN-3", "text": "WCW measured on Hitachi spectrophotometer."},
        {"run_id": "RUN-4", "text": "WCW measured on LABMAN spectrophotometer."},
    ]
    findings = check_metadata_anomalies(
        char_id=char_id,
        bundle=bundle,
        dossier={},
        narrative_observations=narratives,
    )
    instrument_findings = [
        f for f in findings
        if (f.statistics or {}).get("anomaly_kind") == "instrument_change"
    ]
    assert len(instrument_findings) == 1
    f = instrument_findings[0]
    assert "[METADATA-ANOMALY]" in f.summary
    assert "spectrophotometer" in f.summary
    assert f.statistics["instruments_by_run"]["RUN-4"] == "LABMAN"


def test_narratives_without_run_id_are_skipped() -> None:
    """File-level narratives can't be cross-batch evidence."""
    char_id = uuid4()
    bundle, _ = _bundle_with_runs("RUN-1", "RUN-2")
    narratives = [
        {"run_id": None, "text": "Hitachi spectrophotometer used."},
        {"run_id": "RUN-1", "text": "LABMAN spectrophotometer used."},
    ]
    findings = check_metadata_anomalies(
        char_id=char_id,
        bundle=bundle,
        dossier={},
        narrative_observations=narratives,
    )
    instrument_findings = [
        f for f in findings
        if (f.statistics or {}).get("anomaly_kind") == "instrument_change"
    ]
    assert instrument_findings == []


# ---------- header-inconsistency wire-up ----------


def test_header_inconsistency_fires_via_dossier_golden_columns() -> None:
    char_id = uuid4()
    bundle, _ = _bundle_with_runs("RUN-1", "RUN-2")
    dossier = {
        "golden_columns": {
            "wcw_g_l": {
                "observations": [
                    {
                        "observation_id": "OBS-RUN-1-1",
                        "raw_header": "WCW (mg/3 mL)",
                        "source": {"locator": {"run_id": "RUN-1"}},
                        "value": 1.0, "unit": "g/L",
                    },
                    {
                        "observation_id": "OBS-RUN-2-1",
                        "raw_header": "Wet cell weight (mg)",
                        "source": {"locator": {"run_id": "RUN-2"}},
                        "value": 1.0, "unit": "g/L",
                    },
                ]
            }
        }
    }
    findings = check_metadata_anomalies(
        char_id=char_id,
        bundle=bundle,
        dossier=dossier,
        narrative_observations=[],
    )
    header_findings = [
        f for f in findings
        if (f.statistics or {}).get("anomaly_kind") == "header_inconsistency"
    ]
    assert len(header_findings) == 1
    f = header_findings[0]
    assert "wcw_g_l" in f.summary
    assert f.statistics["variable"] == "wcw_g_l"


# ---------- h0-outlier wire-up ----------


def test_h0_outlier_fires_via_trajectories() -> None:
    char_id = uuid4()
    # Build 5 normal runs + 1 outlier run on biomass at h=0
    trajs = []
    for i, (run_id, h0_value) in enumerate([
        ("RUN-1", 1.0), ("RUN-2", 1.05), ("RUN-3", 0.95),
        ("RUN-4", 1.02), ("RUN-5", 0.98), ("RUN-OUT", 8.0),
    ]):
        trajs.append(_make_traj(
            traj_id=f"T-{i+1:04d}",
            run_id=run_id,
            variable="biomass_g_l",
            time_grid=[0.0, 6.0],
            values=[h0_value, h0_value + 1.0],
            obs_id=f"OBS-{run_id}-1",
        ))
    bundle = _BundleView(
        characterization_id="BUNDLE-1",
        run_ids=[t.run_id for t in trajs],
        trajectories=trajs,
        organism=None, process_family=None,
    )
    findings = check_metadata_anomalies(
        char_id=char_id,
        bundle=bundle,
        dossier={},
        narrative_observations=[],
    )
    h0_findings = [
        f for f in findings
        if (f.statistics or {}).get("anomaly_kind") == "h0_outlier"
    ]
    assert len(h0_findings) == 1
    f = h0_findings[0]
    assert "RUN-OUT" in f.summary
    assert f.run_ids == ["RUN-OUT"]
    assert f.statistics["run_value"] == 8.0


# ---------- finding-id namespacing ----------


def test_finding_ids_are_namespaced_to_char_id() -> None:
    char_id = uuid4()
    bundle, _ = _bundle_with_runs("RUN-1", "RUN-2")
    narratives = [
        {"run_id": "RUN-1", "text": "Hitachi spectrophotometer used."},
        {"run_id": "RUN-2", "text": "LABMAN spectrophotometer used."},
    ]
    findings = check_metadata_anomalies(
        char_id=char_id,
        bundle=bundle,
        dossier={},
        narrative_observations=narratives,
        starting_index=10,
    )
    assert len(findings) >= 1
    for f in findings:
        assert f.finding_id.startswith(f"{char_id}:F-")


# ---------- REGRESSION: clean bundle emits zero ----------


def test_clean_bundle_emits_zero_metadata_anomalies() -> None:
    """Penicillin-shaped uniform bundle: no instruments, no header drift,
    no h0 outliers → zero anomaly findings."""
    char_id = uuid4()
    bundle, _ = _bundle_with_runs("RUN-1", "RUN-2", "RUN-3")
    narratives = [
        {"run_id": r, "text": "Run completed normally."}
        for r in ["RUN-1", "RUN-2", "RUN-3"]
    ]
    dossier = {
        "golden_columns": {
            "biomass_g_l": {
                "observations": [
                    {
                        "observation_id": f"OBS-{r}-1",
                        "raw_header": "Biomass (g/L)",
                        "source": {"locator": {"run_id": r}},
                        "value": 1.0, "unit": "g/L",
                    }
                    for r in ["RUN-1", "RUN-2", "RUN-3"]
                ]
            }
        }
    }
    findings = check_metadata_anomalies(
        char_id=char_id,
        bundle=bundle,
        dossier=dossier,
        narrative_observations=narratives,
    )
    assert findings == []


def test_anomaly_kind_set_on_every_finding() -> None:
    """Every emitted finding carries statistics.anomaly_kind so the
    synthesizer can route them in its prompt context."""
    char_id = uuid4()
    bundle, _ = _bundle_with_runs("RUN-1", "RUN-2")
    narratives = [
        {"run_id": "RUN-1", "text": "Hitachi spectrophotometer used."},
        {"run_id": "RUN-2", "text": "LABMAN spectrophotometer used."},
    ]
    findings = check_metadata_anomalies(
        char_id=char_id,
        bundle=bundle,
        dossier={},
        narrative_observations=narratives,
    )
    for f in findings:
        assert (f.statistics or {}).get("anomaly_kind") in {
            "instrument_change",
            "header_inconsistency",
            "h0_outlier",
            "scale_change",
            "bioreactor_change",
        }
        assert (f.statistics or {}).get("pattern_kind") == "metadata_anomaly"
