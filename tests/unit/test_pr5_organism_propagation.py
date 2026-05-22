"""Organism flows from dossier → analyzer prompt → Tier C catalog.

Production bug from run 099b40f4: dossier extracted
organism="Saccharomyces cerevisiae" correctly via the LLM identity
extractor, but the trajectory_analyzer never received the value, so
every Tier C metric (C2, C5, C10) data-gapped with "no organism in
priors" even though priors registry has the entry.

Fix: pipeline reads dossier.experiment.process.observed.organism
and passes it to analyzer.analyze(); analyzer threads it into the
[IDENTITY] block of the user prompt and into the per-bundle metric
checklist.
"""

from __future__ import annotations

from fermdocs_characterize.agents.trajectory_analyzer import TrajectoryAnalyzerAgent


# ---------- checklist: organism gates Tier C metrics ----------


def test_checklist_c2_data_gap_when_organism_missing() -> None:
    out = TrajectoryAnalyzerAgent._build_metric_checklist(
        variables={"biomass_g_l"}, n_runs=1, organism=None
    )
    assert "[DATA_GAP] C2" in out
    assert "no organism in dossier identity layer" in out


def test_checklist_c5_c10_data_gap_when_organism_missing() -> None:
    out = TrajectoryAnalyzerAgent._build_metric_checklist(
        variables={"biomass_g_l"}, n_runs=1, organism=None
    )
    assert "[DATA_GAP] C5" in out
    assert "[DATA_GAP] C10" in out


def test_checklist_c_tier_applicable_when_organism_present() -> None:
    out = TrajectoryAnalyzerAgent._build_metric_checklist(
        variables={"biomass_g_l"}, n_runs=1, organism="Saccharomyces cerevisiae"
    )
    # C2/C5/C10 should pass the organism gate.
    assert "[APPLICABLE] C2" in out
    assert "[APPLICABLE] C5" in out
    assert "[APPLICABLE] C10" in out


def test_checklist_c3_unaffected_by_organism_pure_chemistry() -> None:
    """C3 (Henry's-law O2 saturation) is pure chemistry — no organism
    prior needed. Should remain DATA_GAP only when temperature variable
    is absent, regardless of organism."""
    no_temp = TrajectoryAnalyzerAgent._build_metric_checklist(
        variables={"biomass_g_l"}, n_runs=1, organism=None
    )
    assert "[DATA_GAP] C3" in no_temp
    with_temp = TrajectoryAnalyzerAgent._build_metric_checklist(
        variables={"biomass_g_l", "temperature_k"}, n_runs=1, organism=None
    )
    # C3 should now light up — temperature present, organism not needed.
    assert "[APPLICABLE] C3" in with_temp


# ---------- user-text injection: [IDENTITY] block ----------


def _make_traj(run_id: str, variable: str, n: int = 5):
    from fermdocs_characterize.schema import DataQuality, Trajectory
    return Trajectory(
        trajectory_id="T-0001",
        run_id=run_id,
        variable=variable,
        time_grid=[float(i) for i in range(n)],
        values=[float(i + 1) for i in range(n)],
        imputation_flags=[False] * n,
        unit="g/L",
        quality=1.0,
        data_quality=DataQuality(pct_missing=0.0, pct_imputed=0.0, pct_real=1.0),
        source_observation_ids=[f"OBS-{i:04d}" for i in range(n)],
    )


def test_user_text_includes_identity_block_when_organism_provided(tmp_path) -> None:
    obs_path = tmp_path / "observations.csv"
    obs_path.write_text("dummy")
    agent = TrajectoryAnalyzerAgent(client=None)
    text = agent._build_user_text(
        obs_path=obs_path,
        trajectories=[_make_traj("RUN-0001", "biomass_g_l")],
        spec_findings=[],
        organism="Saccharomyces cerevisiae",
        process_family="aerobic_fed_batch_glucose",
    )
    assert "[IDENTITY]" in text
    assert "organism: Saccharomyces cerevisiae" in text
    assert "process_family: aerobic_fed_batch_glucose" in text


def test_user_text_marks_unknown_organism_explicitly(tmp_path) -> None:
    obs_path = tmp_path / "observations.csv"
    obs_path.write_text("dummy")
    agent = TrajectoryAnalyzerAgent(client=None)
    text = agent._build_user_text(
        obs_path=obs_path,
        trajectories=[_make_traj("RUN-0001", "biomass_g_l")],
        spec_findings=[],
        organism=None,
        process_family=None,
    )
    assert "[IDENTITY]" in text
    assert "(unknown" in text
    # Marker so a future regression where organism stays null but the
    # field reads "None" instead of an explanation is loud.
    assert "Tier C" in text
