"""Trajectory analyzer prompt: [ALREADY COMPUTED] block + anti-rebuild.

Plan ref: plans/2026-05-07-characterize-determinism.md commit 1, A1 fix.

Covers (REGRESSION tests):
  1. When catalog_findings include B10 × RUN-0001 + B10 × RUN-0002, the
     prompt's [ALREADY COMPUTED] block lists B10 with both runs and
     includes a hard rule against re-emitting that metric_id.
  2. Empty catalog_findings → prompt says '(none)' and still includes
     the hard rule.
  3. data_gap findings from the catalog runner are NOT in the
     [ALREADY COMPUTED] block (only computed_metric findings are).
  4. Prompt no longer asks the LLM to iterate the metric checklist
     itself — the catalog runner owns iteration now.
"""

from __future__ import annotations

from fermdocs_characterize.agents.trajectory_analyzer import (
    TrajectoryAnalyzerAgent,
)
from fermdocs_characterize.schema import (
    EvidenceStrength,
    ExtractedVia,
    Finding,
    FindingType,
    Severity,
    Tier,
)


CHAR_ID = "00000000-0000-0000-0000-000000000001"


def _computed_finding(metric_id: str, run_id: str, idx: int) -> Finding:
    return Finding(
        finding_id=f"{CHAR_ID}:F-{idx:04d}",
        type=FindingType.KINETIC_ANOMALY,
        severity=Severity.MINOR,
        tier=Tier.A,
        summary=f"{metric_id} computed on {run_id}.",
        confidence=0.85,
        extracted_via=ExtractedVia.DETERMINISTIC,
        evidence_strength=EvidenceStrength(n_observations=10, n_independent_runs=1),
        evidence_observation_ids=[f"obs-{run_id}-{metric_id}-1"],
        variables_involved=["biomass_g_l"],
        run_ids=[run_id],
        statistics={
            "pattern_kind": "computed_metric",
            "metric_id": metric_id,
            "tier": "A",
            "value": 0.5,
        },
    )


def _gap_finding(metric_id: str, run_id: str, idx: int) -> Finding:
    return Finding(
        finding_id=f"{CHAR_ID}:F-{idx:04d}",
        type=FindingType.KINETIC_ANOMALY,
        severity=Severity.INFO,
        tier=Tier.A,
        summary=f"{metric_id} skipped on {run_id}: precondition not met.",
        confidence=0.5,
        extracted_via=ExtractedVia.DETERMINISTIC,
        evidence_strength=EvidenceStrength(n_observations=0, n_independent_runs=0),
        evidence_observation_ids=["deterministic-runner"],
        variables_involved=[],
        run_ids=[run_id],
        statistics={
            "pattern_kind": "data_gap",
            "metric_id": metric_id,
            "tier": "A",
            "reason": "precondition not met",
        },
    )


# ---------- 1. computed metrics surface in the block ----------


def test_already_computed_block_lists_metric_run_pairs() -> None:
    findings = [
        _computed_finding("B10", "RUN-0001", 1),
        _computed_finding("B10", "RUN-0002", 2),
        _computed_finding("A8",  "RUN-0001", 3),
    ]
    block = TrajectoryAnalyzerAgent._build_already_computed_block(
        catalog_findings=findings
    )
    assert "[ALREADY COMPUTED" in block
    assert "B10" in block
    assert "A8" in block
    assert "RUN-0001" in block
    assert "RUN-0002" in block
    # Both runs appear on the B10 line.
    b10_line = next(line for line in block.splitlines() if "B10" in line)
    assert "RUN-0001" in b10_line and "RUN-0002" in b10_line


# ---------- 2. empty catalog_findings ----------


def test_already_computed_block_empty_emits_none_marker() -> None:
    block = TrajectoryAnalyzerAgent._build_already_computed_block(
        catalog_findings=[]
    )
    assert "[ALREADY COMPUTED" in block
    assert "none" in block.lower()


# ---------- 3. data_gap findings excluded ----------


def test_already_computed_block_excludes_data_gap_findings() -> None:
    findings = [
        _computed_finding("B10", "RUN-0001", 1),
        _gap_finding("B10", "RUN-0002", 2),  # data_gap, must not appear
        _gap_finding("A8", "RUN-0001", 3),
    ]
    block = TrajectoryAnalyzerAgent._build_already_computed_block(
        catalog_findings=findings
    )
    # B10 listed (one computed) but only with the RUN that succeeded.
    b10_line = next(line for line in block.splitlines() if "B10" in line)
    assert "RUN-0001" in b10_line
    assert "RUN-0002" not in b10_line, (
        "data_gap finding for B10×RUN-0002 should NOT appear in"
        " [ALREADY COMPUTED]; it's a tool gap, not a successful computation."
    )
    # A8 not present at all (only data_gap for it).
    assert "A8" not in block


# ---------- 4. prompt anti-rebuild guard ----------


def test_full_user_text_includes_already_computed_and_anti_rebuild_rule() -> None:
    """REGRESSION: the analyzer's prompt no longer tells the LLM to
    iterate the catalog. It now tells the LLM the catalog has been
    computed and to stick to open-ended findings."""
    from pathlib import Path

    from fermdocs_characterize.schema import (
        DataQuality,
        Trajectory,
    )

    agent = TrajectoryAnalyzerAgent(client=None)
    # Build minimal trajectories so the prompt builder doesn't bail.
    traj = Trajectory(
        trajectory_id="T-0001",
        run_id="RUN-0001",
        variable="biomass_g_l",
        time_grid=[0.0, 4.0, 8.0, 12.0],
        values=[0.5, 1.0, 2.0, 4.0],
        imputation_flags=[False, False, False, False],
        source_observation_ids=["obs-1", "obs-2", "obs-3", "obs-4"],
        unit="g/L",
        quality=1.0,
        data_quality=DataQuality(pct_missing=0.0, pct_imputed=0.0, pct_real=1.0),
    )
    catalog = [_computed_finding("B10", "RUN-0001", 1)]

    text = agent._build_user_text(
        obs_path=Path("/tmp/dummy.csv"),
        trajectories=[traj],
        spec_findings=[],
        organism="S. cerevisiae",
        process_family="yeast_batch",
        catalog_findings=catalog,
    )

    # Block is in the prompt.
    assert "[ALREADY COMPUTED" in text
    assert "B10" in text
    # Hard rule against re-emitting catalog metric_ids.
    assert "Do NOT emit" in text or "do NOT emit" in text.lower()
    # The new task framing replaces the old 'work the catalog
    # checklist top-to-bottom' sentence.
    assert "open-ended" in text.lower()
    # The old per-metric loop instruction is gone (anti-rebuild).
    assert "checklist top-to-bottom" not in text
