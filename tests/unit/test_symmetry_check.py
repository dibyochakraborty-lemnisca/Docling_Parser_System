"""check_symmetry: emit data_gaps for missing (metric, run) coverage.

Plan ref: plans/2026-05-07-characterize-determinism.md commit 5.

Covers:
  1. symmetric coverage → no extra findings
  2. B10 only on RUN-1 → emits one [SYMMETRY] data_gap for B10×RUN-2
  3. B10 fully missing on both runs → no extra findings (uniformly
     missing is not asymmetric — commit-1's adapters already emit
     data_gaps for those)
  4. cross-run metric (A19) ignored — has no run_ids
  5. **REGRESSION**: synthesizer prompt USER QUESTION + new TOOL GAP
     rule both fire; neither breaks the other
  6. **CRITICAL `[tool-gap-axis]` does NOT over-fire**: bundle with
     legitimately missing run data (no symmetry_violation findings)
     — critic prompt tells the LLM to accept insufficient_data
     when it's a real DATA gap, not a TOOL gap.
"""

from __future__ import annotations

import re

from fermdocs_characterize.agents.catalog_runner import _BundleView
from fermdocs_characterize.agents.symmetry_check import check_symmetry
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
from fermdocs_hypothesis.agents.critic import CRITIC_INVARIANTS
from fermdocs_hypothesis.agents.synthesizer import SYNTHESIZER_INVARIANTS


CHAR_ID = "00000000-0000-0000-0000-000000000001"


def _trajectory(run_id: str, variable: str) -> Trajectory:
    times = [float(i * 4) for i in range(8)]
    values = [0.5 * 1.05**i for i in range(8)]
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


def _finding(metric_id: str, run_id: str, idx: int, *, pattern_kind: str = "computed_metric") -> Finding:
    return Finding(
        finding_id=f"{CHAR_ID}:F-{idx:04d}",
        type=FindingType.KINETIC_ANOMALY,
        severity=Severity.MINOR if pattern_kind == "computed_metric" else Severity.INFO,
        tier=Tier.A,
        summary=f"{metric_id} on {run_id}.",
        confidence=0.85 if pattern_kind == "computed_metric" else 0.5,
        extracted_via=ExtractedVia.DETERMINISTIC,
        evidence_strength=EvidenceStrength(n_observations=10, n_independent_runs=1),
        evidence_observation_ids=[f"obs-{run_id}"],
        variables_involved=["biomass_g_l"],
        run_ids=[run_id],
        statistics={
            "pattern_kind": pattern_kind,
            "metric_id": metric_id,
            "tier": "A",
        },
    )


def _bundle(run_ids: list[str]) -> _BundleView:
    trajs = [_trajectory(r, "biomass_g_l") for r in run_ids]
    return _BundleView(
        characterization_id=CHAR_ID,
        run_ids=run_ids,
        trajectories=trajs,
        organism=None,
        process_family=None,
    )


# ---------- 1. symmetric coverage → no extras ----------


def test_symmetric_coverage_emits_no_extras() -> None:
    bundle = _bundle(["RUN-1", "RUN-2"])
    findings = [
        _finding("A8", "RUN-1", 1),
        _finding("A8", "RUN-2", 2),
        _finding("B10", "RUN-1", 3),
        _finding("B10", "RUN-2", 4),
    ]
    extras = check_symmetry(bundle, findings, starting_index=len(findings))
    # No metric is asymmetric.
    assert extras == []


# ---------- 2. one-sided coverage → [SYMMETRY] data_gap ----------


def test_b10_only_on_run1_emits_symmetry_gap_for_run2() -> None:
    """The exact IndPenSim bug class: B10 computed for RUN-1 only,
    silently absent for RUN-2."""
    bundle = _bundle(["RUN-1", "RUN-2"])
    findings = [
        # A8 has full coverage — should not fire.
        _finding("A8", "RUN-1", 1),
        _finding("A8", "RUN-2", 2),
        # B10 only on RUN-1 — should fire for RUN-2.
        _finding("B10", "RUN-1", 3),
    ]
    extras = check_symmetry(bundle, findings, starting_index=len(findings))
    # Exactly one symmetry data_gap, for B10×RUN-2.
    by_metric_run = {
        (f.statistics["metric_id"], f.run_ids[0])
        for f in extras
    }
    assert ("B10", "RUN-2") in by_metric_run
    # Sanity: it does NOT also emit one for A8 (symmetric there).
    assert not any(m == "A8" for m, _ in by_metric_run)

    # The emitted finding carries the symmetry_violation marker.
    sym = next(f for f in extras if f.statistics["metric_id"] == "B10")
    assert sym.statistics["symmetry_violation"] is True
    assert sym.statistics["covered_runs"] == ["RUN-1"]
    assert sym.statistics["pattern_kind"] == "data_gap"
    assert "asymmetric" in sym.statistics["reason"]
    assert sym.summary.startswith("[SYMMETRY]")


# ---------- 3. fully missing → no extras (not asymmetric) ----------


def test_metric_missing_on_all_runs_emits_no_extras() -> None:
    """If a metric is uniformly absent (e.g. no adapter today, every
    run equally lacks it), symmetry check stays quiet. Not its job to
    flag total-absence — commit 1's pre-flight WARNINGs do that."""
    bundle = _bundle(["RUN-1", "RUN-2"])
    findings = [
        _finding("A8", "RUN-1", 1),
        _finding("A8", "RUN-2", 2),
        # B10 absent on BOTH runs — not asymmetric, no symmetry findings
    ]
    extras = check_symmetry(bundle, findings, starting_index=len(findings))
    assert all(
        f.statistics["metric_id"] != "B10" for f in extras
    )


# ---------- 4. cross-run metric ignored ----------


def test_cross_run_metric_a19_ignored() -> None:
    """A19/A20/A21 emit ONE finding without per-run scoping. Symmetry
    check skips them entirely — there's no asymmetry possible."""
    bundle = _bundle(["RUN-1", "RUN-2", "RUN-3"])
    findings = [
        _finding("A8", "RUN-1", 1),
        _finding("A8", "RUN-2", 2),
        _finding("A8", "RUN-3", 3),
    ]
    extras = check_symmetry(bundle, findings, starting_index=len(findings))
    # No A19/A20/A21 in extras even though they have no findings at all.
    assert all(
        f.statistics["metric_id"] not in {"A19", "A20", "A21"}
        for f in extras
    )


# ---------- 5. REGRESSION: synth prompt rules compose ----------


def test_synthesizer_prompt_user_question_and_tool_gap_both_present() -> None:
    """Both the PR-A USER QUESTION rule and the commit-5 TOOL GAP rule
    must coexist in SYNTHESIZER_INVARIANTS — a future prompt rewrite
    that loses either re-opens IndPenSim regressions."""
    flat = " ".join(SYNTHESIZER_INVARIANTS)
    # PR-A bias-posture rule still present
    assert "USER QUESTION" in flat
    assert "question_answered" in flat
    # Commit-5 tool-gap rule
    assert "TOOL GAP vs DATA GAP" in flat
    assert "symmetry_violation" in flat
    assert "insufficient_data" in flat
    # Anti-overfire: rule explicitly tells synth NOT to use
    # insufficient_data when only TOOL gap fired
    assert "the BUNDLE itself lacks the data" in flat


# ---------- 6. CRITICAL: tool-gap-axis does NOT over-fire ----------


def test_critic_tool_gap_axis_rule_present_with_anti_overfire_clause() -> None:
    """REGRESSION: the [tool-gap-axis] critic rule MUST include an
    explicit anti-over-fire clause. Without it, every bundle with
    legitimately missing run data would have its 'insufficient_data'
    answer red-flagged — the critic would loop forever."""
    flat = " ".join(CRITIC_INVARIANTS)
    assert "[TOOL-GAP-AXIS]" in flat or "[tool-gap-axis]" in flat
    # Anti-over-fire: critic only red-flags when symmetry_violation
    # findings exist; legitimate data-missing bundles still get
    # 'insufficient_data' accepted.
    assert "Do NOT over-fire" in flat or "do NOT over-fire" in flat.lower()
    assert "BUNDLE itself lacks" in flat
    # Tagged with [tool-gap-axis] prefix for retry distinguishability
    assert "[tool-gap-axis]" in flat
