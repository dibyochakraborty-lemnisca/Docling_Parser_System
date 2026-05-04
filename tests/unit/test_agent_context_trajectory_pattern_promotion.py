"""Tests for trajectory_pattern promotion in agent_context ranker.

Regression: on IndPenSim-shape bundles (~80 deterministic spec findings,
2 trajectory_pattern findings), the ranker sorted strictly by severity.
The trajectory patterns (often severity=info or severity=major mixed
with critical specs) got pushed below the agent's visible cap and
diagnose never cited them.

The ranker now puts trajectory_pattern findings first regardless of
severity. They're biology-grounded and high-information; this is the
right default for unknown_process bundles.
"""

from __future__ import annotations

from uuid import UUID

from fermdocs_characterize.agent_context import _rank_finding_ids
from fermdocs_characterize.schema import (
    EvidenceStrength,
    ExtractedVia,
    Finding,
    FindingType,
    Severity,
    Tier,
)


CHAR_ID = UUID("33333333-3333-3333-3333-333333333333")


def _spec_finding(idx: int, severity: Severity = Severity.MAJOR) -> Finding:
    return Finding(
        finding_id=f"{CHAR_ID}:F-{idx:04d}",
        type=FindingType.RANGE_VIOLATION,
        severity=severity,
        tier=Tier.A,
        summary=f"spec range violation #{idx}",
        confidence=0.9,
        extracted_via=ExtractedVia.DETERMINISTIC,
        evidence_strength=EvidenceStrength(n_observations=10, n_independent_runs=1),
        evidence_observation_ids=[f"OBS-{idx:04d}"],
        variables_involved=["biomass_g_l"],
        run_ids=["RUN-0001"],
    )


def _pattern_finding(idx: int, severity: Severity = Severity.INFO) -> Finding:
    return Finding(
        finding_id=f"{CHAR_ID}:F-{idx:04d}",
        type=FindingType.TRAJECTORY_PATTERN,
        severity=severity,
        tier=Tier.B,
        summary=f"trajectory pattern #{idx}",
        confidence=0.8,
        extracted_via=ExtractedVia.LLM_JUDGED,
        evidence_strength=EvidenceStrength(n_observations=2, n_independent_runs=2),
        evidence_observation_ids=[f"OBS-{idx:04d}"],
        variables_involved=["biomass_g_l"],
        run_ids=["RUN-0001", "RUN-0002"],
        statistics={"pattern_kind": "phase_boundary"},
    )


def test_trajectory_pattern_ranks_above_critical_spec_findings():
    """Even an INFO-severity trajectory_pattern beats a CRITICAL spec
    finding. The premise: trajectory patterns are biology-grounded;
    critical spec violations on unknown_process bundles are mostly
    schema artifacts."""
    spec = _spec_finding(1, Severity.CRITICAL)
    pattern = _pattern_finding(101, Severity.INFO)
    by_id = {spec.finding_id: spec, pattern.finding_id: pattern}
    ranked = _rank_finding_ids([spec.finding_id, pattern.finding_id], by_id)
    assert ranked[0] == pattern.finding_id, (
        "trajectory_pattern must rank first regardless of severity"
    )


def test_indpensim_shape_pattern_makes_visible_cap():
    """Reproduces the IndPenSim case: 80 deterministic CRITICAL spec
    findings + 2 trajectory_pattern findings. Without the promotion,
    the patterns would land at indices 80, 81 and get truncated below
    the agent's MAX_TOP_FINDINGS visible cap. With promotion, they
    occupy positions 0 and 1."""
    spec_findings = [
        _spec_finding(i, Severity.CRITICAL) for i in range(1, 81)
    ]
    pattern_findings = [
        _pattern_finding(101, Severity.MAJOR),
        _pattern_finding(102, Severity.INFO),
    ]
    all_findings = spec_findings + pattern_findings
    by_id = {f.finding_id: f for f in all_findings}
    ranked = _rank_finding_ids([f.finding_id for f in all_findings], by_id)
    # Both trajectory_pattern findings come first
    assert ranked[0] in {f.finding_id for f in pattern_findings}
    assert ranked[1] in {f.finding_id for f in pattern_findings}
    # Spec findings follow
    assert ranked[2] in {f.finding_id for f in spec_findings}


def test_multiple_patterns_sorted_by_severity_among_themselves():
    """Within the trajectory_pattern bucket, severity desc still applies."""
    p_info = _pattern_finding(101, Severity.INFO)
    p_major = _pattern_finding(102, Severity.MAJOR)
    p_critical = _pattern_finding(103, Severity.CRITICAL)
    by_id = {f.finding_id: f for f in (p_info, p_major, p_critical)}
    ranked = _rank_finding_ids(
        [p_info.finding_id, p_major.finding_id, p_critical.finding_id],
        by_id,
    )
    assert ranked == [p_critical.finding_id, p_major.finding_id, p_info.finding_id]


def test_no_patterns_falls_back_to_severity_only():
    """Sanity: when no trajectory_pattern findings exist, ranker behaves
    exactly as before (severity desc, then aggregated, then id)."""
    f_critical = _spec_finding(1, Severity.CRITICAL)
    f_major = _spec_finding(2, Severity.MAJOR)
    f_minor = _spec_finding(3, Severity.MINOR)
    by_id = {f.finding_id: f for f in (f_critical, f_major, f_minor)}
    ranked = _rank_finding_ids(
        [f_minor.finding_id, f_critical.finding_id, f_major.finding_id],
        by_id,
    )
    assert ranked == [f_critical.finding_id, f_major.finding_id, f_minor.finding_id]


# ---------- diagnose prompt content ----------


def test_diagnose_bundle_prompt_calls_out_trajectory_pattern_as_primary():
    """The diagnose prompt must explicitly recognize trajectory_pattern
    findings as primary evidence under UNKNOWN_PROCESS — otherwise the
    agent ignores them and falls back to spec-mismatch failures (the
    May 2026 regression)."""
    from fermdocs_diagnose.agent import _BUNDLE_SYSTEM_PROMPT

    flat = _BUNDLE_SYSTEM_PROMPT
    assert "trajectory_pattern" in flat.lower()
    # The "PRIMARY EVIDENCE" framing for trajectory_pattern is what flips
    # the agent's preference order.
    assert "PRIMARY EVIDENCE" in flat
    # And the rule explicitly tells the agent to PREFER pattern findings
    # over spec-mismatch range_violation findings.
    assert "PREFER" in flat or "Prefer" in flat or "prefer" in flat


def test_diagnose_meta_claim_contract_includes_trajectory_pattern():
    """The meta-claim contract violation now also fires when
    trajectory_pattern findings exist but the agent emits only meta
    claims. Without this, the agent could see patterns AND ignore them
    cleanly via the data_quality_caveat dodge."""
    from fermdocs_diagnose.agent import _BUNDLE_SYSTEM_PROMPT

    flat = _BUNDLE_SYSTEM_PROMPT
    # The contract now mentions trajectory_pattern as a triggering signal
    assert "trajectory_pattern" in flat.lower()
    # And mentions that patterns ARE either failures or cross_run_observation
    assert "trajectory pattern" in flat.lower() or "Trajectory pattern" in flat
