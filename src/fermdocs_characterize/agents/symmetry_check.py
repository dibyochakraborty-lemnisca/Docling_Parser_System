"""Symmetry post-condition: emit data_gaps for missing (metric, run) pairs.

Plan ref: plans/2026-05-07-characterize-determinism.md commit 5.

The IndPenSim feedback exposed a class of bugs where a metric got
computed for some runs but not others. Even with commit 1's per-run
iteration in place, edge cases in the toolkit (insufficient points,
NaN windows, etc.) can leave coverage asymmetric. Without a symmetry
check the synthesizer cannot tell apart 'data is missing on RUN-2'
(legitimate) from 'tool failed on RUN-2 but works on RUN-1' (a bug).

This module emits explicit `data_gap` Findings for every
(metric_id, run_id) pair where:
  - the metric was computed for some runs but not this one, AND
  - the metric is per-run (not cross-run; A19/A20/A21 don't fire here).

The synthesizer (commit 5 prompt amendment) reads these as TOOL gaps,
not DATA gaps, and continues to draw conclusions from what DID compute.
The critic's [tool-gap-axis] rule (commit 5) red-flags hypotheses that
use 'insufficient_data' as the answer when symmetry findings indicate
the data IS in the bundle but the toolchain failed.
"""

from __future__ import annotations

from collections import defaultdict

from fermdocs_characterize.agents.catalog_runner import (
    CROSS_RUN_METRIC_IDS,
    applicable_metric_run_pairs,
    _BundleView,
)
from fermdocs_characterize.schema import (
    EvidenceStrength,
    ExtractedVia,
    Finding,
    FindingType,
    Severity,
    Tier,
)


def check_symmetry(
    bundle: _BundleView,
    findings: list[Finding],
    *,
    starting_index: int = 1,
) -> list[Finding]:
    """Detect (metric_id, run_id) pairs where the catalog runner emitted
    findings for some runs but not all, and emit explicit data_gap
    Findings for the gaps.

    The runner's data_gap findings (precondition not met, tool error)
    ALSO count as 'this run was attempted'; we only fire when a metric
    is *entirely* absent for a run that the iteration helper says it
    should have been. This is rare today (commit 1 ensures the runner
    always emits SOMETHING) but defense-in-depth: future bugs that
    drop emissions silently get caught here.

    Returns a list of Findings to APPEND to the bundle's existing
    findings. starting_index controls finding-id namespacing.

    Plan ref: commit 5 of plans/2026-05-07-characterize-determinism.md.
    """
    # Group existing findings by metric_id → set of run_ids covered.
    # Both computed_metric and data_gap count as covered — both attest
    # the runner attempted that pair.
    covered: dict[str, set[str]] = defaultdict(set)
    for f in findings:
        stats = f.statistics or {}
        mid = stats.get("metric_id")
        if not isinstance(mid, str):
            continue
        for r in f.run_ids or []:
            covered[mid].add(r)

    # Enumerate the universe the runner SHOULD have hit.
    expected: dict[str, set[str]] = defaultdict(set)
    for metric_id, run_id in applicable_metric_run_pairs(bundle):
        # Cross-run metrics intentionally don't have run_ids; symmetry
        # check skips them entirely.
        if metric_id in CROSS_RUN_METRIC_IDS:
            continue
        if run_id is None:
            continue
        expected[metric_id].add(run_id)

    # Diff: for each metric, which runs are missing?
    out: list[Finding] = []
    counter = starting_index
    char_id = bundle.characterization_id
    for metric_id, expected_runs in expected.items():
        covered_runs = covered.get(metric_id, set())
        if not covered_runs:
            # Metric entirely absent — either no adapter (commit 1
            # soft-skip) or the runner didn't run. Symmetry check
            # ignores: 'no coverage anywhere' is not asymmetric, it's
            # uniformly missing. The user can investigate via the
            # runner's pre-flight warnings.
            continue
        if covered_runs == expected_runs:
            # All expected runs covered — no asymmetry.
            continue
        missing = expected_runs - covered_runs
        for run_id in sorted(missing):
            counter += 1
            out.append(_symmetry_data_gap(
                finding_id=f"{char_id}:F-{counter:04d}",
                metric_id=metric_id,
                run_id=run_id,
                covered_runs=sorted(covered_runs),
            ))
    return out


def _symmetry_data_gap(
    *,
    finding_id: str,
    metric_id: str,
    run_id: str,
    covered_runs: list[str],
) -> Finding:
    """One Finding per missing (metric, run) pair.

    pattern_kind='data_gap' so the validator passes it through, the
    seed_topic_extractor's existing data_gap suppression skips it,
    and the synthesizer prompt rule (commit 5) treats it as a TOOL
    gap (not a DATA gap) when reasoning about coverage.
    """
    return Finding(
        finding_id=finding_id,
        type=FindingType.KINETIC_ANOMALY,
        severity=Severity.INFO,
        tier=Tier.A,
        summary=(
            f"[SYMMETRY] {metric_id} computed on {covered_runs}"
            f" but missing on {run_id}: tool gap, not a data gap."
            f" Investigate why the toolchain emitted nothing for this"
            f" (run, metric) pair."
        ),
        confidence=0.5,
        extracted_via=ExtractedVia.DETERMINISTIC,
        evidence_strength=EvidenceStrength(
            n_observations=0, n_independent_runs=0,
        ),
        evidence_observation_ids=["deterministic-runner"],
        variables_involved=[],
        run_ids=[run_id],
        statistics={
            "pattern_kind": "data_gap",
            "metric_id": metric_id,
            "tier": "A",
            "reason": (
                f"asymmetric coverage — {metric_id} computed for"
                f" {covered_runs} but not for {run_id}; investigate tool path"
            ),
            "symmetry_violation": True,
            "covered_runs": covered_runs,
        },
    )
