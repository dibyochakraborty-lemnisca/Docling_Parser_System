"""validate_finding: physicality validator.

Plan ref: plans/2026-05-07-characterize-determinism.md commit 3.

The IndPenSim feedback exposed: 'PAA yield 204.5 g/g' passed every
stage with no sanity check. validate_finding catches that class of bug.

Covers:
  1. valid yield (0.5 g/g) passes through unchanged
  2. yield > 1 g/g → data_gap with reason naming the violation;
     raw_invalid preserved for audit (the canonical IndPenSim case)
  3. negative yield → data_gap (sign bug)
  4. NaN values → data_gap
  5. RQ > 3 → data_gap (thermodynamically suspicious)
  6. percentage > 100 (frac_below_threshold > 1) → data_gap
  7. data_gap input passes through unchanged (idempotent)
  8. config_mismatch input passes through unchanged
  9. unknown statistics field passes through (no over-rejection)
"""

from __future__ import annotations

from fermdocs_characterize.agents.finding_validator import validate_finding
from fermdocs_characterize.schema import (
    EvidenceStrength,
    ExtractedVia,
    Finding,
    FindingType,
    Severity,
    Tier,
)

CHAR_ID = "00000000-0000-0000-0000-000000000001"


def _finding(
    metric_id: str,
    statistics: dict,
    *,
    pattern_kind: str = "computed_metric",
) -> Finding:
    return Finding(
        finding_id=f"{CHAR_ID}:F-0001",
        type=FindingType.KINETIC_ANOMALY,
        severity=Severity.MINOR,
        tier=Tier.A,
        summary=f"{metric_id} on RUN-1.",
        confidence=0.85,
        extracted_via=ExtractedVia.DETERMINISTIC,
        evidence_strength=EvidenceStrength(n_observations=10, n_independent_runs=1),
        evidence_observation_ids=["obs-1"],
        variables_involved=["biomass_g_l"],
        run_ids=["RUN-1"],
        statistics={
            "pattern_kind": pattern_kind,
            "metric_id": metric_id,
            "tier": "B",
            **statistics,
        },
    )


# ---------- 1. valid yield passes through ----------


def test_valid_yield_passes_through_unchanged() -> None:
    f = _finding("B6", {"yield_g_per_g": 0.5, "delta_p": 5.0, "delta_x": 10.0})
    out = validate_finding(f)
    assert out is f or (
        out.statistics["pattern_kind"] == "computed_metric"
        and out.statistics["yield_g_per_g"] == 0.5
    )
    assert out.statistics.get("raw_invalid") is None


# ---------- 2. yield > 1 (the canonical IndPenSim bug) ----------


def test_yield_above_one_converts_to_data_gap() -> None:
    """PAA yield 204.5 g/g — the bug from IndPenSim feedback. Yields
    > 1 g/g are non-physical (mass balance limit)."""
    f = _finding("B6", {"yield_g_per_g": 204.5, "delta_p": 5203.0, "delta_x": 25.4})
    out = validate_finding(f)
    assert out.statistics["pattern_kind"] == "data_gap"
    assert "yield_g_per_g" in out.statistics["reason"]
    assert "204.5" in out.statistics["reason"]
    # Original value preserved for audit
    assert out.statistics["raw_invalid"]["yield_g_per_g"] == 204.5
    # Severity stays INFO (data_gap is informational)
    assert out.severity == Severity.INFO
    assert out.summary.startswith("B6: computed value violated physical bounds")


# ---------- 3. negative yield (sign bug) ----------


def test_negative_yield_rejected() -> None:
    f = _finding("B6", {"yield_g_per_g": -0.3})
    out = validate_finding(f)
    assert out.statistics["pattern_kind"] == "data_gap"
    assert "-0.3" in out.statistics["reason"]


# ---------- 4. NaN ----------


def test_nan_value_rejected() -> None:
    nan = float("nan")
    f = _finding("B10", {"mean_rq": nan})
    out = validate_finding(f)
    assert out.statistics["pattern_kind"] == "data_gap"
    assert "NaN" in out.statistics["reason"]
    assert out.statistics["raw_invalid"]["mean_rq"] == "NaN"


# ---------- 5. RQ > 3 ----------


def test_rq_above_three_rejected() -> None:
    """Mean RQ > 3 means CO2 evolved 3x more than O2 consumed - very
    rare biology, more often a units/sign bug."""
    f = _finding("B10", {"mean_rq": 5.2})
    out = validate_finding(f)
    assert out.statistics["pattern_kind"] == "data_gap"
    assert "mean_rq" in out.statistics["reason"]


# ---------- 6. fraction > 1 ----------


def test_fraction_above_one_rejected() -> None:
    f = _finding("A14", {"frac_below_threshold": 1.5})
    out = validate_finding(f)
    assert out.statistics["pattern_kind"] == "data_gap"
    assert "frac_below_threshold" in out.statistics["reason"]


# ---------- 7. data_gap input passes through ----------


def test_data_gap_input_passes_through_unchanged() -> None:
    f = _finding(
        "A8",
        {"reason": "precondition not met"},
        pattern_kind="data_gap",
    )
    out = validate_finding(f)
    assert out is f, "data_gap should pass through without copy"


# ---------- 8. config_mismatch input passes through ----------


def test_config_mismatch_input_passes_through_unchanged() -> None:
    f = _finding(
        "P1",
        {"reason": "process_families.yaml routes ..."},
        pattern_kind="config_mismatch",
    )
    out = validate_finding(f)
    assert out is f


# ---------- 9. unknown statistics field passes through ----------


def test_unknown_field_passes_through() -> None:
    """Anything we don't have a bound for: leave alone. Over-rejection
    is worse than under-rejection."""
    f = _finding("X99", {"some_obscure_stat": 999999.0, "yield_g_per_g": 0.5})
    out = validate_finding(f)
    assert out.statistics["pattern_kind"] == "computed_metric"
