"""Regression tests covering two AgentContext bugs found during the IndPenSim
end-to-end run.

1. AgentContext defaulted to DictSpecsProvider.from_dossier (which only reads
   `_specs` from the dossier). Real ingestion dossiers don't carry _specs,
   so AgentContext built a summary with no specs even though the
   CharacterizationPipeline used schema-based specs. Result:
   variables_with_specs=[] always, and the SPECS_MOSTLY_MISSING flag fired
   incorrectly. Fix: AgentContext now uses from_schema_with_overrides by
   default, matching the pipeline's resolution.

2. _rank_finding_ids tied within severity by finding_id alone, which is
   sigma-ordered. A per-row biomass violation (sigma=484, n=1) outranked a
   substrate aggregated rollup (sigma=428, n=2242) even though the rollup
   carries strictly more information. Fix: aggregated rollups now win
   within the same severity tier, before per-row tiebreakers.
"""

from __future__ import annotations

import json
import uuid
from datetime import datetime
from pathlib import Path

from fermdocs_characterize.agent_context import (
    _rank_finding_ids,
    build_agent_context,
)
from fermdocs_characterize.flags import ProcessFlag
from fermdocs_characterize.pipeline import CharacterizationPipeline
from fermdocs_characterize.schema import (
    CharacterizationOutput,
    EvidenceStrength,
    ExtractedVia,
    Finding,
    FindingType,
    Meta,
    Severity,
    TimeWindow,
)


# ---------- Issue 1: AgentContext uses schema specs ----------


def _real_ingestion_dossier_shape() -> dict:
    """A dossier matching what fermdocs.dossier.build_dossier emits — has
    golden_columns observations but NO _specs key. Mirrors what IndPenSim
    looked like.
    """
    char_id = uuid.UUID(int=42)
    return {
        "dossier_schema_version": "1.1",
        "experiment": {
            "experiment_id": "EXP-X",
            "process": {
                "observed": {
                    "organism": "Penicillium chrysogenum",
                    "provenance": "manifest",
                },
                "registered": {
                    "process_id": "penicillin_indpensim",
                    "provenance": "manifest",
                },
            },
        },
        "ingestion_summary": {
            "schema_version": "2.0",
            "stale_schema_versions": [],
            "golden_coverage_percent": 80,
            "total_observations": 4,
        },
        "golden_columns": {
            "biomass_g_l": {
                "canonical_unit": "g/L",
                "observations": [
                    _obs(0, 1.0, "biomass_g_l", t=10.0),
                    _obs(1, 1.5, "biomass_g_l", t=20.0),
                ],
            },
            "ph": {
                "canonical_unit": "pH",
                "observations": [
                    _obs(2, 6.5, "ph", t=10.0, unit="pH"),
                    _obs(3, 6.6, "ph", t=20.0, unit="pH"),
                ],
            },
        },
    }


def _obs(idx: int, value: float, var: str, *, t: float, unit: str = "g/L") -> dict:
    return {
        "observation_id": f"O-{idx:04d}",
        "value": value,
        "unit": unit,
        "value_raw": str(value),
        "unit_raw": unit,
        "observation_type": "measured",
        "confidence": {"mapping": 0.9, "extraction": 1.0, "combined": 0.9},
        "needs_review": False,
        "source": {
            "file_id": "F-1",
            "raw_header": var,
            "locator": {"section": "table", "run_id": "RUN-0001", "timestamp_h": t},
        },
        "conversion_status": "ok",
        "extractor_version": "v0.1.0",
        "via": "pint",
    }


def _empty_output() -> CharacterizationOutput:
    return CharacterizationOutput(
        meta=Meta(
            schema_version="2.0",
            characterization_version="v1.0.0",
            characterization_id=uuid.UUID(int=42),
            generation_timestamp=datetime(2026, 5, 2),
            source_dossier_ids=["EXP-X"],
        ),
    )


def test_agent_context_picks_up_schema_specs_for_real_dossier():
    """variables_with_specs must populate from the schema YAML for ingestion
    dossiers that lack _specs. This is the IndPenSim regression: every
    measurement variable was incorrectly classified as 'no specs'.
    """
    dossier = _real_ingestion_dossier_shape()
    output = _empty_output()

    ctx = build_agent_context(dossier, output)

    # biomass_g_l and ph both have nominal+std_dev in golden_schema.yaml
    assert "biomass_g_l" in ctx.variables_with_specs
    assert "ph" in ctx.variables_with_specs
    assert ctx.variables_without_specs == []


def test_specs_mostly_missing_flag_does_not_fire_when_schema_provides():
    """Regression: with schema-based specs, the SPECS_MOSTLY_MISSING flag
    must not fire on a dossier that has no _specs. Pre-fix it always fired.
    """
    dossier = _real_ingestion_dossier_shape()
    output = _empty_output()
    ctx = build_agent_context(dossier, output)
    assert ProcessFlag.SPECS_MOSTLY_MISSING not in ctx.flags


def test_explicit_specs_provider_still_wins():
    """An explicit specs_provider must override the schema default. This is
    how production tests inject a controlled provider.
    """
    from fermdocs_characterize.specs import DictSpecsProvider

    dossier = _real_ingestion_dossier_shape()
    output = _empty_output()
    empty_provider = DictSpecsProvider({})
    ctx = build_agent_context(dossier, output, specs_provider=empty_provider)
    assert ctx.variables_with_specs == []


# ---------- Issue 2: ranking prefers rollups within severity ----------


def _finding(
    fid: str, severity: Severity, *, aggregated: bool = False
) -> Finding:
    stats = {"sigmas": 100.0}
    if aggregated:
        stats = {"max_abs_sigmas": 50.0, "n_violations": 2000, "aggregated": True}
    return Finding(
        finding_id=fid,
        type=FindingType.RANGE_VIOLATION,
        severity=severity,
        summary=f"finding {fid}",
        confidence=0.9,
        extracted_via=ExtractedVia.DETERMINISTIC,
        evidence_strength=EvidenceStrength(n_observations=1, n_independent_runs=1),
        evidence_observation_ids=["O-1"],
        variables_involved=["x"],
        time_window=TimeWindow(start=0, end=1),
        statistics=stats,
    )


def test_aggregated_rollup_outranks_per_row_within_same_severity():
    """The substrate rollup (n=2242, sigma=428, aggregated) must win over a
    per-row biomass finding (sigma=484, not aggregated) when both are
    critical. Without this fix, the agent's top-10 was 10 biomass per-row
    findings and zero rollups, even though rollups carry more signal.
    """
    char_id = uuid.UUID(int=42)
    per_row_id = f"{char_id}:F-0001"  # high-sigma per-row
    rollup_id = f"{char_id}:F-0050"  # lower-sigma but aggregated
    by_id = {
        per_row_id: _finding(per_row_id, Severity.CRITICAL, aggregated=False),
        rollup_id: _finding(rollup_id, Severity.CRITICAL, aggregated=True),
    }
    ranked = _rank_finding_ids([per_row_id, rollup_id], by_id)
    assert ranked[0] == rollup_id, (
        "aggregated rollup should rank before per-row finding within same severity"
    )


def test_severity_still_dominates_aggregation():
    """A critical per-row finding must still rank above a major aggregated
    rollup. Severity is the primary key.
    """
    char_id = uuid.UUID(int=42)
    crit_per_row = f"{char_id}:F-0001"
    major_rollup = f"{char_id}:F-0010"
    by_id = {
        crit_per_row: _finding(crit_per_row, Severity.CRITICAL, aggregated=False),
        major_rollup: _finding(major_rollup, Severity.MAJOR, aggregated=True),
    }
    ranked = _rank_finding_ids([major_rollup, crit_per_row], by_id)
    assert ranked[0] == crit_per_row


def test_two_aggregated_rollups_tiebreak_by_id():
    """When both are aggregated rollups at the same severity, fall back to
    finding_id order — the pipeline already pre-sorts by sigma so the order
    is stable.
    """
    char_id = uuid.UUID(int=42)
    a = f"{char_id}:F-0001"
    b = f"{char_id}:F-0002"
    by_id = {
        a: _finding(a, Severity.CRITICAL, aggregated=True),
        b: _finding(b, Severity.CRITICAL, aggregated=True),
    }
    ranked = _rank_finding_ids([b, a], by_id)
    assert ranked == [a, b]


def test_two_per_row_findings_tiebreak_by_id():
    char_id = uuid.UUID(int=42)
    a = f"{char_id}:F-0001"
    b = f"{char_id}:F-0002"
    by_id = {
        a: _finding(a, Severity.CRITICAL, aggregated=False),
        b: _finding(b, Severity.CRITICAL, aggregated=False),
    }
    ranked = _rank_finding_ids([b, a], by_id)
    assert ranked == [a, b]


def test_missing_finding_id_does_not_crash_ranking():
    """If a finding_id has no entry in by_id (shouldn't happen in practice),
    the ranker must not raise.
    """
    char_id = uuid.UUID(int=42)
    real = f"{char_id}:F-0001"
    orphan = f"{char_id}:F-9999"
    by_id = {real: _finding(real, Severity.CRITICAL, aggregated=False)}
    ranked = _rank_finding_ids([real, orphan], by_id)
    assert real in ranked
    assert orphan in ranked
