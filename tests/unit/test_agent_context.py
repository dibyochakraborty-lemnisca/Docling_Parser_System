"""Tests for the AgentContext builder + serializer.

The AgentContext is what every future LLM agent reads as its primary
prompt prefix. These tests assert:
  - it always carries observed identity when present (the v4.1 yeast case)
  - it stays under the 1500-token budget on every existing fixture
  - rollups are computed at serialize time, not stored
  - oversized inputs gracefully truncate findings rather than overflow
"""

from __future__ import annotations

import json
import logging
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any

import pytest

from fermdocs_characterize.agent_context import (
    DEFAULT_MAX_TOKENS,
    MAX_TOP_FINDINGS,
    AgentContext,
    build_agent_context,
    serialize_for_agent,
)
from fermdocs_characterize.flags import ProcessFlag
from fermdocs_characterize.pipeline import CharacterizationPipeline

FIXTURES_DIR = Path(__file__).resolve().parents[2] / "evals" / "characterize" / "fixtures"


# ---------- Fixture-based smoke and budget checks ----------


@pytest.mark.parametrize(
    "fixture_name", ["01_boundary", "02_missing_data", "03_multi_run"]
)
def test_existing_fixtures_serialize_under_budget(fixture_name: str):
    dossier = json.loads((FIXTURES_DIR / fixture_name / "dossier.json").read_text())
    output = CharacterizationPipeline(validate=False).run(
        dossier,
        characterization_id=uuid.UUID(int=0),
        generation_timestamp=datetime(2026, 1, 1),
    )
    ctx = build_agent_context(dossier, output)
    rendered = serialize_for_agent(ctx, output)
    approx_tokens = len(rendered) // 4
    assert approx_tokens <= DEFAULT_MAX_TOKENS, (
        f"{fixture_name}: ~{approx_tokens} tokens exceeds budget {DEFAULT_MAX_TOKENS}"
    )


# ---------- Identity surfacing ----------


def _minimal_output(finding_count: int = 0):
    """Build a tiny CharacterizationOutput for tests that don't need findings."""
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

    char_id = uuid.UUID(int=42)
    findings = [
        Finding(
            finding_id=f"{char_id}:F-{i+1:04d}",
            type=FindingType.RANGE_VIOLATION,
            severity=Severity.MINOR,
            summary=f"finding {i}",
            confidence=0.85,
            extracted_via=ExtractedVia.DETERMINISTIC,
            evidence_strength=EvidenceStrength(
                n_observations=1, n_independent_runs=1
            ),
            evidence_observation_ids=["O-1"],
            variables_involved=["biomass_g_l"],
            time_window=TimeWindow(start=0, end=1),
        )
        for i in range(finding_count)
    ]
    meta = Meta(
        schema_version="2.0",
        characterization_version="v1.0.0",
        characterization_id=char_id,
        generation_timestamp=datetime(2026, 1, 1),
        source_dossier_ids=["EXP-TEST"],
    )
    return CharacterizationOutput(meta=meta, findings=findings)


def _dossier_with_identity(
    *,
    organism: str | None = "Penicillium chrysogenum",
    process_id: str | None = "penicillin_indpensim",
    registered_provenance: str = "llm_whitelisted",
) -> dict[str, Any]:
    return {
        "experiment": {
            "experiment_id": "EXP-TEST",
            "process": {
                "observed": {
                    "organism": organism,
                    "product": "penicillin" if organism else None,
                    "provenance": "llm_whitelisted" if organism else "unknown",
                },
                "registered": {
                    "process_id": process_id,
                    "provenance": registered_provenance,
                },
            },
        },
        "ingestion_summary": {
            "schema_version": "2.0",
            "stale_schema_versions": [],
            "golden_coverage_percent": 75,
        },
        "golden_columns": {},
    }


def test_observed_organism_surfaces_when_registered_unknown():
    """v4.1 regression guard: yeast-style case where registry doesn't match
    but organism is in the dossier.
    """
    dossier = _dossier_with_identity(
        organism="Saccharomyces cerevisiae",
        process_id=None,
        registered_provenance="unknown",
    )
    ctx = build_agent_context(dossier, _minimal_output())
    assert ctx.process["observed"]["organism"] == "Saccharomyces cerevisiae"
    assert ctx.process["registered"]["process_id"] is None
    assert ProcessFlag.UNKNOWN_PROCESS in ctx.flags
    assert ProcessFlag.UNKNOWN_ORGANISM not in ctx.flags


def test_unknown_organism_flag_when_no_organism():
    dossier = _dossier_with_identity(
        organism=None,
        process_id=None,
        registered_provenance="unknown",
    )
    ctx = build_agent_context(dossier, _minimal_output())
    assert ProcessFlag.UNKNOWN_ORGANISM in ctx.flags
    assert ProcessFlag.UNKNOWN_PROCESS in ctx.flags


# ---------- Serialize-time rollups, not stored ----------


def test_severity_rollup_computed_at_serialize_time():
    """AgentContext stores raw finding_ids; rollup happens on serialize.
    This makes the model reusable across prompt formats without drift.
    """
    dossier = _dossier_with_identity()
    output = _minimal_output(finding_count=3)
    ctx = build_agent_context(dossier, output)

    # Stored shape: only ids, no rollup attribute.
    assert isinstance(ctx.finding_ids, list)
    assert len(ctx.finding_ids) == 3
    assert not hasattr(ctx, "n_findings_by_severity")
    assert not hasattr(ctx, "top_findings")

    # Serialize: rollup appears.
    rendered = serialize_for_agent(ctx, output)
    parsed = json.loads(rendered)
    assert parsed["findings"]["n_total"] == 3
    assert parsed["findings"]["by_severity"]["minor"] == 3
    assert len(parsed["findings"]["top"]) == 3


def test_serialize_caps_top_findings_at_max():
    dossier = _dossier_with_identity()
    output = _minimal_output(finding_count=MAX_TOP_FINDINGS + 5)
    ctx = build_agent_context(dossier, output)
    parsed = json.loads(serialize_for_agent(ctx, output))
    assert len(parsed["findings"]["top"]) == MAX_TOP_FINDINGS
    assert parsed["findings"]["n_total"] == MAX_TOP_FINDINGS + 5


def test_serialize_no_findings():
    dossier = _dossier_with_identity()
    output = _minimal_output(finding_count=0)
    ctx = build_agent_context(dossier, output)
    parsed = json.loads(serialize_for_agent(ctx, output))
    assert parsed["findings"]["n_total"] == 0
    assert parsed["findings"]["top"] == []


# ---------- Runtime truncation ----------


def test_serialize_truncates_when_oversized(caplog):
    """Tiny token budget -> drop findings until it fits, log a warning."""
    dossier = _dossier_with_identity()
    output = _minimal_output(finding_count=5)
    ctx = build_agent_context(dossier, output)
    with caplog.at_level(logging.WARNING):
        rendered = serialize_for_agent(ctx, output, max_tokens=100)
    parsed = json.loads(rendered)
    # At least some findings dropped.
    assert len(parsed["findings"]["top"]) < 5
    # Warning was logged.
    assert any("truncated" in rec.message.lower() for rec in caplog.records)


def test_serialize_drops_low_severity_first(caplog):
    """Truncation order: lowest severity first."""
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

    char_id = uuid.UUID(int=99)
    findings = []
    severities = [Severity.CRITICAL, Severity.MAJOR, Severity.MINOR, Severity.INFO]
    for i, sev in enumerate(severities):
        findings.append(
            Finding(
                finding_id=f"{char_id}:F-{i+1:04d}",
                type=FindingType.RANGE_VIOLATION,
                severity=sev,
                summary=f"finding-{sev.value}",
                confidence=0.85,
                extracted_via=ExtractedVia.DETERMINISTIC,
                evidence_strength=EvidenceStrength(
                    n_observations=1, n_independent_runs=1
                ),
                evidence_observation_ids=["O-1"],
                variables_involved=["x"],
                time_window=TimeWindow(start=0, end=1),
            )
        )
    output = CharacterizationOutput(
        meta=Meta(
            schema_version="2.0",
            characterization_version="v1.0.0",
            characterization_id=char_id,
            generation_timestamp=datetime(2026, 1, 1),
            source_dossier_ids=["EXP-X"],
        ),
        findings=findings,
    )
    dossier = _dossier_with_identity()
    ctx = build_agent_context(dossier, output)

    # Tight budget that forces dropping ~half.
    rendered = serialize_for_agent(ctx, output, max_tokens=120)
    parsed = json.loads(rendered)
    kept_severities = [f["severity"] for f in parsed["findings"]["top"]]
    # Whatever survived, info should be the first to go before critical.
    if "info" in kept_severities and "critical" not in kept_severities:
        pytest.fail(
            "Truncation kept INFO but dropped CRITICAL; ranking is wrong"
        )


# ---------- Stable serialization ----------


def test_serialize_produces_stable_output_for_identical_input():
    """Byte-stable serialization is what makes prompt caching work."""
    dossier = _dossier_with_identity()
    output = _minimal_output(finding_count=2)
    ctx1 = build_agent_context(dossier, output)
    ctx2 = build_agent_context(dossier, output)
    assert serialize_for_agent(ctx1, output) == serialize_for_agent(ctx2, output)
