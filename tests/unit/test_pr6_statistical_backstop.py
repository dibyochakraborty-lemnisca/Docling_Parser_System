"""Statistical-findings backstop: when diagnose emits zero non-meta
claims while the upstream characterize output has metric_id-tagged
trajectory_pattern findings, synthesize a TrendClaim per metric_id.

Production bug from runs a64b38c9 (PDF, 36 catalog findings) and
2305f6af (IndPenSim, 28 catalog findings): both runs produced rich
characterize output but diagnose emitted ~0-2 claims and shipped
mostly empty results to hypothesis. The narrative backstop only fires
on closure_event/intervention narratives; it didn't cover the case
where the only signal is statistical math.

Fix: extend _synthesize_narrative_backstop_if_needed to also synthesize
TrendClaims from catalog findings (one per metric_id, citing all
findings of that metric_id, with confidence_basis=statistical_toolkit
since the math is verified).
"""

from __future__ import annotations

from datetime import datetime, timezone
from uuid import UUID

from fermdocs_characterize.schema import (
    CharacterizationOutput,
    EvidenceStrength,
    ExtractedVia,
    Finding,
    FindingType,
    Meta,
    Severity,
    Tier,
)
from fermdocs_diagnose.agent import _synthesize_narrative_backstop_if_needed
from fermdocs_diagnose.schema import (
    ConfidenceBasis,
    DiagnosisMeta,
    DiagnosisOutput,
)

CHAR_ID = UUID("88888888-8888-8888-8888-888888888888")
DIAG_ID = UUID("99999999-9999-9999-9999-999999999999")


def _meta() -> Meta:
    return Meta(
        schema_version="1.0",
        characterization_version="0.1.0",
        characterization_id=CHAR_ID,
        generation_timestamp=datetime(2026, 5, 5, tzinfo=timezone.utc),
        source_dossier_ids=["dossier-test"],
    )


def _toolkit_finding(idx: int, metric_id: str, run_id: str = "RUN-0001") -> Finding:
    return Finding(
        finding_id=f"{CHAR_ID}:F-{idx:04d}",
        type=FindingType.TRAJECTORY_PATTERN,
        severity=Severity.MINOR,
        tier=Tier.A,
        summary=f"{metric_id} computed for {run_id}: mu_max=0.42",
        confidence=0.95,
        extracted_via=ExtractedVia.STATISTICAL,
        evidence_strength=EvidenceStrength(n_observations=10, n_independent_runs=1),
        evidence_observation_ids=[f"OBS-{idx:04d}"],
        variables_involved=["biomass_g_l"],
        run_ids=[run_id],
        statistics={"metric_id": metric_id, "tier": "A"},
    )


def _data_gap_finding(idx: int, metric_id: str) -> Finding:
    return Finding(
        finding_id=f"{CHAR_ID}:F-{idx:04d}",
        type=FindingType.TRAJECTORY_PATTERN,
        severity=Severity.INFO,
        tier=Tier.A,
        summary=f"{metric_id} skipped: missing input",
        confidence=0.5,
        extracted_via=ExtractedVia.STATISTICAL,
        evidence_strength=EvidenceStrength(n_observations=1, n_independent_runs=1),
        evidence_observation_ids=[f"OBS-{idx:04d}"],
        variables_involved=[],
        run_ids=[],
        statistics={"metric_id": metric_id, "pattern_kind": "data_gap"},
    )


def _empty_diagnosis() -> DiagnosisOutput:
    return DiagnosisOutput(
        meta=DiagnosisMeta(
            schema_version="1.0",
            diagnosis_version="0.1.0",
            diagnosis_id=DIAG_ID,
            supersedes_characterization_id=CHAR_ID,
            generation_timestamp=datetime(2026, 5, 5, tzinfo=timezone.utc),
            model="test",
            provider="gemini",
        ),
        failures=[],
        analysis=[],
        trends=[],
        open_questions=[],
    )


def _upstream(findings: list[Finding]) -> CharacterizationOutput:
    return CharacterizationOutput(
        meta=_meta(), findings=findings, narrative_observations=[], trajectories=[]
    )


# ---------- backstop fires on catalog findings ----------


def test_backstop_synthesizes_trend_per_metric_id_on_empty_emit() -> None:
    """The headline case: 5 metric_id-tagged findings, agent emitted
    nothing → backstop synthesizes one trend per metric_id."""
    upstream = _upstream(
        [
            _toolkit_finding(idx=1, metric_id="A8", run_id="RUN-0001"),
            _toolkit_finding(idx=2, metric_id="A8", run_id="RUN-0002"),
            _toolkit_finding(idx=3, metric_id="A9", run_id="RUN-0001"),
            _toolkit_finding(idx=4, metric_id="B16", run_id="RUN-0002"),
            _toolkit_finding(idx=5, metric_id="A11", run_id="RUN-0001"),
        ]
    )
    out = _synthesize_narrative_backstop_if_needed(_empty_diagnosis(), upstream)
    # 4 distinct metric_ids → 4 trends
    assert len(out.trends) == 4
    metric_ids_in_trends = {
        ln.split("Catalog metric ")[1].split(":")[0]
        for ln in [t.summary for t in out.trends]
    }
    assert metric_ids_in_trends == {"A8", "A9", "B16", "A11"}
    # Each trend gets STATISTICAL_TOOLKIT basis
    for t in out.trends:
        assert t.confidence_basis == ConfidenceBasis.STATISTICAL_TOOLKIT


def test_backstop_groups_multiple_runs_per_metric_id() -> None:
    """A8 fired on 2 runs → ONE trend citing both finding_ids,
    not two trends."""
    upstream = _upstream(
        [
            _toolkit_finding(idx=1, metric_id="A8", run_id="RUN-0001"),
            _toolkit_finding(idx=2, metric_id="A8", run_id="RUN-0002"),
        ]
    )
    out = _synthesize_narrative_backstop_if_needed(_empty_diagnosis(), upstream)
    assert len(out.trends) == 1
    [trend] = out.trends
    assert sorted(trend.cited_finding_ids) == [
        f"{CHAR_ID}:F-0001",
        f"{CHAR_ID}:F-0002",
    ]


def test_backstop_skips_data_gap_findings() -> None:
    """data_gap entries describe missing inputs, not anomalies. Backstop
    should ignore them — zero trends synthesized."""
    upstream = _upstream(
        [
            _data_gap_finding(idx=1, metric_id="B10"),
            _data_gap_finding(idx=2, metric_id="C5"),
        ]
    )
    out = _synthesize_narrative_backstop_if_needed(_empty_diagnosis(), upstream)
    assert out.trends == []
    # No real signal at all → empty diag stays empty (passed through).
    assert out.failures == []
    assert out.analysis == []


def test_backstop_no_op_when_agent_already_emitted_claims() -> None:
    """If the agent emitted any non-meta claim, backstop is a no-op
    even when statistical findings exist."""
    from fermdocs_diagnose.schema import TrendClaim

    upstream = _upstream(
        [_toolkit_finding(idx=1, metric_id="A8", run_id="RUN-0001")]
    )
    diag = _empty_diagnosis().model_copy(
        update={
            "trends": [
                TrendClaim(
                    claim_id="D-T-0001",
                    summary="agent emitted this",
                    cited_finding_ids=[f"{CHAR_ID}:F-0001"],
                    confidence=0.85,
                    confidence_basis=ConfidenceBasis.SCHEMA_ONLY,
                    affected_variables=["biomass_g_l"],
                    direction="increasing",
                )
            ]
        }
    )
    out = _synthesize_narrative_backstop_if_needed(diag, upstream)
    # Single agent-emitted trend stays; no synthesis (agent's emit was
    # not "meta-only or empty").
    assert len(out.trends) == 1
    assert out.trends[0].summary == "agent emitted this"


def test_backstop_skips_when_no_catalog_findings_no_narratives() -> None:
    """Empty bundle → empty diag passes through unchanged."""
    upstream = _upstream([])
    out = _synthesize_narrative_backstop_if_needed(_empty_diagnosis(), upstream)
    assert out.trends == []
    assert out.failures == []


def test_backstop_synthesized_trend_cites_findings_real_ids() -> None:
    """Sanity: every cited finding_id in a synthesized trend exists in
    upstream.findings — otherwise downstream validators reject."""
    upstream = _upstream(
        [
            _toolkit_finding(idx=1, metric_id="A8"),
            _toolkit_finding(idx=2, metric_id="A8"),
        ]
    )
    out = _synthesize_narrative_backstop_if_needed(_empty_diagnosis(), upstream)
    [trend] = out.trends
    upstream_ids = {f.finding_id for f in upstream.findings}
    for cited in trend.cited_finding_ids:
        assert cited in upstream_ids
