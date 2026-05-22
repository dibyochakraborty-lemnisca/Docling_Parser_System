"""Tests for the diagnose runtime narrative backstop.

When the diagnose agent dodges its own contract by emitting only
meta-kind analyses (data_quality_caveat / spec_alignment) while the
bundle has narrative_observations describing closure_events or
interventions, the runtime synthesizes deterministic FailureClaims
(from closure_events) and AnalysisClaims (from interventions) so the
diagnosis isn't empty and downstream hypothesis stage has real topics.

Generalisable: keys off narrative tags, not registry or organism.
Works for any process the narrative extractor processed.

Idempotent: if the agent emitted any failures or non-meta analyses,
this is a no-op.
"""

from __future__ import annotations

from datetime import datetime
from uuid import UUID

import pytest

from fermdocs_characterize.schema import (
    CharacterizationOutput,
    Meta,
    NarrativeObservation,
    NarrativeSourceLocator,
    NarrativeTag,
    Severity,
)
from fermdocs_diagnose.agent import (
    _META_ANALYSIS_KINDS,
    _is_meta_only_emit,
    _synthesize_narrative_backstop_if_needed,
)
from fermdocs_diagnose.schema import (
    AnalysisClaim,
    ConfidenceBasis,
    DiagnosisMeta,
    DiagnosisOutput,
    FailureClaim,
    TrajectoryRef,
    TrendClaim,
)

CHAR_ID = UUID("11111111-1111-1111-1111-111111111111")


# ---------- helpers ----------


def _diag_meta() -> DiagnosisMeta:
    return DiagnosisMeta(
        schema_version="1.0",
        diagnosis_version="v1.0.0",
        diagnosis_id=UUID(int=1),
        supersedes_characterization_id=CHAR_ID,
        generation_timestamp=datetime(2026, 5, 3),
        model="claude-opus-4-7",
        provider="gemini",
    )


def _empty_output() -> CharacterizationOutput:
    return CharacterizationOutput(
        meta=Meta(
            schema_version="2.0",
            characterization_version="v1.0.0",
            characterization_id=CHAR_ID,
            generation_timestamp=datetime(2026, 5, 3),
            source_dossier_ids=["EXP-TEST"],
        ),
        findings=[],
        narrative_observations=[],
    )


def _output_with_narratives(
    *, closures: int = 0, interventions: int = 0
) -> CharacterizationOutput:
    nobs: list[NarrativeObservation] = []
    counter = 0
    for i in range(closures):
        counter += 1
        nobs.append(
            NarrativeObservation(
                narrative_id=f"{CHAR_ID}:N-{counter:04d}",
                tag=NarrativeTag.CLOSURE_EVENT,
                text=f"Cultivation terminated at {72 + i}h, white cells observed.",
                source_locator=NarrativeSourceLocator(),
                run_id=f"RUN-000{i + 1}",
                time_h=72.0 + i,
                affected_variables=["wcw_g_l"],
                confidence=0.8,
                extraction_model="gemini-3.1-pro-preview",
            )
        )
    for i in range(interventions):
        counter += 1
        nobs.append(
            NarrativeObservation(
                narrative_id=f"{CHAR_ID}:N-{counter:04d}",
                tag=NarrativeTag.INTERVENTION,
                text="200 mL Isopropyl Myristate added at 24h.",
                source_locator=NarrativeSourceLocator(),
                run_id=f"RUN-000{i + 1}",
                time_h=24.0,
                affected_variables=[],
                confidence=0.85,
                extraction_model="gemini-3.1-pro-preview",
            )
        )
    return CharacterizationOutput(
        meta=Meta(
            schema_version="2.0",
            characterization_version="v1.0.0",
            characterization_id=CHAR_ID,
            generation_timestamp=datetime(2026, 5, 3),
            source_dossier_ids=["EXP-TEST"],
        ),
        findings=[],
        narrative_observations=nobs,
    )


def _meta_only_diagnosis() -> DiagnosisOutput:
    """Reproduces the exact carotenoid bug shape: failures=[], trends=[],
    one data_quality_caveat analysis, one spec_alignment analysis."""
    return DiagnosisOutput(
        meta=_diag_meta(),
        failures=[],
        trends=[],
        analysis=[
            AnalysisClaim(
                claim_id="D-A-0001",
                summary="The dataset is highly sparse.",
                cited_finding_ids=[f"{CHAR_ID}:F-0001"],
                affected_variables=[],
                confidence=0.85,
                confidence_basis=ConfidenceBasis.SCHEMA_ONLY,
                kind="data_quality_caveat",
            ),
            AnalysisClaim(
                claim_id="D-A-0002",
                summary="Process specifications are missing.",
                cited_finding_ids=[f"{CHAR_ID}:F-0001"],
                affected_variables=[],
                confidence=0.85,
                confidence_basis=ConfidenceBasis.SCHEMA_ONLY,
                kind="spec_alignment",
            ),
        ],
    )


# ---------- _is_meta_only_emit predicate ----------


def test_meta_only_detects_dual_kind_dodge():
    """Both data_quality_caveat AND spec_alignment present, no
    failures/trends → meta-only."""
    diag = _meta_only_diagnosis()
    assert _is_meta_only_emit(diag) is True


def test_meta_only_returns_false_when_failures_present():
    diag = DiagnosisOutput(
        meta=_diag_meta(),
        failures=[
            FailureClaim(
                claim_id="D-F-0001",
                summary="biomass plateau",
                cited_finding_ids=[f"{CHAR_ID}:F-0001"],
                affected_variables=["biomass_g_l"],
                confidence=0.7,
                confidence_basis=ConfidenceBasis.SCHEMA_ONLY,
                severity=Severity.MAJOR,
            ),
        ],
        trends=[],
        analysis=[],
    )
    assert _is_meta_only_emit(diag) is False


def test_meta_only_returns_false_when_cross_run_present():
    """A cross_run_observation is non-meta — saves the emit from being
    meta-only."""
    diag = DiagnosisOutput(
        meta=_diag_meta(),
        failures=[],
        trends=[],
        analysis=[
            AnalysisClaim(
                claim_id="D-A-0001",
                summary="data is sparse",
                cited_finding_ids=[f"{CHAR_ID}:F-0001"],
                affected_variables=[],
                confidence=0.85,
                confidence_basis=ConfidenceBasis.SCHEMA_ONLY,
                kind="data_quality_caveat",
            ),
            AnalysisClaim(
                claim_id="D-A-0002",
                summary="cross-run divergence at 30h",
                cited_finding_ids=[f"{CHAR_ID}:F-0001"],
                affected_variables=["od600_au"],
                confidence=0.8,
                confidence_basis=ConfidenceBasis.CROSS_RUN,
                kind="cross_run_observation",
            ),
        ],
    )
    assert _is_meta_only_emit(diag) is False


def test_meta_only_returns_true_when_emit_is_fully_empty():
    """Updated contract (May 2026): a fully-empty emit (failures=[],
    trends=[], analysis=[]) is now treated as a dodge case, not as an
    intentional 'no signal' verdict.

    Why the change: yeast/unknown_process bundles started producing
    fully-empty emits where the agent interpreted the meta-flags
    (sparse_data, specs_mostly_missing, unknown_process) as 'I have
    nothing to claim.' Hypothesis stage then exited no_topics_left.

    The predicate now flags these as meta-only-equivalent. The actual
    decision of whether to inject claims is the caller's:
    `_synthesize_narrative_backstop_if_needed` checks whether the
    bundle has narrative_observations to ground claims on. If there's
    nothing in the narrative either, the empty emit passes through
    unchanged (covered by test_backstop_no_op_when_no_actionable_narratives)."""
    diag = DiagnosisOutput(meta=_diag_meta(), failures=[], trends=[], analysis=[])
    assert _is_meta_only_emit(diag) is True


# ---------- backstop synthesis ----------


def test_backstop_synthesizes_failure_per_closure_event():
    """6 closure events → 6 synthesized FailureClaims, each citing a
    distinct narrative_id."""
    output = _output_with_narratives(closures=6)
    diag = _meta_only_diagnosis()
    new = _synthesize_narrative_backstop_if_needed(diag, output)
    assert len(new.failures) == 6
    cited = {fid for f in new.failures for fid in f.cited_narrative_ids}
    assert len(cited) == 6
    # Original meta analyses preserved
    assert len(new.analysis) == 2


def test_backstop_synthesizes_analysis_per_intervention():
    """1 intervention → 1 synthesized AnalysisClaim with kind=
    cross_run_observation citing the narrative_id."""
    output = _output_with_narratives(interventions=1)
    diag = _meta_only_diagnosis()
    new = _synthesize_narrative_backstop_if_needed(diag, output)
    # Original 2 meta + 1 synthesized non-meta
    assert len(new.analysis) == 3
    synth = new.analysis[-1]
    assert synth.kind == "cross_run_observation"
    assert len(synth.cited_narrative_ids) == 1
    assert synth.provenance_downgraded is True


def test_backstop_handles_mixed_narrative_tags():
    """Carotenoid shape: 6 closures + 1 intervention → 6 failures
    synthesized + 1 analysis synthesized. Original meta analyses kept
    so diagnose's audit trail isn't lost."""
    output = _output_with_narratives(closures=6, interventions=1)
    diag = _meta_only_diagnosis()
    new = _synthesize_narrative_backstop_if_needed(diag, output)
    assert len(new.failures) == 6
    # 2 original meta + 1 synthesized
    assert len(new.analysis) == 3


def test_backstop_idempotent_when_failures_already_present():
    """If the agent already emitted at least one failure, the backstop
    must not fire — idempotency / agent-first semantics."""
    output = _output_with_narratives(closures=6)
    diag = DiagnosisOutput(
        meta=_diag_meta(),
        failures=[
            FailureClaim(
                claim_id="D-F-0001",
                summary="real anomaly the agent found",
                cited_finding_ids=[f"{CHAR_ID}:F-0001"],
                affected_variables=["wcw_g_l"],
                confidence=0.8,
                confidence_basis=ConfidenceBasis.SCHEMA_ONLY,
                severity=Severity.MAJOR,
            ),
        ],
        trends=[],
        analysis=[],
    )
    new = _synthesize_narrative_backstop_if_needed(diag, output)
    # Unchanged
    assert len(new.failures) == 1
    assert new.failures[0].claim_id == "D-F-0001"


def test_backstop_idempotent_when_cross_run_already_present():
    """If the agent already emitted a non-meta analysis (e.g.,
    cross_run_observation), don't fire."""
    output = _output_with_narratives(closures=6)
    diag = DiagnosisOutput(
        meta=_diag_meta(),
        failures=[],
        trends=[],
        analysis=[
            AnalysisClaim(
                claim_id="D-A-0001",
                summary="cross-run pattern the agent did identify",
                cited_finding_ids=[f"{CHAR_ID}:F-0001"],
                affected_variables=["od600_au"],
                confidence=0.8,
                confidence_basis=ConfidenceBasis.CROSS_RUN,
                kind="cross_run_observation",
            ),
        ],
    )
    new = _synthesize_narrative_backstop_if_needed(diag, output)
    assert len(new.failures) == 0
    assert len(new.analysis) == 1


def test_backstop_no_op_when_no_actionable_narratives():
    """Agent emits meta-only AND there are no closure_events or
    interventions to ground claims on. Pass-through unchanged — we
    don't invent claims out of nothing."""
    output = _empty_output()
    diag = _meta_only_diagnosis()
    new = _synthesize_narrative_backstop_if_needed(diag, output)
    assert len(new.failures) == 0
    assert len(new.analysis) == 2  # original meta preserved


# ---------- yeast empty-emit regression (May 2026) ----------


def _empty_diagnosis() -> DiagnosisOutput:
    """Reproduces the yeast/unknown_process bug shape from run a5b7b43b:
    failures=[], trends=[], analysis=[], open_questions=[]. The agent
    interpreted unknown_process + sparse_data + specs_mostly_missing as
    'I have nothing to claim.' Hypothesis stage then exited
    no_topics_left."""
    return DiagnosisOutput(
        meta=_diag_meta(),
        failures=[],
        trends=[],
        analysis=[],
        open_questions=[],
        narrative=(
            "The system flagged the data as sparse and lacking specifications. "
            "Zero automated findings reported."
        ),
    )


def test_backstop_fires_on_empty_emit_when_closure_events_exist():
    """The yeast regression: agent emits zero claims, but the bundle has
    closure_events in narrative_observations. Backstop must inject
    FailureClaims so hypothesis stage has something to debate.

    Mirrors run a5b7b43b which exited no_topics_left at turn 0 with 0
    tokens spent — the symptom that prompted this fix."""
    output = _output_with_narratives(closures=3)
    diag = _empty_diagnosis()
    new = _synthesize_narrative_backstop_if_needed(diag, output)
    assert len(new.failures) == 3, (
        f"expected 3 synthesized failures from 3 closure_events, got "
        f"{len(new.failures)} — the empty-emit case is not firing"
    )
    cited = {fid for f in new.failures for fid in f.cited_narrative_ids}
    assert len(cited) == 3
    # All synthesized claims marked provenance_downgraded so the
    # validator + downstream UI know they came from the safety net.
    assert all(f.provenance_downgraded is True for f in new.failures)


def test_backstop_fires_on_empty_emit_when_interventions_exist():
    output = _output_with_narratives(interventions=2)
    diag = _empty_diagnosis()
    new = _synthesize_narrative_backstop_if_needed(diag, output)
    assert len(new.analysis) == 2
    assert all(a.kind == "cross_run_observation" for a in new.analysis)
    assert all(a.provenance_downgraded is True for a in new.analysis)


def test_backstop_passes_through_empty_emit_when_no_narratives():
    """Empty emit + empty narratives = nothing to ground a claim on.
    Backstop respects this — it only injects when narrative content
    exists. The downstream hypothesis stage will still exit
    no_topics_left in this case, but that's a reflection of upstream
    extraction having found nothing, not a dodge."""
    output = _empty_output()
    diag = _empty_diagnosis()
    new = _synthesize_narrative_backstop_if_needed(diag, output)
    assert len(new.failures) == 0
    assert len(new.analysis) == 0


def test_backstop_marks_synthesized_claims_as_provenance_downgraded():
    """Synthesized claims must carry provenance_downgraded=True so
    audit / dashboards can distinguish runtime-injected from
    agent-emitted."""
    output = _output_with_narratives(closures=2, interventions=1)
    diag = _meta_only_diagnosis()
    new = _synthesize_narrative_backstop_if_needed(diag, output)
    for f in new.failures:
        assert f.provenance_downgraded is True
    # Synthesized analysis (last one) must also be downgraded
    assert new.analysis[-1].provenance_downgraded is True


def test_backstop_failure_summary_includes_run_id_when_available():
    """Synthesized failure summaries should name the run_id where
    available so the hypothesis stage's topic carries useful context."""
    output = _output_with_narratives(closures=1)
    diag = _meta_only_diagnosis()
    new = _synthesize_narrative_backstop_if_needed(diag, output)
    summary = new.failures[0].summary
    assert "RUN-0001" in summary
    assert "72h" in summary or "72.0" in summary  # time tag


def test_backstop_caps_failure_confidence_below_agent_max():
    """Synthesized claims are deterministic narrative-citations, not
    LLM-judged inferences. Cap at 0.7 (below the agent's 0.85 max) so
    the rank ordering favors agent claims when both exist."""
    output = _output_with_narratives(closures=1)
    diag = _meta_only_diagnosis()
    new = _synthesize_narrative_backstop_if_needed(diag, output)
    assert new.failures[0].confidence == 0.7


# ---------- meta kinds set drift guard ----------


def test_agent_meta_kinds_match_seed_extractor_kinds():
    """Drift guard: the diagnose-agent meta-kinds set must stay in
    sync with seed_topic_extractor's suppression set. If a new meta
    kind is added in one place, this test fails until it's added in
    both."""
    from fermdocs_hypothesis.seed_topic_extractor import (
        _SUPPRESSED_ANALYSIS_KINDS,
    )

    assert _META_ANALYSIS_KINDS == _SUPPRESSED_ANALYSIS_KINDS, (
        "diagnose agent's _META_ANALYSIS_KINDS drifted from seed_topic_"
        "extractor's _SUPPRESSED_ANALYSIS_KINDS. Both define what counts "
        "as a 'meta' AnalysisClaim — they must match exactly. Update both."
    )
