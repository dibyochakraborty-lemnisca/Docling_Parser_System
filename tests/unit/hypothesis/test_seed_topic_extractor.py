"""seed_topic_extractor — every claim type produces a SeedTopic with a
sensible severity and priority."""

from __future__ import annotations

from datetime import datetime
from uuid import UUID

import pytest

from fermdocs_characterize.schema import Severity, TimeWindow
from fermdocs_diagnose.schema import (
    AnalysisClaim,
    ConfidenceBasis,
    DiagnosisMeta,
    DiagnosisOutput,
    FailureClaim,
    OpenQuestion,
    TrajectoryRef,
    TrendClaim,
)
from fermdocs_hypothesis.schema import TopicSourceType
from fermdocs_hypothesis.seed_topic_extractor import extract_seed_topics

CHAR_ID = "00000000-0000-0000-0000-000000000042"


def _meta() -> DiagnosisMeta:
    return DiagnosisMeta(
        schema_version="1.0",
        diagnosis_version="v1.0.0",
        diagnosis_id=UUID(int=1),
        supersedes_characterization_id=UUID(int=42),
        generation_timestamp=datetime(2026, 5, 3),
        model="claude-opus-4-7",
        provider="gemini",
    )


def test_failure_becomes_seed_topic():
    diag = DiagnosisOutput(
        meta=_meta(),
        failures=[
            FailureClaim(
                claim_id="D-F-0001",
                summary="biomass plateau at 40h",
                cited_finding_ids=[f"{CHAR_ID}:F-0001"],
                affected_variables=["biomass_g_l"],
                confidence=0.7,
                confidence_basis=ConfidenceBasis.SCHEMA_ONLY,
                severity=Severity.MAJOR,
            ),
        ],
    )
    seeds = extract_seed_topics(diag)
    assert len(seeds) == 1
    assert seeds[0].source_type == TopicSourceType.FAILURE
    assert seeds[0].severity == Severity.MAJOR
    assert seeds[0].source_id == "D-F-0001"


def test_open_question_becomes_seed_topic():
    diag = DiagnosisOutput(
        meta=_meta(),
        open_questions=[
            OpenQuestion(
                question_id="D-Q-0001",
                question="why did DO crash at 30h?",
                why_it_matters="anomaly",
                cited_finding_ids=[f"{CHAR_ID}:F-0001"],
                answer_format_hint="free_text",
            ),
        ],
    )
    seeds = extract_seed_topics(diag)
    assert seeds[0].source_type == TopicSourceType.OPEN_QUESTION


def test_critical_failures_get_higher_priority_than_minor():
    diag = DiagnosisOutput(
        meta=_meta(),
        failures=[
            FailureClaim(
                claim_id="D-F-0001",
                summary="x",
                cited_finding_ids=[f"{CHAR_ID}:F-0001"],
                confidence=0.7,
                confidence_basis=ConfidenceBasis.SCHEMA_ONLY,
                severity=Severity.CRITICAL,
            ),
            FailureClaim(
                claim_id="D-F-0002",
                summary="y",
                cited_finding_ids=[f"{CHAR_ID}:F-0001"],
                confidence=0.7,
                confidence_basis=ConfidenceBasis.SCHEMA_ONLY,
                severity=Severity.MINOR,
            ),
        ],
    )
    seeds = extract_seed_topics(diag)
    crit = next(s for s in seeds if s.source_id == "D-F-0001")
    minor = next(s for s in seeds if s.source_id == "D-F-0002")
    assert crit.priority > minor.priority


def test_topic_ids_assigned_in_order():
    diag = DiagnosisOutput(
        meta=_meta(),
        failures=[
            FailureClaim(
                claim_id=f"D-F-000{i+1}",
                summary=f"f{i}",
                cited_finding_ids=[f"{CHAR_ID}:F-0001"],
                confidence=0.7,
                confidence_basis=ConfidenceBasis.SCHEMA_ONLY,
                severity=Severity.MAJOR,
            )
            for i in range(3)
        ],
    )
    seeds = extract_seed_topics(diag)
    assert [s.topic_id for s in seeds] == ["T-0001", "T-0002", "T-0003"]


def test_summary_truncated_to_200_chars():
    diag = DiagnosisOutput(
        meta=_meta(),
        failures=[
            FailureClaim(
                claim_id="D-F-0001",
                summary="x" * 500,
                cited_finding_ids=[f"{CHAR_ID}:F-0001"],
                confidence=0.7,
                confidence_basis=ConfidenceBasis.SCHEMA_ONLY,
                severity=Severity.MAJOR,
            ),
        ],
    )
    seeds = extract_seed_topics(diag)
    assert len(seeds[0].summary) == 200


# ---------- data_quality_caveat suppression ----------
#
# Carotenoid regression: when diagnose ran on a yeast PDF where the
# registry's only entry is Penicillium, every analysis claim was a
# data_quality_caveat ("registry doesn't match", "data is sparse").
# Those propagated to seed topics as T-0001, T-0002 — and hypothesis
# specialists then debated the registry mismatch as if it were the
# central biological question. Both hypotheses got rejected because
# they were arguing about data plumbing, not biology.
#
# data_quality_caveat is a meta-observation about the data, not a
# hypothesizable claim. Skip it at the topic level. cross_run_observation
# and phase_characterization are still legit topics and pass through.


def test_data_quality_caveat_analysis_suppressed():
    """A data_quality_caveat analysis must NOT become a seed topic."""
    diag = DiagnosisOutput(
        meta=_meta(),
        analysis=[
            AnalysisClaim(
                claim_id="D-A-0001",
                summary="The observed organism is Saccharomyces cerevisiae, "
                "which conflicts with the registered Penicillium chrysogenum.",
                cited_finding_ids=[f"{CHAR_ID}:F-0001"],
                affected_variables=[],
                confidence=0.85,
                confidence_basis=ConfidenceBasis.SCHEMA_ONLY,
                kind="data_quality_caveat",
            ),
        ],
    )
    seeds = extract_seed_topics(diag)
    assert seeds == [], (
        "data_quality_caveat analysis claims must not seed topics — they "
        "bias specialists toward arguing about data plumbing instead of biology"
    )


def test_cross_run_observation_analysis_still_seeds_topic():
    """cross_run_observation IS a hypothesizable claim — keep it."""
    diag = DiagnosisOutput(
        meta=_meta(),
        analysis=[
            AnalysisClaim(
                claim_id="D-A-0001",
                summary="Run-A reaches 304 OD while Run-B plateaus at 195",
                cited_finding_ids=[f"{CHAR_ID}:F-0001"],
                affected_variables=["od600_au"],
                confidence=0.8,
                confidence_basis=ConfidenceBasis.CROSS_RUN,
                kind="cross_run_observation",
            ),
        ],
    )
    seeds = extract_seed_topics(diag)
    assert len(seeds) == 1
    assert seeds[0].source_type == TopicSourceType.ANALYSIS
    assert seeds[0].source_id == "D-A-0001"


def test_spec_alignment_analysis_suppressed():
    """Carotenoid follow-up bug: agent learned to dodge the
    data_quality_caveat suppression by emitting the SAME meta-claim
    under kind=spec_alignment ('process specifications are missing').
    spec_alignment is structurally meta (about system configuration,
    not about the experiment) and must also be suppressed."""
    diag = DiagnosisOutput(
        meta=_meta(),
        analysis=[
            AnalysisClaim(
                claim_id="D-A-0001",
                summary="Process specifications are missing for all "
                "recorded variables, preventing automated deviation detection.",
                cited_finding_ids=[f"{CHAR_ID}:F-0001"],
                affected_variables=["od600_au"],
                confidence=0.85,
                confidence_basis=ConfidenceBasis.SCHEMA_ONLY,
                kind="spec_alignment",
            ),
        ],
    )
    seeds = extract_seed_topics(diag)
    assert seeds == [], (
        "spec_alignment analysis claims must not seed topics — same "
        "meta-claim concern as data_quality_caveat. Without this, the "
        "agent dodges suppression by relabeling."
    )


def test_phase_characterization_analysis_still_seeds_topic():
    """phase_characterization is also legit — exponential phase, decline
    phase, plateau — these are real topics specialists can discuss."""
    diag = DiagnosisOutput(
        meta=_meta(),
        analysis=[
            AnalysisClaim(
                claim_id="D-A-0001",
                summary="Exponential phase observed from 0-30h",
                cited_finding_ids=[f"{CHAR_ID}:F-0001"],
                affected_variables=["od600_au"],
                confidence=0.8,
                confidence_basis=ConfidenceBasis.CROSS_RUN,
                kind="phase_characterization",
            ),
        ],
    )
    seeds = extract_seed_topics(diag)
    assert len(seeds) == 1


# ---------- spec-only failure suppression (IndPenSim regression) ----------
#
# May 2026: when running on IndPenSim CSV (unknown_process, no priors),
# the diagnose agent emitted three FailureClaims that were pure
# measured-vs-nominal-spec deltas:
#
#   D-F-0001: "DO ... violating the nominal specification of 15 mg/L"
#   D-F-0002: "Temperature ... exceeding the nominal specification of 297 K"
#   D-F-0003: "Biomass ... exceeding nominal specification of 0.5 g/L by
#             over 470 sigma"
#
# All three: confidence_basis=SCHEMA_ONLY, no narrative or trajectory
# citations. Hypothesis specialists were forced to debate each one,
# correctly rejected (no real evidence beyond schema interpretation),
# and the run produced 3 rejected hypotheses + 0 finals.
#
# `_is_spec_only_failure` filters these at seed extraction so the
# debate never starts. Failures that mix spec language with narrative
# or trajectory citations are kept — real evidence makes them debatable.


def test_spec_only_failure_dropped_indpensim_biomass_shape():
    """The exact shape from run a5b7b43b → must not seed a topic."""
    diag = DiagnosisOutput(
        meta=_meta(),
        failures=[
            FailureClaim(
                claim_id="D-F-0001",
                summary=(
                    "Biomass measurements reached up to 24.7 g/L between "
                    "120h and 156h, exceeding the nominal specification of "
                    "0.5 g/L by over 470 sigma."
                ),
                cited_finding_ids=[f"{CHAR_ID}:F-0001"],
                affected_variables=["biomass_g_l"],
                confidence=0.85,
                confidence_basis=ConfidenceBasis.SCHEMA_ONLY,
                severity=Severity.CRITICAL,
            ),
        ],
    )
    seeds = extract_seed_topics(diag)
    assert seeds == [], (
        "spec-only FailureClaim must not seed a topic — these waste "
        "debate turns when specs are misaligned (unknown_process bundles)"
    )


def test_spec_only_failure_dropped_indpensim_do_shape():
    diag = DiagnosisOutput(
        meta=_meta(),
        failures=[
            FailureClaim(
                claim_id="D-F-0001",
                summary=(
                    "Dissolved oxygen consistently measured between 9.2 "
                    "and 14.0 mg/L, violating the nominal specification "
                    "of 15 mg/L."
                ),
                cited_finding_ids=[f"{CHAR_ID}:F-0001"],
                affected_variables=["dissolved_o2_mg_l"],
                confidence=0.85,
                confidence_basis=ConfidenceBasis.SCHEMA_ONLY,
                severity=Severity.MAJOR,
            ),
        ],
    )
    seeds = extract_seed_topics(diag)
    assert seeds == []


def test_spec_only_failure_with_setpoint_language_dropped():
    """Variant vocabulary: 'setpoint' is also a spec-frame word."""
    diag = DiagnosisOutput(
        meta=_meta(),
        failures=[
            FailureClaim(
                claim_id="D-F-0001",
                summary="Temperature held above the 30C setpoint for 12h",
                cited_finding_ids=[f"{CHAR_ID}:F-0001"],
                affected_variables=["temperature_k"],
                confidence=0.85,
                confidence_basis=ConfidenceBasis.SCHEMA_ONLY,
                severity=Severity.MAJOR,
            ),
        ],
    )
    seeds = extract_seed_topics(diag)
    assert seeds == []


def test_failure_with_spec_language_AND_narrative_citation_kept():
    """The narrative citation is the anchoring evidence — even though
    the summary mentions specs, this is debatable because there's a
    real operator-witnessed event behind it."""
    diag = DiagnosisOutput(
        meta=_meta(),
        failures=[
            FailureClaim(
                claim_id="D-F-0001",
                summary="DO crashed below nominal spec at 30h, mixer trip recorded",
                cited_finding_ids=[f"{CHAR_ID}:F-0001"],
                cited_narrative_ids=[f"{CHAR_ID}:N-0001"],
                affected_variables=["dissolved_o2_mg_l"],
                confidence=0.85,
                confidence_basis=ConfidenceBasis.SCHEMA_ONLY,
                severity=Severity.MAJOR,
            ),
        ],
    )
    seeds = extract_seed_topics(diag)
    assert len(seeds) == 1, (
        "narrative-cited failures stay even with spec language — the "
        "operator narrative is the real evidence"
    )


def test_failure_with_spec_language_AND_trajectory_citation_kept():
    """Same principle — a trajectory citation anchors the claim in real
    time-series data, so spec language doesn't disqualify it."""
    diag = DiagnosisOutput(
        meta=_meta(),
        failures=[
            FailureClaim(
                claim_id="D-F-0001",
                summary="Biomass exceeded nominal at 24h",
                cited_finding_ids=[f"{CHAR_ID}:F-0001"],
                cited_trajectories=[
                    TrajectoryRef(run_id="RUN-0001", variable="biomass_g_l")
                ],
                affected_variables=["biomass_g_l"],
                confidence=0.85,
                confidence_basis=ConfidenceBasis.SCHEMA_ONLY,
                severity=Severity.MAJOR,
            ),
        ],
    )
    seeds = extract_seed_topics(diag)
    assert len(seeds) == 1


def test_failure_with_process_priors_basis_kept_even_without_narrative():
    """When confidence_basis=process_priors, the claim was grounded in
    organism+process priors (not raw schema). These ARE debatable — the
    spec-only filter targets schema_only basis specifically."""
    diag = DiagnosisOutput(
        meta=_meta(),
        failures=[
            FailureClaim(
                claim_id="D-F-0001",
                summary="DO held below typical range vs spec",
                cited_finding_ids=[f"{CHAR_ID}:F-0001"],
                affected_variables=["dissolved_o2_mg_l"],
                confidence=0.85,
                confidence_basis=ConfidenceBasis.PROCESS_PRIORS,
                severity=Severity.MAJOR,
            ),
        ],
    )
    seeds = extract_seed_topics(diag)
    assert len(seeds) == 1


def test_failure_without_spec_language_kept():
    """No spec/nominal/sigma vocabulary → not spec-only, kept as topic.
    Sanity that the carotenoid-shape biological failures still flow."""
    diag = DiagnosisOutput(
        meta=_meta(),
        failures=[
            FailureClaim(
                claim_id="D-F-0001",
                summary="Cells lost pigmentation across all 6 batches by 72h",
                cited_finding_ids=[f"{CHAR_ID}:F-0001"],
                affected_variables=["pigmentation"],
                confidence=0.85,
                confidence_basis=ConfidenceBasis.SCHEMA_ONLY,
                severity=Severity.CRITICAL,
            ),
        ],
    )
    seeds = extract_seed_topics(diag)
    assert len(seeds) == 1


def test_spec_only_open_question_dropped():
    """'What is the correct specification for X?' open questions seed
    topics that route specialists into spec-arguing — same dodge as
    spec-only failures."""
    diag = DiagnosisOutput(
        meta=_meta(),
        open_questions=[
            OpenQuestion(
                question_id="D-Q-0001",
                question="What is the correct specification for dissolved_o2_mg_l?",
                why_it_matters=(
                    "Hundreds of observations violate the current 15 mg/L "
                    "nominal, which may be a setpoint rather than a range."
                ),
                cited_finding_ids=[f"{CHAR_ID}:F-0001"],
                answer_format_hint="numeric",
            ),
        ],
    )
    seeds = extract_seed_topics(diag)
    assert seeds == []


def test_indpensim_full_diagnosis_drops_to_empty_topics():
    """End-to-end shape from the real IndPenSim run: 3 spec-only
    failures + 2 spec_alignment analyses + 2 spec-only open questions
    → all filtered → 0 seed topics. This reproduces the exact
    a5b7b43b regression at the seed extraction layer."""
    diag = DiagnosisOutput(
        meta=_meta(),
        failures=[
            FailureClaim(
                claim_id="D-F-0001",
                summary="DO violating nominal specification of 15 mg/L",
                cited_finding_ids=[f"{CHAR_ID}:F-0001"],
                affected_variables=["dissolved_o2_mg_l"],
                confidence=0.85,
                confidence_basis=ConfidenceBasis.SCHEMA_ONLY,
                severity=Severity.MAJOR,
            ),
            FailureClaim(
                claim_id="D-F-0002",
                summary="Temperature exceeding nominal specification of 297 K",
                cited_finding_ids=[f"{CHAR_ID}:F-0002"],
                affected_variables=["temperature_k"],
                confidence=0.85,
                confidence_basis=ConfidenceBasis.SCHEMA_ONLY,
                severity=Severity.MAJOR,
            ),
            FailureClaim(
                claim_id="D-F-0003",
                summary="Biomass exceeding nominal specification by over 470 sigma",
                cited_finding_ids=[f"{CHAR_ID}:F-0003"],
                affected_variables=["biomass_g_l"],
                confidence=0.85,
                confidence_basis=ConfidenceBasis.SCHEMA_ONLY,
                severity=Severity.CRITICAL,
            ),
        ],
        analysis=[
            AnalysisClaim(
                claim_id="D-A-0001",
                summary="Spec misaligned with observed values",
                cited_finding_ids=[f"{CHAR_ID}:F-0001"],
                affected_variables=["biomass_g_l"],
                confidence=0.85,
                confidence_basis=ConfidenceBasis.SCHEMA_ONLY,
                kind="spec_alignment",
            ),
        ],
        open_questions=[
            OpenQuestion(
                question_id="D-Q-0001",
                question="What are the intended trajectory bounds for biomass_g_l?",
                why_it_matters="Observations exceed schema nominal by 470 sigma",
                cited_finding_ids=[f"{CHAR_ID}:F-0001"],
                answer_format_hint="numeric",
            ),
        ],
    )
    seeds = extract_seed_topics(diag)
    assert seeds == [], (
        "All-spec IndPenSim shape must produce zero seed topics — better "
        "than wasting debate cycles on schema interpretations"
    )


def test_topic_id_numbering_skips_suppressed():
    """When a data_quality_caveat is sandwiched between a cross_run and
    a trend, the failure → cross_run → trend numbering must be
    contiguous (T-0001, T-0002, T-0003) — the suppressed claim doesn't
    consume an ID."""
    diag = DiagnosisOutput(
        meta=_meta(),
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
                summary="Run-A diverges from Run-B post-30h",
                cited_finding_ids=[f"{CHAR_ID}:F-0001"],
                affected_variables=["od600_au"],
                confidence=0.8,
                confidence_basis=ConfidenceBasis.CROSS_RUN,
                kind="cross_run_observation",
            ),
        ],
        trends=[
            TrendClaim(
                claim_id="D-T-0001",
                summary="OD plateau 60-90h",
                cited_finding_ids=[],
                cited_trajectories=[
                    TrajectoryRef(run_id="RUN-0001", variable="od600_au")
                ],
                affected_variables=["od600_au"],
                confidence=0.7,
                confidence_basis=ConfidenceBasis.SCHEMA_ONLY,
                direction="plateau",
            ),
        ],
    )
    seeds = extract_seed_topics(diag)
    assert [s.topic_id for s in seeds] == ["T-0001", "T-0002", "T-0003"]
    assert [s.source_type for s in seeds] == [
        TopicSourceType.FAILURE,
        TopicSourceType.ANALYSIS,
        TopicSourceType.TREND,
    ]
    assert [s.source_id for s in seeds] == ["D-F-0001", "D-A-0002", "D-T-0001"]
