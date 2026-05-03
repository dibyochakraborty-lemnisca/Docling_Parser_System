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
