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
