"""Shape-aware seed topic extraction (PR-A2 commit 3, drive posture).

Plan ref: plans/2026-05-05-hitl-followup.md commit 3.

Each shape gets its own branch:
  - mechanistic → ONE synthetic USER_MECHANISM topic (diag ignored)
  - comparative → ONE synthetic USER_COMPARISON topic (diag ignored)
  - scoping with overlap → filtered bias topics
  - scoping with NO overlap → ONE USER_SCOPE placeholder (D3)
  - open / shape=None → bias path (full extract_seed_topics)
"""

from __future__ import annotations

from datetime import datetime, timezone
from uuid import UUID

from fermdocs.domain.user_question import UserQuestion
from fermdocs_characterize.schema import Severity
from fermdocs_diagnose.schema import (
    ConfidenceBasis,
    DiagnosisMeta,
    DiagnosisOutput,
    FailureClaim,
    TrajectoryRef,
)
from fermdocs_hypothesis.schema import TopicSourceType
from fermdocs_hypothesis.seed_topic_extractor import (
    extract_seed_topics_for_followup,
)

CHAR_ID = UUID("aaaaaaaa-aaaa-aaaa-aaaa-aaaaaaaaaaaa")
DIAG_ID = UUID("bbbbbbbb-bbbb-bbbb-bbbb-bbbbbbbbbbbb")


def _diag(failures=()) -> DiagnosisOutput:
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
        failures=list(failures),
        analysis=[],
        trends=[],
        open_questions=[],
    )


def _failure(idx: int, *, var: str, run_id: str, severity=Severity.MAJOR) -> FailureClaim:
    return FailureClaim(
        claim_id=f"D-F-{idx:04d}",
        summary=f"failure on {var} in {run_id}",
        cited_finding_ids=[f"{CHAR_ID}:F-{idx:04d}"],
        cited_trajectories=[TrajectoryRef(run_id=run_id, variable=var)],
        affected_variables=[var],
        confidence=0.85,
        confidence_basis=ConfidenceBasis.SCHEMA_ONLY,
        severity=severity,
    )


# ---------- mechanistic shape ----------


def test_mechanistic_emits_single_synthetic_topic() -> None:
    diag = _diag(failures=[_failure(1, var="biomass", run_id="RUN-0001")])
    q = UserQuestion(
        text="Was the RQ peak caused by oxygen limitation?",
        shape="mechanistic",
        raised_by="user_followup",
    )

    topics = extract_seed_topics_for_followup(diag, question=q)

    assert len(topics) == 1
    t = topics[0]
    assert t.source_type == TopicSourceType.USER_MECHANISM
    assert t.summary == q.text
    assert t.priority == 1.0


def test_mechanistic_ignores_diag_failures() -> None:
    """Drive-posture mechanistic mode discards diag-derived topics —
    the user's mechanism IS the topic."""
    diag = _diag(
        failures=[
            _failure(1, var="biomass", run_id="RUN-0001", severity=Severity.CRITICAL),
            _failure(2, var="DO", run_id="RUN-0001", severity=Severity.CRITICAL),
        ]
    )
    q = UserQuestion(text="Was X the cause?", shape="mechanistic")

    topics = extract_seed_topics_for_followup(diag, question=q)

    assert len(topics) == 1
    assert topics[0].source_type == TopicSourceType.USER_MECHANISM


def test_mechanistic_carries_question_affected_variables() -> None:
    q = UserQuestion(
        text="Was glucose limitation the cause?",
        shape="mechanistic",
        affected_variables=["glucose_g_l", "biomass_g_l"],
    )

    topics = extract_seed_topics_for_followup(_diag(), question=q)

    assert topics[0].affected_variables == ["glucose_g_l", "biomass_g_l"]


# ---------- comparative shape ----------


def test_comparative_emits_single_synthetic_topic() -> None:
    diag = _diag(failures=[_failure(1, var="biomass", run_id="RUN-0001")])
    q = UserQuestion(
        text="Compare RUN-0001 to RUN-0002 on RQ behavior",
        shape="comparative",
        affected_runs=["RUN-0001", "RUN-0002"],
    )

    topics = extract_seed_topics_for_followup(diag, question=q)

    assert len(topics) == 1
    assert topics[0].source_type == TopicSourceType.USER_COMPARISON
    assert topics[0].priority == 1.0


def test_comparative_ignores_diag() -> None:
    diag = _diag(
        failures=[
            _failure(1, var="biomass", run_id="RUN-0001", severity=Severity.CRITICAL),
        ]
    )
    q = UserQuestion(text="Compare A to B", shape="comparative")

    topics = extract_seed_topics_for_followup(diag, question=q)

    assert len(topics) == 1
    assert topics[0].source_type == TopicSourceType.USER_COMPARISON


# ---------- scoping shape with overlap ----------


def test_scoping_with_variable_overlap_returns_filtered_bias_topics() -> None:
    diag = _diag(
        failures=[
            _failure(1, var="biomass_g_l", run_id="RUN-0001"),
            _failure(2, var="paa_mg_l", run_id="RUN-0002"),
        ]
    )
    q = UserQuestion(
        text="What's happening with biomass?",
        shape="scoping",
        affected_variables=["biomass_g_l"],
    )

    topics = extract_seed_topics_for_followup(diag, question=q)

    # Only the biomass topic survives the scope filter
    assert len(topics) == 1
    assert "biomass_g_l" in topics[0].affected_variables
    # Bias-posture priority bump still applies
    assert topics[0].priority > 0.5
    # Source type is still FAILURE (it's a real diag topic, not synthetic)
    assert topics[0].source_type == TopicSourceType.FAILURE


def test_scoping_with_run_overlap_filters_to_run() -> None:
    diag = _diag(
        failures=[
            _failure(1, var="biomass_g_l", run_id="RUN-0001"),
            _failure(2, var="biomass_g_l", run_id="RUN-0002"),
        ]
    )
    q = UserQuestion(
        text="What about RUN-0002?",
        shape="scoping",
        affected_runs=["RUN-0002"],
    )

    topics = extract_seed_topics_for_followup(diag, question=q)

    assert len(topics) == 1
    runs = {ref.run_id for ref in topics[0].cited_trajectories}
    assert runs == {"RUN-0002"}


# ---------- scoping shape with NO overlap (D3) ----------


def test_scoping_empty_match_emits_user_scope_placeholder() -> None:
    diag = _diag(
        failures=[_failure(1, var="biomass_g_l", run_id="RUN-0001")]
    )
    q = UserQuestion(
        text="What about RUN-9999?",
        shape="scoping",
        affected_runs=["RUN-9999"],  # nonexistent
    )

    topics = extract_seed_topics_for_followup(diag, question=q)

    assert len(topics) == 1
    assert topics[0].source_type == TopicSourceType.USER_SCOPE
    assert "scope did not match" in topics[0].summary.lower()


def test_scoping_empty_match_var_emits_user_scope() -> None:
    diag = _diag(
        failures=[_failure(1, var="biomass_g_l", run_id="RUN-0001")]
    )
    q = UserQuestion(
        text="What about phosphorus?",
        shape="scoping",
        affected_variables=["paa_mg_l"],  # not in any topic
    )

    topics = extract_seed_topics_for_followup(diag, question=q)

    assert len(topics) == 1
    assert topics[0].source_type == TopicSourceType.USER_SCOPE


# ---------- open shape: bias path ----------


def test_open_shape_falls_back_to_bias_path() -> None:
    """open shape (or shape=None) reuses extract_seed_topics with the bump."""
    diag = _diag(
        failures=[
            _failure(1, var="biomass_g_l", run_id="RUN-0001"),
            _failure(2, var="paa_mg_l", run_id="RUN-0002"),
        ]
    )
    q = UserQuestion(
        text="Anything weird in this run?",
        shape="open",
        affected_variables=["biomass_g_l"],
    )

    topics = extract_seed_topics_for_followup(diag, question=q)

    # All diag topics surface; bumped one ranks higher
    assert len(topics) == 2
    biomass = next(t for t in topics if "biomass_g_l" in t.affected_variables)
    paa = next(t for t in topics if "paa_mg_l" in t.affected_variables)
    assert biomass.priority > paa.priority


def test_shape_none_falls_back_to_open_path() -> None:
    """When the classifier failed to determine a shape, treat as open
    so we don't drop topics."""
    diag = _diag(failures=[_failure(1, var="biomass_g_l", run_id="RUN-0001")])
    q = UserQuestion(text="huh?")  # shape defaults to None

    topics = extract_seed_topics_for_followup(diag, question=q)

    assert len(topics) == 1  # diag topic survives; nothing dropped
    assert topics[0].source_type == TopicSourceType.FAILURE


# ---------- new TopicSourceType enum members ----------


def test_topic_source_type_has_user_drive_members() -> None:
    """Pin the enum additions so we'd notice if someone removed them."""
    assert TopicSourceType.USER_MECHANISM.value == "user_mechanism"
    assert TopicSourceType.USER_COMPARISON.value == "user_comparison"
    assert TopicSourceType.USER_SCOPE.value == "user_scope"
