"""Seed topic priority bump when user question matches.

PR-A on caisc-hitl, commit 4. Plan ref:
plans/2026-05-04-user-question-and-hitl.md (S2: kept ranker as-is,
added small bump in seed_topic_extractor as the safety net).

Invariants:
- Back-compat: when user_question=None (or omitted), output is unchanged.
- Bump only fires on relevance > 0 (variable, run, or text overlap).
- Bump never starves critical real anomalies — capped at 1.0 absolute.
- Bump is multiplicative (1.3x) so high-priority topics don't get a free
  ride to 1.0; they're already there.
"""

from __future__ import annotations

from datetime import datetime, timezone
from uuid import UUID

from fermdocs.domain.user_question import UserQuestion
from fermdocs_characterize.schema import Severity
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
from fermdocs_hypothesis.seed_topic_extractor import (
    USER_QUESTION_PRIORITY_MULTIPLIER,
    extract_seed_topics,
)

CHAR_ID = UUID("aaaaaaaa-aaaa-aaaa-aaaa-aaaaaaaaaaaa")
DIAG_ID = UUID("bbbbbbbb-bbbb-bbbb-bbbb-bbbbbbbbbbbb")


def _diag(failures=(), analyses=(), trends=(), open_qs=()) -> DiagnosisOutput:
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
        analysis=list(analyses),
        trends=list(trends),
        open_questions=list(open_qs),
    )


def _failure(idx: int, *, severity: Severity, var: str, run_id: str) -> FailureClaim:
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


# ---------- back-compat ----------


def test_extract_seed_topics_unchanged_when_no_user_question() -> None:
    diag = _diag(failures=[_failure(1, severity=Severity.MAJOR, var="biomass_g_l", run_id="RUN-0001")])
    topics_no_q = extract_seed_topics(diag)
    topics_explicit_none = extract_seed_topics(diag, user_question=None)
    assert len(topics_no_q) == 1 == len(topics_explicit_none)
    assert topics_no_q[0].priority == topics_explicit_none[0].priority


# ---------- bump fires on overlap ----------


def test_bump_fires_on_variable_overlap() -> None:
    diag = _diag(
        failures=[_failure(1, severity=Severity.MINOR, var="biomass_g_l", run_id="RUN-0001")]
    )
    no_bump = extract_seed_topics(diag)[0]
    q = UserQuestion(text="Why?", affected_variables=["biomass_g_l"])
    bumped = extract_seed_topics(diag, user_question=q)[0]
    assert bumped.priority > no_bump.priority
    expected = min(no_bump.priority * USER_QUESTION_PRIORITY_MULTIPLIER, 1.0)
    assert bumped.priority == expected


def test_bump_fires_on_run_overlap() -> None:
    diag = _diag(
        failures=[_failure(1, severity=Severity.MINOR, var="biomass_g_l", run_id="RUN-0002")]
    )
    no_bump = extract_seed_topics(diag)[0]
    q = UserQuestion(text="?", affected_runs=["RUN-0002"])
    bumped = extract_seed_topics(diag, user_question=q)[0]
    assert bumped.priority > no_bump.priority


def test_bump_does_not_fire_on_no_overlap() -> None:
    diag = _diag(
        failures=[_failure(1, severity=Severity.MINOR, var="biomass_g_l", run_id="RUN-0001")]
    )
    no_bump = extract_seed_topics(diag)[0]
    q = UserQuestion(
        text="What about phosphorus?",
        affected_variables=["paa_mg_l"],
        affected_runs=["RUN-0099"],
    )
    not_bumped = extract_seed_topics(diag, user_question=q)[0]
    assert not_bumped.priority == no_bump.priority


# ---------- non-starvation ----------


def test_bump_capped_at_1_0() -> None:
    """A high-priority topic (critical severity, lots of citations) is
    already near 1.0; the bump must not exceed 1.0."""
    diag = _diag(
        failures=[_failure(1, severity=Severity.CRITICAL, var="biomass_g_l", run_id="RUN-0002")]
    )
    no_bump = extract_seed_topics(diag)[0]
    q = UserQuestion(
        text="?",
        affected_variables=["biomass_g_l"],
        affected_runs=["RUN-0002"],
    )
    bumped = extract_seed_topics(diag, user_question=q)[0]
    assert bumped.priority <= 1.0
    # Specifically: 0.5 + 0.4*1.0 + 0.1*small = 0.92ish, * 1.3 = 1.196 → cap to 1.0
    assert bumped.priority == 1.0


def test_critical_unrelated_outranks_bumped_minor() -> None:
    """The non-starvation invariant: a critical-severity topic that
    DOESN'T match the question still outranks a minor topic that DOES
    match — the question bumps preferences, doesn't invert priorities."""
    diag = _diag(
        failures=[
            _failure(1, severity=Severity.CRITICAL, var="paa_mg_l", run_id="RUN-9999"),
            _failure(2, severity=Severity.MINOR, var="biomass_g_l", run_id="RUN-0002"),
        ]
    )
    q = UserQuestion(
        text="?",
        affected_variables=["biomass_g_l"],
        affected_runs=["RUN-0002"],
    )
    topics = extract_seed_topics(diag, user_question=q)
    critical_topic = next(t for t in topics if "paa_mg_l" in t.affected_variables)
    minor_topic = next(t for t in topics if "biomass_g_l" in t.affected_variables)
    assert critical_topic.priority > minor_topic.priority


# ---------- text-substring relevance ----------


def test_bump_fires_when_question_text_mentions_variable() -> None:
    """Even when classifier didn't pull the variable into affected_variables,
    a topic whose own affected_variables appear as substring in the
    question text should still get the bump (worth 0.2 in
    question_relevance, which is > 0)."""
    diag = _diag(
        failures=[_failure(1, severity=Severity.MINOR, var="biomass", run_id="RUN-0001")]
    )
    no_bump = extract_seed_topics(diag)[0]
    q = UserQuestion(
        text="Why did biomass plateau?",
        # classifier didn't extract anything (worst case)
    )
    bumped = extract_seed_topics(diag, user_question=q)[0]
    assert bumped.priority > no_bump.priority


# ---------- multi-topic ordering ----------


def test_bump_changes_ordering_when_minor_relevant_outranks_other_minors() -> None:
    """Two minor topics, one matches the question — the matching one
    should now have higher priority than the non-matching one."""
    diag = _diag(
        failures=[
            _failure(1, severity=Severity.MINOR, var="paa_mg_l", run_id="RUN-9999"),
            _failure(2, severity=Severity.MINOR, var="biomass_g_l", run_id="RUN-0002"),
        ]
    )
    q = UserQuestion(text="?", affected_variables=["biomass_g_l"])
    topics = extract_seed_topics(diag, user_question=q)
    paa_topic = next(t for t in topics if "paa_mg_l" in t.affected_variables)
    biomass_topic = next(t for t in topics if "biomass_g_l" in t.affected_variables)
    assert biomass_topic.priority > paa_topic.priority
