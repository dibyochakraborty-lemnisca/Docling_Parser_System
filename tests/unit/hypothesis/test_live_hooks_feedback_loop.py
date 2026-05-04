"""Integration test: LiveHooks production wire surfaces the feedback loop.

This is the regression guard for the carotenoid bug — it asserts that a
real LiveHooks instance, given a non-empty event log with a prior
rejected attempt on the current topic, produces a synthesizer prompt
that contains the prior critic_reasons. Without this, all the unit
tests would pass while the production path silently kept the loop
broken (the exact failure mode we just fixed).

Approach: minimal fake LoadedBundle (empty pools), fake Gemini client
that records prompts and returns canned JSON. Exercise synthesize and
summarize_lessons directly — full runner.run_stage with mocked LLM would
require also stubbing all 3 specialists + critic + judge, which is out
of scope for "does feedback reach the prompt."
"""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock

from fermdocs_characterize.schema import Severity
from fermdocs_diagnose.schema import ConfidenceBasis
from fermdocs_hypothesis.bundle_loader import LoadedBundle
from fermdocs_hypothesis.events import (
    CritiqueFiledEvent,
    Event,
    HypothesisRejectedEvent,
    HypothesisSynthesizedEvent,
    JudgeRulingEvent,
    TopicSelectedEvent,
)
from fermdocs_hypothesis.live_hooks import LiveHooks
from fermdocs_hypothesis.runner import RunnerState
from fermdocs_hypothesis.schema import (
    BudgetSnapshot,
    CritiqueFull,
    FacetFull,
    HypothesisFull,
    SeedTopic,
    TopicSourceType,
    TopicSpec,
)

NOW = datetime(2026, 5, 3, 12, 0, 0, tzinfo=timezone.utc)
F1 = "00000000-0000-0000-0000-000000000042:F-0001"


# ---------- fake gemini client ----------


class _RecordingGeminiClient:
    """Records each .call() and returns canned responses keyed off
    schema fingerprint. Real GeminiHypothesisClient signature shadowed
    just enough for synthesizer + lessons summarizer to drive."""

    def __init__(self):
        self.calls: list[dict[str, Any]] = []
        self._model = "fake-gemini-test"
        self._api_key = "x"

    @property
    def model_name(self) -> str:
        return self._model

    def call(self, *, system, user_text, response_schema, temperature=0.0):
        self.calls.append(
            {
                "system": system,
                "user_text": user_text,
                "schema": response_schema,
            }
        )
        # Schema fingerprint tells us which agent is calling
        required = set(response_schema.get("required") or [])
        if {"summary", "facet_ids"}.issubset(required):
            # synthesizer
            return (
                {
                    "summary": "T-0001: narrowed claim addressing prior reasons",
                    "facet_ids": ["FCT-0001"],
                    "cited_finding_ids": [F1],
                    "cited_narrative_ids": [],
                    "cited_trajectories": [],
                    "affected_variables": ["biomass_g_l"],
                    "confidence": 0.55,
                    "confidence_basis": "schema_only",
                },
                250,
                90,
            )
        if "lessons" in required:
            # lessons summarizer
            return (
                {"lessons": ["distilled lesson A", "distilled lesson B"]},
                120,
                40,
            )
        # default — shouldn't be hit by these tests
        return ({}, 0, 0)


# ---------- bundle stub ----------


def _make_minimal_bundle(tmp_path: Path) -> LoadedBundle:
    """Empty pools; pass-through characterization. Enough for projector
    + LiveHooks._build_citation_lookups not to crash.

    The real bundle reader does a lot; we sidestep it because this test
    isolates LiveHooks behavior, not bundle loading.
    """
    char = MagicMock()
    char.findings = []
    char.narrative_observations = []
    diag = MagicMock()
    return LoadedBundle(
        hyp_input=MagicMock(),
        diagnosis=diag,
        characterization=char,
        findings_pool=[],
        narratives_pool=[],
        trajectories_pool=[],
        priors_pool=[],
        analyses_pool=[],
        bundle_dir=tmp_path,
    )


# ---------- helpers ----------


def _topic_spec(topic_id="T-0001"):
    return TopicSpec(
        topic_id=topic_id,
        summary="biomass plateau 40-60h",
        source_type=TopicSourceType.FAILURE,
        cited_finding_ids=[F1],
        affected_variables=["biomass_g_l"],
    )


def _facet():
    return FacetFull(
        facet_id="FCT-0001",
        specialist="kinetics",
        summary="growth phase analysis",
        cited_finding_ids=[F1],
        confidence=0.6,
        confidence_basis=ConfidenceBasis.SCHEMA_ONLY,
    )


def _seed_topic(topic_id="T-0001"):
    return SeedTopic(
        topic_id=topic_id,
        summary="biomass plateau 40-60h",
        source_type=TopicSourceType.FAILURE,
        source_id="D-F-0001",
        cited_finding_ids=[F1],
        affected_variables=["biomass_g_l"],
        severity=Severity.MAJOR,
        priority=0.8,
    )


def _runner_state(topic_id="T-0001"):
    return RunnerState(
        phase="synthesize",
        budget=BudgetSnapshot(),
        seed_topics=(_seed_topic(topic_id),),
        current_turn=2,
        current_topic=_topic_spec(topic_id),
        current_facets=(_facet(),),
    )


def _events_with_one_rejected_attempt() -> list[Event]:
    """A complete one-attempt cycle that left a rejected hypothesis with
    a specific critic_reason — the carotenoid-shape regression test."""
    return [
        TopicSelectedEvent(
            ts=NOW,
            turn=1,
            topic_id="T-0001",
            summary="biomass plateau 40-60h",
            rationale="ranker top-1",
        ),
        HypothesisSynthesizedEvent(
            ts=NOW,
            turn=1,
            hyp_id="H-0001",
            topic_id="T-0001",
            summary="prior claim that overreached on documented absence",
            facet_ids=["FCT-0001"],
            cited_finding_ids=[F1],
            confidence=0.6,
            confidence_basis=ConfidenceBasis.SCHEMA_ONLY,
        ),
        CritiqueFiledEvent(
            ts=NOW,
            turn=1,
            hyp_id="H-0001",
            flag="red",
            reasons=["documented absence treated as ruled out"],
        ),
        JudgeRulingEvent(
            ts=NOW,
            turn=1,
            hyp_id="H-0001",
            criticism_valid=True,
            rationale="critic stands",
        ),
        HypothesisRejectedEvent(
            ts=NOW,
            turn=1,
            hyp_id="H-0001",
            topic_id="T-0001",
            reason="upheld",
        ),
    ]


# ---------- the actual tests ----------


def test_synthesize_first_attempt_no_previous_attempts_in_prompt(tmp_path):
    """Sanity: on a fresh topic with no prior cycles, the synthesizer
    prompt does NOT contain previous_attempts content."""
    bundle = _make_minimal_bundle(tmp_path)
    client = _RecordingGeminiClient()
    hooks = LiveHooks(bundle=bundle, client=client)  # type: ignore[arg-type]

    state = _runner_state()
    hooks.synthesize(state, hyp_id="H-0001", events=[])

    assert len(client.calls) == 1
    user_text = client.calls[0]["user_text"]
    # Empty list serializes — that's fine. The carotenoid-shape phrase
    # from a prior critic_reason MUST not appear.
    assert "documented absence treated as ruled out" not in user_text


def test_synthesize_retry_carries_prior_critic_reason_into_prompt(tmp_path):
    """Regression guard: on retry (with events showing a prior rejected
    attempt on this topic), the synthesizer's user_text contains the
    prior critic_reason verbatim. This is exactly what was missing in
    H-001/2/3 carotenoid loop — the fix."""
    bundle = _make_minimal_bundle(tmp_path)
    client = _RecordingGeminiClient()
    hooks = LiveHooks(bundle=bundle, client=client)  # type: ignore[arg-type]

    events = _events_with_one_rejected_attempt()
    state = _runner_state()
    hooks.synthesize(state, hyp_id="H-0002", events=events)

    assert len(client.calls) == 1
    user_text = client.calls[0]["user_text"]
    # The prior critic_reason must appear in the prompt — projector
    # populated previous_attempts → SynthesizerView serializes into
    # user_text via build_prompt → reaches the LLM.
    assert "documented absence treated as ruled out" in user_text
    # And the prior hyp_id should be there too so the model can reason
    # about which attempt is being followed up.
    assert "H-0001" in user_text


def test_critique_retry_carries_prior_attempts_excluding_current(tmp_path):
    """Critic on retry must see prior attempts on this topic but NOT the
    hypothesis currently under review (it would self-reference)."""
    bundle = _make_minimal_bundle(tmp_path)
    client = _RecordingGeminiClient()
    hooks = LiveHooks(bundle=bundle, client=client)  # type: ignore[arg-type]

    events = _events_with_one_rejected_attempt()
    # Critic is reviewing a NEW hypothesis (H-0002) that came after H-0001
    new_hyp = HypothesisFull(
        hyp_id="H-0002",
        summary="narrowed claim",
        facet_ids=["FCT-0001"],
        cited_finding_ids=[F1],
        confidence=0.55,
        confidence_basis=ConfidenceBasis.SCHEMA_ONLY,
    )
    state = RunnerState(
        phase="critique",
        budget=BudgetSnapshot(),
        seed_topics=(_seed_topic(),),
        current_turn=2,
        current_topic=_topic_spec(),
        current_facets=(_facet(),),
        current_hypothesis=new_hyp,
    )

    # We can't easily drive the critic ReAct loop without mocking tool
    # responses too — that's a unit-test concern of CriticAgent. What
    # we CAN verify cheaply: project_critic, called via LiveHooks's
    # critique() entry, populates the view correctly. Inspect by
    # patching the critic agent's critique() to capture the view.
    captured: dict[str, Any] = {}
    real_critique = hooks._critic.critique  # noqa: SLF001

    def _capture(view):
        captured["view"] = view
        # Return a minimal CritiqueResult-shaped object
        from fermdocs_hypothesis.agents.critic import CriticResult
        return CriticResult(
            critique=CritiqueFull(hyp_id=view.hypothesis.hyp_id, flag="green", reasons=[]),
            input_tokens=10,
            output_tokens=5,
            tool_calls=0,
        )

    hooks._critic.critique = _capture  # type: ignore[assignment]
    try:
        hooks.critique(state, events=events)
    finally:
        hooks._critic.critique = real_critique  # type: ignore[assignment]

    view = captured["view"]
    # Prior H-0001 attempt visible to critic
    assert [a.hyp_id for a in view.previous_attempts] == ["H-0001"]
    # Current hyp_id NOT in previous_attempts (would be self-reference)
    assert "H-0002" not in [a.hyp_id for a in view.previous_attempts]


def test_summarize_lessons_calls_real_agent_and_returns_digest(tmp_path):
    """LiveHooks.summarize_lessons hands off to LessonsSummarizerAgent
    with the real Gemini client; on success the digest text is non-empty
    and tokens flow through."""
    bundle = _make_minimal_bundle(tmp_path)
    client = _RecordingGeminiClient()
    hooks = LiveHooks(bundle=bundle, client=client)  # type: ignore[arg-type]

    digest, in_tok, out_tok = hooks.summarize_lessons(
        _runner_state(),
        recent_reasons=["documented absence ≠ ruled out", "scope drift on batch claims"],
        source_reason_count=2,
    )
    assert "distilled lesson A" in digest
    assert "distilled lesson B" in digest
    assert in_tok == 120
    assert out_tok == 40

    # The recorded call's user_text should mention the input reasons
    assert len(client.calls) == 1
    user_text = client.calls[0]["user_text"]
    assert "documented absence" in user_text


def test_summarize_lessons_swallows_client_errors(tmp_path):
    """A Gemini outage during summarization must not blow up the runner.
    LiveHooks wraps the call in try/except so the runner's higher-level
    fallback path stays consistent."""

    class _BoomClient(_RecordingGeminiClient):
        def call(self, **kwargs):
            raise RuntimeError("fake gemini outage")

    bundle = _make_minimal_bundle(tmp_path)
    hooks = LiveHooks(bundle=bundle, client=_BoomClient())  # type: ignore[arg-type]

    digest, in_tok, out_tok = hooks.summarize_lessons(
        _runner_state(),
        recent_reasons=["any reason"],
        source_reason_count=1,
    )
    assert digest == ""
    assert in_tok == 0
    assert out_tok == 0
