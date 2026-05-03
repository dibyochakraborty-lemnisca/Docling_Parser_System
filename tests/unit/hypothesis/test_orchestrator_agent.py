"""Orchestrator agent — offline parser tests + gated live smoke."""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from fermdocs_hypothesis.agents.orchestrator import OrchestratorAgent, _parse_action
from fermdocs_hypothesis.bundle_loader import load_bundle
from fermdocs_hypothesis.projector import project_orchestrator
from fermdocs_hypothesis.schema import (
    BudgetSnapshot,
    OrchestratorView,
    RankedTopic,
)

CAROTENOID = Path("out/bundle_multi_20260502T220318Z_1a29e4")
LIVE = os.environ.get("FERMDOCS_RUN_LIVE_TESTS") == "1"


def _make_view(top_ids: list[str]) -> OrchestratorView:
    return OrchestratorView(
        current_turn=1,
        budget_remaining=BudgetSnapshot(),
        top_topics=[RankedTopic(topic_id=tid, summary=f"sum {tid}", score=0.5) for tid in top_ids],
        open_questions=[],
    )


# ---------- offline parser ----------


def test_parse_action_select_topic_happy_path():
    view = _make_view(["T-0001", "T-0002"])
    parsed = {
        "action": "select_topic",
        "topic_id": "T-0002",
        "rationale": "stronger citations",
    }
    out = _parse_action(parsed, view)
    assert out.action == "select_topic"
    assert out.topic_id == "T-0002"
    assert out.rationale == "stronger citations"


def test_parse_action_hallucinated_topic_id_falls_back():
    view = _make_view(["T-0001"])
    parsed = {"action": "select_topic", "topic_id": "T-9999", "rationale": "x"}
    out = _parse_action(parsed, view)
    # Auto-corrected to first top topic
    assert out.topic_id == "T-0001"


def test_parse_action_hallucinated_topic_id_no_topics_exits():
    view = _make_view([])
    parsed = {"action": "select_topic", "topic_id": "T-9999", "rationale": "x"}
    out = _parse_action(parsed, view)
    assert out.action == "exit_stage"


def test_parse_action_add_open_question():
    view = _make_view(["T-0001"])
    parsed = {
        "action": "add_open_question",
        "question": "what was DO at 30h?",
        "tags": ["DO"],
    }
    out = _parse_action(parsed, view)
    assert out.action == "add_open_question"
    assert out.question == "what was DO at 30h?"
    assert out.tags == ["DO"]


def test_parse_action_exit_stage():
    view = _make_view([])
    parsed = {"action": "exit_stage", "exit_reason": "no_topics_left"}
    out = _parse_action(parsed, view)
    assert out.action == "exit_stage"
    assert out.exit_reason == "no_topics_left"


def test_parse_action_unknown_falls_back_to_exit():
    view = _make_view(["T-0001"])
    parsed = {"action": "weird"}
    out = _parse_action(parsed, view)
    assert out.action == "exit_stage"


def test_parse_action_rationale_truncated_to_200():
    view = _make_view(["T-0001"])
    parsed = {"action": "select_topic", "topic_id": "T-0001", "rationale": "x" * 500}
    out = _parse_action(parsed, view)
    assert len(out.rationale) == 200


# ---------- live smoke (gated) ----------


@pytest.mark.skipif(not LIVE, reason="set FERMDOCS_RUN_LIVE_TESTS=1 to run")
@pytest.mark.skipif(not CAROTENOID.exists(), reason="carotenoid bundle not present")
def test_orchestrator_live_picks_topic_on_carotenoid():
    """End-to-end: real Gemini call, real OrchestratorView from carotenoid.

    Asserts the orchestrator returns a usable action — either select_topic
    with a valid topic_id or exit_stage with a reason. Token counts must
    be > 0 (instrumentation working).
    """
    from dotenv import load_dotenv

    load_dotenv()
    from fermdocs_hypothesis.llm_clients import GeminiHypothesisClient

    loaded = load_bundle(CAROTENOID)
    view = project_orchestrator(
        events=[],
        seed_topics=loaded.hyp_input.seed_topics,
        budget=BudgetSnapshot(),
        current_turn=1,
    )
    agent = OrchestratorAgent(GeminiHypothesisClient())
    action, in_tok, out_tok = agent.decide(view)

    assert in_tok > 0
    assert out_tok > 0
    assert action.action in {"select_topic", "add_open_question", "exit_stage"}
    if action.action == "select_topic":
        valid = {t.topic_id for t in view.top_topics}
        assert action.topic_id in valid
        assert action.rationale
