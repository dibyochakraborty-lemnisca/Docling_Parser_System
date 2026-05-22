"""Event log: roundtrip, atomic write, append-only, all event types."""

from __future__ import annotations

import json
from datetime import datetime, timezone

from fermdocs_hypothesis.event_log import Observer, read_events
from fermdocs_hypothesis.events import (
    HumanInputReceivedEvent,
    QuestionAddedEvent,
    StagePausedEvent,
    StageStartedEvent,
    TopicSelectedEvent,
)
from fermdocs_hypothesis.schema import BudgetSnapshot

NOW = datetime(2026, 5, 3, 12, 0, 0, tzinfo=timezone.utc)


def test_roundtrip_single_event(tmp_path):
    p = tmp_path / "global.md"
    obs = Observer(p)
    obs.write(StageStartedEvent(ts=NOW, turn=0, input_diagnosis_id="diag-1", budget=BudgetSnapshot()))
    parsed = read_events(p)
    assert len(parsed) == 1
    assert parsed[0].type == "stage_started"


def test_roundtrip_multiple_events_preserves_order(tmp_path):
    p = tmp_path / "global.md"
    obs = Observer(p)
    obs.write(StageStartedEvent(ts=NOW, turn=0, input_diagnosis_id="diag-1", budget=BudgetSnapshot()))
    obs.write(TopicSelectedEvent(ts=NOW, turn=1, topic_id="T-0001", summary="x", rationale="r"))
    obs.write(QuestionAddedEvent(ts=NOW, turn=1, qid="Q-0001", question="why?", raised_by="kinetics", tags=["DO"]))
    parsed = read_events(p)
    assert [e.type for e in parsed] == ["stage_started", "topic_selected", "question_added"]


def test_reserved_hitl_events_roundtrip(tmp_path):
    """stage_paused / human_input_received are reserved seams; must serialize now."""
    p = tmp_path / "global.md"
    obs = Observer(p)
    obs.write(StagePausedEvent(ts=NOW, turn=2, context={"awaiting": "questions"}))
    obs.write(HumanInputReceivedEvent(ts=NOW, turn=2, input_type="answer", payload={"qid": "Q-0001", "answer": "yes"}))
    parsed = read_events(p)
    assert parsed[0].type == "stage_paused"
    assert parsed[1].type == "human_input_received"
    assert parsed[1].input_type == "answer"


def test_observer_loads_existing_events_on_init(tmp_path):
    """Re-opening Observer on an existing file must not lose history."""
    p = tmp_path / "global.md"
    obs1 = Observer(p)
    obs1.write(StageStartedEvent(ts=NOW, turn=0, input_diagnosis_id="diag-1", budget=BudgetSnapshot()))
    obs2 = Observer(p)
    assert len(obs2.events) == 1
    obs2.write(TopicSelectedEvent(ts=NOW, turn=1, topic_id="T-0001", summary="x", rationale="r"))
    parsed = read_events(p)
    assert len(parsed) == 2


def test_atomic_write_no_temp_files_left_behind(tmp_path):
    p = tmp_path / "global.md"
    obs = Observer(p)
    for i in range(5):
        obs.write(TopicSelectedEvent(ts=NOW, turn=i + 1, topic_id=f"T-000{i+1}", summary="x", rationale="r"))
    leftovers = [f for f in tmp_path.iterdir() if f.name.startswith(".global.md.")]
    assert leftovers == []


def test_jsonl_block_format_is_valid_json(tmp_path):
    p = tmp_path / "global.md"
    obs = Observer(p)
    obs.write(TopicSelectedEvent(ts=NOW, turn=1, topic_id="T-0001", summary="x", rationale="r"))
    text = p.read_text()
    assert "```jsonl" in text
    # Extract the jsonl block and confirm each line parses as JSON.
    body = text.split("```jsonl\n")[1].split("```")[0]
    for line in body.strip().splitlines():
        json.loads(line)


def test_empty_observer_renders_header_only(tmp_path):
    p = tmp_path / "global.md"
    Observer(p)._flush()
    text = p.read_text()
    assert text.startswith("# Hypothesis stage event log")
    assert "```jsonl" not in text
