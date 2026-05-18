from __future__ import annotations

from pathlib import Path

from fermdocs_eval.harness import EvalRun, RunStatus, append_jsonl, now_iso, read_jsonl


def test_append_and_read_roundtrip(tmp_path: Path) -> None:
    out = tmp_path / "results.jsonl"
    run = EvalRun(
        suite="e2",
        trial_id="t-001",
        status=RunStatus.OK,
        started_at=now_iso(),
        finished_at=now_iso(),
        payload={"fired_axes": ["trajectory-axis"], "labeled_axis": "trajectory-axis"},
    )
    append_jsonl(out, run)
    append_jsonl(out, run)

    rows = read_jsonl(out)
    assert len(rows) == 2
    assert rows[0]["suite"] == "e2"
    assert rows[0]["status"] == "ok"
    assert rows[0]["payload"]["labeled_axis"] == "trajectory-axis"


def test_read_jsonl_missing_returns_empty(tmp_path: Path) -> None:
    assert read_jsonl(tmp_path / "nope.jsonl") == []


def test_error_row_carries_message(tmp_path: Path) -> None:
    out = tmp_path / "results.jsonl"
    run = EvalRun(
        suite="e1",
        trial_id="bundle-x-warm",
        status=RunStatus.ERROR,
        started_at=now_iso(),
        finished_at=now_iso(),
        error="LLM timeout after 60s",
    )
    append_jsonl(out, run)
    rows = read_jsonl(out)
    assert rows[0]["status"] == "error"
    assert "timeout" in rows[0]["error"]
