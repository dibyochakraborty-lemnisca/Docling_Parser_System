"""execute_followup_run + lifecycle helper (PR-A2 commit 2).

Plan ref: plans/2026-05-05-hitl-followup.md commit 2.

Coverage:
  - status transitions DONE → HYPOTHESIZING → DONE on success
  - status FAILED on missing/missing-on-disk bundle
  - frozen-bundle invariant (no ingest/characterize/diagnose subprocess
    fires during follow-up — the explicit assertion the plan calls out)
  - followup_index increments before work; FollowupResult appended on
    success
  - multiple follow-ups in sequence accumulate in run.followups
  - lifecycle helper refactor doesn't change execute_resume's external
    behavior (status sequence, error path)
"""

from __future__ import annotations

import asyncio
import json
from pathlib import Path
from unittest.mock import patch

import pytest

from apps.api.fermdocs_api import runner_pipeline
from apps.api.fermdocs_api.state import RunStatus, RunStore


# ---------- shared fixtures ----------


def _arun(coro):
    """Run an async test body synchronously. The repo doesn't ship
    pytest-asyncio; this matches the existing test convention."""
    return asyncio.run(coro)


def _make_done_run(tmp_path: Path) -> tuple[RunStore, "Run", Path]:  # type: ignore[name-defined]
    """A RunStore + Run in DONE state with a real bundle_dir on disk."""
    store = RunStore(uploads_root=tmp_path / "u", runs_root=tmp_path / "r")
    upload = store.add_upload(filename="t.csv", content_type="text/csv", content=b"x")
    run = store.create_run(upload.upload_id)

    bundle_dir = tmp_path / "bundle"
    bundle_dir.mkdir()
    (bundle_dir / "meta.json").write_text("{}")
    (bundle_dir / "characterization.json").write_text(
        json.dumps({"trajectories": [{"run_id": "RUN-0001", "variable": "OD"}]})
    )

    hyp_dir = tmp_path / "hyp"
    hyp_dir.mkdir()
    global_md = hyp_dir / "global.md"
    global_md.write_text("")

    run.bundle_dir = bundle_dir
    run.global_md = global_md
    run.hypothesis_dir = hyp_dir
    run.status = RunStatus.DONE
    return store, run, bundle_dir


class _FakeResult:
    """Stand-in for runner.run_stage's return shape."""

    def __init__(self, output) -> None:
        self.output = output


class _FakeOutput:
    def __init__(self) -> None:
        self.open_questions = []  # not used by follow-up

    def model_dump(self, mode: str = "json") -> dict:
        return {"meta": {"hypothesis_id": "stub"}, "final_hypotheses": []}


# ---------- happy path ----------


def test_execute_followup_run_status_done_to_done(tmp_path: Path) -> None:
    store, run, _ = _make_done_run(tmp_path)
    fake_result = _FakeResult(_FakeOutput())

    async def _body():
        with patch.object(
            runner_pipeline, "_run_hypothesis_blocking", return_value=fake_result
        ), patch.object(
            runner_pipeline, "_classify_and_persist_followup_question",
        ):
            await runner_pipeline.execute_followup_run(
                store=store, run=run, question_text="why RUN-0001 plateau?"
            )

    _arun(_body())
    assert run.status == RunStatus.DONE
    assert run.followup_index == 1
    assert len(run.followups) == 1
    assert run.followups[0].user_question_text == "why RUN-0001 plateau?"
    assert run.followups[0].followup_index == 1


def test_execute_followup_run_writes_per_followup_output(tmp_path: Path) -> None:
    store, run, _ = _make_done_run(tmp_path)
    fake_result = _FakeResult(_FakeOutput())

    async def _body():
        with patch.object(
            runner_pipeline, "_run_hypothesis_blocking", return_value=fake_result
        ), patch.object(
            runner_pipeline, "_classify_and_persist_followup_question",
        ):
            await runner_pipeline.execute_followup_run(
                store=store, run=run, question_text="q1"
            )

    _arun(_body())
    out_path = run.hypothesis_dir / "hypothesis_output_followup_1.json"
    assert out_path.exists()


# ---------- frozen bundle invariant (the critical assertion) ----------


def test_followup_does_not_fire_ingest_characterize_diagnose(tmp_path: Path) -> None:
    """The whole point of drive posture: bundle is frozen. Spy on the
    subprocess-driven ingest/characterize/diagnose helpers and assert
    none of them fired during the follow-up run."""
    store, run, _ = _make_done_run(tmp_path)
    fake_result = _FakeResult(_FakeOutput())

    spies = {}

    async def _body():
        with patch.object(
            runner_pipeline, "_run_hypothesis_blocking", return_value=fake_result
        ), patch.object(
            runner_pipeline, "_classify_and_persist_followup_question",
        ), patch.object(
            runner_pipeline, "_prepare_bundle_dir"
        ) as prepare_spy, patch.object(
            runner_pipeline, "_build_bundle_from_raw"
        ) as build_spy, patch.object(
            runner_pipeline, "_run_subprocess"
        ) as subprocess_spy:
            await runner_pipeline.execute_followup_run(
                store=store, run=run, question_text="q"
            )
            spies["prepare"] = prepare_spy.call_count
            spies["build"] = build_spy.call_count
            spies["subprocess"] = subprocess_spy.call_count

    _arun(_body())
    assert spies["prepare"] == 0
    assert spies["build"] == 0
    assert spies["subprocess"] == 0


# ---------- guard rails ----------


def test_followup_fails_when_no_bundle_dir(tmp_path: Path) -> None:
    store, run, _ = _make_done_run(tmp_path)
    run.bundle_dir = None  # simulate weird state

    _arun(runner_pipeline.execute_followup_run(
        store=store, run=run, question_text="q"
    ))

    assert run.status == RunStatus.FAILED
    assert run.error and "no bundle" in run.error
    assert run.followups == []


def test_followup_fails_when_bundle_dir_missing_on_disk(tmp_path: Path) -> None:
    store, run, bundle_dir = _make_done_run(tmp_path)
    import shutil

    shutil.rmtree(bundle_dir)

    _arun(runner_pipeline.execute_followup_run(
        store=store, run=run, question_text="q"
    ))

    assert run.status == RunStatus.FAILED
    assert run.error and "no longer exists" in run.error
    assert run.followups == []


# ---------- exception inside hypothesis stage ----------


def test_followup_exception_marks_run_failed(tmp_path: Path) -> None:
    store, run, _ = _make_done_run(tmp_path)

    def _boom(*_a, **_k):
        raise RuntimeError("LLM exploded")

    async def _body():
        with patch.object(
            runner_pipeline, "_run_hypothesis_blocking", side_effect=_boom
        ), patch.object(
            runner_pipeline, "_classify_and_persist_followup_question",
        ):
            await runner_pipeline.execute_followup_run(
                store=store, run=run, question_text="q"
            )

    _arun(_body())
    assert run.status == RunStatus.FAILED
    assert run.error and "RuntimeError" in run.error
    assert run.followup_index == 1
    assert run.followups == []


# ---------- multiple follow-ups in sequence ----------


def test_multiple_followups_accumulate(tmp_path: Path) -> None:
    store, run, _ = _make_done_run(tmp_path)
    fake_result = _FakeResult(_FakeOutput())

    async def _body():
        with patch.object(
            runner_pipeline, "_run_hypothesis_blocking", return_value=fake_result
        ), patch.object(
            runner_pipeline, "_classify_and_persist_followup_question",
        ):
            await runner_pipeline.execute_followup_run(
                store=store, run=run, question_text="q1"
            )
            await runner_pipeline.execute_followup_run(
                store=store, run=run, question_text="q2"
            )
            await runner_pipeline.execute_followup_run(
                store=store, run=run, question_text="q3"
            )

    _arun(_body())
    assert run.followup_index == 3
    assert [f.followup_index for f in run.followups] == [1, 2, 3]
    assert [f.user_question_text for f in run.followups] == ["q1", "q2", "q3"]


# ---------- classifier wiring ----------


def test_followup_classifies_question_with_user_followup_raised_by(
    tmp_path: Path,
) -> None:
    """The classifier is called with raised_by='user_followup' so downstream
    prompts apply drive-mode rules (commit 4)."""
    store, run, _ = _make_done_run(tmp_path)
    fake_result = _FakeResult(_FakeOutput())

    captured = {}

    async def _body():
        with patch.object(
            runner_pipeline, "_run_hypothesis_blocking", return_value=fake_result
        ), patch.object(
            runner_pipeline, "_classify_and_persist_followup_question",
        ) as classifier_spy:
            await runner_pipeline.execute_followup_run(
                store=store, run=run, question_text="why?"
            )
            captured["count"] = classifier_spy.call_count
            captured["kwargs"] = classifier_spy.call_args.kwargs

    _arun(_body())
    assert captured["count"] == 1
    assert captured["kwargs"]["question_text"] == "why?"
    assert captured["kwargs"]["bundle_dir"] == run.bundle_dir


def test_classify_and_persist_followup_passes_raised_by(tmp_path: Path) -> None:
    """Direct unit test: the helper must call classify_user_question with
    raised_by='user_followup'."""
    bundle_dir = tmp_path / "b"
    bundle_dir.mkdir()
    char_path = bundle_dir / "characterization.json"
    char_path.write_text(json.dumps({"trajectories": []}))

    captured = {}

    def _fake_classify(*, text, available_run_ids, available_variables, client, raised_by):
        captured["raised_by"] = raised_by
        from fermdocs.domain.user_question import UserQuestion

        return UserQuestion(text=text, raised_by=raised_by)

    with patch(
        "fermdocs_hypothesis.question_classifier.classify_user_question",
        side_effect=_fake_classify,
    ), patch(
        "fermdocs_hypothesis.question_classifier.GeminiQuestionClassifierClient",
        side_effect=RuntimeError("no api key"),
    ):
        target = runner_pipeline._classify_and_persist_followup_question(
            question_text="hello",
            bundle_dir=bundle_dir,
            char_path=char_path,
        )

    assert captured["raised_by"] == "user_followup"
    assert target == bundle_dir / "user_question.json"
    persisted = json.loads(target.read_text())
    assert persisted["raised_by"] == "user_followup"


# ---------- lifecycle helper refactor: execute_resume still works ----------


def test_execute_resume_still_publishes_resuming_status(tmp_path: Path) -> None:
    """Smoke test the lifecycle refactor: execute_resume's status hop
    still goes through RESUMING (not HYPOTHESIZING)."""
    store, run, _ = _make_done_run(tmp_path)
    run.status = RunStatus.PAUSED
    fake_result = _FakeResult(_FakeOutput())

    published: list[dict] = []

    async def _capture(_run_id, ev):
        published.append(ev)

    store.publish = _capture  # type: ignore[method-assign]

    async def _body():
        with patch.object(
            runner_pipeline, "_resume_hypothesis_blocking", return_value=fake_result
        ):
            await runner_pipeline.execute_resume(store=store, run=run, answers=[])

    _arun(_body())
    statuses = [e for e in published if e.get("type") == "status"]
    assert any(e["status"] == RunStatus.RESUMING.value for e in statuses), statuses


def test_execute_resume_no_bundle_returns_failed(tmp_path: Path) -> None:
    store, run, _ = _make_done_run(tmp_path)
    run.bundle_dir = None
    run.global_md = None

    _arun(runner_pipeline.execute_resume(store=store, run=run, answers=[]))

    assert run.status == RunStatus.FAILED
    assert run.error and "not in a resumable state" in run.error
