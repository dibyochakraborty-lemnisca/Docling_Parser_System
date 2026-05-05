"""POST /api/runs/{run_id}/followup endpoint + state surfacing.

PR-A2 commit 6. Plan ref: plans/2026-05-05-hitl-followup.md commit 6.

Coverage:
  - happy path: DONE run accepts a follow-up, background task runs,
    GET reflects new state
  - 404 on unknown run
  - 409 on non-DONE run (RUNNING / HYPOTHESIZING / FAILED)
  - 410 on missing bundle_dir (GC simulated by deleting the dir)
  - 400 on empty question
  - 422 on too-long question (FastAPI/Pydantic enforces max_length=2000)
  - back-compat: GET on a legacy run returns followups=[] and
    bundle_followup_eligible computed correctly
  - multiple follow-ups in sequence accumulate in the GET response
  - Pydantic FollowupRequest schema directly: empty / too-long
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

import pytest
from fastapi.testclient import TestClient


# ---------- direct schema ----------


def test_followup_request_rejects_empty_question() -> None:
    from pydantic import ValidationError

    from apps.api.fermdocs_api.main import FollowupRequest  # type: ignore[import]

    with pytest.raises(ValidationError):
        FollowupRequest.model_validate({"question": ""})


def test_followup_request_rejects_oversize_question() -> None:
    from pydantic import ValidationError

    from apps.api.fermdocs_api.main import FollowupRequest  # type: ignore[import]

    with pytest.raises(ValidationError):
        FollowupRequest.model_validate({"question": "x" * 2001})


def test_followup_request_accepts_valid_question() -> None:
    from apps.api.fermdocs_api.main import FollowupRequest  # type: ignore[import]

    body = FollowupRequest.model_validate({"question": "Why did RUN-0002 plateau?"})
    assert body.question == "Why did RUN-0002 plateau?"


# ---------- helper: build a minimal app with a swapped STORE ----------


def _app_with_store(tmp_path: Path):
    """Build a fresh app + STORE pair for an isolated test."""
    from apps.api.fermdocs_api import main as api_main  # type: ignore[import]
    from apps.api.fermdocs_api.state import RunStore  # type: ignore[import]

    store = RunStore(uploads_root=tmp_path / "u", runs_root=tmp_path / "r")
    # Patch the module-global STORE in-place so create_app's closure picks it up.
    api_main.STORE = store
    app = api_main.create_app()
    return app, store


def _make_done_run(store, tmp_path: Path):
    from apps.api.fermdocs_api.state import RunStatus  # type: ignore[import]

    upload = store.add_upload(filename="t.csv", content_type="text/csv", content=b"x")
    run = store.create_run(upload.upload_id)
    bundle_dir = tmp_path / f"bundle-{run.run_id[:8]}"
    bundle_dir.mkdir()
    (bundle_dir / "meta.json").write_text("{}")
    run.bundle_dir = bundle_dir
    run.global_md = tmp_path / f"global-{run.run_id[:8]}.md"
    run.global_md.write_text("")
    run.hypothesis_dir = tmp_path / f"hyp-{run.run_id[:8]}"
    run.hypothesis_dir.mkdir()
    run.status = RunStatus.DONE
    return run, bundle_dir


# ---------- endpoint: 404 / 409 / 410 / 400 ----------


def test_followup_returns_404_on_unknown_run(tmp_path: Path) -> None:
    app, _ = _app_with_store(tmp_path)
    with TestClient(app) as client:
        r = client.post("/api/runs/nonexistent/followup", json={"question": "?"})
    assert r.status_code == 404


def test_followup_returns_409_when_run_not_done(tmp_path: Path) -> None:
    from apps.api.fermdocs_api.state import RunStatus  # type: ignore[import]

    app, store = _app_with_store(tmp_path)
    run, _ = _make_done_run(store, tmp_path)
    run.status = RunStatus.HYPOTHESIZING

    with TestClient(app) as client:
        r = client.post(f"/api/runs/{run.run_id}/followup", json={"question": "?"})
    assert r.status_code == 409
    assert "must be done" in r.text


def test_followup_returns_410_when_bundle_missing(tmp_path: Path) -> None:
    app, store = _app_with_store(tmp_path)
    run, bundle_dir = _make_done_run(store, tmp_path)
    import shutil

    shutil.rmtree(bundle_dir)

    with TestClient(app) as client:
        r = client.post(f"/api/runs/{run.run_id}/followup", json={"question": "?"})
    assert r.status_code == 410
    assert "no longer available" in r.text


def test_followup_returns_422_on_empty_question(tmp_path: Path) -> None:
    app, store = _app_with_store(tmp_path)
    run, _ = _make_done_run(store, tmp_path)

    with TestClient(app) as client:
        r = client.post(f"/api/runs/{run.run_id}/followup", json={"question": ""})
    # Pydantic min_length=1 → 422 (FastAPI validation), not the inner 400 path.
    assert r.status_code == 422


def test_followup_returns_422_on_oversize_question(tmp_path: Path) -> None:
    app, store = _app_with_store(tmp_path)
    run, _ = _make_done_run(store, tmp_path)

    with TestClient(app) as client:
        r = client.post(
            f"/api/runs/{run.run_id}/followup",
            json={"question": "x" * 2001},
        )
    assert r.status_code == 422


# ---------- endpoint: happy path ----------


def test_followup_happy_path_queues_background_task(tmp_path: Path) -> None:
    """Endpoint accepts the question, returns anticipated index, and
    schedules execute_followup_run as a background task. We patch the
    runner so the real LLM-driven hypothesis stage doesn't fire."""
    from apps.api.fermdocs_api import runner_pipeline  # type: ignore[import]

    app, store = _app_with_store(tmp_path)
    run, _ = _make_done_run(store, tmp_path)

    captured = {}

    async def _fake_followup(*, store, run, question_text):
        captured["question"] = question_text
        captured["run_id"] = run.run_id

    with patch.object(runner_pipeline, "execute_followup_run", side_effect=_fake_followup):
        # Re-create app so the patched function is what /followup calls.
        # (FastAPI captures the import at create_app time; patching after
        # is too late. Easier: patch the symbol main.execute_followup_run.)
        from apps.api.fermdocs_api import main as api_main  # type: ignore[import]

        with patch.object(api_main, "execute_followup_run", side_effect=_fake_followup):
            with TestClient(app) as client:
                r = client.post(
                    f"/api/runs/{run.run_id}/followup",
                    json={"question": "why?"},
                )

    assert r.status_code == 200
    body = r.json()
    assert body["run_id"] == run.run_id
    assert body["status"] == "queued"
    assert body["anticipated_followup_index"] == 1
    assert captured["question"] == "why?"
    assert captured["run_id"] == run.run_id


# ---------- GET surfacing of follow-up state ----------


def test_get_run_includes_followup_fields_back_compat(tmp_path: Path) -> None:
    """A run with no follow-ups returns followups=[] and the eligibility
    flag — back-compat for legacy runs that pre-date PR-A2."""
    app, store = _app_with_store(tmp_path)
    run, _ = _make_done_run(store, tmp_path)

    with TestClient(app) as client:
        r = client.get(f"/api/runs/{run.run_id}")
    assert r.status_code == 200
    body = r.json()
    assert body["followups"] == []
    assert body["followup_index"] == 0
    assert body["bundle_followup_eligible"] is True


def test_get_run_followups_eligible_false_when_bundle_gc(tmp_path: Path) -> None:
    app, store = _app_with_store(tmp_path)
    run, bundle_dir = _make_done_run(store, tmp_path)
    import shutil

    shutil.rmtree(bundle_dir)

    with TestClient(app) as client:
        r = client.get(f"/api/runs/{run.run_id}")
    body = r.json()
    assert body["bundle_followup_eligible"] is False


def test_get_run_includes_accumulated_followups(tmp_path: Path) -> None:
    """After 2 follow-ups land, GET returns both in order."""
    from apps.api.fermdocs_api.state import FollowupResult  # type: ignore[import]

    app, store = _app_with_store(tmp_path)
    run, _ = _make_done_run(store, tmp_path)

    # Synthesize 2 follow-up results (skip the runner — we're testing
    # the GET shape, not the runner; runner has its own commit-2 tests).
    run.followup_index = 2
    run.followups.append(
        FollowupResult(followup_index=1, user_question_text="q1", output={"final_hypotheses": [{"hyp_id": "H-FOLLOW-1"}]})
    )
    run.followups.append(
        FollowupResult(followup_index=2, user_question_text="q2", output={"final_hypotheses": []})
    )

    with TestClient(app) as client:
        r = client.get(f"/api/runs/{run.run_id}")
    body = r.json()
    assert len(body["followups"]) == 2
    assert body["followups"][0]["followup_index"] == 1
    assert body["followups"][0]["user_question_text"] == "q1"
    assert body["followups"][0]["output"]["final_hypotheses"][0]["hyp_id"] == "H-FOLLOW-1"
    assert body["followups"][1]["followup_index"] == 2
    assert body["followup_index"] == 2
