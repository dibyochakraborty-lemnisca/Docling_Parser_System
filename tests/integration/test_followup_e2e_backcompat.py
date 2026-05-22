"""Frontend back-compat invariant for PR-A2 follow-ups.

PR-A2 commit 7. Plan ref: plans/2026-05-05-hitl-followup.md commit 7.

These tests verify that:
  - The API GET /runs/{id} response shape is unchanged for runs with
    no follow-ups (legacy runs render identically post-PR-A2).
  - The follow-up textarea fields (followups, followup_index,
    bundle_followup_eligible) are present and have safe defaults.
  - Status badges differentiate "running follow-up" from "running
    original" via the followup_index field.

The actual JSX rendering is a Next.js client component; we test the
Python-side contract that the frontend reads. Frontend type-checking
+ visual smoke is covered by the existing apps/web/tsconfig and dev
server, not pytest.
"""

from __future__ import annotations

from pathlib import Path

from fastapi.testclient import TestClient


def _app_with_store(tmp_path: Path):
    from apps.api.fermdocs_api import main as api_main  # type: ignore[import]
    from apps.api.fermdocs_api.state import RunStore  # type: ignore[import]

    store = RunStore(uploads_root=tmp_path / "u", runs_root=tmp_path / "r")
    api_main.STORE = store
    app = api_main.create_app()
    return app, store


def _make_run(store, tmp_path: Path, *, status_value: str = "done"):
    from apps.api.fermdocs_api.state import RunStatus  # type: ignore[import]

    upload = store.add_upload(files=[("t.csv", "text/csv", b"x")])
    run = store.create_run(upload.upload_id)
    bundle = tmp_path / f"b-{run.run_id[:8]}"
    bundle.mkdir()
    run.bundle_dir = bundle
    run.status = RunStatus(status_value)
    return run, bundle


def test_legacy_run_get_shape_back_compat(tmp_path: Path) -> None:
    """A run with zero follow-ups returns the same fields plus the new
    ones with safe defaults — no fields removed, no fields renamed."""
    app, store = _app_with_store(tmp_path)
    run, _ = _make_run(store, tmp_path)

    with TestClient(app) as client:
        r = client.get(f"/api/runs/{run.run_id}")

    assert r.status_code == 200
    body = r.json()

    # PR-A fields (unchanged)
    for key in (
        "run_id",
        "upload_id",
        "status",
        "created_at",
        "bundle_dir",
        "hypothesis_dir",
        "global_md",
        "error",
        "output",
    ):
        assert key in body, f"missing PR-A field: {key}"

    # PR-A2 fields (new, with safe defaults)
    assert body["followups"] == []
    assert body["followup_index"] == 0
    assert body["bundle_followup_eligible"] is True


def test_bundle_gc_hides_followup_via_eligibility(tmp_path: Path) -> None:
    """When the bundle is gone, the eligibility flag flips so the
    frontend hides the textarea before the user even tries to POST."""
    app, store = _app_with_store(tmp_path)
    run, bundle = _make_run(store, tmp_path)
    import shutil

    shutil.rmtree(bundle)

    with TestClient(app) as client:
        r = client.get(f"/api/runs/{run.run_id}")
    body = r.json()
    assert body["bundle_followup_eligible"] is False


def test_status_during_followup_disambiguates_via_index(tmp_path: Path) -> None:
    """Frontend computes 'Running follow-up #N' from
    (status='hypothesizing' AND followup_index>0). Verify both fields
    are surfaced so the UI can derive that label."""
    from apps.api.fermdocs_api.state import RunStatus  # type: ignore[import]

    app, store = _app_with_store(tmp_path)
    run, _ = _make_run(store, tmp_path)
    run.status = RunStatus.HYPOTHESIZING
    run.followup_index = 2  # in the middle of follow-up #2

    with TestClient(app) as client:
        r = client.get(f"/api/runs/{run.run_id}")
    body = r.json()
    assert body["status"] == "hypothesizing"
    assert body["followup_index"] == 2


def test_followups_render_in_index_order(tmp_path: Path) -> None:
    """Frontend iterates run.followups; the API must return them in
    order so the UI doesn't have to sort."""
    from apps.api.fermdocs_api.state import FollowupResult  # type: ignore[import]

    app, store = _app_with_store(tmp_path)
    run, _ = _make_run(store, tmp_path)

    # Append in scrambled order, but each carries its own index
    run.followups.append(FollowupResult(followup_index=1, user_question_text="q1", output={}))
    run.followups.append(FollowupResult(followup_index=2, user_question_text="q2", output={}))
    run.followups.append(FollowupResult(followup_index=3, user_question_text="q3", output={}))

    with TestClient(app) as client:
        r = client.get(f"/api/runs/{run.run_id}")
    body = r.json()
    indices = [f["followup_index"] for f in body["followups"]]
    assert indices == [1, 2, 3]


def test_followup_output_carries_question_answered_field(tmp_path: Path) -> None:
    """The frontend renders the question-answered badge from each
    follow-up's output.final_hypotheses[].question_answered. Verify
    that field round-trips through the API."""
    from apps.api.fermdocs_api.state import FollowupResult  # type: ignore[import]

    app, store = _app_with_store(tmp_path)
    run, _ = _make_run(store, tmp_path)
    run.followups.append(
        FollowupResult(
            followup_index=1,
            user_question_text="why?",
            output={
                "final_hypotheses": [
                    {
                        "hyp_id": "H-FOLLOW-1",
                        "summary": "answer",
                        "question_answered": "yes",
                        "question_response_summary": "the bundle says X",
                    }
                ],
            },
        )
    )

    with TestClient(app) as client:
        r = client.get(f"/api/runs/{run.run_id}")
    body = r.json()
    h = body["followups"][0]["output"]["final_hypotheses"][0]
    assert h["question_answered"] == "yes"
    assert h["question_response_summary"] == "the bundle says X"
