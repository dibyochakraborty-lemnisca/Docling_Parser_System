"""Offline tests for the FastAPI surface — endpoints, WS, state machine.

No live LLM. Uses a stubbed pipeline by patching execute_run / execute_resume
so we can verify routing + WS plumbing without spending tokens.
"""

from __future__ import annotations

import asyncio
import io
import json
import zipfile
from pathlib import Path

import pytest
from fastapi.testclient import TestClient


def _make_zip(files: dict[str, str | bytes]) -> bytes:
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as zf:
        for name, content in files.items():
            if isinstance(content, str):
                content = content.encode()
            zf.writestr(name, content)
    return buf.getvalue()


@pytest.fixture
def app(tmp_path, monkeypatch):
    monkeypatch.setenv("FERMDOCS_API_ROOT", str(tmp_path / "api"))
    # Re-import so the module-level STORE picks up the env override
    import importlib

    import fermdocs_api.main as main_mod
    importlib.reload(main_mod)
    return main_mod.app


def test_health(app):
    client = TestClient(app)
    r = client.get("/api/health")
    assert r.status_code == 200
    assert r.json() == {"status": "ok"}


def test_upload_returns_id(app):
    client = TestClient(app)
    r = client.post(
        "/api/uploads",
        files={"file": ("hello.txt", b"hi", "text/plain")},
    )
    assert r.status_code == 200
    body = r.json()
    assert body["filename"] == "hello.txt"
    assert body["size_bytes"] == 2
    assert "upload_id" in body


def test_create_run_unknown_upload_404(app):
    client = TestClient(app)
    r = client.post("/api/runs", json={"upload_id": "nope"})
    assert r.status_code == 404


def test_list_runs_empty(app):
    client = TestClient(app)
    r = client.get("/api/runs")
    assert r.status_code == 200
    assert r.json() == {"runs": []}


def test_get_run_404(app):
    client = TestClient(app)
    r = client.get("/api/runs/does-not-exist")
    assert r.status_code == 404


def test_create_run_with_zip_upload_then_get(app, monkeypatch):
    """End-to-end happy path with the pipeline patched to a no-op so we
    don't spend tokens. We verify upload → run create → run get returns
    the expected status flow."""
    import fermdocs_api.main as main_mod

    async def fake_execute_run(*, store, run, upload):
        # Simulate: pipeline unzips the bundle and finishes
        run.status = main_mod.RunStatus.DONE

    monkeypatch.setattr(main_mod, "execute_run", fake_execute_run)

    client = TestClient(app)

    zip_bytes = _make_zip({"meta.json": '{"bundle_schema_version": "1.0"}'})
    r = client.post(
        "/api/uploads",
        files={"file": ("bundle.zip", zip_bytes, "application/zip")},
    )
    upload_id = r.json()["upload_id"]

    r = client.post("/api/runs", json={"upload_id": upload_id})
    assert r.status_code == 200
    run_id = r.json()["run_id"]

    # The background task may have already run. Whichever — get should work.
    r = client.get(f"/api/runs/{run_id}")
    assert r.status_code == 200
    body = r.json()
    assert body["run_id"] == run_id
    assert body["upload_id"] == upload_id


def test_submit_answers_requires_paused_or_done(app):
    """Pre-create a run object directly; submit answers when status is
    pending → should reject with 409."""
    import fermdocs_api.main as main_mod

    upload = main_mod.STORE.add_upload(
        filename="x.zip", content_type="application/zip", content=b"hi"
    )
    run = main_mod.STORE.create_run(upload.upload_id)
    # status starts as PENDING

    client = TestClient(app)
    r = client.post(
        f"/api/runs/{run.run_id}/answers",
        json={"answers": [{"qid": "Q-0001", "resolution": "yes"}]},
    )
    assert r.status_code == 409


def test_submit_answers_empty_list_rejected(app):
    import fermdocs_api.main as main_mod

    upload = main_mod.STORE.add_upload(
        filename="x.zip", content_type="application/zip", content=b"hi"
    )
    run = main_mod.STORE.create_run(upload.upload_id)
    run.status = main_mod.RunStatus.PAUSED

    client = TestClient(app)
    r = client.post(
        f"/api/runs/{run.run_id}/answers",
        json={"answers": []},
    )
    assert r.status_code == 400


def test_submit_answers_kicks_off_resume(app, monkeypatch):
    import fermdocs_api.main as main_mod

    called = {"args": None}

    async def fake_resume(*, store, run, answers):
        called["args"] = (run.run_id, answers)
        run.status = main_mod.RunStatus.DONE

    monkeypatch.setattr(main_mod, "execute_resume", fake_resume)

    upload = main_mod.STORE.add_upload(
        filename="x.zip", content_type="application/zip", content=b"hi"
    )
    run = main_mod.STORE.create_run(upload.upload_id)
    run.status = main_mod.RunStatus.PAUSED

    client = TestClient(app)
    r = client.post(
        f"/api/runs/{run.run_id}/answers",
        json={"answers": [{"qid": "Q-0001", "resolution": "yes"}]},
    )
    assert r.status_code == 200
    # Background task should have been scheduled and run
    assert called["args"] is not None
    assert called["args"][0] == run.run_id
    assert called["args"][1] == [("Q-0001", "yes")]


def test_websocket_subscribes_and_receives_published_events(app):
    """A WS subscriber should receive events published via the store."""
    import fermdocs_api.main as main_mod

    upload = main_mod.STORE.add_upload(
        filename="x.zip", content_type="application/zip", content=b"hi"
    )
    run = main_mod.STORE.create_run(upload.upload_id)

    client = TestClient(app)
    with client.websocket_connect(f"/api/runs/{run.run_id}/events") as ws:
        # Publish from another task
        async def publish_one():
            await asyncio.sleep(0.05)
            await main_mod.STORE.publish(run.run_id, {"type": "status", "status": "ingesting"})

        # TestClient runs the WS in its own thread; we use the WS's own
        # loop to publish.
        import threading

        def runner():
            loop = asyncio.new_event_loop()
            try:
                loop.run_until_complete(publish_one())
            finally:
                loop.close()

        t = threading.Thread(target=runner)
        t.start()
        msg = ws.receive_json()
        t.join()
        assert msg == {"type": "status", "status": "ingesting"}
