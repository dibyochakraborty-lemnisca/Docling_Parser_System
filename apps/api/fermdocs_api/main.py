"""FastAPI app — endpoints for upload, run, watch, answer.

Plan ref: plans/2026-05-03-hypothesis-debate-v0.md (v0.5a backend).

Endpoints (all under `/api`):

  POST /uploads               — multipart upload; returns {upload_id}
  POST /runs                  — body {upload_id}; kicks off background pipeline; returns {run_id}
  GET  /runs                  — list runs (status, timestamps)
  GET  /runs/{run_id}         — full run state
  WS   /runs/{run_id}/events  — live event stream for a run
  POST /runs/{run_id}/answers — body {answers: [{qid, resolution}]}; triggers resume

Local-only by design — no auth, no CORS lockdown beyond the dev frontend
on localhost:3000.
"""

from __future__ import annotations

import asyncio
import json
import os
from pathlib import Path

import uvicorn
from dotenv import load_dotenv
from fastapi import (
    BackgroundTasks,
    FastAPI,
    File,
    HTTPException,
    UploadFile,
    WebSocket,
    WebSocketDisconnect,
)
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from fermdocs_api.runner_pipeline import (
    execute_followup_run,
    execute_resume,
    execute_run,
)
from fermdocs_api.state import RunStatus, RunStore


# Pydantic request/response models — defined at module scope so FastAPI's
# introspection sees them as proper top-level types (nested-in-closure
# classes can confuse the dependency resolver and cause 422s on valid
# bodies).


class CreateRunRequest(BaseModel):
    upload_id: str
    # PR-A on caisc-hitl: optional question typed by the user above the
    # upload box. Empty string and None both mean "no question, run as
    # today". Capped at 2000 chars per UserQuestion schema; FastAPI rejects
    # longer with 422.
    user_question: str | None = Field(default=None, max_length=2000)


class Answer(BaseModel):
    qid: str
    resolution: str


class AnswersRequest(BaseModel):
    answers: list[Answer]


class FollowupRequest(BaseModel):
    """Body for POST /api/runs/{run_id}/followup (PR-A2 drive posture).

    Only the question is needed — the run already knows its bundle.
    Same 2000-char cap as the original CreateRunRequest.user_question
    so users can't bypass the limit by submitting a follow-up.
    """

    question: str = Field(min_length=1, max_length=2000)

load_dotenv()

# Local-only state roots; override via env if needed.
_API_ROOT = Path(os.environ.get("FERMDOCS_API_ROOT", "out/api"))
STORE = RunStore(
    uploads_root=_API_ROOT / "uploads",
    runs_root=_API_ROOT / "runs",
)


def create_app() -> FastAPI:
    app = FastAPI(title="fermdocs-api", version="0.1.0")
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["http://localhost:3000", "http://127.0.0.1:3000"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    @app.get("/api/health")
    async def health() -> dict:
        return {"status": "ok"}

    # ---- uploads ----

    @app.post("/api/uploads")
    async def upload(file: UploadFile = File(...)) -> dict:
        content = await file.read()
        upload = STORE.add_upload(
            filename=file.filename or "upload.bin",
            content_type=file.content_type or "application/octet-stream",
            content=content,
        )
        return {
            "upload_id": upload.upload_id,
            "filename": upload.filename,
            "size_bytes": upload.size_bytes,
            "content_type": upload.content_type,
        }

    # ---- runs ----

    @app.post("/api/runs")
    async def create_run(
        body: CreateRunRequest, background: BackgroundTasks
    ) -> dict:
        upload = STORE.get_upload(body.upload_id)
        if upload is None:
            raise HTTPException(404, f"upload {body.upload_id} not found")
        run = STORE.create_run(
            body.upload_id, user_question_text=body.user_question
        )
        background.add_task(execute_run, store=STORE, run=run, upload=upload)
        return {
            "run_id": run.run_id,
            "status": run.status.value,
            "user_question": run.user_question_text,
        }

    @app.get("/api/runs")
    async def list_runs() -> dict:
        return {
            "runs": [
                {
                    "run_id": r.run_id,
                    "upload_id": r.upload_id,
                    "status": r.status.value,
                    "created_at": r.created_at.isoformat(),
                    "error": r.error,
                }
                for r in STORE.list_runs()
            ]
        }

    @app.get("/api/runs/{run_id}")
    async def get_run(run_id: str) -> dict:
        run = STORE.get_run(run_id)
        if run is None:
            raise HTTPException(404, f"run {run_id} not found")
        output_path = (
            (run.hypothesis_dir / "hypothesis_output.json")
            if run.hypothesis_dir is not None
            else None
        )
        output = None
        if output_path and output_path.exists():
            output = json.loads(output_path.read_text())
        # PR-A2: surface follow-up state. `followups` is a list of
        # {followup_index, user_question_text, output, created_at}.
        # `bundle_followup_eligible` lets the frontend hide the textarea
        # when the bundle has been GC'd (would 410 on POST).
        followups = [
            {
                "followup_index": f.followup_index,
                "user_question_text": f.user_question_text,
                "output": (
                    f.output.model_dump(mode="json")
                    if hasattr(f.output, "model_dump")
                    else f.output
                ),
                "created_at": f.created_at.isoformat(),
            }
            for f in run.followups
        ]
        return {
            "run_id": run.run_id,
            "upload_id": run.upload_id,
            "status": run.status.value,
            "created_at": run.created_at.isoformat(),
            "bundle_dir": str(run.bundle_dir) if run.bundle_dir else None,
            "hypothesis_dir": str(run.hypothesis_dir) if run.hypothesis_dir else None,
            "global_md": str(run.global_md) if run.global_md else None,
            "error": run.error,
            "output": output,
            "followups": followups,
            "followup_index": run.followup_index,
            "bundle_followup_eligible": run.bundle_followup_eligible,
        }

    # ---- live event stream ----

    @app.websocket("/api/runs/{run_id}/events")
    async def stream_events(websocket: WebSocket, run_id: str) -> None:
        run = STORE.get_run(run_id)
        if run is None:
            await websocket.close(code=4404)
            return
        await websocket.accept()
        # Replay any existing events from global.md so a late-joining
        # subscriber sees the full timeline.
        if run.global_md and run.global_md.exists():
            from fermdocs_hypothesis.event_log import read_events as _read

            for ev in _read(run.global_md):
                await websocket.send_json(
                    {"type": "event", "event": ev.model_dump(mode="json")}
                )
        # Subscribe to future events
        q = await STORE.subscribe(run_id)
        try:
            while True:
                msg = await q.get()
                await websocket.send_json(msg)
        except WebSocketDisconnect:
            pass
        finally:
            STORE.unsubscribe(run_id, q)

    # ---- follow-up (drive posture, PR-A2) ----

    @app.post("/api/runs/{run_id}/followup")
    async def submit_followup(
        run_id: str, body: FollowupRequest, background: BackgroundTasks
    ) -> dict:
        """Drive posture: a DONE run accepts a new question against the
        same bundle. No re-ingest. Status flips to HYPOTHESIZING and
        runs the debate cycle once with shape-aware seed topics.

        Returns 404 if the run doesn't exist, 409 if the run isn't DONE,
        410 Gone if the bundle has been deleted (so frontend can hide
        the textarea on subsequent loads).
        """
        run = STORE.get_run(run_id)
        if run is None:
            raise HTTPException(404, f"run {run_id} not found")
        if run.status != RunStatus.DONE:
            raise HTTPException(
                409,
                f"run {run_id} is in {run.status.value!r}; must be done to follow up",
            )
        if not run.bundle_followup_eligible:
            # Bundle dir is missing — no work to do, no point queueing.
            raise HTTPException(
                410, f"run {run_id} bundle is no longer available for follow-up"
            )
        question = body.question.strip()
        if not question:
            raise HTTPException(400, "question is empty")
        background.add_task(
            execute_followup_run,
            store=STORE,
            run=run,
            question_text=question,
        )
        # followup_index is incremented by the runner just before work
        # starts; report the *anticipated* index so the UI can render
        # "Running follow-up #N" immediately.
        return {
            "run_id": run.run_id,
            "status": "queued",
            "anticipated_followup_index": run.followup_index + 1,
        }

    # ---- answers (resume) ----

    @app.post("/api/runs/{run_id}/answers")
    async def submit_answers(
        run_id: str, body: AnswersRequest, background: BackgroundTasks
    ) -> dict:
        run = STORE.get_run(run_id)
        if run is None:
            raise HTTPException(404, f"run {run_id} not found")
        if run.status not in (RunStatus.PAUSED, RunStatus.DONE):
            raise HTTPException(
                409,
                f"run {run_id} is in {run.status.value!r}; must be paused or done to resume",
            )
        if not body.answers:
            raise HTTPException(400, "no answers provided")
        background.add_task(
            execute_resume,
            store=STORE,
            run=run,
            answers=[(a.qid, a.resolution) for a in body.answers],
        )
        return {"run_id": run.run_id, "status": "resuming"}

    return app


app = create_app()


def run() -> None:
    """Console-script entry: `fermdocs-api`."""
    uvicorn.run(
        "fermdocs_api.main:app",
        host=os.environ.get("FERMDOCS_API_HOST", "127.0.0.1"),
        port=int(os.environ.get("FERMDOCS_API_PORT", "8000")),
        reload=bool(os.environ.get("FERMDOCS_API_RELOAD")),
    )


if __name__ == "__main__":
    run()
