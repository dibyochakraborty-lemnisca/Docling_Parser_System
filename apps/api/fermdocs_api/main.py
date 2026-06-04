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
    Form,
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
    async def upload(
        files: list[UploadFile] = File(...),
        process_family: str | None = Form(default=None),
    ) -> dict:
        """Multipart upload of one OR many files (PR-A3, frontend-redesign).

        Frontend sends `multipart/form-data` with one or more `files=`
        parts. The endpoint validates:
          - non-empty (else 400)
          - extensions in {.csv, .xlsx, .pdf, .zip} (else 400)
          - if any file is a .zip, it must be the only file (zips are
            pre-built bundles that bypass ingest; mixing them with raw
            data has no coherent meaning) (else 400)
          - duplicate filenames are rejected by RunStore.add_upload
            (else 400)

        Response carries lists for filenames and content_types so the
        frontend can show what was actually accepted; size_bytes is the
        sum across all files. For N=1 the response shape stays
        compatible with what existed before — the legacy `filename` /
        `content_type` keys are also returned so any older client
        doesn't break.
        """
        if not files:
            raise HTTPException(400, "at least one file required")

        # Read all bodies up front so we can validate before touching disk.
        # This also means a partial upload (network drop mid-stream)
        # surfaces as a fastapi error before we ever call add_upload.
        ALLOWED_SUFFIXES = {".csv", ".xlsx", ".pdf", ".zip"}
        triples: list[tuple[str, str, bytes]] = []
        for f in files:
            fname = f.filename or "upload.bin"
            suffix = ("." + fname.rsplit(".", 1)[-1].lower()) if "." in fname else ""
            if suffix not in ALLOWED_SUFFIXES:
                raise HTTPException(
                    400,
                    f"unsupported file type {fname!r};"
                    " supported: .csv, .xlsx, .pdf, .zip",
                )
            triples.append((
                fname,
                f.content_type or "application/octet-stream",
                await f.read(),
            ))

        # Zip-mixing rule: zips are pre-built bundles, must be standalone.
        zip_count = sum(1 for fname, _, _ in triples if fname.endswith(".zip"))
        if zip_count > 0 and len(triples) > 1:
            raise HTTPException(
                400,
                "zip uploads must be standalone—cannot mix .zip with other files",
            )

        # Operator-supplied process_family (upload-process-family-ui).
        # Treat empty string, "auto", "auto-detect", "unknown" all as
        # None — those are the dropdown's "let the LLM figure it out"
        # picks. The dossier loader's _validate_manifest_family will
        # also enforce the closed enum, but normalising here keeps the
        # API contract clean.
        pf_raw = (process_family or "").strip().lower()
        canonical_pf: str | None = None
        if pf_raw and pf_raw not in {"auto", "auto-detect", "auto_detect", "unknown", ""}:
            canonical_pf = pf_raw

        try:
            upload = STORE.add_upload(files=triples, process_family=canonical_pf)
        except ValueError as exc:
            # add_upload raises on empty list (handled above) and
            # duplicate filenames (defense-in-depth).
            raise HTTPException(400, str(exc))

        return {
            "upload_id": upload.upload_id,
            "filenames": list(upload.filenames),
            "content_types": list(upload.content_types),
            "size_bytes": upload.size_bytes,
            "process_family": upload.process_family,
            # Legacy single-file keys, populated when N=1 so older
            # clients keep working. None when N>1 — callers should switch
            # to filenames/content_types.
            "filename": upload.filenames[0] if len(upload.filenames) == 1 else None,
            "content_type": upload.content_types[0] if len(upload.content_types) == 1 else None,
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
            
        recommendation_output = None
        if run.recommend_dir and (run.recommend_dir / "recommendation.json").exists():
            recommendation_output = json.loads((run.recommend_dir / "recommendation.json").read_text())

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
            "recommend_dir": str(run.recommend_dir) if run.recommend_dir else None,
            "global_md": str(run.global_md) if run.global_md else None,
            "error": run.error,
            "output": output,
            "recommendation_output": recommendation_output,
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
