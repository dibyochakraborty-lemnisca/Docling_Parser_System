"""In-process run-state store.

For local-only v0.5a we keep run state in memory + on disk. A persistent
store (SQLite or Postgres) is a v1 concern when we want runs to survive
server restarts.

A `Run` is one execution of the full pipeline against an upload. State
machine:
  pending     — uploaded, not started
  ingesting   — building dossier (deterministic)
  characterizing
  diagnosing  — Gemini calls (slow)
  hypothesizing — Gemini calls (slow)
  paused      — exited with unresolved open questions; awaiting answers
  resuming    — running resume_stage with provided answers
  done        — completed (final or after answer rounds)
  failed      — exception
"""

from __future__ import annotations

import asyncio
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any


class RunStatus(str, Enum):
    PENDING = "pending"
    INGESTING = "ingesting"
    CHARACTERIZING = "characterizing"
    DIAGNOSING = "diagnosing"
    HYPOTHESIZING = "hypothesizing"
    PAUSED = "paused"
    RESUMING = "resuming"
    DONE = "done"
    FAILED = "failed"


@dataclass
class Upload:
    upload_id: str
    filename: str
    path: Path
    content_type: str
    size_bytes: int
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))


@dataclass
class FollowupResult:
    """One follow-up question + its hypothesis output, appended to a Run.

    PR-A2 on hitl-followup. Drive posture: after a run reaches DONE, the
    user can submit a follow-up question against the *same* bundle (no
    re-ingest). Each follow-up produces its own HypothesisOutput, stored
    here in order. The bundle's user_question.json holds only the most
    recent question; full history lives in Run.followups.
    """

    followup_index: int  # 1-indexed
    user_question_text: str
    output: Any  # HypothesisOutput; loose to avoid a cross-package import here
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))


@dataclass
class Run:
    run_id: str
    upload_id: str
    status: RunStatus = RunStatus.PENDING
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    bundle_dir: Path | None = None
    hypothesis_dir: Path | None = None
    global_md: Path | None = None
    error: str | None = None
    # PR-A on caisc-hitl: optional human-typed question that biases the
    # debate. Empty string and None both mean "no question, run as today".
    user_question_text: str | None = None
    # PR-A2 on hitl-followup: ordered follow-up results (drive posture).
    # Empty for runs that never received a follow-up. Lives in-memory only;
    # API restart wipes follow-up history (matches today's Run lifetime).
    followups: list[FollowupResult] = field(default_factory=list)
    # 0 means "no follow-ups yet". Incremented before each execute_followup_run
    # so the FollowupResult records which # it is. Equals len(followups)
    # after the run completes; may briefly be ahead while a follow-up is
    # in-flight (status=HYPOTHESIZING and followup_index > 0 means
    # "running follow-up #N").
    followup_index: int = 0
    # Live event subscribers — WebSockets connect and receive future events.
    # We keep one queue per subscriber.
    subscribers: list[asyncio.Queue] = field(default_factory=list, repr=False)

    @property
    def bundle_followup_eligible(self) -> bool:
        """True iff a follow-up against this run could succeed today.

        Follow-up requires the bundle still on disk; retention/GC may
        delete bundle_dir between original run and follow-up. Frontend
        uses this to hide the follow-up textarea when it would 410 Gone.
        """
        return self.bundle_dir is not None and self.bundle_dir.exists()


class RunStore:
    """In-process run + upload registry. Thread-unsafe by design — meant
    to be accessed only from the asyncio event loop."""

    def __init__(self, *, uploads_root: Path, runs_root: Path) -> None:
        # Resolve to absolute paths so subprocess CLIs work regardless of cwd.
        self.uploads_root = uploads_root.resolve()
        self.runs_root = runs_root.resolve()
        self.uploads_root.mkdir(parents=True, exist_ok=True)
        self.runs_root.mkdir(parents=True, exist_ok=True)
        self._uploads: dict[str, Upload] = {}
        self._runs: dict[str, Run] = {}

    # ---- uploads ----

    def add_upload(
        self, *, filename: str, content_type: str, content: bytes
    ) -> Upload:
        upload_id = str(uuid.uuid4())
        target = self.uploads_root / upload_id / filename
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(content)
        upload = Upload(
            upload_id=upload_id,
            filename=filename,
            path=target,
            content_type=content_type,
            size_bytes=len(content),
        )
        self._uploads[upload_id] = upload
        return upload

    def get_upload(self, upload_id: str) -> Upload | None:
        return self._uploads.get(upload_id)

    # ---- runs ----

    def create_run(
        self, upload_id: str, *, user_question_text: str | None = None
    ) -> Run:
        run_id = str(uuid.uuid4())
        # Empty/whitespace string = legacy run (no question), normalize to None
        # so downstream code only checks `is None`.
        cleaned = (user_question_text or "").strip() or None
        run = Run(
            run_id=run_id, upload_id=upload_id, user_question_text=cleaned
        )
        self._runs[run_id] = run
        return run

    def get_run(self, run_id: str) -> Run | None:
        return self._runs.get(run_id)

    def list_runs(self) -> list[Run]:
        return sorted(
            self._runs.values(), key=lambda r: r.created_at, reverse=True
        )

    # ---- follow-ups (PR-A2) ----

    def add_followup(self, run_id: str, result: FollowupResult) -> None:
        """Append a FollowupResult to a run. Caller is responsible for
        having already incremented run.followup_index to result.followup_index
        before kicking off the work."""
        run = self._runs.get(run_id)
        if run is None:
            raise KeyError(f"unknown run_id {run_id!r}")
        run.followups.append(result)

    # ---- pub/sub ----

    async def subscribe(self, run_id: str) -> asyncio.Queue:
        run = self._runs[run_id]
        q: asyncio.Queue = asyncio.Queue()
        run.subscribers.append(q)
        return q

    def unsubscribe(self, run_id: str, q: asyncio.Queue) -> None:
        run = self._runs.get(run_id)
        if run is not None:
            try:
                run.subscribers.remove(q)
            except ValueError:
                pass

    async def publish(self, run_id: str, event: dict[str, Any]) -> None:
        run = self._runs.get(run_id)
        if run is None:
            return
        for q in list(run.subscribers):
            await q.put(event)
