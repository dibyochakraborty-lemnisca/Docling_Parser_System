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
    """One upload group — may carry one or many files.

    Single-file invariant: when N=1 (the path that existed before
    multi-file landed), all three list fields have len 1. Downstream
    callers that previously read `upload.path` now read `upload.paths[0]`,
    and likewise for filename/content_type — there is exactly one item.

    Multi-file invariant: every list has the same length, items in the
    same index correspond to the same physical file. Order is the order
    the user picked them.

    `size_bytes` is the SUM across all files (not per-file) — callers
    that surface 'how big was the upload' want the total.
    """

    upload_id: str
    filenames: list[str]
    paths: list[Path]
    content_types: list[str]
    size_bytes: int
    # Operator-supplied process family for the upload-time dropdown
    # (upload-process-family-ui branch). None = auto-detect (LLM
    # extractor runs as before). Closed enum value from
    # process_families.yaml (e.g. "penicillin_fedbatch") forces the
    # manifest path and skips LLM identity extraction. "unknown" is
    # equivalent to None — explicit pick of "no idea, please auto-detect".
    process_family: str | None = None
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
        self,
        *,
        files: list[tuple[str, str, bytes]],
        process_family: str | None = None,
    ) -> Upload:
        """Register one upload group from N files. files is a list of
        (filename, content_type, content) tuples. Atomic: writes to a
        tempdir first, only moves to the final destination on full
        success. On any failure, no Upload record is created and the
        partial tempdir is cleaned up.

        `process_family`: optional operator-supplied closed-vocab name
        from the upload-time UI dropdown. None means auto-detect. The
        ingest pipeline will turn this into a process-manifest if set.

        Raises:
          ValueError("at least one file required") on empty list.
          ValueError("duplicate filename: <name>") on collisions — we
            don't auto-suffix because silent renames are surprising.
            Frontend pre-validates so this should be a defense-in-depth
            backend 400 rather than a user-facing path.
        """
        import shutil
        import tempfile

        if not files:
            raise ValueError("at least one file required")

        # Duplicate-name check first — cheaper than disk I/O.
        seen: set[str] = set()
        for filename, _, _ in files:
            if filename in seen:
                raise ValueError(f"duplicate filename: {filename!r}")
            seen.add(filename)

        upload_id = str(uuid.uuid4())
        final_dir = self.uploads_root / upload_id

        # Atomic write: stage to tempdir, move into place only when every
        # file has succeeded. shutil.move handles cross-filesystem moves
        # via copy-then-delete fallback (slower but correct).
        with tempfile.TemporaryDirectory(
            prefix=f"upload-{upload_id}-", dir=self.uploads_root
        ) as staging:
            staging_path = Path(staging)
            for filename, _content_type, content in files:
                (staging_path / filename).write_bytes(content)
            # All files written successfully; promote to final location.
            final_dir.mkdir(parents=True, exist_ok=False)
            for filename, _, _ in files:
                shutil.move(
                    str(staging_path / filename), str(final_dir / filename)
                )

        paths = [final_dir / fname for fname, _, _ in files]
        upload = Upload(
            upload_id=upload_id,
            filenames=[fname for fname, _, _ in files],
            paths=paths,
            content_types=[ct for _, ct, _ in files],
            size_bytes=sum(len(c) for _, _, c in files),
            process_family=process_family,
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
