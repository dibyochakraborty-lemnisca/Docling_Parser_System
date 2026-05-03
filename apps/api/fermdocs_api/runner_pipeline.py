"""Async wrapper around the fermdocs pipeline.

Plan ref: plans/2026-05-03-hypothesis-debate-v0.md (v0.5a backend).

Three entry shapes:

1. Bundle upload (.zip of an existing diagnose bundle dir)
   → unzip → load_bundle → run hypothesis stage

2. CSV upload (.csv with experiment data)
   → fermdocs ingest → fermdocs dossier → fermdocs-characterize
   → fermdocs-diagnose → load_bundle → run hypothesis stage

3. PDF upload (.pdf — uses DoclingPdfParser inside ingest)
   → same as CSV

CSV / PDF paths require:
  - DATABASE_URL env var (Postgres for the ingest pipeline)
  - GEMINI_API_KEY for header mapper / diagnose / hypothesis

Each stage publishes a `status` event so the frontend can show progress.
The hypothesis stage's events stream as `event` messages via the
global.md tailer.

Heavy CPU/IO work runs in a thread pool so the asyncio event loop stays
responsive for WebSocket subscribers.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import subprocess
import sys
import uuid
import zipfile
from datetime import datetime, timezone
from pathlib import Path

from fermdocs_hypothesis.bundle_loader import load_bundle
from fermdocs_hypothesis.event_log import read_events
from fermdocs_hypothesis.live_hooks import LiveHooks
from fermdocs_hypothesis.runner import resume_stage, run_stage
from fermdocs_hypothesis.schema import BudgetSnapshot

from fermdocs_api.state import Run, RunStatus, RunStore, Upload

_log = logging.getLogger(__name__)


async def execute_run(
    *,
    store: RunStore,
    run: Run,
    upload: Upload,
) -> None:
    """Background task: take an upload, run the full pipeline, publish events."""
    try:
        run.status = RunStatus.PENDING
        await store.publish(run.run_id, {"type": "status", "status": run.status.value})

        # 1. Resolve to a bundle dir (this may run ingest/characterize/diagnose
        # for CSV/PDF uploads and emit per-stage status updates).
        bundle_dir = await _prepare_bundle_dir(
            upload=upload,
            store=store,
            run=run,
        )
        run.bundle_dir = bundle_dir

        # 2. Run hypothesis stage with live event publishing
        run.status = RunStatus.HYPOTHESIZING
        hyp_dir = store.runs_root / run.run_id
        hyp_dir.mkdir(parents=True, exist_ok=True)
        global_md = hyp_dir / "global.md"
        run.hypothesis_dir = hyp_dir
        run.global_md = global_md

        await store.publish(
            run.run_id,
            {
                "type": "status",
                "status": run.status.value,
                "bundle_dir": str(bundle_dir),
                "hypothesis_dir": str(hyp_dir),
            },
        )

        watcher_task = asyncio.create_task(
            _watch_global_md(store=store, run=run, global_md=global_md)
        )

        result = await asyncio.to_thread(
            _run_hypothesis_blocking, bundle_dir, global_md
        )

        watcher_task.cancel()
        try:
            await watcher_task
        except asyncio.CancelledError:
            pass

        out_path = hyp_dir / "hypothesis_output.json"
        out_path.write_text(
            json.dumps(result.output.model_dump(mode="json"), indent=2, default=str)
        )

        unresolved = [q for q in result.output.open_questions if not q.resolved]
        run.status = RunStatus.PAUSED if unresolved else RunStatus.DONE

        await store.publish(
            run.run_id,
            {
                "type": "result",
                "status": run.status.value,
                "output": result.output.model_dump(mode="json"),
            },
        )
    except Exception as e:
        _log.exception("run %s failed", run.run_id)
        run.status = RunStatus.FAILED
        run.error = f"{type(e).__name__}: {e}"
        await store.publish(
            run.run_id,
            {"type": "error", "status": run.status.value, "error": run.error},
        )


async def execute_resume(
    *,
    store: RunStore,
    run: Run,
    answers: list[tuple[str, str]],
) -> None:
    """Resume a paused run with human answers, run another debate round."""
    if run.bundle_dir is None or run.global_md is None:
        run.status = RunStatus.FAILED
        run.error = "run is not in a resumable state"
        await store.publish(
            run.run_id, {"type": "error", "status": run.status.value, "error": run.error}
        )
        return
    try:
        run.status = RunStatus.RESUMING
        await store.publish(run.run_id, {"type": "status", "status": run.status.value})

        watcher_task = asyncio.create_task(
            _watch_global_md(
                store=store, run=run, global_md=run.global_md, start_from_eof=True
            )
        )
        result = await asyncio.to_thread(
            _resume_hypothesis_blocking, run.bundle_dir, run.global_md, answers
        )
        watcher_task.cancel()
        try:
            await watcher_task
        except asyncio.CancelledError:
            pass

        if run.hypothesis_dir is not None:
            (run.hypothesis_dir / "hypothesis_output.json").write_text(
                json.dumps(result.output.model_dump(mode="json"), indent=2, default=str)
            )

        unresolved = [q for q in result.output.open_questions if not q.resolved]
        run.status = RunStatus.PAUSED if unresolved else RunStatus.DONE
        await store.publish(
            run.run_id,
            {
                "type": "result",
                "status": run.status.value,
                "output": result.output.model_dump(mode="json"),
            },
        )
    except Exception as e:
        _log.exception("resume %s failed", run.run_id)
        run.status = RunStatus.FAILED
        run.error = f"{type(e).__name__}: {e}"
        await store.publish(
            run.run_id,
            {"type": "error", "status": run.status.value, "error": run.error},
        )


# ---------- prepare bundle ----------


async def _prepare_bundle_dir(
    *, upload: Upload, store: RunStore, run: Run
) -> Path:
    """Resolve an upload to a bundle directory. Branches by extension."""
    suffix = upload.path.suffix.lower()
    if suffix == ".zip":
        return await asyncio.to_thread(_unzip_bundle, upload)
    if suffix in (".csv", ".pdf", ".xlsx"):
        return await _build_bundle_from_raw(
            upload=upload, store=store, run=run
        )
    raise ValueError(
        f"upload type not supported: {upload.filename!r}."
        " Supported: .csv, .pdf, .xlsx, or .zip of an existing bundle."
    )


def _unzip_bundle(upload: Upload) -> Path:
    target = upload.path.parent / "bundle"
    if target.exists():
        return _find_bundle_root(target)
    target.mkdir(exist_ok=True)
    with zipfile.ZipFile(upload.path) as zf:
        zf.extractall(target)
    return _find_bundle_root(target)


def _find_bundle_root(extracted: Path) -> Path:
    if (extracted / "meta.json").exists():
        return extracted
    for child in extracted.iterdir():
        if child.is_dir() and (child / "meta.json").exists():
            return child
    raise ValueError(f"no meta.json found in {extracted}")


async def _build_bundle_from_raw(
    *, upload: Upload, store: RunStore, run: Run
) -> Path:
    """Run ingest → dossier → characterize → diagnose to produce a bundle.

    Requires DATABASE_URL (for ingest) and GEMINI_API_KEY (for diagnose).
    """
    if not os.environ.get("DATABASE_URL"):
        raise RuntimeError(
            "DATABASE_URL not set; required for CSV/PDF ingest. Set it in"
            " your .env or upload a pre-built bundle .zip instead."
        )

    # All paths absolute so subprocess CLIs work regardless of cwd.
    work_root = upload.path.parent.resolve()
    experiment_id = f"web-upload-{uuid.uuid4().hex[:8]}"
    dossier_path = work_root / "dossier.json"
    char_path = work_root / "characterization.json"
    bundle_root = work_root / "bundles"
    bundle_root.mkdir(exist_ok=True)

    # 1. Ingest
    run.status = RunStatus.INGESTING
    await store.publish(
        run.run_id,
        {"type": "status", "status": run.status.value, "message": f"ingesting {upload.filename}"},
    )
    await _run_subprocess(
        [
            sys.executable, "-m", "fermdocs.cli", "ingest",
            "--experiment-id", experiment_id,
            "--files", str(upload.path.resolve()),
            "--out", str(dossier_path),
        ],
        cwd=Path(os.environ.get("FERMDOCS_REPO_ROOT", Path.cwd())),
    )
    if not dossier_path.exists():
        raise RuntimeError(
            f"ingest exited cleanly but did not write dossier to"
            f" {dossier_path}. Check ingest logs (DATABASE_URL set? file"
            f" format supported?)."
        )

    # 2. Characterize (with --bundle to write a proper bundle dir)
    run.status = RunStatus.CHARACTERIZING
    await store.publish(
        run.run_id,
        {"type": "status", "status": run.status.value},
    )
    await _run_subprocess(
        [
            sys.executable, "-m", "fermdocs_characterize.cli",
            str(dossier_path),
            "--out", str(char_path),
            "--bundle", str(bundle_root),
        ],
        cwd=Path(os.environ.get("FERMDOCS_REPO_ROOT", Path.cwd())),
    )

    # Locate the bundle that characterize just wrote
    bundles = sorted(
        (p for p in bundle_root.iterdir() if p.is_dir() and (p / "meta.json").exists()),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    if not bundles:
        raise RuntimeError(
            f"characterize did not write a bundle under {bundle_root}"
        )
    bundle_dir = bundles[0]

    # 3. Diagnose
    run.status = RunStatus.DIAGNOSING
    await store.publish(
        run.run_id,
        {"type": "status", "status": run.status.value, "bundle_dir": str(bundle_dir)},
    )
    diagnosis_path = bundle_dir / "diagnosis" / "diagnosis.json"
    await _run_subprocess(
        [
            sys.executable, "-m", "fermdocs_diagnose.cli", "run",
            "--dossier", str(dossier_path),
            "--characterization", str(char_path),
            "--output", str(diagnosis_path),
        ],
        cwd=Path(os.environ.get("FERMDOCS_REPO_ROOT", Path.cwd())),
    )
    if not diagnosis_path.exists():
        raise RuntimeError(f"diagnose did not produce {diagnosis_path}")

    return bundle_dir


async def _run_subprocess(cmd: list[str], cwd: Path | None = None) -> None:
    """Run a subprocess; on failure include stderr+stdout in the error.

    Logs the command + cwd at INFO so server logs show what fired.
    """
    _log.info("subprocess: %s (cwd=%s)", " ".join(cmd), cwd)
    proc = await asyncio.create_subprocess_exec(
        *cmd,
        cwd=str(cwd) if cwd else None,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
        env=os.environ.copy(),
    )
    stdout, stderr = await proc.communicate()
    if proc.returncode != 0:
        err = stderr.decode(errors="replace") if stderr else ""
        out = stdout.decode(errors="replace") if stdout else ""
        # Show stderr first; some CLIs print structured errors to stdout
        body = (err + ("\n--- stdout ---\n" + out if out else "")).strip()
        raise RuntimeError(
            f"command failed (exit {proc.returncode}): {' '.join(cmd[:3])}…\n"
            f"{body[-2500:] or '(no output)'}"
        )


# ---------- blocking hypothesis-stage helpers ----------


def _run_hypothesis_blocking(bundle_dir: Path, global_md: Path):
    loaded = load_bundle(bundle_dir)
    hooks = LiveHooks(loaded)
    return run_stage(
        hyp_input=loaded.hyp_input,
        hooks=hooks,
        global_md_path=global_md,
        diagnosis_id=loaded.diagnosis.meta.diagnosis_id,
        provider="gemini",
        model_name=hooks._client.model_name,
        budget=BudgetSnapshot(),
        validate=True,
        now_factory=lambda: datetime.now(timezone.utc),
    )


def _resume_hypothesis_blocking(
    bundle_dir: Path, global_md: Path, answers: list[tuple[str, str]]
):
    loaded = load_bundle(bundle_dir)
    hooks = LiveHooks(loaded)
    return resume_stage(
        hyp_input=loaded.hyp_input,
        hooks=hooks,
        global_md_path=global_md,
        diagnosis_id=loaded.diagnosis.meta.diagnosis_id,
        answers=answers,
        provider="gemini",
        model_name=hooks._client.model_name,
        budget=BudgetSnapshot(),
        validate=True,
        now_factory=lambda: datetime.now(timezone.utc),
    )


async def _watch_global_md(
    *, store: RunStore, run: Run, global_md: Path, start_from_eof: bool = False
) -> None:
    """Tail global.md, publish each new event to subscribers."""
    seen = 0
    if start_from_eof and global_md.exists():
        seen = len(read_events(global_md))
    try:
        while True:
            if global_md.exists():
                events = read_events(global_md)
                if len(events) > seen:
                    for ev in events[seen:]:
                        await store.publish(
                            run.run_id,
                            {"type": "event", "event": ev.model_dump(mode="json")},
                        )
                    seen = len(events)
            await asyncio.sleep(0.5)
    except asyncio.CancelledError:
        if global_md.exists():
            events = read_events(global_md)
            if len(events) > seen:
                for ev in events[seen:]:
                    await store.publish(
                        run.run_id,
                        {"type": "event", "event": ev.model_dump(mode="json")},
                    )
        raise
