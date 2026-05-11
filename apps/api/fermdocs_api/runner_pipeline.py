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
from fermdocs_hypothesis.schema import (
    BudgetSnapshot,
    FollowupContext,
    HypothesisOutput,
    PriorFollowupRef,
    PriorHypothesisRef,
)

from fermdocs_api.state import FollowupResult, Run, RunStatus, RunStore, Upload

_log = logging.getLogger(__name__)


# API-runner budget: 2× BudgetSnapshot defaults so debates explore more
# topics and run more critic cycles per topic before terminating. Trade-off
# is ~2× cost and wall time per run. Bump these together if you raise one.
_API_BUDGET = BudgetSnapshot(
    max_turns=20,                     # 2× default 10 — more topics covered
    max_critic_cycles_per_topic=6,    # 2× default 3 — deeper per-topic refinement
    max_tool_calls_total=160,         # 2× default 80 — covers raised turn cap
    max_total_input_tokens=400_000,   # 2× default 200k — hard token ceiling
    max_open_questions=30,            # 2× default 15 — more OQs allowed
)


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

    bundle_dir = run.bundle_dir
    global_md = run.global_md
    hypothesis_dir = run.hypothesis_dir

    def _do_work():
        return _resume_hypothesis_blocking(bundle_dir, global_md, answers)

    def _on_success(result) -> None:
        if hypothesis_dir is not None:
            (hypothesis_dir / "hypothesis_output.json").write_text(
                json.dumps(result.output.model_dump(mode="json"), indent=2, default=str)
            )
        unresolved = [q for q in result.output.open_questions if not q.resolved]
        run.status = RunStatus.PAUSED if unresolved else RunStatus.DONE

    await _run_hypothesis_with_lifecycle(
        store=store,
        run=run,
        target_status=RunStatus.RESUMING,
        global_md=global_md,
        do_blocking_work=_do_work,
        on_success=_on_success,
        log_label="resume",
    )


async def execute_followup_run(
    *,
    store: RunStore,
    run: Run,
    question_text: str,
) -> None:
    """Background task: run a follow-up question against an already-DONE run.

    Drive posture (PR-A2). Bundle is frozen — we DO NOT re-run
    ingest/characterize/diagnose. Only the hypothesis stage fires, with
    the new user_question.json overwriting the prior one in the bundle.

    Caller is responsible for asserting `run.status == DONE` and
    `run.bundle_followup_eligible` *before* spawning this task; we re-
    assert here as a defense-in-depth invariant.
    """
    if run.bundle_dir is None or run.global_md is None:
        run.status = RunStatus.FAILED
        run.error = "run has no bundle for follow-up"
        await store.publish(
            run.run_id, {"type": "error", "status": run.status.value, "error": run.error}
        )
        return
    if not run.bundle_dir.exists():
        run.status = RunStatus.FAILED
        run.error = f"bundle_dir {run.bundle_dir} no longer exists on disk"
        await store.publish(
            run.run_id, {"type": "error", "status": run.status.value, "error": run.error}
        )
        return

    bundle_dir = run.bundle_dir
    global_md = run.global_md
    hypothesis_dir = run.hypothesis_dir

    # Increment first so any concurrent GET sees status=HYPOTHESIZING +
    # followup_index=N "running follow-up #N" before the work starts.
    run.followup_index += 1
    this_index = run.followup_index

    # Classify the new question against the bundle's actual run_ids/variables
    # (same path as PR-A's _classify_and_persist_user_question, but with
    # raised_by="user_followup" baked in by question_classifier).
    char_path = bundle_dir / "characterization.json"
    if not char_path.exists():
        # Some legacy bundles don't ship characterization.json next to
        # meta.json — try the standard nested location.
        alt = bundle_dir / "characterization" / "characterization.json"
        if alt.exists():
            char_path = alt
    try:
        await asyncio.to_thread(
            _classify_and_persist_followup_question,
            question_text=question_text,
            bundle_dir=bundle_dir,
            char_path=char_path,
        )
    except Exception as exc:
        _log.warning(
            "follow-up classification failed (%s: %s); continuing with raw text",
            exc.__class__.__name__, str(exc)[:200],
        )

    followup_context = _build_followup_context(run)

    def _do_work():
        return _run_hypothesis_blocking(
            bundle_dir, global_md, followup_context=followup_context
        )

    def _on_success(result) -> None:
        if hypothesis_dir is not None:
            # Per-followup output file so prior follow-up outputs don't clobber.
            out_path = hypothesis_dir / f"hypothesis_output_followup_{this_index}.json"
            out_path.write_text(
                json.dumps(result.output.model_dump(mode="json"), indent=2, default=str)
            )
        store.add_followup(
            run.run_id,
            FollowupResult(
                followup_index=this_index,
                user_question_text=question_text,
                output=result.output,
            ),
        )
        # Follow-ups don't honor open_questions → PAUSED — drive posture
        # treats the answer as terminal. Goes straight back to DONE so
        # the user can submit yet another follow-up.
        run.status = RunStatus.DONE

    await _run_hypothesis_with_lifecycle(
        store=store,
        run=run,
        target_status=RunStatus.HYPOTHESIZING,
        global_md=global_md,
        do_blocking_work=_do_work,
        on_success=_on_success,
        log_label="follow-up",
        extra_status_payload={"followup_index": this_index},
    )


async def _run_hypothesis_with_lifecycle(
    *,
    store: RunStore,
    run: Run,
    target_status: RunStatus,
    global_md: Path,
    do_blocking_work,
    on_success,
    log_label: str,
    extra_status_payload: dict | None = None,
) -> None:
    """Shared skeleton for execute_resume and execute_followup_run.

    Pattern:
        set status → publish status → start watcher → run blocking work
        → cancel watcher → on_success(result) → publish result event
        on exception → set FAILED → publish error event

    execute_run is intentionally NOT routed through this — its bundle-prep
    stages have unique status hops and are well-tested; refactoring them
    here would risk regressing PR-A's pipeline path for no DRY win on the
    skeleton we're trying to dedupe.
    """
    try:
        run.status = target_status
        status_event = {"type": "status", "status": run.status.value}
        if extra_status_payload:
            status_event.update(extra_status_payload)
        await store.publish(run.run_id, status_event)

        watcher_task = asyncio.create_task(
            _watch_global_md(
                store=store, run=run, global_md=global_md, start_from_eof=True
            )
        )
        result = await asyncio.to_thread(do_blocking_work)
        watcher_task.cancel()
        try:
            await watcher_task
        except asyncio.CancelledError:
            pass

        on_success(result)

        await store.publish(
            run.run_id,
            {
                "type": "result",
                "status": run.status.value,
                "output": result.output.model_dump(mode="json"),
            },
        )
    except Exception as e:
        _log.exception("%s %s failed", log_label, run.run_id)
        run.status = RunStatus.FAILED
        run.error = f"{type(e).__name__}: {e}"
        await store.publish(
            run.run_id,
            {"type": "error", "status": run.status.value, "error": run.error},
        )


def _classify_and_persist_followup_question(
    *,
    question_text: str,
    bundle_dir: Path,
    char_path: Path,
) -> Path:
    """Same shape as _classify_and_persist_user_question (PR-A) but
    raised_by='user_followup'. Overwrites bundle/user_question.json.
    """
    import json as _json

    from fermdocs_hypothesis.question_classifier import (
        GeminiQuestionClassifierClient,
        classify_user_question,
    )

    if char_path.exists():
        char_data = _json.loads(char_path.read_text())
        run_ids = sorted({t.get("run_id") for t in char_data.get("trajectories") or [] if t.get("run_id")})
        variables = sorted({t.get("variable") for t in char_data.get("trajectories") or [] if t.get("variable")})
    else:
        run_ids, variables = [], []

    client: GeminiQuestionClassifierClient | None
    try:
        client = GeminiQuestionClassifierClient()
    except Exception:
        client = None

    question = classify_user_question(
        text=question_text,
        available_run_ids=run_ids,
        available_variables=variables,
        client=client,
        raised_by="user_followup",
    )

    target = bundle_dir / "user_question.json"
    target.write_text(_json.dumps(question.model_dump(mode="json"), indent=2))
    return target


def _build_followup_context(run: Run) -> FollowupContext | None:
    """Build compact prior-answer context for a follow-up hypothesis run.

    The agents should see prior conclusions, not a full prior debate transcript
    or rendered chart payloads. Missing/corrupt prior output degrades to prior
    completed follow-ups only; the follow-up still runs over the frozen bundle.
    """
    original_finals: list[PriorHypothesisRef] = []
    if run.hypothesis_dir is not None:
        output_path = run.hypothesis_dir / "hypothesis_output.json"
        if output_path.exists():
            try:
                output = HypothesisOutput.model_validate_json(output_path.read_text())
                original_finals = [
                    _prior_hypothesis_ref(h)
                    for h in output.final_hypotheses
                ]
            except Exception as exc:
                _log.warning(
                    "follow-up context: failed to load prior output (%s: %s)",
                    exc.__class__.__name__, str(exc)[:200],
                )

    prior_followups = [
        PriorFollowupRef(
            followup_index=f.followup_index,
            user_question_text=f.user_question_text,
            final_hypotheses=[
                _prior_hypothesis_ref(h) for h in f.output.final_hypotheses
            ],
        )
        for f in run.followups
    ]

    if not original_finals and not prior_followups:
        return None
    return FollowupContext(
        original_final_hypotheses=original_finals,
        previous_followups=prior_followups,
    )


def _prior_hypothesis_ref(h) -> PriorHypothesisRef:
    return PriorHypothesisRef(
        hyp_id=h.hyp_id,
        summary=h.summary,
        question_answered=h.question_answered,
        question_response_summary=h.question_response_summary,
        affected_variables=list(h.affected_variables),
        confidence=h.confidence,
        actionable_recommendation=h.actionable_recommendation,
        parent_hypothesis_ids=list(h.parent_hypothesis_ids),
    )


# ---------- prepare bundle ----------


async def _prepare_bundle_dir(
    *, upload: Upload, store: RunStore, run: Run
) -> Path:
    """Resolve an upload to a bundle directory. Branches by extension.

    Multi-file uploads (PR-A3, branch frontend-redesign): when N>1 every
    file must be a raw data type (.csv/.xlsx/.pdf) — .zip uploads are
    standalone (one zip per upload) because they're pre-built bundles
    that bypass ingest entirely. Validation happens upstream at the API
    boundary; this branch is defense-in-depth.
    """
    if len(upload.paths) == 1:
        suffix = upload.paths[0].suffix.lower()
        if suffix == ".zip":
            return await asyncio.to_thread(_unzip_bundle, upload)
        if suffix in (".csv", ".pdf", ".xlsx"):
            return await _build_bundle_from_raw(
                upload=upload, store=store, run=run
            )
        raise ValueError(
            f"upload type not supported: {upload.filenames[0]!r}."
            " Supported: .csv, .pdf, .xlsx, or .zip of an existing bundle."
        )
    # N > 1: every file must be raw data, never a zip.
    bad = [p for p in upload.paths if p.suffix.lower() not in (".csv", ".pdf", ".xlsx")]
    if bad:
        raise ValueError(
            f"multi-file upload contained unsupported file: {bad[0].name!r}."
            " Multi-file uploads must be all .csv/.xlsx/.pdf; zips are standalone."
        )
    return await _build_bundle_from_raw(upload=upload, store=store, run=run)


def _unzip_bundle(upload: Upload) -> Path:
    """Single-file zip path. Caller guarantees len(upload.paths)==1."""
    zip_path = upload.paths[0]
    target = zip_path.parent / "bundle"
    if target.exists():
        return _find_bundle_root(target)
    target.mkdir(exist_ok=True)
    with zipfile.ZipFile(zip_path) as zf:
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

    # All paths absolute so subprocess CLIs work regardless of cwd. Every
    # file in a multi-file upload shares the same parent dir (one upload
    # group lives in one directory), so paths[0].parent is the work root
    # for the whole batch.
    work_root = upload.paths[0].parent.resolve()
    experiment_id = f"web-upload-{uuid.uuid4().hex[:8]}"
    dossier_path = work_root / "dossier.json"
    char_path = work_root / "characterization.json"
    bundle_root = work_root / "bundles"
    bundle_root.mkdir(exist_ok=True)

    # 1. Ingest
    run.status = RunStatus.INGESTING
    # Status message lists all filenames so the user sees what's being
    # ingested; for N=1 this is the original single-file behavior.
    msg_files = ", ".join(upload.filenames)
    await store.publish(
        run.run_id,
        {"type": "status", "status": run.status.value, "message": f"ingesting {msg_files}"},
    )
    # `fermdocs ingest --files` is multi-valued (`multiple=True` in the
    # click definition); we expand to one --files <path> per upload file.
    ingest_cmd = [
        sys.executable, "-m", "fermdocs.cli", "ingest",
        "--experiment-id", experiment_id,
    ]
    for p in upload.paths:
        ingest_cmd.extend(["--files", str(p.resolve())])
    ingest_cmd.extend(["--out", str(dossier_path)])

    # Operator-supplied process_family from the upload dropdown
    # (upload-process-family-ui). Writes a minimal manifest YAML next
    # to the dossier and passes --process-manifest to the ingest CLI,
    # which forces provenance=MANIFEST on the resulting RegisteredProcess
    # and skips the LLM identity extractor. This is how the dropdown
    # short-circuits the CSV-only path that otherwise can't classify
    # families without narrative text.
    if upload.process_family:
        manifest_path = work_root / "_upload_manifest.yaml"
        manifest_path.write_text(
            f"process_family: {upload.process_family}\n"
            f"rationale: operator-supplied at upload (UI dropdown)\n"
            f"confidence: 1.0\n"
        )
        ingest_cmd.extend(["--process-manifest", str(manifest_path)])
        _log.info(
            "ingest: using operator-supplied process_family=%r via manifest %s",
            upload.process_family, manifest_path,
        )

    await _run_subprocess(
        ingest_cmd,
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

    # 2b. Classify user question (PR-A) once we know the bundle's actual
    # run_ids and variables. Write the resulting UserQuestion JSON into
    # the bundle so diagnose's CLI can load it via --user-question-path
    # and the in-process hypothesis stage can re-load it from the same
    # path. Empty/None question_text → no file written, no flag passed.
    user_question_path: Path | None = None
    if run.user_question_text:
        try:
            user_question_path = await asyncio.to_thread(
                _classify_and_persist_user_question,
                question_text=run.user_question_text,
                bundle_dir=bundle_dir,
                char_path=char_path,
            )
        except Exception as exc:  # never block the run on classifier issues
            _log.warning(
                "user_question classification failed (%s: %s); continuing without it",
                exc.__class__.__name__, str(exc)[:200],
            )
            user_question_path = None

    # 3. Diagnose
    run.status = RunStatus.DIAGNOSING
    await store.publish(
        run.run_id,
        {"type": "status", "status": run.status.value, "bundle_dir": str(bundle_dir)},
    )
    diagnosis_path = bundle_dir / "diagnosis" / "diagnosis.json"
    diagnose_cmd = [
        sys.executable, "-m", "fermdocs_diagnose.cli", "run",
        "--dossier", str(dossier_path),
        "--characterization", str(char_path),
        "--output", str(diagnosis_path),
    ]
    if user_question_path is not None:
        diagnose_cmd.extend(["--user-question-path", str(user_question_path)])
    await _run_subprocess(
        diagnose_cmd,
        cwd=Path(os.environ.get("FERMDOCS_REPO_ROOT", Path.cwd())),
    )
    if not diagnosis_path.exists():
        raise RuntimeError(f"diagnose did not produce {diagnosis_path}")

    return bundle_dir


def _classify_and_persist_user_question(
    *,
    question_text: str,
    bundle_dir: Path,
    char_path: Path,
) -> Path:
    """Classify the question against the bundle's actual metadata, write
    the resulting UserQuestion JSON to <bundle_dir>/user_question.json.

    Returns the path. Pure function (deterministic given inputs + LLM
    response). Lives outside _build_bundle_from_raw so the asyncio
    event loop doesn't block on the LLM call.
    """
    import json as _json

    from fermdocs.domain.user_question import UserQuestion
    from fermdocs_hypothesis.question_classifier import (
        GeminiQuestionClassifierClient,
        classify_user_question,
    )

    # Pull run_ids + variables from the just-written characterization.
    char_data = _json.loads(char_path.read_text())
    run_ids = sorted({t.get("run_id") for t in char_data.get("trajectories") or [] if t.get("run_id")})
    variables = sorted({t.get("variable") for t in char_data.get("trajectories") or [] if t.get("variable")})

    # Production: Gemini classifier. Stub mode (no API key) falls back
    # cleanly to UserQuestion(shape='open', no hints) inside the function.
    client: GeminiQuestionClassifierClient | None
    try:
        client = GeminiQuestionClassifierClient()
    except Exception:
        client = None

    question = classify_user_question(
        text=question_text,
        available_run_ids=run_ids,
        available_variables=variables,
        client=client,
    )

    target = bundle_dir / "user_question.json"
    target.write_text(
        _json.dumps(question.model_dump(mode="json"), indent=2)
    )
    return target


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


def _build_memory_backend():
    """Construct the memory backend from env config.

    Env:
      FERMDOCS_MEMORY=synap  → SynapBackend (requires SYNAP_API_KEY)
      FERMDOCS_MEMORY=noop or unset → NoopBackend (default; off)

    Failures (missing dep, missing key) downgrade gracefully to
    NoopBackend so a misconfigured prod env doesn't break runs.
    """
    from fermdocs_memory import NoopBackend
    mode = (os.environ.get("FERMDOCS_MEMORY") or "noop").strip().lower()
    if mode == "synap":
        try:
            from fermdocs_memory import _build_synap_backend
            return _build_synap_backend()
        except Exception as exc:
            _log.warning(
                "FERMDOCS_MEMORY=synap requested but backend construction"
                " failed (%s); falling back to NoopBackend",
                exc.__class__.__name__,
            )
            return NoopBackend()
    return NoopBackend()


def _run_hypothesis_blocking(
    bundle_dir: Path,
    global_md: Path,
    *,
    followup_context: FollowupContext | None = None,
):
    loaded = load_bundle(bundle_dir, followup_context=followup_context)
    memory = _build_memory_backend()
    hooks = LiveHooks(loaded, memory=memory)
    return run_stage(
        hyp_input=loaded.hyp_input,
        hooks=hooks,
        global_md_path=global_md,
        diagnosis_id=loaded.diagnosis.meta.diagnosis_id,
        provider="gemini",
        model_name=hooks._client.model_name,
        budget=_API_BUDGET,
        memory=memory,
        validate=True,
        now_factory=lambda: datetime.now(timezone.utc),
    )


def _resume_hypothesis_blocking(
    bundle_dir: Path, global_md: Path, answers: list[tuple[str, str]]
):
    loaded = load_bundle(bundle_dir)
    memory = _build_memory_backend()
    hooks = LiveHooks(loaded, memory=memory)
    return resume_stage(
        hyp_input=loaded.hyp_input,
        hooks=hooks,
        global_md_path=global_md,
        diagnosis_id=loaded.diagnosis.meta.diagnosis_id,
        answers=answers,
        provider="gemini",
        model_name=hooks._client.model_name,
        budget=_API_BUDGET,
        memory=memory,
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
