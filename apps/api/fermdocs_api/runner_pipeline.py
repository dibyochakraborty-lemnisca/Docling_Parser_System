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

from fermdocs_recommend.agent import RecommendationAgent
from fermdocs_recommend.llm_clients import build_recommend_client
from fermdocs_recommend.schema import RecommendationOutput

def _run_recommendation_blocking(
    bundle_dir: Path,
    hypothesis_output_path: Path | None = None,
    run_id: str | None = None,
) -> RecommendationOutput:
    from fermdocs.bundle import BundleReader

    reader = BundleReader(bundle_dir)
    client = build_recommend_client()
    agent = RecommendationAgent(client=client)
    return agent.recommend(
        reader, hypothesis_output_path=hypothesis_output_path, run_id=run_id
    )

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
        
        if run.status == RunStatus.DONE:
            await _try_recommendation(store, run, bundle_dir)

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


async def execute_optimization_run(
    *,
    store: RunStore,
    run: Run,
    upload: Upload,
) -> None:
    """Optimization workflow: ingest/characterize the upload, run the opportunity
    debate (reuses the debate engine — Gemini), then build the optimization result.

    The closed-loop optimizer (fit → propose → simulate-on-oracle) only runs when a
    real process simulator is configured (see `_optimizer_simulator_available`);
    otherwise the debated levers stand as the optimization plan. Either way the
    result carries a MODEL LOG (the governing equations + any fits) so the UI can
    show how the agent uses the model.
    """
    try:
        run.status = RunStatus.PENDING
        await store.publish(run.run_id, {"type": "status", "status": run.status.value,
                                         "workflow": run.workflow.value})

        bundle_dir = await _prepare_bundle_dir(upload=upload, store=store, run=run)
        run.bundle_dir = bundle_dir

        # Opportunity debate. Reuse the global.md tailer so events stream live.
        run.status = RunStatus.DEBATING_OPPORTUNITIES
        opt_dir = store.runs_root / run.run_id
        opt_dir.mkdir(parents=True, exist_ok=True)
        global_md = opt_dir / "global.md"
        run.optimize_dir = opt_dir
        run.global_md = global_md
        await store.publish(run.run_id, {
            "type": "status", "status": run.status.value,
            "bundle_dir": str(bundle_dir), "optimize_dir": str(opt_dir),
        })

        watcher_task = asyncio.create_task(
            _watch_global_md(store=store, run=run, global_md=global_md)
        )
        debate_result = await asyncio.to_thread(
            _run_opportunity_debate_blocking, bundle_dir, global_md
        )
        watcher_task.cancel()
        try:
            await watcher_task
        except asyncio.CancelledError:
            pass

        (opt_dir / "optimization_debate.json").write_text(
            json.dumps(debate_result.output.model_dump(mode="json"), indent=2, default=str)
        )

        # Assemble the optimization output (levers + model log; closed loop if a
        # simulator is available). Status flips to OPTIMIZING for that phase.
        run.status = RunStatus.OPTIMIZING
        await store.publish(run.run_id, {"type": "status", "status": run.status.value})
        opt_output = await asyncio.to_thread(
            _assemble_optimization_output, bundle_dir, debate_result, run.run_id
        )
        run.optimization_output = opt_output
        (opt_dir / "optimization.json").write_text(json.dumps(opt_output, indent=2, default=str))

        run.status = RunStatus.DONE
        await store.publish(run.run_id, {
            "type": "result", "status": run.status.value,
            "optimization_output": opt_output,
        })
    except Exception as e:
        _log.exception("optimization run %s failed", run.run_id)
        run.status = RunStatus.FAILED
        run.error = f"{type(e).__name__}: {e}"
        await store.publish(
            run.run_id, {"type": "error", "status": run.status.value, "error": run.error}
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


async def _try_recommendation(store: RunStore, run: Run, bundle_dir: Path) -> None:
    """Run the recommendation stage on a DONE run.

    Best-effort: any failure logs and leaves the run DONE — the recommendation
    stage must NEVER flip a completed run to FAILED. The agent itself already
    returns a structured refusal rather than raising, but we wrap defensively
    (client construction, disk, threading) so nothing here can break the run.
    """
    try:
        run.status = RunStatus.RECOMMENDING
        run.recommend_dir = bundle_dir / "recommend"
        run.recommend_dir.mkdir(parents=True, exist_ok=True)
        await store.publish(
            run.run_id,
            {
                "type": "status",
                "status": run.status.value,
                "recommend_dir": str(run.recommend_dir),
            },
        )

        hyp_output_path = None
        if run.hypothesis_dir is not None:
            candidate = Path(run.hypothesis_dir) / "hypothesis_output.json"
            if candidate.exists():
                hyp_output_path = candidate

        rec_result = await asyncio.to_thread(
            _run_recommendation_blocking, bundle_dir, hyp_output_path, run.run_id
        )
        out_path = run.recommend_dir / "recommendation.json"
        out_path.write_text(
            json.dumps(rec_result.model_dump(mode="json"), indent=2, default=str)
        )
    except Exception:  # noqa: BLE001 — recommendation must never fail the run
        _log.exception("recommendation stage errored for run %s (run stays DONE)", run.run_id)
    finally:
        run.status = RunStatus.DONE
    # The caller (execute_run / on_success) publishes the final result event.

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

        if run.status == RunStatus.DONE and run.bundle_dir is not None:
            await _try_recommendation(store, run, run.bundle_dir)

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


# ---------- optimization-workflow blocking helpers ----------


def _run_opportunity_debate_blocking(bundle_dir: Path, global_md: Path):
    """Run the opportunity debate over a bundle (reuses the debate engine)."""
    from fermdocs_optimize_debate.loader import load_optimization_bundle
    from fermdocs_optimize_debate.run import run_optimization_debate

    loaded = load_optimization_bundle(bundle_dir)
    return run_optimization_debate(
        loaded, global_md_path=global_md, provider="gemini",
        budget=_API_BUDGET, validate=True,
    )


def _optimizer_simulator_available(bundle_dir: Path) -> bool:
    """True iff the closed-loop optimizer can run: a process simulator (oracle)
    is configured. Today that means the LABS `generate-batches` CLI on PATH plus
    a mech-params file via FERMDOCS_OPTIMIZE_MECH_PARAMS. Absent → debate-only
    (the honest default; we never optimize against a fake oracle)."""
    import shutil

    mech = os.environ.get("FERMDOCS_OPTIMIZE_MECH_PARAMS")
    gen_bin = os.environ.get("FERMDOCS_GENERATE_BATCHES_BIN", "generate-batches")
    return bool(mech and Path(mech).exists() and shutil.which(gen_bin))


def _assemble_optimization_output(bundle_dir: Path, debate_result, run_id: str) -> dict:
    """Build the OptimizationOutput-shaped dict the frontend renders.

    Always: the debated levers + a MODEL LOG (governing equations). When a real
    simulator is configured, also runs the closed-loop optimizer and folds in its
    achieved titer + per-round fit logs."""
    from fermdocs_optimize.models.mechanistic import MechanisticModel
    from fermdocs_optimize_debate.schema import levers_from_output

    levers = [lev.model_dump() for lev in levers_from_output(debate_result.output)]
    base = {
        "meta": {"run_id": run_id, "stage": "opportunity_debate"},
        "best_candidate": None, "best_achieved_titer": None,
        "baseline_titer": None, "improvement": None,
        "levers": levers,
        "simulator_available": _optimizer_simulator_available(bundle_dir),
    }

    if base["simulator_available"]:
        try:
            out = {**base, **_run_closed_loop_optimizer(bundle_dir, run_id, levers)}
            # Equation discovery: agent writes the ODE structure, oracle refines it,
            # scipy searches the equation, oracle verifies. Best-effort: a failure
            # here never sinks the optimizer result.
            if os.environ.get("FERMDOCS_OPTIMIZE_DISCOVERY", "1") != "0":
                try:
                    out["discovery"] = _run_equation_discovery(bundle_dir)
                except Exception as exc:  # noqa: BLE001
                    _log.exception("equation discovery failed; omitting from output")
                    out.setdefault("meta", {})["discovery_error"] = f"{type(exc).__name__}: {exc}"
            return out
        except Exception as exc:  # noqa: BLE001 — fall back to debate-only, never fail the run
            _log.exception("closed-loop optimizer failed; reporting debate levers only")
            base["meta"]["optimizer_error"] = f"{type(exc).__name__}: {exc}"

    # Debate-only: still show the equations the agent's model uses.
    model_log = [MechanisticModel.model_card(), {
        "kind": "note",
        "title": "Closed-loop optimizer not run",
        "detail": ("No process simulator (oracle) is configured for this experiment, "
                   "so the agent reported the debated levers as the optimization plan. "
                   "With a simulator, the loop would fit the model above, propose knob "
                   "settings, and verify the achieved titer on the oracle."),
    }]
    summary = getattr(debate_result.output, "debate_summary", "") or \
        "Opportunity debate complete. Prioritized levers are listed below."
    return {
        **base,
        "confident": bool(levers),
        "refusal_reason": None if levers else "no_levers",
        "selection_rationale": summary,
        "model_log": model_log,
    }


def _seed_training_from_bundle(bundle_dir: Path):
    """Best-effort seed batches from the uploaded experiment.

    Looks for a wide-schema observations CSV (batch/run_id, t, X, S, P, M, V) in
    the bundle. Returns a DataFrame in the optimizer's training schema, or None
    when the columns aren't present (caller falls back to a configured CSV). We
    deliberately do NOT pivot long-format here — a wrong reshape would silently
    corrupt the seed; better to fall back."""
    import pandas as pd

    candidates = [
        bundle_dir / "characterization" / "observations.csv",
        bundle_dir / "observations.csv",
    ]
    needed = {"t", "X", "S", "P", "M", "V"}
    for path in candidates:
        if not path.exists():
            continue
        try:
            df = pd.read_csv(path)
        except Exception:  # noqa: BLE001
            continue
        cols = set(df.columns)
        if "batch" not in cols and "run_id" in cols:
            df = df.rename(columns={"run_id": "batch"})
            cols = set(df.columns)
        if needed <= cols and "batch" in cols:
            keep = ["batch", "t", "X", "S", "P", "M", "V"]
            return df[keep].copy()
    return None


def _run_closed_loop_optimizer(bundle_dir: Path, run_id: str, levers: list[dict]) -> dict:
    """Run the closed-loop optimizer against a configured simulator. Returns the
    fields that overlay the debate-only base. Requires LABS + mech-params (gated
    upstream by `_optimizer_simulator_available`)."""
    import pandas as pd

    from fermdocs_optimize.agent import OptimizerAgent
    from fermdocs_optimize.llm_clients import build_optimize_client
    from fermdocs_optimize.schema import Box
    from fermdocs_optimize.simulators.labs import LABSSimulator

    mech = os.environ["FERMDOCS_OPTIMIZE_MECH_PARAMS"]
    gen_bin = os.environ.get("FERMDOCS_GENERATE_BATCHES_BIN", "generate-batches")
    box_cfg = os.environ.get("FERMDOCS_OPTIMIZE_BOX")  # config.json with var_params
    if not box_cfg:
        raise RuntimeError("FERMDOCS_OPTIMIZE_BOX must be set (config.json with var_params)")

    import json as _json
    vp = _json.loads(Path(box_cfg).read_text()).get("var_params")
    box = Box(**{k: (vp[k]["lb"], vp[k]["ub"]) for k in
                 ("biomass", "total_sub", "malt_frac", "dilution")})

    # Seed data: prefer the uploaded experiment (bundle), fall back to a
    # configured train CSV. The bundle data IS the experiment we're optimizing.
    train = _seed_training_from_bundle(bundle_dir)
    if train is None:
        train_csv = os.environ.get("FERMDOCS_OPTIMIZE_TRAIN")
        if not train_csv:
            raise RuntimeError(
                "no seed batches: bundle has no wide-schema observations and "
                "FERMDOCS_OPTIMIZE_TRAIN is unset")
        train = pd.read_csv(train_csv)

    simulator = LABSSimulator(mech, generate_batches_bin=gen_bin)
    client = build_optimize_client(os.environ.get("FERMDOCS_OPTIMIZE_PROVIDER", "none"))
    agent = OptimizerAgent(client, provider=os.environ.get("FERMDOCS_OPTIMIZE_PROVIDER", "none") if client else "none")
    out = agent.optimize(training_data=train, box=box, simulator=simulator)
    return {
        "confident": out.confident,
        "refusal_reason": out.refusal_reason,
        "selection_rationale": out.selection_rationale,
        "best_candidate": out.best_candidate.knobs() if out.best_candidate else None,
        "best_achieved_titer": out.best_achieved_titer,
        "baseline_titer": out.baseline_titer,
        "improvement": out.improvement,
        "model_log": out.model_log,
    }


def _optimizer_box_and_train(bundle_dir: Path):
    """Shared loader: the search box (from config var_params) + seed batches
    (uploaded bundle, falling back to a configured CSV). Used by the closed-loop
    optimizer and equation discovery."""
    import json as _json

    import pandas as pd

    from fermdocs_optimize.schema import Box

    box_cfg = os.environ.get("FERMDOCS_OPTIMIZE_BOX")
    if not box_cfg:
        raise RuntimeError("FERMDOCS_OPTIMIZE_BOX must be set (config.json with var_params)")
    vp = _json.loads(Path(box_cfg).read_text()).get("var_params")
    box = Box(**{k: (vp[k]["lb"], vp[k]["ub"]) for k in
                 ("biomass", "total_sub", "malt_frac", "dilution")})
    train = _seed_training_from_bundle(bundle_dir)
    if train is None:
        train_csv = os.environ.get("FERMDOCS_OPTIMIZE_TRAIN")
        if not train_csv:
            raise RuntimeError("no seed batches and FERMDOCS_OPTIMIZE_TRAIN is unset")
        train = pd.read_csv(train_csv)[["batch", "t", "X", "S", "P", "M", "V"]]
    return box, train


def _append_setpoint_to_training(setpoint_df, knobs) -> dict | None:
    """Append the recommended setpoint's oracle trajectory to the training CSV as
    a NEW batch, so the next run learns from (recommended condition -> LABS titer).

    This is the active-learning accumulation the operator asked for: every run the
    agent proposes a next condition, the oracle says what titer it gives, and that
    real (condition, outcome) batch is folded into train.csv for next time. No-op
    if no persistent training CSV is configured (e.g. bundle-only runs)."""
    import pandas as pd

    train_csv = os.environ.get("FERMDOCS_OPTIMIZE_TRAIN")
    if not train_csv or not Path(train_csv).exists():
        return None
    existing = pd.read_csv(train_csv)
    ids = pd.to_numeric(existing["batch"], errors="coerce")
    next_id = int(ids.max()) + 1 if ids.notna().any() else 0
    new = setpoint_df.copy()
    new["batch"] = next_id
    for col in existing.columns:  # align to the train schema; fill gaps with NaN
        if col not in new.columns:
            new[col] = float("nan")
    new = new[existing.columns]
    pd.concat([existing, new], ignore_index=True).to_csv(train_csv, index=False)
    _log.info("appended recommended setpoint to %s as batch %d (peak %.2f g/L)",
              train_csv, next_id, float(setpoint_df["P"].max()))
    return {
        "path": train_csv, "batch_id": next_id, "rows": int(len(new)),
        "peak_titer": round(float(setpoint_df["P"].max()), 3),
        "total_batches": int(existing["batch"].nunique()) + 1,
        "knobs": {k: round(float(getattr(knobs, k)), 5) for k in
                  ("biomass", "total_sub", "malt_frac", "dilution")},
    }


def _run_equation_discovery(bundle_dir: Path) -> dict:
    """The discover -> scipy-search -> oracle-verify pattern, shaped for the UI.

    The agent (gemini by default; set FERMDOCS_OPTIMIZE_DISCOVERY_PROPOSER=template
    for the deterministic structural search) proposes the ODE structure, the
    oracle scores each structure, scipy.optimize searches the best equation, and
    the predicted setpoint is verified back on the oracle."""
    from fermdocs_optimize.discovery import CandidateModel, discover_model
    from fermdocs_optimize.discovery.proposers import LLMSpecProposer, TemplateProposer
    from fermdocs_optimize.oracle_search import oracle_global_search
    from fermdocs_optimize.schema import KNOB_NAMES
    from fermdocs_optimize.scipy_search import scipy_global_search
    from fermdocs_optimize.simulators.labs import LABSSimulator
    from fermdocs_optimize.simulators.model_backed import ModelSimulator

    mech = os.environ["FERMDOCS_OPTIMIZE_MECH_PARAMS"]
    gen_bin = os.environ.get("FERMDOCS_GENERATE_BATCHES_BIN", "generate-batches")
    box, train = _optimizer_box_and_train(bundle_dir)
    oracle = LABSSimulator(mech, generate_batches_bin=gen_bin)

    use_llm = os.environ.get("FERMDOCS_OPTIMIZE_DISCOVERY_PROPOSER", "llm").lower() == "llm"
    proposer = LLMSpecProposer() if use_llm else TemplateProposer()
    rounds = int(os.environ.get("FERMDOCS_OPTIMIZE_DISCOVERY_ROUNDS", "5"))
    probes = int(os.environ.get("FERMDOCS_OPTIMIZE_DISCOVERY_PROBES", "14"))

    # 1. discover + refine the equation against the oracle
    rep = discover_model(training_data=train, simulator=oracle, box=box,
                         proposer=proposer, max_rounds=rounds, n_probes=probes)
    if rep.best_spec is None:
        raise RuntimeError("discovery produced no compilable equation")

    # 2. scipy global search ON the discovered equation (cheap, vectorized)
    best_model = CandidateModel(rep.best_spec); best_model.fit(train)
    search = scipy_global_search(ModelSimulator(best_model), box, method="de",
                                 maxiter=30, popsize=15)
    c = search.best_candidate

    # 3. verify the predicted setpoint on the oracle + reference true max
    setpoint_df = oracle.simulate([c], v0=10.0)
    verified = float(setpoint_df["P"].max())
    try:
        ref = oracle_global_search(oracle, box, v0=10.0, n_lhs=120, refine_iters=6)
        true_max = ref.best_titer
        capture = round(100.0 * verified / true_max, 1) if true_max else None
    except Exception:  # noqa: BLE001
        true_max, capture = None, None

    # 4. ACTIVE LEARNING: fold the recommended (condition -> LABS titer) batch back
    # into the training data so the next run learns from it.
    appended = None
    if os.environ.get("FERMDOCS_OPTIMIZE_APPEND_RECOMMENDATIONS", "1") != "0":
        try:
            appended = _append_setpoint_to_training(setpoint_df, c)
        except Exception:  # noqa: BLE001 — accumulation must never sink the run
            _log.exception("appending recommended setpoint to training data failed")

    def _eqs(spec):
        return [f"{k} = {v}" for k, v in spec.aux.items()] + \
               [f"d{k}/dt = {v}" for k, v in spec.odes.items()]

    return {
        "proposer": "llm" if use_llm else "template",
        "rounds": [{
            "round_index": r.round_index, "name": r.spec.name,
            "oracle_peak_rmse": r.oracle_peak_rmse, "oracle_peak_r2": r.oracle_peak_r2,
            "compile_ok": r.compile_ok, "mu": r.spec.aux.get("mu", ""),
            "equations": _eqs(r.spec),
            "fitted_params": {k: round(v, 5) for k, v in r.fitted_params.items()},
            "notes": r.spec.notes or "",
            "error": r.error or "",
        } for r in rep.rounds],
        "best_name": rep.best_spec.name,
        "best_equations": _eqs(rep.best_spec),
        "best_fitted_params": {k: round(v, 5) for k, v in best_model.fitted_params.items()},
        "best_oracle_peak_rmse": rep.oracle_peak_rmse,
        "best_oracle_peak_r2": rep.oracle_peak_r2,
        "search_method": "scipy differential_evolution (vectorized)",
        "search_evals": search.n_oracle_evals,
        "predicted_optimum_titer": search.best_titer,
        "predicted_knobs": {k: round(getattr(c, k), 5) for k in KNOB_NAMES},
        "oracle_verified_titer": round(verified, 3),
        "oracle_true_max": true_max,
        "capture_pct": capture,
        "knobs_on_boundary": search.knobs_on_boundary,
        "appended_to_training": appended,
    }


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
