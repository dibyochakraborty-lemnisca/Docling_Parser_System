"""E2 suite — critic axes precision/recall.

For each fixture in eval/fixtures/e2/specs.py:
  1. Build the fixture bundle from its DefectSpec (idempotent — skip if exists)
  2. Optionally classify the leading_question against the bundle
  3. Run the full hypothesis stage
  4. Extract fired axes from CritiqueFiledEvent.reasons across all critiques
  5. Append one EvalRun row to eval/results/e2.jsonl

Resumability: trial_id is the fixture_id. On startup the runner reads the
existing JSONL and skips any fixture already present with status=ok. Errors
are also recorded as rows but NOT skipped on retry — re-running picks them
up again.

Model config: all-pro (gemini-3-pro). LiveHooks shares one client across
the hypothesis stage, so per-agent overrides aren't possible without a
refactor. The runner sets FERMDOCS_HYPOTHESIS_MODEL=gemini-3-pro and
restores prior env on exit. This decision is disclosed in the paper.
"""

from __future__ import annotations

import os
import re
import uuid
from datetime import datetime, timezone
from pathlib import Path

from fermdocs_eval.fixture_builder import DefectSpec, build_fixture
from fermdocs_eval.harness import EvalRun, RunStatus, append_jsonl, now_iso, read_jsonl

# Tenant scope used for memory-axis fixtures. Matches FERMDOCS_TENANT_ID
# default. The eval runs in a hermetic StubBackend instance per fixture so
# no production memory is touched.
EVAL_TENANT_ID = "eval-e2"

# Critic prefixes its rejection reasons with "[axis-name]:". We scan all
# reasons across every CritiqueFiledEvent for the fixture's run.
AXIS_RE = re.compile(r"\[([a-z-]+-axis)\]", re.IGNORECASE)


def extract_fired_axes(events: list) -> list[str]:
    """Scan all CritiqueFiledEvent.reasons for [axis-name] prefixes.

    Returns deduplicated, lowercased axis names in first-seen order.
    """
    seen: list[str] = []
    for ev in events:
        # Event objects are pydantic models; CritiqueFiledEvent has .type and .reasons.
        ev_type = getattr(ev, "type", None) or (
            ev.get("type") if isinstance(ev, dict) else None
        )
        if ev_type != "critique_filed":
            continue
        reasons = getattr(ev, "reasons", None)
        if reasons is None and isinstance(ev, dict):
            reasons = ev.get("reasons") or []
        for r in reasons or []:
            for axis in AXIS_RE.findall(str(r)):
                axis_l = axis.lower()
                if axis_l not in seen:
                    seen.append(axis_l)
    return seen


def _completed_trial_ids(out_path: Path) -> set[str]:
    """Read existing JSONL and return trial_ids that already completed OK."""
    rows = read_jsonl(out_path)
    return {r["trial_id"] for r in rows if r.get("status") == "ok"}


def _make_memory_backend(spec: DefectSpec):
    """Build a hermetic StubBackend pre-populated with spec.memory_seed.

    Each seed becomes a MemoryRecord with kind='lesson', tenant_id=EVAL_TENANT_ID,
    and the spec's process_family. The lesson_id is deterministic per fixture
    so the same fixture re-run produces the same record IDs (helpful when
    diffing JSONL outputs across runs).

    For non-memory-axis fixtures (empty memory_seed), returns an empty
    StubBackend so the pipeline still has a valid backend but retrieves
    nothing.
    """
    from fermdocs_memory.base import MemoryRecord
    from fermdocs_memory.stub import StubBackend

    backend = StubBackend()
    for i, seed in enumerate(spec.memory_seed):
        family, summary = seed[0], seed[1]
        record = MemoryRecord(
            memory_id=f"L-eval-{spec.fixture_id}-{i:04d}",
            kind="lesson",
            summary=summary,
            process_family=family,
            organism=None,
            tenant_id=EVAL_TENANT_ID,
            provenance={"source": "e2_eval_seed", "fixture_id": spec.fixture_id},
            embedding_provider="stub",
            embedding_model="stub",
            embedding_version="1",
        )
        backend.write(record)
    return backend


def _set_e2_env() -> dict[str, str]:
    """Set env vars for the E2 pipeline (all-pro). Returns prior values for cleanup.

    LiveHooks today shares one client across the whole hypothesis stage, so
    per-agent overrides aren't possible without a refactor. We run all agents
    on gemini-3-pro for fidelity to the deployed config. This decision is
    disclosed in the paper.
    """
    overrides = {
        "FERMDOCS_HYPOTHESIS_PROVIDER": "gemini",
        # GeminiHypothesisClient reads FERMDOCS_GEMINI_MODEL (not the
        # FERMDOCS_HYPOTHESIS_MODEL we set earlier — that env var is for a
        # different code path). Dry run on 2026-05-18 surfaced this:
        # gemini-3-pro returns 404, the actual available model is the preview.
        "FERMDOCS_GEMINI_MODEL": "gemini-3.1-pro-preview",
        "FERMDOCS_HYPOTHESIS_MODEL": "gemini-3.1-pro-preview",
        # Memory-axis fixtures use a hermetic StubBackend under this tenant.
        "FERMDOCS_TENANT_ID": EVAL_TENANT_ID,
    }
    prior: dict[str, str] = {}
    for k, v in overrides.items():
        prior[k] = os.environ.get(k, "")
        os.environ[k] = v
    return prior


def _restore_env(prior: dict[str, str]) -> None:
    for k, v in prior.items():
        if v:
            os.environ[k] = v
        else:
            os.environ.pop(k, None)


def _run_one_fixture(
    spec: DefectSpec,
    *,
    fixtures_root: Path,
    template_dir: Path,
    runs_root: Path,
) -> EvalRun:
    """Build and run a single fixture. Always returns an EvalRun row."""
    started = now_iso()
    try:
        bundle_dir = build_fixture(spec, template_dir=template_dir, out_root=fixtures_root)
    except Exception as exc:  # noqa: BLE001
        return EvalRun(
            suite="e2",
            trial_id=spec.fixture_id,
            status=RunStatus.ERROR,
            started_at=started,
            finished_at=now_iso(),
            payload={"phase": "build", "labeled_axis": spec.labeled_axis},
            error=f"{type(exc).__name__}: {exc}",
        )

    # Lazy imports — keep eval module light when not invoking pipeline.
    from fermdocs_hypothesis.bundle_loader import load_bundle
    from fermdocs_hypothesis.live_hooks import LiveHooks
    from fermdocs_hypothesis.runner import run_stage
    from fermdocs_hypothesis.schema import BudgetSnapshot
    from fermdocs_memory.stub import StubBackend

    try:
        loaded = load_bundle(bundle_dir)
        hyp_id = uuid.uuid4()
        run_dir = runs_root / spec.fixture_id / str(hyp_id)
        run_dir.mkdir(parents=True, exist_ok=True)
        global_md = run_dir / "global.md"

        # Per-fixture hermetic memory backend. Pre-populated from
        # spec.memory_seed for memory-axis fixtures; empty otherwise.
        memory_backend = _make_memory_backend(spec)

        budget = BudgetSnapshot(
            max_turns=10,
            max_critic_cycles_per_topic=3,
            max_tool_calls_total=80,
            max_total_input_tokens=200_000,
        )
        hooks = LiveHooks(loaded, memory=memory_backend)
        diagnosis_id = loaded.diagnosis.meta.diagnosis_id

        result = run_stage(
            hyp_input=loaded.hyp_input,
            hooks=hooks,
            global_md_path=global_md,
            diagnosis_id=diagnosis_id,
            provider="gemini",
            model_name=hooks._client.model_name,
            budget=budget,
            memory=memory_backend,
            validate=False,  # eval bundles intentionally play with structure
            now_factory=lambda: datetime.now(timezone.utc),
        )

        fired = extract_fired_axes(result.events)

        # Count distinct hypotheses and critiques for context in the report.
        n_critiques = sum(
            1 for e in result.events if getattr(e, "type", None) == "critique_filed"
        )
        n_hypotheses = len(getattr(result.output, "final_hypotheses", []) or [])

        return EvalRun(
            suite="e2",
            trial_id=spec.fixture_id,
            status=RunStatus.OK,
            started_at=started,
            finished_at=now_iso(),
            payload={
                "labeled_axis": spec.labeled_axis,
                "difficulty": spec.difficulty,
                "fired_axes": fired,
                "n_critiques_filed": n_critiques,
                "n_final_hypotheses": n_hypotheses,
                "exit_reason": getattr(result.state, "exit_reason", None),
                "hyp_id": str(hyp_id),
                "run_dir": str(run_dir),
                "leading_question": spec.leading_question,
            },
        )
    except Exception as exc:  # noqa: BLE001
        return EvalRun(
            suite="e2",
            trial_id=spec.fixture_id,
            status=RunStatus.ERROR,
            started_at=started,
            finished_at=now_iso(),
            payload={
                "phase": "pipeline",
                "labeled_axis": spec.labeled_axis,
                "difficulty": spec.difficulty,
            },
            error=f"{type(exc).__name__}: {exc}",
        )


def run(
    *,
    out_path: str = "eval/results/e2.jsonl",
    fixtures_root: str = "eval/fixtures/e2",
    template_dir: str = "out/bundle_indpensim",
    runs_root: str = "eval/runs/e2",
    specs: list[DefectSpec] | None = None,
    resume: bool = True,
) -> None:
    """Run the E2 batch. `specs` defaults to fermdocs_eval.fixtures.e2_specs.SPECS."""
    out = Path(out_path)
    fixtures_p = Path(fixtures_root)
    runs_p = Path(runs_root)

    if specs is None:
        # Authored spec list — created in E2d.
        try:
            from fermdocs_eval.fixtures.e2_specs import SPECS  # type: ignore

            specs = SPECS
        except ImportError:
            print("[e2] no specs registered yet (fermdocs_eval/fixtures/e2_specs.py)")
            return

    if not specs:
        print("[e2] empty spec list; nothing to run.")
        return

    skip = _completed_trial_ids(out) if resume else set()
    prior_env = _set_e2_env()

    n = len(specs)
    print(f"[e2] running {n} fixtures (skipping {len(skip)} already-complete)")
    try:
        for i, spec in enumerate(specs, 1):
            if spec.fixture_id in skip:
                print(f"[e2] {i}/{n} {spec.fixture_id} — skip (already ok)")
                continue
            print(f"[e2] {i}/{n} {spec.fixture_id} ({spec.labeled_axis}, {spec.difficulty})")
            row = _run_one_fixture(
                spec,
                fixtures_root=fixtures_p,
                template_dir=Path(template_dir),
                runs_root=runs_p,
            )
            append_jsonl(out, row)
            status_str = row.status.value
            extra = ""
            if row.status == RunStatus.OK:
                extra = f" fired={row.payload.get('fired_axes', [])}"
            elif row.error:
                extra = f" err={row.error[:80]}"
            print(f"[e2]   -> {status_str}{extra}")
    finally:
        _restore_env(prior_env)
    print(f"[e2] done. results: {out}")
