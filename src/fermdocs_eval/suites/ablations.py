"""E3a — Ablation studies: full system vs component-removed variants.

For each (question, config) pair:
  1. Run the pipeline with the config's overrides applied.
  2. Record structured output: citation counts, finding-class coverage,
     n_final_hypotheses, n_critiques, turn_count, exit_reason.

Configs:
  - full:        no overrides (reference column)
  - no_critic:   skip_critic=True — synthesizer draft becomes final
  - single_spec: specialist_order=("kinetics",) only
  - no_memory:   NoopBackend (cold run, no cross-run lessons)
  - baseline:    single-shot Gemini call (same as old E3)

Trial IDs: "q{N}-{config_name}" e.g. "q3-full", "q3-no_critic", "q3-baseline"

Resumability: completed trials are skipped on re-run. Errors are retried.
"""

from __future__ import annotations

import json
import os
import traceback
import uuid
from pathlib import Path

from fermdocs_eval.ablation_configs import (
    ABLATION_QUESTIONS,
    ALL_CONFIGS,
    BASELINE,
    AblationConfig,
    make_budget,
)
from fermdocs_eval.harness import EvalRun, RunStatus, append_jsonl, now_iso, read_jsonl

EVAL_TENANT_ID = "eval-ablation"
BASELINE_MODEL = "gemini-3.1-pro-preview"


def _completed(out_path: Path) -> set[str]:
    return {r["trial_id"] for r in read_jsonl(out_path) if r.get("status") == "ok"}


def _set_env() -> dict[str, str]:
    overrides = {
        "FERMDOCS_HYPOTHESIS_PROVIDER": "gemini",
        "FERMDOCS_GEMINI_MODEL": BASELINE_MODEL,
        "FERMDOCS_HYPOTHESIS_MODEL": BASELINE_MODEL,
        "FERMDOCS_TENANT_ID": EVAL_TENANT_ID,
    }
    prior = {k: os.environ.get(k, "") for k in overrides}
    os.environ.update(overrides)
    return prior


def _restore_env(prior: dict[str, str]) -> None:
    for k, v in prior.items():
        if v:
            os.environ[k] = v
        else:
            os.environ.pop(k, None)


def _run_pipeline(
    bundle_dir: Path,
    question: str,
    run_dir: Path,
    config: AblationConfig,
) -> dict:
    from fermdocs.domain.user_question import UserQuestion
    from fermdocs_hypothesis.bundle_loader import load_bundle
    from fermdocs_hypothesis.live_hooks import LiveHooks
    from fermdocs_hypothesis.runner import run_stage
    from fermdocs_hypothesis.schema import BudgetSnapshot
    from fermdocs_memory.stub import StubBackend

    uq = UserQuestion(text=question)
    (bundle_dir / "user_question.json").write_text(
        json.dumps(uq.model_dump(mode="json"), indent=2)
    )

    loaded = load_bundle(bundle_dir)
    hyp_id = uuid.uuid4()
    out_dir = run_dir / str(hyp_id)
    out_dir.mkdir(parents=True, exist_ok=True)
    global_md = out_dir / "global.md"

    budget = make_budget(config)

    if config.use_memory:
        backend = StubBackend()
    else:
        from fermdocs_memory.noop import NoopBackend
        backend = NoopBackend()

    hooks = LiveHooks(loaded, memory=backend)

    result = run_stage(
        hyp_input=loaded.hyp_input,
        hooks=hooks,
        global_md_path=global_md,
        diagnosis_id=loaded.diagnosis.meta.diagnosis_id,
        provider="gemini",
        model_name=hooks._client.model_name,
        budget=budget,
        memory=backend,
        validate=False,
        specialist_order=config.specialist_order,
    )

    hyps = list(getattr(result.output, "final_hypotheses", []) or [])
    final_text = "\n\n".join(getattr(h, "summary", "") for h in hyps) if hyps else ""

    total_cited_findings = set()
    total_cited_narratives = set()
    finding_classes: set[str] = set()
    for h in hyps:
        total_cited_findings.update(h.cited_finding_ids)
        total_cited_narratives.update(h.cited_narrative_ids)
        for fid in h.cited_finding_ids:
            prefix = fid.rsplit("-", 1)[0] if "-" in fid else fid
            finding_classes.add(prefix)

    n_critiques = sum(
        1 for e in result.events if getattr(e, "type", None) == "critique_filed"
    )

    return {
        "hyp_id": str(hyp_id),
        "run_dir": str(out_dir),
        "config": config.name,
        "n_final_hypotheses": len(hyps),
        "n_critiques": n_critiques,
        "n_cited_findings": len(total_cited_findings),
        "n_cited_narratives": len(total_cited_narratives),
        "cited_finding_ids": sorted(total_cited_findings),
        "cited_narrative_ids": sorted(total_cited_narratives),
        "finding_class_count": len(finding_classes),
        "finding_classes": sorted(finding_classes),
        "exit_reason": getattr(result.state, "exit_reason", None),
        "turns_used": result.state.budget.turns_used,
        "final_text": final_text,
    }


def _build_baseline_prompt(bundle_dir: Path, question: str) -> str:
    parts = [
        "You are an expert fermentation analyst. Read the bundle below and"
        " answer the user's question. Be specific and cite findings by ID"
        " where possible."
    ]
    for label, rel in [
        ("DOSSIER", "dossier.json"),
        ("CHARACTERIZATION", "characterization/characterization.json"),
        ("DIAGNOSIS", "diagnosis/diagnosis.json"),
    ]:
        p = bundle_dir / rel
        if not p.exists():
            continue
        try:
            content = p.read_text()
        except OSError:
            continue
        if len(content) > 30_000:
            content = content[:30_000] + "\n... [truncated for context window]"
        parts.append(f"### {label}\n```json\n{content}\n```")
    parts.append(f"### QUESTION\n{question}")
    return "\n\n".join(parts)


def _baseline_call(prompt: str) -> str:
    from google import genai  # type: ignore

    client = genai.Client(api_key=os.environ["GEMINI_API_KEY"])
    response = client.models.generate_content(model=BASELINE_MODEL, contents=prompt)
    return response.text or ""


def run(
    *,
    bundle_dir: str,
    questions_path: str = "eval/questions.json",
    out_path: str = "eval/results/ablations.jsonl",
    runs_root: str = "eval/runs/ablations",
    resume: bool = True,
    questions: tuple[str, ...] | None = None,
    configs: tuple[AblationConfig, ...] | None = None,
) -> None:
    """Run the ablation matrix.

    questions: subset of question IDs to run (default: ABLATION_QUESTIONS).
    configs: subset of configs to run (default: ALL_CONFIGS).
    """
    bundle = Path(bundle_dir)
    out = Path(out_path)
    runs = Path(runs_root)
    qpath = Path(questions_path)
    if not qpath.exists():
        raise FileNotFoundError(f"questions file not found: {qpath}")

    all_questions: dict[str, str] = json.loads(qpath.read_text())
    qids = questions or ABLATION_QUESTIONS
    cfgs = configs or ALL_CONFIGS
    skip = _completed(out) if resume else set()
    prior_env = _set_env()

    n_trials = len(qids) * len(cfgs)
    print(f"[ablation] {len(qids)} questions × {len(cfgs)} configs = {n_trials} trials, {len(skip)} pre-completed")

    try:
        for qid in qids:
            qtext = all_questions[qid]
            for config in cfgs:
                trial_id = f"{qid}-{config.name}"
                if trial_id in skip:
                    print(f"[ablation] {trial_id} skip")
                    continue

                print(f"[ablation] {trial_id} running...")
                started = now_iso()

                if config.name == "baseline":
                    try:
                        prompt = _build_baseline_prompt(bundle, qtext)
                        baseline_text = _baseline_call(prompt)
                        row = EvalRun(
                            suite="ablation", trial_id=trial_id, status=RunStatus.OK,
                            started_at=started, finished_at=now_iso(),
                            payload={
                                "qid": qid, "question": qtext, "config": "baseline",
                                "model": BASELINE_MODEL,
                                "final_text": baseline_text,
                                "n_final_hypotheses": 0,
                                "n_critiques": 0,
                                "n_cited_findings": 0,
                                "finding_class_count": 0,
                                "exit_reason": "single_shot",
                                "turns_used": 0,
                            },
                        )
                    except Exception as exc:
                        row = EvalRun(
                            suite="ablation", trial_id=trial_id, status=RunStatus.ERROR,
                            started_at=started, finished_at=now_iso(),
                            payload={"config": "baseline", "qid": qid,
                                     "traceback": traceback.format_exc()},
                            error=f"{type(exc).__name__}: {exc}",
                        )
                else:
                    try:
                        payload = _run_pipeline(bundle, qtext, runs / qid / config.name, config)
                        payload["question"] = qtext
                        payload["qid"] = qid
                        row = EvalRun(
                            suite="ablation", trial_id=trial_id, status=RunStatus.OK,
                            started_at=started, finished_at=now_iso(), payload=payload,
                        )
                    except Exception as exc:
                        row = EvalRun(
                            suite="ablation", trial_id=trial_id, status=RunStatus.ERROR,
                            started_at=started, finished_at=now_iso(),
                            payload={"config": config.name, "qid": qid, "question": qtext,
                                     "traceback": traceback.format_exc()},
                            error=f"{type(exc).__name__}: {exc}",
                        )

                append_jsonl(out, row)
                status = row.status.value
                ft_len = len(row.payload.get("final_text", ""))
                print(f"[ablation]   -> {status} final_len={ft_len}")
    finally:
        _restore_env(prior_env)
    print(f"[ablation] done. results: {out}")
