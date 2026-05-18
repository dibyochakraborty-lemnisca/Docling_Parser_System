"""E3 suite — case studies vs single-shot Gemini baseline.

For each bundle (yeast, indpensim):
  Treatment: full fermdocs pipeline (load_bundle + LiveHooks + run_stage).
             Output = final ratified hypothesis text.
  Baseline:  one Gemini call (gemini-3.1-pro-preview) given the raw
             bundle contents + the same user question dumped into a
             single prompt. No tool use, no agents, no critique.

Both outputs are then blind-judged by a separate Gemini call (preference
judge) for specificity / grounding / actionability. Order is
counterbalanced (A=treatment for half the seeds, B=treatment for the
other half) to control for position bias.

This is a case-study eval at N=2. We do NOT compute an aggregate
preference rate; instead we report per-bundle which output won and
across-seed agreement.
"""

from __future__ import annotations

import json
import os
import uuid
from pathlib import Path

from fermdocs_eval.harness import EvalRun, RunStatus, append_jsonl, now_iso, read_jsonl

EVAL_TENANT_ID = "eval-e3"
BASELINE_MODEL = "gemini-3.1-pro-preview"
JUDGE_SEEDS = ["s0", "s1", "s2"]  # 3-seed estimate of judge variance


def _set_e3_env() -> dict[str, str]:
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


def _build_baseline_prompt(bundle_dir: Path, question: str) -> str:
    """Assemble a single-shot prompt with bundle context + user question.

    Strategy: dossier summary + characterization findings + diagnosis
    failures, capped at ~80k chars to stay well under model context
    limits. The synthesizer in the real pipeline sees structured views;
    here the baseline gets the raw JSON it would have to reason over
    itself.
    """
    parts = ["You are an expert fermentation analyst. Read the bundle below and answer the user's question."]

    dossier_p = bundle_dir / "dossier.json"
    char_p = bundle_dir / "characterization" / "characterization.json"
    diag_p = bundle_dir / "diagnosis" / "diagnosis.json"

    for label, path in [("DOSSIER", dossier_p), ("CHARACTERIZATION", char_p), ("DIAGNOSIS", diag_p)]:
        if path.exists():
            try:
                content = path.read_text()
                if len(content) > 30000:
                    content = content[:30000] + "\n... [truncated for context window]"
                parts.append(f"### {label}\n```json\n{content}\n```")
            except OSError:
                pass

    parts.append(f"### USER QUESTION\n{question}")
    parts.append(
        "Produce a single hypothesis: a concrete causal claim that names"
        " specific findings, time windows, and runs. End with one"
        " actionable recommendation for the next experiment."
    )
    return "\n\n".join(parts)


def _baseline_call(prompt: str) -> str:
    """Single-shot Gemini call. Returns text or raises."""
    from google import genai  # type: ignore

    client = genai.Client(api_key=os.environ["GEMINI_API_KEY"])
    response = client.models.generate_content(model=BASELINE_MODEL, contents=prompt)
    return response.text or ""


def _run_treatment(bundle_dir: Path, question: str, run_dir: Path) -> dict:
    """Full pipeline run. Returns payload with final_text."""
    import uuid as _uuid

    from fermdocs.domain.user_question import UserQuestion
    from fermdocs_hypothesis.bundle_loader import load_bundle
    from fermdocs_hypothesis.live_hooks import LiveHooks
    from fermdocs_hypothesis.runner import run_stage
    from fermdocs_hypothesis.schema import BudgetSnapshot
    from fermdocs_memory.stub import StubBackend

    uq_path = bundle_dir / "user_question.json"
    if not uq_path.exists():
        uq = UserQuestion(text=question)
        uq_path.write_text(json.dumps(uq.model_dump(mode="json"), indent=2))

    loaded = load_bundle(bundle_dir)
    hyp_id = _uuid.uuid4()
    out_dir = run_dir / str(hyp_id)
    out_dir.mkdir(parents=True, exist_ok=True)
    global_md = out_dir / "global.md"

    budget = BudgetSnapshot(
        max_turns=10, max_critic_cycles_per_topic=3,
        max_tool_calls_total=80, max_total_input_tokens=200_000,
    )
    backend = StubBackend()
    hooks = LiveHooks(loaded, memory=backend)
    result = run_stage(
        hyp_input=loaded.hyp_input, hooks=hooks, global_md_path=global_md,
        diagnosis_id=loaded.diagnosis.meta.diagnosis_id,
        provider="gemini", model_name=hooks._client.model_name,
        budget=budget, memory=backend, validate=False,
    )

    hyps = list(getattr(result.output, "final_hypotheses", []) or [])
    final_text = "\n\n".join(getattr(h, "summary", "") for h in hyps) if hyps else ""

    return {
        "hyp_id": str(hyp_id),
        "run_dir": str(out_dir),
        "n_final_hypotheses": len(hyps),
        "final_text": final_text,
    }


def _completed_trials(out_path: Path) -> set[str]:
    return {r["trial_id"] for r in read_jsonl(out_path) if r.get("status") == "ok"}


def run(
    *,
    bundle_dir: str,
    question: str,
    out_path: str = "eval/results/e3.jsonl",
    runs_root: str = "eval/runs/e3",
    resume: bool = True,
) -> None:
    """Run treatment + baseline + 3-seed judge preference for one bundle."""
    from fermdocs_eval.judges import judge_preference

    bundle = Path(bundle_dir)
    bundle_name = bundle.name
    out = Path(out_path)
    runs = Path(runs_root)

    skip = _completed_trials(out) if resume else set()
    prior_env = _set_e3_env()

    try:
        # 1. Treatment (full pipeline)
        trial_t = f"{bundle_name}-treatment"
        treatment_text = ""
        if trial_t not in skip:
            print(f"[e3] treatment: {bundle_name}")
            started = now_iso()
            try:
                payload = _run_treatment(bundle, question, runs / bundle_name / "treatment")
                treatment_text = payload["final_text"]
                row = EvalRun(
                    suite="e3", trial_id=trial_t, status=RunStatus.OK,
                    started_at=started, finished_at=now_iso(), payload=payload,
                )
            except Exception as exc:  # noqa: BLE001
                row = EvalRun(
                    suite="e3", trial_id=trial_t, status=RunStatus.ERROR,
                    started_at=started, finished_at=now_iso(),
                    payload={"phase": "treatment", "bundle": bundle_name},
                    error=f"{type(exc).__name__}: {exc}",
                )
            append_jsonl(out, row)
            print(f"[e3]   -> {row.status.value}")
        else:
            # We need the text for the judge step; re-read from prior row.
            for r in read_jsonl(out):
                if r["trial_id"] == trial_t:
                    treatment_text = (r.get("payload") or {}).get("final_text", "")
                    break

        # 2. Baseline (single-shot)
        trial_b = f"{bundle_name}-baseline"
        baseline_text = ""
        if trial_b not in skip:
            print(f"[e3] baseline: {bundle_name}")
            started = now_iso()
            try:
                prompt = _build_baseline_prompt(bundle, question)
                baseline_text = _baseline_call(prompt)
                row = EvalRun(
                    suite="e3", trial_id=trial_b, status=RunStatus.OK,
                    started_at=started, finished_at=now_iso(),
                    payload={"final_text": baseline_text, "model": BASELINE_MODEL},
                )
            except Exception as exc:  # noqa: BLE001
                row = EvalRun(
                    suite="e3", trial_id=trial_b, status=RunStatus.ERROR,
                    started_at=started, finished_at=now_iso(),
                    payload={"phase": "baseline", "bundle": bundle_name},
                    error=f"{type(exc).__name__}: {exc}",
                )
            append_jsonl(out, row)
            print(f"[e3]   -> {row.status.value}")
        else:
            for r in read_jsonl(out):
                if r["trial_id"] == trial_b:
                    baseline_text = (r.get("payload") or {}).get("final_text", "")
                    break

        # 3. Judge preference, 3 seeds, counterbalanced order
        if treatment_text and baseline_text:
            for i, seed in enumerate(JUDGE_SEEDS):
                # Counterbalance: even seeds → treatment=A; odd → treatment=B
                trial_j = f"{bundle_name}-judge-{seed}"
                if trial_j in skip:
                    continue
                treatment_first = (i % 2 == 0)
                a_text = treatment_text if treatment_first else baseline_text
                b_text = baseline_text if treatment_first else treatment_text
                print(f"[e3] judge {seed} (treatment={'A' if treatment_first else 'B'}): {bundle_name}")
                started = now_iso()
                verdict = judge_preference(a_text, b_text, seed_label=seed)
                # Normalize: rewrite winner relative to TREATMENT, not A/B order.
                if verdict.get("status") == "ok":
                    winner_axis = verdict["winner"]
                    if winner_axis == "tie":
                        treatment_won = "tie"
                    elif treatment_first:
                        treatment_won = "treatment" if winner_axis == "A" else "baseline"
                    else:
                        treatment_won = "treatment" if winner_axis == "B" else "baseline"
                    row = EvalRun(
                        suite="e3", trial_id=trial_j, status=RunStatus.OK,
                        started_at=started, finished_at=now_iso(),
                        payload={
                            "bundle": bundle_name, "seed": seed,
                            "treatment_position": "A" if treatment_first else "B",
                            "raw_winner": winner_axis,
                            "treatment_won": treatment_won,
                            "rationale": verdict.get("rationale", ""),
                            "axes": verdict.get("axes", {}),
                        },
                    )
                else:
                    row = EvalRun(
                        suite="e3", trial_id=trial_j, status=RunStatus.ERROR,
                        started_at=started, finished_at=now_iso(),
                        payload={"seed": seed, "bundle": bundle_name},
                        error=verdict.get("error", "judge_unknown_error"),
                    )
                append_jsonl(out, row)
                print(f"[e3]   -> {row.status.value} treatment_won={row.payload.get('treatment_won', '?')}")
    finally:
        _restore_env(prior_env)
    print(f"[e3] done: {bundle_name}")
