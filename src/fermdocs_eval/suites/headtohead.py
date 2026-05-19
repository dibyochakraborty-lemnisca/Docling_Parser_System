"""Head-to-head suite — fermdocs agent vs single-shot Gemini baseline.

For each question on a given bundle:
  1. Treatment: full fermdocs hypothesis pipeline (load_bundle + LiveHooks
     + run_stage). Output = concatenated final hypothesis summaries.
  2. Baseline: one Gemini 3.1 Pro call with the bundle JSON dumped in +
     the same question. No tools, no agents, no critique.
  3. Judge: judge_head_to_head with 3 seeds, counterbalanced order
     (even seeds -> treatment=A, odd -> treatment=B). The judge scores
     each answer 1-10 on specificity, grounding, actionability, honesty,
     and picks a winner. Winner is normalized to "treatment/baseline/tie".

Trial IDs (stable across resumes):
  - "q{N}-treatment": one row per question
  - "q{N}-baseline": one row per question
  - "q{N}-judge-s{S}": one row per judge seed

Resumability: a question's treatment/baseline/judge rows are written
once and re-runs skip them. Errors don't get skipped — re-firing retries
them. Final JSONL has up to 10 * (1+1+3) = 50 rows.
"""

from __future__ import annotations

import json
import os
import traceback
import uuid
from pathlib import Path

from fermdocs_eval.harness import EvalRun, RunStatus, append_jsonl, now_iso, read_jsonl

EVAL_TENANT_ID = "eval-headtohead"
BASELINE_MODEL = "gemini-3.1-pro-preview"
JUDGE_SEEDS = ("s0", "s1", "s2")
QUESTION_COUNT = 10  # number of questions used for stable trial-id formatting


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


def _completed(out_path: Path) -> set[str]:
    return {r["trial_id"] for r in read_jsonl(out_path) if r.get("status") == "ok"}


def _build_baseline_prompt(bundle_dir: Path, question: str) -> str:
    """Assemble a single-shot prompt with bundle context + the question.

    Strategy: dossier + characterization + diagnosis JSON dumped inline,
    capped per file so the total prompt stays well under context limits.
    """
    parts = [
        "You are an expert fermentation analyst. Read the bundle below and"
        " answer the user's question. Be specific, cite findings by ID where"
        " possible, and end with concrete next steps when the question admits"
        " a recommendation."
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


def _run_treatment(bundle_dir: Path, question: str, run_dir: Path) -> dict:
    """Full pipeline run on the bundle with the given question.

    The bundle's user_question.json is overwritten per-question so each
    run sees the right user question. Returns payload with final_text.
    """
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

    budget = BudgetSnapshot(
        max_turns=10,
        max_critic_cycles_per_topic=3,
        max_tool_calls_total=80,
        max_total_input_tokens=200_000,
    )
    backend = StubBackend()  # hermetic, no memory carried across questions
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
    )

    hyps = list(getattr(result.output, "final_hypotheses", []) or [])
    final_text = "\n\n".join(getattr(h, "summary", "") for h in hyps) if hyps else ""
    n_critiques = sum(
        1 for e in result.events if getattr(e, "type", None) == "critique_filed"
    )
    return {
        "hyp_id": str(hyp_id),
        "run_dir": str(out_dir),
        "n_final_hypotheses": len(hyps),
        "n_critiques": n_critiques,
        "exit_reason": getattr(result.state, "exit_reason", None),
        "final_text": final_text,
    }


def _treatment_text_from_jsonl(out: Path, trial_id: str) -> str:
    for r in read_jsonl(out):
        if r["trial_id"] == trial_id and r.get("status") == "ok":
            return (r.get("payload") or {}).get("final_text", "")
    return ""


def _baseline_text_from_jsonl(out: Path, trial_id: str) -> str:
    return _treatment_text_from_jsonl(out, trial_id)


def run(
    *,
    bundle_dir: str,
    questions_path: str = "eval/questions.json",
    out_path: str = "eval/results/headtohead.jsonl",
    runs_root: str = "eval/runs/headtohead",
    resume: bool = True,
) -> None:
    """Run the full head-to-head batch.

    questions_path: JSON file mapping {"q1": "text...", "q2": "...", ...}.
    Authored separately at eval/questions.json (see eval/questions.md).
    """
    from fermdocs_eval.judges import judge_head_to_head

    bundle = Path(bundle_dir)
    out = Path(out_path)
    runs = Path(runs_root)
    qpath = Path(questions_path)
    if not qpath.exists():
        raise FileNotFoundError(f"questions file not found: {qpath}")

    questions: dict[str, str] = json.loads(qpath.read_text())
    skip = _completed(out) if resume else set()
    prior_env = _set_env()

    print(f"[h2h] {len(questions)} questions, {len(skip)} pre-completed trials")

    try:
        for qid, qtext in questions.items():
            # 1. Treatment
            trial_t = f"{qid}-treatment"
            treatment_text = ""
            if trial_t in skip:
                treatment_text = _treatment_text_from_jsonl(out, trial_t)
                print(f"[h2h] {trial_t} skip")
            else:
                print(f"[h2h] {trial_t} running...")
                started = now_iso()
                try:
                    payload = _run_treatment(bundle, qtext, runs / qid / "treatment")
                    payload["question"] = qtext
                    treatment_text = payload["final_text"]
                    row = EvalRun(
                        suite="headtohead", trial_id=trial_t, status=RunStatus.OK,
                        started_at=started, finished_at=now_iso(), payload=payload,
                    )
                except Exception as exc:  # noqa: BLE001
                    row = EvalRun(
                        suite="headtohead", trial_id=trial_t, status=RunStatus.ERROR,
                        started_at=started, finished_at=now_iso(),
                        payload={"phase": "treatment", "qid": qid, "question": qtext,
                                  "traceback": traceback.format_exc()},
                        error=f"{type(exc).__name__}: {exc}",
                    )
                append_jsonl(out, row)
                print(f"[h2h]   -> {row.status.value}"
                      f" final_len={len(treatment_text)}")

            # 2. Baseline
            trial_b = f"{qid}-baseline"
            baseline_text = ""
            if trial_b in skip:
                baseline_text = _baseline_text_from_jsonl(out, trial_b)
                print(f"[h2h] {trial_b} skip")
            else:
                print(f"[h2h] {trial_b} running...")
                started = now_iso()
                try:
                    prompt = _build_baseline_prompt(bundle, qtext)
                    baseline_text = _baseline_call(prompt)
                    row = EvalRun(
                        suite="headtohead", trial_id=trial_b, status=RunStatus.OK,
                        started_at=started, finished_at=now_iso(),
                        payload={
                            "qid": qid, "question": qtext,
                            "model": BASELINE_MODEL,
                            "final_text": baseline_text,
                        },
                    )
                except Exception as exc:  # noqa: BLE001
                    row = EvalRun(
                        suite="headtohead", trial_id=trial_b, status=RunStatus.ERROR,
                        started_at=started, finished_at=now_iso(),
                        payload={"phase": "baseline", "qid": qid,
                                  "traceback": traceback.format_exc()},
                        error=f"{type(exc).__name__}: {exc}",
                    )
                append_jsonl(out, row)
                print(f"[h2h]   -> {row.status.value}"
                      f" final_len={len(baseline_text)}")

            # 3. Judge (3 seeds, counterbalanced)
            if not (treatment_text and baseline_text):
                print(f"[h2h] {qid} skipping judges — empty treatment/baseline text")
                continue

            for i, seed in enumerate(JUDGE_SEEDS):
                trial_j = f"{qid}-judge-{seed}"
                if trial_j in skip:
                    print(f"[h2h] {trial_j} skip")
                    continue
                treatment_first = (i % 2 == 0)
                a_text = treatment_text if treatment_first else baseline_text
                b_text = baseline_text if treatment_first else treatment_text
                print(f"[h2h] {trial_j} (treatment={'A' if treatment_first else 'B'})...")
                started = now_iso()
                verdict = judge_head_to_head(
                    question=qtext, a_text=a_text, b_text=b_text, seed_label=seed,
                )
                if verdict.get("status") == "ok":
                    raw_winner = verdict["winner"]
                    if raw_winner == "tie":
                        treatment_won = "tie"
                    elif treatment_first:
                        treatment_won = "treatment" if raw_winner == "A" else "baseline"
                    else:
                        treatment_won = "treatment" if raw_winner == "B" else "baseline"
                    # Re-label scores so payload always speaks in
                    # treatment/baseline terms, not A/B.
                    raw_scores = verdict["scores"]
                    norm_scores = {
                        "treatment": raw_scores["A"] if treatment_first else raw_scores["B"],
                        "baseline":  raw_scores["B"] if treatment_first else raw_scores["A"],
                    }
                    row = EvalRun(
                        suite="headtohead", trial_id=trial_j, status=RunStatus.OK,
                        started_at=started, finished_at=now_iso(),
                        payload={
                            "qid": qid, "question": qtext, "seed": seed,
                            "treatment_position": "A" if treatment_first else "B",
                            "raw_winner": raw_winner,
                            "treatment_won": treatment_won,
                            "scores": norm_scores,
                            "rationale": verdict.get("rationale", ""),
                            "judge_model": verdict.get("judge_model"),
                            "prompt_version": verdict.get("prompt_version"),
                            # Mirror as winner=A/B/tie for preference_rate() reuse —
                            # always treats treatment as A regardless of presentation.
                            "winner": (
                                "tie" if treatment_won == "tie"
                                else "A" if treatment_won == "treatment"
                                else "B"
                            ),
                        },
                    )
                else:
                    row = EvalRun(
                        suite="headtohead", trial_id=trial_j, status=RunStatus.ERROR,
                        started_at=started, finished_at=now_iso(),
                        payload={"qid": qid, "seed": seed},
                        error=verdict.get("error", "judge_unknown_error"),
                    )
                append_jsonl(out, row)
                if row.status == RunStatus.OK:
                    print(f"[h2h]   -> ok treatment_won={row.payload['treatment_won']}")
                else:
                    print(f"[h2h]   -> error {row.error[:80]}")
    finally:
        _restore_env(prior_env)
    print(f"[h2h] done. results: {out}")
