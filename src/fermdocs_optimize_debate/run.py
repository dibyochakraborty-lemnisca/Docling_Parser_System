"""Run the optimization debate over a loaded bundle.

Thin orchestration: build OptimizeHooks, drive the reused `run_stage`, and return
the engine's RunResult (whose `.output` is a HypothesisOutput whose
final_hypotheses ARE the debated optimization levers).
"""
from __future__ import annotations

from pathlib import Path

from fermdocs_hypothesis.runner import run_stage
from fermdocs_hypothesis.schema import BudgetSnapshot

from fermdocs_optimize_debate.hooks import OptimizeHooks
from fermdocs_optimize_debate.loader import OptimizeLoadedBundle


def run_optimization_debate(
    loaded: OptimizeLoadedBundle,
    *,
    global_md_path: Path,
    provider: str = "gemini",
    budget: BudgetSnapshot | None = None,
    validate: bool = True,
    run_id: str | None = None,
    hooks: OptimizeHooks | None = None,
):
    """Drive the debate. `hooks` is injectable so tests can pass stub hooks
    (no LLM); production builds OptimizeHooks (real Gemini specialists)."""
    hooks = hooks or OptimizeHooks(loaded, run_id=run_id)
    model_name = getattr(getattr(hooks, "_client", None), "model_name", "stub")
    return run_stage(
        hyp_input=loaded.hyp_input,
        hooks=hooks,
        global_md_path=global_md_path,
        diagnosis_id=loaded.diagnosis_id,
        provider=provider,  # type: ignore[arg-type]
        model_name=model_name,
        budget=budget or BudgetSnapshot(),
        validate=validate,
    )
