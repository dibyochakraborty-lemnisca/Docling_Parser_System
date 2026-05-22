"""fermdocs_eval — quantitative evals for the CAISc 2026 paper.

Three suites:
- E1 memory mechanism (cold/warm same-bundle)
- E2 critic axes precision/recall (synthetic hypotheses)
- E3 case studies vs single-shot baseline

Outputs are written to eval/results/ as JSONL so metrics can be recomputed
without re-running pipelines. Judge prompts live in eval/prompts/ and are
checked into the repo for paper reproducibility.
"""

from fermdocs_eval.harness import EvalRun, RunStatus

__all__ = ["EvalRun", "RunStatus"]
