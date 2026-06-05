"""fermdocs-optimize-debate — run the opportunity debate over a bundle.

    fermdocs-optimize-debate run <bundle_dir> [--out-dir DIR] [--objective P]

Writes <out_dir>/optimization_debate.json (HypothesisOutput shape; final
hypotheses are the debated levers) and optimization_debate.md (event log).
"""
from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from pathlib import Path

from fermdocs_hypothesis.schema import BudgetSnapshot

from fermdocs_optimize_debate.loader import load_optimization_bundle
from fermdocs_optimize_debate.run import run_optimization_debate
from fermdocs_optimize_debate.schema import levers_from_output


def main(argv: list[str] | None = None) -> int:
    import argparse

    ap = argparse.ArgumentParser(prog="fermdocs-optimize-debate")
    sub = ap.add_subparsers(dest="cmd", required=True)
    runp = sub.add_parser("run", help="run the optimization debate over a bundle")
    runp.add_argument("bundle_dir")
    runp.add_argument("--out-dir", default=None, help="defaults to the bundle dir")
    runp.add_argument("--objective", default="P", help="objective species to maximize")
    runp.add_argument("--max-trend-topics", type=int, default=4)
    runp.add_argument("--max-turns", type=int, default=20)
    runp.add_argument("--max-critic-cycles-per-topic", type=int, default=6)
    runp.add_argument("--max-tool-calls", type=int, default=160)
    runp.add_argument("--max-total-input-tokens", type=int, default=400_000)
    runp.add_argument("--provider", default="gemini")
    runp.add_argument("--no-validate", action="store_true")
    args = ap.parse_args(argv)

    loaded = load_optimization_bundle(
        args.bundle_dir, objective_species=args.objective,
        max_trend_topics=args.max_trend_topics)
    out_dir = Path(args.out_dir) if args.out_dir else loaded.bundle_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    global_md = out_dir / "optimization_debate.md"

    budget = BudgetSnapshot(
        max_turns=args.max_turns,
        max_critic_cycles_per_topic=args.max_critic_cycles_per_topic,
        max_tool_calls_total=args.max_tool_calls,
        max_total_input_tokens=args.max_total_input_tokens,
    )

    print(f"running optimization debate over {loaded.bundle_dir} "
          f"({len(loaded.hyp_input.seed_topics)} opportunity topics)")
    result = run_optimization_debate(
        loaded, global_md_path=global_md, provider=args.provider,
        budget=budget, validate=not args.no_validate)

    out_path = out_dir / "optimization_debate.json"
    out_path.write_text(json.dumps(result.output.model_dump(mode="json"), indent=2, default=str))
    levers = levers_from_output(result.output)
    print(f"✓ wrote {out_path}")
    print(f"exit reason: {result.state.exit_reason}")
    print(f"debated levers: {len(levers)}")
    for lev in levers[:5]:
        knobs = ",".join(lev.knobs) or "—"
        print(f"  [{lev.lever_id}] knobs={knobs} conf={lev.confidence:.2f}  {lev.summary[:80]}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
