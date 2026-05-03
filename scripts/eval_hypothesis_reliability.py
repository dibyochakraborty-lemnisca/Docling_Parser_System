"""N-rerun reliability evaluation for the hypothesis stage.

Plan ref: plans/2026-05-03-hypothesis-debate-v0.md §11 Stage 4, §12.

Runs the full hypothesis stage N times against each target bundle.
Measures how stable the system's answer is across reruns by computing
Jaccard overlap of root-cause variables (affected_variables across all
final hypotheses) per run, and comparing pairwise.

Pass criterion (from yaml specs):
  carotenoid_pdf:    Jaccard ≥ 0.6 (high stability expected)
  indpensim_csv:     Jaccard ≥ 0.6
  mixed_bundle:      Jaccard ≥ 0.5 (synthetic, less stable expected)

Outputs a markdown report at out/hypothesis/reliability_report.md.

Usage:
    python scripts/eval_hypothesis_reliability.py [--n 5] [--bundles carotenoid,indpensim]

Cost: roughly N × bundles × ~150-180k Gemini tokens. Default N=5 across
2 bundles is ~$3-5 in Gemini spend.
"""

from __future__ import annotations

import argparse
import json
import time
import uuid
from datetime import datetime, timezone
from itertools import combinations
from pathlib import Path

try:
    from dotenv import load_dotenv

    load_dotenv()
except ImportError:
    pass

from fermdocs_hypothesis.bundle_loader import load_bundle
from fermdocs_hypothesis.live_hooks import LiveHooks
from fermdocs_hypothesis.runner import run_stage
from fermdocs_hypothesis.schema import BudgetSnapshot

ROOT = Path(__file__).resolve().parents[1]

BUNDLES = {
    "carotenoid": {
        "path": ROOT / "out" / "bundle_multi_20260502T220318Z_1a29e4",
        "pass_jaccard": 0.6,
    },
    "indpensim": {
        "path": ROOT / "out" / "bundle_indpensim",
        "pass_jaccard": 0.6,
    },
}


def _run_once(bundle_path: Path, out_root: Path) -> dict:
    """One full hypothesis stage run. Returns a summary dict."""
    loaded = load_bundle(bundle_path)
    hyp_id = uuid.uuid4()
    out_dir = out_root / str(hyp_id)
    out_dir.mkdir(parents=True, exist_ok=True)
    global_md = out_dir / "global.md"

    budget = BudgetSnapshot(
        max_total_input_tokens=200_000,
        max_tool_calls_total=80,
    )
    hooks = LiveHooks(loaded)
    t0 = time.time()
    result = run_stage(
        hyp_input=loaded.hyp_input,
        hooks=hooks,
        global_md_path=global_md,
        diagnosis_id=loaded.diagnosis.meta.diagnosis_id,
        provider="gemini",
        model_name=hooks._client.model_name,
        budget=budget,
        validate=True,
        now_factory=lambda: datetime.now(timezone.utc),
    )
    elapsed_s = time.time() - t0
    out_path = out_dir / "hypothesis_output.json"
    out_path.write_text(
        json.dumps(result.output.model_dump(mode="json"), indent=2, default=str)
    )

    final_vars: set[str] = set()
    for h in result.output.final_hypotheses:
        final_vars.update(v.lower() for v in h.affected_variables)

    return {
        "hyp_id": str(hyp_id),
        "out_dir": str(out_dir),
        "exit_reason": result.state.exit_reason,
        "n_final": len(result.output.final_hypotheses),
        "n_rejected": len(result.output.rejected_hypotheses),
        "n_open_questions": len(
            [q for q in result.output.open_questions if not q.resolved]
        ),
        "tokens_input": result.output.token_report.total_input,
        "tokens_output": result.output.token_report.total_output,
        "elapsed_s": round(elapsed_s, 1),
        "final_summaries": [
            h.summary[:200] for h in result.output.final_hypotheses
        ],
        "final_variables": sorted(final_vars),
    }


def _jaccard(a: set, b: set) -> float:
    if not a and not b:
        return 1.0
    if not a or not b:
        return 0.0
    return len(a & b) / len(a | b)


def _pairwise_jaccards(runs: list[dict]) -> list[float]:
    sets = [set(r["final_variables"]) for r in runs]
    return [_jaccard(a, b) for a, b in combinations(sets, 2)]


def _render_markdown(report: dict) -> str:
    lines = [
        "# Hypothesis-stage reliability report",
        "",
        f"Generated: {datetime.now(timezone.utc).isoformat()}",
        f"N reruns per bundle: {report['n']}",
        "",
        "## Summary",
        "",
        "| Bundle | Pass | Mean Jaccard | Min Jaccard | Pairs | Avg tokens (in/out) | Avg time (s) |",
        "|---|---|---|---|---|---|---|",
    ]
    for bundle_name, r in report["per_bundle"].items():
        pass_str = "✓" if r["passed"] else "✗"
        lines.append(
            f"| `{bundle_name}` | {pass_str} | {r['mean_jaccard']:.2f} (≥{r['pass_threshold']}) |"
            f" {r['min_jaccard']:.2f} | {len(r['pairwise_jaccards'])} |"
            f" {r['avg_tokens_input']:,}/{r['avg_tokens_output']:,} |"
            f" {r['avg_elapsed_s']:.0f} |"
        )
    lines.append("")
    for bundle_name, r in report["per_bundle"].items():
        lines.append(f"## `{bundle_name}` runs")
        lines.append("")
        for i, run in enumerate(r["runs"], 1):
            lines.append(f"### Run {i} ({run['hyp_id'][:8]})")
            lines.append("")
            lines.append(f"- exit: `{run['exit_reason']}`")
            lines.append(f"- final / rejected: {run['n_final']} / {run['n_rejected']}")
            lines.append(f"- unresolved open questions: {run['n_open_questions']}")
            lines.append(f"- tokens: {run['tokens_input']:,} in / {run['tokens_output']:,} out")
            lines.append(f"- elapsed: {run['elapsed_s']}s")
            lines.append(f"- root-cause vars: `{', '.join(run['final_variables']) or '(none)'}`")
            for j, summary in enumerate(run["final_summaries"], 1):
                lines.append(f"- H{j}: {summary}")
            lines.append("")
    return "\n".join(lines) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=5)
    parser.add_argument(
        "--bundles", default=",".join(BUNDLES.keys()),
        help="comma-separated bundle keys",
    )
    parser.add_argument(
        "--out-root", type=Path, default=ROOT / "out" / "hypothesis" / "reliability",
    )
    args = parser.parse_args()

    selected = [b.strip() for b in args.bundles.split(",") if b.strip()]
    args.out_root.mkdir(parents=True, exist_ok=True)

    report: dict = {"n": args.n, "per_bundle": {}}
    overall_pass = True

    for bundle_name in selected:
        if bundle_name not in BUNDLES:
            print(f"[skip] unknown bundle: {bundle_name}")
            continue
        bundle_meta = BUNDLES[bundle_name]
        bundle_path = bundle_meta["path"]
        if not bundle_path.exists():
            print(f"[skip] {bundle_name}: bundle not present at {bundle_path}")
            continue

        print(f"\n=== {bundle_name} (N={args.n}) ===")
        runs: list[dict] = []
        for i in range(args.n):
            print(f"[{bundle_name}] run {i+1}/{args.n}")
            run = _run_once(bundle_path, args.out_root / bundle_name)
            runs.append(run)
            print(
                f"  exit={run['exit_reason']}"
                f" final={run['n_final']} rejected={run['n_rejected']}"
                f" tokens={run['tokens_input']:,}in/{run['tokens_output']:,}out"
                f" t={run['elapsed_s']}s"
            )

        jaccards = _pairwise_jaccards(runs)
        mean_j = sum(jaccards) / len(jaccards) if jaccards else 1.0
        min_j = min(jaccards) if jaccards else 1.0
        threshold = bundle_meta["pass_jaccard"]
        passed = mean_j >= threshold
        overall_pass = overall_pass and passed

        report["per_bundle"][bundle_name] = {
            "runs": runs,
            "pairwise_jaccards": jaccards,
            "mean_jaccard": mean_j,
            "min_jaccard": min_j,
            "pass_threshold": threshold,
            "passed": passed,
            "avg_tokens_input": sum(r["tokens_input"] for r in runs) // len(runs),
            "avg_tokens_output": sum(r["tokens_output"] for r in runs) // len(runs),
            "avg_elapsed_s": sum(r["elapsed_s"] for r in runs) / len(runs),
        }

    report_md = _render_markdown(report)
    report_path = args.out_root / "reliability_report.md"
    report_path.write_text(report_md)
    json_path = args.out_root / "reliability_report.json"
    json_path.write_text(json.dumps(report, indent=2, default=str))

    print(f"\n✓ wrote {report_path}")
    print(f"✓ wrote {json_path}")
    print()
    for bundle_name, r in report["per_bundle"].items():
        status = "PASS" if r["passed"] else "FAIL"
        print(
            f"  {status} {bundle_name}: mean Jaccard {r['mean_jaccard']:.2f}"
            f" (≥{r['pass_threshold']}), min {r['min_jaccard']:.2f}"
        )
    return 0 if overall_pass else 1


if __name__ == "__main__":
    raise SystemExit(main())
