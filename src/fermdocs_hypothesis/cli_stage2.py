"""Stage 2 CLI — `python -m fermdocs_hypothesis.cli_stage2 <bundle_path>`.

Loads a bundle, runs the hypothesis stage with LiveHooks (real LLM agents),
and writes hypothesis_output.json + global.md to a sibling
`out/hypothesis/<hypothesis_id>/` directory.

Stage 4 will replace this with `fermdocs-hypothesize` console-script.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import uuid
from datetime import datetime, timezone
from pathlib import Path

from fermdocs_hypothesis.bundle_loader import load_bundle
from fermdocs_hypothesis.live_hooks import LiveHooks
from fermdocs_hypothesis.runner import run_stage
from fermdocs_hypothesis.schema import BudgetSnapshot


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="hypothesize-stage2")
    parser.add_argument("bundle_dir", help="Path to a diagnose bundle directory.")
    parser.add_argument(
        "--out-root",
        default="out/hypothesis",
        help="Root directory for hypothesis bundles (default: out/hypothesis).",
    )
    parser.add_argument("--max-turns", type=int, default=2)
    parser.add_argument("--max-tool-calls-total", type=int, default=40)
    parser.add_argument("--max-total-input-tokens", type=int, default=120_000)
    args = parser.parse_args(argv)

    try:
        from dotenv import load_dotenv

        load_dotenv()
    except Exception:
        pass

    bundle_dir = Path(args.bundle_dir)
    if not bundle_dir.exists():
        print(f"bundle dir not found: {bundle_dir}", file=sys.stderr)
        return 2

    print(f"loading bundle: {bundle_dir}")
    loaded = load_bundle(bundle_dir)
    print(
        f"  seed_topics={len(loaded.hyp_input.seed_topics)} "
        f"findings={len(loaded.findings_pool)} "
        f"narratives={len(loaded.narratives_pool)} "
        f"trajectories={len(loaded.trajectories_pool)} "
        f"organism={loaded.hyp_input.organism!r}"
    )

    hyp_id = uuid.uuid4()
    out_dir = Path(args.out_root) / str(hyp_id)
    out_dir.mkdir(parents=True, exist_ok=True)
    global_md = out_dir / "global.md"

    budget = BudgetSnapshot(
        max_turns=args.max_turns,
        max_tool_calls_total=args.max_tool_calls_total,
        max_total_input_tokens=args.max_total_input_tokens,
    )

    hooks = LiveHooks(loaded)
    diagnosis_id = loaded.diagnosis.meta.diagnosis_id

    print(f"running hypothesis stage (hyp_id={hyp_id})")
    result = run_stage(
        hyp_input=loaded.hyp_input,
        hooks=hooks,
        global_md_path=global_md,
        diagnosis_id=diagnosis_id,
        provider="gemini",
        model_name=hooks._client.model_name,
        budget=budget,
        validate=True,
        now_factory=lambda: datetime.now(timezone.utc),
    )

    out_path = out_dir / "hypothesis_output.json"
    out_path.write_text(json.dumps(result.output.model_dump(mode="json"), indent=2, default=str))

    print(f"\n✓ wrote {out_path}")
    print(f"✓ wrote {global_md}")
    print(f"\nfinal_hypotheses: {len(result.output.final_hypotheses)}")
    for h in result.output.final_hypotheses:
        print(f"  - {h.hyp_id}: {h.summary[:120]}")
    print(f"rejected_hypotheses: {len(result.output.rejected_hypotheses)}")
    print(f"open_questions: {len(result.output.open_questions)}")
    print(f"exit reason: {result.state.exit_reason}")
    print(
        f"tokens: input={result.output.token_report.total_input}"
        f" output={result.output.token_report.total_output}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
