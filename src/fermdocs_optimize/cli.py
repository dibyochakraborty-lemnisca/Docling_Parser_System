"""fermdocs-optimize — run the closed-loop titer optimizer.

Example:
    fermdocs-optimize \\
        --train train_data.csv \\
        --mech-params mech_params.json \\
        --box config.json \\
        --rounds 6 --proposals 4 --out optimization.json

`--mech-params` is the LABS simulator's TRUE params; it is passed to the
generate-batches subprocess only — the agent's model never reads it.
`--box` is a JSON with `var_params` (e.g. the provided config.json).
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

from fermdocs_optimize.agent import OptimizerAgent
from fermdocs_optimize.evaluate import peak_titer_per_batch
from fermdocs_optimize.llm_clients import build_optimize_client
from fermdocs_optimize.schema import Box, OptimizationInput
from fermdocs_optimize.simulators.labs import LABSSimulator


def _load_box(path: str) -> Box:
    cfg = json.loads(Path(path).read_text())
    vp = cfg.get("var_params", cfg)
    return Box(**{k: (vp[k]["lb"], vp[k]["ub"]) for k in
                  ("biomass", "total_sub", "malt_frac", "dilution")})


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="fermdocs-optimize")
    ap.add_argument("--train", required=True, help="seed training CSV (batch,t,X,S,P,M,V)")
    ap.add_argument("--mech-params", required=True, help="LABS true-params JSON (oracle only)")
    ap.add_argument("--box", required=True, help="JSON with var_params bounds (config.json)")
    ap.add_argument("--rounds", type=int, default=8)
    ap.add_argument("--proposals", type=int, default=4)
    ap.add_argument("--delta", type=float, default=2.0, help="ΔP convergence threshold")
    ap.add_argument("--v0", type=float, default=10.0)
    ap.add_argument("--generate-batches-bin", default="generate-batches")
    ap.add_argument("--provider", default="none",
                    help="LLM orchestrator: gemini | anthropic | none (deterministic loop)")
    ap.add_argument("--debate", default=None,
                    help="optional optimization_debate.json — advisory levers (inform-only)")
    ap.add_argument("--oracle-search", dest="oracle_search", action="store_true", default=True,
                    help="after the loop, search the simulator directly for the TRUE box max (default on)")
    ap.add_argument("--no-oracle-search", dest="oracle_search", action="store_false")
    ap.add_argument("--n-lhs", type=int, default=200, help="dense oracle sweep size")
    ap.add_argument("--refine-iters", type=int, default=10, help="oracle pattern-search steps")
    ap.add_argument("--out", default="optimization.json")
    args = ap.parse_args(argv)

    train = pd.read_csv(args.train)
    box = _load_box(args.box)
    simulator = LABSSimulator(args.mech_params,
                              generate_batches_bin=args.generate_batches_bin)

    # baseline = best peak titer present in the seed data
    baseline = max(peak_titer_per_batch(train, "P").values(), default=None)

    # The agent orchestrates the deterministic loop. With provider=none it runs
    # the loop directly with defaults; the simulator (which holds the oracle's
    # true-params path) is injected — the agent's model never reads it.
    client = build_optimize_client(args.provider)
    agent = OptimizerAgent(client, provider=args.provider if client else "none")
    fallback = OptimizationInput(box=box, max_rounds=args.rounds,
                                 proposals_per_round=args.proposals,
                                 delta_titer_threshold=args.delta, v0=args.v0,
                                 oracle_search=args.oracle_search,
                                 n_lhs=args.n_lhs, refine_iters=args.refine_iters)
    out = agent.optimize(training_data=train, box=box, simulator=simulator,
                         baseline_titer=baseline, v0=args.v0, fallback_spec=fallback,
                         debate_output_path=args.debate)

    Path(args.out).write_text(out.model_dump_json(indent=2))
    if out.confident:
        print(out.selection_rationale)
        print(f"trajectory: {out.convergence.titer_trajectory}")
        if out.oracle_search is not None:
            r = out.oracle_search
            print(f"oracle search: true box max {r.best_titer} g/L over {r.n_oracle_evals} "
                  f"simulator evals; on-boundary knobs: {r.knobs_on_boundary or 'none'}")
    else:
        print(f"REFUSED: {out.refusal_reason} — {out.selection_rationale}")
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
