"""CLI for equation discovery: the agent proposes ODE structure, the oracle
(LABS) scores it, the agent revises, repeat. Keeps the best structure found.

    python -m fermdocs_optimize.discovery.cli \\
        --config config.json --mech-params mech_params.json \\
        --train train_data.csv --bin /tmp/labs_venv/bin/generate-batches \\
        --proposer llm --rounds 4 --probes 24

`--proposer template` (default) runs the deterministic structural search with no
LLM. `--proposer llm` lets Gemini/Anthropic write the equations (needs a key).
`--mech-params` is the oracle's true params — passed to LABS only, never read by
the agent's models.
"""
from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import pandas as pd

from fermdocs_optimize.discovery.loop import discover_model
from fermdocs_optimize.discovery.proposers import LLMSpecProposer, TemplateProposer
from fermdocs_optimize.schema import Box
from fermdocs_optimize.simulators.labs import LABSSimulator


def _box_from_config(path: str) -> Box:
    vp = json.loads(Path(path).read_text())["var_params"]
    return Box(**{k: (vp[k]["lb"], vp[k]["ub"])
                  for k in ("biomass", "total_sub", "malt_frac", "dilution")})


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Discover the kinetic ODE structure against the oracle.")
    ap.add_argument("--config", required=True, help="config.json with var_params (the box)")
    ap.add_argument("--mech-params", required=True, help="LABS true-params JSON (oracle only)")
    ap.add_argument("--train", required=True, help="seed batches CSV (batch,t,X,S,P,M,V)")
    ap.add_argument("--bin", default="generate-batches", help="LABS generate-batches binary")
    ap.add_argument("--proposer", choices=("template", "llm"), default="template")
    ap.add_argument("--rounds", type=int, default=5)
    ap.add_argument("--probes", type=int, default=24)
    ap.add_argument("--v0", type=float, default=10.0)
    ap.add_argument("--json", action="store_true", help="print the full report as JSON")
    args = ap.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(message)s")
    train = pd.read_csv(args.train)[["batch", "t", "X", "S", "P", "M", "V"]]
    box = _box_from_config(args.config)
    sim = LABSSimulator(args.mech_params, generate_batches_bin=args.bin)
    proposer = LLMSpecProposer() if args.proposer == "llm" else TemplateProposer()

    rep = discover_model(training_data=train, simulator=sim, box=box, proposer=proposer,
                         max_rounds=args.rounds, n_probes=args.probes, v0=args.v0)

    if args.json:
        print(rep.model_dump_json(indent=2))
        return 0

    print(f"\nbaseline (fixed mechanistic) oracle peak RMSE = {rep.baseline_peak_rmse} g/L")
    print(f"best structure: '{rep.best_spec.name if rep.best_spec else None}' "
          f"(round {rep.best_round})")
    print(f"  oracle peak RMSE = {rep.oracle_peak_rmse} g/L | peak R2 = {rep.oracle_peak_r2}")
    print(f"  improved over baseline? {rep.improved} | exit: {rep.exit_reason} | "
          f"oracle evals: {rep.n_oracle_evals}")
    print("\nleaderboard:")
    for r in rep.rounds:
        flag = "" if r.compile_ok else " [COMPILE ERROR]"
        print(f"  r{r.round_index} {r.spec.name:24s} RMSE={r.oracle_peak_rmse:8.3f} "
              f"peakR2={r.oracle_peak_r2:7.3f} trajR2={r.oracle_traj_r2:7.3f}{flag}")
    if rep.best_spec:
        print(f"\nbest equations ('{rep.best_spec.name}'):")
        for k, v in rep.best_spec.aux.items():
            print(f"  {k} = {v}")
        for s, v in rep.best_spec.odes.items():
            print(f"  d{s}/dt = {v}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
