"""LABS simulator adapter — the ground-truth oracle for lactic-acid optimization.

Drives the LABS `generate-batches` console script via subprocess:
  1. write the candidates into an explicit-batch config (with fixed o2/kla/noise),
  2. run `generate-batches --config <cfg> --mech-params <true_params> --output <csv>`,
  3. read the resulting trajectories.

`--mech-params` makes LABS skip its (Pyomo/IPOPT) fitting and use the true
params as the oracle — so this path needs no IPOPT. The optimizer passes the
true-params file *path* to the subprocess; it never reads its contents into the
agent's model.
"""
from __future__ import annotations

import json
import subprocess
import tempfile
from pathlib import Path

import pandas as pd

from fermdocs_optimize.schema import Candidate

# Default singlezone physics + clean (noiseless) generation for the oracle.
_DEFAULT_O2 = {"K_O2": 5e-4, "q_O2_max": 0.02, "O2_sat": 7e-3}
_DEFAULT_KLA = {"a": 0.0083, "b": 0.62, "c": 0.49, "P_g_over_V": 1000.0, "v_s": 0.0015}


class LABSSimulator:
    """Ground-truth oracle backed by the LABS `generate-batches` CLI."""

    def __init__(
        self,
        mech_params_path: str | Path,
        *,
        generate_batches_bin: str = "generate-batches",
        o2_params: dict | None = None,
        kla_params: dict | None = None,
        noise_fracs=None,  # None -> LABS default; pass None-literal via config for clean
        seed: int = 11,
        reactor_model: str = "singlezone",
        timeout_s: int = 600,
    ):
        self._mech = str(mech_params_path)
        if not Path(self._mech).exists():
            raise FileNotFoundError(f"mech-params file not found: {self._mech}")
        self._bin = generate_batches_bin
        self._o2 = o2_params or dict(_DEFAULT_O2)
        self._kla = kla_params or dict(_DEFAULT_KLA)
        self._noise = noise_fracs
        self._seed = seed
        self._reactor = reactor_model
        self._timeout = timeout_s

    def simulate(self, candidates: list[Candidate], *, v0: float) -> pd.DataFrame:
        cfg = {
            "reactor_model": self._reactor,
            "batches": [
                {"name": f"cand_{i}", "V0": v0, **c.knobs()}
                for i, c in enumerate(candidates)
            ],
            "o2_params": self._o2,
            "kla_params": self._kla,
            "noise_fracs": self._noise,
            "seed": self._seed,
        }
        with tempfile.TemporaryDirectory() as d:
            cfg_path = Path(d) / "explicit.json"
            out_path = Path(d) / "sim.csv"
            cfg_path.write_text(json.dumps(cfg))
            proc = subprocess.run(
                [self._bin, "--config", str(cfg_path),
                 "--mech-params", self._mech, "--output", str(out_path)],
                capture_output=True, text=True, timeout=self._timeout,
            )
            if proc.returncode != 0 or not out_path.exists():
                raise RuntimeError(
                    f"generate-batches failed (rc={proc.returncode}): {proc.stderr[-500:]}"
                )
            return pd.read_csv(out_path)
