"""Offline pre-training of the IndPenSim penicillin base model.

Trains a control-augmented LSTM (inputs: Fs, Fg, RPM, Fpaa; output: penicillin P)
on the 800-batch IndPenSim dataset with a small hyperparameter search, a 3-way
train/val/test split by batch, and saves the winning model + normalisation scales
+ metadata as a cached artifact the live stage can load.

Run:
    parsevenv/bin/python -m fermdocs_recommend.pretrain \
        --data data/indpensim_800/output_800/batch_csv \
        --out  src/fermdocs_recommend/models/penicillin_fedbatch

Column resolution handles BOTH the 800-batch descriptive headers
("Sugar feed rate(Fs:L/h)") and the raw short headers ("Fs") used by uploaded
runs, so the same model applies at inference time on the raw uploaded CSV.
"""

from __future__ import annotations

import argparse
import glob
import json
import statistics
from pathlib import Path

import numpy as np

# Logical channels -> (exact short name, descriptive substring)
CONTROL_CHANNELS = ["Fs", "Fg", "RPM", "Fpaa"]
OUTPUT_CHANNELS = ["P"]
_COL_PATTERNS = {
    "Fs": "Fs:L/h", "Fg": "Fg:L/h", "RPM": "RPM:RPM", "Fpaa": "Fpaa", "P": "P:g/L",
}


def resolve_col(df, key: str) -> str:
    if key in df.columns:           # raw short headers (uploaded runs)
        return key
    sub = _COL_PATTERNS[key]        # descriptive headers (800-batch)
    for c in df.columns:
        if sub in c:
            return c
    raise KeyError(f"column for {key!r} not found")


def load_run(df, stride: int):
    """Return (U[n,4] controls, Y[n,1] penicillin) on the strided grid."""
    import pandas as pd
    U = np.column_stack([pd.to_numeric(df[resolve_col(df, k)], errors="coerce").to_numpy(float)[::stride] for k in CONTROL_CHANNELS])
    Y = np.column_stack([pd.to_numeric(df[resolve_col(df, k)], errors="coerce").to_numpy(float)[::stride] for k in OUTPUT_CHANNELS])
    return np.nan_to_num(U), np.nan_to_num(Y)


def _r2(yp, yo):
    m = np.isfinite(yp) & np.isfinite(yo)
    ss = float(np.sum((yo[m] - yo[m].mean()) ** 2))
    return float(1 - np.sum((yp[m] - yo[m]) ** 2) / ss) if ss > 0 and m.sum() > 2 else float("nan")


def train_penicillin_base(data_dir: str, out_dir: str, *, n_train=500, n_val=100,
                          n_test=100, stride=5, grid=None) -> dict:
    import pandas as pd
    import jax; jax.config.update("jax_enable_x64", True)
    import jax.numpy as jnp, jax.random as jr
    import equinox as eqx
    from brewtwin.data.schemas import Trajectory
    from brewtwin.surrogates.rnn import LSTMSurrogate
    from brewtwin.surrogates.train import fit_surrogate

    grid = grid or [{"hidden": 64, "lr": 1e-3}, {"hidden": 96, "lr": 1e-3},
                    {"hidden": 64, "lr": 2e-3}, {"hidden": 96, "lr": 5e-4}]
    files = sorted(glob.glob(f"{data_dir}/*.csv"))
    tr_f = files[:n_train]
    va_f = files[n_train:n_train + n_val]
    te_f = files[n_train + n_val:n_train + n_val + n_test]

    def load_set(fs):
        return [load_run(pd.read_csv(f), stride) for f in fs]

    train, val, test = load_set(tr_f), load_set(va_f), load_set(te_f)
    allU = np.vstack([u for u, _ in train]); allY = np.vstack([y for _, y in train])
    u_mu, u_sc = allU.mean(0), np.where(allU.std(0) > 0, allU.std(0), 1.0)
    y_mu, y_sc = allY.mean(0), np.where(allY.std(0) > 0, allY.std(0), 1.0)

    def trajs_and_u(dataset):
        trajs, useqs = [], []
        for U, Y in dataset:
            Yn = (Y - y_mu) / y_sc
            trajs.append(Trajectory.from_dense(t=np.arange(len(Yn), dtype=float),
                                               concentrations={"penicillin": Yn[:, 0]}))
            useqs.append(jnp.asarray(((U - u_mu) / u_sc)[:-1]))
        return trajs, useqs

    tr_trajs, tr_u = trajs_and_u(train)

    def eval_set(model, dataset):
        scores = []
        for U, Y in dataset:
            y0 = jnp.asarray((Y[0] - y_mu) / y_sc)
            u = jnp.asarray((U - u_mu) / u_sc)[:-1]
            pred = np.asarray(model.rollout(y0, u)) * y_sc + y_mu
            L = min(len(pred), len(Y))
            scores.append(_r2(pred[:L, 0], Y[:L, 0]))
        return [s for s in scores if np.isfinite(s)]

    best = None
    for i, cfg in enumerate(grid):
        lstm = LSTMSurrogate(state_size=1, input_size=len(CONTROL_CHANNELS),
                             hidden_size=cfg["hidden"], key=jr.key(i))
        res = fit_surrogate(lstm, tr_trajs, ["penicillin"], u_seqs=tr_u,
                            n_epochs=150, lr=cfg["lr"], progress=False)
        vs = eval_set(res.surrogate, val)
        med = statistics.median(vs) if vs else float("-inf")
        print(f"  cfg {cfg}: val penicillin R2 median={med:.3f} (n={len(vs)})")
        if best is None or med > best["val_median"]:
            best = {"cfg": cfg, "val_median": med, "model": res.surrogate}

    ts = eval_set(best["model"], test)
    test_median = statistics.median(ts) if ts else float("nan")
    test_hit = float(np.mean(np.array(ts) > 0.75)) if ts else float("nan")
    print(f"BEST {best['cfg']}: TEST penicillin R2 median={test_median:.3f}, %>0.75={test_hit*100:.0f}%")

    out = Path(out_dir); out.mkdir(parents=True, exist_ok=True)
    eqx.tree_serialise_leaves(out / "model.eqx", best["model"])
    meta = {
        "family": "penicillin_fedbatch",
        "model_type": "surrogate_lstm_control_augmented",
        "state_size": 1, "input_size": len(CONTROL_CHANNELS),
        "hidden_size": best["cfg"]["hidden"], "stride": stride,
        "control_channels": CONTROL_CHANNELS, "output_channels": OUTPUT_CHANNELS,
        "u_mu": u_mu.tolist(), "u_sc": u_sc.tolist(),
        "y_mu": y_mu.tolist(), "y_sc": y_sc.tolist(),
        "hyperparameters": best["cfg"],
        "test_penicillin_r2_median": test_median, "test_pct_above_0_75": test_hit,
        "n_train": len(train), "n_val": len(val), "n_test": len(test),
    }
    (out / "meta.json").write_text(json.dumps(meta, indent=2))
    print(f"saved artifact to {out}")
    return meta


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--n-train", type=int, default=500)
    ap.add_argument("--n-val", type=int, default=100)
    ap.add_argument("--n-test", type=int, default=100)
    ap.add_argument("--stride", type=int, default=5)
    args = ap.parse_args()
    train_penicillin_base(args.data, args.out, n_train=args.n_train, n_val=args.n_val,
                          n_test=args.n_test, stride=args.stride)
