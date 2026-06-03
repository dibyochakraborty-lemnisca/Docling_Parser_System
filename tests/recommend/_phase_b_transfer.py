"""Transfer experiment: train control-augmented LSTM on the 800-batch data,
predict the 2 unseen runs in the root IndPenSim_V2_export_V7.csv.

Tests whether a model trained on history generalises to a brand-new run (the
warm-start / train-on-history-predict-new-run path). Inputs: Fs, Fg, RPM, Fpaa.
Outputs: substrate (S), penicillin (P), dissolved O2 (DO2).
"""
import glob, time, numpy as np
import jax; jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp, jax.random as jr
import pandas as pd
from brewtwin.data.schemas import Trajectory
from brewtwin.surrogates.rnn import LSTMSurrogate
from brewtwin.surrogates.train import fit_surrogate

STRIDE = 5
CTRL_KEYS = ["Fs:L/h", "Fg:L/h", "RPM:RPM", "Fpaa"]      # 800-batch descriptive cols
STATE_KEYS = ["S:g/L", "P:g/L", "DO2:mg/L"]
STATE_NAMES = ["substrate", "penicillin", "do2"]
ROOT_CTRL = ["Fs", "Fg", "RPM", "Fpaa"]                   # root CSV short cols
ROOT_STATE = ["S", "P", "DO2"]

def col(df, key):
    for c in df.columns:
        if key in c:
            return c
    raise KeyError(key)

def load_800(path):
    df = pd.read_csv(path)
    U = np.column_stack([pd.to_numeric(df[col(df, k)], errors="coerce").to_numpy(float)[::STRIDE] for k in CTRL_KEYS])
    Y = np.column_stack([pd.to_numeric(df[col(df, k)], errors="coerce").to_numpy(float)[::STRIDE] for k in STATE_KEYS])
    t = df[col(df, "Time (h)")].to_numpy(float)[::STRIDE]
    return t, np.nan_to_num(U), np.nan_to_num(Y)

def load_root_runs(path):
    df = pd.read_csv(path)
    t = df["Time (h)"].to_numpy(float)
    cuts = np.where(np.diff(t) < 0)[0] + 1
    bounds = [0, *cuts.tolist(), len(df)]
    runs = []
    for a, b in zip(bounds[:-1], bounds[1:]):
        d = df.iloc[a:b]
        U = np.column_stack([pd.to_numeric(d[c], errors="coerce").to_numpy(float)[::STRIDE] for c in ROOT_CTRL])
        Y = np.column_stack([pd.to_numeric(d[c], errors="coerce").to_numpy(float)[::STRIDE] for c in ROOT_STATE])
        runs.append((np.nan_to_num(U), np.nan_to_num(Y)))
    return runs

FILES = sorted(glob.glob("data/indpensim_800/output_800/batch_csv/*.csv"))[:400]
train = [load_800(f) for f in FILES]
allY = np.vstack([y for _, _, y in train]); allU = np.vstack([u for _, u, _ in train])
y_sc = np.where(allY.std(0) > 0, allY.std(0), 1.0); y_mu = allY.mean(0)
u_sc = np.where(allU.std(0) > 0, allU.std(0), 1.0); u_mu = allU.mean(0)

tr_trajs, tr_u = [], []
for t, U, Y in train:
    Yn = (Y - y_mu) / y_sc
    tr_trajs.append(Trajectory.from_dense(t=t, concentrations={n: Yn[:, j] for j, n in enumerate(STATE_NAMES)}))
    tr_u.append(jnp.asarray(((U - u_mu) / u_sc)[:-1]))

lstm = LSTMSurrogate(state_size=3, input_size=4, hidden_size=64, key=jr.key(0))
t0 = time.time()
res = fit_surrogate(lstm, tr_trajs, STATE_NAMES, u_seqs=tr_u, n_epochs=120, lr=1e-3, progress=False)
model = res.surrogate
print(f"trained on {len(FILES)} batches in {time.time()-t0:.0f}s; loss {res.train_loss_history[0]:.3f} -> {res.train_loss_history[-1]:.3f}")

# sanity: value ranges (units consistency between 800 and root)
root = load_root_runs("IndPenSim_V2_export_V7.csv")
print("scale check (800 train means):", dict(zip(STATE_NAMES, np.round(y_mu, 2))))
print("root run-1 state means:", dict(zip(STATE_NAMES, np.round(root[0][1].mean(0), 2))))

def r2(yp, yo):
    m = np.isfinite(yp) & np.isfinite(yo)
    ss = float(np.sum((yo[m]-yo[m].mean())**2))
    return float(1 - np.sum((yp[m]-yo[m])**2)/ss) if ss > 0 and m.sum() > 2 else float("nan")

print("\nTRANSFER -> root IndPenSim runs (trained on 800, predicting these unseen runs):")
for ri, (U, Y) in enumerate(root, 1):
    y0 = jnp.asarray((Y[0] - y_mu) / y_sc)
    u = jnp.asarray((U - u_mu) / u_sc)[:-1]
    pred = np.asarray(model.rollout(y0, u)) * y_sc + y_mu
    L = min(len(pred), len(Y))
    r2s = {n: round(r2(pred[:L, j], Y[:L, j]), 3) for j, n in enumerate(STATE_NAMES)}
    print(f"  run {ri}: R2 = {r2s}")
