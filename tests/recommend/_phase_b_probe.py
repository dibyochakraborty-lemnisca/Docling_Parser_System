"""Phase B proof: control-augmented LSTM surrogate on the 800-batch IndPenSim data.

The Phase-A/IndPenSim failure was a FEED-BLIND autonomous model. Here the
surrogate sees the controls (Fs, Fg, RPM, Fpaa) as time-varying inputs and
predicts the dense online states (S, P, DO2). Train on many batches, validate on
HELD-OUT batches (true cross-run generalisation). If control inputs are the
missing piece, held-out R^2 on the online channels should be high.
"""
import glob, json, time, numpy as np
import jax; jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp, jax.random as jr
import pandas as pd
from brewtwin.data.schemas import Trajectory
from brewtwin.surrogates.rnn import LSTMSurrogate
from brewtwin.surrogates.train import fit_surrogate

FILES = sorted(glob.glob("data/indpensim_800/output_800/batch_csv/*.csv"))
N_TRAIN, N_TEST, STRIDE = 400, 50, 5
train_files = FILES[:N_TRAIN]
test_files = FILES[N_TRAIN:N_TRAIN + N_TEST]

def col(df, key):
    for c in df.columns:
        if key in c:
            return c
    raise KeyError(key)

CTRL_KEYS = ["Fs:L/h", "Fg:L/h", "RPM:RPM", "Fpaa"]
STATE_KEYS = ["S:g/L", "P:g/L", "DO2:mg/L"]
STATE_NAMES = ["substrate", "penicillin", "do2"]

def load(path):
    df = pd.read_csv(path)
    t = df[col(df, "Time (h)")].to_numpy(float)[::STRIDE]
    U = np.column_stack([pd.to_numeric(df[col(df, k)], errors="coerce").to_numpy(float)[::STRIDE] for k in CTRL_KEYS])
    Y = np.column_stack([pd.to_numeric(df[col(df, k)], errors="coerce").to_numpy(float)[::STRIDE] for k in STATE_KEYS])
    return t, np.nan_to_num(U), np.nan_to_num(Y)

# fit normalisation scales on training data
allY = np.vstack([load(f)[2] for f in train_files])
allU = np.vstack([load(f)[1] for f in train_files])
y_sc = np.where(allY.std(0) > 0, allY.std(0), 1.0); y_mu = allY.mean(0)
u_sc = np.where(allU.std(0) > 0, allU.std(0), 1.0); u_mu = allU.mean(0)

def prep(files):
    trajs, useqs, raws = [], [], []
    for f in files:
        t, U, Y = load(f)
        Yn = (Y - y_mu) / y_sc
        Un = (U - u_mu) / u_sc
        trajs.append(Trajectory.from_dense(t=t, concentrations={n: Yn[:, j] for j, n in enumerate(STATE_NAMES)}))
        useqs.append(jnp.asarray(Un[:-1]))   # (n_steps, input_size)
        raws.append((t, Y))
    return trajs, useqs, raws

tr_trajs, tr_u, _ = prep(train_files)
te_trajs, te_u, te_raw = prep(test_files)

lstm = LSTMSurrogate(state_size=len(STATE_NAMES), input_size=len(CTRL_KEYS), hidden_size=64, key=jr.key(0))
t0 = time.time()
res = fit_surrogate(lstm, tr_trajs, STATE_NAMES, u_seqs=tr_u, n_epochs=120, lr=1e-3, progress=False)
model = res.surrogate
print(f"trained on {N_TRAIN} batches in {time.time()-t0:.0f}s; loss {res.train_loss_history[0]:.3f} -> {res.train_loss_history[-1]:.3f}")

def r2(yp, yo):
    m = np.isfinite(yp) & np.isfinite(yo)
    ss = float(np.sum((yo[m]-yo[m].mean())**2))
    return float(1 - np.sum((yp[m]-yo[m])**2)/ss) if ss > 0 and m.sum() > 2 else float("nan")

# held-out evaluation
agg = {n: [] for n in STATE_NAMES}
for (t, Yreal), u in zip(te_raw, te_u):
    y0 = jnp.asarray(((Yreal[0] - y_mu) / y_sc))
    pred_n = np.asarray(model.rollout(y0, u))           # (n_steps+1, state)
    pred = pred_n * y_sc + y_mu
    L = min(len(pred), len(Yreal))
    for j, n in enumerate(STATE_NAMES):
        agg[n].append(r2(pred[:L, j], Yreal[:L, j]))

print("HELD-OUT R^2 over %d test batches (median is robust to diverging outliers):" % N_TEST)
for n in STATE_NAMES:
    vals = np.array([v for v in agg[n] if np.isfinite(v)])
    hit = float(np.mean(vals > 0.75)) if len(vals) else float("nan")
    print(f"  {n:11s} median={np.median(vals):.3f}  mean={np.mean(vals):.3f}  "
          f"%>0.75={hit*100:.0f}%  max={np.max(vals):.3f}  (n={len(vals)})")
