"""Motor-cortex decoding on ModelDataRightContra with six models.

Units: MOp/MOs only. Neurons are a *set* — shared weights plus masked
pooling, never a fixed neuron index — because unit 1 in session A is not
unit 1 in session B. Trials have variable T (bins) and N (units); batches
pad both axes and mask.

Tasks
    trial_holdout   10% of trials from 10% of sessions (same session seen)
    session_holdout hold out one CSH_ZAD_026 session; other sessions of
                    that mouse stay in train (new neurons, known mouse)
    mouse_holdout   hold out ZM_2241 (single-session mouse; new animal)

Models
    cnn, gru, lstm, cnn_gru, transformer, ridge
    Ridge uses a causal sliding window on permutation-invariant population
    statistics (not raw unit-aligned counts). Deep models use MSE on
    z-scored wheel |ω| and 2D paw speed, early stopping, disk cache.
"""

from __future__ import annotations

import argparse
import json
import math
import pickle
import random
import warnings
from datetime import datetime, timezone
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.preprocessing import StandardScaler
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from torch.utils.data import DataLoader, Dataset

import slurm_utils

ROOT = Path(__file__).resolve().parent
DESKTOP = ROOT.parent
DATA = ROOT / "ModelDataRightContra"
SESS = DATA / "sessions"
OUT = ROOT / "MLtestcontra"
CACHE = OUT / "cache"

BINSIZE = 0.02
W_LAGS = 10
MIN_BINS = W_LAGS + 3
MIN_UNITS = 5
MAX_BINS = 128
VAL_FRAC = 0.12
BATCH = 32
MAX_EPOCHS = 25
PATIENCE = 5
LR = 1e-3
WEIGHT_DECAY = 1e-4
SEED = 0
D_MODEL = 64
CNN_CH = (32, 64)
KERNEL = 5
RNN_HIDDEN = 64
N_HEADS = 4
N_LAYERS = 2
FF_DIM = 128
DROPOUT = 0.1
ALPHAS = np.array([0.1, 1.0, 10.0, 100.0, 1000.0, 1e4])

MODELS = ("cnn", "gru", "lstm", "cnn_gru", "transformer", "ridge")
TASKS = ("trial_holdout", "session_holdout", "mouse_holdout")
TARGETS = ("wheel_speed", "paw_speed")
TARGET_LABELS = {"wheel_speed": "wheel speed |ω|", "paw_speed": "paw speed (2D)"}

# Fixed holdouts (documented decisions).
HOLD_MOUSE = "ZM_2241"  # single session, 419 trials, 69 motor units
HOLD_SESSION_MOUSE = "CSH_ZAD_026"
HOLD_SESSION_EID = "626126d5-eecf-4e9b-900e-ec29a17ece07"  # 2020-08-18


def log(msg):
    print(f"[{datetime.now().strftime('%H:%M:%S')}] {msg}", flush=True)


def set_seed(seed=SEED):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def device_of():
    return slurm_utils.device_of()


def job_grid():
    return [{"task": t, "model": m} for t in TASKS for m in MODELS]


def motor_mask(units):
    area = units["brain_area"].astype(str)
    return area.str.startswith("MOp") | area.str.startswith("MOs")


def trial_targets(rec):
    wheel = np.abs(np.asarray(rec.get("wheel_velocity", []), dtype=np.float32))
    paw = np.asarray(rec.get("lp_speed", []), dtype=np.float32)
    if paw.size == 0 or not np.isfinite(paw).any():
        paw = np.asarray(rec.get("dlc_speed", []), dtype=np.float32)
    return wheel, paw


def load_corpus():
    """Load every RightContra session; keep motor units and usable trials."""
    pkls = sorted(p for p in SESS.glob("*.pkl") if not p.name.endswith(".tmp"))
    trials = []
    session_meta = []
    log(f"loading {len(pkls)} sessions from ModelDataRightContra")
    for path in pkls:
        with path.open("rb") as fh:
            payload = pickle.load(fh)
        units = payload["units"]
        m = np.flatnonzero(motor_mask(units).to_numpy())
        mouse = str(payload.get("mouse_id"))
        date = str(payload.get("session_date"))
        eid = str(payload["eid"])
        session_meta.append(
            {
                "eid": eid,
                "mouse_id": mouse,
                "session_date": date,
                "n_motor": int(m.size),
                "n_trials_raw": len(payload["trials"]),
            }
        )
        if m.size < MIN_UNITS:
            continue
        for rec in payload["trials"]:
            spikes = np.asarray(rec.get("spike_counts", []), dtype=np.float32)
            if spikes.ndim != 2 or spikes.shape[1] <= m.max():
                continue
            wheel, paw = trial_targets(rec)
            n = min(spikes.shape[0], wheel.size, paw.size, MAX_BINS)
            if n < MIN_BINS:
                continue
            spikes = spikes[:n, m]
            y = np.stack([wheel[:n], paw[:n]], axis=1)
            finite = np.isfinite(spikes).all(axis=1) & np.isfinite(y).all(axis=1)
            if finite.sum() < MIN_BINS:
                continue
            if finite.sum() < n:
                spikes = spikes[finite]
                y = y[finite]
                if spikes.shape[0] < MIN_BINS:
                    continue
            trials.append(
                {
                    "eid": eid,
                    "mouse_id": mouse,
                    "session_date": date,
                    "trial_index": rec.get("trial_index"),
                    "spikes": np.log1p(np.clip(spikes, 0, None)),
                    "y": y.astype(np.float32),
                }
            )
    log(f"corpus: {len(trials)} trials, {pd.DataFrame(session_meta).eid.nunique()} sessions")
    return trials, session_meta


def make_splits(trials):
    """Three generalization tests. Test indices are disjoint from train."""
    df = pd.DataFrame(
        [
            {
                "i": i,
                "eid": t["eid"],
                "mouse_id": t["mouse_id"],
                "trial_index": t["trial_index"],
            }
            for i, t in enumerate(trials)
        ]
    )
    rng = np.random.default_rng(SEED)
    eids = np.array(sorted(df.eid.unique()))
    n_hold_sess = max(1, int(round(0.10 * len(eids))))
    hold_eids = set(rng.choice(eids, size=n_hold_sess, replace=False).tolist())

    trial_test = []
    for eid in sorted(hold_eids):
        idx = df.index[df.eid == eid].to_numpy()
        n_te = max(1, int(round(0.10 * len(idx))))
        pick = rng.choice(idx, size=n_te, replace=False)
        trial_test.extend(df.loc[pick, "i"].tolist())
    trial_test = np.array(sorted(set(trial_test)), dtype=int)
    trial_train = np.array(sorted(set(df.i) - set(trial_test)), dtype=int)

    sess_test = df.loc[df.eid == HOLD_SESSION_EID, "i"].to_numpy()
    sess_train = df.loc[df.eid != HOLD_SESSION_EID, "i"].to_numpy()
    same_mouse_in_train = df.loc[
        (df.mouse_id == HOLD_SESSION_MOUSE) & (df.eid != HOLD_SESSION_EID), "i"
    ]
    if len(same_mouse_in_train) == 0:
        raise RuntimeError("session_holdout mouse has no other session in train")

    mouse_test = df.loc[df.mouse_id == HOLD_MOUSE, "i"].to_numpy()
    mouse_train = df.loc[df.mouse_id != HOLD_MOUSE, "i"].to_numpy()
    if df.loc[df.mouse_id == HOLD_MOUSE, "eid"].nunique() != 1:
        raise RuntimeError(f"{HOLD_MOUSE} is not a single-session mouse")

    splits = {
        "trial_holdout": {
            "train": trial_train,
            "test": trial_test,
            "note": (
                f"{n_hold_sess}/{len(eids)} sessions contribute held-out trials "
                f"({', '.join(e[:8] for e in sorted(hold_eids))})"
            ),
            "hold_eids": sorted(hold_eids),
        },
        "session_holdout": {
            "train": sess_train,
            "test": sess_test,
            "note": (
                f"held-out session {HOLD_SESSION_EID[:8]} of {HOLD_SESSION_MOUSE}; "
                f"{same_mouse_in_train.size} trials from other sessions of that mouse remain in train"
            ),
            "hold_eids": [HOLD_SESSION_EID],
        },
        "mouse_holdout": {
            "train": mouse_train,
            "test": mouse_test,
            "note": f"held-out mouse {HOLD_MOUSE} (one session, unseen animal)",
            "hold_eids": sorted(df.loc[df.mouse_id == HOLD_MOUSE, "eid"].unique()),
        },
    }
    for name, sp in splits.items():
        log(f"split {name}: train={len(sp['train'])}  test={len(sp['test'])}  {sp['note']}")
    return splits


def train_val_split(train_idx, seed=SEED):
    rng = np.random.default_rng(seed)
    train_idx = np.asarray(train_idx)
    rng.shuffle(train_idx)
    n_val = max(1, int(round(VAL_FRAC * len(train_idx))))
    return train_idx[n_val:], train_idx[:n_val]


def y_scaler_from(trials, idx):
    chunks = [trials[i]["y"][W_LAGS:] for i in idx]
    y = np.concatenate(chunks, axis=0)
    mean = y.mean(axis=0)
    std = y.std(axis=0)
    std = np.where(std < 1e-6, 1.0, std)
    return mean.astype(np.float32), std.astype(np.float32)


class TrialDS(Dataset):
    def __init__(self, trials, idx, y_mean, y_std):
        self.trials = trials
        self.idx = np.asarray(idx)
        self.y_mean = y_mean
        self.y_std = y_std

    def __len__(self):
        return len(self.idx)

    def __getitem__(self, j):
        t = self.trials[int(self.idx[j])]
        y = (t["y"] - self.y_mean) / self.y_std
        return t["spikes"], y.astype(np.float32)


def collate(batch):
    spikes, ys = zip(*batch)
    B = len(batch)
    T = max(s.shape[0] for s in spikes)
    N = max(s.shape[1] for s in spikes)
    x = torch.zeros(B, N, T, dtype=torch.float32)
    y = torch.zeros(B, T, 2, dtype=torch.float32)
    mask_t = torch.zeros(B, T, dtype=torch.bool)
    mask_n = torch.zeros(B, N, dtype=torch.bool)
    lengths = torch.zeros(B, dtype=torch.long)
    for i, (s, yi) in enumerate(zip(spikes, ys)):
        n, tt = s.shape[1], s.shape[0]
        x[i, :n, :tt] = torch.from_numpy(s.T)
        y[i, :tt] = torch.from_numpy(yi)
        mask_t[i, :tt] = True
        mask_n[i, :n] = True
        lengths[i] = tt
    return {"x": x, "y": y, "mask_t": mask_t, "mask_n": mask_n, "lengths": lengths}


class CausalConv1d(nn.Module):
    def __init__(self, cin, cout, k, dilation=1):
        super().__init__()
        self.left = (k - 1) * dilation
        self.conv = nn.Conv1d(cin, cout, k, dilation=dilation)

    def forward(self, x):
        return self.conv(F.pad(x, (self.left, 0)))


class SetPool(nn.Module):
    """Masked mean / max / attention over the neuron axis. Permutation-invariant."""

    def __init__(self, d):
        super().__init__()
        self.attn = nn.Linear(d, 1)

    def forward(self, h, mask_n):
        # h: (B, N, T, D)
        mask = mask_n[:, :, None, None]
        h0 = h.masked_fill(~mask, 0.0)
        denom = mask_n.sum(dim=1).clamp(min=1).to(h.dtype)[:, None, None]
        mean = h0.sum(dim=1) / denom
        hmax = h.masked_fill(~mask, -1e9).max(dim=1).values
        logits = self.attn(h).squeeze(-1)
        logits = logits.masked_fill(~mask_n[:, :, None], -1e9)
        w = torch.softmax(logits, dim=1)
        attn = (w.unsqueeze(-1) * h0).sum(dim=1)
        return torch.cat([mean, hmax, attn], dim=-1)  # (B, T, 3D)


class SharedCausalCNN(nn.Module):
    """Same causal CNN applied independently to every neuron."""

    def __init__(self, channels=CNN_CH, k=KERNEL):
        super().__init__()
        layers = []
        cin = 1
        for cout in channels:
            layers += [CausalConv1d(cin, cout, k), nn.GELU()]
            cin = cout
        self.net = nn.Sequential(*layers)
        self.out_ch = channels[-1]

    def forward(self, x, mask_n):
        B, N, T = x.shape
        h = self.net(x.reshape(B * N, 1, T)).reshape(B, N, self.out_ch, T)
        return h.permute(0, 1, 3, 2)  # (B, N, T, C)


class InstantEmbed(nn.Module):
    def __init__(self, d):
        super().__init__()
        self.lin = nn.Sequential(nn.Linear(1, d), nn.GELU(), nn.Linear(d, d))

    def forward(self, x):
        return self.lin(x.unsqueeze(-1))  # (B, N, T, d)


class Head(nn.Module):
    def __init__(self, d, p=DROPOUT):
        super().__init__()
        self.net = nn.Sequential(nn.Dropout(p), nn.Linear(d, d), nn.GELU(), nn.Linear(d, 2))

    def forward(self, h):
        return self.net(h)


class TemporalCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.per_neuron = SharedCausalCNN()
        self.pool = SetPool(self.per_neuron.out_ch)
        d_in = 3 * self.per_neuron.out_ch
        self.mix = nn.Linear(d_in, D_MODEL)
        self.temporal = nn.Sequential(
            CausalConv1d(D_MODEL, D_MODEL, KERNEL),
            nn.GELU(),
            CausalConv1d(D_MODEL, D_MODEL, KERNEL, dilation=2),
            nn.GELU(),
        )
        self.head = Head(D_MODEL)

    def forward(self, x, mask_n, mask_t, lengths):
        h = self.pool(self.per_neuron(x, mask_n), mask_n)
        h = self.mix(h) * mask_t.unsqueeze(-1)
        h = self.temporal(h.transpose(1, 2)).transpose(1, 2)
        return self.head(h)


class PooledRNN(nn.Module):
    def __init__(self, cell="gru"):
        super().__init__()
        self.embed = InstantEmbed(D_MODEL)
        self.pool = SetPool(D_MODEL)
        self.mix = nn.Linear(3 * D_MODEL, D_MODEL)
        rnn_cls = nn.GRU if cell == "gru" else nn.LSTM
        self.rnn = rnn_cls(D_MODEL, RNN_HIDDEN, num_layers=1, batch_first=True, dropout=0.0)
        self.head = Head(RNN_HIDDEN)

    def forward(self, x, mask_n, mask_t, lengths):
        h = self.mix(self.pool(self.embed(x), mask_n))
        packed = pack_padded_sequence(
            h, lengths.cpu(), batch_first=True, enforce_sorted=False
        )
        out, _ = self.rnn(packed)
        out, _ = pad_packed_sequence(out, batch_first=True, total_length=x.shape[-1])
        return self.head(out)


class CNNGRU(nn.Module):
    def __init__(self):
        super().__init__()
        self.per_neuron = SharedCausalCNN()
        self.pool = SetPool(self.per_neuron.out_ch)
        self.mix = nn.Linear(3 * self.per_neuron.out_ch, D_MODEL)
        self.rnn = nn.GRU(D_MODEL, RNN_HIDDEN, num_layers=1, batch_first=True)
        self.head = Head(RNN_HIDDEN)

    def forward(self, x, mask_n, mask_t, lengths):
        h = self.mix(self.pool(self.per_neuron(x, mask_n), mask_n))
        packed = pack_padded_sequence(
            h, lengths.cpu(), batch_first=True, enforce_sorted=False
        )
        out, _ = self.rnn(packed)
        out, _ = pad_packed_sequence(out, batch_first=True, total_length=x.shape[-1])
        return self.head(out)


class CausalTransformer(nn.Module):
    """Decoder-only (causal) transformer over pooled population tokens.

    Token t is the permutation-invariant summary of motor spikes in bin t.
    Position t may attend to tokens 0..t. No kinematics are fed back, so
    the model is comparable to the others: spikes → speed, not AR on y.
    """

    def __init__(self):
        super().__init__()
        self.embed = InstantEmbed(D_MODEL)
        self.pool = SetPool(D_MODEL)
        self.mix = nn.Linear(3 * D_MODEL, D_MODEL)
        self.pos = nn.Parameter(torch.zeros(1, MAX_BINS, D_MODEL))
        nn.init.normal_(self.pos, std=0.02)
        layer = nn.TransformerEncoderLayer(
            d_model=D_MODEL,
            nhead=N_HEADS,
            dim_feedforward=FF_DIM,
            dropout=DROPOUT,
            batch_first=True,
            activation="gelu",
        )
        self.enc = nn.TransformerEncoder(layer, num_layers=N_LAYERS)
        self.head = Head(D_MODEL)

    def forward(self, x, mask_n, mask_t, lengths):
        T = x.shape[-1]
        h = self.mix(self.pool(self.embed(x), mask_n))
        h = h + self.pos[:, :T]
        causal = torch.triu(torch.ones(T, T, device=x.device, dtype=torch.bool), 1)
        h = self.enc(h, mask=causal, src_key_padding_mask=~mask_t, is_causal=True)
        return self.head(h)


def build_model(name):
    if name == "cnn":
        return TemporalCNN()
    if name == "gru":
        return PooledRNN("gru")
    if name == "lstm":
        return PooledRNN("lstm")
    if name == "cnn_gru":
        return CNNGRU()
    if name == "transformer":
        return CausalTransformer()
    raise ValueError(name)


def masked_mse(pred, y, mask_t):
    """MSE on z-scored targets; time bins t < W_LAGS are ignored (Ridge warmup)."""
    B, T, _ = pred.shape
    t = torch.arange(T, device=pred.device)
    warm = mask_t & (t >= W_LAGS)
    w = warm.unsqueeze(-1).to(pred.dtype)
    denom = w.sum() * pred.shape[-1]
    denom = denom.clamp(min=1.0)
    return ((pred - y).pow(2) * w).sum() / denom


def run_epoch(model, loader, opt, dev, train=True):
    model.train(train)
    total = 0.0
    n = 0
    for batch in loader:
        x = batch["x"].to(dev)
        y = batch["y"].to(dev)
        mask_t = batch["mask_t"].to(dev)
        mask_n = batch["mask_n"].to(dev)
        lengths = batch["lengths"]
        if train:
            opt.zero_grad(set_to_none=True)
        pred = model(x, mask_n, mask_t, lengths)
        loss = masked_mse(pred, y, mask_t)
        if train:
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
        total += float(loss.item()) * x.size(0)
        n += x.size(0)
    return total / max(n, 1)


def cache_dir(task, model):
    d = CACHE / task / model
    d.mkdir(parents=True, exist_ok=True)
    return d


def save_json(path, obj):
    Path(path).write_text(json.dumps(obj, indent=2, default=float))


def train_torch(name, trials, train_idx, val_idx, y_mean, y_std, task):
    cdir = cache_dir(task, name)
    ckpt = cdir / "model.pt"
    hist_path = cdir / "history.json"
    if ckpt.exists() and hist_path.exists():
        log(f"  cache hit {task}/{name}")
        blob = torch.load(ckpt, map_location="cpu", weights_only=False)
        hist = json.loads(hist_path.read_text())
        return blob, hist

    dev = device_of()
    model = build_model(name).to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    tr_ds = TrialDS(trials, train_idx, y_mean, y_std)
    va_ds = TrialDS(trials, val_idx, y_mean, y_std)
    tr_ld = DataLoader(tr_ds, batch_size=BATCH, shuffle=True, collate_fn=collate)
    va_ld = DataLoader(va_ds, batch_size=BATCH, shuffle=False, collate_fn=collate)
    best_val = math.inf
    best_state = None
    bad = 0
    hist = {"train": [], "val": [], "best_epoch": 0}
    log(f"  train {name} on {dev}  n_tr={len(tr_ds)} n_val={len(va_ds)}")
    for epoch in range(1, MAX_EPOCHS + 1):
        tr_loss = run_epoch(model, tr_ld, opt, dev, train=True)
        with torch.no_grad():
            va_loss = run_epoch(model, va_ld, opt, dev, train=False)
        hist["train"].append(tr_loss)
        hist["val"].append(va_loss)
        log(f"    {name} epoch {epoch:02d}  train={tr_loss:.4f}  val={va_loss:.4f}")
        if va_loss + 1e-5 < best_val:
            best_val = va_loss
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            hist["best_epoch"] = epoch
            bad = 0
        else:
            bad += 1
        slurm_utils.write_status(
            state="training",
            epoch=epoch,
            train=tr_loss,
            val=va_loss,
            best_epoch=hist["best_epoch"],
            best_val=best_val,
        )
        if bad >= PATIENCE:
            log(f"    early stop at epoch {epoch}")
            break
    blob = {
        "state_dict": best_state,
        "y_mean": y_mean,
        "y_std": y_std,
        "name": name,
        "best_val": best_val,
    }
    torch.save(blob, ckpt)
    save_json(hist_path, hist)
    return blob, hist


def predict_torch(blob, trials, idx, name):
    dev = device_of()
    model = build_model(name).to(dev)
    model.load_state_dict(blob["state_dict"])
    model.eval()
    ds = TrialDS(trials, idx, blob["y_mean"], blob["y_std"])
    ld = DataLoader(ds, batch_size=BATCH, shuffle=False, collate_fn=collate)
    recs = []
    cursor = 0
    with torch.no_grad():
        for batch in ld:
            x = batch["x"].to(dev)
            mask_t = batch["mask_t"].to(dev)
            mask_n = batch["mask_n"].to(dev)
            pred = model(x, mask_n, mask_t, batch["lengths"]).cpu().numpy()
            y_std = batch["y"].numpy()
            mt = batch["mask_t"].numpy()
            for b in range(pred.shape[0]):
                T = int(mt[b].sum())
                t0 = W_LAGS
                p = pred[b, t0:T] * blob["y_std"] + blob["y_mean"]
                y = y_std[b, t0:T] * blob["y_std"] + blob["y_mean"]
                recs.append({"pred": p.astype(np.float32), "y": y.astype(np.float32)})
                cursor += 1
    return recs


def pop_stats(spikes):
    """Permutation-invariant per-bin population features (T, 4)."""
    mean = spikes.mean(axis=1)
    std = spikes.std(axis=1)
    mx = spikes.max(axis=1)
    frac = (spikes > 1e-8).mean(axis=1)
    return np.stack([mean, std, mx, frac], axis=1).astype(np.float32)


def lag_features(stats, w=W_LAGS):
    feats = []
    for k in range(w + 1):
        rolled = np.roll(stats, k, axis=0)
        if k:
            rolled[:k] = 0.0
        feats.append(rolled)
    return np.concatenate(feats, axis=1)[w:]


def flatten_ridge(trials, idx):
    Xs, Ys, bounds = [], [], []
    start = 0
    for i in idx:
        stats = pop_stats(trials[i]["spikes"])
        X = lag_features(stats)
        y = trials[i]["y"][W_LAGS : W_LAGS + len(X)]
        n = min(len(X), len(y))
        Xs.append(X[:n])
        Ys.append(y[:n])
        bounds.append((int(i), start, start + n))
        start += n
    return np.vstack(Xs), np.vstack(Ys), bounds


def train_ridge(trials, train_idx, val_idx, task):
    cdir = cache_dir(task, "ridge")
    ckpt = cdir / "model.pkl"
    hist_path = cdir / "history.json"
    if ckpt.exists() and hist_path.exists():
        log("  cache hit ridge")
        with ckpt.open("rb") as fh:
            blob = pickle.load(fh)
        hist = json.loads(hist_path.read_text())
        return blob, hist
    Xtr, ytr, _ = flatten_ridge(trials, train_idx)
    Xva, yva, _ = flatten_ridge(trials, val_idx)
    xsc = StandardScaler().fit(Xtr)
    ysc = StandardScaler().fit(ytr)
    Xtr_s, Xva_s = xsc.transform(Xtr), xsc.transform(Xva)
    ytr_s, yva_s = ysc.transform(ytr), ysc.transform(yva)
    hist = {"train": [], "val": [], "alphas": ALPHAS.tolist(), "best_epoch": 0}
    best = None
    best_val = math.inf
    for i, alpha in enumerate(ALPHAS, start=1):
        model = Ridge(alpha=float(alpha), fit_intercept=True)
        model.fit(Xtr_s, ytr_s)
        tr = float(np.mean((model.predict(Xtr_s) - ytr_s) ** 2))
        va = float(np.mean((model.predict(Xva_s) - yva_s) ** 2))
        hist["train"].append(tr)
        hist["val"].append(va)
        log(f"    ridge alpha={alpha:g}  train={tr:.4f}  val={va:.4f}")
        if va < best_val:
            best_val = va
            best = (alpha, model)
            hist["best_epoch"] = i
    alpha, _ = best
    Xall = np.vstack([Xtr, Xva])
    yall = np.vstack([ytr, yva])
    xsc = StandardScaler().fit(Xall)
    ysc = StandardScaler().fit(yall)
    model = Ridge(alpha=float(alpha), fit_intercept=True)
    model.fit(xsc.transform(Xall), ysc.transform(yall))
    blob = {"model": model, "xsc": xsc, "ysc": ysc, "alpha": float(alpha)}
    with ckpt.open("wb") as fh:
        pickle.dump(blob, fh)
    save_json(hist_path, hist)
    return blob, hist


def predict_ridge(blob, trials, idx):
    X, y, bounds = flatten_ridge(trials, idx)
    pred = blob["ysc"].inverse_transform(blob["model"].predict(blob["xsc"].transform(X)))
    recs = []
    for _i, a, b in bounds:
        recs.append({"pred": pred[a:b].astype(np.float32), "y": y[a:b].astype(np.float32)})
    return recs


def score_recs(recs):
    y = np.concatenate([r["y"] for r in recs], axis=0)
    p = np.concatenate([r["pred"] for r in recs], axis=0)
    out = {}
    for j, name in enumerate(TARGETS):
        yt, yp = y[:, j], p[:, j]
        m = np.isfinite(yt) & np.isfinite(yp)
        yt, yp = yt[m], yp[m]
        out[name] = {
            "r2": float(r2_score(yt, yp)) if m.sum() >= 10 else float("nan"),
            "rmse": float(np.sqrt(np.mean((yt - yp) ** 2))) if m.sum() else float("nan"),
            "mae": float(mean_absolute_error(yt, yp)) if m.sum() else float("nan"),
            "pearson": float(np.corrcoef(yt, yp)[0, 1]) if m.sum() >= 10 else float("nan"),
            "n_bins": int(m.sum()),
        }
    trial_r2 = {t: [] for t in TARGETS}
    for r in recs:
        for j, name in enumerate(TARGETS):
            yt, yp = r["y"][:, j], r["pred"][:, j]
            m = np.isfinite(yt) & np.isfinite(yp)
            if m.sum() < 8:
                continue
            trial_r2[name].append(float(r2_score(yt[m], yp[m])))
    for name in TARGETS:
        arr = np.asarray(trial_r2[name], dtype=float)
        out[name]["median_trial_r2"] = float(np.nanmedian(arr)) if arr.size else float("nan")
        out[name]["n_trials"] = int(arr.size)
    return out, recs


def style_plots():
    plt.rcParams.update(
        {
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "axes.spines.top": False,
            "axes.spines.right": False,
            "font.size": 11,
        }
    )


def plot_training(histories, out_dir):
    style_plots()
    for task in TASKS:
        fig, axes = plt.subplots(2, 3, figsize=(11.2, 6.4), sharex=False)
        for ax, name in zip(axes.ravel(), MODELS):
            h = histories[task][name]
            xs = np.arange(1, len(h["train"]) + 1)
            ax.plot(xs, h["train"], color="#334155", lw=1.4, label="train")
            ax.plot(xs, h["val"], color="#C47B3B", lw=1.4, label="val")
            ax.axvline(h.get("best_epoch", 1), color="0.7", ls="--", lw=0.8)
            ax.set_title(name if name != "ridge" else "ridge (α grid)")
            ax.set_ylabel("MSE (z-scored targets)")
            ax.set_xlabel("epoch" if name != "ridge" else "α index")
            ax.legend(frameon=False, fontsize=8)
        fig.suptitle(f"Training curves · {task}")
        fig.tight_layout()
        fig.savefig(out_dir / f"train_{task}.png", dpi=150)
        plt.close(fig)


def plot_r2_bars(rows, out_dir):
    style_plots()
    df = pd.DataFrame(rows)
    colors = {
        "cnn": "#4C6A92",
        "gru": "#C47B3B",
        "lstm": "#7A5C8A",
        "cnn_gru": "#3B7A57",
        "transformer": "#B4554A",
        "ridge": "#6B7280",
    }
    for target in TARGETS:
        fig, ax = plt.subplots(figsize=(9.6, 4.6))
        x = np.arange(len(TASKS))
        width = 0.13
        for i, name in enumerate(MODELS):
            vals = []
            for task in TASKS:
                sub = df[(df.task == task) & (df.model == name) & (df.target == target)]
                vals.append(float(sub.r2.iloc[0]) if len(sub) else np.nan)
            ax.bar(x + (i - 2.5) * width, vals, width, color=colors[name], label=name)
        ax.axhline(0.0, color="0.6", lw=0.8)
        ax.set_xticks(x)
        ax.set_xticklabels(TASKS)
        ax.set_ylabel("held-out R² (concatenated bins)")
        ax.set_title(TARGET_LABELS[target])
        ax.legend(frameon=False, ncol=3, fontsize=8)
        fig.tight_layout()
        fig.savefig(out_dir / f"r2_{target}.png", dpi=150)
        plt.close(fig)


def plot_examples(example_bank, out_dir):
    style_plots()
    colors = {
        "cnn": "#4C6A92",
        "gru": "#C47B3B",
        "lstm": "#7A5C8A",
        "cnn_gru": "#3B7A57",
        "transformer": "#B4554A",
        "ridge": "#6B7280",
        "true": "#111827",
    }
    for task, pack in example_bank.items():
        if not pack:
            continue
        n = min(2, len(pack["true"]))
        fig, axes = plt.subplots(n * 2, 1, figsize=(9.2, 2.2 * n * 2), sharex=False)
        axes = np.atleast_1d(axes)
        k = 0
        for j in range(n):
            t = np.arange(len(pack["true"][j])) * BINSIZE
            for ti, ylab in enumerate(["|ω| (rad/s)", "px/s (left-cam equiv.)"]):
                ax = axes[k]
                ax.plot(t, pack["true"][j][:, ti], color=colors["true"], lw=1.5, label="actual")
                for name in MODELS:
                    if name in pack and j < len(pack[name]):
                        ax.plot(t, pack[name][j][:, ti], color=colors[name], lw=1.1, alpha=0.85, label=name)
                ax.set_ylabel(ylab)
                ax.set_title(f"{task}  example trial {j + 1}  {TARGETS[ti]}")
                if k == 0:
                    ax.legend(frameon=False, ncol=4, fontsize=8)
                k += 1
        axes[-1].set_xlabel("time after lag crop (s)")
        fig.tight_layout()
        fig.savefig(out_dir / f"examples_{task}.png", dpi=150)
        plt.close(fig)


def write_report(splits, rows, histories, session_meta, path):
    df = pd.DataFrame(rows)
    lines = []
    lines.append("# MLtestcontra — motor-cortex decoding on ModelDataRightContra")
    lines.append("")
    lines.append(f"Built {datetime.now(timezone.utc).isoformat()}")
    lines.append("")
    lines.append("## Dataset and the neuron-identity problem")
    lines.append("")
    lines.append(
        "ModelDataRightContra is right-paw / right-hemisphere BWM trials. "
        "Only MOp/MOs units enter the models. Sessions still differ in how many "
        "motor units they have (5–226) and those units are *different cells* — "
        "column 0 of `spike_counts` in session A is not the same neuron as "
        "column 0 in session B. Concatenating raw (T, N) tensors across sessions "
        "would pretend otherwise. Every model here is permutation-invariant over "
        "the neuron axis: a shared encoder runs independently on each unit, then "
        "a masked mean / max / attention pool builds a population token per time bin."
    )
    lines.append("")
    lines.append(
        f"Trials shorter than {MIN_BINS} bins after motor filtering are dropped. "
        f"Sequences longer than {MAX_BINS} bins (2.56 s) are cropped from stimOn; "
        "that truncates <1% of raw RightContra trials. Spike counts are "
        "`log1p`. Wheel target is `|ω|` (rad/s). Paw target is Lightning Pose 2D "
        "speed, falling back to DLC."
    )
    lines.append("")
    lines.append("## Holdout tasks")
    lines.append("")
    for task in TASKS:
        sp = splits[task]
        lines.append(f"- **{task}**: {sp['note']}. train n={len(sp['train'])}, test n={len(sp['test'])}.")
    lines.append("")
    lines.append(
        "Trial holdout tests within-session decoding (neurons *are* the same cells "
        "on train and test, but the model is still not allowed to use a fixed index). "
        "Session holdout tests a new probe insertion in a mouse the model has seen. "
        "Mouse holdout tests a new animal."
    )
    lines.append("")
    lines.append("## Architectures")
    lines.append("")
    lines.append("### Shared pooling")
    lines.append("")
    lines.append(
        "Let `x` be `(B, N, T)` log-counts with neuron mask `m`. A shared encoder "
        "produces `h ∈ R^{B×N×T×D}`. SetPool returns "
        "`[masked_mean(h); masked_max(h); attention_pool(h)] ∈ R^{B×T×3D}`. "
        "Attention scores are `softmax_N(W h)`, with masked neurons set to −∞. "
        "Because encoder weights do not depend on the neuron index, any permutation "
        "of columns of `x` leaves the pooled sequence unchanged."
    )
    lines.append("")
    lines.append("### 1. Temporal CNN (`cnn`)")
    lines.append("")
    lines.append(
        f"Per-neuron causal Conv1d stack, channels {CNN_CH}, kernel {KERNEL}, "
        "left-padded so output time t uses bins ≤ t. Pool, Linear to "
        f"{D_MODEL}, then two more causal convs (dilation 1 then 2). MLP head → 2."
    )
    lines.append("")
    lines.append("### 2. GRU (`gru`)")
    lines.append("")
    lines.append(
        f"Instantaneous Linear(1→{D_MODEL}) embed per neuron, pool, Linear to "
        f"{D_MODEL}, then a 1-layer GRU (hidden {RNN_HIDDEN}) on the packed "
        "variable-length sequence. Hidden state at each t → MLP head."
    )
    lines.append("")
    lines.append("### 3. LSTM (`lstm`)")
    lines.append("")
    lines.append("Identical to GRU with `nn.LSTM` in place of `nn.GRU`.")
    lines.append("")
    lines.append("### 4. CNN–RNN (`cnn_gru`)")
    lines.append("")
    lines.append(
        "Same per-neuron causal CNN as (1), then pool, then the GRU of (2). "
        "CNN supplies local temporal features per cell; GRU integrates them."
    )
    lines.append("")
    lines.append("### 5. Causal autoregressive transformer (`transformer`)")
    lines.append("")
    lines.append(
        f"Instantaneous embed + pool → {D_MODEL}-d tokens, plus a learned "
        f"positional table of length {MAX_BINS}. {N_LAYERS} TransformerEncoder "
        f"layers, {N_HEADS} heads, FFN {FF_DIM}, dropout {DROPOUT}. A strictly "
        "upper-triangular attention mask makes the stack decoder-only: token t "
        "attends to 0…t (current bin included, matching BWM lag 0). Kinematics "
        "are **not** fed back, so test-time predictions come from spikes alone. "
        "That is autoregressive over neural tokens, not over past |ω|."
    )
    lines.append("")
    lines.append("### 6. Ridge, fixed sliding window (`ridge`)")
    lines.append("")
    lines.append(
        f"Each bin is reduced to four set-statistics: population mean, std, max, "
        f"and fraction of neurons with a spike. A causal window of W={W_LAGS} "
        "lags (11 bins × 4 = 44 features) predicts the two speeds at time t from "
        "bins t−W…t. First W bins are dropped. Features and targets are "
        "StandardScaled on the training bins. α is chosen on the validation "
        f"split from {ALPHAS.tolist()}, then Ridge is refit on train+val. This is "
        "the linear analogue of early stopping: complexity is selected on held-out "
        "MSE, not on train fit."
    )
    lines.append("")
    lines.append("## Training, loss, and error")
    lines.append("")
    lines.append(
        "Deep models minimize **masked MSE on z-scored targets**. Train-set bin "
        "means and stds (both speeds, after the W-bin crop) standardize y so "
        "rad/s and px/s contribute on the same scale:"
    )
    lines.append("")
    lines.append("    L = mean_{t ≥ W, valid}  Σ_{k=1,2} (ŷ_{t,k} − ỹ_{t,k})²")
    lines.append("")
    lines.append(
        f"AdamW (lr={LR}, weight decay={WEIGHT_DECAY}), batch {BATCH}, grad clip 1.0, "
        f"max {MAX_EPOCHS} epochs, early stop when val MSE does not improve for "
        f"{PATIENCE} epochs. Best checkpoint (lowest val MSE) is cached under "
        "`BehaviorRes/MLtestcontra/cache/<task>/<model>/` and reused."
    )
    lines.append("")
    lines.append(
        "Reported error is **not** that training loss. After inverting the y "
        "standardization we compute, on concatenated test bins with t ≥ W:"
    )
    lines.append("")
    lines.append("- R² = 1 − Σ(y−ŷ)² / Σ(y−ȳ_test)²  (sklearn, original units)")
    lines.append("- RMSE and MAE in rad/s or px/s")
    lines.append("- Pearson r")
    lines.append("- median per-trial R² (trials with ≥ 8 scored bins)")
    lines.append("")
    lines.append("## Results")
    lines.append("")
    lines.append("| task | model | target | R² | RMSE | MAE | Pearson | median trial R² | n bins | n trials |")
    lines.append("|---|---|---|---:|---:|---:|---:|---:|---:|---:|")
    for r in rows:
        lines.append(
            f"| {r['task']} | {r['model']} | {r['target']} | "
            f"{r['r2']:.3f} | {r['rmse']:.3f} | {r['mae']:.3f} | "
            f"{r['pearson']:.3f} | {r['median_trial_r2']:.3f} | "
            f"{r['n_bins']} | {r['n_trials']} |"
        )
    lines.append("")
    lines.append("### Best model per task × target (test R²)")
    lines.append("")
    for task in TASKS:
        for target in TARGETS:
            sub = df[(df.task == task) & (df.target == target)]
            if sub.empty:
                continue
            best = sub.loc[sub.r2.idxmax()]
            lines.append(
                f"- {task} / {target}: **{best.model}** R²={best.r2:.3f} "
                f"(ridge={float(sub.loc[sub.model=='ridge','r2'].iloc[0]):.3f})"
            )
    lines.append("")
    lines.append("## Training diagnostics")
    lines.append("")
    for task in TASKS:
        lines.append(f"### {task}")
        lines.append("")
        for name in MODELS:
            h = histories[task][name]
            be = h.get("best_epoch", 0)
            vtr = h["train"][be - 1] if be and be <= len(h["train"]) else float("nan")
            vva = h["val"][be - 1] if be and be <= len(h["val"]) else float("nan")
            lines.append(
                f"- {name}: best @ {be}/{len(h['train'])}  "
                f"train MSE={vtr:.4f}  val MSE={vva:.4f}"
            )
        lines.append("")
    lines.append("Plots: `train_<task>.png`, `r2_wheel_speed.png`, `r2_paw_speed.png`, `examples_<task>.png`.")
    lines.append("")
    Path(path).write_text("\n".join(lines))


def pick_examples(trials, test_idx, preds_by_model, k=2):
    lengths = [(len(trials[i]["y"]) - W_LAGS, j) for j, i in enumerate(test_idx)]
    lengths = [p for p in lengths if p[0] >= 12]
    lengths.sort(reverse=True)
    chosen = [j for _, j in lengths[:k]]
    pack = {"true": []}
    for name in MODELS:
        pack[name] = []
    for j in chosen:
        pack["true"].append(preds_by_model[MODELS[0]][j]["y"])
        for name in MODELS:
            pack[name].append(preds_by_model[name][j]["pred"])
    return pack


def process(models=MODELS, tasks=TASKS, make_plots=True, job_index=None):
    warnings.filterwarnings("ignore")
    set_seed(SEED)
    OUT.mkdir(parents=True, exist_ok=True)
    CACHE.mkdir(parents=True, exist_ok=True)
    log(f"device={device_of()}")
    trials, session_meta = load_corpus()
    splits = make_splits(trials)
    save_json(OUT / "splits.json", {t: {"note": splits[t]["note"], "n_train": int(len(splits[t]["train"])), "n_test": int(len(splits[t]["test"])), "hold_eids": splits[t]["hold_eids"]} for t in TASKS})

    rows = []
    histories = {t: {} for t in TASKS}
    example_bank = {}
    n_job = 0
    for task in tasks:
        log(f"==== {task} ====")
        train_full = splits[task]["train"]
        test_idx = splits[task]["test"]
        tr_idx, va_idx = train_val_split(train_full)
        y_mean, y_std = y_scaler_from(trials, tr_idx)
        preds_by_model = {}
        for name in models:
            log(f"-- {task} / {name}")
            tag = f"{task}_{name}"
            idx = job_index if job_index is not None else n_job
            jdir = slurm_utils.begin_job(OUT, idx, tag, task=task, model=name)
            slurm_utils.write_status(state="fitting" if name == "ridge" else "training")
            if name == "ridge":
                blob, hist = train_ridge(trials, tr_idx, va_idx, task)
                recs = predict_ridge(blob, trials, test_idx)
            else:
                blob, hist = train_torch(name, trials, tr_idx, va_idx, y_mean, y_std, task)
                recs = predict_torch(blob, trials, test_idx, name)
            histories[task][name] = hist
            preds_by_model[name] = recs
            scores, _ = score_recs(recs)
            these = []
            for target, sc in scores.items():
                row = {"task": task, "model": name, "target": target, **sc}
                rows.append(row)
                these.append(row)
                log(
                    f"   {target:12s} R²={sc['r2']:+.3f}  RMSE={sc['rmse']:.3f}  "
                    f"r={sc['pearson']:.3f}  n={sc['n_bins']}"
                )
            pd.DataFrame(these).to_csv(jdir / "scores.csv", index=False)
            slurm_utils.write_status(state="done", n_rows=len(these), best_epoch=hist.get("best_epoch"))
            n_job += 1
            if make_plots:
                pd.DataFrame(rows).to_csv(OUT / "scores.csv", index=False)
        if make_plots:
            example_bank[task] = pick_examples(trials, test_idx, preds_by_model)

    if make_plots:
        plot_training(histories, OUT)
        plot_r2_bars(rows, OUT)
        plot_examples(example_bank, OUT)
        write_report(splits, rows, histories, session_meta, OUT / "REPORT.md")
        save_json(OUT / "scores.json", rows)
        log("wrote " + str(OUT))
        print(pd.DataFrame(rows).to_string(index=False))
    else:
        log("wrote job dir " + str(slurm_utils.job_dir() or OUT / "jobs"))
        if rows:
            print(pd.DataFrame(rows).to_string(index=False))
    return rows


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--models", nargs="*", default=list(MODELS))
    p.add_argument("--tasks", nargs="*", default=list(TASKS))
    slurm_utils.add_common_args(p)
    args = p.parse_args()
    if args.list_jobs:
        slurm_utils.print_jobs(job_grid())
        raise SystemExit(0)
    slurm_utils.set_device(args.device)
    if args.aggregate:
        slurm_utils.aggregate_scores(OUT)
        raise SystemExit(0)
    job = slurm_utils.resolve_job_index(args.job)
    if job is None:
        process(models=tuple(args.models), tasks=tuple(args.tasks))
    else:
        cfg = slurm_utils.pick_config(job_grid(), job)
        process(
            models=(cfg["model"],),
            tasks=(cfg["task"],),
            make_plots=False,
            job_index=job,
        )
