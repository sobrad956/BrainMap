"""Baselines for Modelv1 / Formulation A on ModelDataRightContra.

Extends Modelv1.py: same corpus, holdouts, z-scored trial-balanced MSE,
and evaluation. Implements PDF §6 baselines 0, 1, 2, 3, 6, 7, 9, 10, plus
a Brain-Wide Map lagged linear decoder (session-specific, W=10, Ridge)
adapted to this RightContra motor-cortex task.
"""

from __future__ import annotations

import argparse
import json
import math
import pickle
import sys
import warnings
from datetime import datetime, timezone
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.linear_model import Ridge, RidgeCV
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from torch.utils.data import DataLoader, Dataset

sys.path.insert(0, str(Path(__file__).resolve().parent))
import Modelv1 as mv
import slurm_utils

OUT = mv.ROOT / "Baselinev1"
CACHE = OUT / "cache"
MODEL_ROOT = mv.ROOT / "Baselinev1"


def configure_out(target):
    global OUT, CACHE
    if target not in mv.TARGETS:
        raise SystemExit(f"unknown target {target!r}; choose from {mv.TARGETS}")
    OUT = MODEL_ROOT / target
    CACHE = OUT / "cache"
    return OUT

W_WINDOW = 10  # PDF B2 width; BWM n_bins_lag
ALPHAS = np.array([0.1, 1.0, 10.0, 100.0, 1000.0, 10000.0])
N_FOLDS = 5
MIN_SESSION_TRIALS = 8

CROSS_MODELS = ("b0_const", "b0_traj", "b1", "b2", "b3", "b6", "b7")
WITHIN_MODELS = ("b9", "b10", "bwm")
ALL_MODELS = CROSS_MODELS + WITHIN_MODELS

MODEL_LABELS = {
    "b0_const": "B0 mean",
    "b0_traj": "B0 traj",
    "b1": "B1 pop ridge",
    "b2": "B2 pop window",
    "b3": "B3 region ridge",
    "b6": "B6 pop GRU",
    "b7": "B7 region GRU",
    "b9": "B9 sess ridge",
    "b10": "B10 sess GRU",
    "bwm": "BWM lag ridge",
    "A_mean": "A mean pool",
    "A_attn": "A attn pool",
    "A_mean_a1": "A1 mean",
    "A_attn_a1": "A1 attn",
}

MODEL_COLORS = {
    "b0_const": "#9CA3AF",
    "b0_traj": "#6B7280",
    "b1": "#86B087",
    "b2": "#4F8A6B",
    "b3": "#2F6B4F",
    "b6": "#7BA3C9",
    "b7": "#4C6A92",
    "b9": "#E0A06B",
    "b10": "#C47B3B",
    "bwm": "#8B6BB0",
    "A_mean": "#1E3A5F",
    "A_attn": "#111827",
    "A_mean_a1": "#94A3B8",
    "A_attn_a1": "#CBD5E1",
}


def job_grid():
    return [
        {"target": beh, "task": t, "model": m}
        for beh in mv.TARGETS
        for t in mv.TASKS
        for m in ALL_MODELS
    ]


def log(msg):
    mv.log(msg)


def _open_job(task, model, job_index, n_job, target):
    idx = job_index if job_index is not None else n_job
    tag = f"{target}_{task}_{model}"
    jdir = slurm_utils.begin_job(OUT, idx, tag, task=task, model=model, target=target)
    return jdir


def cache_dir(*parts):
    d = CACHE.joinpath(*[str(p) for p in parts])
    d.mkdir(parents=True, exist_ok=True)
    return d


def save_json(path, obj):
    Path(path).write_text(json.dumps(obj, indent=2, default=float))


def dump_pkl(path, obj):
    Path(path).write_bytes(pickle.dumps(obj, protocol=pickle.HIGHEST_PROTOCOL))


def load_pkl(path):
    return pickle.loads(Path(path).read_bytes())


def region_list(trials, idx):
    seen = set()
    for i in idx:
        seen.update(str(a) for a in trials[int(i)]["areas"])
    return sorted(seen)


def pop_rate(spikes):
    return spikes.mean(axis=1, keepdims=True).astype(np.float32)


def region_features(spikes, areas, vocab):
    """PDF eq. 81: [g_r; M_r; log(1+C_r)] at each bin."""
    T = spikes.shape[0]
    R = len(vocab)
    index = {a: r for r, a in enumerate(vocab)}
    g = np.zeros((T, R), dtype=np.float32)
    c = np.zeros(R, dtype=np.float32)
    for n, a in enumerate(areas):
        r = index.get(str(a))
        if r is None:
            continue
        g[:, r] += spikes[:, n]
        c[r] += 1.0
    m = (c > 0).astype(np.float32)
    for r in range(R):
        if c[r] > 0:
            g[:, r] /= c[r]
    logc = np.log1p(c).astype(np.float32)
    return np.concatenate(
        [g, np.broadcast_to(m, (T, R)), np.broadcast_to(logc, (T, R))],
        axis=1,
    ).astype(np.float32)


def causal_window(x, w=W_WINDOW):
    """PDF eq. 75: [x_{t-W+1}, ..., x_t], dropping the first W-1 bins."""
    T, F = x.shape
    if T < w:
        return np.zeros((0, w * F), dtype=np.float32)
    out = np.zeros((T - w + 1, w * F), dtype=np.float32)
    for k in range(w):
        out[:, k * F : (k + 1) * F] = x[k : k + T - w + 1]
    return out


def bwm_lag_matrix(spikes, n_lags=W_WINDOW):
    """Paper build_predictor_matrix: hstack of rolls 0..n_lags, crop first n_lags."""
    mat = np.hstack([np.roll(spikes, i, axis=0) for i in range(n_lags + 1)])
    return mat[n_lags:].astype(np.float32)


def align_xy(x, y):
    """If features are a causal window, they line up with the trailing bins of y."""
    x = np.asarray(x, dtype=np.float32)
    y = np.asarray(y, dtype=np.float32)
    n = min(x.shape[0], y.shape[0])
    return x[-n:], y[-n:]


def stack_xy(trials, idx, feat_fn, crop=0, drop_invalid=True):
    xs, ys = [], []
    for i in idx:
        t = trials[int(i)]
        x, y = align_xy(feat_fn(t), t["y"])
        if crop:
            if x.shape[0] <= crop:
                continue
            x, y = x[crop:], y[crop:]
        y = np.asarray(y, dtype=np.float32).reshape(y.shape[0], -1)
        if drop_invalid:
            valid = np.isfinite(y).all(axis=1)
            if not valid.any():
                continue
            x, y = x[valid], y[valid]
        if x.shape[0] < 3:
            continue
        xs.append(x)
        ys.append(y)
    if not xs:
        return None, None
    return np.vstack(xs).astype(np.float32), np.vstack(ys).astype(np.float32)


def fit_ridge_val(Xtr, ytr, Xva, yva, Xte):
    scaler = StandardScaler()
    Xtr_s = scaler.fit_transform(Xtr)
    Xva_s = scaler.transform(Xva)
    best_a, best_mse = float(ALPHAS[0]), math.inf
    for a in ALPHAS:
        model = Ridge(alpha=float(a), fit_intercept=True)
        model.fit(Xtr_s, ytr)
        pred = model.predict(Xva_s)
        mse = float(np.mean((pred - yva) ** 2))
        if mse < best_mse:
            best_a, best_mse = float(a), mse
    Xtv = np.vstack([Xtr, Xva])
    ytv = np.vstack([ytr, yva]) if ytr.ndim == 2 else np.concatenate([ytr, yva])
    scaler = StandardScaler()
    model = Ridge(alpha=best_a, fit_intercept=True)
    model.fit(scaler.fit_transform(Xtv), ytv)
    pred = model.predict(scaler.transform(Xte))
    return pred, {"alpha": best_a, "val_mse": best_mse, "scaler": scaler, "model": model}


def fit_ridgecv(Xtr, ytr, Xte):
    scaler = StandardScaler()
    Xtr_s = scaler.fit_transform(Xtr)
    model = RidgeCV(alphas=ALPHAS)
    model.fit(Xtr_s, ytr)
    pred = model.predict(scaler.transform(Xte))
    return pred, {"alpha": float(model.alpha_), "scaler": scaler, "model": model}


def recs_from_map(trials, test_idx, pred_map, crop=0):
    recs = []
    for i in test_idx:
        y = trials[int(i)]["y"].astype(np.float32)
        p = pred_map.get(int(i))
        if p is None:
            recs.append({"pred": np.full_like(y, np.nan), "y": y})
            continue
        p = np.asarray(p, dtype=np.float32)
        p, y = align_xy(p, y)
        if crop:
            p, y = p[crop:], y[crop:]
        recs.append({"pred": p, "y": y})
    return recs


class SeqDS(Dataset):
    def __init__(self, feats, ys, y_mean, y_std):
        self.feats = feats
        self.ys = ys
        self.y_mean = y_mean
        self.y_std = y_std

    def __len__(self):
        return len(self.feats)

    def __getitem__(self, j):
        y = np.asarray(self.ys[j], dtype=np.float32).reshape(self.ys[j].shape[0], -1)
        valid = np.isfinite(y).all(axis=1)
        y = np.where(valid[:, None], (y - self.y_mean) / self.y_std, 0.0)
        return self.feats[j], y.astype(np.float32), valid


def collate_seq(batch):
    xs, ys, valids = zip(*batch)
    B = len(batch)
    T = max(x.shape[0] for x in xs)
    F = xs[0].shape[1]
    y_dim = int(np.asarray(ys[0]).reshape(ys[0].shape[0], -1).shape[-1])
    x = torch.zeros(B, T, F, dtype=torch.float32)
    y = torch.zeros(B, T, y_dim, dtype=torch.float32)
    mask_t = torch.zeros(B, T, dtype=torch.bool)
    lengths = torch.zeros(B, dtype=torch.long)
    for i, (xi, yi, vi) in enumerate(zip(xs, ys, valids)):
        tt = xi.shape[0]
        x[i, :tt] = torch.from_numpy(np.asarray(xi, dtype=np.float32))
        y[i, :tt] = torch.from_numpy(np.asarray(yi, dtype=np.float32).reshape(tt, -1))
        mask_t[i, :tt] = torch.from_numpy(np.asarray(vi, dtype=bool))
        lengths[i] = tt
    return {"x": x, "y": y, "mask_t": mask_t, "lengths": lengths}


class SpikesDS(Dataset):
    def __init__(self, trials, idx, y_mean, y_std):
        self.trials = trials
        self.idx = np.asarray(idx)
        self.y_mean = y_mean
        self.y_std = y_std

    def __len__(self):
        return len(self.idx)

    def __getitem__(self, j):
        t = self.trials[int(self.idx[j])]
        y = np.asarray(t["y"], dtype=np.float32).reshape(t["y"].shape[0], -1)
        valid = t.get("y_valid")
        if valid is None:
            valid = np.isfinite(y).all(axis=1)
        valid = np.asarray(valid, dtype=bool)
        y = np.where(valid[:, None], (y - self.y_mean) / self.y_std, 0.0)
        return t["spikes"], y.astype(np.float32), valid


def collate_spikes(batch):
    spikes, ys, valids = zip(*batch)
    B = len(batch)
    T = max(s.shape[0] for s in spikes)
    N = max(s.shape[1] for s in spikes)
    y_dim = int(np.asarray(ys[0]).reshape(ys[0].shape[0], -1).shape[-1])
    x = torch.zeros(B, N, T, dtype=torch.float32)
    y = torch.zeros(B, T, y_dim, dtype=torch.float32)
    mask_t = torch.zeros(B, T, dtype=torch.bool)
    mask_n = torch.zeros(B, N, dtype=torch.bool)
    lengths = torch.zeros(B, dtype=torch.long)
    for i, (s, yi, vi) in enumerate(zip(spikes, ys, valids)):
        n, tt = s.shape[1], s.shape[0]
        x[i, :n, :tt] = torch.from_numpy(s.T)
        y[i, :tt] = torch.from_numpy(np.asarray(yi, dtype=np.float32).reshape(tt, -1))
        mask_t[i, :tt] = torch.from_numpy(np.asarray(vi, dtype=bool))
        mask_n[i, :n] = True
        lengths[i] = tt
    return {"x": x, "y": y, "mask_t": mask_t, "mask_n": mask_n, "lengths": lengths}


class SequenceGRU(nn.Module):
    """Matched temporal backbone: Linear → GRU(64) → Modelv1 Head."""

    def __init__(self, in_dim, d=mv.D_MODEL, h=mv.RNN_HIDDEN):
        super().__init__()
        self.in_proj = nn.Linear(in_dim, d)
        self.rnn = nn.GRU(d, h, num_layers=1, batch_first=True)
        self.head = mv.Head(h)

    def forward(self, x, mask_t, lengths):
        h = self.in_proj(x) * mask_t.unsqueeze(-1)
        packed = pack_padded_sequence(h, lengths.cpu(), batch_first=True, enforce_sorted=False)
        out, _ = self.rnn(packed)
        out, _ = pad_packed_sequence(out, batch_first=True, total_length=x.shape[1])
        return self.head(out)


class SessionGRU(nn.Module):
    """Baseline 10: session-specific encoder on a fixed neuron axis, then the same GRU."""

    def __init__(self, n_units, d=mv.D_MODEL, h=mv.RNN_HIDDEN):
        super().__init__()
        self.n_units = n_units
        self.enc = nn.Sequential(nn.Linear(n_units, d), nn.GELU(), nn.Linear(d, d))
        self.rnn = nn.GRU(d, h, num_layers=1, batch_first=True)
        self.head = mv.Head(h)

    def forward(self, x, mask_n, mask_t, lengths):
        xt = x.transpose(1, 2)
        if xt.shape[-1] != self.n_units:
            if xt.shape[-1] > self.n_units:
                xt = xt[..., : self.n_units]
            else:
                pad = self.n_units - xt.shape[-1]
                xt = nn.functional.pad(xt, (0, pad))
        h = self.enc(xt) * mask_t.unsqueeze(-1)
        packed = pack_padded_sequence(h, lengths.cpu(), batch_first=True, enforce_sorted=False)
        out, _ = self.rnn(packed)
        out, _ = pad_packed_sequence(out, batch_first=True, total_length=x.shape[-1])
        return self.head(out)


def run_epoch_seq(model, loader, opt, dev, train=True, kind="seq"):
    model.train(train)
    total = 0.0
    n = 0
    for batch in loader:
        x = batch["x"].to(dev)
        y = batch["y"].to(dev)
        mask_t = batch["mask_t"].to(dev)
        lengths = batch["lengths"]
        if train:
            opt.zero_grad(set_to_none=True)
        if kind == "seq":
            pred = model(x, mask_t, lengths)
        else:
            pred = model(x, batch["mask_n"].to(dev), mask_t, lengths)
        loss = mv.trial_balanced_mse(pred, y, mask_t)
        if train:
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
        total += float(loss.item()) * x.size(0)
        n += x.size(0)
    return total / max(n, 1)


def train_torch(model, tr_ld, va_ld, cdir, name, kind="seq"):
    ckpt = cdir / "model.pt"
    hist_path = cdir / "history.json"
    start_epoch = 1
    resume = False
    if ckpt.exists() and hist_path.exists():
        hist = json.loads(hist_path.read_text())
        blob = torch.load(ckpt, map_location="cpu", weights_only=False)
        if hist.get("finished", True):
            log(f"  cache hit {name}")
            return blob, hist
        resume = True
        log(f"  resume {name} from epoch {len(hist.get('train', []))}")

    dev = mv.device_of()
    model = model.to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=mv.LR, weight_decay=mv.WEIGHT_DECAY)
    if resume:
        live = blob.get("live_state", blob.get("state_dict"))
        if live is not None:
            model.load_state_dict(live)
        if blob.get("opt_state") is not None:
            opt.load_state_dict(blob["opt_state"])
        best_state = blob.get("state_dict")
        best_val = float(blob.get("best_val", math.inf))
        hist = json.loads(hist_path.read_text())
        hist["finished"] = False
        bad = int(hist.get("bad", 0))
        start_epoch = len(hist.get("train", [])) + 1
        log(f"  continue {name} at epoch {start_epoch}  best_val={best_val:.4f}")
    else:
        hist = {
            "train": [],
            "val": [],
            "best_epoch": 0,
            "n_params": mv.n_params(model),
            "finished": False,
        }
        best_val = math.inf
        best_state = None
        bad = 0
        start_epoch = 1
        log(f"  {name} params={mv.n_params(model)}  device={dev}")

    for epoch in range(start_epoch, mv.MAX_EPOCHS + 1):
        tr_loss = run_epoch_seq(model, tr_ld, opt, dev, train=True, kind=kind)
        with torch.no_grad():
            va_loss = run_epoch_seq(model, va_ld, opt, dev, train=False, kind=kind)
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
        hist["bad"] = bad
        hist["finished"] = False
        blob = {
            "state_dict": best_state,
            "live_state": {k: v.detach().cpu().clone() for k, v in model.state_dict().items()},
            "opt_state": {k: v for k, v in opt.state_dict().items()},
            "best_val": best_val,
            "in_dim": getattr(model, "n_units", None),
        }
        torch.save(blob, ckpt)
        save_json(hist_path, hist)
        slurm_utils.write_status(
            state="training",
            epoch=epoch,
            train=tr_loss,
            val=va_loss,
            best_epoch=hist["best_epoch"],
            best_val=best_val,
        )
        if bad >= mv.PATIENCE:
            log(f"    early stop at epoch {epoch}")
            break
    hist["finished"] = True
    blob = {
        "state_dict": best_state,
        "best_val": best_val,
        "in_dim": getattr(model, "n_units", None),
    }
    torch.save(blob, ckpt)
    save_json(hist_path, hist)
    return blob, hist


def predict_seq(model, loader, y_mean, y_std, kind="seq"):
    dev = mv.device_of()
    model = model.to(dev)
    model.eval()
    recs = []
    with torch.no_grad():
        for batch in loader:
            if kind == "seq":
                pred = model(batch["x"].to(dev), batch["mask_t"].to(dev), batch["lengths"])
            else:
                pred = model(
                    batch["x"].to(dev),
                    batch["mask_n"].to(dev),
                    batch["mask_t"].to(dev),
                    batch["lengths"],
                )
            pred = pred.cpu().numpy()
            yz = batch["y"].numpy()
            mt = batch["mask_t"].numpy()
            lengths = batch["lengths"].numpy()
            for b in range(pred.shape[0]):
                T = int(lengths[b])
                p = pred[b, :T] * y_std + y_mean
                y = yz[b, :T] * y_std + y_mean
                valid = mt[b, :T]
                p = p.astype(np.float32)
                y = y.astype(np.float32)
                p[~valid] = np.nan
                y[~valid] = np.nan
                recs.append({"pred": p, "y": y})
    return recs


def row_from_scores(task, model, scores, oracle=False, extra=None):
    rows = []
    for target, sc in scores.items():
        row = {
            "task": task,
            "model": model,
            "target": target,
            "oracle": bool(oracle),
            "family": "within" if model in WITHIN_MODELS else "cross",
            **sc,
        }
        if extra:
            row.update(extra)
        rows.append(row)
    return rows


def log_scores(model, scores, oracle=False):
    tag = "  oracle" if oracle else ""
    for target, sc in scores.items():
        log(
            f"   {model:10s}{tag:8s} {target:12s} R²={sc['r2']:+.3f}  "
            f"trialR²={sc['mean_trial_r2']:+.3f}  r={sc['pearson']:.3f}"
        )


# ---------------------------------------------------------------------------
# Cross-recording baselines
# ---------------------------------------------------------------------------

def run_b0(trials, tr_idx, test_idx, which):
    chunks = []
    for i in tr_idx:
        y = np.asarray(trials[int(i)]["y"], dtype=np.float32).reshape(-1, trials[int(i)]["y"].shape[-1] if trials[int(i)]["y"].ndim > 1 else 1)
        valid = trials[int(i)].get("y_valid")
        if valid is None:
            valid = np.isfinite(y).all(axis=1)
        y = y[np.asarray(valid, dtype=bool)]
        if y.size:
            chunks.append(y)
    y_dim = int(np.asarray(trials[int(tr_idx[0])]["y"]).reshape(len(trials[int(tr_idx[0])]["y"]), -1).shape[-1])
    if chunks:
        y_tr = np.concatenate(chunks, axis=0)
        const = np.nanmean(y_tr, axis=0).astype(np.float32)
    else:
        const = np.zeros((y_dim,), dtype=np.float32)
    if which == "b0_const":
        pred_map = {int(i): np.broadcast_to(const, trials[int(i)]["y"].shape).copy() for i in test_idx}
        return recs_from_map(trials, test_idx, pred_map)

    max_t = max(trials[int(i)]["y"].shape[0] for i in tr_idx)
    acc = np.zeros((max_t, y_dim), dtype=np.float64)
    cnt = np.zeros((max_t, y_dim), dtype=np.float64)
    for i in tr_idx:
        y = np.asarray(trials[int(i)]["y"], dtype=np.float64).reshape(-1, y_dim)
        valid = np.isfinite(y)
        acc[: y.shape[0]] += np.where(valid, y, 0.0)
        cnt[: y.shape[0]] += valid.astype(np.float64)
    traj = (acc / np.maximum(cnt, 1.0)).astype(np.float32)
    pred_map = {}
    for i in test_idx:
        T = trials[int(i)]["y"].shape[0]
        pred_map[int(i)] = traj[:T] if T <= len(traj) else np.vstack(
            [traj, np.repeat(traj[-1:], T - len(traj), axis=0)]
        )
    return recs_from_map(trials, test_idx, pred_map)


def run_global_ridge(trials, tr_idx, va_idx, test_idx, feat_fn, crop, cdir, name):
    pkl = cdir / "ridge.pkl"
    if pkl.exists():
        blob = load_pkl(pkl)
        log(f"  cache hit {name}  alpha={blob['alpha']}")
        return recs_from_map(trials, test_idx, blob["pred_map"], crop=crop), blob

    Xtr, ytr = stack_xy(trials, tr_idx, feat_fn, crop=crop)
    Xva, yva = stack_xy(trials, va_idx, feat_fn, crop=crop)
    if Xtr is None or Xva is None:
        raise RuntimeError(f"{name}: empty train/val features")
    pred_map = {}
    Xte_parts, te_keys, te_lens = [], [], []
    for i in test_idx:
        x, y = align_xy(feat_fn(trials[int(i)]), trials[int(i)]["y"])
        if crop:
            x, y = x[crop:], y[crop:]
        Xte_parts.append(x)
        te_keys.append(int(i))
        te_lens.append(x.shape[0])
    Xte = np.vstack(Xte_parts).astype(np.float32)
    pred, info = fit_ridge_val(Xtr, ytr, Xva, yva, Xte)
    start = 0
    for key, n in zip(te_keys, te_lens):
        pred_map[key] = pred[start : start + n]
        start += n
    blob = {"alpha": info["alpha"], "val_mse": info["val_mse"], "pred_map": pred_map}
    dump_pkl(pkl, blob)
    log(f"  {name} alpha={info['alpha']}  val MSE={info['val_mse']:.4f}")
    return recs_from_map(trials, test_idx, pred_map, crop=crop), blob


def run_seq_gru(trials, tr_idx, va_idx, test_idx, feat_fn, y_mean, y_std, cdir, name):
    def pack(idx):
        feats = [feat_fn(trials[int(i)]) for i in idx]
        ys = [trials[int(i)]["y"] for i in idx]
        return feats, ys

    tr_f, tr_y = pack(tr_idx)
    va_f, va_y = pack(va_idx)
    te_f, te_y = pack(test_idx)
    in_dim = tr_f[0].shape[1]
    model = SequenceGRU(in_dim)
    tr_ld = DataLoader(SeqDS(tr_f, tr_y, y_mean, y_std), batch_size=mv.BATCH, shuffle=True, collate_fn=collate_seq)
    va_ld = DataLoader(SeqDS(va_f, va_y, y_mean, y_std), batch_size=mv.BATCH, shuffle=False, collate_fn=collate_seq)
    blob, hist = train_torch(model, tr_ld, va_ld, cdir, name, kind="seq")
    model = SequenceGRU(in_dim)
    model.load_state_dict(blob["state_dict"])
    te_ld = DataLoader(SeqDS(te_f, te_y, y_mean, y_std), batch_size=mv.BATCH, shuffle=False, collate_fn=collate_seq)
    recs = predict_seq(model, te_ld, y_mean, y_std, kind="seq")
    return recs, hist


# ---------------------------------------------------------------------------
# Within-session baselines
# ---------------------------------------------------------------------------

def session_groups(trials, idx):
    buckets = {}
    for i in np.asarray(idx):
        buckets.setdefault(trials[int(i)]["eid"], []).append(int(i))
    return buckets


def run_b9_session(trials, tr_ids, te_ids, cdir):
    """Instantaneous session-specific ridge (PDF eq. 101)."""
    pkl = cdir / "ridge.pkl"
    if pkl.exists():
        return load_pkl(pkl)

    def feat(t):
        return t["spikes"].astype(np.float32)

    if len(tr_ids) >= 2 and len(te_ids):
        oracle = False
        if len(tr_ids) >= MIN_SESSION_TRIALS:
            tr_fit, va_fit = mv.train_val_split(np.asarray(tr_ids))
            Xtr, ytr = stack_xy(trials, tr_fit, feat)
            Xva, yva = stack_xy(trials, va_fit, feat)
            Xte, _ = stack_xy(trials, te_ids, feat, drop_invalid=False)
            pred, info = fit_ridge_val(Xtr, ytr, Xva, yva, Xte)
        else:
            Xtr, ytr = stack_xy(trials, tr_ids, feat)
            Xte, _ = stack_xy(trials, te_ids, feat, drop_invalid=False)
            pred, info = fit_ridgecv(Xtr, ytr, Xte)
    elif len(te_ids) >= 2:
        ids = list(te_ids)
        pred_parts = {i: None for i in te_ids}
        n_splits = min(N_FOLDS, len(ids))
        n_splits = max(2, n_splits) if len(ids) >= 2 else 2
        n_splits = min(n_splits, len(ids))
        kf = KFold(n_splits=n_splits, shuffle=True, random_state=mv.SEED)
        info = {"alpha": float("nan")}
        for fold_tr, fold_te in kf.split(np.arange(len(ids))):
            tr = [ids[k] for k in fold_tr]
            te = [ids[k] for k in fold_te]
            Xtr, ytr = stack_xy(trials, tr, feat)
            Xte, _ = stack_xy(trials, te, feat, drop_invalid=False)
            p, info = fit_ridgecv(Xtr, ytr, Xte)
            start = 0
            for i in te:
                n = trials[i]["spikes"].shape[0]
                pred_parts[i] = p[start : start + n]
                start += n
        blob = {"pred_map": pred_parts, "oracle": True, "alpha": info.get("alpha")}
        dump_pkl(pkl, blob)
        return blob
    else:
        blob = {"pred_map": {i: np.full_like(trials[i]["y"], np.nan) for i in te_ids}, "oracle": True, "alpha": None}
        dump_pkl(pkl, blob)
        return blob

    start = 0
    pred_map = {}
    for i in te_ids:
        n = trials[i]["spikes"].shape[0]
        pred_map[i] = pred[start : start + n]
        start += n
    blob = {"pred_map": pred_map, "oracle": oracle, "alpha": info["alpha"]}
    dump_pkl(pkl, blob)
    return blob


def run_bwm_session(trials, tr_ids, te_ids, cdir):
    """BWM paper structure: lagged unit activity, session-specific Ridge, W=10."""
    pkl = cdir / "ridge.pkl"
    if pkl.exists():
        return load_pkl(pkl)

    def feat(t):
        return bwm_lag_matrix(t["spikes"])

    def predict_ids(tr, te):
        pred_map = {}
        Xtr, ytr = stack_xy(trials, tr, feat)
        if Xtr is None or ytr is None:
            for i in te:
                n = bwm_lag_matrix(trials[i]["spikes"]).shape[0]
                pred_map[i] = np.full((n, 1), np.nan, dtype=np.float32)
            return pred_map, {"alpha": float("nan")}
        ytr = ytr.reshape(ytr.shape[0], -1)[:, 0]
        Xte_parts, keys, lens = [], [], []
        for i in te:
            x, _ = align_xy(feat(trials[i]), trials[i]["y"])
            Xte_parts.append(x)
            keys.append(i)
            lens.append(x.shape[0])
        Xte = np.vstack(Xte_parts)
        if len(tr) >= MIN_SESSION_TRIALS:
            tr_fit, va_fit = mv.train_val_split(np.asarray(tr), seed=mv.SEED)
            Xa, ya = stack_xy(trials, tr_fit, feat)
            Xb, yb = stack_xy(trials, va_fit, feat)
            p, info = fit_ridge_val(
                Xa, ya.reshape(ya.shape[0], -1)[:, 0], Xb, yb.reshape(yb.shape[0], -1)[:, 0], Xte
            )
        else:
            p, info = fit_ridgecv(Xtr, ytr, Xte)
        start = 0
        for key, n in zip(keys, lens):
            pred_map[key] = np.asarray(p[start : start + n], dtype=np.float32).reshape(n, 1)
            start += n
        return pred_map, info

    if len(tr_ids) >= 2 and len(te_ids):
        pred_map, info = predict_ids(tr_ids, te_ids)
        blob = {"pred_map": pred_map, "oracle": False, "alpha": info["alpha"]}
    elif len(te_ids) >= 2:
        ids = list(te_ids)
        n_splits = min(N_FOLDS, len(ids))
        n_splits = min(max(n_splits, 2), len(ids))
        kf = KFold(n_splits=n_splits, shuffle=True, random_state=mv.SEED)
        pred_map = {}
        info = {"alpha": float("nan")}
        for fold_tr, fold_te in kf.split(np.arange(len(ids))):
            tr = [ids[k] for k in fold_tr]
            te = [ids[k] for k in fold_te]
            pmap, info = predict_ids(tr, te)
            pred_map.update(pmap)
        blob = {"pred_map": pred_map, "oracle": True, "alpha": info.get("alpha")}
    else:
        blob = {
            "pred_map": {i: np.full_like(trials[i]["y"], np.nan) for i in te_ids},
            "oracle": True,
            "alpha": None,
        }
    dump_pkl(pkl, blob)
    return blob


def run_b10_session(trials, tr_ids, te_ids, y_mean, y_std, cdir, name):
    n_units = trials[tr_ids[0] if tr_ids else te_ids[0]]["spikes"].shape[1]
    oracle = len(tr_ids) < 2
    pred_map = {}
    hists = []
    if len(tr_ids) >= 2 and len(te_ids):
        oracle = False
        if len(tr_ids) >= MIN_SESSION_TRIALS:
            tr_fit, va_fit = mv.train_val_split(np.asarray(tr_ids))
        else:
            tr_fit, va_fit = np.asarray(tr_ids[1:]), np.asarray(tr_ids[:1])
        model = SessionGRU(n_units)
        tr_ld = DataLoader(
            SpikesDS(trials, tr_fit, y_mean, y_std),
            batch_size=min(mv.BATCH, max(len(tr_fit), 1)),
            shuffle=True,
            collate_fn=collate_spikes,
        )
        va_ld = DataLoader(
            SpikesDS(trials, va_fit, y_mean, y_std),
            batch_size=min(mv.BATCH, max(len(va_fit), 1)),
            shuffle=False,
            collate_fn=collate_spikes,
        )
        blob, hist = train_torch(model, tr_ld, va_ld, cdir, name, kind="sess")
        hists.append(hist)
        model = SessionGRU(n_units)
        model.load_state_dict(blob["state_dict"])
        te_ld = DataLoader(
            SpikesDS(trials, te_ids, y_mean, y_std),
            batch_size=min(mv.BATCH, max(len(te_ids), 1)),
            shuffle=False,
            collate_fn=collate_spikes,
        )
        recs = predict_seq(model, te_ld, y_mean, y_std, kind="sess")
        for i, rec in zip(te_ids, recs):
            pred_map[int(i)] = rec["pred"]
    elif len(te_ids) >= 2:
        ids = list(te_ids)
        n_splits = min(N_FOLDS, len(ids))
        n_splits = min(max(n_splits, 2), len(ids))
        kf = KFold(n_splits=n_splits, shuffle=True, random_state=mv.SEED)
        for f, (fold_tr, fold_te) in enumerate(kf.split(np.arange(len(ids)))):
            tr = [ids[k] for k in fold_tr]
            te = [ids[k] for k in fold_te]
            tr_fit, va_fit = mv.train_val_split(np.asarray(tr), seed=mv.SEED + f)
            fold_dir = cdir / f"fold{f}"
            fold_dir.mkdir(parents=True, exist_ok=True)
            model = SessionGRU(n_units)
            tr_ld = DataLoader(
                SpikesDS(trials, tr_fit, y_mean, y_std),
                batch_size=min(mv.BATCH, max(len(tr_fit), 1)),
                shuffle=True,
                collate_fn=collate_spikes,
            )
            va_ld = DataLoader(
                SpikesDS(trials, va_fit, y_mean, y_std),
                batch_size=min(mv.BATCH, max(len(va_fit), 1)),
                shuffle=False,
                collate_fn=collate_spikes,
            )
            blob, hist = train_torch(model, tr_ld, va_ld, fold_dir, f"{name}/f{f}", kind="sess")
            hists.append(hist)
            model = SessionGRU(n_units)
            model.load_state_dict(blob["state_dict"])
            te_ld = DataLoader(
                SpikesDS(trials, te, y_mean, y_std),
                batch_size=min(mv.BATCH, max(len(te), 1)),
                shuffle=False,
                collate_fn=collate_spikes,
            )
            recs = predict_seq(model, te_ld, y_mean, y_std, kind="sess")
            for i, rec in zip(te, recs):
                pred_map[int(i)] = rec["pred"]
    else:
        for i in te_ids:
            pred_map[int(i)] = np.full_like(trials[int(i)]["y"], np.nan)
    if not hists:
        hist = {"train": [], "val": [], "best_epoch": 0, "n_params": 0}
    elif len(hists) == 1:
        hist = hists[0]
    else:
        hist = {
            "train": [h["train"][-1] if h["train"] else float("nan") for h in hists],
            "val": [h["val"][-1] if h["val"] else float("nan") for h in hists],
            "best_epoch": int(np.nanmean([h.get("best_epoch", 0) for h in hists])),
            "n_params": hists[0].get("n_params"),
            "folds": len(hists),
        }
    return pred_map, oracle, hist


def run_within_models(trials, train_full, test_idx, y_mean, y_std, task, models):
    te_groups = session_groups(trials, test_idx)
    tr_groups = session_groups(trials, train_full)
    out = {m: {"pred_map": {}, "oracle": False, "hists": []} for m in models if m in WITHIN_MODELS}
    for eid, te_ids in te_groups.items():
        tr_ids = tr_groups.get(eid, [])
        short = eid[:8]
        log(f"  session {short}  n_train={len(tr_ids)} n_test={len(te_ids)}")
        if "b9" in out:
            blob = run_b9_session(trials, tr_ids, te_ids, cache_dir(task, "b9", short))
            out["b9"]["pred_map"].update(blob["pred_map"])
            out["b9"]["oracle"] = out["b9"]["oracle"] or bool(blob.get("oracle"))
        if "bwm" in out:
            blob = run_bwm_session(trials, tr_ids, te_ids, cache_dir(task, "bwm", short))
            out["bwm"]["pred_map"].update(blob["pred_map"])
            out["bwm"]["oracle"] = out["bwm"]["oracle"] or bool(blob.get("oracle"))
        if "b10" in out:
            pmap, oracle, hist = run_b10_session(
                trials, tr_ids, te_ids, y_mean, y_std, cache_dir(task, "b10", short), f"{task}/b10/{short}"
            )
            out["b10"]["pred_map"].update(pmap)
            out["b10"]["oracle"] = out["b10"]["oracle"] or oracle
            out["b10"]["hists"].append(hist)
    recs = {}
    for m, pack in out.items():
        recs[m] = recs_from_map(trials, test_idx, pack["pred_map"])
        pack["recs"] = recs[m]
    return out


# ---------------------------------------------------------------------------
# Plots / report
# ---------------------------------------------------------------------------

def persist_scores(new_rows):
    path = OUT / "scores.csv"
    new = pd.DataFrame(new_rows)
    if path.exists() and not new.empty:
        old = pd.read_csv(path)
        idx = set(zip(new["task"], new["model"], new["target"]))
        keep = [
            (t, m, g) not in idx
            for t, m, g in zip(old["task"], old["model"], old["target"])
        ]
        merged = pd.concat([old.loc[keep], new], ignore_index=True)
    else:
        merged = new
    merged.to_csv(path, index=False)
    save_json(OUT / "scores.json", merged.to_dict(orient="records"))
    return merged.to_dict(orient="records")


def load_score_rows():
    path = OUT / "scores.csv"
    if not path.exists():
        return []
    return pd.read_csv(path).to_dict(orient="records")


def load_histories_from_cache():
    histories = {t: {} for t in mv.TASKS}
    for task in mv.TASKS:
        for model in ("b6", "b7"):
            path = CACHE / task / model / "history.json"
            if path.exists():
                histories[task][model] = json.loads(path.read_text())
        b10_root = CACHE / task / "b10"
        if not b10_root.exists():
            continue
        candidates = sorted(b10_root.rglob("history.json"))
        chosen = next((p for p in candidates if "fold" not in p.parts), None)
        if chosen is None and candidates:
            chosen = candidates[0]
        if chosen is not None:
            histories[task]["b10"] = json.loads(chosen.read_text())
    return histories


def merge_histories(live):
    disk = load_histories_from_cache()
    for task, by_model in (live or {}).items():
        disk.setdefault(task, {})
        for model, hist in by_model.items():
            if hist and hist.get("train"):
                disk[task][model] = hist
    return disk


def load_modelv1_rows(target=None):
    rows = []
    if target:
        roots = [mv.MODEL_ROOT / target]
    else:
        roots = [p for p in sorted(mv.MODEL_ROOT.iterdir()) if p.is_dir()] if mv.MODEL_ROOT.exists() else []
        if not roots and (mv.MODEL_ROOT / "scores.csv").exists():
            roots = [mv.MODEL_ROOT]
    mapping_names = [("scores.csv", False), ("scores_a1.csv", True)]
    for root in roots:
        for fname, is_a1 in mapping_names:
            path = root / fname
            if not path.exists():
                continue
            df = pd.read_csv(path)
            if target and "target" in df.columns:
                df = df[df.target == target]
            for _, r in df.iterrows():
                pool = str(r["pool"])
                key = f"A_{pool}_a1" if is_a1 or str(r.get("ablation", "none")) == "a1" else f"A_{pool}"
                rows.append(
                    {
                        "task": r["task"],
                        "model": key,
                        "target": r["target"],
                        "oracle": False,
                        "family": "proposed",
                        "r2": float(r["r2"]),
                        "mean_trial_r2": float(r["mean_trial_r2"]),
                        "median_trial_r2": float(r["median_trial_r2"]),
                        "rmse": float(r["rmse"]),
                        "mae": float(r["mae"]),
                        "pearson": float(r["pearson"]),
                        "n_bins": int(r["n_bins"]),
                        "n_trials": int(r["n_trials"]),
                    }
                )
    return rows


def plot_r2(rows, out_dir):
    mv.style_plots()
    df = pd.DataFrame(rows)
    tasks = [t for t in mv.TASKS if t in set(df.task)]
    models = [m for m in list(ALL_MODELS) + ["A_mean", "A_attn", "A_mean_a1", "A_attn_a1"] if m in set(df.model)]
    if not tasks or not models:
        return
    for metric, ylabel, fname in (
        ("r2", "held-out R² (concatenated bins)", "r2_concat"),
        ("mean_trial_r2", "held-out mean trial R²", "r2_trial"),
    ):
        plot_targets = [t for t in mv.TARGETS if t in set(df.target)] or list(set(df.target))
        if not plot_targets:
            continue
        fig, axes = plt.subplots(
            len(plot_targets),
            len(tasks),
            figsize=(4.6 * len(tasks), 4.0 * len(plot_targets)),
            sharey=False,
            squeeze=False,
        )
        width = 0.8
        x = np.arange(len(models))
        for r, target in enumerate(plot_targets):
            for c, task in enumerate(tasks):
                ax = axes[r, c]
                vals, colors, hatches = [], [], []
                for m in models:
                    sub = df[(df.task == task) & (df.model == m) & (df.target == target)]
                    vals.append(float(sub[metric].iloc[0]) if len(sub) else np.nan)
                    colors.append(MODEL_COLORS.get(m, "#64748B"))
                    oracle = bool(sub.oracle.iloc[0]) if len(sub) and "oracle" in sub.columns else False
                    hatches.append("//" if oracle else None)
                bars = ax.bar(x, vals, width, color=colors, edgecolor="#111827", linewidth=0.4)
                for bar, h in zip(bars, hatches):
                    if h:
                        bar.set_hatch(h)
                ax.axhline(0.0, color="0.6", lw=0.8)
                ax.set_xticks(x)
                ax.set_xticklabels([MODEL_LABELS.get(m, m) for m in models], rotation=55, ha="right", fontsize=8)
                ax.set_title(f"{task.replace('_holdout', '')} · {mv.TARGET_LABELS.get(target, target)}")
                if c == 0:
                    ax.set_ylabel(ylabel)
        fig.suptitle("Baselinev1 vs Formulation A  (hatched = within-session oracle)")
        fig.tight_layout()
        fig.savefig(out_dir / f"{fname}.png", dpi=150)
        plt.close(fig)


def plot_cross_vs_within(rows, out_dir):
    mv.style_plots()
    df = pd.DataFrame(rows)
    task = "trial_holdout"
    if task not in set(df.task):
        return
    models = [m for m in ("b0_traj", "b1", "b2", "b3", "b6", "b7", "b9", "b10", "bwm", "A_mean", "A_attn") if m in set(df.model)]
    plot_targets = [t for t in mv.TARGETS if t in set(df.target)] or list(set(df.target))
    n = max(len(plot_targets), 1)
    fig, axes = plt.subplots(1, n, figsize=(5.7 * n, 4.6), sharey=False)
    axes = np.atleast_1d(axes)
    x = np.arange(len(models))
    for ax, target in zip(axes, plot_targets):
        vals = []
        colors = []
        for m in models:
            sub = df[(df.task == task) & (df.model == m) & (df.target == target)]
            vals.append(float(sub.r2.iloc[0]) if len(sub) else np.nan)
            colors.append(MODEL_COLORS.get(m, "#64748B"))
        ax.bar(x, vals, color=colors, edgecolor="#111827", linewidth=0.4)
        ax.axhline(0.0, color="0.6", lw=0.8)
        ax.set_xticks(x)
        ax.set_xticklabels([MODEL_LABELS.get(m, m) for m in models], rotation=50, ha="right", fontsize=8)
        ax.set_ylabel("held-out R² (concatenated bins)")
        ax.set_title(mv.TARGET_LABELS.get(target, target))
    fig.suptitle("trial holdout: cross-recording baselines vs within-session references")
    fig.tight_layout()
    fig.savefig(out_dir / "trial_cross_vs_within.png", dpi=150)
    plt.close(fig)


def plot_training(histories, out_dir):
    mv.style_plots()
    items = []
    for task, by_model in histories.items():
        for model, hist in by_model.items():
            if hist and hist.get("train"):
                items.append((task, model, hist))
    if not items:
        return
    models = list(dict.fromkeys(m for _, m, _ in items))
    tasks = list(dict.fromkeys(t for t, _, _ in items))
    fig, axes = plt.subplots(
        len(tasks),
        len(models),
        figsize=(4.2 * len(models), 2.6 * len(tasks)),
        squeeze=False,
    )
    lookup = {(t, m): h for t, m, h in items}
    for r, task in enumerate(tasks):
        for c, model in enumerate(models):
            ax = axes[r, c]
            h = lookup.get((task, model))
            if not h:
                ax.set_axis_off()
                continue
            xs = np.arange(1, len(h["train"]) + 1)
            ax.plot(xs, h["train"], color="#334155", lw=1.4, label="train")
            ax.plot(xs, h["val"], color="#C47B3B", lw=1.4, label="val")
            ax.axvline(h.get("best_epoch", 1), color="0.7", ls="--", lw=0.8)
            ax.set_title(f"{task.replace('_holdout', '')} · {MODEL_LABELS.get(model, model)}")
            ax.set_ylabel("trial-balanced MSE (z)")
            ax.set_xlabel("epoch")
            ax.legend(frameon=False, fontsize=8)
    fig.suptitle("Matched-GRU baseline training")
    fig.tight_layout()
    fig.savefig(out_dir / "train_curves.png", dpi=150)
    plt.close(fig)


def plot_examples(example_bank, out_dir):
    mv.style_plots()
    show = ["b0_traj", "b2", "b6", "b7", "bwm", "b10", "A_attn"]
    for task, pack in example_bank.items():
        if not pack or not pack.get("true"):
            continue
        n = min(2, len(pack["true"]))
        fig, axes = plt.subplots(n, 1, figsize=(9.4, 2.4 * max(n, 1)), sharex=False)
        axes = np.atleast_1d(axes)
        ylab = mv.TARGET_YLABELS.get(pack.get("target", ""), "signal")
        tname = pack.get("target", "")
        for j in range(n):
            ytrue = np.asarray(pack["true"][j])
            if ytrue.ndim > 1:
                ytrue = ytrue[:, 0]
            t = np.arange(len(ytrue)) * mv.BINSIZE
            ax = axes[j]
            ax.plot(t, ytrue, color="#111827", lw=1.6, label="actual")
            for m in show:
                if m not in pack or j >= len(pack[m]):
                    continue
                pred = np.asarray(pack[m][j])
                if pred.ndim > 1:
                    pred = pred[:, 0]
                if pred.shape[0] != t.size:
                    aligned = np.full(t.size, np.nan, dtype=np.float32)
                    aligned[-pred.shape[0] :] = pred
                    pred = aligned
                ax.plot(
                    t,
                    pred,
                    color=MODEL_COLORS.get(m, "#64748B"),
                    lw=1.05,
                    alpha=0.9,
                    label=MODEL_LABELS.get(m, m),
                )
            ax.set_ylabel(ylab)
            ax.set_title(f"{task}  example {j + 1}  {tname}")
            if j == 0:
                ax.legend(frameon=False, ncol=4, fontsize=8)
        axes[-1].set_xlabel("time from stimOn (s)")
        fig.tight_layout()
        fig.savefig(out_dir / f"examples_{task}.png", dpi=150)
        plt.close(fig)


def write_report(splits, rows, histories, session_meta, path):
    df = pd.DataFrame(rows)
    lines = []
    lines.append("# Baselinev1 — controls for Formulation A on ModelDataRightContra")
    lines.append("")
    lines.append(f"Built {datetime.now(timezone.utc).isoformat()}")
    lines.append("")
    lines.append("## What this is")
    lines.append("")
    lines.append(
        "Baselines from the neural-decoding framework, evaluated on the same "
        "ModelDataRightContra motor-cortex corpus and holdouts as `Modelv1.py`. "
        "Each behavior is a separate 1-d model (this folder is one target). "
        "Cross-recording models do not use a globally fixed neuron index. "
        "Session-specific models (B9, B10, BWM) are within-session references; "
        "when a test session is absent from train they are scored as nested-CV "
        "**oracles** (hatched in the plots) and are not cross-session competitors."
    )
    lines.append("")
    lines.append("## Models")
    lines.append("")
    lines.append("- **B0 mean**: constant train-set mean of each target (PDF eq. 66–67).")
    lines.append("- **B0 traj**: stimOn-aligned mean behavioral trajectory (PDF eq. 68).")
    lines.append("- **B1**: instantaneous ridge on the population mean rate (PDF eq. 69–70).")
    lines.append(
        f"- **B2**: causal W={W_WINDOW} window of the population mean, ridge (PDF eq. 75–76). "
        "First W−1 bins are dropped."
    )
    lines.append(
        "- **B3**: Allen-region pooled ridge with occupancy mask and log unit-count "
        "(PDF eq. 79–82)."
    )
    lines.append(
        "- **B6**: population mean sequence through the same GRU + MLP head as Modelv1 "
        "(PDF eq. 92–94)."
    )
    lines.append("- **B7**: region-feature sequence through that same GRU (PDF eq. 95–97).")
    lines.append("- **B9**: session-specific instantaneous ridge on the unit vector (PDF eq. 101).")
    lines.append(
        "- **B10**: session-specific Linear–GELU–Linear encoder on the fixed neuron axis, "
        "then the matched GRU (PDF eq. 104–106)."
    )
    lines.append(
        "- **BWM lag ridge**: Brain-Wide Map movement decoder structure applied to this "
        "task — session-specific Ridge on motor units with causal lag W=10 "
        f"(paper `n_bins_lag={W_WINDOW}`, 20 ms bins), StandardScaler, alpha chosen on "
        "a validation split. The paper used Lasso; this file uses Ridge as in `BWMtest.py` "
        "because unscaled L1 did not converge on these counts. One 1-d model per "
        "behavior folder. Window is stimOn→end of stored trial, not firstMovement "
        "−0.2:+1.0 s."
    )
    lines.append("")
    if session_meta:
        n_mot = [s["n_motor"] for s in session_meta if s["n_motor"] >= mv.MIN_UNITS]
        n_sess = len(session_meta)
        mot_rng = f"{min(n_mot)}–{max(n_mot)}" if n_mot else "?"
    else:
        n_sess, mot_rng = 52, "5–226"
    lines.append(
        f"RightContra: {n_sess} sessions. Motor units/session "
        f"{mot_rng}. Same trial filter as Modelv1 "
        f"(T≥{mv.MIN_BINS}, crop {mv.MAX_BINS}). Inputs are log1p spike counts."
    )
    lines.append("")
    lines.append("## Holdouts")
    lines.append("")
    for task in mv.TASKS:
        sp = splits[task]
        n_tr = len(sp["train"]) if "train" in sp else int(sp.get("n_train", 0))
        n_te = len(sp["test"]) if "test" in sp else int(sp.get("n_test", 0))
        lines.append(f"- **{task}**: {sp['note']}. train n={n_tr}, test n={n_te}.")
    lines.append("")
    lines.append("## Results")
    lines.append("")
    lines.append(
        "| task | model | oracle | target | R² concat | mean trial R² | median trial R² | "
        "RMSE | MAE | Pearson | n bins | n trials |"
    )
    lines.append("|---|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|")
    order = {m: i for i, m in enumerate(list(ALL_MODELS) + ["A_mean", "A_attn", "A_mean_a1", "A_attn_a1"])}
    task_order = {t: i for i, t in enumerate(mv.TASKS)}
    baseline_rows = [r for r in rows if not str(r["model"]).startswith("A_")]
    baseline_rows.sort(
        key=lambda r: (task_order.get(r["task"], 99), order.get(r["model"], 99), r["target"])
    )
    for r in baseline_rows:
        ora = "yes" if r.get("oracle") else ""
        pear = r["pearson"]
        pear_s = "nan" if pear is None or (isinstance(pear, float) and math.isnan(pear)) else f"{pear:.3f}"
        lines.append(
            f"| {r['task']} | {r['model']} | {ora} | {r['target']} | "
            f"{r['r2']:.3f} | {r['mean_trial_r2']:.3f} | {r['median_trial_r2']:.3f} | "
            f"{r['rmse']:.3f} | {r['mae']:.3f} | {pear_s} | "
            f"{r['n_bins']} | {r['n_trials']} |"
        )
    prop = df[df.model.str.startswith("A_")]
    if not prop.empty:
        lines.append("")
        lines.append("### Formulation A (from Modelv1 scores, same splits)")
        lines.append("")
        lines.append("| task | model | target | R² concat | mean trial R² | Pearson |")
        lines.append("|---|---|---|---:|---:|---:|")
        for _, r in prop.iterrows():
            lines.append(
                f"| {r.task} | {r.model} | {r.target} | {r.r2:.3f} | {r.mean_trial_r2:.3f} | {r.pearson:.3f} |"
            )
    lines.append("")
    lines.append("### Best model per task × target (concatenated-bin R²)")
    lines.append("")
    for task in mv.TASKS:
        for target in mv.TARGETS:
            sub = df[(df.task == task) & (df.target == target)]
            if sub.empty:
                continue
            cross = sub[~sub.model.isin(WITHIN_MODELS)]
            within = sub[sub.model.isin(WITHIN_MODELS)]
            if not cross.empty:
                best = cross.loc[cross.r2.idxmax()]
                lines.append(
                    f"- {task} / {target} (cross-recording, includes Formulation A): "
                    f"**{best.model}** ({MODEL_LABELS.get(best.model, best.model)}) "
                    f"R²={best.r2:.3f}"
                )
            if not within.empty:
                best_w = within.loc[within.r2.idxmax()]
                tag = "oracle" if bool(best_w.get("oracle", False)) else "within-session"
                lines.append(
                    f"  - {tag} reference: **{best_w.model}** "
                    f"({MODEL_LABELS.get(best_w.model, best_w.model)}) R²={best_w.r2:.3f}"
                )
    lines.append("")
    lines.append("## Training diagnostics (GRU baselines)")
    lines.append("")
    for task in mv.TASKS:
        if not histories.get(task):
            continue
        lines.append(f"### {task}")
        lines.append("")
        for model, h in histories[task].items():
            be = int(h.get("best_epoch", 0) or 0)
            n_ep = len(h.get("train") or [])
            vtr = h["train"][be - 1] if be and be <= n_ep else float("nan")
            vva = h["val"][be - 1] if be and be <= len(h.get("val") or []) else float("nan")
            fold_note = f", {h['folds']} nested folds" if h.get("folds") else ""
            lines.append(
                f"- {model}: {h.get('n_params', '?')} params, best @ {be}/{n_ep}{fold_note}  "
                f"train={vtr:.4f}  val={vva:.4f}"
            )
        lines.append("")
    lines.extend(_interpretation_lines(df))
    lines.append(
        "Plots: `r2_concat.png`, `r2_trial.png`, `trial_cross_vs_within.png`, "
        "`train_curves.png`, `examples_<task>.png`."
    )
    lines.append("")
    Path(path).write_text("\n".join(lines))


def _r2(df, task, model, target):
    sub = df[(df.task == task) & (df.model == model) & (df.target == target)]
    return float(sub.r2.iloc[0]) if len(sub) else float("nan")


def _interpretation_lines(df):
    def fmt(task, model, target):
        v = _r2(df, task, model, target)
        return "n/a" if math.isnan(v) else f"{v:.3f}"

    lines = [
        "## Interpretation",
        "",
        "Concatenated-bin R² is the primary number. Mean trial R² is pulled around by "
        "short or nearly-still trials (especially paw speed on mouse holdout); median "
        "trial R² is the more stable per-trial summary.",
        "",
        "1. **Rate-only linear models do not decode this task.** B1 (population mean), "
        "B2 (causal W=10 window of that mean), and B3 (Allen-region pooled ridge) stay "
        f"near the constant-mean null on every holdout (trial wheel "
        f"{fmt('trial_holdout','b1','wheel_speed')} / "
        f"{fmt('trial_holdout','b2','wheel_speed')} / "
        f"{fmt('trial_holdout','b3','wheel_speed')} vs B0 mean "
        f"{fmt('trial_holdout','b0_const','wheel_speed')}). Instantaneous rate is not "
        "the missing ingredient.",
        "",
        "2. **A matched temporal backbone is.** B6 (population-mean GRU) and B7 "
        "(region-feature GRU) use the same GRU+MLP head as Formulation A. On trial "
        f"holdout, B7 reaches wheel/paw {fmt('trial_holdout','b7','wheel_speed')} / "
        f"{fmt('trial_holdout','b7','paw_speed')} vs the stimOn trajectory null "
        f"{fmt('trial_holdout','b0_traj','wheel_speed')} / "
        f"{fmt('trial_holdout','b0_traj','paw_speed')}. On a new session of a seen "
        f"mouse, B6 wheel {fmt('session_holdout','b6','wheel_speed')} beats that null "
        f"({fmt('session_holdout','b0_traj','wheel_speed')}); paw does not "
        f"({fmt('session_holdout','b6','paw_speed')} vs traj "
        f"{fmt('session_holdout','b0_traj','paw_speed')}). On a new mouse, B6 "
        f"({fmt('mouse_holdout','b6','wheel_speed')} / "
        f"{fmt('mouse_holdout','b6','paw_speed')}) is indistinguishable from the "
        f"trajectory null ({fmt('mouse_holdout','b0_traj','wheel_speed')} / "
        f"{fmt('mouse_holdout','b0_traj','paw_speed')}), and B7 is worse.",
        "",
        "3. **Formulation A is the best *cross-recording* model on seen animals, not "
        "on a new mouse.** Trial wheel: A attn "
        f"{fmt('trial_holdout','A_attn','wheel_speed')} vs best baseline B7 "
        f"{fmt('trial_holdout','b7','wheel_speed')}. Session wheel: A mean "
        f"{fmt('session_holdout','A_mean','wheel_speed')} vs B6 "
        f"{fmt('session_holdout','b6','wheel_speed')}. Session paw: A attn "
        f"{fmt('session_holdout','A_attn','paw_speed')} vs traj "
        f"{fmt('session_holdout','b0_traj','paw_speed')} — this is the one place "
        "anatomy + set pooling clearly helps a target that B6/B7 lose. Mouse holdout: "
        f"A attn {fmt('mouse_holdout','A_attn','wheel_speed')} / "
        f"{fmt('mouse_holdout','A_attn','paw_speed')} sits next to B6 and the "
        "trajectory null. The set encoder is not buying mouse transfer beyond a "
        "shared GRU on the population mean.",
        "",
        "4. **Within-session oracles are the ceiling, and they are not reachable from "
        "other recordings.** Hatched B10 (session-specific GRU on the fixed neuron "
        f"axis) is {fmt('session_holdout','b10','wheel_speed')} / "
        f"{fmt('session_holdout','b10','paw_speed')} on the held-out session and "
        f"{fmt('mouse_holdout','b10','wheel_speed')} / "
        f"{fmt('mouse_holdout','b10','paw_speed')} inside the held-out mouse. "
        "Formulation A recovers most of the session-oracle wheel "
        f"({fmt('session_holdout','A_mean','wheel_speed')} / "
        f"{fmt('session_holdout','b10','wheel_speed')}) but only about half of the "
        "within-mouse oracle. B9 (instantaneous session ridge) and the BWM lagged "
        "ridge sit between B10 and the cross-recording models on trial holdout; as "
        "oracles they still beat every linear rate baseline, and BWM session-oracle "
        f"wheel ({fmt('session_holdout','bwm','wheel_speed')}) is a strong linear "
        "within-session number.",
        "",
        "5. **What Formulation A actually buys on this corpus.** Shared per-unit "
        "anatomy + pooling beats a matched GRU on collapsed rate/region features "
        "when the animal has been seen (trial and new-insertion session). It does "
        "not beat a session-specific decoder that is allowed to see that session's "
        "neurons, and it does not beat the behavioral-trajectory null on an unseen "
        "mouse. Control A1 (drop CCF xyz and region) is the remaining check of "
        "whether those anatomical channels are doing the work on the seen-animal gains.",
        "",
    ]
    return lines


def pick_examples(trials, test_idx, recs_by_model, k=2):
    scored = []
    for j, i in enumerate(test_idx):
        y = trials[int(i)]["y"]
        if len(y) < 12:
            continue
        y0 = np.asarray(y).reshape(len(y), -1)[:, 0]
        scored.append((float(np.nanstd(y0)), len(y), j))
    scored.sort(reverse=True)
    chosen = [j for _, _, j in scored[:k]]
    pack = {"true": [], "target": trials[int(test_idx[0])].get("target", "") if len(test_idx) else ""}
    keys = [m for m in recs_by_model if recs_by_model[m]]
    for m in keys:
        pack[m] = []
    for j in chosen:
        y = None
        for m in keys:
            if recs_by_model[m] and j < len(recs_by_model[m]):
                y = recs_by_model[m][j]["y"]
                break
        if y is None:
            y = trials[int(test_idx[j])]["y"]
        pack["true"].append(y)
        for m in keys:
            if recs_by_model[m] and j < len(recs_by_model[m]):
                pack[m].append(recs_by_model[m][j]["pred"])
    return pack


def process(
    models=ALL_MODELS,
    tasks=mv.TASKS,
    make_plots=True,
    job_index=None,
    target="wheel_speed",
    trials=None,
    session_meta=None,
):
    warnings.filterwarnings("ignore")
    mv.set_seed(mv.SEED)
    configure_out(target)
    OUT.mkdir(parents=True, exist_ok=True)
    CACHE.mkdir(parents=True, exist_ok=True)
    models = tuple(models)
    log(f"device={mv.device_of()}  target={target} ({mv.TARGET_LABELS[target]})")
    if trials is None or session_meta is None:
        trials, session_meta = mv.load_corpus()
    mv.set_target(trials, target)
    splits = mv.make_splits(trials)
    mv.save_json(
        OUT / "splits.json",
        {
            t: {
                "note": splits[t]["note"],
                "n_train": int(len(splits[t]["train"])),
                "n_test": int(len(splits[t]["test"])),
                "hold_eids": splits[t]["hold_eids"],
            }
            for t in mv.TASKS
        },
    )

    rows = []
    histories = {t: {} for t in tasks}
    example_bank = {}
    n_job = 0
    for task in tasks:
        log(f"==== {task} / {target} ====")
        train_full = mv.usable_idx(trials, splits[task]["train"])
        test_idx = mv.usable_idx(trials, splits[task]["test"])
        if len(train_full) < 8 or len(test_idx) < 1:
            log(f"  skip {task}: too few usable trials for {target}")
            continue
        tr_idx, va_idx = mv.train_val_split(train_full)
        tr_idx, va_idx = mv.usable_idx(trials, tr_idx), mv.usable_idx(trials, va_idx)
        y_mean, y_std = mv.y_scaler_from(trials, tr_idx)
        vocab = region_list(trials, tr_idx)
        log(f"  regions={len(vocab)}  y_mean={y_mean} y_std={y_std}")
        recs_by_model = {}

        def feat_pop(t):
            return pop_rate(t["spikes"])

        def feat_pop_win(t):
            return causal_window(pop_rate(t["spikes"]), W_WINDOW)

        def feat_region(t):
            return region_features(t["spikes"], t["areas"], vocab)

        def start(model):
            nonlocal n_job
            jdir = _open_job(task, model, job_index, n_job, target)
            n_job += 1
            return jdir

        def finish(model, recs, sc, jdir, oracle=False, extra=None):
            recs_by_model[model] = recs
            these = row_from_scores(task, model, sc, oracle=oracle, extra=extra)
            rows.extend(these)
            pd.DataFrame(these).to_csv(jdir / "scores.csv", index=False)
            slurm_utils.write_status(state="done", n_rows=len(these))
            log_scores(model, sc, oracle=oracle)

        if "b0_const" in models:
            jdir = start("b0_const")
            recs = run_b0(trials, tr_idx, test_idx, "b0_const")
            finish("b0_const", recs, mv.score_recs(recs, target=target), jdir)

        if "b0_traj" in models:
            jdir = start("b0_traj")
            recs = run_b0(trials, tr_idx, test_idx, "b0_traj")
            finish("b0_traj", recs, mv.score_recs(recs, target=target), jdir)

        if "b1" in models:
            jdir = start("b1")
            recs, _ = run_global_ridge(
                trials, tr_idx, va_idx, test_idx, feat_pop, 0, cache_dir(task, "b1"), f"{task}/b1"
            )
            finish("b1", recs, mv.score_recs(recs, target=target), jdir)

        if "b2" in models:
            jdir = start("b2")
            recs, _ = run_global_ridge(
                trials, tr_idx, va_idx, test_idx, feat_pop_win, 0, cache_dir(task, "b2"), f"{task}/b2"
            )
            finish("b2", recs, mv.score_recs(recs, target=target), jdir)

        if "b3" in models:
            jdir = start("b3")
            recs, _ = run_global_ridge(
                trials, tr_idx, va_idx, test_idx, feat_region, 0, cache_dir(task, "b3"), f"{task}/b3"
            )
            finish("b3", recs, mv.score_recs(recs, target=target), jdir)

        if "b6" in models:
            jdir = start("b6")
            recs, hist = run_seq_gru(
                trials, tr_idx, va_idx, test_idx, feat_pop, y_mean, y_std, cache_dir(task, "b6"), f"{task}/b6"
            )
            histories[task]["b6"] = hist
            finish("b6", recs, mv.score_recs(recs, target=target), jdir)

        if "b7" in models:
            jdir = start("b7")
            recs, hist = run_seq_gru(
                trials,
                tr_idx,
                va_idx,
                test_idx,
                feat_region,
                y_mean,
                y_std,
                cache_dir(task, "b7"),
                f"{task}/b7",
            )
            histories[task]["b7"] = hist
            finish("b7", recs, mv.score_recs(recs, target=target), jdir)

        within = [m for m in models if m in WITHIN_MODELS]
        if within:
            lead = "b10" if "b10" in within else within[0]
            jdirs = {lead: start(lead)}
            packed = run_within_models(trials, train_full, test_idx, y_mean, y_std, task, within)
            for m in within:
                if m not in jdirs:
                    jdirs[m] = start(m)
                recs = packed[m]["recs"]
                sc = mv.score_recs(recs, target=target)
                extra = {}
                if packed[m]["hists"]:
                    histories[task][m] = packed[m]["hists"][0]
                finish(m, recs, sc, jdirs[m], oracle=packed[m]["oracle"], extra=extra)

        if make_plots:
            example_bank[task] = pick_examples(trials, test_idx, recs_by_model)
            persist_scores(rows)

    if make_plots:
        rows = load_score_rows()
        extra = load_modelv1_rows(target=target)
        all_rows = rows + extra
        histories = merge_histories(histories)
        plot_r2(all_rows, OUT)
        plot_cross_vs_within(all_rows, OUT)
        plot_training(histories, OUT)
        plot_examples(example_bank, OUT)
        write_report(splits, all_rows, histories, session_meta, OUT / "REPORT.md")
        log("wrote " + str(OUT))
        print(pd.DataFrame(rows).to_string(index=False))
    else:
        log("wrote job dir " + str(slurm_utils.job_dir() or OUT / "jobs"))
        if rows:
            print(pd.DataFrame(rows).to_string(index=False))
    return rows


def render_from_disk(target=None):
    """Rebuild plots and REPORT.md from cached scores/histories (no retraining)."""
    if target:
        configure_out(target)
    OUT.mkdir(parents=True, exist_ok=True)
    rows = load_score_rows() + load_modelv1_rows(target=target)
    if not rows:
        raise SystemExit("no scores.csv — run Baselinev1.py without --plots-only first")
    splits_path = OUT / "splits.json"
    splits = json.loads(splits_path.read_text()) if splits_path.exists() else {}
    histories = load_histories_from_cache()
    plot_r2(rows, OUT)
    plot_cross_vs_within(rows, OUT)
    plot_training(histories, OUT)
    write_report(splits, rows, histories, None, OUT / "REPORT.md")
    log("wrote plots + REPORT to " + str(OUT))
    print(pd.DataFrame(load_score_rows()).to_string(index=False))


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--models", nargs="*", default=list(ALL_MODELS))
    p.add_argument("--tasks", nargs="*", default=list(mv.TASKS))
    p.add_argument("--targets", nargs="*", default=list(mv.TARGETS), choices=list(mv.TARGETS))
    p.add_argument(
        "--plots-only",
        action="store_true",
        help="rebuild comparison plots and REPORT.md from cached scores",
    )
    slurm_utils.add_common_args(p)
    args = p.parse_args()
    if args.list_jobs:
        slurm_utils.print_jobs(job_grid())
        raise SystemExit(0)
    slurm_utils.set_device(args.device)
    if args.aggregate:
        for beh in args.targets:
            d = MODEL_ROOT / beh
            if (d / "jobs").exists():
                slurm_utils.aggregate_scores(d)
        raise SystemExit(0)
    if args.plots_only:
        for beh in args.targets:
            render_from_disk(target=beh)
    else:
        job = slurm_utils.resolve_job_index(args.job)
        if job is None:
            trials, session_meta = mv.load_corpus()
            for beh in args.targets:
                process(
                    models=tuple(args.models),
                    tasks=tuple(args.tasks),
                    target=beh,
                    trials=trials,
                    session_meta=session_meta,
                )
        else:
            cfg = slurm_utils.pick_config(job_grid(), job)
            process(
                models=(cfg["model"],),
                tasks=(cfg["task"],),
                target=cfg["target"],
                make_plots=False,
                job_index=job,
            )
