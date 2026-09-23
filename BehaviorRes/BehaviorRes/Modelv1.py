"""Formulation A set encoder on ModelDataRightContra (motor cortex only).

Shared per-unit MLP + masked mean or attention pool + causal GRU.
Neuron index is never a feature. Three holdouts: seen-session trials,
held-out session of a seen mouse, held-out mouse.

Loss is trial-balanced MSE on z-scored |ω| and 2D paw speed (PDF eq. 7).
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
from sklearn.metrics import mean_absolute_error, r2_score
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from torch.utils.data import DataLoader, Dataset

import slurm_utils

ROOT = Path(__file__).resolve().parent
DESKTOP = ROOT.parent
DATA = ROOT / "ModelDataRightContra"
SESS = DATA / "sessions"
OUT = ROOT / "Modelv1"
CACHE = OUT / "cache"

BINSIZE = 0.02
MIN_BINS = 13
MIN_UNITS = 5
MAX_BINS = 128
VAL_FRAC = 0.12
BATCH = 32
MAX_EPOCHS = 40
PATIENCE = 5
LR = 1e-3
WEIGHT_DECAY = 1e-4
SEED = 0
D_MODEL = 64
RNN_HIDDEN = 64
AREA_DIM = 16
DROPOUT = 0.1
UNIT_DROPOUT = 0.15

POOLS = ("mean", "attn")
TASKS = ("trial_holdout", "session_holdout", "mouse_holdout")
TARGETS = ("wheel_speed", "paw_vx", "paw_vy", "paw_vz", "paw_speed")
TARGET_LABELS = {
    "wheel_speed": "wheel speed |ω|",
    "paw_vx": "paw vx",
    "paw_vy": "paw vy",
    "paw_vz": "paw vz",
    "paw_speed": "paw speed (2D, x-y)",
}
TARGET_YLABELS = {
    "wheel_speed": "|ω| (rad/s)",
    "paw_vx": "vx (px/s)",
    "paw_vy": "vy (px/s)",
    "paw_vz": "vz (px/s)",
    "paw_speed": "2D speed (px/s)",
}
N_OUT = 1
MODEL_ROOT = ROOT / "Modelv1"

HOLD_MOUSE = "ZM_2241"
HOLD_SESSION_MOUSE = "CSH_ZAD_026"
HOLD_SESSION_EID = "626126d5-eecf-4e9b-900e-ec29a17ece07"


def log(msg):
    print(f"[{datetime.now().strftime('%H:%M:%S')}] {msg}", flush=True)


def set_seed(seed=SEED):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def device_of():
    return slurm_utils.device_of()


def job_grid():
    return [
        {"target": beh, "task": t, "pool": p, "ablation": a}
        for beh in TARGETS
        for a in ("none", "a1")
        for t in TASKS
        for p in POOLS
    ]


def configure_out(target):
    global OUT, CACHE
    if target not in TARGETS:
        raise SystemExit(f"unknown target {target!r}; choose from {TARGETS}")
    OUT = MODEL_ROOT / target
    CACHE = OUT / "cache"
    return OUT


def motor_mask(units):
    area = units["brain_area"].astype(str)
    return area.str.startswith("MOp") | area.str.startswith("MOs")


def _trace(rec, lp_key, dlc_key, absval=False):
    y = np.asarray(rec.get(lp_key, []), dtype=np.float32)
    if y.size == 0 or not np.isfinite(y).any():
        y = np.asarray(rec.get(dlc_key, []), dtype=np.float32)
    if absval:
        y = np.abs(y)
    return y


def trial_behaviors(rec):
    vx = _trace(rec, "lp_vx", "dlc_vx")
    vy = _trace(rec, "lp_vy", "dlc_vy")
    speed = _trace(rec, "lp_speed", "dlc_speed")
    if speed.size == 0 or not np.isfinite(speed).any():
        n = min(vx.size, vy.size)
        speed = np.hypot(vx[:n], vy[:n]).astype(np.float32) if n else speed
    return {
        "wheel_speed": _trace(rec, "wheel_velocity", "wheel_velocity", absval=True),
        "paw_vx": vx,
        "paw_vy": vy,
        "paw_vz": _trace(rec, "lp_vz", "dlc_vz"),
        "paw_speed": speed,
    }


def set_target(trials, name):
    """Bind one behavior as trial['y'] with shape (T, 1). Keeps time alignment."""
    if name not in TARGETS:
        raise ValueError(name)
    for t in trials:
        raw = np.asarray(t["behaviors"][name], dtype=np.float32).reshape(-1)
        n = int(t["spikes"].shape[0])
        y = np.full(n, np.nan, dtype=np.float32)
        m = min(n, raw.size)
        if m:
            y[:m] = raw[:m]
        t["y"] = y.reshape(-1, 1)
        t["y_valid"] = np.isfinite(y)
        t["target"] = name
    return trials


def usable_idx(trials, idx, min_bins=MIN_BINS):
    keep = [int(i) for i in np.asarray(idx) if int(trials[int(i)]["y_valid"].sum()) >= min_bins]
    return np.asarray(keep, dtype=int)


def load_corpus():
    """Motor MOp/MOs units only; log1p counts; CCF and area kept as metadata."""
    pkls = sorted(p for p in SESS.glob("*.pkl") if not p.name.endswith(".tmp"))
    trials = []
    session_meta = []
    log(f"loading {len(pkls)} sessions from ModelDataRightContra")
    n_motor = []
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
        n_motor.append(int(m.size))
        xyz = np.stack(
            [
                units["allen_x"].to_numpy(dtype=np.float32)[m],
                units["allen_y"].to_numpy(dtype=np.float32)[m],
                units["allen_z"].to_numpy(dtype=np.float32)[m],
            ],
            axis=1,
        )
        areas = units["brain_area"].astype(str).to_numpy()[m]
        for rec in payload["trials"]:
            spikes = np.asarray(rec.get("spike_counts", []), dtype=np.float32)
            if spikes.ndim != 2 or spikes.shape[1] <= m.max():
                continue
            behaviors = trial_behaviors(rec)
            n = min(spikes.shape[0], MAX_BINS)
            if n < MIN_BINS:
                continue
            spikes = spikes[:n, m]
            packed = {}
            for name, beh in behaviors.items():
                arr = np.full(n, np.nan, dtype=np.float32)
                src = np.asarray(beh, dtype=np.float32).reshape(-1)
                mlen = min(n, src.size)
                if mlen:
                    arr[:mlen] = src[:mlen]
                packed[name] = arr
            finite_spk = np.isfinite(spikes).all(axis=1)
            if finite_spk.sum() < MIN_BINS:
                continue
            trials.append(
                {
                    "eid": eid,
                    "mouse_id": mouse,
                    "session_date": date,
                    "trial_index": rec.get("trial_index"),
                    "spikes": np.log1p(np.clip(spikes, 0, None)).astype(np.float32),
                    "behaviors": packed,
                    "xyz": xyz,
                    "areas": areas,
                }
            )
    log(
        f"corpus: {len(trials)} trials, "
        f"{pd.DataFrame(session_meta).eid.nunique()} sessions, "
        f"motor units/session {min(n_motor)}–{max(n_motor)}"
    )
    return trials, session_meta


def make_splits(trials):
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
    chunks = []
    for i in idx:
        t = trials[int(i)]
        y = np.asarray(t["y"], dtype=np.float32)
        valid = t.get("y_valid")
        if valid is None:
            valid = np.isfinite(y).reshape(y.shape[0], -1).all(axis=1)
        y = y.reshape(y.shape[0], -1)[np.asarray(valid, dtype=bool)]
        if y.size:
            chunks.append(y)
    if not chunks:
        return np.zeros((N_OUT,), dtype=np.float32), np.ones((N_OUT,), dtype=np.float32)
    y = np.concatenate(chunks, axis=0)
    mean = y.mean(axis=0)
    std = y.std(axis=0)
    std = np.where(std < 1e-6, 1.0, std)
    return mean.astype(np.float32), std.astype(np.float32)


def fit_unit_stats(trials, train_idx, fallback_idx):
    """Area vocab, CCF z-score, and session mean log-rate. Train sessions only for scale/vocab."""
    train_eids = {trials[i]["eid"] for i in train_idx}
    seen = set()
    xyz_rows = []
    areas = []
    for i in train_idx:
        t = trials[i]
        if t["eid"] in seen:
            continue
        seen.add(t["eid"])
        xyz_rows.append(t["xyz"])
        areas.extend(t["areas"].tolist())
    xyz = np.concatenate(xyz_rows, axis=0)
    xyz_mean = np.nanmean(xyz, axis=0).astype(np.float32)
    xyz_std = np.nanstd(xyz, axis=0).astype(np.float32)
    xyz_std = np.where(xyz_std < 1e-8, 1.0, xyz_std).astype(np.float32)

    vocab = {"<unk>": 0}
    for a in sorted(set(areas)):
        if a not in vocab:
            vocab[a] = len(vocab)

    def rates_from(idx):
        buckets = {}
        for i in idx:
            t = trials[i]
            buckets.setdefault(t["eid"], []).append(t["spikes"].mean(axis=0))
        return {eid: np.mean(vs, axis=0).astype(np.float32) for eid, vs in buckets.items()}

    train_rates = rates_from(train_idx)
    fb_rates = rates_from(fallback_idx)
    rates = dict(fb_rates)
    rates.update(train_rates)
    return {
        "vocab": vocab,
        "n_areas": len(vocab),
        "xyz_mean": xyz_mean,
        "xyz_std": xyz_std,
        "rates": rates,
        "n_train_eids": len(train_eids),
    }


def encode_areas(areas, vocab):
    return np.array([vocab.get(a, 0) for a in areas], dtype=np.int64)


class TrialDS(Dataset):
    def __init__(self, trials, idx, y_mean, y_std, stats):
        self.trials = trials
        self.idx = np.asarray(idx)
        self.y_mean = y_mean
        self.y_std = y_std
        self.stats = stats

    def __len__(self):
        return len(self.idx)

    def __getitem__(self, j):
        t = self.trials[int(self.idx[j])]
        y = np.asarray(t["y"], dtype=np.float32).reshape(t["y"].shape[0], -1)
        valid = t.get("y_valid")
        if valid is None:
            valid = np.isfinite(y).all(axis=1)
        valid = np.asarray(valid, dtype=bool)
        y = np.where(valid[:, None], y, 0.0)
        y = (y - self.y_mean) / self.y_std
        xyz = (t["xyz"] - self.stats["xyz_mean"]) / self.stats["xyz_std"]
        xyz = np.nan_to_num(xyz, nan=0.0).astype(np.float32)
        areas = encode_areas(t["areas"], self.stats["vocab"])
        rate = self.stats["rates"][t["eid"]]
        return t["spikes"], y.astype(np.float32), valid, xyz, areas, rate


def collate(batch):
    spikes, ys, valids, xyzs, areas, rates = zip(*batch)
    B = len(batch)
    T = max(s.shape[0] for s in spikes)
    N = max(s.shape[1] for s in spikes)
    y_dim = int(ys[0].shape[-1])
    x = torch.zeros(B, N, T, dtype=torch.float32)
    y = torch.zeros(B, T, y_dim, dtype=torch.float32)
    xyz = torch.zeros(B, N, 3, dtype=torch.float32)
    area = torch.zeros(B, N, dtype=torch.long)
    rate = torch.zeros(B, N, dtype=torch.float32)
    mask_t = torch.zeros(B, T, dtype=torch.bool)
    mask_n = torch.zeros(B, N, dtype=torch.bool)
    lengths = torch.zeros(B, dtype=torch.long)
    for i, (s, yi, vi, zi, ai, ri) in enumerate(zip(spikes, ys, valids, xyzs, areas, rates)):
        n, tt = s.shape[1], s.shape[0]
        x[i, :n, :tt] = torch.from_numpy(s.T)
        y[i, :tt] = torch.from_numpy(yi)
        xyz[i, :n] = torch.from_numpy(zi)
        area[i, :n] = torch.from_numpy(ai)
        rate[i, :n] = torch.from_numpy(ri)
        mask_t[i, :tt] = torch.from_numpy(np.asarray(vi, dtype=bool))
        mask_n[i, :n] = True
        lengths[i] = tt
    return {
        "x": x,
        "y": y,
        "xyz": xyz,
        "area": area,
        "rate": rate,
        "mask_t": mask_t,
        "mask_n": mask_n,
        "lengths": lengths,
    }


def apply_unit_dropout(mask_n, p=UNIT_DROPOUT):
    if p <= 0:
        return mask_n
    keep = torch.rand(mask_n.shape, device=mask_n.device) >= p
    dropped = mask_n & keep
    empty = ~dropped.any(dim=1)
    if empty.any():
        dropped = dropped.clone()
        dropped[empty] = mask_n[empty]
    return dropped


class UnitEncoder(nn.Module):
    """φ_θ([x_nt, r̃_n, u_n]) shared across units and recordings.

    Control A1 drops CCF xyz and the region embedding: φ_θ([x_nt, u_n]).
    """

    def __init__(self, n_areas, d=D_MODEL, area_dim=AREA_DIM, use_anatomy=True):
        super().__init__()
        self.use_anatomy = use_anatomy
        if use_anatomy:
            self.area_emb = nn.Embedding(n_areas, area_dim, padding_idx=0)
            in_dim = 1 + 3 + area_dim + 1
        else:
            self.area_emb = None
            in_dim = 1 + 1
        self.mlp = nn.Sequential(
            nn.Linear(in_dim, d),
            nn.GELU(),
            nn.Linear(d, d),
        )

    def forward(self, x, xyz, area, rate, mask_n):
        # x: (B, N, T)
        xt = x.unsqueeze(-1)
        if self.use_anatomy:
            area_e = self.area_emb(area)
            static = torch.cat([xyz, area_e, rate.unsqueeze(-1)], dim=-1)
        else:
            static = rate.unsqueeze(-1)
        static = static.unsqueeze(2).expand(-1, -1, x.shape[-1], -1)
        h = self.mlp(torch.cat([xt, static], dim=-1))
        h = h.masked_fill(~mask_n[:, :, None, None], 0.0)
        return h


class MeanPool(nn.Module):
    def __init__(self, d=D_MODEL):
        super().__init__()
        self.rho = nn.Linear(d, d)

    def forward(self, e, mask_n):
        mask = mask_n[:, :, None, None]
        e0 = e.masked_fill(~mask, 0.0)
        denom = mask_n.sum(dim=1).clamp(min=1).to(e.dtype)[:, None, None]
        return self.rho(e0.sum(dim=1) / denom)


class AttnPool(nn.Module):
    """α_n ∝ exp(q⊤ tanh(W e_n)); h = Σ α_n e_n. Masked over observed units."""

    def __init__(self, d=D_MODEL):
        super().__init__()
        self.W = nn.Linear(d, d)
        self.q = nn.Linear(d, 1, bias=False)

    def forward(self, e, mask_n):
        logits = self.q(torch.tanh(self.W(e))).squeeze(-1)
        logits = logits.masked_fill(~mask_n[:, :, None], -1e9)
        alpha = torch.softmax(logits, dim=1)
        return (alpha.unsqueeze(-1) * e).sum(dim=1)


class Head(nn.Module):
    def __init__(self, d, p=DROPOUT, n_out=N_OUT):
        super().__init__()
        self.n_out = n_out
        self.net = nn.Sequential(nn.Dropout(p), nn.Linear(d, d), nn.GELU(), nn.Linear(d, n_out))

    def forward(self, h):
        return self.net(h)


class SetGRU(nn.Module):
    def __init__(self, n_areas, pool="mean", use_anatomy=True):
        super().__init__()
        self.pool_name = pool
        self.use_anatomy = use_anatomy
        self.enc = UnitEncoder(n_areas, use_anatomy=use_anatomy)
        self.pool = MeanPool(D_MODEL) if pool == "mean" else AttnPool(D_MODEL)
        self.rnn = nn.GRU(D_MODEL, RNN_HIDDEN, num_layers=1, batch_first=True)
        self.head = Head(RNN_HIDDEN)

    def forward(self, x, xyz, area, rate, mask_n, mask_t, lengths, unit_drop=0.0):
        if self.training and unit_drop > 0:
            mask_n = apply_unit_dropout(mask_n, unit_drop)
        e = self.enc(x, xyz, area, rate, mask_n)
        h = self.pool(e, mask_n) * mask_t.unsqueeze(-1)
        packed = pack_padded_sequence(
            h, lengths.cpu(), batch_first=True, enforce_sorted=False
        )
        out, _ = self.rnn(packed)
        out, _ = pad_packed_sequence(out, batch_first=True, total_length=x.shape[-1])
        return self.head(out)


def n_params(model):
    return int(sum(p.numel() for p in model.parameters() if p.requires_grad))


def trial_balanced_mse(pred, y, mask_t):
    """PDF eq. 7 in z-space: mean over trials of (mean over valid bins of ||·||_2^2)."""
    sq = (pred - y).pow(2).sum(dim=-1)
    w = mask_t.to(pred.dtype)
    denom = w.sum(dim=1).clamp(min=1.0)
    per_trial = (sq * w).sum(dim=1) / denom
    return per_trial.mean()


def run_epoch(model, loader, opt, dev, train=True):
    model.train(train)
    total = 0.0
    n = 0
    for batch in loader:
        x = batch["x"].to(dev)
        y = batch["y"].to(dev)
        xyz = batch["xyz"].to(dev)
        area = batch["area"].to(dev)
        rate = batch["rate"].to(dev)
        mask_t = batch["mask_t"].to(dev)
        mask_n = batch["mask_n"].to(dev)
        lengths = batch["lengths"]
        if train:
            opt.zero_grad(set_to_none=True)
        pred = model(
            x,
            xyz,
            area,
            rate,
            mask_n,
            mask_t,
            lengths,
            unit_drop=UNIT_DROPOUT if train else 0.0,
        )
        loss = trial_balanced_mse(pred, y, mask_t)
        if train:
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
        total += float(loss.item()) * x.size(0)
        n += x.size(0)
    return total / max(n, 1)


def model_tag(pool, use_anatomy=True):
    return pool if use_anatomy else f"{pool}_a1"


def cache_dir(task, pool, use_anatomy=True):
    d = CACHE / task / model_tag(pool, use_anatomy)
    d.mkdir(parents=True, exist_ok=True)
    return d


def save_json(path, obj):
    Path(path).write_text(json.dumps(obj, indent=2, default=float))


def train_one(pool, trials, train_idx, val_idx, y_mean, y_std, stats, task, use_anatomy=True):
    tag = model_tag(pool, use_anatomy)
    cdir = cache_dir(task, pool, use_anatomy)
    ckpt = cdir / "model.pt"
    hist_path = cdir / "history.json"
    if ckpt.exists() and hist_path.exists():
        hist = json.loads(hist_path.read_text())
        blob = torch.load(ckpt, map_location="cpu", weights_only=False)
        if hist.get("finished", True):
            log(f"  cache hit {task}/{tag}")
            return blob, hist
        log(f"  resume {task}/{tag} from epoch {len(hist.get('train', []))}")
        resume = True
    else:
        resume = False

    dev = device_of()
    model = SetGRU(stats["n_areas"], pool=pool, use_anatomy=use_anatomy).to(dev)
    extra = "" if use_anatomy else "  Control A1 (no xyz/region)"
    log(f"  {tag} params={n_params(model)}  device={dev}{extra}")
    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    tr_ds = TrialDS(trials, train_idx, y_mean, y_std, stats)
    va_ds = TrialDS(trials, val_idx, y_mean, y_std, stats)
    tr_ld = DataLoader(tr_ds, batch_size=BATCH, shuffle=True, collate_fn=collate)
    va_ld = DataLoader(va_ds, batch_size=BATCH, shuffle=False, collate_fn=collate)
    best_val = math.inf
    best_state = None
    bad = 0
    start_epoch = 1
    hist = {
        "train": [],
        "val": [],
        "best_epoch": 0,
        "n_params": n_params(model),
        "use_anatomy": bool(use_anatomy),
        "ablation": "none" if use_anatomy else "a1",
        "finished": False,
    }
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
    log(f"  train {tag}  n_tr={len(tr_ds)} n_val={len(va_ds)}")

    def snapshot():
        return {
            "state_dict": best_state,
            "live_state": {k: v.detach().cpu().clone() for k, v in model.state_dict().items()},
            "opt_state": {k: v for k, v in opt.state_dict().items()},
            "y_mean": y_mean,
            "y_std": y_std,
            "pool": pool,
            "n_areas": stats["n_areas"],
            "use_anatomy": bool(use_anatomy),
            "best_val": best_val,
            "stats": {
                "vocab": stats["vocab"],
                "n_areas": stats["n_areas"],
                "xyz_mean": stats["xyz_mean"],
                "xyz_std": stats["xyz_std"],
                "rates": stats["rates"],
            },
        }

    for epoch in range(start_epoch, MAX_EPOCHS + 1):
        tr_loss = run_epoch(model, tr_ld, opt, dev, train=True)
        with torch.no_grad():
            va_loss = run_epoch(model, va_ld, opt, dev, train=False)
        hist["train"].append(tr_loss)
        hist["val"].append(va_loss)
        log(f"    {tag} epoch {epoch:02d}  train={tr_loss:.4f}  val={va_loss:.4f}")
        if va_loss + 1e-5 < best_val:
            best_val = va_loss
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            hist["best_epoch"] = epoch
            bad = 0
        else:
            bad += 1
        hist["bad"] = bad
        hist["finished"] = False
        torch.save(snapshot(), ckpt)
        save_json(hist_path, hist)
        slurm_utils.write_status(
            state="training",
            epoch=epoch,
            train=tr_loss,
            val=va_loss,
            best_epoch=hist["best_epoch"],
            best_val=best_val,
            n_params=hist.get("n_params"),
        )
        if bad >= PATIENCE:
            log(f"    early stop at epoch {epoch}")
            break
    hist["finished"] = True
    out = snapshot()
    out.pop("live_state", None)
    out.pop("opt_state", None)
    torch.save(out, ckpt)
    save_json(hist_path, hist)
    return out, hist


def predict(blob, trials, idx, stats):
    stats = {
        "vocab": stats["vocab"],
        "n_areas": int(stats["n_areas"]),
        "xyz_mean": np.asarray(stats["xyz_mean"], dtype=np.float32),
        "xyz_std": np.asarray(stats["xyz_std"], dtype=np.float32),
        "rates": {k: np.asarray(v, dtype=np.float32) for k, v in stats["rates"].items()},
    }
    dev = device_of()
    model = SetGRU(
        blob["n_areas"],
        pool=blob["pool"],
        use_anatomy=blob.get("use_anatomy", True),
    ).to(dev)
    model.load_state_dict(blob["state_dict"])
    model.eval()
    ds = TrialDS(trials, idx, blob["y_mean"], blob["y_std"], stats)
    ld = DataLoader(ds, batch_size=BATCH, shuffle=False, collate_fn=collate)
    recs = []
    with torch.no_grad():
        for batch in ld:
            pred = model(
                batch["x"].to(dev),
                batch["xyz"].to(dev),
                batch["area"].to(dev),
                batch["rate"].to(dev),
                batch["mask_n"].to(dev),
                batch["mask_t"].to(dev),
                batch["lengths"],
                unit_drop=0.0,
            ).cpu().numpy()
            y_z = batch["y"].numpy()
            mt = batch["mask_t"].numpy()
            lengths = batch["lengths"].numpy()
            for b in range(pred.shape[0]):
                T = int(lengths[b])
                p = pred[b, :T] * blob["y_std"] + blob["y_mean"]
                y = y_z[b, :T] * blob["y_std"] + blob["y_mean"]
                valid = mt[b, :T]
                p = p.astype(np.float32)
                y = y.astype(np.float32)
                p[~valid] = np.nan
                y[~valid] = np.nan
                recs.append({"pred": p, "y": y})
    return recs


def _series(arr, j=0):
    arr = np.asarray(arr)
    if arr.ndim == 1:
        return arr
    if arr.shape[-1] <= j:
        return arr.reshape(arr.shape[0], -1)[:, 0]
    return arr[:, j]


def trial_r2_list(recs, j=0):
    out = []
    for r in recs:
        yt, yp = _series(r["y"], j), _series(r["pred"], j)
        m = np.isfinite(yt) & np.isfinite(yp)
        if m.sum() < 8:
            continue
        out.append(float(r2_score(yt[m], yp[m])))
    return np.asarray(out, dtype=float)


def score_recs(recs, target=None):
    y = np.concatenate([np.asarray(r["y"]).reshape(len(r["y"]), -1) for r in recs], axis=0)
    p = np.concatenate([np.asarray(r["pred"]).reshape(len(r["pred"]), -1) for r in recs], axis=0)
    names = (target,) if target else TARGETS[: y.shape[1]]
    out = {}
    for j, name in enumerate(names):
        yt, yp = y[:, j], p[:, j]
        m = np.isfinite(yt) & np.isfinite(yp)
        yt, yp = yt[m], yp[m]
        arr = trial_r2_list(recs, j)
        out[name] = {
            "r2": float(r2_score(yt, yp)) if m.sum() >= 10 else float("nan"),
            "rmse": float(np.sqrt(np.mean((yt - yp) ** 2))) if m.sum() else float("nan"),
            "mae": float(mean_absolute_error(yt, yp)) if m.sum() else float("nan"),
            "pearson": float(np.corrcoef(yt, yp)[0, 1]) if m.sum() >= 10 else float("nan"),
            "n_bins": int(m.sum()),
            "mean_trial_r2": float(np.nanmean(arr)) if arr.size else float("nan"),
            "median_trial_r2": float(np.nanmedian(arr)) if arr.size else float("nan"),
            "n_trials": int(arr.size),
        }
    return out


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


def plot_training(histories, out_dir, fname="train_curves.png", title="Formulation A training"):
    style_plots()
    tasks = [t for t in TASKS if histories.get(t)]
    pools = list(dict.fromkeys(p for t in tasks for p in histories[t]))
    if not tasks or not pools:
        return
    fig, axes = plt.subplots(
        len(tasks),
        len(pools),
        figsize=(4.4 * len(pools), 2.7 * len(tasks)),
        sharex=False,
        squeeze=False,
    )
    for r, task in enumerate(tasks):
        for c, pool in enumerate(pools):
            ax = axes[r, c]
            h = histories.get(task, {}).get(pool)
            if not h:
                ax.set_axis_off()
                continue
            xs = np.arange(1, len(h["train"]) + 1)
            ax.plot(xs, h["train"], color="#334155", lw=1.4, label="train")
            ax.plot(xs, h["val"], color="#C47B3B", lw=1.4, label="val")
            ax.axvline(h.get("best_epoch", 1), color="0.7", ls="--", lw=0.8)
            ax.set_title(f"{task} · {pool} pool")
            ax.set_ylabel("trial-balanced MSE (z)")
            ax.set_xlabel("epoch")
            ax.legend(frameon=False, fontsize=8)
    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(out_dir / fname, dpi=150)
    plt.close(fig)


def plot_r2_bars(rows, out_dir, suffix=""):
    style_plots()
    df = pd.DataFrame(rows)
    colors = {"mean": "#4C6A92", "attn": "#C47B3B"}
    tasks = [t for t in TASKS if t in set(df.task)]
    pools = [p for p in POOLS if p in set(df.pool)]
    if not tasks or not pools:
        return
    for metric, ylabel, fname in (
        ("r2", "held-out R² (concatenated bins)", "r2_concat"),
        ("mean_trial_r2", "held-out mean trial R²", "r2_trial"),
    ):
        fig, axes = plt.subplots(1, max(len(set(df.target)), 1), figsize=(5.2 * max(len(set(df.target)), 1), 4.2), sharey=False)
        axes = np.atleast_1d(axes)
        plot_targets = [t for t in TARGETS if t in set(df.target)] or list(set(df.target))
        for ax, target in zip(axes, plot_targets):
            x = np.arange(len(tasks))
            width = 0.8 / max(len(pools), 1)
            for i, pool in enumerate(pools):
                vals = []
                for task in tasks:
                    sub = df[(df.task == task) & (df.pool == pool) & (df.target == target)]
                    vals.append(float(sub[metric].iloc[0]) if len(sub) else np.nan)
                ax.bar(
                    x + (i - (len(pools) - 1) / 2) * width,
                    vals,
                    width,
                    color=colors.get(pool, "#64748B"),
                    label=f"{pool} pool",
                )
            ax.axhline(0.0, color="0.6", lw=0.8)
            ax.set_xticks(x)
            ax.set_xticklabels([t.replace("_holdout", "") for t in tasks])
            ax.set_ylabel(ylabel)
            ax.set_title(TARGET_LABELS[target])
            ax.legend(frameon=False, fontsize=8)
        fig.tight_layout()
        fig.savefig(out_dir / f"{fname}{suffix}.png", dpi=150)
        plt.close(fig)


def plot_examples(example_bank, out_dir, suffix=""):
    style_plots()
    colors = {"mean": "#4C6A92", "attn": "#C47B3B", "true": "#111827"}
    for task, pack in example_bank.items():
        if not pack:
            continue
        n = min(2, len(pack["true"]))
        fig, axes = plt.subplots(n, 1, figsize=(9.2, 2.4 * max(n, 1)), sharex=False)
        axes = np.atleast_1d(axes)
        k = 0
        pools_here = [p for p in POOLS if p in pack]
        ylab = TARGET_YLABELS.get(pack.get("target", ""), "signal")
        tname = pack.get("target", "")
        for j in range(n):
            ytrue = np.asarray(pack["true"][j])
            if ytrue.ndim > 1:
                ytrue = ytrue[:, 0]
            t = np.arange(len(ytrue)) * BINSIZE
            ax = axes[k]
            ax.plot(t, ytrue, color=colors["true"], lw=1.5, label="actual")
            for pool in pools_here:
                if j < len(pack[pool]):
                    yp = np.asarray(pack[pool][j])
                    if yp.ndim > 1:
                        yp = yp[:, 0]
                    ax.plot(t, yp, color=colors[pool], lw=1.1, alpha=0.9, label=f"{pool} pool")
            ax.set_ylabel(ylab)
            ax.set_title(f"{task}  example {j + 1}  {tname}")
            if k == 0:
                ax.legend(frameon=False, ncol=3, fontsize=8)
            k += 1
        axes[-1].set_xlabel("time from stimOn (s)")
        fig.tight_layout()
        fig.savefig(out_dir / f"examples_{task}{suffix}.png", dpi=150)
        plt.close(fig)


def plot_ablation_compare(out_dir):
    """Control A1 vs full Formulation A (same splits, mean and attn pools)."""
    full_path = out_dir / "scores.csv"
    a1_path = out_dir / "scores_a1.csv"
    if not full_path.exists() or not a1_path.exists():
        log("skip a1_vs_full plot (need both scores.csv and scores_a1.csv)")
        return
    style_plots()
    full = pd.read_csv(full_path)
    a1 = pd.read_csv(a1_path)
    colors = {
        ("mean", "none"): "#4C6A92",
        ("mean", "a1"): "#A7C1D9",
        ("attn", "none"): "#C47B3B",
        ("attn", "a1"): "#E6C4A0",
    }
    labels = {
        ("mean", "none"): "mean + anatomy",
        ("mean", "a1"): "mean, no anatomy",
        ("attn", "none"): "attn + anatomy",
        ("attn", "a1"): "attn, no anatomy",
    }
    series = [("mean", "none", full), ("mean", "a1", a1), ("attn", "none", full), ("attn", "a1", a1)]
    tasks = [t for t in TASKS if t in set(full.task) and t in set(a1.task)]
    if not tasks:
        return
    plot_targets = [t for t in TARGETS if t in set(full.target) or t in set(a1.target)]
    fig, axes = plt.subplots(1, max(len(plot_targets), 1), figsize=(5.6 * max(len(plot_targets), 1), 4.4), sharey=False)
    axes = np.atleast_1d(axes)
    width = 0.18
    for ax, target in zip(axes, plot_targets):
        x = np.arange(len(tasks))
        for i, (pool, ab, df) in enumerate(series):
            vals = []
            for task in tasks:
                sub = df[(df.task == task) & (df.pool == pool) & (df.target == target)]
                vals.append(float(sub.r2.iloc[0]) if len(sub) else np.nan)
            ax.bar(
                x + (i - 1.5) * width,
                vals,
                width,
                color=colors[(pool, ab)],
                label=labels[(pool, ab)],
            )
        ax.axhline(0.0, color="0.6", lw=0.8)
        ax.set_xticks(x)
        ax.set_xticklabels([t.replace("_holdout", "") for t in tasks])
        ax.set_ylabel("held-out R² (concatenated bins)")
        ax.set_title(TARGET_LABELS[target])
        ax.legend(frameon=False, fontsize=8)
    fig.suptitle("Control A1: drop CCF xyz and region labels")
    fig.tight_layout()
    fig.savefig(out_dir / "a1_vs_full.png", dpi=150)
    plt.close(fig)


def write_report(splits, rows, histories, session_meta, path, use_anatomy=True):
    df = pd.DataFrame(rows)
    lines = []
    if use_anatomy:
        lines.append("# Modelv1 — Formulation A set encoder on ModelDataRightContra")
    else:
        lines.append("# Modelv1 — Control A1 (set encoder without anatomical metadata)")
    lines.append("")
    lines.append(f"Built {datetime.now(timezone.utc).isoformat()}")
    lines.append("")
    lines.append("## What this is")
    lines.append("")
    if use_anatomy:
        lines.append(
            "Formulation A from the neural-decoding framework: a shared per-unit encoder "
            "φ_θ([log1p count, CCF xyz, region embedding, session mean log-rate]), "
            "permutation-invariant pooling, then a causal GRU and a 1-d head for one "
            f"behavior ({TARGET_LABELS.get(rows[0]['target'], 'target') if rows else 'behavior'}). "
            "Only MOp/MOs units are used. Column *n* is not a global neuron identity."
        )
    else:
        lines.append(
            "Control A1 from the neural-decoding framework (PDF §6.12.1): the same set "
            "encoder, pooling, causal GRU, and 1-d behavior head as Formulation A, but the "
            "per-unit encoder is φ_θ([log1p count, session mean log-rate]). CCF xyz and "
        "Allen-region embeddings are removed. This tests whether the direct set model "
        "learns a useful cross-recording representation from activity alone."
        )
    lines.append("")
    n_mot = [s["n_motor"] for s in session_meta if s["n_motor"] >= MIN_UNITS]
    tname = rows[0]["target"] if rows else "behavior"
    lines.append(
        f"RightContra: {len(session_meta)} sessions. Motor units/session "
        f"{min(n_mot)}–{max(n_mot)}. Trials shorter than {MIN_BINS} bins after "
        f"motor filtering are dropped; T is cropped at {MAX_BINS} bins (2.56 s). "
        f"This folder trains a 1-d head on **{TARGET_LABELS.get(tname, tname)}** only; "
        "sibling folders hold the other behaviors."
    )
    lines.append("")
    lines.append("## Holdouts")
    lines.append("")
    for task in TASKS:
        if task not in splits:
            continue
        sp = splits[task]
        lines.append(
            f"- **{task}**: {sp['note']}. train n={len(sp['train'])}, test n={len(sp['test'])}."
        )
    lines.append("")
    lines.append("## Model")
    lines.append("")
    if use_anatomy:
        enc_in = 1 + 3 + AREA_DIM + 1
        enc_desc = "log1p count, CCF xyz, region embedding, session mean log-rate"
    else:
        enc_in = 1 + 1
        enc_desc = "log1p count, session mean log-rate (Control A1)"
    lines.append(
        f"Unit MLP ({enc_desc}): Linear({enc_in}→{D_MODEL}), GELU, Linear({D_MODEL}→{D_MODEL}). "
        f"Mean pool applies Linear({D_MODEL}→{D_MODEL}) to the masked average. "
        "Attention pool uses α ∝ exp(q⊤ tanh(W e)) over units. "
        f"GRU hidden {RNN_HIDDEN}, decoder Dropout–Linear–GELU–Linear → 1. "
        f"Train-time unit dropout {UNIT_DROPOUT}."
    )
    lines.append("")
    lines.append(
        "Training loss is trial-balanced MSE on z-scored targets (PDF eq. 7): "
        "each trial contributes equally, then trials are averaged. "
        f"AdamW lr={LR}, weight decay={WEIGHT_DECAY}, batch {BATCH}, "
        f"max {MAX_EPOCHS} epochs, patience {PATIENCE}."
    )
    lines.append("")
    lines.append("## Results")
    lines.append("")
    lines.append(
        "| task | pool | target | R² concat | mean trial R² | median trial R² | "
        "RMSE | MAE | Pearson | n bins | n trials |"
    )
    lines.append("|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|")
    for r in rows:
        lines.append(
            f"| {r['task']} | {r['pool']} | {r['target']} | "
            f"{r['r2']:.3f} | {r['mean_trial_r2']:.3f} | {r['median_trial_r2']:.3f} | "
            f"{r['rmse']:.3f} | {r['mae']:.3f} | {r['pearson']:.3f} | "
            f"{r['n_bins']} | {r['n_trials']} |"
        )
    lines.append("")
    lines.append("### Best pool per task × target (concatenated-bin R²)")
    lines.append("")
    run_tasks = [t for t in TASKS if t in set(df.task)]
    run_pools = [p for p in POOLS if p in set(df.pool)]
    for task in run_tasks:
        for target in TARGETS:
            sub = df[(df.task == task) & (df.target == target)]
            if sub.empty:
                continue
            best = sub.loc[sub.r2.idxmax()]
            lines.append(
                f"- {task} / {target}: **{best.pool}** R²={best.r2:.3f} "
                f"(mean trial R²={best.mean_trial_r2:.3f})"
            )
    lines.append("")
    lines.append("## Training diagnostics")
    lines.append("")
    for task in run_tasks:
        lines.append(f"### {task}")
        lines.append("")
        for pool in run_pools:
            h = histories.get(task, {}).get(pool)
            if not h:
                continue
            be = h.get("best_epoch", 0)
            vtr = h["train"][be - 1] if be and be <= len(h["train"]) else float("nan")
            vva = h["val"][be - 1] if be and be <= len(h["val"]) else float("nan")
            lines.append(
                f"- {pool}: {h.get('n_params', '?')} params, best @ {be}/{len(h['train'])}  "
                f"train={vtr:.4f}  val={vva:.4f}"
            )
        lines.append("")
    if use_anatomy:
        lines.append("Plots: `train_curves.png`, `r2_concat.png`, `r2_trial.png`, `examples_<task>.png`.")
    else:
        lines.append(
            "Plots: `train_curves_a1.png`, `r2_concat_a1.png`, `r2_trial_a1.png`, "
            "`examples_<task>_a1.png`, and `a1_vs_full.png` when the full model scores exist."
        )
    lines.append("")
    Path(path).write_text("\n".join(lines))


def pick_examples(trials, test_idx, preds_by_pool, k=2):
    scored = []
    for j, i in enumerate(test_idx):
        y = trials[i]["y"]
        if len(y) < 12:
            continue
        y0 = np.asarray(y).reshape(len(y), -1)[:, 0]
        scored.append((float(np.nanstd(y0)), len(y), j))
    scored.sort(reverse=True)
    chosen = [j for _, _, j in scored[:k]]
    pools_here = [p for p in POOLS if p in preds_by_pool]
    lead = pools_here[0] if pools_here else next(iter(preds_by_pool))
    pack = {"true": [], "target": trials[int(test_idx[0])]["target"] if len(test_idx) else ""}
    for pool in POOLS:
        pack[pool] = []
    for j in chosen:
        pack["true"].append(preds_by_pool[lead][j]["y"])
        for pool in pools_here:
            pack[pool].append(preds_by_pool[pool][j]["pred"])
    return pack


def process(pools=POOLS, tasks=TASKS, ablation="none", make_plots=True, job_index=None, target="wheel_speed", trials=None, session_meta=None):
    warnings.filterwarnings("ignore")
    set_seed(SEED)
    use_anatomy = ablation != "a1"
    suffix = "" if use_anatomy else "_a1"
    configure_out(target)
    OUT.mkdir(parents=True, exist_ok=True)
    CACHE.mkdir(parents=True, exist_ok=True)
    log(f"device={device_of()}  target={target} ({TARGET_LABELS[target]})")
    if use_anatomy:
        log("Formulation A (full anatomical metadata)")
    else:
        log("Control A1: encoder is φ_θ([log1p count, session mean log-rate]); no xyz/region")
    if trials is None or session_meta is None:
        trials, session_meta = load_corpus()
    set_target(trials, target)
    splits = make_splits(trials)
    save_json(
        OUT / "splits.json",
        {
            t: {
                "note": splits[t]["note"],
                "n_train": int(len(splits[t]["train"])),
                "n_test": int(len(splits[t]["test"])),
                "hold_eids": splits[t]["hold_eids"],
            }
            for t in TASKS
        },
    )

    rows = []
    histories = {t: {} for t in tasks}
    example_bank = {}
    scores_csv = OUT / f"scores{suffix}.csv"
    n_job = 0
    for task in tasks:
        log(f"==== {task} ====")
        train_full = usable_idx(trials, splits[task]["train"])
        test_idx = usable_idx(trials, splits[task]["test"])
        if len(train_full) < 8 or len(test_idx) < 1:
            log(f"  skip {task}: too few usable trials for {target}")
            continue
        tr_idx, va_idx = train_val_split(train_full)
        tr_idx, va_idx = usable_idx(trials, tr_idx), usable_idx(trials, va_idx)
        y_mean, y_std = y_scaler_from(trials, tr_idx)
        stats = fit_unit_stats(trials, tr_idx, np.concatenate([tr_idx, va_idx, test_idx]))
        log(
            f"  areas={stats['n_areas']}  train_eids={stats['n_train_eids']}  "
            f"y_mean={y_mean} y_std={y_std}"
        )
        preds_by_pool = {}
        for pool in pools:
            log(f"-- {task} / {pool}" + ("" if use_anatomy else " / A1"))
            tag = f"{target}_{task}_{model_tag(pool, use_anatomy)}"
            idx = job_index if job_index is not None else n_job
            jdir = slurm_utils.begin_job(
                OUT,
                idx,
                tag,
                task=task,
                pool=pool,
                ablation="none" if use_anatomy else "a1",
                target=target,
            )
            blob, hist = train_one(
                pool, trials, tr_idx, va_idx, y_mean, y_std, stats, task, use_anatomy=use_anatomy
            )
            histories[task][pool] = hist
            recs = predict(blob, trials, test_idx, blob.get("stats", stats))
            preds_by_pool[pool] = recs
            scores = score_recs(recs, target=target)
            these = []
            for target, sc in scores.items():
                row = {
                    "task": task,
                    "pool": pool,
                    "target": target,
                    "ablation": "none" if use_anatomy else "a1",
                    **sc,
                }
                rows.append(row)
                these.append(row)
                log(
                    f"   {target:12s} R²={sc['r2']:+.3f}  trialR²={sc['mean_trial_r2']:+.3f}  "
                    f"RMSE={sc['rmse']:.3f}  r={sc['pearson']:.3f}"
                )
            pd.DataFrame(these).to_csv(jdir / "scores.csv", index=False)
            slurm_utils.write_status(state="done", n_rows=len(these), best_epoch=hist.get("best_epoch"))
            n_job += 1
            if make_plots:
                pd.DataFrame(rows).to_csv(scores_csv, index=False)
        if make_plots:
            example_bank[task] = pick_examples(trials, test_idx, preds_by_pool)

    if make_plots:
        title = "Formulation A training" if use_anatomy else "Control A1 training (no anatomical metadata)"
        plot_training(histories, OUT, fname=f"train_curves{suffix}.png", title=title)
        plot_r2_bars(rows, OUT, suffix=suffix)
        plot_examples(example_bank, OUT, suffix=suffix)
        write_report(
            splits,
            rows,
            histories,
            session_meta,
            OUT / f"REPORT{suffix}.md",
            use_anatomy=use_anatomy,
        )
        save_json(OUT / f"scores{suffix}.json", rows)
        if not use_anatomy:
            plot_ablation_compare(OUT)
        log("wrote " + str(OUT))
        print(pd.DataFrame(rows).to_string(index=False))
    else:
        log("wrote job dir " + str(slurm_utils.job_dir() or OUT / "jobs"))
        if rows:
            print(pd.DataFrame(rows).to_string(index=False))
    return rows


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--pools", nargs="*", default=list(POOLS))
    p.add_argument("--tasks", nargs="*", default=list(TASKS))
    p.add_argument(
        "--ablation",
        choices=["none", "a1"],
        default="none",
        help="none: full Formulation A. a1: Control A1, drop CCF xyz and region labels.",
    )
    p.add_argument("--targets", nargs="*", default=list(TARGETS), choices=list(TARGETS))
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
    job = slurm_utils.resolve_job_index(args.job)
    if job is None:
        trials, session_meta = load_corpus()
        for beh in args.targets:
            process(
                pools=tuple(args.pools),
                tasks=tuple(args.tasks),
                ablation=args.ablation,
                target=beh,
                trials=trials,
                session_meta=session_meta,
            )
    else:
        cfg = slurm_utils.pick_config(job_grid(), job)
        process(
            pools=(cfg["pool"],),
            tasks=(cfg["task"],),
            ablation=cfg["ablation"],
            target=cfg["target"],
            make_plots=False,
            job_index=job,
        )
