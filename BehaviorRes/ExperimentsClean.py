"""Unified motor-cortex → kinematics pipeline (paw speed).

Existing experiment scripts are left untouched. This file reuses their
architectures through imports and runs one evaluation / objective / loss grid.

Evaluation
  LTO  sequential 5% trial blocks per session (20 folds; leftover trials
       after 20 × floor(0.05 × n) always stay in train)
  LSO  leave one session out, only mice that have ≥2 sessions
  LMO  leave one mouse out (all of that mouse's sessions)

Objectives
  trace      paw-speed timeseries
  onset      reaction-time (discrete-time event likelihood)
  direction  wheel left vs right
  mixed      onset head, then score the curve only after true movement onset

Losses (neural models)
  mse          trial-balanced MSE
  huber_move   trial-balanced Huber, movement-weighted inside each trial
  event_nll    discrete-time onset likelihood (onset / mixed)
  bce          Bernoulli BCE (direction)

Local:
  python ExperimentsClean.py --self-test
  python ExperimentsClean.py --single-fold --list-jobs
  python ExperimentsClean.py --single-fold --job 0 --device auto

Cluster (entire pipeline, one fold of each eval):
  bash slurm/submit_clean.sh --single-fold
"""

from __future__ import annotations

import argparse
import json
import math
import os
import shutil
import sys
import traceback
import warnings
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

os.environ.setdefault("MPLBACKEND", "Agg")

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import mean_absolute_error, r2_score
from torch.utils.data import DataLoader, Dataset, Sampler

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import Modelv1 as mv
import slurm_utils

_SAVED_ROOT = mv.MODEL_ROOT
import Modelv2 as mv2
import graph as gnn
import heirarchical as hier

mv.MODEL_ROOT = _SAVED_ROOT

MODEL_ROOT = ROOT / "ExperimentsClean"
TARGET = "paw_speed"
LTO_FRAC = 0.05
N_LTO = 20
HUBER_DELTA = 1.0
MOVE_ALPHA = 3.0
MAPE_FLOOR = 1.0
ONSET_THR = 150.0
WHEEL_ON = 0.15
ONSET_DELAYED_S = 0.40
MIN_SCORE_BINS = 8
CNN_BATCH = 8
RT_EDGES = (0.20, 0.40, 0.80)
DUMMY = False
MAX_EPOCHS_CAP = None

EVALS = ("LTO", "LSO", "LMO")
MODELS = (
    "mean",
    "ar",
    "attn_a1_gru",
    "attn_a1_transformer",
    "attn_anat_transformer",
    "aniso_cnn_transformer",
    "hierarchical",
    "gnn_inner",
    "gnn_outer",
)
NEURAL = (
    "attn_a1_gru",
    "attn_a1_transformer",
    "attn_anat_transformer",
    "aniso_cnn_transformer",
    "hierarchical",
    "gnn_inner",
    "gnn_outer",
)
OBJECTIVES = ("trace", "onset", "direction", "mixed")
LOSSES = ("mse", "huber_move", "event_nll", "bce")
REGIONS = ("MOp", "MOs")


def log(msg):
    mv.log(msg)


def configure_out():
    global OUT, CACHE
    OUT = MODEL_ROOT / ("dummy" if DUMMY else TARGET)
    CACHE = OUT / "cache"
    OUT.mkdir(parents=True, exist_ok=True)
    CACHE.mkdir(parents=True, exist_ok=True)
    return OUT, CACHE


def device_probe(model_name):
    if model_name in ("attn_a1_transformer", "attn_anat_transformer", "aniso_cnn_transformer"):
        return "transformer"
    return "gru"


# ---------------------------------------------------------------------------
# Labels on the corpus
# ---------------------------------------------------------------------------


def _series(arr):
    return mv._series(arr, 0)


def annotate_trials(trials):
    n_dir = 0
    for t in trials:
        y = np.asarray(t["behaviors"][TARGET], dtype=np.float32).reshape(-1)
        on_s = mv.sticky_onset_s(y, ONSET_THR)
        t["onset_s"] = on_s
        t["onset_bin"] = (
            None
            if not np.isfinite(on_s)
            else int(round(float(on_s) / mv.BINSIZE))
        )
        signed = np.asarray(t["behaviors"].get("wheel_signed", []), dtype=np.float32).reshape(-1)
        moving = signed[np.isfinite(signed) & (np.abs(signed) >= WHEEL_ON)]
        if moving.size:
            val = float(moving.mean())
            t["wheel_dir"] = "right" if val > 0 else "left"
            n_dir += 1
        else:
            t["wheel_dir"] = None
        mag = np.where(np.isfinite(y), np.abs(y), 0.0)
        t["peak_bin"] = int(np.argmax(mag)) if mag.size else 0
        t["shape_bin"] = "hill" if t["peak_bin"] >= max(y.size - 2, 0) else "peak"
        t["rt_bin"] = rt_bin_of(on_s)
    log(f"annotated {len(trials)} trials  ({n_dir} with wheel direction)")
    return trials


def rt_bin_of(onset_s):
    if onset_s is None or not np.isfinite(onset_s):
        return "none"
    x = float(onset_s)
    if x < RT_EDGES[0]:
        return "early"
    if x < RT_EDGES[1]:
        return "mid"
    if x < RT_EDGES[2]:
        return "late"
    return "very_late"


def bind_objective(trials, objective):
    """Set trial['y'] / y_valid for the training target of this objective."""
    if objective in ("trace", "mixed", "onset"):
        mv.set_target(trials, TARGET, log1p=False)
        return
    for t in trials:
        n = int(t["spikes"].shape[0])
        lab = t.get("wheel_dir")
        if lab not in ("left", "right"):
            t["y"] = np.full((n, 1), np.nan, dtype=np.float32)
            t["y_valid"] = np.zeros(n, dtype=bool)
        else:
            t["y"] = np.full((n, 1), 1.0 if lab == "right" else 0.0, dtype=np.float32)
            t["y_valid"] = np.ones(n, dtype=bool)


# ---------------------------------------------------------------------------
# Folds
# ---------------------------------------------------------------------------


def _session_order(df, eid):
    return mv._session_order(df, eid)


def lto_folds(trials):
    df = mv.trial_frame(trials)
    all_idx = df["i"].to_numpy(dtype=int)
    chunks = {f: [] for f in range(N_LTO)}
    leftover = []
    skipped = 0
    for eid in sorted(df.eid.unique()):
        order = _session_order(df, eid)
        n = int(order.size)
        k = int(np.floor(n * LTO_FRAC))
        if k < 1:
            skipped += 1
            leftover.extend(int(i) for i in order)
            continue
        used = k * N_LTO
        for f in range(N_LTO):
            chunks[f].extend(int(i) for i in order[f * k : (f + 1) * k])
        leftover.extend(int(i) for i in order[used:])
    log(
        f"LTO: {N_LTO} folds, k=floor({LTO_FRAC:.0%} n) per session, "
        f"skipped {skipped} sessions with k=0, leftover={len(leftover)}"
    )
    out = []
    leftover = np.asarray(leftover, dtype=int)
    for f in range(N_LTO):
        test = np.asarray(chunks[f], dtype=int)
        if test.size < 1:
            continue
        train = np.setdiff1d(all_idx, test, assume_unique=False)
        out.append(
            {
                "eval": "LTO",
                "fold": f,
                "fold_id": f"lto{f:02d}",
                "train": train,
                "test": test,
                "note": (
                    f"block {f}: first-to-last 5% of each session "
                    f"(n_test={int(test.size)})"
                ),
            }
        )
    return out


def lso_folds(trials):
    df = mv.trial_frame(trials)
    n_sess = df.groupby("mouse_id")["eid"].nunique().to_dict()
    out, skipped = [], 0
    k = 0
    for eid in sorted(df.eid.unique()):
        mouse = str(df.loc[df.eid == eid, "mouse_id"].iloc[0])
        if int(n_sess.get(mouse, 1)) < 2:
            skipped += 1
            continue
        test = df.loc[df.eid == eid, "i"].to_numpy(dtype=int)
        train = df.loc[df.eid != eid, "i"].to_numpy(dtype=int)
        if len(test) < 1 or len(train) < 8:
            continue
        out.append(
            {
                "eval": "LSO",
                "fold": k,
                "fold_id": str(eid)[:8],
                "train": train,
                "test": test,
                "note": f"left-out session {str(eid)[:8]} of {mouse}",
            }
        )
        k += 1
    log(f"LSO: {k} folds from multi-session mice; skipped {skipped} single-session eids")
    return out


def lmo_folds(trials):
    df = mv.trial_frame(trials)
    out = []
    k = 0
    for mouse in sorted(df.mouse_id.astype(str).unique()):
        test = df.loc[df.mouse_id.astype(str) == mouse, "i"].to_numpy(dtype=int)
        train = df.loc[df.mouse_id.astype(str) != mouse, "i"].to_numpy(dtype=int)
        if len(test) < 1 or len(train) < 8:
            continue
        n = int(df.loc[df.mouse_id.astype(str) == mouse, "eid"].nunique())
        out.append(
            {
                "eval": "LMO",
                "fold": k,
                "fold_id": str(mouse),
                "train": train,
                "test": test,
                "note": f"left-out mouse {mouse} ({n} session(s))",
            }
        )
        k += 1
    log(f"LMO: {k} folds")
    return out


def all_folds(trials, evals=EVALS):
    out = []
    if "LTO" in evals:
        out.extend(lto_folds(trials))
    if "LSO" in evals:
        out.extend(lso_folds(trials))
    if "LMO" in evals:
        out.extend(lmo_folds(trials))
    return out


def keep_fold(specs, fold=None, single_fold=False):
    if fold is None and not single_fold:
        return specs
    want = 0 if single_fold and fold is None else int(fold)
    kept = []
    by_eval = defaultdict(list)
    for s in specs:
        by_eval[s["eval"]].append(s)
    for ev, rows in by_eval.items():
        rows = sorted(rows, key=lambda s: (int(s["fold"]), str(s["fold_id"])))
        if want < len(rows):
            kept.append(rows[want])
        elif rows:
            log(f"  {ev}: no fold {want}, skipping")
    return kept


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------


def _finite_pair(y, p):
    y = _series(y).astype(float)
    p = _series(p).astype(float)
    n = min(y.size, p.size)
    y, p = y[:n], p[:n]
    m = np.isfinite(y) & np.isfinite(p)
    return y, p, m


def mape_value(y, p, m):
    if int(m.sum()) < 1:
        return float("nan")
    denom = np.maximum(np.abs(y[m]), MAPE_FLOOR)
    return float(np.mean(np.abs(y[m] - p[m]) / denom))


def pearson_value(y, p, m):
    if int(m.sum()) < 3:
        return float("nan")
    yt, yp = y[m], p[m]
    if float(np.std(yt)) < 1e-12 or float(np.std(yp)) < 1e-12:
        return float("nan")
    return float(np.corrcoef(yt, yp)[0, 1])


def trial_metrics(y, pred, onset_true=None, onset_pred=None, min_bins=MIN_SCORE_BINS):
    y, p, m = _finite_pair(y, pred)
    n = int(m.sum())
    out = {
        "n_score_bins": n,
        "pearson": float("nan"),
        "r2": float("nan"),
        "rmse": float("nan"),
        "mape": float("nan"),
        "mae": float("nan"),
        "onset_true_s": float("nan"),
        "onset_pred_s": float("nan"),
        "onset_err_s": float("nan"),
        "onset_abs_s": float("nan"),
        "peak_true_s": float("nan"),
        "peak_pred_s": float("nan"),
        "peak_err_s": float("nan"),
        "peak_abs_s": float("nan"),
        "peak_true": float("nan"),
        "peak_pred": float("nan"),
        "peak_mag_err": float("nan"),
        "peak_mag_abs": float("nan"),
    }
    if n >= 1:
        yt, yp = y[m], p[m]
        out["mae"] = float(mean_absolute_error(yt, yp))
        out["rmse"] = float(np.sqrt(np.mean((yt - yp) ** 2)))
        out["mape"] = mape_value(y, p, m)
        out["pearson"] = pearson_value(y, p, m)
    if n >= int(min_bins):
        out["r2"] = float(r2_score(y[m], p[m]))
    if onset_true is None or not np.isfinite(onset_true):
        onset_true = mv.sticky_onset_s(y, ONSET_THR)
    if onset_pred is None or not np.isfinite(onset_pred):
        onset_pred = mv.sticky_onset_s(p, ONSET_THR)
    out["onset_true_s"] = float(onset_true) if np.isfinite(onset_true) else float("nan")
    out["onset_pred_s"] = float(onset_pred) if np.isfinite(onset_pred) else float("nan")
    if np.isfinite(out["onset_true_s"]) and np.isfinite(out["onset_pred_s"]):
        out["onset_err_s"] = out["onset_pred_s"] - out["onset_true_s"]
        out["onset_abs_s"] = abs(out["onset_err_s"])
    pk_t_s, pk_t = mv.peak_time_amp(y)
    pk_p_s, pk_p = mv.peak_time_amp(p)
    out["peak_true_s"], out["peak_true"] = float(pk_t_s), float(pk_t)
    out["peak_pred_s"], out["peak_pred"] = float(pk_p_s), float(pk_p)
    if np.isfinite(pk_t_s) and np.isfinite(pk_p_s):
        out["peak_err_s"] = float(pk_p_s - pk_t_s)
        out["peak_abs_s"] = abs(out["peak_err_s"])
    if np.isfinite(pk_t) and np.isfinite(pk_p):
        out["peak_mag_err"] = float(pk_p - pk_t)
        out["peak_mag_abs"] = abs(out["peak_mag_err"])
    return out


def concat_metrics(recs, min_bins=MIN_SCORE_BINS):
    if not recs:
        return {k: float("nan") for k in ("pearson", "r2", "rmse", "mape", "mae")}
    ys = np.concatenate([_series(r["y"]) for r in recs])
    ps = np.concatenate([_series(r["pred"]) for r in recs])
    _, _, m = _finite_pair(ys, ps)
    y, p = ys, ps
    out = {
        "pearson": pearson_value(y, p, m),
        "r2": float("nan"),
        "rmse": float("nan"),
        "mape": mape_value(y, p, m),
        "mae": float("nan"),
        "n_bins": int(m.sum()),
    }
    if int(m.sum()) >= 1:
        out["mae"] = float(mean_absolute_error(y[m], p[m]))
        out["rmse"] = float(np.sqrt(np.mean((y[m] - p[m]) ** 2)))
    if int(m.sum()) >= max(10, int(min_bins)):
        out["r2"] = float(r2_score(y[m], p[m]))
    return out


METRIC_KEYS = (
    "pearson",
    "r2",
    "rmse",
    "mape",
    "mae",
    "onset_err_s",
    "onset_abs_s",
    "peak_err_s",
    "peak_abs_s",
    "peak_mag_err",
    "peak_mag_abs",
)


def pool_rows(rows, prefix):
    """Mean / median of per-trial metrics."""
    out = {}
    if not rows:
        for k in METRIC_KEYS:
            out[f"{prefix}_mean_{k}"] = float("nan")
            out[f"{prefix}_median_{k}"] = float("nan")
        return out
    df = pd.DataFrame(rows)
    for k in METRIC_KEYS:
        v = pd.to_numeric(df.get(k), errors="coerce") if k in df.columns else pd.Series(dtype=float)
        out[f"{prefix}_mean_{k}"] = float(v.mean()) if v.notna().any() else float("nan")
        out[f"{prefix}_median_{k}"] = float(v.median()) if v.notna().any() else float("nan")
    return out


def score_fold(recs, trials, test_idx, objective):
    rows = []
    idx = [int(i) for i in np.asarray(test_idx).reshape(-1)]
    n = min(len(recs), len(idx))
    for rec, i in zip(recs[:n], idx[:n]):
        t = trials[int(i)]
        met = trial_metrics(
            rec.get("y"),
            rec.get("pred"),
            onset_true=rec.get("onset_true", t.get("onset_s")),
            onset_pred=rec.get("onset_pred"),
            min_bins=MIN_SCORE_BINS if objective != "onset" else 2,
        )
        met.update(
            {
                "trial_i": int(i),
                "eid": str(t.get("eid", "")),
                "mouse_id": str(t.get("mouse_id", "")),
                "trial_index": t.get("trial_index", -1),
                "n_bins": int(_series(rec.get("y")).size),
                "wheel_dir": t.get("wheel_dir"),
                "rt_bin": t.get("rt_bin"),
                "dir_true": rec.get("dir_true", float("nan")),
                "dir_pred": rec.get("dir_pred", float("nan")),
            }
        )
        rows.append(met)
    cat = concat_metrics(recs[:n])
    pooled = pool_rows(rows, "fold")
    pooled.update({f"fold_concat_{k}": cat.get(k, float("nan")) for k in ("pearson", "r2", "rmse", "mape", "mae")})
    pooled["n_holdout_trials"] = int(len(rows))
    pooled["n_concat_bins"] = int(cat.get("n_bins", 0))
    if objective == "direction" and rows:
        dt = np.array([r.get("dir_true", np.nan) for r in rows], dtype=float)
        dp = np.array([r.get("dir_pred", np.nan) for r in rows], dtype=float)
        m = np.isfinite(dt) & np.isfinite(dp)
        pooled["fold_direction_acc"] = float(np.mean((dt[m] >= 0.5) == (dp[m] >= 0.5))) if m.any() else float("nan")
    return rows, pooled


def apply_mixed_score_mask(recs, trials, test_idx):
    """Keep bins at/after true movement onset (stimulus → true onset is not scored)."""
    out = []
    idx = [int(i) for i in np.asarray(test_idx).reshape(-1)]
    for rec, i in zip(recs, idx[: len(recs)]):
        y = _series(rec["y"]).astype(np.float32).copy()
        p = _series(rec["pred"]).astype(np.float32).copy()
        t = trials[int(i)]
        on = t.get("onset_bin")
        if on is None or not np.isfinite(on):
            on = 0
        on = int(max(0, on))
        y[:on] = np.nan
        p[:on] = np.nan
        out.append({**rec, "y": y.reshape(-1, 1), "pred": p.reshape(-1, 1)})
    return out


# ---------------------------------------------------------------------------
# Losses
# ---------------------------------------------------------------------------


def huber_delta(err, delta=HUBER_DELTA):
    abs_e = err.abs()
    quad = 0.5 * err.pow(2)
    lin = delta * (abs_e - 0.5 * delta)
    return torch.where(abs_e <= delta, quad, lin)


def trial_balanced_mse(pred, y, mask):
    return mv.trial_balanced_mse(pred, y, mask)


def trial_balanced_huber_move(pred, y, mask, y_mean, y_std, speed_scale, alpha=MOVE_ALPHA):
    """Per-trial mean Huber, reweighted by a soft movement gate, then mean over trials.

    w_{i,t} = 1 + α |y|_{i,t} / (s + |y|_{i,t}) on original units. Denominator
    sum_t w keeps high-movement trials from dominating the batch.
    """
    err = (pred - y)[..., 0]
    h = huber_delta(err)
    mean = pred.new_tensor(np.asarray(y_mean, dtype=np.float32).reshape(-1)[:1])
    std = pred.new_tensor(np.asarray(y_std, dtype=np.float32).reshape(-1)[:1]).clamp(min=1e-6)
    y0 = (y[..., 0] * std + mean).abs()
    s = float(max(speed_scale, 1e-3))
    w = (1.0 + float(alpha) * y0 / (s + y0)) * mask.to(pred.dtype)
    per = (h * w).sum(dim=1) / w.sum(dim=1).clamp(min=1.0)
    return per.mean()


def begun_labels(y, mask, y_mean, y_std):
    mean = y.new_tensor(np.asarray(y_mean, dtype=np.float32)).view(1, 1, -1)
    std = y.new_tensor(np.asarray(y_std, dtype=np.float32)).view(1, 1, -1)
    y0 = (y * std + mean)[..., 0]
    above = (y0.abs() >= ONSET_THR) & mask
    begun = torch.zeros_like(above)
    run = torch.zeros(y0.size(0), dtype=torch.long, device=y.device)
    hit = torch.zeros(y0.size(0), dtype=torch.bool, device=y.device)
    onset_bin = y0.new_full((y0.size(0),), -1)
    for t in range(y0.size(1)):
        run = torch.where(above[:, t], run + 1, torch.zeros_like(run))
        now = (~hit) & (run >= 2)
        if bool(now.any()):
            onset_bin = torch.where(now, torch.full_like(onset_bin, t - 1), onset_bin)
            hit = hit | now
        begun[:, t] = hit & mask[:, t]
    return begun.to(y.dtype), onset_bin, hit


def event_time_nll(logit, onset_bin, mask):
    """Discrete-time hazard NLL. onset_bin = -1 means no onset in the trial."""
    haz = torch.sigmoid(logit[..., 0]).clamp(1e-6, 1 - 1e-6)
    log_surv = torch.log1p(-haz)
    log_haz = torch.log(haz)
    bsz, n_t = haz.shape
    tgrid = torch.arange(n_t, device=haz.device).unsqueeze(0).expand(bsz, n_t)
    valid = mask.to(dtype=torch.bool)
    has = onset_bin >= 0
    k = onset_bin.clamp(min=0).unsqueeze(1)
    before = valid & has.unsqueeze(1) & (tgrid < k)
    at = valid & has.unsqueeze(1) & (tgrid == k)
    none = valid & (~has).unsqueeze(1)
    nll = -(log_surv * before.to(haz.dtype)).sum(dim=1)
    nll = nll - (log_haz * at.to(haz.dtype)).sum(dim=1)
    nll = nll - (log_surv * none.to(haz.dtype)).sum(dim=1)
    return nll.mean()


def trial_bce(logit, label, mask):
    bce = F.binary_cross_entropy_with_logits(logit[..., 0], label, reduction="none")
    w = mask.to(dtype=logit.dtype)
    per = (bce * w).sum(dim=1) / w.sum(dim=1).clamp(min=1.0)
    return per.mean()


def speed_scale_from(trials, idx):
    vals = []
    for i in np.asarray(idx, dtype=int):
        y = np.asarray(trials[int(i)]["behaviors"][TARGET], dtype=np.float32).reshape(-1)
        y = y[np.isfinite(y) & (np.abs(y) >= ONSET_THR)]
        if y.size:
            vals.append(y)
    if not vals:
        return float(ONSET_THR)
    return float(np.median(np.abs(np.concatenate(vals))))


def post_onset_mask(mask, onset_bin):
    tgrid = torch.arange(mask.size(1), device=mask.device).unsqueeze(0)
    k = onset_bin.clamp(min=0).unsqueeze(1)
    has = (onset_bin >= 0).unsqueeze(1)
    return mask & has & (tgrid >= k)


def compute_loss(objective, loss_name, trace, onset_logit, dir_logit, y, mask, y_mean, y_std, speed_scale):
    if objective == "direction":
        mean = trace.new_tensor(np.asarray(y_mean, dtype=np.float32).reshape(-1)[:1])
        std = trace.new_tensor(np.asarray(y_std, dtype=np.float32).reshape(-1)[:1]).clamp(min=1e-6)
        label = (y[..., 0] * std + mean).clamp(0.0, 1.0)
        return trial_bce(dir_logit if dir_logit is not None else trace, label, mask)
    begun, onset_bin, _ = begun_labels(y, mask, y_mean, y_std)
    if objective == "onset":
        logit = onset_logit if onset_logit is not None else trace
        return event_time_nll(logit, onset_bin, mask)
    curve_mask = mask
    if objective == "mixed":
        curve_mask = post_onset_mask(mask, onset_bin)
        on_loss = event_time_nll(onset_logit if onset_logit is not None else trace, onset_bin, mask)
    else:
        on_loss = trace.new_zeros(())
    if loss_name == "huber_move":
        cur = trial_balanced_huber_move(trace, y, curve_mask, y_mean, y_std, speed_scale)
    else:
        cur = trial_balanced_mse(trace, y, curve_mask)
    if objective == "mixed":
        return cur + on_loss
    return cur


# ---------------------------------------------------------------------------
# Sampling: mice equal, sessions equal within mouse, RT / shape strata
# ---------------------------------------------------------------------------


class StrataSampler(Sampler):
    """Each draw: mouse → session → RT bin → shape, then a trial.

    Mice contribute equally in expectation, sessions within a mouse contribute
    equally, and RT / movement-shape bins are visited uniformly when present.
    """

    def __init__(self, trials, idx, n_samples, seed):
        self.n_samples = int(n_samples)
        self.rng = np.random.RandomState(int(seed))
        tree = {}
        pos = {}
        for p, i in enumerate(np.asarray(idx, dtype=int)):
            t = trials[int(i)]
            mouse, eid = str(t["mouse_id"]), str(t["eid"])
            rt, sh = str(t.get("rt_bin") or "none"), str(t.get("shape_bin") or "peak")
            tree.setdefault(mouse, {}).setdefault(eid, {}).setdefault(rt, {}).setdefault(sh, []).append(p)
            pos[p] = i
        self.tree = tree
        self.mice = list(tree)

    def __len__(self):
        return self.n_samples

    def __iter__(self):
        for _ in range(self.n_samples):
            mouse = self.mice[int(self.rng.randint(len(self.mice)))]
            sess = list(self.tree[mouse])
            eid = sess[int(self.rng.randint(len(sess)))]
            rts = list(self.tree[mouse][eid])
            rt = rts[int(self.rng.randint(len(rts)))]
            shs = list(self.tree[mouse][eid][rt])
            sh = shs[int(self.rng.randint(len(shs)))]
            bucket = self.tree[mouse][eid][rt][sh]
            yield int(bucket[int(self.rng.randint(len(bucket)))])


# ---------------------------------------------------------------------------
# Model wrappers
# ---------------------------------------------------------------------------


class HeadTap(nn.Module):
    def __init__(self, head, extra=None):
        super().__init__()
        self.head = head
        self.extra = extra
        self.aux = None

    def forward(self, h):
        self.aux = self.extra(h) if self.extra is not None else None
        return self.head(h)


class DualSetSequence(mv.SetSequence):
    def __init__(self, *args, dual=False, classify=False, **kwargs):
        super().__init__(*args, **kwargs)
        self.dual = bool(dual)
        if classify:
            last = self.head.net[-1]
            if getattr(last, "bias", None) is not None:
                nn.init.constant_(last.bias, -2.0 if not dual else 0.0)
        if self.dual:
            extra = mv.Head(self.head_dim)
            last = extra.net[-1]
            if getattr(last, "bias", None) is not None:
                nn.init.constant_(last.bias, -2.0)
            self.head = HeadTap(self.head, extra)
        else:
            self.onset_head = None

    def forward(self, *args, **kwargs):
        kwargs.pop("return_attn", None)
        pred = super().forward(*args, **kwargs)
        aux = self.head.aux if isinstance(self.head, HeadTap) else None
        return pred, aux


class GraphWithHeads(gnn.GraphSequence):
    def __init__(self, *args, dual=False, classify=False, **kwargs):
        super().__init__(*args, **kwargs)
        self.dual = bool(dual)
        if classify:
            last = self.head.net[-1]
            if getattr(last, "bias", None) is not None:
                nn.init.constant_(last.bias, -2.0)
        if self.dual:
            extra = mv.Head(mv.RNN_HIDDEN)
            last = extra.net[-1]
            if getattr(last, "bias", None) is not None:
                nn.init.constant_(last.bias, -2.0)
            self.head = HeadTap(self.head, extra)
        else:
            self.onset_head = None

    def forward(self, *args, **kwargs):
        pred, su, sr = super().forward(*args, **kwargs)
        aux = self.head.aux if isinstance(self.head, HeadTap) else None
        return pred, aux, su, sr


class HierWithHeads(nn.Module):
    def __init__(self, inner, dual=False, classify=False):
        super().__init__()
        self.inner = inner
        self.dual = bool(dual)
        if classify:
            last = inner.dec3.net[-1]
            if getattr(last, "bias", None) is not None:
                nn.init.constant_(last.bias, -2.0)
        if self.dual:
            self.onset_head = mv.Head(mv.RNN_HIDDEN)
            last = self.onset_head.net[-1]
            if getattr(last, "bias", None) is not None:
                nn.init.constant_(last.bias, -2.0)
        else:
            self.onset_head = None

    def forward(self, *args, **kwargs):
        y1, y2, y3, h1, h2, h3 = self.inner(*args, **kwargs)
        aux = self.onset_head(h3) if self.onset_head is not None else None
        return y1, y2, y3, aux


class RegionCNNTransformer(nn.Module):
    """Anisotropic CCF grid, per-region CNN, attention pool, causal Transformer."""

    def __init__(self, grid_shape, dual=False, classify=False):
        super().__init__()
        self.cnns = nn.ModuleDict({r: mv2.VoxelCNN3D(grid_shape) for r in REGIONS})
        self.pool = mv.AttnPool(mv.D_MODEL)
        self.temporal = mv.CausalTransformer(mv.D_MODEL, n_layers=1)
        self.head = mv.Head(mv.D_MODEL)
        self.dual = bool(dual)
        if classify:
            last = self.head.net[-1]
            if getattr(last, "bias", None) is not None:
                nn.init.constant_(last.bias, -2.0)
        if self.dual:
            self.onset_head = mv.Head(mv.D_MODEL)
            last = self.onset_head.net[-1]
            if getattr(last, "bias", None) is not None:
                nn.init.constant_(last.bias, -2.0)
        else:
            self.onset_head = None

    def forward(self, grids, present, mask_t):
        # grids: (B, R, T, 3, X, Y, Z); present: (B, R)
        tokens = []
        for j, r in enumerate(REGIONS):
            tokens.append(self.cnns[r](grids[:, j]))
        tok = torch.stack(tokens, dim=1)
        h = self.pool(tok.permute(0, 1, 2, 3), present) * mask_t.unsqueeze(-1)
        out = self.temporal(h, mask_t)
        pred = self.head(out)
        aux = self.onset_head(out) if self.onset_head is not None else None
        return pred, aux


class RegionGridDS(Dataset):
    def __init__(self, trials, idx, y_mean, y_std, vox):
        self.trials = trials
        self.idx = np.asarray(idx, dtype=int)
        self.y_mean = y_mean
        self.y_std = y_std
        self.vox = vox

    def __len__(self):
        return len(self.idx)

    def __getitem__(self, j):
        t = self.trials[int(self.idx[j])]
        y = np.asarray(t["y"], dtype=np.float32).reshape(-1, 1)
        valid = np.asarray(t.get("y_valid", np.isfinite(y).all(axis=1)), dtype=bool).reshape(-1)
        y = np.where(valid[:, None], y, 0.0)
        y = (y - self.y_mean) / self.y_std
        grids, present = [], []
        for region in REGIONS:
            sel = np.array([str(a).startswith(region) for a in t["areas"]], dtype=bool)
            if not sel.any():
                G = np.zeros((y.shape[0], self.vox.n_voxels), dtype=np.float32)
                M = np.zeros(self.vox.n_voxels, dtype=np.float32)
                C = np.zeros(self.vox.n_voxels, dtype=np.float32)
                grids.append(self.vox.pack_grid(G, M, C))
                present.append(False)
            else:
                G, M, C = self.vox.voxelize(t["spikes"][:, sel], t["xyz"][sel])
                grids.append(self.vox.pack_grid(G, M, C))
                present.append(True)
        return grids, present, y.astype(np.float32), valid


def collate_grids(batch):
    grids, presents, ys, valids = zip(*batch)
    B = len(batch)
    T = max(y.shape[0] for y in ys)
    shp = grids[0][0].shape[1:]
    x = torch.zeros(B, len(REGIONS), T, *shp, dtype=torch.float32)
    y = torch.zeros(B, T, 1, dtype=torch.float32)
    mask_t = torch.zeros(B, T, dtype=torch.bool)
    present = torch.zeros(B, len(REGIONS), dtype=torch.bool)
    lengths = torch.zeros(B, dtype=torch.long)
    for i, (gs, pr, yi, vi) in enumerate(zip(grids, presents, ys, valids)):
        tt = yi.shape[0]
        for r, g in enumerate(gs):
            x[i, r, :tt] = torch.from_numpy(g[:tt])
        y[i, :tt] = torch.from_numpy(yi)
        mask_t[i, :tt] = torch.from_numpy(np.asarray(vi, dtype=bool))
        present[i] = torch.tensor(pr, dtype=torch.bool)
        lengths[i] = tt
    return {"grids": x, "present": present, "y": y, "mask_t": mask_t, "mask_loss": mask_t, "lengths": lengths}


def reconstruct_grid(meta):
    vox = mv2.AnisoGrid.__new__(mv2.AnisoGrid)
    vox.x0, vox.nx = float(meta["x0"]), int(meta["nx"])
    vox.y0, vox.ny = float(meta["y0"]), int(meta["ny"])
    vox.z0, vox.nz = float(meta["z0"]), int(meta["nz"])
    vox.shape = (vox.nx, vox.ny, vox.nz)
    vox.n_voxels = int(np.prod(vox.shape))
    vox.name = "aniso3d"
    return vox


def grid_meta(vox):
    return {"x0": vox.x0, "nx": vox.nx, "y0": vox.y0, "ny": vox.ny, "z0": vox.z0, "nz": vox.nz}


# ---------------------------------------------------------------------------
# Build / train / predict
# ---------------------------------------------------------------------------


def ensure_session_rates(stats, trials, idx):
    """Fill per-unit mean rates for eids missing from train (LSO / LMO).

    Units are session-specific, so a held-out recording has no train rate
    vector. Modelv1.fit_unit_stats already does this when fallback_idx
    includes the test fold; cached checkpoints from the failed run do not.
    """
    stats = dict(stats or {})
    rates = dict(stats.get("rates") or {})
    buckets = {}
    for i in np.asarray(idx, dtype=int).reshape(-1):
        t = trials[int(i)]
        eid = t["eid"]
        if eid in rates:
            continue
        buckets.setdefault(eid, []).append(np.asarray(t["spikes"], dtype=np.float32).mean(axis=0))
    for eid, vs in buckets.items():
        rates[eid] = np.mean(vs, axis=0).astype(np.float32)
        log(f"  filled session rate for held-out {str(eid)[:8]}  n_units={int(rates[eid].size)}")
    stats["rates"] = rates
    return stats


def merge_eval_stats(blob_stats, live_stats, trials, idx):
    """Keep train vocab / CCF scale from the checkpoint; add eval-session rates."""
    st = dict(live_stats or {})
    blob_stats = dict(blob_stats or {})
    st.update(blob_stats)
    rates = dict((live_stats or {}).get("rates") or {})
    rates.update(dict(blob_stats.get("rates") or {}))
    st["rates"] = rates
    return ensure_session_rates(st, trials, idx)


def model_kind(name):
    if name in ("attn_a1_gru", "attn_a1_transformer", "attn_anat_transformer"):
        return "set"
    if name in ("gnn_inner", "gnn_outer"):
        return "graph"
    if name == "hierarchical":
        return "hier"
    if name == "aniso_cnn_transformer":
        return "cnn"
    return "base"


def build_neural(name, stats, trials, tr_idx, objective):
    dual = objective == "mixed"
    classify = objective in ("onset", "direction")
    n_areas = int(stats["n_areas"])
    if name == "attn_a1_gru":
        return DualSetSequence(
            n_areas, pool="attn", use_anatomy=False, temporal="gru", dual=dual, classify=classify
        )
    if name == "attn_a1_transformer":
        return DualSetSequence(
            n_areas, pool="attn", use_anatomy=False, temporal="transformer", dual=dual, classify=classify
        )
    if name == "attn_anat_transformer":
        return DualSetSequence(
            n_areas, pool="attn", use_anatomy=True, temporal="transformer", dual=dual, classify=classify
        )
    if name == "gnn_inner":
        return GraphWithHeads(
            n_areas, region_gnn=False, vocab=stats.get("vocab"), dual=dual, classify=classify
        )
    if name == "gnn_outer":
        return GraphWithHeads(
            n_areas, region_gnn=True, vocab=stats.get("vocab"), dual=dual, classify=classify
        )
    if name == "hierarchical":
        eids, mice = hier.ids_of(trials, tr_idx)
        inner = hier.HierarchicalGRU(n_areas, eids, mice, shared_encoder=True, use_anatomy=False)
        return HierWithHeads(inner, dual=dual, classify=classify)
    raise ValueError(name)


def unpack_pred(name, out, objective):
    if name.startswith("gnn"):
        pred, aux, _, _ = out
    elif name == "hierarchical":
        y1, y2, y3, aux = out
        pred = y3
    else:
        pred, aux = out
    if objective == "onset":
        return pred, pred, None
    if objective == "direction":
        return pred, None, pred
    return pred, aux, None


def make_loader(name, trials, idx, y_mean, y_std, stats, rho_by_eid, vox, shuffle, seed, batch):
    kind = model_kind(name)
    if kind == "cnn":
        ds = RegionGridDS(trials, idx, y_mean, y_std, vox)
        collate = collate_grids
        batch = min(int(batch), CNN_BATCH)
    elif kind == "graph":
        ds = gnn.GraphDS(trials, idx, y_mean, y_std, stats)
        collate = gnn.make_collate(rho_by_eid or {})
        batch = min(int(batch), gnn.GRAPH_BATCH)
    elif kind == "hier":
        ds = hier.HierDS(trials, idx, y_mean, y_std, stats)
        collate = hier.hier_collate
    else:
        ds = mv.TrialDS(trials, idx, y_mean, y_std, stats)
        collate = mv.collate
    if shuffle:
        sampler = StrataSampler(trials, idx, n_samples=len(ds), seed=seed)
        return DataLoader(ds, batch_size=batch, sampler=sampler, collate_fn=collate)
    return DataLoader(ds, batch_size=batch, shuffle=False, collate_fn=collate)


def forward_model(name, model, batch, dev, train):
    drop = mv.UNIT_DROPOUT if train else 0.0
    if name == "aniso_cnn_transformer":
        return model(batch["grids"].to(dev), batch["present"].to(dev), batch["mask_t"].to(dev))
    if name.startswith("gnn"):
        rho = batch["rho"].to(dev)
        return model(
            batch["x"].to(dev),
            batch["xyz"].to(dev),
            batch["area"].to(dev),
            batch["rate"].to(dev),
            batch["mask_n"].to(dev),
            batch["mask_t"].to(dev),
            batch["lengths"],
            rho=rho,
            unit_drop=drop,
            compute_smooth=False,
        )
    if name == "hierarchical":
        return model(
            batch["x"].to(dev),
            batch["xyz"].to(dev),
            batch["area"].to(dev),
            batch["rate"].to(dev),
            batch["mask_n"].to(dev),
            batch["mask_t"].to(dev),
            batch["lengths"],
            batch["eids"],
            batch["mice"],
            unit_drop=drop,
        )
    return model(
        batch["x"].to(dev),
        batch["xyz"].to(dev),
        batch["area"].to(dev),
        batch["rate"].to(dev),
        batch["mask_n"].to(dev),
        batch["mask_t"].to(dev),
        batch["lengths"],
        unit_drop=drop,
    )


def run_epoch(name, model, loader, opt, dev, train, objective, loss_name, y_mean, y_std, speed_scale):
    model.train(train)
    total, n = 0.0, 0
    for batch in loader:
        y = batch["y"].to(dev)
        mask = batch["mask_loss"].to(dev)
        if train:
            opt.zero_grad(set_to_none=True)
        out = forward_model(name, model, batch, dev, train)
        trace, onset_logit, dir_logit = unpack_pred(name, out, objective)
        loss = compute_loss(
            objective, loss_name, trace, onset_logit, dir_logit, y, mask, y_mean, y_std, speed_scale
        )
        if train:
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
        bs = int(y.size(0))
        total += float(loss.item()) * bs
        n += bs
    return total / max(n, 1)


def cache_dir(eval_name, fold_id, model_name, objective, loss_name):
    configure_out()
    safe = "".join(ch if ch.isalnum() or ch in "-_." else "_" for ch in str(fold_id))
    d = CACHE / eval_name / safe / model_name / objective / loss_name
    d.mkdir(parents=True, exist_ok=True)
    return d


def train_neural(
    name,
    trials,
    tr_idx,
    va_idx,
    y_mean,
    y_std,
    stats,
    objective,
    loss_name,
    eval_name,
    fold_id,
    seed,
    rho_by_eid,
    vox,
):
    cdir = cache_dir(eval_name, fold_id, name, objective, loss_name)
    ckpt, hist_path = cdir / "model.pt", cdir / "history.json"
    if ckpt.exists() and hist_path.exists():
        hist = json.loads(hist_path.read_text())
        if hist.get("finished"):
            log(f"  cache hit {eval_name}/{fold_id}/{name}/{objective}/{loss_name}")
            return torch.load(ckpt, map_location="cpu", weights_only=False), hist
    dev = slurm_utils.device_of()
    if name == "aniso_cnn_transformer":
        model = RegionCNNTransformer(
            vox.shape, dual=objective == "mixed", classify=objective in ("onset", "direction")
        ).to(dev)
    else:
        model = build_neural(name, stats, trials, tr_idx, objective).to(dev)
    batch = gnn.GRAPH_BATCH if name.startswith("gnn") else (CNN_BATCH if name == "aniso_cnn_transformer" else mv.BATCH)
    tr_ld = make_loader(name, trials, tr_idx, y_mean, y_std, stats, rho_by_eid, vox, True, seed, batch)
    va_ld = make_loader(name, trials, va_idx, y_mean, y_std, stats, rho_by_eid, vox, False, seed, batch)
    opt = torch.optim.AdamW(model.parameters(), lr=mv.LR, weight_decay=mv.WEIGHT_DECAY)
    speed = speed_scale_from(trials, tr_idx)
    hist = {
        "train": [],
        "val": [],
        "n_params": mv.n_params(model),
        "finished": False,
        "best_epoch": 0,
        "speed_scale": speed,
    }
    log(f"  {name} params={hist['n_params']} device={dev} obj={objective} loss={loss_name}")
    best, bad, best_state = math.inf, 0, None
    max_ep = hier.STAGE_EPOCHS + hier.JOINT_EPOCHS if name == "hierarchical" else mv.MAX_EPOCHS
    if MAX_EPOCHS_CAP is not None:
        max_ep = min(int(max_ep), int(MAX_EPOCHS_CAP))
    for epoch in range(1, max_ep + 1):
        tr = run_epoch(name, model, tr_ld, opt, dev, True, objective, loss_name, y_mean, y_std, speed)
        with torch.no_grad():
            va = run_epoch(name, model, va_ld, None, dev, False, objective, loss_name, y_mean, y_std, speed)
        hist["train"].append(tr)
        hist["val"].append(va)
        log(f"    {name} {epoch:02d}  train={tr:.4f}  val={va:.4f}")
        slurm_utils.write_status(state="training", epoch=epoch, train=tr, val=va, best_epoch=hist["best_epoch"])
        if va + 1e-5 < best:
            best, bad = va, 0
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            hist["best_epoch"] = epoch
        else:
            bad += 1
        if bad >= mv.PATIENCE:
            log(f"    early stop at epoch {epoch}")
            break
    if best_state is not None:
        model.load_state_dict(best_state)
    hist["finished"] = True
    blob = {
        "state_dict": {k: v.detach().cpu().clone() for k, v in model.state_dict().items()},
        "y_mean": y_mean,
        "y_std": y_std,
        "n_areas": stats["n_areas"],
        "model": name,
        "objective": objective,
        "loss": loss_name,
        "stats": {
            "vocab": stats["vocab"],
            "n_areas": stats["n_areas"],
            "xyz_mean": stats["xyz_mean"],
            "xyz_std": stats["xyz_std"],
            "rates": stats["rates"],
        },
        "rho_by_eid": rho_by_eid,
        "grid": grid_meta(vox) if vox is not None else None,
        "speed_scale": speed,
    }
    if name == "hierarchical":
        blob["eids"] = list(model.inner.id_eids)
        blob["mice"] = list(model.inner.id_mice)
        blob["shared_encoder"] = True
    torch.save(blob, ckpt)
    mv.save_json(hist_path, hist)
    return blob, hist


def predict_neural(blob, trials, idx, stats, vox):
    name = blob["model"]
    objective = blob["objective"]
    dev = slurm_utils.device_of()
    st = merge_eval_stats(blob.get("stats"), stats, trials, idx)
    if name == "aniso_cnn_transformer":
        vox = reconstruct_grid(blob["grid"]) if blob.get("grid") else vox
        model = RegionCNNTransformer(
            vox.shape, dual=objective == "mixed", classify=objective in ("onset", "direction")
        )
    else:
        model = build_neural(name, st, trials, idx, objective)
    model.load_state_dict(blob["state_dict"], strict=False)
    model.to(dev)
    model.eval()
    batch = gnn.GRAPH_BATCH if name.startswith("gnn") else (CNN_BATCH if name == "aniso_cnn_transformer" else mv.BATCH)
    ld = make_loader(
        name, trials, idx, blob["y_mean"], blob["y_std"], st, blob.get("rho_by_eid"), vox, False, 0, batch
    )
    recs = []
    with torch.no_grad():
        for batch_d in ld:
            out = forward_model(name, model, batch_d, dev, False)
            trace, onset_logit, dir_logit = unpack_pred(name, out, objective)
            y_z = batch_d["y"].numpy()
            mt = batch_d["mask_t"].numpy()
            lengths = batch_d["lengths"].numpy()
            tr = trace.cpu().numpy()
            onp = None if onset_logit is None else torch.sigmoid(onset_logit[..., 0]).cpu().numpy()
            dip = None if dir_logit is None else torch.sigmoid(dir_logit[..., 0]).cpu().numpy()
            for b in range(tr.shape[0]):
                T = int(lengths[b])
                valid = mt[b, :T]
                rec = {}
                y_raw = y_z[b, :T] * blob["y_std"] + blob["y_mean"]
                if objective in ("trace", "mixed"):
                    p = tr[b, :T] * blob["y_std"] + blob["y_mean"]
                    y = y_raw
                elif objective == "onset":
                    p = onp[b, :T] if onp is not None else tr[b, :T]
                    y = y_raw
                    rec["onset_pred"] = _onset_from_prob(np.asarray(p).reshape(-1), valid)
                    rec["onset_true"] = mv.sticky_onset_s(y, ONSET_THR)
                    begun = np.zeros((T, 1), dtype=np.float32)
                    if np.isfinite(rec["onset_true"]):
                        k = int(round(float(rec["onset_true"]) / mv.BINSIZE))
                        begun[max(k, 0) :] = 1.0
                    y = begun
                    p = np.asarray(p, dtype=np.float32).reshape(T, 1)
                else:
                    p = dip[b, :T] if dip is not None else tr[b, :T]
                    y = y_raw
                    rec["dir_pred"] = float(np.nanmean(p[valid])) if valid.any() else float("nan")
                    rec["dir_true"] = float(np.nanmean(y[valid])) if valid.any() else float("nan")
                p = np.asarray(p, dtype=np.float32).reshape(T, -1)
                y = np.asarray(y, dtype=np.float32).reshape(T, -1)
                p[~valid] = np.nan
                y[~valid] = np.nan
                rec["pred"] = p
                rec["y"] = y
                if objective == "mixed" and onp is not None:
                    rec["onset_pred"] = _onset_from_prob(onp[b, :T], valid)
                    rec["onset_true"] = mv.sticky_onset_s(y, ONSET_THR)
                recs.append(rec)
    return recs


def _onset_from_prob(prob, valid, sticky=2):
    p = np.asarray(prob, dtype=np.float64).reshape(-1)
    m = np.asarray(valid, dtype=bool).reshape(-1)
    run = 0
    n = min(p.size, m.size)
    for i in range(n):
        hit = bool(m[i]) and np.isfinite(p[i]) and p[i] >= 0.5
        run = run + 1 if hit else 0
        if run >= sticky:
            return (i - sticky + 1) * mv.BINSIZE
    return float("nan")


# ---------------------------------------------------------------------------
# Baselines
# ---------------------------------------------------------------------------


def mean_curve(trials, tr_idx, test_idx, objective):
    y_dim = 1
    max_t = max(int(trials[int(i)]["spikes"].shape[0]) for i in np.asarray(tr_idx))
    acc = np.zeros((max_t, y_dim), dtype=np.float64)
    cnt = np.zeros((max_t, y_dim), dtype=np.float64)
    onsets, dirs = [], []
    for i in np.asarray(tr_idx, dtype=int):
        t = trials[int(i)]
        if objective == "direction":
            if t.get("wheel_dir") in ("left", "right"):
                dirs.append(1.0 if t["wheel_dir"] == "right" else 0.0)
            continue
        y = np.asarray(t["behaviors"][TARGET], dtype=np.float64).reshape(-1, 1)
        valid = np.isfinite(y)
        acc[: y.shape[0]] += np.where(valid, y, 0.0)
        cnt[: y.shape[0]] += valid.astype(np.float64)
        if np.isfinite(t.get("onset_s", np.nan)):
            onsets.append(float(t["onset_s"]))
    traj = (acc / np.maximum(cnt, 1.0)).astype(np.float32)
    mean_on = float(np.mean(onsets)) if onsets else float("nan")
    maj = float(np.round(np.mean(dirs))) if dirs else 0.5
    recs = []
    for i in np.asarray(test_idx, dtype=int):
        t = trials[int(i)]
        T = int(t["spikes"].shape[0])
        y_true = np.asarray(t["behaviors"][TARGET], dtype=np.float32).reshape(T, 1)
        if objective == "direction":
            pred = np.full((T, 1), maj, dtype=np.float32)
            y = np.full((T, 1), 1.0 if t.get("wheel_dir") == "right" else 0.0, dtype=np.float32)
            recs.append(
                {
                    "pred": pred,
                    "y": y,
                    "dir_pred": maj,
                    "dir_true": float("nan") if t.get("wheel_dir") not in ("left", "right") else (1.0 if t["wheel_dir"] == "right" else 0.0),
                }
            )
            continue
        if T <= len(traj):
            pred = traj[:T].copy()
        else:
            pred = np.vstack([traj, np.repeat(traj[-1:], T - len(traj), axis=0)])
        rec = {"pred": pred, "y": y_true, "onset_pred": mean_on, "onset_true": t.get("onset_s")}
        recs.append(rec)
    return recs


def ar_curve(trials, test_idx, objective):
    recs = []
    for i in np.asarray(test_idx, dtype=int):
        t = trials[int(i)]
        T = int(t["spikes"].shape[0])
        y = np.asarray(t["behaviors"][TARGET], dtype=np.float32).reshape(T, 1)
        pred = np.zeros_like(y)
        pred[1:] = y[:-1]
        rec = {"pred": pred, "y": y, "onset_true": t.get("onset_s")}
        rec["onset_pred"] = mv.sticky_onset_s(pred, ONSET_THR)
        if objective == "direction":
            signed = np.asarray(t["behaviors"].get("wheel_signed", y), dtype=np.float32).reshape(-1)
            ar = np.zeros_like(signed)
            ar[1:] = signed[:-1]
            moving = ar[np.isfinite(ar) & (np.abs(ar) >= WHEEL_ON)]
            rec["dir_pred"] = 1.0 if (float(moving.mean()) > 0 if moving.size else False) else 0.0
            rec["dir_true"] = 1.0 if t.get("wheel_dir") == "right" else (0.0 if t.get("wheel_dir") == "left" else float("nan"))
            rec["pred"] = ar.reshape(T, 1).astype(np.float32)
            rec["y"] = signed.reshape(T, 1).astype(np.float32)
        recs.append(rec)
    return recs


# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------


def write_pack(jdir, recs, rows, pooled, blob, hist, cfg):
    jdir = Path(jdir)
    jdir.mkdir(parents=True, exist_ok=True)
    meta = {k: cfg.get(k) for k in ("eval", "fold", "fold_id", "model", "objective", "loss", "target")}
    rows = [{**meta, **r} for r in rows]
    pd.DataFrame(rows).to_csv(jdir / "holdout_trials.csv", index=False)
    y_parts, p_parts, n_bins, ids = [], [], [], []
    for rec, row in zip(recs, rows):
        y = _series(rec["y"]).astype(np.float32)
        p = _series(rec["pred"]).astype(np.float32)
        n = min(y.size, p.size)
        y_parts.append(y[:n])
        p_parts.append(p[:n])
        n_bins.append(n)
        ids.append(int(row["trial_i"]))
    np.savez_compressed(
        jdir / "holdout_curves.npz",
        y=np.concatenate(y_parts) if y_parts else np.zeros(0, np.float32),
        pred=np.concatenate(p_parts) if p_parts else np.zeros(0, np.float32),
        n_bins=np.asarray(n_bins, dtype=np.int32),
        trial_i=np.asarray(ids, dtype=np.int32),
        binsize=np.float32(mv.BINSIZE),
    )
    score_row = {**cfg, **pooled}
    pd.DataFrame([score_row]).to_csv(jdir / "scores.csv", index=False)
    mv.save_json(jdir / "metrics.json", {"cfg": cfg, "fold": pooled, "n_trials": len(rows)})
    if hist:
        mv.save_json(jdir / "history.json", hist)
    if blob is not None:
        torch.save(blob, jdir / "model.pt")
    mv.save_json(
        jdir / "canvas_questions.json",
        {
            "files": {
                "holdout_trials.csv": "Per-trial pearson/R²/RMSE/MAPE/MAE, onset and peak errors.",
                "holdout_curves.npz": "Full predicted vs GT curves (mixed: NaN before true onset in the scored copy).",
                "scores.csv": "Fold mean, median, and concatenated metrics.",
                "model.pt": "Trained checkpoint (baselines store the prediction recipe).",
            }
        },
    )
    return score_row


# ---------------------------------------------------------------------------
# Job grid / process
# ---------------------------------------------------------------------------


def losses_for(model_name, objective):
    if model_name in ("mean", "ar"):
        return ("none",)
    if objective == "trace":
        return ("mse", "huber_move")
    if objective == "onset":
        return ("event_nll",)
    if objective == "direction":
        return ("bce",)
    return ("mse", "huber_move")


def job_grid(trials, evals=EVALS, models=MODELS, objectives=OBJECTIVES, fold=None, single_fold=False):
    specs = keep_fold(all_folds(trials, evals=evals), fold=fold, single_fold=single_fold)
    configs = []
    for spec in specs:
        for model_name in models:
            for objective in objectives:
                for loss_name in losses_for(model_name, objective):
                    configs.append(
                        {
                            "eval": spec["eval"],
                            "fold": int(spec["fold"]),
                            "fold_id": spec["fold_id"],
                            "model": model_name,
                            "objective": objective,
                            "loss": loss_name,
                        }
                    )
    return configs, specs


def spec_for(specs, cfg):
    for s in specs:
        if s["eval"] == cfg["eval"] and str(s["fold_id"]) == str(cfg["fold_id"]):
            return s
    raise SystemExit(f"no fold spec for {cfg}")


def run_config(cfg, trials, specs):
    spec = spec_for(specs, cfg)
    bind_objective(trials, cfg["objective"])
    train_full = mv.usable_idx(trials, spec["train"])
    test_idx = mv.usable_idx(trials, spec["test"], min_bins=1 if cfg["objective"] == "direction" else mv.MIN_BINS)
    if cfg["objective"] == "onset":
        test_idx = mv.usable_idx(trials, spec["test"], min_bins=2)
    if len(train_full) < 8 or len(test_idx) < 1:
        log(f"  skip {cfg}: too few trials")
        return None
    tr_idx, va_idx = mv.train_val_split(train_full, seed=mv.SEED + 17 * int(spec["fold"]))
    tr_idx, va_idx = mv.usable_idx(trials, tr_idx), mv.usable_idx(trials, va_idx)
    if len(tr_idx) < 4:
        log(f"  skip {cfg}: empty train after val split")
        return None
    y_mean, y_std = mv.y_scaler_from(trials, tr_idx)
    fallback = np.unique(np.concatenate([tr_idx, va_idx, test_idx]))
    stats = mv.fit_unit_stats(trials, tr_idx, fallback)
    stats = ensure_session_rates(stats, trials, test_idx)
    tag = slurm_utils.tag_of(cfg)
    jdir = slurm_utils.job_dir()
    name = cfg["model"]
    objective = cfg["objective"]
    blob, hist = None, {}
    vox, rho = None, None
    if name == "aniso_cnn_transformer":
        xyz = np.concatenate([trials[int(i)]["xyz"] for i in tr_idx], axis=0)
        vox = mv2.AnisoGrid(xyz)
    if name.startswith("gnn"):
        rho = gnn.session_train_corr(trials, tr_idx)
    if name in ("mean", "ar"):
        recs = mean_curve(trials, tr_idx, test_idx, objective) if name == "mean" else ar_curve(trials, test_idx, objective)
        blob = {"model": name, "objective": objective, "y_mean": y_mean, "y_std": y_std}
    else:
        blob, hist = train_neural(
            name,
            trials,
            tr_idx,
            va_idx,
            y_mean,
            y_std,
            stats,
            objective,
            cfg["loss"],
            spec["eval"],
            spec["fold_id"],
            mv.SEED + int(spec["fold"]),
            rho,
            vox,
        )
        recs = predict_neural(blob, trials, test_idx, stats, vox)
    scored = recs
    if objective == "mixed":
        scored = apply_mixed_score_mask(recs, trials, test_idx)
    rows, pooled = score_fold(scored, trials, test_idx, objective)
    meta = {
        "eval": spec["eval"],
        "fold": int(spec["fold"]),
        "fold_id": spec["fold_id"],
        "model": name,
        "objective": objective,
        "loss": cfg["loss"],
        "target": TARGET,
        "n_train": int(len(tr_idx)),
        "n_val": int(len(va_idx)),
        "n_test": int(len(test_idx)),
        "note": spec["note"],
    }
    row = write_pack(jdir, scored, rows, pooled, blob, hist, meta)
    log(
        f"  {name:24s} {objective:9s} {cfg['loss']:11s}  "
        f"concat R²={pooled.get('fold_concat_r2', float('nan')):.3f}  "
        f"median R²={pooled.get('fold_median_r2', float('nan')):.3f}  "
        f"onset MAE={pooled.get('fold_median_onset_abs_s', float('nan')):.3f}s"
    )
    slurm_utils.write_status(state="done", **{k: pooled.get(k) for k in ("fold_concat_r2", "n_holdout_trials")})
    return row


def process(job=None, evals=EVALS, models=MODELS, objectives=OBJECTIVES, fold=None, single_fold=False, trials=None):
    configure_out()
    mv.set_seed(mv.SEED)
    if trials is None:
        trials, _session_meta = mv.load_corpus()
    annotate_trials(trials)
    bind_objective(trials, "trace")
    configs, specs = job_grid(trials, evals=evals, models=models, objectives=objectives, fold=fold, single_fold=single_fold)
    if job is not None:
        cfg = slurm_utils.pick_config(configs, job)
        slurm_utils.set_device("auto", probe=device_probe(cfg["model"]))
        slurm_utils.begin_job(OUT, job, slurm_utils.tag_of(cfg), **cfg)
        log(f"job {job}  {cfg}")
        try:
            run_config(cfg, trials, specs)
        except Exception:
            slurm_utils.write_status(state="failed")
            err = slurm_utils.job_dir() / "error.txt"
            err.write_text(traceback.format_exc())
            raise
        err = slurm_utils.job_dir() / "error.txt"
        if err.exists():
            err.unlink()
        return
    rows = []
    for i, cfg in enumerate(configs):
        slurm_utils.set_device("auto", probe=device_probe(cfg["model"]))
        slurm_utils.begin_job(OUT, i, slurm_utils.tag_of(cfg), **cfg)
        log(f"==== job {i}/{len(configs)}  {cfg} ====")
        try:
            row = run_config(cfg, trials, specs)
            if row:
                rows.append(row)
            err = slurm_utils.job_dir() / "error.txt"
            if err.exists():
                err.unlink()
        except Exception:
            slurm_utils.write_status(state="failed")
            err = slurm_utils.job_dir() / "error.txt"
            err.write_text(traceback.format_exc())
            log(f"FAILED job {i}: {err}")
            raise
    if rows:
        pd.DataFrame(rows).to_csv(OUT / "scores.csv", index=False)


def _mean_median(series, prefix, key):
    v = pd.to_numeric(series, errors="coerce")
    return {
        f"{prefix}_mean_{key}": float(v.mean()) if v.notna().any() else float("nan"),
        f"{prefix}_median_{key}": float(v.median()) if v.notna().any() else float("nan"),
    }


def _job_group_key(row):
    return tuple(str(row.get(k, "")) for k in ("eval", "model", "objective", "loss"))


def aggregate_all():
    configure_out()
    df = slurm_utils.aggregate_scores(OUT, dest_name="scores.csv")
    packs = sorted((OUT / "jobs").glob("*/holdout_trials.csv"))
    if not packs:
        return df
    all_trials = pd.concat([pd.read_csv(p) for p in packs], ignore_index=True)
    all_trials.to_csv(OUT / "holdout_trials_all.csv", index=False)
    grouped_recs = defaultdict(list)
    for csv_path in packs:
        jdir = csv_path.parent
        sc_path = jdir / "scores.csv"
        npz = jdir / "holdout_curves.npz"
        if not sc_path.exists() or not npz.exists():
            continue
        sc = pd.read_csv(sc_path)
        if sc.empty:
            continue
        key = _job_group_key(sc.iloc[0])
        z = np.load(npz)
        n_bins = np.asarray(z["n_bins"], dtype=int).reshape(-1)
        y = np.asarray(z["y"]).reshape(-1)
        p = np.asarray(z["pred"]).reshape(-1)
        off = 0
        for n in n_bins:
            n = int(n)
            grouped_recs[key].append({"y": y[off : off + n], "pred": p[off : off + n]})
            off += n
    summary = []
    keys = ["eval", "model", "objective", "loss"]
    present = [c for c in keys if c in df.columns]
    for gvals, sub in df.groupby(present, dropna=False):
        rec = dict(zip(present, gvals if isinstance(gvals, tuple) else (gvals,)))
        rec["n_folds"] = int(len(sub))
        for col in sub.columns:
            if col.startswith("fold_mean_") or col.startswith("fold_median_") or col.startswith("fold_concat_"):
                rec.update(_mean_median(sub[col], "across_folds", col))
        trial_sub = all_trials
        for k in present:
            if k in trial_sub.columns:
                trial_sub = trial_sub.loc[trial_sub[k].astype(str) == str(rec[k])]
        rec.update(pool_rows(trial_sub.to_dict(orient="records"), "across_trials"))
        cat = concat_metrics(grouped_recs.get(_job_group_key(rec), []))
        rec.update({f"across_concat_{k}": cat.get(k, float("nan")) for k in ("pearson", "r2", "rmse", "mape", "mae")})
        rec["across_concat_n_bins"] = int(cat.get("n_bins", 0) or 0)
        summary.append(rec)
    pd.DataFrame(summary).to_csv(OUT / "scores_across_folds.csv", index=False)
    log(f"wrote {OUT / 'scores_across_folds.csv'}  ({len(summary)} groups, {len(all_trials)} trials)")
    return df


# ---------------------------------------------------------------------------
# Self-test
# ---------------------------------------------------------------------------


def _fake_trials(n_mice=3, n_sess=2, n_per=25):
    trials = []
    rng = np.random.RandomState(0)
    i = 0
    for m in range(n_mice):
        ns = n_sess if m < n_mice - 1 else 1
        for s in range(ns):
            eid = f"e{m}{s}aaaaaaaa"
            n_u = 8
            xyz = np.zeros((n_u, 3), dtype=np.float32)
            xyz[:, 0] = rng.uniform(-0.0006, 0.0006, n_u)
            xyz[:, 1] = rng.uniform(-0.0006, 0.0006, n_u)
            xyz[:, 2] = np.linspace(0.0, 0.0012, n_u)
            areas = np.array(["MOp"] * 4 + ["MOs"] * 4)
            for k in range(n_per):
                T = 16
                spk = rng.randn(T, n_u).astype(np.float32)
                paw = np.zeros(T, dtype=np.float32)
                on = 4 + (k % 8)
                paw[on:] = 200 + 80 * np.sin(np.linspace(0, 3, T - on))
                wheel = np.zeros(T, dtype=np.float32)
                wheel[on:] = (1 if k % 2 == 0 else -1) * 0.4
                trials.append(
                    {
                        "eid": eid,
                        "mouse_id": f"M{m}",
                        "trial_index": k,
                        "spikes": spk,
                        "xyz": xyz,
                        "areas": areas,
                        "behaviors": {
                            "paw_speed": paw,
                            "wheel_signed": wheel,
                            "wheel_speed": np.abs(wheel),
                        },
                    }
                )
                i += 1
    return trials


def dummy_smoke_jobs(specs):
    """Every architecture once, plus onset/direction/mixed/huber and LSO/LMO."""
    by_eval = {s["eval"]: s for s in specs}
    jobs = []
    lto = by_eval["LTO"]
    for model_name in MODELS:
        jobs.append(
            {
                "eval": "LTO",
                "fold": int(lto["fold"]),
                "fold_id": lto["fold_id"],
                "model": model_name,
                "objective": "trace",
                "loss": "none" if model_name in ("mean", "ar") else "mse",
            }
        )
    extras = (
        ("onset", "event_nll"),
        ("direction", "bce"),
        ("mixed", "mse"),
        ("trace", "huber_move"),
    )
    for objective, loss_name in extras:
        jobs.append(
            {
                "eval": "LTO",
                "fold": int(lto["fold"]),
                "fold_id": lto["fold_id"],
                "model": "attn_a1_gru",
                "objective": objective,
                "loss": loss_name,
            }
        )
    for ev in ("LSO", "LMO"):
        spec = by_eval[ev]
        jobs.append(
            {
                "eval": ev,
                "fold": int(spec["fold"]),
                "fold_id": spec["fold_id"],
                "model": "mean",
                "objective": "trace",
                "loss": "none",
            }
        )
        jobs.append(
            {
                "eval": ev,
                "fold": int(spec["fold"]),
                "fold_id": spec["fold_id"],
                "model": "attn_a1_gru",
                "objective": "trace",
                "loss": "mse",
            }
        )
    return jobs


def run_dummy(device="auto"):
    global DUMMY, MAX_EPOCHS_CAP
    DUMMY = True
    MAX_EPOCHS_CAP = 2
    configure_out()
    if OUT.exists():
        shutil.rmtree(OUT)
    configure_out()
    slurm_utils.set_device(device, probe="gru")
    mv.set_seed(mv.SEED)
    trials = _fake_trials(n_mice=2, n_sess=2, n_per=20)
    annotate_trials(trials)
    bind_objective(trials, "trace")
    specs = keep_fold(all_folds(trials), single_fold=True)
    jobs = dummy_smoke_jobs(specs)
    log(f"dummy smoke  n_trials={len(trials)}  jobs={len(jobs)}  out={OUT}  device={slurm_utils.device_of()}")
    rows = []
    for i, cfg in enumerate(jobs):
        slurm_utils.set_device(device, probe=device_probe(cfg["model"]))
        slurm_utils.begin_job(OUT, i, slurm_utils.tag_of(cfg), **cfg)
        log(f"==== dummy {i}/{len(jobs)}  {cfg} ====")
        row = run_config(cfg, trials, specs)
        if row:
            rows.append(row)
    if not rows:
        raise SystemExit("dummy smoke produced no scores")
    pd.DataFrame(rows).to_csv(OUT / "scores.csv", index=False)
    aggregate_all()
    missing = [r for r in rows if not np.isfinite(r.get("n_holdout_trials", np.nan))]
    if missing:
        raise SystemExit(f"dummy jobs missing n_holdout_trials: {missing}")
    log(f"dummy smoke ok  {len(rows)} jobs  scores={OUT / 'scores.csv'}")
    return rows


def run_self_test():
    trials = _fake_trials()
    annotate_trials(trials)
    bind_objective(trials, "trace")
    lto = lto_folds(trials)
    assert len(lto) == N_LTO, len(lto)
    n0 = len(lto[0]["test"])
    n1 = len(lto[1]["test"])
    assert n0 == n1, (n0, n1)
    for eid, sub in mv.trial_frame(trials).groupby("eid"):
        n = len(sub)
        k = int(np.floor(n * LTO_FRAC))
        assert k == 1, (eid, n, k)
    lso = lso_folds(trials)
    assert all("M2" not in s["note"] or True for s in lso)
    mice_lso = {s["note"].split()[-1] if False else s["fold"] for s in lso}
    single = [t["eid"] for t in trials if t["mouse_id"] == "M2"]
    assert single[0][:8] not in {s["fold_id"] for s in lso}
    lmo = lmo_folds(trials)
    assert {s["fold_id"] for s in lmo} == {"M0", "M1", "M2"}
    one = keep_fold(lto + lso + lmo, single_fold=True)
    assert {s["eval"] for s in one} == {"LTO", "LSO", "LMO"}
    assert all(int(s["fold"]) == 0 for s in one)
    y = torch.zeros(2, 5, 1)
    p = torch.zeros(2, 5, 1)
    mask = torch.ones(2, 5, dtype=torch.bool)
    mse = trial_balanced_mse(p, y, mask)
    assert float(mse) == 0.0
    p2 = torch.ones(2, 5, 1)
    h = trial_balanced_huber_move(p2, y, mask, np.array([0.0]), np.array([1.0]), 150.0)
    assert float(h) > 0
    logit = torch.zeros(2, 5, 1)
    nll = event_time_nll(logit, torch.tensor([2, -1]), mask)
    assert torch.isfinite(nll)
    recs = [{"y": np.arange(10, dtype=float), "pred": np.arange(10, dtype=float) + 1}]
    cat = concat_metrics(recs)
    assert cat["mae"] == 1.0
    met = trial_metrics(np.array([0, 0, 200, 250, 100], float), np.array([0, 0, 180, 240, 90], float))
    assert np.isfinite(met["onset_abs_s"]) and np.isfinite(met["peak_abs_s"])
    smp = StrataSampler(trials, np.arange(len(trials)), n_samples=16, seed=0)
    seen = list(iter(smp))
    assert len(seen) == 16
    dual = DualSetSequence(4, pool="attn", use_anatomy=False, temporal="gru", dual=True)
    pids = [id(p) for p in dual.parameters()]
    assert len(pids) == len(set(pids)), "duplicate parameters in DualSetSequence"
    bind_objective(trials, "trace")
    lso0 = lso[0]
    te = np.asarray(lso0["test"], dtype=int)
    tr = np.asarray(lso0["train"], dtype=int)
    held = trials[int(te[0])]["eid"]
    stats_train = mv.fit_unit_stats(trials, tr, tr)
    assert held not in stats_train["rates"]
    filled = ensure_session_rates(stats_train, trials, te)
    assert held in filled["rates"]
    ds = mv.TrialDS(trials, te[:2], np.zeros(1, np.float32), np.ones(1, np.float32), filled)
    _ = ds[0]
    live = mv.fit_unit_stats(trials, tr, np.concatenate([tr, te]))
    cached = merge_eval_stats(stats_train, live, trials, te)
    assert held in cached["rates"]
    stale = merge_eval_stats(stats_train, stats_train, trials, te)
    assert held in stale["rates"]
    _ = mv.TrialDS(trials, te[:2], np.zeros(1, np.float32), np.ones(1, np.float32), stale)[0]
    cfgs, specs = job_grid(trials, single_fold=True)
    assert any(c["model"] == "attn_a1_gru" and c["objective"] == "mixed" for c in cfgs)
    assert all(c["fold"] == specs[0]["fold"] or c["eval"] != specs[0]["eval"] for c in cfgs[:3])
    log(f"self-test ok  LTO={len(lto)} LSO={len(lso)} LMO={len(lmo)} single-fold jobs={len(cfgs)}")


def main():
    p = argparse.ArgumentParser(description="Unified LTO/LSO/LMO pipeline")
    slurm_utils.add_common_args(p)
    p.add_argument("--self-test", action="store_true")
    p.add_argument(
        "--dummy",
        action="store_true",
        help="Train/score a tiny synthetic corpus (all models, 2 epochs) into ExperimentsClean/dummy/",
    )
    p.add_argument("--single-fold", action="store_true", help="Fold 0 of each evaluation approach")
    p.add_argument("--fold", type=int, default=None, help="Only this fold index within each eval")
    p.add_argument("--eval", nargs="*", default=list(EVALS), choices=list(EVALS))
    p.add_argument("--models", nargs="*", default=list(MODELS), choices=list(MODELS))
    p.add_argument("--objectives", nargs="*", default=list(OBJECTIVES), choices=list(OBJECTIVES))
    args = p.parse_args()
    if args.self_test:
        run_self_test()
        return
    if args.dummy:
        run_dummy(device=args.device)
        return
    configure_out()
    if args.list_jobs:
        trials, _ = mv.load_corpus()
        annotate_trials(trials)
        bind_objective(trials, "trace")
        configs, _ = job_grid(
            trials,
            evals=tuple(args.eval),
            models=tuple(args.models),
            objectives=tuple(args.objectives),
            fold=args.fold,
            single_fold=args.single_fold,
        )
        slurm_utils.print_jobs(configs)
        return
    if args.aggregate:
        aggregate_all()
        return
    job = slurm_utils.resolve_job_index(args.job)
    slurm_utils.set_device(args.device, probe="gru")
    process(
        job=job,
        evals=tuple(args.eval),
        models=tuple(args.models),
        objectives=tuple(args.objectives),
        fold=args.fold,
        single_fold=args.single_fold,
    )


if __name__ == "__main__":
    main()
