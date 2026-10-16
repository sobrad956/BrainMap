"""Modelv1 cluster experiment: attn A1 GRU, loss × sampling, atlas slices.

Copy of the Modelv1 training stack (imports Modelv1) with a fixed architecture
and a 2×2 training comparison.

Architecture (every job):
  GRU + attention pool, Control A1 (no CCF xyz / region), one target at a time
  (wheel_speed or paw_speed).

Tasks:
  trial_repeat   within-trial 10% bin interpolation (default 1 repeat)
  session_loso   leave one session out, but only if that mouse has another
                 session in the corpus (never a fold that is secretly LOMO)
  mouse_lomo     leave one mouse out

Comparison axes (train only; no trials are dropped):
  loss       mse          trial-balanced MSE (PDF eq. 7)
             onset_peak   that MSE plus differentiable timing terms:
                          soft-argmax peak-time L1, soft-onset L1 (seconds,
                          scaled into MSE units), and a Gaussian peak-window MSE
  sampling   uniform      shuffle, one pass per epoch
             inverse      WeightedRandomSampler, p ∝ 1/n_stratum, weights
                          clipped so max/min ≤ INV_MAX_RATIO, then renormalized.
                          Strata on the current fold's train indices:
                          contrast × length × early/delayed onset

Checkpoints are stored under cache/v2 so the original discrete-L1 grid is not reused.

Local:
  python modelv1clusters.py --self-test
  python modelv1clusters.py --smoke --device mps
  python modelv1clusters.py --pilot --list-jobs
  python modelv1clusters.py --list-jobs
  python modelv1clusters.py --job 0 --device cpu
  python modelv1clusters.py --aggregate

Reports per fold, then mean / median across folds with delete-1 jackknife SE:
  concatenated R², mean trial R², median trial R²
  the same three sliced by coarse-axis level and by Trajectory Atlas Removed
  cluster (wheel clusters for wheel_speed, paw clusters for paw_speed)

Slurm (iLab; results under Modelv1ClustersV2):
  python modelv1clusters.py --pilot --list-jobs
  bash slurm/submit.sh modelv1clusters.py --pilot
  bash slurm/submit.sh modelv1clusters.py --pilot --loso-folds 5 --lomo-folds 5
  bash slurm/submit.sh modelv1clusters.py --pilot --onset-rt
  bash slurm/submit.sh modelv1clusters.py
  python modelv1clusters.py --aggregate

Onset-RT mode (--onset-rt): causal GRU still sees neural bins ≤ t, but the
readout is a per-bin Bernoulli logit “has movement begun?”. Horizon axis:
  now   logit_t  → begun_t     (current 20 ms bin is allowed)
  next  logit_t  → begun_{t+1} (current bin’s spikes cannot label themselves)
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import warnings
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, WeightedRandomSampler

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import Modelv1 as mv
import slurm_utils

ATLAS_ROOT = ROOT / "Modelv1Clusters"
mv.MODEL_ROOT = ROOT / "Modelv1ClustersV2"
CLUSTER_JSON = ATLAS_ROOT / "atlas_trial_clusters.json"

TARGETS = ("wheel_speed", "paw_speed")
TASKS = ("trial_repeat", "session_loso", "mouse_lomo")
LOSSES = ("mse", "onset_peak")
SAMPLINGS = ("uniform", "inverse")
HORIZONS = ("now", "next")
POOL = "attn"
ABLATION = "a1"
CACHE_VERSION = "v2"

WHEEL_ON = 0.15
PAW_ON = 150.0
ONSET_DELAYED_S = 0.40
LEN_SHORT = 20
LEN_MID = 32
LAM_ONSET = 1.0
LAM_PEAK = 1.0
# Softmax temperature in z-space for peak time. Smaller → closer to argmax.
SOFT_PEAK_TAU = 0.4
# Onset sigmoid temperature as a fraction of the raw-unit threshold.
SOFT_ONSET_FRAC = 0.25
# Seconds → loss units so ~200 ms of timing error is O(1), like z-space MSE.
TIME_LOSS_SCALE = 5.0
PEAK_WINDOW_SIGMA = 2.0
INV_MAX_RATIO = 10.0
SMOKE_FOLDS = 8
PILOT_LOSO = 5
PILOT_LOMO = 5
PILOT_INTERP = 1
CONTRAST_KEYS = (625, 1250, 2500, 10000)
CONTRAST_LABEL = {
    625: "6.25%",
    1250: "12.5%",
    2500: "25%",
    10000: "100%",
}


def log(msg):
    mv.log(msg)


def onset_bins(y, thr, sticky=2):
    y = np.asarray(y, dtype=np.float32).reshape(-1)
    above = np.isfinite(y) & (np.abs(y) >= thr)
    run = 0
    for i, a in enumerate(above):
        run = run + 1 if a else 0
        if run >= sticky:
            return i - sticky + 1
    return None


def br_key(brightness):
    if brightness is None:
        return None
    try:
        b = float(brightness)
    except (TypeError, ValueError):
        return None
    if not np.isfinite(b):
        return None
    return int(round(b * 10000))


def contrast_label(key):
    if key in CONTRAST_LABEL:
        return CONTRAST_LABEL[key]
    return "unknown"


def length_bin(n):
    n = int(n)
    if n <= LEN_SHORT:
        return "short"
    if n <= LEN_MID:
        return "mid"
    return "long"


def onset_bin(onset_s):
    if onset_s is None or not np.isfinite(onset_s):
        return "early"
    return "delayed" if float(onset_s) >= ONSET_DELAYED_S else "early"


def load_cluster_table():
    if not CLUSTER_JSON.exists():
        raise SystemExit(
            f"missing {CLUSTER_JSON}. It is written next to this script under "
            "Modelv1Clusters/atlas_trial_clusters.json."
        )
    blob = json.loads(CLUSTER_JSON.read_text())
    by = {}
    for rec in blob["trials"]:
        tid = rec.get("trial_index")
        tid = None if tid is None else int(tid)
        by[(str(rec["eid"]), tid)] = rec
    return blob, by


def annotate_trials(trials, target):
    """Attach contrast, length, onset, stratum, and atlas cluster to each trial."""
    blob, by = load_cluster_table()
    names = blob["wheel_names"] if target == "wheel_speed" else blob["paw_names"]
    n_miss = 0
    for t in trials:
        tid = t.get("trial_index")
        tid = None if tid is None else int(tid)
        rec = by.get((str(t["eid"]), tid))
        n = int(t["spikes"].shape[0])
        y = np.asarray(t["behaviors"][target], dtype=np.float32).reshape(-1)
        thr = WHEEL_ON if target == "wheel_speed" else PAW_ON
        on = onset_bins(y, thr)
        on_s = None if on is None else float(on) * mv.BINSIZE
        br = None if rec is None else rec.get("brightness")
        if br is None:
            br = t.get("brightness")
        ck = br_key(br)
        cl = -2
        if rec is not None:
            cl = int(rec["wheel_c"] if target == "wheel_speed" else rec["paw_c"])
        else:
            n_miss += 1
        t["brightness"] = br
        t["contrast_key"] = ck
        t["contrast"] = contrast_label(ck)
        t["len_bin"] = length_bin(n)
        t["onset_s"] = on_s
        t["onset_bin"] = onset_bin(on_s)
        t["stratum"] = f"{t['contrast']}|{t['len_bin']}|{t['onset_bin']}"
        t["atlas_c"] = cl
        if cl >= 0:
            t["atlas_name"] = names[cl] if cl < len(names) else f"c{cl}"
        elif cl == -1:
            t["atlas_name"] = "unclustered"
        else:
            t["atlas_name"] = "not_in_atlas"
    if n_miss:
        log(f"annotate: {n_miss} trials missing from atlas sidecar (matched on eid, trial_index)")
    return blob


def inverse_weights(trials, idx, target=None, max_ratio=INV_MAX_RATIO):
    """p ∝ 1/n_stratum on this train index list. Epoch size stays len(idx).

    After the 1/n weights, clip so max/min ≤ `max_ratio` and renormalize.
    Uncapped inverse over-weights singleton delayed/long strata (~50–100×)
    and starves the abundant short-early trials the pooled metric cares about.
    """
    idx = np.asarray(idx, dtype=int)
    counts = defaultdict(int)
    keys = []
    for i in idx:
        s = trials[int(i)]["stratum"]
        keys.append(s)
        counts[s] += 1
    w = np.array([1.0 / counts[s] for s in keys], dtype=np.float64)
    w *= len(w) / max(w.sum(), 1e-12)
    raw_ratio = float(w.max() / max(w.min(), 1e-12))
    if max_ratio and raw_ratio > max_ratio:
        cap = float(w.min() * max_ratio)
        w = np.clip(w, 0.0, cap)
        w *= len(w) / max(w.sum(), 1e-12)
    ratio = float(w.max() / max(w.min(), 1e-12))
    log(
        f"    inverse sampling: {len(counts)} strata in train, "
        f"weight min/max={w.min():.3f}/{w.max():.3f}  "
        f"max/min={ratio:.1f} (raw {raw_ratio:.1f}, cap {max_ratio:g})"
    )
    return torch.as_tensor(w, dtype=torch.double)


def cv_folds_clusters(trials, n_trial_repeats=1):
    """Interpolation, multi-session LOSO, and LOMO."""
    df = mv.trial_frame(trials)
    out = []
    all_idx = df.i.to_numpy(dtype=int)
    for k in range(int(n_trial_repeats)):
        seed = mv.SEED + 1000 + k
        out.append(
            {
                "task": "trial_repeat",
                "fold": k,
                "fold_id": f"rep{k}_interp",
                "train": all_idx,
                "test": all_idx,
                "bin_mode": "interp",
                "bin_seed": seed,
                "note": (
                    f"all {len(all_idx)} trials; hold out {mv.EXTRAP_FRAC:.0%} of finite "
                    f"bins at random (seed={seed}); interpolate"
                ),
                "hold_eids": [],
            }
        )
    n_sess = df.groupby("mouse_id")["eid"].nunique().to_dict()
    k = 0
    skipped = 0
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
                "task": "session_loso",
                "fold": k,
                "fold_id": str(eid)[:8],
                "train": train,
                "test": test,
                "note": (
                    f"left-out session {str(eid)[:8]} of {mouse} "
                    f"({int(n_sess[mouse])} sessions; not a mouse holdout)"
                ),
                "hold_eids": [eid],
            }
        )
        k += 1
    log(f"session_loso: {k} folds from multi-session mice; skipped {skipped} single-session eids")
    k = 0
    for mouse in sorted(df.mouse_id.astype(str).unique()):
        test = df.loc[df.mouse_id.astype(str) == mouse, "i"].to_numpy(dtype=int)
        train = df.loc[df.mouse_id.astype(str) != mouse, "i"].to_numpy(dtype=int)
        if len(test) < 1 or len(train) < 8:
            continue
        n = int(df.loc[df.mouse_id.astype(str) == mouse, "eid"].nunique())
        out.append(
            {
                "task": "mouse_lomo",
                "fold": k,
                "fold_id": str(mouse),
                "train": train,
                "test": test,
                "note": f"left-out mouse {mouse} ({n} session(s))",
                "hold_eids": sorted(df.loc[df.mouse_id.astype(str) == mouse, "eid"].unique()),
            }
        )
        k += 1
    log(
        f"cv folds: {sum(1 for s in out if s['task']=='trial_repeat')} trial_repeat, "
        f"{sum(1 for s in out if s['task']=='session_loso')} session_loso, "
        f"{sum(1 for s in out if s['task']=='mouse_lomo')} mouse_lomo"
    )
    return out


def subsample_folds(folds, n_loso=None, n_lomo=None, n_interp=None, seed=0):
    """Deterministic subset of CV folds. Same seed → same job indices on every worker."""
    grouped = {"trial_repeat": [], "session_loso": [], "mouse_lomo": []}
    for spec in folds:
        grouped.setdefault(spec["task"], []).append(spec)
    rng = np.random.RandomState(int(mv.SEED) + int(seed))

    def pick(task, n):
        specs = list(grouped.get(task, []))
        if n is None or int(n) < 0 or int(n) >= len(specs):
            return specs
        n = int(n)
        if n == 0:
            return []
        chosen = [specs[int(i)] for i in rng.choice(len(specs), size=n, replace=False)]
        chosen.sort(key=lambda s: (int(s["fold"]), str(s["fold_id"])))
        return chosen

    out = pick("trial_repeat", n_interp) + pick("session_loso", n_loso) + pick("mouse_lomo", n_lomo)
    log(
        f"fold subset seed={seed}  "
        f"interp={','.join(s['fold_id'] for s in out if s['task']=='trial_repeat') or '-'}  "
        f"loso={','.join(s['fold_id'] for s in out if s['task']=='session_loso') or '-'}  "
        f"lomo={','.join(s['fold_id'] for s in out if s['task']=='mouse_lomo') or '-'}"
    )
    return out


def job_grid(
    trials=None,
    n_trial_repeats=1,
    targets=None,
    tasks=None,
    losses=None,
    samplings=None,
    smoke=False,
    smoke_folds=SMOKE_FOLDS,
    pilot=False,
    loso_folds=None,
    lomo_folds=None,
    interp_folds=None,
    fold_seed=0,
    onset_rt=False,
    horizons=None,
):
    if trials is None:
        trials, _ = mv.load_corpus()
    folds = cv_folds_clusters(trials, n_trial_repeats=n_trial_repeats)
    targets = list(targets or TARGETS)
    tasks = list(tasks or TASKS)
    losses = list(losses or LOSSES)
    samplings = list(samplings or SAMPLINGS)
    if smoke:
        targets = ["wheel_speed"]
        tasks = ["session_loso"]
        n_keep = max(1, int(smoke_folds))
        folds = [
            spec
            for spec in folds
            if spec["task"] == "session_loso" and int(spec["fold"]) < n_keep
        ]
    else:
        n_loso = n_lomo = n_interp = None
        if pilot:
            n_loso = PILOT_LOSO if loso_folds is None else int(loso_folds)
            n_lomo = PILOT_LOMO if lomo_folds is None else int(lomo_folds)
            n_interp = PILOT_INTERP if interp_folds is None else int(interp_folds)
        else:
            n_loso = None if loso_folds is None else int(loso_folds)
            n_lomo = None if lomo_folds is None else int(lomo_folds)
            n_interp = None if interp_folds is None else int(interp_folds)
        if any(v is not None for v in (n_loso, n_lomo, n_interp)):
            folds = subsample_folds(
                folds, n_loso=n_loso, n_lomo=n_lomo, n_interp=n_interp, seed=fold_seed
            )
    onset_rt = bool(onset_rt)
    if onset_rt:
        losses = ["bce"]
        horizons = list(horizons or HORIZONS)
    else:
        horizons = [None]
    configs = []
    for beh in targets:
        for spec in folds:
            if spec["task"] not in tasks:
                continue
            for loss in losses:
                for sampling in samplings:
                    for horizon in horizons:
                        cfg = {
                            "protocol": "cv",
                            "target": beh,
                            "task": spec["task"],
                            "pool": POOL,
                            "ablation": ABLATION,
                            "loss": loss,
                            "sampling": sampling,
                            "fold": int(spec["fold"]),
                            "fold_id": spec["fold_id"],
                        }
                        if onset_rt:
                            cfg["head"] = "onset"
                            cfg["horizon"] = horizon
                        configs.append(cfg)
    return configs


def sticky_onset_times(series, mask, thr, binsize, sticky=2):
    """First time |series| stays ≥ thr for `sticky` bins. NaN if never."""
    above = mask & (series.abs() >= thr)
    B, T = series.shape
    run = torch.zeros(B, dtype=torch.long, device=series.device)
    found = torch.zeros(B, dtype=torch.bool, device=series.device)
    on = torch.full((B,), float("nan"), device=series.device, dtype=series.dtype)
    for t in range(T):
        run = torch.where(above[:, t], run + 1, torch.zeros_like(run))
        hit = (~found) & (run >= sticky)
        if hit.any():
            on = torch.where(hit, torch.full_like(on, (t - sticky + 1) * binsize), on)
            found = found | hit
        if bool(found.all()):
            break
    return on, found


def _masked_softmax(logits, mask):
    return torch.softmax(logits.masked_fill(~mask, -1e9), dim=1)


def soft_peak_time(series, mask, binsize, tau):
    """Expected time of |series| mass. Differentiable in `series`."""
    tau = max(float(tau), 1e-4)
    w = _masked_softmax(series.abs() / tau, mask)
    tgrid = torch.arange(series.size(1), device=series.device, dtype=series.dtype) * binsize
    return (w * tgrid).sum(dim=1)


def hard_peak_time(series, mask, binsize):
    filled = series.abs().masked_fill(~mask, float("-inf"))
    return filled.argmax(dim=1).to(dtype=series.dtype) * binsize


def soft_peak_l1(pred, y, mask, binsize, tau):
    ok = mask.any(dim=1)
    if not bool(ok.any()):
        return pred.new_zeros(())
    return (soft_peak_time(pred, mask, binsize, tau)[ok] - hard_peak_time(y, mask, binsize)[ok]).abs().mean()


def peak_window_mse(pred, y, mask, sigma=PEAK_WINDOW_SIGMA):
    """z-space MSE concentrated around the true |y| peak (dense amplitude grad)."""
    idx = y.abs().masked_fill(~mask, float("-inf")).argmax(dim=1).to(dtype=pred.dtype)
    t = torch.arange(pred.size(1), device=pred.device, dtype=pred.dtype)
    w = torch.exp(-0.5 * ((t.unsqueeze(0) - idx.unsqueeze(1)) / max(float(sigma), 0.5)) ** 2)
    w = w * mask.to(dtype=pred.dtype)
    w = w / w.sum(dim=1, keepdim=True).clamp(min=1e-6)
    return ((pred - y).pow(2) * w).sum(dim=1).mean()


def soft_onset_time(series, mask, thr, binsize, tau, sticky=2):
    """Expected first sticky-crossing time. Differentiable in `series`."""
    bsz, n_t = series.shape
    tau = max(float(tau), 1e-6)
    p_above = torch.sigmoid((series.abs() - thr) / tau) * mask.to(dtype=series.dtype)
    if sticky >= 2:
        hit = p_above.clone()
        for k in range(1, sticky):
            hit = hit * torch.cat([p_above[:, k:], p_above.new_zeros(bsz, k)], dim=1)
    else:
        hit = p_above
    hit = hit.clamp(1e-6, 1.0 - 1e-6)
    log_surv = torch.cumsum(torch.log1p(-hit), dim=1)
    log_before = torch.cat([hit.new_zeros(bsz, 1), log_surv[:, :-1]], dim=1)
    p_t = hit * torch.exp(log_before)
    p_never = torch.exp(log_surv[:, -1])
    tgrid = torch.arange(n_t, device=series.device, dtype=series.dtype) * binsize
    last = mask.to(dtype=series.dtype).sum(dim=1).clamp(min=1.0) * binsize
    return (p_t * tgrid).sum(dim=1) + p_never * last


def soft_onset_l1(pred, y, mask, thr, binsize, tau, sticky=2):
    on_y, has_y = sticky_onset_times(y, mask, thr, binsize, sticky=sticky)
    if not bool(has_y.any()):
        return pred.new_zeros(())
    on_p = soft_onset_time(pred, mask, thr, binsize, tau, sticky=sticky)
    return (on_p[has_y] - on_y[has_y]).abs().mean()


def cluster_objective(pred, y, mask_loss, y_mean, y_std, thr, lengths, use_aux, lam_on, lam_pk):
    mse = mv.trial_balanced_mse(pred, y, mask_loss)
    if not use_aux:
        z = pred.new_zeros(())
        return mse, mse, z, z
    p_z = pred[..., 0]
    y_z = y[..., 0]
    pk_t = soft_peak_l1(p_z, y_z, mask_loss, mv.BINSIZE, SOFT_PEAK_TAU)
    pk_w = peak_window_mse(p_z, y_z, mask_loss, PEAK_WINDOW_SIGMA)
    mean = pred.new_tensor(np.asarray(y_mean, dtype=np.float32)).view(1, 1, -1)
    std = pred.new_tensor(np.asarray(y_std, dtype=np.float32)).view(1, 1, -1)
    y0 = (y * std + mean)[..., 0]
    p0 = (pred * std + mean)[..., 0]
    tau_on = max(float(thr) * SOFT_ONSET_FRAC, 1e-6)
    on = soft_onset_l1(p0, y0, mask_loss, thr, mv.BINSIZE, tau_on)
    aux = lam_on * TIME_LOSS_SCALE * on + lam_pk * TIME_LOSS_SCALE * pk_t + lam_pk * pk_w
    return mse + aux, mse, on, pk_t


def begun_step_labels(y, mask, y_mean, y_std, thr, binsize):
    """Per-bin 1 if movement has begun (sticky onset on the speed trace)."""
    mean = y.new_tensor(np.asarray(y_mean, dtype=np.float32)).view(1, 1, -1)
    std = y.new_tensor(np.asarray(y_std, dtype=np.float32)).view(1, 1, -1)
    y0 = (y * std + mean)[..., 0]
    on, has = sticky_onset_times(y0, mask, thr, binsize)
    tgrid = torch.arange(y0.size(1), device=y.device, dtype=y0.dtype) * binsize
    label = (tgrid.unsqueeze(0) >= on.unsqueeze(1)) & has.unsqueeze(1) & mask
    return label.to(dtype=y.dtype), has


def shift_horizon(label, mask, horizon):
    """Align targets to a causal readout.

    now:  logit_t ↔ begun_t  (neural bins ≤ t, including the current 20 ms)
    next: logit_t ↔ begun_{t+1}  (the labeled bin’s spikes are not in the input)
    """
    if horizon != "next":
        return label, mask
    lab = torch.zeros_like(label)
    msk = torch.zeros_like(mask)
    lab[:, :-1] = label[:, 1:]
    msk[:, :-1] = mask[:, :-1] & mask[:, 1:]
    return lab, msk


def onset_bce(pred, label, mask):
    """Trial-balanced BCE on per-bin ‘has movement begun?’ logits."""
    logit = pred[..., 0]
    bce = F.binary_cross_entropy_with_logits(logit, label, reduction="none")
    w = mask.to(dtype=pred.dtype)
    per = (bce * w).sum(dim=1) / w.sum(dim=1).clamp(min=1.0)
    return per.mean()


def build_onset_model(n_areas, y_mean, y_std):
    """Same causal GRU+attn A1 backbone; readout is a Bernoulli logit, not speed."""
    model = mv.build_model(
        n_areas,
        pool=POOL,
        use_anatomy=False,
        temporal="gru",
        double=False,
        log1p=False,
        y_mean=y_mean,
        y_std=y_std,
    )
    last = model.head.net[-1]
    if getattr(last, "bias", None) is not None:
        torch.nn.init.constant_(last.bias, -2.0)
    return model


def _align_logit_to_label_clock(logit, horizon):
    """Place model outputs on the begun_t time axis."""
    logit = np.asarray(logit, dtype=np.float32).reshape(-1)
    if horizon != "next":
        return logit
    aligned = np.full_like(logit, np.nan)
    if logit.size:
        aligned[1:] = logit[:-1]
    return aligned


def _onset_from_prob(prob, valid, binsize, sticky=2):
    p = np.asarray(prob, dtype=np.float64).reshape(-1)
    m = np.asarray(valid, dtype=bool).reshape(-1)
    run = 0
    n = min(p.size, m.size)
    for i in range(n):
        hit = bool(m[i]) and np.isfinite(p[i]) and p[i] >= 0.5
        run = run + 1 if hit else 0
        if run >= sticky:
            return (i - sticky + 1) * binsize
    return float("nan")


def predict_onset(blob, trials, idx, stats, horizon, thr):
    """Causal logits → begun probabilities on the label clock (not denormalized speed)."""
    stats = {
        "vocab": stats["vocab"],
        "n_areas": int(stats["n_areas"]),
        "xyz_mean": np.asarray(stats["xyz_mean"], dtype=np.float32),
        "xyz_std": np.asarray(stats["xyz_std"], dtype=np.float32),
        "rates": {k: np.asarray(v, dtype=np.float32) for k, v in stats["rates"].items()},
    }
    dev = mv.device_of()
    model = build_onset_model(blob["n_areas"], blob.get("y_mean"), blob.get("y_std")).to(dev)
    model.load_state_dict(blob["state_dict"])
    model.eval()
    ds = mv.TrialDS(trials, idx, blob["y_mean"], blob["y_std"], stats)
    ld = DataLoader(ds, batch_size=mv.BATCH, shuffle=False, collate_fn=mv.collate)
    recs = []
    y_mean = np.asarray(blob["y_mean"], dtype=np.float32).reshape(-1)
    y_std = np.asarray(blob["y_std"], dtype=np.float32).reshape(-1)
    with torch.no_grad():
        for batch in ld:
            logit = (
                model(
                    batch["x"].to(dev),
                    batch["xyz"].to(dev),
                    batch["area"].to(dev),
                    batch["rate"].to(dev),
                    batch["mask_n"].to(dev),
                    batch["mask_t"].to(dev),
                    batch["lengths"].to(dev),
                    unit_drop=0.0,
                )[..., 0]
                .cpu()
                .numpy()
            )
            y_z = batch["y"].numpy()[..., 0]
            mt = batch["mask_t"].numpy()
            lengths = batch["lengths"].numpy()
            for b in range(logit.shape[0]):
                T = int(lengths[b])
                raw = logit[b, :T]
                aligned = _align_logit_to_label_clock(raw, horizon)
                valid = mt[b, :T].astype(bool)
                if horizon == "next" and T:
                    valid[0] = False
                finite = valid & np.isfinite(aligned)
                prob = np.full(T, np.nan, dtype=np.float32)
                prob[finite] = 1.0 / (1.0 + np.exp(-np.clip(aligned[finite], -20, 20)))
                y0 = y_z[b, :T] * y_std[0] + y_mean[0]
                onset_true = _onset_from_prob(
                    (np.abs(y0) >= thr).astype(np.float64),
                    mt[b, :T].astype(bool),
                    mv.BINSIZE,
                )
                label = np.zeros(T, dtype=np.float32)
                if np.isfinite(onset_true):
                    i0 = int(round(onset_true / mv.BINSIZE))
                    label[max(i0, 0) :] = 1.0
                recs.append(
                    {
                        "pred": prob,
                        "y": label,
                        "prob": prob,
                        "logit": aligned.astype(np.float32),
                        "onset_true": onset_true,
                        "onset_pred": _onset_from_prob(prob, valid, mv.BINSIZE),
                        "horizon": horizon,
                    }
                )
    return recs


def run_epoch_cluster(
    model,
    loader,
    opt,
    dev,
    train=True,
    y_mean=None,
    y_std=None,
    thr=WHEEL_ON,
    use_aux=False,
    lam_on=LAM_ONSET,
    lam_pk=LAM_PEAK,
    head="trace",
    horizon="now",
):
    model.train(train)
    total = mse_tot = on_tot = pk_tot = 0.0
    n = 0
    for batch in loader:
        x = batch["x"].to(dev)
        y = batch["y"].to(dev)
        xyz = batch["xyz"].to(dev)
        area = batch["area"].to(dev)
        rate = batch["rate"].to(dev)
        mask_t = batch["mask_t"].to(dev)
        mask_loss = batch["mask_loss"].to(dev)
        mask_n = batch["mask_n"].to(dev)
        lengths = batch["lengths"].to(dev)
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
            unit_drop=mv.UNIT_DROPOUT if train else 0.0,
        )
        if head == "onset":
            label, _ = begun_step_labels(y, mask_loss, y_mean, y_std, thr, mv.BINSIZE)
            label, mask_on = shift_horizon(label, mask_loss, horizon)
            loss = onset_bce(pred, label, mask_on)
            mse = loss
            on = pk = pred.new_zeros(())
        else:
            loss, mse, on, pk = cluster_objective(
                pred, y, mask_loss, y_mean, y_std, thr, lengths, use_aux, lam_on, lam_pk
            )
        if train:
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
        bs = int(x.size(0))
        total += float(loss.item()) * bs
        mse_tot += float(mse.item()) * bs
        on_tot += float(on.item()) * bs
        pk_tot += float(pk.item()) * bs
        n += bs
    d = max(n, 1)
    return total / d, mse_tot / d, on_tot / d, pk_tot / d


def cache_dir_cluster(task, fold_id, loss, sampling, head="trace", horizon=""):
    base = mv.cache_dir(task, POOL, use_anatomy=False, fold_id=fold_id, temporal="gru")
    if head == "onset":
        d = base / CACHE_VERSION / "onset" / str(horizon or "now") / sampling
    else:
        d = base / CACHE_VERSION / loss / sampling
    d.mkdir(parents=True, exist_ok=True)
    return d


def train_one_cluster(
    trials,
    train_idx,
    val_idx,
    y_mean,
    y_std,
    stats,
    task,
    fold_id,
    loss,
    sampling,
    target,
    bin_key=None,
    lam_on=LAM_ONSET,
    lam_pk=LAM_PEAK,
    head="trace",
    horizon="now",
):
    tag = f"attn_a1_{loss}_{sampling}" if head != "onset" else f"attn_a1_onset_{horizon}_{sampling}"
    cdir = cache_dir_cluster(task, fold_id, loss, sampling, head=head, horizon=horizon)
    ckpt = cdir / "model.pt"
    hist_path = cdir / "history.json"
    resume = False
    blob = {}
    if ckpt.exists() and hist_path.exists():
        hist = json.loads(hist_path.read_text())
        if str(hist.get("cache_version", "")) != CACHE_VERSION:
            log(f"  ignore stale cache {task}/{fold_id}/{tag} (version {hist.get('cache_version')})")
        else:
            blob = torch.load(ckpt, map_location="cpu", weights_only=False)
            if hist.get("finished", True):
                log(f"  cache hit {task}/{fold_id}/{tag}")
                return blob, hist
            log(f"  resume {task}/{fold_id}/{tag} from epoch {len(hist.get('train', []))}")
            resume = True

    dev = mv.device_of()
    if head == "onset":
        model = build_onset_model(stats["n_areas"], y_mean, y_std).to(dev)
        log(f"  {tag} params={mv.n_params(model)}  device={dev}  A1 attn GRU onset-RT horizon={horizon}")
    else:
        model = mv.build_model(
            stats["n_areas"],
            pool=POOL,
            use_anatomy=False,
            temporal="gru",
            double=False,
            log1p=False,
            y_mean=y_mean,
            y_std=y_std,
        ).to(dev)
        log(f"  {tag} params={mv.n_params(model)}  device={dev}  A1 attn GRU")
    opt = torch.optim.AdamW(model.parameters(), lr=mv.LR, weight_decay=mv.WEIGHT_DECAY)
    tr_ds = mv.TrialDS(trials, train_idx, y_mean, y_std, stats, bin_key=bin_key)
    va_ds = mv.TrialDS(trials, val_idx, y_mean, y_std, stats, bin_key=bin_key)
    if sampling == "inverse":
        w = inverse_weights(trials, train_idx, target)
        sampler = WeightedRandomSampler(w, num_samples=len(w), replacement=True)
        tr_ld = DataLoader(tr_ds, batch_size=mv.BATCH, sampler=sampler, collate_fn=mv.collate)
    else:
        tr_ld = DataLoader(tr_ds, batch_size=mv.BATCH, shuffle=True, collate_fn=mv.collate)
    va_ld = DataLoader(va_ds, batch_size=mv.BATCH, shuffle=False, collate_fn=mv.collate)
    thr = WHEEL_ON if target == "wheel_speed" else PAW_ON
    use_aux = loss == "onset_peak"
    best_val = math.inf
    best_state = None
    bad = 0
    start_epoch = 1
    hist = {
        "train": [],
        "val": [],
        "train_mse": [],
        "val_mse": [],
        "train_onset": [],
        "val_onset": [],
        "train_peak": [],
        "val_peak": [],
        "best_epoch": 0,
        "n_params": mv.n_params(model),
        "use_anatomy": False,
        "ablation": ABLATION,
        "loss": loss,
        "sampling": sampling,
        "head": head,
        "horizon": horizon,
        "temporal": "gru",
        "cache_version": CACHE_VERSION,
        "soft_peak_tau": SOFT_PEAK_TAU,
        "soft_onset_frac": SOFT_ONSET_FRAC,
        "time_loss_scale": TIME_LOSS_SCALE,
        "inv_max_ratio": INV_MAX_RATIO,
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
    log(f"  train {tag}  n_tr={len(tr_ds)} n_val={len(va_ds)}  aux={int(use_aux)}")

    def snapshot():
        return {
            "state_dict": best_state,
            "live_state": {k: v.detach().cpu().clone() for k, v in model.state_dict().items()},
            "opt_state": {k: v for k, v in opt.state_dict().items()},
            "y_mean": y_mean,
            "y_std": y_std,
            "pool": POOL,
            "n_areas": stats["n_areas"],
            "use_anatomy": False,
            "temporal": "gru",
            "double": False,
            "log1p": False,
            "loss": loss,
            "sampling": sampling,
            "head": head,
            "horizon": horizon,
            "cache_version": CACHE_VERSION,
            "best_val": best_val,
            "stats": {
                "vocab": stats["vocab"],
                "n_areas": stats["n_areas"],
                "xyz_mean": stats["xyz_mean"],
                "xyz_std": stats["xyz_std"],
                "rates": stats["rates"],
            },
        }

    kw = dict(
        y_mean=y_mean,
        y_std=y_std,
        thr=thr,
        use_aux=use_aux,
        lam_on=lam_on,
        lam_pk=lam_pk,
        head=head,
        horizon=horizon,
    )
    for epoch in range(start_epoch, mv.MAX_EPOCHS + 1):
        tr_loss, tr_mse, tr_on, tr_pk = run_epoch_cluster(model, tr_ld, opt, dev, train=True, **kw)
        with torch.no_grad():
            va_loss, va_mse, va_on, va_pk = run_epoch_cluster(model, va_ld, None, dev, train=False, **kw)
        hist["train"].append(tr_loss)
        hist["val"].append(va_loss)
        hist["train_mse"].append(tr_mse)
        hist["val_mse"].append(va_mse)
        hist["train_onset"].append(tr_on)
        hist["val_onset"].append(va_on)
        hist["train_peak"].append(tr_pk)
        hist["val_peak"].append(va_pk)
        if head == "onset":
            log(
                f"    {tag} epoch {epoch:02d}  train={tr_loss:.4f}  val={va_loss:.4f}  "
                f"bce={tr_mse:.4f}/{va_mse:.4f}  horizon={horizon}"
            )
        else:
            log(
                f"    {tag} epoch {epoch:02d}  train={tr_loss:.4f}  val={va_loss:.4f}  "
                f"mse={tr_mse:.4f}/{va_mse:.4f}  on={tr_on:.4f}  pk={tr_pk:.4f}"
            )
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
        mv.save_json(hist_path, hist)
        slurm_utils.write_status(
            state="training",
            epoch=epoch,
            train=tr_loss,
            val=va_loss,
            best_epoch=hist["best_epoch"],
            best_val=best_val,
            n_params=hist.get("n_params"),
            loss=loss,
            sampling=sampling,
        )
        if bad >= mv.PATIENCE:
            log(f"    early stop at epoch {epoch}")
            break
    hist["finished"] = True
    out = snapshot()
    out.pop("live_state", None)
    out.pop("opt_state", None)
    torch.save(out, ckpt)
    mv.save_json(hist_path, hist)
    return out, hist


def score_onset_pack(recs):
    """Reaction-time metrics from per-bin begun probabilities."""
    empty = {
        "r2": float("nan"),
        "mean_trial_r2": float("nan"),
        "median_trial_r2": float("nan"),
        "rmse": float("nan"),
        "mae": float("nan"),
        "pearson": float("nan"),
        "onset_mae": float("nan"),
        "onset_pearson": float("nan"),
        "frac_40ms": float("nan"),
        "frac_100ms": float("nan"),
        "bin_acc": float("nan"),
        "brier": float("nan"),
        "n_bins": 0,
        "n_trials": 0,
        "n_onset": 0,
    }
    if not recs:
        return empty
    tt, tp = [], []
    y_all, p_all = [], []
    for r in recs:
        ot, op = r.get("onset_true"), r.get("onset_pred")
        if ot is not None and op is not None and np.isfinite(ot) and np.isfinite(op):
            tt.append(float(ot))
            tp.append(float(op))
        y = np.asarray(r.get("y"), dtype=float).reshape(-1)
        p = np.asarray(r.get("prob", r.get("pred")), dtype=float).reshape(-1)
        m = np.isfinite(y) & np.isfinite(p)
        if m.any():
            y_all.append(y[m])
            p_all.append(p[m])
    out = dict(empty)
    out["n_trials"] = len(recs)
    out["n_onset"] = len(tt)
    if tt:
        tt, tp = np.asarray(tt), np.asarray(tp)
        err = np.abs(tt - tp)
        out["onset_mae"] = float(err.mean())
        out["mae"] = out["onset_mae"]
        out["frac_40ms"] = float((err <= 0.04).mean())
        out["frac_100ms"] = float((err <= 0.10).mean())
        if tt.size >= 8 and tt.std() > 1e-9 and tp.std() > 1e-9:
            out["onset_pearson"] = float(np.corrcoef(tt, tp)[0, 1])
            out["pearson"] = out["onset_pearson"]
    if y_all:
        y = np.concatenate(y_all)
        p = np.clip(np.concatenate(p_all), 0.0, 1.0)
        out["n_bins"] = int(y.size)
        out["bin_acc"] = float(((p >= 0.5) == (y >= 0.5)).mean())
        out["brier"] = float(np.mean((p - y) ** 2))
        out["rmse"] = float(np.sqrt(out["brier"]))
    return out


def score_pack(recs, min_trial_bins=8, target="wheel_speed"):
    if recs and "onset_pred" in recs[0]:
        return score_onset_pack(recs)
    if not recs:
        return {
            "r2": float("nan"),
            "mean_trial_r2": float("nan"),
            "median_trial_r2": float("nan"),
            "rmse": float("nan"),
            "mae": float("nan"),
            "pearson": float("nan"),
            "n_bins": 0,
            "n_trials": 0,
        }
    packed = mv.score_recs(recs, target=target, min_trial_bins=min_trial_bins)
    return packed[target]


def slice_rows(recs, trials, test_idx, min_trial_bins=8, target="wheel_speed"):
    """Long-format R² by contrast, length, onset, stratum, and atlas cluster."""
    recs = list(recs)
    idx = [int(i) for i in np.asarray(test_idx)]
    groups = defaultdict(list)
    for rec, i in zip(recs, idx):
        t = trials[i]
        groups[("contrast", t["contrast"])].append(rec)
        groups[("length", t["len_bin"])].append(rec)
        groups[("onset", t["onset_bin"])].append(rec)
        groups[("stratum", t["stratum"])].append(rec)
        cid = int(t["atlas_c"])
        if cid >= 0:
            key = f"{cid}:{t['atlas_name']}"
        elif cid == -1:
            key = "U:unclustered"
        else:
            key = "NA:not_in_atlas"
        groups[("cluster", key)].append(rec)
    rows = []
    for (kind, key), sub in sorted(groups.items(), key=lambda kv: (kv[0][0], kv[0][1])):
        sc = score_pack(sub, min_trial_bins=min_trial_bins, target=target)
        rows.append({"slice_kind": kind, "slice_key": key, **sc})
    return rows


def jackknife_mean_se(vals):
    x = np.asarray(vals, dtype=float)
    x = x[np.isfinite(x)]
    n = int(x.size)
    if n == 0:
        return float("nan"), float("nan"), 0
    if n == 1:
        return float(x[0]), float("nan"), 1
    theta = float(x.mean())
    loo = (theta * n - x) / (n - 1)
    se = float(np.sqrt((n - 1) / n * np.sum((loo - loo.mean()) ** 2)))
    return theta, se, n


def summarize(rows, slice_rows_all):
    df = pd.DataFrame(rows)
    keys = ["task", "target", "loss", "sampling"]
    if not df.empty:
        for extra in ("head", "horizon"):
            if extra in df.columns:
                keys.append(extra)
    metrics = ["r2", "mean_trial_r2", "median_trial_r2"]
    if not df.empty:
        for extra in ("onset_mae", "onset_pearson", "frac_40ms", "frac_100ms", "bin_acc", "brier"):
            if extra in df.columns:
                metrics.append(extra)
    out = []
    if not df.empty:
        for name, sub in df.groupby(keys, dropna=False):
            rec = dict(zip(keys, name if isinstance(name, tuple) else (name,)))
            rec["n_folds"] = int(len(sub))
            for m in metrics:
                mean, se, n = jackknife_mean_se(sub[m].to_numpy())
                rec[f"{m}_mean"] = mean
                rec[f"{m}_jackknife_se"] = se
                rec[f"{m}_median"] = float(np.nanmedian(sub[m].to_numpy()))
                rec[f"{m}_n"] = n
            out.append(rec)
    sdf = pd.DataFrame(slice_rows_all)
    slices = []
    if not sdf.empty:
        skeys = keys + ["slice_kind", "slice_key"]
        for name, sub in sdf.groupby(skeys, dropna=False):
            rec = dict(zip(skeys, name if isinstance(name, tuple) else (name,)))
            rec["n_folds"] = int(len(sub))
            rec["n_trials_mean"] = float(sub["n_trials"].mean()) if "n_trials" in sub else float("nan")
            for m in metrics:
                mean, se, n = jackknife_mean_se(sub[m].to_numpy())
                rec[f"{m}_mean"] = mean
                rec[f"{m}_jackknife_se"] = se
                rec[f"{m}_median"] = float(np.nanmedian(sub[m].to_numpy()))
            slices.append(rec)
    return out, slices


def plot_summary(summary, out_dir):
    mv.style_plots()
    import matplotlib.pyplot as plt

    df = pd.DataFrame(summary)
    if df.empty or "r2_mean" not in df.columns:
        return
    tasks = [t for t in TASKS if t in set(df.task)]
    combos = [(lo, sa) for lo in LOSSES for sa in SAMPLINGS]
    labels = [f"{lo}\n{sa}" for lo, sa in combos]
    for metric, ylabel in (
        ("r2_mean", "concatenated R²"),
        ("mean_trial_r2_mean", "mean trial R²"),
        ("median_trial_r2_mean", "median trial R²"),
    ):
        fig, axes = plt.subplots(len(TARGETS), 1, figsize=(8.4, 3.2 * len(TARGETS)), sharex=True)
        if len(TARGETS) == 1:
            axes = [axes]
        x = np.arange(len(tasks))
        width = 0.18
        for ax, beh in zip(axes, TARGETS):
            sub = df[df.target == beh]
            for k, (lo, sa) in enumerate(combos):
                means, ses = [], []
                for task in tasks:
                    hit = sub[(sub.loss == lo) & (sub.sampling == sa) & (sub.task == task)]
                    means.append(float(hit[metric].iloc[0]) if len(hit) else np.nan)
                    ses.append(float(hit[metric.replace("_mean", "_jackknife_se")].iloc[0]) if len(hit) else np.nan)
                ses = [0.0 if not np.isfinite(s) else float(s) for s in ses]
                ax.bar(
                    x + (k - 1.5) * width,
                    means,
                    width,
                    yerr=ses,
                    capsize=2,
                    label=labels[k],
                    color=["#334155", "#1D4ED8", "#C47B3B", "#0F766E"][k],
                )
            ax.axhline(0, color="0.85", lw=0.8)
            ax.set_ylabel(ylabel)
            ax.set_title(mv.TARGET_LABELS.get(beh, beh))
            ax.set_xticks(x)
            ax.set_xticklabels([mv.TASK_LABELS.get(t, t) for t in tasks])
        axes[0].legend(frameon=False, fontsize=8, ncol=4)
        fig.tight_layout()
        fig.savefig(out_dir / f"{metric}_jackknife.png", dpi=140)
        plt.close(fig)


def plot_coarse_slices(slices, out_dir):
    mv.style_plots()
    import matplotlib.pyplot as plt

    df = pd.DataFrame(slices)
    if df.empty or "slice_kind" not in df.columns:
        return
    combos = [(lo, sa) for lo in LOSSES for sa in SAMPLINGS]
    labels = [f"{lo}/{sa}" for lo, sa in combos]
    colors = ["#334155", "#1D4ED8", "#C47B3B", "#0F766E"]
    for kind in ("contrast", "length", "onset"):
        subk = df[df.slice_kind == kind]
        if subk.empty:
            continue
        keys = sorted(subk.slice_key.astype(str).unique())
        tasks = [t for t in TASKS if t in set(subk.task)]
        fig, axes = plt.subplots(
            len(TARGETS),
            max(len(tasks), 1),
            figsize=(4.2 * max(len(tasks), 1), 3.0 * len(TARGETS)),
            sharey=True,
            squeeze=False,
        )
        x = np.arange(len(keys))
        width = 0.18
        for r, beh in enumerate(TARGETS):
            for c, task in enumerate(tasks or ["_"]):
                ax = axes[r][c]
                hit = subk[(subk.target == beh) & (subk.task == task)]
                for k, (lo, sa) in enumerate(combos):
                    means, ses = [], []
                    for key in keys:
                        row = hit[(hit.loss == lo) & (hit.sampling == sa) & (hit.slice_key.astype(str) == key)]
                        means.append(float(row["mean_trial_r2_mean"].iloc[0]) if len(row) else np.nan)
                        se = float(row["mean_trial_r2_jackknife_se"].iloc[0]) if len(row) else np.nan
                        ses.append(0.0 if not np.isfinite(se) else se)
                    ax.bar(
                        x + (k - 1.5) * width,
                        means,
                        width,
                        yerr=ses,
                        capsize=2,
                        label=labels[k] if r == 0 and c == 0 else None,
                        color=colors[k],
                    )
                ax.axhline(0, color="0.85", lw=0.8)
                ax.set_xticks(x)
                ax.set_xticklabels(keys, rotation=20, ha="right")
                if c == 0:
                    ax.set_ylabel(f"{mv.TARGET_LABELS.get(beh, beh)}\nmean trial R²")
                if r == 0:
                    ax.set_title(mv.TASK_LABELS.get(task, task))
        handles, labs = axes[0][0].get_legend_handles_labels()
        if handles:
            fig.legend(handles, labs, frameon=False, ncol=4, loc="upper center")
        fig.tight_layout(rect=(0, 0, 1, 0.92))
        fig.savefig(out_dir / f"slices_{kind}_jackknife.png", dpi=140)
        plt.close(fig)


def _fmt_pm(mean, se):
    try:
        m = float(mean)
    except (TypeError, ValueError):
        return "—"
    if not np.isfinite(m):
        return "—"
    try:
        s = float(se)
    except (TypeError, ValueError):
        s = float("nan")
    if not np.isfinite(s):
        return f"{m:.3f}"
    return f"{m:.3f} ± {s:.3f}"


def write_report(summary, slices, path, n_jobs):
    lines = [
        "# Modelv1Clusters",
        "",
        "GRU + attention, Control A1 (no anatomy). Separate heads for wheel |ω| and paw speed.",
        "LOSO folds omit only sessions from mice that have another session in the corpus.",
        "",
        f"Jobs completed in this aggregate: {n_jobs}.",
        "",
        "## Overall (mean across folds ± jackknife SE)",
        "",
        "| task | target | loss | sampling | concat R² | mean trial R² | median trial R² | n |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for rec in summary:
        lines.append(
            "| {task} | {target} | {loss} | {sampling} | {r2} | {mean_t} | {med_t} | {n_folds} |".format(
                task=rec.get("task"),
                target=rec.get("target"),
                loss=rec.get("loss"),
                sampling=rec.get("sampling"),
                r2=_fmt_pm(rec.get("r2_mean"), rec.get("r2_jackknife_se")),
                mean_t=_fmt_pm(rec.get("mean_trial_r2_mean"), rec.get("mean_trial_r2_jackknife_se")),
                med_t=_fmt_pm(rec.get("median_trial_r2_mean"), rec.get("median_trial_r2_jackknife_se")),
                n_folds=rec.get("n_folds"),
            )
        )
    lines += ["", "## Slices (mean across folds ± jackknife SE)", ""]
    for kind in ("contrast", "length", "onset", "cluster"):
        sub = [s for s in slices if s.get("slice_kind") == kind]
        if not sub:
            continue
        lines += [
            f"### {kind}",
            "",
            "| task | target | loss | sampling | key | concat R² | mean trial R² | median trial R² | n folds |",
            "|---|---|---|---|---|---|---|---|---|",
        ]
        for rec in sub:
            lines.append(
                "| {task} | {target} | {loss} | {sampling} | {slice_key} | {r2} | {mean_t} | {med_t} | {n_folds} |".format(
                    task=rec.get("task"),
                    target=rec.get("target"),
                    loss=rec.get("loss"),
                    sampling=rec.get("sampling"),
                    slice_key=rec.get("slice_key"),
                    r2=_fmt_pm(rec.get("r2_mean"), rec.get("r2_jackknife_se")),
                    mean_t=_fmt_pm(rec.get("mean_trial_r2_mean"), rec.get("mean_trial_r2_jackknife_se")),
                    med_t=_fmt_pm(rec.get("median_trial_r2_mean"), rec.get("median_trial_r2_jackknife_se")),
                    n_folds=rec.get("n_folds"),
                )
            )
        lines.append("")
    Path(path).write_text("\n".join(lines))


def process(
    target,
    task,
    loss,
    sampling,
    fold=None,
    fold_id=None,
    trials=None,
    session_meta=None,
    n_trial_repeats=1,
    job_index=None,
    make_plots=True,
    lam_on=LAM_ONSET,
    lam_pk=LAM_PEAK,
    max_fold=None,
    head="trace",
    horizon="now",
):
    warnings.filterwarnings("ignore")
    mv.set_seed(mv.SEED)
    mv.configure_out(target, "cv")
    mv.OUT.mkdir(parents=True, exist_ok=True)
    mv.CACHE.mkdir(parents=True, exist_ok=True)
    log(
        f"device={mv.device_of()}  target={target}  task={task}  "
        f"loss={loss}  sampling={sampling}  head={head}  horizon={horizon}  A1 attn GRU"
    )
    if trials is None or session_meta is None:
        trials, session_meta = mv.load_corpus()
    mv.set_target(trials, target, log1p=False)
    annotate_trials(trials, target)
    specs = cv_folds_clusters(trials, n_trial_repeats=n_trial_repeats)
    specs = [s for s in specs if s["task"] == task]
    if fold is not None:
        specs = [s for s in specs if int(s["fold"]) == int(fold)]
    if fold_id is not None:
        specs = [s for s in specs if str(s["fold_id"]) == str(fold_id)]
    if max_fold is not None:
        specs = [s for s in specs if int(s["fold"]) <= int(max_fold)]
    if not specs:
        log(f"no folds for task={task} fold={fold}")
        return [], []
    rows = []
    slice_all = []
    n_job = 0
    for spec in specs:
        fold_i = int(spec["fold"])
        fid = spec["fold_id"]
        log(f"==== {task}  fold={fold_i} {fid}  {loss}/{sampling}  head={head} horizon={horizon} ====")
        mv.clear_bin_masks(trials)
        bin_mode = spec.get("bin_mode")
        if bin_mode:
            mv.apply_bin_masks(trials, spec)
            train_full = mv.usable_idx(
                trials, spec["train"], min_bins=max(4, mv.MIN_BINS // 2), mask_key="train_bins"
            )
            test_ok = mv.usable_idx(trials, spec["test"], min_bins=1, mask_key="test_bins")
            both = np.intersect1d(train_full, test_ok)
            train_full, test_idx = both, both
        else:
            train_full = mv.usable_idx(trials, spec["train"])
            test_idx = mv.usable_idx(trials, spec["test"])
        if len(train_full) < 8 or len(test_idx) < 1:
            log(f"  skip {task}/{fid}: too few usable trials")
            continue
        tr_idx, va_idx = mv.train_val_split(train_full, seed=mv.SEED + 17 * fold_i)
        if bin_mode:
            tr_idx = mv.usable_idx(
                trials, tr_idx, min_bins=max(4, mv.MIN_BINS // 2), mask_key="train_bins"
            )
            va_idx = mv.usable_idx(trials, va_idx, min_bins=1, mask_key="train_bins")
        else:
            tr_idx, va_idx = mv.usable_idx(trials, tr_idx), mv.usable_idx(trials, va_idx)
        if len(tr_idx) < 4:
            log(f"  skip {task}/{fid}: empty train after val split")
            continue
        y_mean, y_std = mv.y_scaler_from(trials, tr_idx)
        stats = mv.fit_unit_stats(trials, tr_idx, np.concatenate([tr_idx, va_idx, test_idx]))
        if head == "onset":
            tag = f"{target}_{task}_{fid}_attn_a1_onset_{horizon}_{sampling}_{CACHE_VERSION}"
        else:
            tag = f"{target}_{task}_{fid}_attn_a1_{loss}_{sampling}_{CACHE_VERSION}"
        idx = job_index if job_index is not None else n_job
        jdir = slurm_utils.begin_job(
            mv.OUT,
            idx,
            tag,
            task=task,
            pool=POOL,
            ablation=ABLATION,
            target=target,
            protocol="cv",
            fold=fold_i,
            fold_id=fid,
            loss=loss,
            sampling=sampling,
            head=head,
            horizon=horizon,
        )
        blob, hist = train_one_cluster(
            trials,
            tr_idx,
            va_idx,
            y_mean,
            y_std,
            stats,
            task,
            fid,
            loss,
            sampling,
            target,
            bin_key="train_bins" if bin_mode else None,
            lam_on=lam_on,
            lam_pk=lam_pk,
            head=head,
            horizon=horizon,
        )
        if head == "onset":
            recs = predict_onset(
                blob,
                trials,
                test_idx,
                blob.get("stats", stats),
                horizon=horizon,
                thr=WHEEL_ON if target == "wheel_speed" else PAW_ON,
            )
            min_bins = 8
        else:
            recs = mv.predict(blob, trials, test_idx, blob.get("stats", stats))
            if bin_mode:
                recs = mv.restrict_recs(recs, trials, test_idx, "test_bins")
            min_bins = 2 if bin_mode else 8
        sc = score_pack(recs, min_trial_bins=min_bins, target=target)
        row = {
            "protocol": "cv",
            "task": task,
            "pool": POOL,
            "target": target,
            "ablation": ABLATION,
            "loss": loss,
            "sampling": sampling,
            "head": head,
            "horizon": horizon,
            "cache_version": CACHE_VERSION,
            "fold": fold_i,
            "fold_id": fid,
            "bin_mode": bin_mode or "",
            **sc,
        }
        rows.append(row)
        slices = slice_rows(recs, trials, test_idx, min_trial_bins=min_bins, target=target)
        for sl in slices:
            slice_all.append({**row, **sl})
        pd.DataFrame([row]).to_csv(jdir / "scores.csv", index=False)
        pd.DataFrame(slices).to_csv(jdir / "scores_slices.csv", index=False)
        slurm_utils.write_status(state="done", n_rows=1, best_epoch=hist.get("best_epoch"))
        if head == "onset":
            log(
                f"   {target:12s} horizon={horizon:4s}  onset_mae={sc.get('onset_mae', float('nan'))*1000:.0f}ms  "
                f"acc={sc.get('bin_acc', float('nan')):.3f}  n={sc['n_trials']}"
            )
        else:
            log(
                f"   {target:12s} R²={sc['r2']:+.3f}  mean_trial={sc['mean_trial_r2']:+.3f}  "
                f"median_trial={sc['median_trial_r2']:+.3f}  n={sc['n_trials']}"
            )
        n_job += 1
        if make_plots:
            pd.DataFrame(rows).to_csv(mv.OUT / "scores.csv", index=False)
            pd.DataFrame(slice_all).to_csv(mv.OUT / "scores_slices.csv", index=False)
    if make_plots and rows:
        summary, slice_sum = summarize(rows, slice_all)
        pd.DataFrame(summary).to_csv(mv.OUT / "scores_summary.csv", index=False)
        pd.DataFrame(slice_sum).to_csv(mv.OUT / "scores_summary_slices.csv", index=False)
        mv.save_json(mv.OUT / "scores_summary.json", summary)
        plot_summary(summary, mv.OUT)
        plot_coarse_slices(slice_sum, mv.OUT)
        write_report(summary, slice_sum, mv.OUT / "REPORT.md", n_jobs=len(rows))
        log("wrote " + str(mv.OUT))
    return rows, slice_all


def aggregate_all():
    root = mv.MODEL_ROOT
    rows, slices = [], []
    for beh in TARGETS:
        d = root / beh / "cv"
        jobs = d / "jobs"
        if not jobs.exists():
            continue
        for p in sorted(jobs.glob("*/scores.csv")):
            rows.append(pd.read_csv(p))
        for p in sorted(jobs.glob("*/scores_slices.csv")):
            slices.append(pd.read_csv(p))
        if rows:
            df = pd.concat(rows, ignore_index=True)
            df = df[df["target"] == beh] if "target" in df.columns else df
            d.mkdir(parents=True, exist_ok=True)
            df.to_csv(d / "scores.csv", index=False)
        if slices:
            sdf = pd.concat(slices, ignore_index=True)
            sdf = sdf[sdf["target"] == beh] if "target" in sdf.columns else sdf
            sdf.to_csv(d / "scores_slices.csv", index=False)
        rows, slices = [], []
    all_rows, all_slices = [], []
    for beh in TARGETS:
        d = root / beh / "cv"
        sp, slp = d / "scores.csv", d / "scores_slices.csv"
        if sp.exists():
            all_rows.append(pd.read_csv(sp))
        if slp.exists():
            all_slices.append(pd.read_csv(slp))
    if not all_rows:
        raise SystemExit(f"no per-job scores under {root}")
    df = pd.concat(all_rows, ignore_index=True)
    sdf = pd.concat(all_slices, ignore_index=True) if all_slices else pd.DataFrame()
    root.mkdir(parents=True, exist_ok=True)
    df.to_csv(root / "scores.csv", index=False)
    if not sdf.empty:
        sdf.to_csv(root / "scores_slices.csv", index=False)
    summary, slice_sum = summarize(df.to_dict(orient="records"), sdf.to_dict(orient="records") if not sdf.empty else [])
    pd.DataFrame(summary).to_csv(root / "scores_summary.csv", index=False)
    pd.DataFrame(slice_sum).to_csv(root / "scores_summary_slices.csv", index=False)
    mv.save_json(root / "scores_summary.json", summary)
    plot_summary(summary, root)
    plot_coarse_slices(slice_sum, root)
    write_report(summary, slice_sum, root / "REPORT.md", n_jobs=len(df))
    log(f"aggregated {len(df)} fold-rows → {root}")
    print(pd.DataFrame(summary).to_string(index=False))


def run_self_test():
    """Gradient + inverse-cap checks. No corpus, no cache."""
    torch.manual_seed(0)
    bsz, n_t = 8, 40
    t_idx = torch.arange(n_t, dtype=torch.float32)
    y = torch.zeros(bsz, n_t, 1)
    for i in range(bsz):
        peak = 8.0 + 3.0 * i
        y[i, :, 0] = torch.exp(-0.5 * ((t_idx - peak) / 2.0) ** 2) * (0.9 + 0.05 * i)
    mask = torch.ones(bsz, n_t, dtype=torch.bool)
    lengths = torch.full((bsz,), n_t)
    y_mean = np.array([0.0], dtype=np.float32)
    y_std = np.array([1.0], dtype=np.float32)
    pred = (0.25 * y + 0.04).detach().clone().requires_grad_(True)
    loss, mse, on, pk = cluster_objective(
        pred, y, mask, y_mean, y_std, WHEEL_ON, lengths, True, 1.0, 1.0
    )
    (on + pk).backward()
    g = float(pred.grad.abs().sum())
    if not np.isfinite(g) or g <= 0:
        raise SystemExit(f"self-test failed: timing terms have no gradient (g={g})")
    pred_a = torch.randn(bsz, n_t, 1, requires_grad=True)
    l_mse, _, _, _ = cluster_objective(
        pred_a, y, mask, y_mean, y_std, WHEEL_ON, lengths, False, 1.0, 1.0
    )
    pred_b = pred_a.detach().clone().requires_grad_(True)
    l_aux, _, on_b, pk_b = cluster_objective(
        pred_b, y, mask, y_mean, y_std, WHEEL_ON, lengths, True, 1.0, 1.0
    )
    if abs((l_aux - l_mse).detach().item()) < 1e-8 and (on_b + pk_b).detach().item() <= 0:
        raise SystemExit("self-test failed: onset_peak equals mse with zero aux")
    fake = [{"stratum": "common"}] * 90 + [{"stratum": "rare"}]
    w = inverse_weights(fake, np.arange(len(fake)))
    ratio = float(w.max() / w.min())
    if ratio > INV_MAX_RATIO + 1e-6:
        raise SystemExit(f"self-test failed: inverse ratio {ratio:.2f} > cap {INV_MAX_RATIO}")
    logit = torch.randn(bsz, n_t, 1, requires_grad=True)
    lab = torch.zeros(bsz, n_t)
    lab[:, 12:] = 1.0
    bce = onset_bce(logit, lab, mask)
    bce.backward()
    if float(logit.grad.abs().sum()) <= 0:
        raise SystemExit("self-test failed: onset BCE has no gradient")
    log(
        f"self-test ok  timing_grad={g:.4g}  "
        f"Δloss={abs((l_aux - l_mse).detach().item()):.4g}  inv_ratio={ratio:.2f}  "
        f"onset_bce_grad={float(logit.grad.abs().sum()):.4g}"
    )
    return True


def _hist_path(fold_id, loss, sampling, target="wheel_speed"):
    safe = "".join(ch if ch.isalnum() or ch in "-_." else "_" for ch in str(fold_id))
    return (
        mv.MODEL_ROOT
        / target
        / "cv"
        / "cache"
        / "session_loso"
        / safe
        / "attn_a1"
        / CACHE_VERSION
        / loss
        / sampling
        / "history.json"
    )


def write_smoke_report(n_folds=SMOKE_FOLDS):
    """Compare mse vs onset_peak on the smoke jobs. Does not walk cache/."""
    root = mv.MODEL_ROOT / "wheel_speed" / "cv"
    jobs = root / "jobs"
    if not jobs.exists():
        raise SystemExit(f"no smoke jobs under {jobs}")
    recs = []
    for sc_path in sorted(jobs.glob("*/scores.csv")):
        df = pd.read_csv(sc_path)
        if df.empty:
            continue
        row = df.iloc[0].to_dict()
        sl_path = sc_path.parent / "scores_slices.csv"
        delayed = long = None
        if sl_path.exists():
            sdf = pd.read_csv(sl_path)
            for _, sl in sdf.iterrows():
                if sl.get("slice_kind") == "onset" and str(sl.get("slice_key")) == "delayed":
                    delayed = float(sl["r2"])
                if sl.get("slice_kind") == "length" and str(sl.get("slice_key")) == "long":
                    long = float(sl["r2"])
        hp = _hist_path(row["fold_id"], row["loss"], row["sampling"])
        train_mse0 = None
        n_ep = 0
        if hp.exists():
            hist = json.loads(hp.read_text())
            tm = hist.get("train_mse") or []
            n_ep = len(tm)
            if tm:
                train_mse0 = float(tm[0])
        recs.append(
            {
                "fold": int(row["fold"]),
                "fold_id": str(row["fold_id"]),
                "loss": str(row["loss"]),
                "sampling": str(row["sampling"]),
                "r2": float(row["r2"]),
                "median_trial_r2": float(row.get("median_trial_r2", float("nan"))),
                "delayed_r2": delayed,
                "long_r2": long,
                "train_mse0": train_mse0,
                "n_epochs": n_ep,
            }
        )
    if not recs:
        raise SystemExit(f"empty smoke scores under {jobs}")

    def _mean(vals):
        x = [v for v in vals if v is not None and np.isfinite(v)]
        return float(np.mean(x)) if x else float("nan")

    combos = {}
    for rec in recs:
        key = (rec["loss"], rec["sampling"])
        combos.setdefault(key, []).append(rec)

    ident = []
    deltas = []
    for sampling in SAMPLINGS:
        mse_map = {r["fold_id"]: r for r in combos.get(("mse", sampling), [])}
        op_map = {r["fold_id"]: r for r in combos.get(("onset_peak", sampling), [])}
        for fid, a in mse_map.items():
            b = op_map.get(fid)
            if not b or a["train_mse0"] is None or b["train_mse0"] is None:
                continue
            d = abs(a["train_mse0"] - b["train_mse0"])
            deltas.append(d)
            ident.append(d < 1e-8)

    n_ident = int(sum(ident)) if ident else 0
    n_pair = len(ident)
    mean_d = _mean(deltas)
    diverged = n_pair > 0 and n_ident == 0
    table = []
    for loss, sampling in ((ls, sm) for ls in LOSSES for sm in SAMPLINGS):
        sub = combos.get((loss, sampling), [])
        table.append(
            {
                "loss": loss,
                "sampling": sampling,
                "n": len(sub),
                "r2": _mean([r["r2"] for r in sub]),
                "median_trial_r2": _mean([r["median_trial_r2"] for r in sub]),
                "delayed_r2": _mean([r["delayed_r2"] for r in sub]),
                "long_r2": _mean([r["long_r2"] for r in sub]),
            }
        )
    mse_u = next((t for t in table if t["loss"] == "mse" and t["sampling"] == "uniform"), {})
    op_u = next((t for t in table if t["loss"] == "onset_peak" and t["sampling"] == "uniform"), {})
    delayed_moved = (
        np.isfinite(mse_u.get("delayed_r2", float("nan")))
        and np.isfinite(op_u.get("delayed_r2", float("nan")))
        and abs(op_u["delayed_r2"] - mse_u["delayed_r2"]) > 0.02
    )
    report = {
        "cache_version": CACHE_VERSION,
        "n_jobs": len(recs),
        "n_folds": n_folds,
        "train_mse_pairs": n_pair,
        "train_mse_identical": n_ident,
        "train_mse_mean_abs_delta": mean_d,
        "train_mse_diverged": diverged,
        "delayed_r2_moved": bool(delayed_moved),
        "combos": table,
        "folds": recs,
        "pass": bool(diverged),
    }
    out = mv.MODEL_ROOT / "smoke_report.json"
    mv.MODEL_ROOT.mkdir(parents=True, exist_ok=True)
    mv.save_json(out, report)
    log(f"smoke report → {out}")
    log(
        f"train_mse epoch-1: {n_pair} pairs, {n_ident} identical, "
        f"mean |Δ|={mean_d:.4g}  diverged={diverged}"
    )
    print("loss         sampling   n   concat R²   med trial   delayed R²   long R²")
    for t in table:
        print(
            f"{t['loss']:12s} {t['sampling']:8s} {t['n']:2d}  "
            f"{t['r2']:+8.3f}   {t['median_trial_r2']:+8.3f}   "
            f"{t['delayed_r2']:+8.3f}    {t['long_r2']:+8.3f}"
        )
    if diverged:
        log("SMOKE PASS: onset_peak train_mse diverged from mse at epoch 1")
    else:
        log("SMOKE FAIL: onset_peak train_mse still matches mse (timing still detached?)")
    if delayed_moved:
        log(
            f"delayed R² moved {mse_u['delayed_r2']:+.3f} → {op_u['delayed_r2']:+.3f} "
            "(uniform mse vs onset_peak)"
        )
    else:
        log("delayed R² did not move >0.02 on uniform mse vs onset_peak")
    return report


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--targets", nargs="*", default=list(TARGETS), choices=list(TARGETS))
    p.add_argument("--tasks", nargs="*", default=list(TASKS), choices=list(TASKS))
    p.add_argument("--losses", nargs="*", default=list(LOSSES), choices=list(LOSSES))
    p.add_argument("--samplings", nargs="*", default=list(SAMPLINGS), choices=list(SAMPLINGS))
    p.add_argument("--trial-repeats", type=int, default=1)
    p.add_argument("--lam-onset", type=float, default=LAM_ONSET)
    p.add_argument("--lam-peak", type=float, default=LAM_PEAK)
    p.add_argument("--self-test", action="store_true", help="Gradient and inverse-cap checks, then exit")
    p.add_argument("--smoke", action="store_true", help="8-session wheel LOSO × loss × sampling into Modelv1ClustersV2")
    p.add_argument("--smoke-folds", type=int, default=SMOKE_FOLDS)
    p.add_argument(
        "--pilot",
        action="store_true",
        help=f"Cheap CV grid: {PILOT_LOSO} random LOSO sessions, {PILOT_LOMO} random LOMO mice, {PILOT_INTERP} interp (seeded)",
    )
    p.add_argument("--loso-folds", type=int, default=None, help="Keep this many random session_loso folds")
    p.add_argument("--lomo-folds", type=int, default=None, help="Keep this many random mouse_lomo folds")
    p.add_argument("--interp-folds", type=int, default=None, help="Keep this many trial_repeat folds")
    p.add_argument("--fold-seed", type=int, default=0, help="Added to SEED when sampling pilot folds")
    p.add_argument(
        "--onset-rt",
        action="store_true",
        help="Decode movement onset (per-bin ‘has begun?’) instead of the speed trace; adds horizon now vs next",
    )
    p.add_argument(
        "--horizons",
        nargs="*",
        default=list(HORIZONS),
        choices=list(HORIZONS),
        help="Onset-RT only: now = label_t from bins ≤ t; next = label_{t+1} from bins ≤ t",
    )
    slurm_utils.add_common_args(p)
    args = p.parse_args()
    n_repeats = max(1, int(args.trial_repeats))
    smoke_folds = max(1, int(args.smoke_folds))
    grid_kw = dict(
        n_trial_repeats=n_repeats,
        targets=args.targets,
        tasks=args.tasks,
        losses=args.losses,
        samplings=args.samplings,
        smoke=args.smoke,
        smoke_folds=smoke_folds,
        pilot=args.pilot,
        loso_folds=args.loso_folds,
        lomo_folds=args.lomo_folds,
        interp_folds=args.interp_folds,
        fold_seed=args.fold_seed,
        onset_rt=args.onset_rt,
        horizons=args.horizons,
    )
    if args.list_jobs:
        slurm_utils.print_jobs(job_grid(**grid_kw))
        raise SystemExit(0)
    if args.self_test:
        run_self_test()
        raise SystemExit(0)
    slurm_utils.set_device(args.device, probe="gru")
    if args.aggregate:
        aggregate_all()
        raise SystemExit(0)
    if args.smoke:
        args.targets = ["wheel_speed"]
        args.tasks = ["session_loso"]
        run_self_test()
    job = slurm_utils.resolve_job_index(args.job)
    use_grid = bool(
        args.smoke
        or args.pilot
        or args.onset_rt
        or args.loso_folds is not None
        or args.lomo_folds is not None
        or args.interp_folds is not None
    )
    if job is None:
        trials, session_meta = mv.load_corpus()
        if use_grid:
            configs = job_grid(trials=trials, **grid_kw)
            for i, cfg in enumerate(configs):
                process(
                    target=cfg["target"],
                    task=cfg["task"],
                    loss=cfg["loss"],
                    sampling=cfg["sampling"],
                    fold=cfg["fold"],
                    fold_id=cfg["fold_id"],
                    trials=trials,
                    session_meta=session_meta,
                    n_trial_repeats=n_repeats,
                    lam_on=args.lam_onset,
                    lam_pk=args.lam_peak,
                    make_plots=False,
                    job_index=i,
                    head=cfg.get("head", "trace"),
                    horizon=cfg.get("horizon", "now"),
                )
            if args.smoke:
                write_smoke_report(n_folds=smoke_folds)
            else:
                aggregate_all()
        else:
            for beh in args.targets:
                for task in args.tasks:
                    for loss in args.losses:
                        for sampling in args.samplings:
                            process(
                                target=beh,
                                task=task,
                                loss=loss,
                                sampling=sampling,
                                trials=trials,
                                session_meta=session_meta,
                                n_trial_repeats=n_repeats,
                                lam_on=args.lam_onset,
                                lam_pk=args.lam_peak,
                            )
            aggregate_all()
    else:
        trials, session_meta = mv.load_corpus()
        cfg = slurm_utils.pick_config(job_grid(trials=trials, **grid_kw), job)
        process(
            target=cfg["target"],
            task=cfg["task"],
            loss=cfg["loss"],
            sampling=cfg["sampling"],
            fold=cfg["fold"],
            fold_id=cfg["fold_id"],
            trials=trials,
            session_meta=session_meta,
            make_plots=False,
            job_index=job,
            n_trial_repeats=n_repeats,
            lam_on=args.lam_onset,
            lam_pk=args.lam_peak,
            head=cfg.get("head", "trace"),
            horizon=cfg.get("horizon", "now"),
        )
