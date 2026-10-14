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
             onset_peak   that MSE plus onset-time L1 and peak-time L1 (seconds)
  sampling   uniform      shuffle, one pass per epoch
             inverse      WeightedRandomSampler, p ∝ 1/n_stratum, strata
                          computed on the current fold's train indices:
                          contrast × length × early/delayed onset

Reports per fold, then mean / median across folds with delete-1 jackknife SE:
  concatenated R², mean trial R², median trial R²
  the same three sliced by coarse-axis level and by Trajectory Atlas Removed
  cluster (wheel clusters for wheel_speed, paw clusters for paw_speed)

Local:
  python modelv1clusters.py --list-jobs
  python modelv1clusters.py --job 0 --device cpu
  python modelv1clusters.py --aggregate

Slurm:
  bash slurm/submit.sh modelv1clusters.py
  python modelv1clusters.py --aggregate
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
from torch.utils.data import DataLoader, WeightedRandomSampler

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import Modelv1 as mv
import slurm_utils

mv.MODEL_ROOT = ROOT / "Modelv1Clusters"
CLUSTER_JSON = mv.MODEL_ROOT / "atlas_trial_clusters.json"

TARGETS = ("wheel_speed", "paw_speed")
TASKS = ("trial_repeat", "session_loso", "mouse_lomo")
LOSSES = ("mse", "onset_peak")
SAMPLINGS = ("uniform", "inverse")
POOL = "attn"
ABLATION = "a1"

WHEEL_ON = 0.15
PAW_ON = 150.0
ONSET_DELAYED_S = 0.40
LEN_SHORT = 20
LEN_MID = 32
LAM_ONSET = 1.0
LAM_PEAK = 1.0
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


def inverse_weights(trials, idx, target=None):
    """p ∝ 1/n_stratum on this train index list. Epoch size stays len(idx)."""
    idx = np.asarray(idx, dtype=int)
    counts = defaultdict(int)
    keys = []
    for i in idx:
        s = trials[int(i)]["stratum"]
        keys.append(s)
        counts[s] += 1
    w = np.array([1.0 / counts[s] for s in keys], dtype=np.float64)
    w *= len(w) / w.sum()
    log(
        f"    inverse sampling: {len(counts)} strata in train, "
        f"weight min/max={w.min():.3f}/{w.max():.3f}  "
        f"max/min={w.max() / max(w.min(), 1e-12):.1f}"
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


def job_grid(trials=None, n_trial_repeats=1):
    if trials is None:
        trials, _ = mv.load_corpus()
    folds = cv_folds_clusters(trials, n_trial_repeats=n_trial_repeats)
    configs = []
    for beh in TARGETS:
        for spec in folds:
            for loss in LOSSES:
                for sampling in SAMPLINGS:
                    configs.append(
                        {
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
                    )
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


def onset_l1(pred, y, mask, thr, binsize, lengths):
    on_y, has_y = sticky_onset_times(y, mask, thr, binsize)
    on_p, has_p = sticky_onset_times(pred, mask, thr, binsize)
    last = lengths.to(dtype=pred.dtype) * binsize
    pen = torch.zeros(pred.shape[0], device=pred.device, dtype=pred.dtype)
    both = has_y & has_p
    pen = torch.where(both, (on_y - on_p).abs(), pen)
    pen = torch.where(has_y & ~has_p, (on_y - last).abs(), pen)
    pen = torch.where(has_p & ~has_y, (on_p - last).abs(), pen)
    use = has_y | has_p
    if not bool(use.any()):
        return pred.new_zeros(())
    return pen[use].mean()


def peak_l1(pred, y, mask, binsize):
    y_m = y.masked_fill(~mask, float("-inf"))
    p_m = pred.masked_fill(~mask, float("-inf"))
    ty = y_m.argmax(dim=1).to(dtype=pred.dtype) * binsize
    tp = p_m.argmax(dim=1).to(dtype=pred.dtype) * binsize
    ok = mask.any(dim=1)
    if not bool(ok.any()):
        return pred.new_zeros(())
    return (ty[ok] - tp[ok]).abs().mean()


def cluster_objective(pred, y, mask_loss, y_mean, y_std, thr, lengths, use_aux, lam_on, lam_pk):
    mse = mv.trial_balanced_mse(pred, y, mask_loss)
    if not use_aux:
        z = pred.new_zeros(())
        return mse, mse, z, z
    mean = pred.new_tensor(np.asarray(y_mean, dtype=np.float32)).view(1, 1, -1)
    std = pred.new_tensor(np.asarray(y_std, dtype=np.float32)).view(1, 1, -1)
    y0 = (y * std + mean)[..., 0]
    p0 = (pred * std + mean)[..., 0]
    on = onset_l1(p0, y0, mask_loss, thr, mv.BINSIZE, lengths)
    pk = peak_l1(p0, y0, mask_loss, mv.BINSIZE)
    return mse + lam_on * on + lam_pk * pk, mse, on, pk


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


def cache_dir_cluster(task, fold_id, loss, sampling):
    base = mv.cache_dir(task, POOL, use_anatomy=False, fold_id=fold_id, temporal="gru")
    d = base / loss / sampling
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
):
    tag = f"attn_a1_{loss}_{sampling}"
    cdir = cache_dir_cluster(task, fold_id, loss, sampling)
    ckpt = cdir / "model.pt"
    hist_path = cdir / "history.json"
    if ckpt.exists() and hist_path.exists():
        hist = json.loads(hist_path.read_text())
        blob = torch.load(ckpt, map_location="cpu", weights_only=False)
        if hist.get("finished", True):
            log(f"  cache hit {task}/{fold_id}/{tag}")
            return blob, hist
        log(f"  resume {task}/{fold_id}/{tag} from epoch {len(hist.get('train', []))}")
        resume = True
    else:
        resume = False
        blob = {}

    dev = mv.device_of()
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
        "temporal": "gru",
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
            "best_val": best_val,
            "stats": {
                "vocab": stats["vocab"],
                "n_areas": stats["n_areas"],
                "xyz_mean": stats["xyz_mean"],
                "xyz_std": stats["xyz_std"],
                "rates": stats["rates"],
            },
        }

    kw = dict(y_mean=y_mean, y_std=y_std, thr=thr, use_aux=use_aux, lam_on=lam_on, lam_pk=lam_pk)
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


def score_pack(recs, min_trial_bins=8, target="wheel_speed"):
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
    metrics = ["r2", "mean_trial_r2", "median_trial_r2"]
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
    if df.empty:
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
):
    warnings.filterwarnings("ignore")
    mv.set_seed(mv.SEED)
    mv.configure_out(target, "cv")
    mv.OUT.mkdir(parents=True, exist_ok=True)
    mv.CACHE.mkdir(parents=True, exist_ok=True)
    log(
        f"device={mv.device_of()}  target={target}  task={task}  "
        f"loss={loss}  sampling={sampling}  A1 attn GRU"
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
    if not specs:
        log(f"no folds for task={task} fold={fold}")
        return [], []
    rows = []
    slice_all = []
    n_job = 0
    for spec in specs:
        fold_i = int(spec["fold"])
        fid = spec["fold_id"]
        log(f"==== {task}  fold={fold_i} {fid}  {loss}/{sampling} ====")
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
        tag = f"{target}_{task}_{fid}_attn_a1_{loss}_{sampling}"
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
        )
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


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--targets", nargs="*", default=list(TARGETS), choices=list(TARGETS))
    p.add_argument("--tasks", nargs="*", default=list(TASKS), choices=list(TASKS))
    p.add_argument("--losses", nargs="*", default=list(LOSSES), choices=list(LOSSES))
    p.add_argument("--samplings", nargs="*", default=list(SAMPLINGS), choices=list(SAMPLINGS))
    p.add_argument("--trial-repeats", type=int, default=1)
    p.add_argument("--lam-onset", type=float, default=LAM_ONSET)
    p.add_argument("--lam-peak", type=float, default=LAM_PEAK)
    slurm_utils.add_common_args(p)
    args = p.parse_args()
    n_repeats = max(1, int(args.trial_repeats))
    if args.list_jobs:
        slurm_utils.print_jobs(job_grid(n_trial_repeats=n_repeats))
        raise SystemExit(0)
    slurm_utils.set_device(args.device, probe="gru")
    if args.aggregate:
        aggregate_all()
        raise SystemExit(0)
    job = slurm_utils.resolve_job_index(args.job)
    if job is None:
        trials, session_meta = mv.load_corpus()
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
        cfg = slurm_utils.pick_config(job_grid(trials=trials, n_trial_repeats=n_repeats), job)
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
        )
