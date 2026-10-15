#!/usr/bin/env python3
"""Score Modelv1 CV2 attn-A1 GRU checkpoints: holdout metrics + atlas clusters."""
from __future__ import annotations

import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import r2_score

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

import Modelv1 as mv
import slurm_utils

OUT = Path("/tmp/modelv1_cv2_holdout.json")
CLUSTER_JSON = ROOT / "Modelv1Clusters" / "atlas_trial_clusters.json"
TARGETS = ("wheel_speed", "paw_speed")
TASKS = ("trial_repeat", "session_loso", "mouse_lomo")
POOL = "attn"
USE_ANATOMY = False
WHEEL_ON = 0.15
PAW_ON = 150.0
STICKY = 2
MIN_BINS_FULL = 8
MIN_BINS_INTERP = 2


def jack(vals):
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


def series(arr):
    a = np.asarray(arr, dtype=float).reshape(-1)
    return a


def onset_bin(y, thr, sticky=STICKY):
    y = series(y)
    above = np.isfinite(y) & (np.abs(y) >= thr)
    run = 0
    for i, a in enumerate(above):
        run = run + 1 if a else 0
        if run >= sticky:
            return i - sticky + 1
    return None


def peak_bin(y):
    y = series(y)
    m = np.isfinite(y)
    if not m.any():
        return None, None
    i = int(np.nanargmax(np.where(m, np.abs(y), -np.inf)))
    return i, float(y[i])


def trial_r2(y, p, min_bins):
    yt, yp = series(y), series(p)
    m = np.isfinite(yt) & np.isfinite(yp)
    if int(m.sum()) < min_bins:
        return float("nan")
    yt, yp = yt[m], yp[m]
    tot = float(np.sum((yt - yt.mean()) ** 2))
    if tot < 1e-12:
        return float("nan")
    return float(1.0 - np.sum((yt - yp) ** 2) / tot)


def concat_r2(recs, min_bins=2):
    """All finite bins, matching Modelv1.score_recs concatenated R²."""
    ys, ps = [], []
    for r in recs:
        yt, yp = series(r["y"]), series(r["pred"])
        m = np.isfinite(yt) & np.isfinite(yp)
        if not m.any():
            continue
        ys.append(yt[m])
        ps.append(yp[m])
    if not ys:
        return float("nan")
    yt = np.concatenate(ys)
    yp = np.concatenate(ps)
    if yt.size < 10:
        return float("nan")
    return float(r2_score(yt, yp))


def pearson(a, b):
    a, b = np.asarray(a, float), np.asarray(b, float)
    m = np.isfinite(a) & np.isfinite(b)
    if int(m.sum()) < 8:
        return float("nan")
    if np.std(a[m]) < 1e-12 or np.std(b[m]) < 1e-12:
        return float("nan")
    return float(np.corrcoef(a[m], b[m])[0, 1])


def score_recs_extra(recs_r2, recs_time, idx, trials, target, min_bins):
    thr = WHEEL_ON if target == "wheel_speed" else PAW_ON
    r2s, on_err, pk_err, pk_ratio = [], [], [], []
    t_true, t_pred = [], []
    for rec, rec_t, i in zip(recs_r2, recs_time, idx):
        r2 = trial_r2(rec["y"], rec["pred"], min_bins)
        r2s.append(r2)
        yt, yp = series(rec_t["y"]), series(rec_t["pred"])
        on_y = onset_bin(yt, thr)
        on_p = onset_bin(yp, thr)
        if on_y is not None and on_p is not None:
            on_err.append(abs(on_y - on_p) * mv.BINSIZE)
        py, ay = peak_bin(yt)
        pp, ap = peak_bin(yp)
        if py is not None and pp is not None:
            pk_err.append(abs(py - pp) * mv.BINSIZE)
            t_true.append(py * mv.BINSIZE)
            t_pred.append(pp * mv.BINSIZE)
            if ay is not None and abs(ay) > 1e-6:
                pk_ratio.append(abs(ap) / abs(ay))
    r2s = np.asarray(r2s, float)
    clip = np.clip(r2s, -1.0, None)
    finite = clip[np.isfinite(clip)]
    pack = {
        "concat_r2": concat_r2(recs_r2, min_bins=min_bins),
        "med_r2_clip": float(np.nanmedian(finite)) if finite.size else float("nan"),
        "mean_r2_clip": float(np.nanmean(finite)) if finite.size else float("nan"),
        "pct_r2_04": float(100.0 * np.mean(finite > 0.4)) if finite.size else float("nan"),
        "n_trials": int(finite.size),
        "onset_mae": float(np.mean(on_err)) if on_err else float("nan"),
        "peak_mae": float(np.mean(pk_err)) if pk_err else float("nan"),
        "peak_pearson": pearson(t_true, t_pred),
        "peak_ratio": float(np.median(pk_ratio)) if pk_ratio else float("nan"),
        "n_onset": int(len(on_err)),
        "n_peak": int(len(pk_err)),
        "clusters": {},
    }
    by = defaultdict(list)
    for k, i in enumerate(idx):
        t = trials[int(i)]
        cid = int(t.get("atlas_c", -2))
        by[cid].append(k)
    names = {int(t["atlas_c"]): t.get("atlas_name", str(t["atlas_c"])) for t in trials}
    for cid, loc in by.items():
        sub_r2 = [recs_r2[j] for j in loc]
        sub_tm = [recs_time[j] for j in loc]
        r2s_c, on_c, pk_c, pr_c = [], [], [], []
        tt, tp = [], []
        for rec, rec_t in zip(sub_r2, sub_tm):
            r2s_c.append(trial_r2(rec["y"], rec["pred"], min_bins))
            yt, yp = series(rec_t["y"]), series(rec_t["pred"])
            on_y, on_p = onset_bin(yt, thr), onset_bin(yp, thr)
            if on_y is not None and on_p is not None:
                on_c.append(abs(on_y - on_p) * mv.BINSIZE)
            py, ay = peak_bin(yt)
            pp, ap = peak_bin(yp)
            if py is not None and pp is not None:
                pk_c.append(abs(py - pp) * mv.BINSIZE)
                tt.append(py * mv.BINSIZE)
                tp.append(pp * mv.BINSIZE)
                if ay is not None and abs(ay) > 1e-6:
                    pr_c.append(abs(ap) / abs(ay))
        arr = np.asarray(r2s_c, float)
        clip_c = np.clip(arr, -1.0, None)
        fin = clip_c[np.isfinite(clip_c)]
        pack["clusters"][str(cid)] = {
            "id": int(cid),
            "name": names.get(cid, str(cid)),
            "concat_r2": concat_r2(sub_r2, min_bins=min_bins),
            "med_r2_clip": float(np.nanmedian(fin)) if fin.size else float("nan"),
            "mean_r2_clip": float(np.nanmean(fin)) if fin.size else float("nan"),
            "pct_r2_04": float(100.0 * np.mean(fin > 0.4)) if fin.size else float("nan"),
            "n_trials": int(fin.size),
            "onset_mae": float(np.mean(on_c)) if on_c else float("nan"),
            "peak_mae": float(np.mean(pk_c)) if pk_c else float("nan"),
            "peak_pearson": pearson(tt, tp),
            "peak_ratio": float(np.median(pr_c)) if pr_c else float("nan"),
        }
    return pack


def load_atlas(trials, target):
    blob = json.loads(CLUSTER_JSON.read_text())
    by = {}
    for rec in blob["trials"]:
        tid = rec.get("trial_index")
        tid = None if tid is None else int(tid)
        by[(str(rec["eid"]), tid)] = rec
    names = blob["wheel_names"] if target == "wheel_speed" else blob["paw_names"]
    n_miss = 0
    for t in trials:
        tid = t.get("trial_index")
        tid = None if tid is None else int(tid)
        rec = by.get((str(t["eid"]), tid))
        if rec is None:
            n_miss += 1
            t["atlas_c"] = -2
            t["atlas_name"] = "not_in_atlas"
            continue
        cl = int(rec["wheel_c"] if target == "wheel_speed" else rec["paw_c"])
        t["atlas_c"] = cl
        if cl >= 0:
            t["atlas_name"] = names[cl] if cl < len(names) else f"c{cl}"
        elif cl == -1:
            t["atlas_name"] = "unclustered"
        else:
            t["atlas_name"] = "not_in_atlas"
    print(f"atlas {target}: miss={n_miss}", flush=True)
    return names


def summarize(fold_packs):
    keys = ["concat_r2", "med_r2_clip", "mean_r2_clip", "pct_r2_04", "onset_mae", "peak_mae", "peak_pearson", "peak_ratio"]
    out = {"n_folds": len(fold_packs)}
    for k in keys:
        m, se, n = jack([p[k] for p in fold_packs])
        out[k] = m
        out[k + "_se"] = se
        out[k + "_n"] = n
    # clusters
    ids = sorted({cid for p in fold_packs for cid in p["clusters"]}, key=lambda s: int(s))
    clusters = []
    for cid in ids:
        rows = [p["clusters"][cid] for p in fold_packs if cid in p["clusters"] and p["clusters"][cid]["n_trials"] >= 1]
        if not rows:
            continue
        rec = {
            "id": int(cid),
            "name": rows[0]["name"],
            "n_folds": len(rows),
            "n_trials_mean": float(np.mean([r["n_trials"] for r in rows])),
        }
        for k in keys:
            m, se, n = jack([r[k] for r in rows])
            rec[k] = m
            rec[k + "_se"] = se
        clusters.append(rec)
    out["clusters"] = clusters
    return out


def main():
    slurm_utils.set_device("auto", probe="gru")
    print("device", slurm_utils.device_of(), flush=True)
    trials, _ = mv.load_corpus()
    result = {"device": str(slurm_utils.device_of()), "model": "attn_a1_gru", "targets": {}}
    for target in TARGETS:
        mv.configure_out(target, "cv")
        mv.set_target(trials, target, log1p=False)
        names = load_atlas(trials, target)
        specs = [s for s in mv.cv_folds(trials) if s["task"] in TASKS]
        by_task = defaultdict(list)
        for spec in specs:
            task, fid = spec["task"], spec["fold_id"]
            mv.clear_bin_masks(trials)
            bin_mode = spec.get("bin_mode")
            if bin_mode:
                mv.apply_bin_masks(trials, spec)
                train_full = mv.usable_idx(
                    trials, spec["train"], min_bins=max(4, mv.MIN_BINS // 2), mask_key="train_bins"
                )
                test_ok = mv.usable_idx(trials, spec["test"], min_bins=1, mask_key="test_bins")
                test_idx = np.intersect1d(train_full, test_ok)
            else:
                test_idx = mv.usable_idx(trials, spec["test"])
            if len(test_idx) < 1:
                print(f"  skip empty {target} {task}/{fid}", flush=True)
                continue
            cdir = mv.cache_dir(task, POOL, USE_ANATOMY, fold_id=fid, temporal="gru")
            ckpt = cdir / "model.pt"
            if not ckpt.exists():
                print(f"  missing {ckpt}", flush=True)
                continue
            blob = torch.load(ckpt, map_location="cpu", weights_only=False)
            stats = blob.get("stats")
            if stats is None:
                print(f"  no stats {ckpt}", flush=True)
                continue
            recs = mv.predict(blob, trials, test_idx, stats)
            recs_time = recs
            min_bins = MIN_BINS_INTERP if bin_mode else MIN_BINS_FULL
            if bin_mode:
                recs_r2 = mv.restrict_recs(recs, trials, test_idx, "test_bins")
            else:
                recs_r2 = recs
            pack = score_recs_extra(recs_r2, recs_time, test_idx, trials, target, min_bins)
            pack["task"] = task
            pack["fold_id"] = fid
            by_task[task].append(pack)
            print(
                f"  {target:12s} {task:14s} {fid:12s}  "
                f"R²={pack['concat_r2']:+.3f}  medclip={pack['med_r2_clip']:+.3f}  "
                f"pct>0.4={pack['pct_r2_04']:.1f}  on={pack['onset_mae']:.3f}s  "
                f"pk={pack['peak_mae']:.3f}s  n={pack['n_trials']}",
                flush=True,
            )
        result["targets"][target] = {
            "cluster_names": names,
            "tasks": {task: summarize(packs) for task, packs in by_task.items()},
        }
        OUT.write_text(json.dumps(result, indent=2, default=float))
        print("wrote", OUT, flush=True)
    print("done", flush=True)


if __name__ == "__main__":
    main()
