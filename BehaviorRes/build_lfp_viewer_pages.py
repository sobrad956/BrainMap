#!/usr/bin/env python3
"""Build LFP-only traces + metrics HTML pages from ExperimentsClean jobs.

Spine is the LFP-only GRU holdout for each LFP-tested fold. Spikes and
spikes+LFP curves are joined on (eid, trial_index). Spike predictions may
come from any fold of the same eval so LTO slice mismatch does not drop them.

Never glob model.pt. Never iterdir() jobs.
"""
from __future__ import annotations

import csv
import json
import math
import re
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

ROOT = Path("/Users/Stefan/Desktop/BrainMap/BehaviorRes")
JOBS = ROOT / "ExperimentsClean" / "paw_speed" / "jobs"
ATLAS = ROOT / "Modelv1Clusters" / "atlas_trial_clusters.json"
TRACES_SRC = ROOT / "experiments_clean_traces_clusters_full.html"
METRICS_SRC = ROOT / "experiments_clean_metrics.html"
TRACES_OUT = ROOT / "experiments_clean_traces_lfp.html"
METRICS_OUT = ROOT / "experiments_clean_metrics_lfp.html"

EVALS = ("LTO", "LSO", "LMO")
EXPECTED = {"LTO": 20, "LSO": 35, "LMO": 32}
DT = 0.02
MAX_JSON_BYTES = 40_000_000
MIN_CLUSTER_N = 5
PAW_ON = 150.0

LONG = {
    "gru": "a1_gru",
    "a1tf": "a1_transformer",
    "anat": "anat_transformer",
}
SHORT_FROM_LONG = {v: k for k, v in LONG.items()}
INPUTS = ("spikes", "lfp", "spike_lfp")
KEY_OF = {
    ("gru", "spikes"): "gru",
    ("gru", "lfp"): "gru_lfp",
    ("gru", "spike_lfp"): "gru_slfp",
    ("a1tf", "spikes"): "a1tf",
    ("a1tf", "lfp"): "a1tf_lfp",
    ("a1tf", "spike_lfp"): "a1tf_slfp",
    ("anat", "spikes"): "anat",
    ("anat", "lfp"): "anat_lfp",
    ("anat", "spike_lfp"): "anat_slfp",
}
MODELS_JS = """const MODELS = [
  { id: "gru", mse: "gru", huber: "gru_h", name: "A1 GRU (spikes)", color: "#6ea8fe" },
  { id: "gru_lfp", mse: "gru_lfp", huber: "gru_lfp_h", name: "A1 GRU (LFP)", color: "#4fc3f7" },
  { id: "gru_slfp", mse: "gru_slfp", huber: "gru_slfp_h", name: "A1 GRU (spikes+LFP)", color: "#0277bd" },
  { id: "a1tf", mse: "a1tf", huber: "a1tf_h", name: "A1 Transformer (spikes)", color: "#7dcea0" },
  { id: "a1tf_lfp", mse: "a1tf_lfp", huber: "a1tf_lfp_h", name: "A1 TF (LFP)", color: "#a9dfbf" },
  { id: "a1tf_slfp", mse: "a1tf_slfp", huber: "a1tf_slfp_h", name: "A1 TF (spikes+LFP)", color: "#1e8449" },
  { id: "anat", mse: "anat", huber: "anat_h", name: "Anat. Transformer (spikes)", color: "#c39bd3" },
  { id: "anat_lfp", mse: "anat_lfp", huber: "anat_lfp_h", name: "Anat. TF (LFP)", color: "#d7bde2" },
  { id: "anat_slfp", mse: "anat_slfp", huber: "anat_slfp_h", name: "Anat. TF (spikes+LFP)", color: "#6c3483" },
];"""

JOB_RE = re.compile(
    r"^(?P<job>\d+)_(?P<ev>LTO|LSO|LMO)_(?P<fold>\d+)_(?P<fid>.+)"
    r"_attn_(?P<model>a1_gru|a1_transformer|anat_transformer)"
    r"_trace_mse(?:_(?P<inp>lfp|spike_lfp|electrode))?$"
)

LEN_LABELS = ["8–17", "18–27", "28–47", "48+"]
R2_LABELS = ["<0", "0–0.2", "0.2–0.4", "0.4–0.6", "≥0.6"]
ONSET_EDGES = [-0.40, -0.20, -0.08, 0.08, 0.20, 0.40]
ONSET_LABELS = ["≤−0.40s", "−0.40–−0.20", "−0.20–−0.08", "−0.08–0.08", "0.08–0.20", "0.20–0.40", ">0.40s"]


def log(msg):
    print(msg, flush=True)


def fnum(x):
    try:
        v = float(x)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(v):
        return None
    return v


def eid8(eid):
    s = str(eid or "")
    return s[:8] if s else ""


def pair_key(eid, trial_index):
    try:
        ti = int(float(trial_index))
    except (TypeError, ValueError):
        return None
    e = str(eid or "").strip()
    if not e:
        return None
    return (e, ti)


def parse_job_dir(name):
    m = JOB_RE.match(name)
    if not m:
        return None
    inp = m.group("inp") or "spikes"
    if inp == "electrode":
        return None
    return {
        "job": int(m.group("job")),
        "ev": m.group("ev"),
        "fold": int(m.group("fold")),
        "fid": m.group("fid"),
        "model": SHORT_FROM_LONG[m.group("model")],
        "inp": inp,
    }


def index_jobs():
    """Map (ev, fid, model, inp) -> best holdout_trials.csv."""
    best = {}
    for csv_path in JOBS.glob("*_trace_mse*/holdout_trials.csv"):
        meta = parse_job_dir(csv_path.parent.name)
        if not meta:
            continue
        key = (meta["ev"], meta["fid"], meta["model"], meta["inp"])
        npz = csv_path.parent / "holdout_curves.npz"
        rank = (npz.exists(), meta["job"], csv_path.stat().st_mtime)
        prev = best.get(key)
        if prev is None or rank > prev[0]:
            best[key] = (rank, csv_path, meta)
    out = {k: (p, m) for k, (rank, p, m) in best.items()}
    log(f"indexed {len(out)} unique (eval, fold, model, input) jobs")
    return out


_CSV_CACHE = {}
_CURVE_CACHE = {}


def load_csv(path):
    key = str(path)
    hit = _CSV_CACHE.get(key)
    if hit is not None:
        return hit
    rows = []
    with path.open(newline="") as fh:
        for rec in csv.DictReader(fh):
            rows.append(rec)
    _CSV_CACHE[key] = rows
    return rows


def load_curves(csv_path):
    key = str(csv_path)
    hit = _CURVE_CACHE.get(key)
    if hit is not None:
        return hit
    npz_path = csv_path.parent / "holdout_curves.npz"
    if not npz_path.exists():
        empty = ({}, {})
        _CURVE_CACHE[key] = empty
        return empty
    z = np.load(npz_path)
    ids = np.asarray(z["trial_i"], dtype=int).reshape(-1)
    n_bins = np.asarray(z["n_bins"], dtype=int).reshape(-1)
    pred = np.asarray(z["pred"]).reshape(-1)
    y = np.asarray(z["y"]).reshape(-1)
    preds, ys = {}, {}
    off = 0
    for tid, n in zip(ids, n_bins):
        n = int(n)
        preds[int(tid)] = pred[off : off + n]
        ys[int(tid)] = y[off : off + n]
        off += n
    rec = (preds, ys)
    _CURVE_CACHE[key] = rec
    return rec


def trim_arr(arr):
    if arr is None:
        return []
    a = np.asarray(arr, dtype=np.float64).reshape(-1)
    n = a.size
    while n > 0:
        v = a[n - 1]
        if not np.isfinite(v) or v == 0.0:
            n -= 1
            continue
        break
    a = a[:n]
    out = []
    for v in a:
        if not np.isfinite(v):
            out.append(None)
        else:
            out.append(int(round(float(v))))
    return out


def extras(rec):
    out = {}
    mapping = {
        "pe": "pearson",
        "rm": "rmse",
        "ma": "mae",
        "mp": "mape",
        "on": "onset_abs_s",
        "pk": "peak_abs_s",
    }
    for short, col in mapping.items():
        v = fnum(rec.get(col))
        if v is None:
            continue
        out[short] = round(v, 4) if short in ("pe", "on", "pk") else round(v, 2)
    return out


def shape_bin(y):
    y = np.asarray(y, dtype=np.float32).reshape(-1)
    finite = np.isfinite(y)
    if int(finite.sum()) < 3:
        return "hill"
    mag = np.where(finite, np.abs(y), 0.0)
    peak_i = int(np.argmax(mag))
    peak = float(mag[peak_i])
    end_n = max(2, int(round(0.10 * mag.size)))
    if peak_i >= mag.size - end_n:
        return "hill"
    end = float(np.mean(mag[-end_n:]))
    floor = max(0.20 * peak, PAW_ON)
    return "peak" if end <= floor else "hill"


def len_bin(n):
    n = int(n)
    if n <= 17:
        return 0
    if n <= 27:
        return 1
    if n <= 47:
        return 2
    return 3


def r2_bin(v):
    if v is None or not math.isfinite(v):
        return None
    if v < 0:
        return 0
    if v < 0.2:
        return 1
    if v < 0.4:
        return 2
    if v < 0.6:
        return 3
    return 4


def onset_bin(err):
    if err is None or not math.isfinite(err):
        return None
    for i, e in enumerate(ONSET_EDGES):
        if err <= e:
            return i
    return len(ONSET_EDGES)


def load_atlas():
    blob = json.loads(ATLAS.read_text())
    by = {}
    for rec in blob.get("trials") or []:
        key = pair_key(rec.get("eid"), rec.get("trial_index"))
        if key:
            by[key] = rec
    paw_names = list(blob.get("paw_names") or [])
    wheel_names = list(blob.get("wheel_names") or [])
    sample = next(iter(by.values())) if by else {}
    log(f"atlas trials={len(by)} sample_keys={sorted(sample)[:24]}")
    return by, paw_names, wheel_names


def cluster_fields(atlas_rec, paw_names, wheel_names, y):
    sg = shape_bin(y) if y is not None and len(y) else "unclustered"
    if not atlas_rec:
        return {
            "c": -2,
            "cn": "not_in_atlas",
            "wc": -2,
            "wn": "not_in_atlas",
            "sg": sg,
            "sgn": sg,
        }
    pc = atlas_rec.get("paw_c", atlas_rec.get("atlas_c", -2))
    wc = atlas_rec.get("wheel_c", -2)
    try:
        pc = int(pc)
    except (TypeError, ValueError):
        pc = -2
    try:
        wc = int(wc)
    except (TypeError, ValueError):
        wc = -2
    if pc >= 0 and pc < len(paw_names):
        cn = paw_names[pc]
    elif pc == -1:
        cn = "unclustered"
    else:
        cn = "not_in_atlas"
    if wc >= 0 and wc < len(wheel_names):
        wn = wheel_names[wc]
    elif wc == -1:
        wn = "unclustered"
    else:
        wn = "not_in_atlas"
    man_p = atlas_rec.get("paw_manual") or atlas_rec.get("manual_paw") or cn
    man_w = atlas_rec.get("wheel_manual") or atlas_rec.get("manual_wheel") or wn
    return {
        "c": pc,
        "cn": str(man_p),
        "wc": wc,
        "wn": str(man_w),
        "sg": sg,
        "sgn": sg,
    }


def scores_row(csv_path):
    sc = csv_path.parent / "scores.csv"
    if not sc.exists():
        return {}
    rows = load_csv(sc)
    return rows[0] if rows else {}


def attach_model(trial, key, rec, curve):
    r2 = fnum(rec.get("r2")) if rec else None
    if r2 is not None:
        trial.setdefault("r", {})[key] = round(r2, 3)
    extras_d = extras(rec) if rec else {}
    if extras_d:
        trial.setdefault("k", {})[key] = extras_d
    if curve is not None and len(curve):
        trial.setdefault("p", {})[key] = trim_arr(curve)


def build_eval_pair_index(job_index, ev, model, inp):
    """(eid, trial_index) -> (rec, pred, y) across all folds of one eval."""
    out = {}
    for (e, fid, m, i), (csv_path, meta) in job_index.items():
        if e != ev or m != model or i != inp:
            continue
        rows = load_csv(csv_path)
        preds, ys = load_curves(csv_path)
        for rec in rows:
            key = pair_key(rec.get("eid"), rec.get("trial_index"))
            if not key:
                continue
            tid = rec.get("trial_i")
            curve = None
            y = None
            if tid not in (None, ""):
                curve = preds.get(int(float(tid)))
                y = ys.get(int(float(tid)))
            if key not in out:
                out[key] = (rec, curve, y)
    return out


def fid_sort_key(fid):
    m = re.match(r"^lto(\d+)$", str(fid))
    if m:
        return (0, int(m.group(1)))
    return (1, str(fid))


def build_traces_data(job_index, atlas, paw_names, wheel_names):
    fids = defaultdict(set)
    for (ev, fid, model, inp), _ in job_index.items():
        if inp == "lfp" and model == "gru":
            fids[ev].add(fid)
    data = {"evals": {}, "pref_fids": {}, "dt": DT}
    fold_scores = defaultdict(list)  # (ev, key) -> list of fold concat stats
    trial_rows = []  # for metrics clusters / extras

    for ev in EVALS:
        packs = {}
        ordered = sorted(fids.get(ev, []), key=fid_sort_key)
        log(f"{ev}: {len(ordered)} LFP folds")
        spike_ix = {
            stem: build_eval_pair_index(job_index, ev, stem, "spikes") for stem in LONG
        }
        slfp_ix = {
            stem: build_eval_pair_index(job_index, ev, stem, "spike_lfp") for stem in LONG
        }
        lfp_ix = {
            stem: build_eval_pair_index(job_index, ev, stem, "lfp") for stem in LONG
        }
        for fid in ordered:
            spine_key = (ev, fid, "gru", "lfp")
            if spine_key not in job_index:
                continue
            csv_path, meta = job_index[spine_key]
            rows = load_csv(csv_path)
            preds, ys = load_curves(csv_path)
            trials = []
            for rec in rows:
                key = pair_key(rec.get("eid"), rec.get("trial_index"))
                if not key:
                    continue
                tid = rec.get("trial_i")
                tid_i = int(float(tid)) if tid not in (None, "") else None
                y = ys.get(tid_i) if tid_i is not None else None
                y_list = trim_arr(y)
                if len(y_list) < 2:
                    continue
                atlas_rec = atlas.get(key)
                fields = cluster_fields(atlas_rec, paw_names, wheel_names, y)
                try:
                    session_i = int(float(rec.get("trial_index")))
                except (TypeError, ValueError):
                    session_i = tid_i if tid_i is not None else 0
                tr = {
                    "i": session_i,
                    "m": str(rec.get("mouse_id") or ""),
                    "e": eid8(rec.get("eid")),
                    "n": len(y_list),
                    "y": y_list,
                    "r": {},
                    "p": {},
                    "k": {},
                    **fields,
                }
                for stem in LONG:
                    for inp, ix in (("lfp", lfp_ix), ("spike_lfp", slfp_ix), ("spikes", spike_ix)):
                        short = KEY_OF[(stem, inp)]
                        hit = ix[stem].get(key)
                        if not hit:
                            continue
                        rec_m, curve_m, _y_m = hit
                        attach_model(tr, short, rec_m, curve_m)
                if "gru_lfp" not in tr["p"]:
                    curve = preds.get(tid_i) if tid_i is not None else None
                    attach_model(tr, "gru_lfp", rec, curve)
                trials.append(tr)
                trial_rows.append(
                    {
                        "ev": ev,
                        "fid": fid,
                        "key": key,
                        "n": tr["n"],
                        "c": str(tr["c"] if tr["c"] >= 0 else -1),
                        "cn": tr["cn"],
                        "sg": tr["sg"],
                        "r": dict(tr["r"]),
                        "on": {k: (tr["k"].get(k) or {}).get("on") for k in tr["r"]},
                        "onset_err": fnum(rec.get("onset_err_s")),
                    }
                )
            keys = sorted({k for t in trials for k in list(t.get("r", {}) ) + list(t.get("p", {}))})
            n_max = max((t["n"] for t in trials), default=0)
            packs[fid] = {
                "fid": fid,
                "n": len(trials),
                "n_max": n_max,
                "dt": DT,
                "keys": keys,
                "has_curves": any(t.get("p") for t in trials),
                "note": (
                    f"LFP holdout {ev} {fid}: {len(trials)} trials. "
                    "Spikes / LFP / spikes+LFP for A1 GRU, A1 Transformer, Anatomical Transformer (TRACE mse)."
                ),
                "trials": trials,
            }
            for stem in LONG:
                for inp in INPUTS:
                    jk = (ev, fid, stem, inp)
                    if jk not in job_index:
                        continue
                    sc = scores_row(job_index[jk][0])
                    short = KEY_OF[(stem, inp)]
                    fold_scores[(ev, short)].append(sc)
        if packs:
            pref = max(packs.items(), key=lambda kv: kv[1]["n"])[0]
            data["evals"][ev] = {"packs": packs}
            data["pref_fids"][ev] = pref
            log(f"  {ev} packs={len(packs)} pref={pref} n={packs[pref]['n']}")
    return data, fold_scores, trial_rows


def maybe_strip_curves(data):
    payload = json.dumps(data, separators=(",", ":"), ensure_ascii=False)
    nbytes = len(payload.encode("utf-8"))
    log(f"traces json {nbytes/1e6:.1f} MB before strip")
    if nbytes <= MAX_JSON_BYTES:
        return data, payload, False
    keep = {}
    for ev, evobj in data["evals"].items():
        packs = evobj["packs"]
        ranked = sorted(packs.items(), key=lambda kv: kv[1]["n"], reverse=True)
        keep[ev] = {fid for fid, _ in ranked[:2]}
    n_drop = 0
    for ev, evobj in data["evals"].items():
        for fid, pk in evobj["packs"].items():
            if fid in keep[ev]:
                continue
            pk["has_curves"] = False
            for t in pk["trials"]:
                if t.get("p"):
                    n_drop += 1
                t.pop("p", None)
                t.pop("y", None)
    payload = json.dumps(data, separators=(",", ":"), ensure_ascii=False)
    log(f"stripped pred/y from extra folds ({n_drop} trials); now {len(payload.encode('utf-8'))/1e6:.1f} MB")
    return data, payload, True


def stat_blob(vals):
    x = [float(v) for v in vals if v is not None and math.isfinite(float(v))]
    if not x:
        return {
            "n": 0,
            "mean": None,
            "med": None,
            "std": None,
            "sem": None,
            "p25": None,
            "p75": None,
            "lo": None,
            "hi": None,
        }
    arr = np.asarray(x, dtype=np.float64)
    mean = float(arr.mean())
    std = float(arr.std(ddof=1)) if arr.size > 1 else 0.0
    sem = std / math.sqrt(arr.size) if arr.size else 0.0
    p25 = float(np.percentile(arr, 25))
    p75 = float(np.percentile(arr, 75))
    med = float(np.median(arr))
    return {
        "n": int(arr.size),
        "mean": round(mean, 4),
        "med": round(med, 4),
        "std": round(std, 4),
        "sem": round(sem, 4),
        "p25": round(p25, 4),
        "p75": round(p75, 4),
        "lo": round(mean - sem, 4),
        "hi": round(mean + sem, 4),
    }


def metrics_row(ev, short, sc_rows):
    r2s, r2m, ons = [], [], []
    for sc in sc_rows:
        r2s.append(fnum(sc.get("fold_concat_r2")))
        r2m.append(fnum(sc.get("fold_mean_r2")))
        ons.append(fnum(sc.get("fold_median_onset_abs_s")))
    return {
        "e": ev,
        "m": short,
        "o": "trace",
        "l": "mse",
        "nf": len(sc_rows),
        "r2": stat_blob(r2s),
        "r2_med": stat_blob(r2m),
        "on": stat_blob(ons),
    }


def cluster_id(tr, mode):
    if mode == "learned":
        return tr["c"]
    if mode == "manual":
        return tr["cn"]
    return tr["sg"]


def cluster_across(trial_rows, mode):
    """mode: learned uses t.c; manual uses t.cn; shape uses t.sg."""
    r2_buckets = defaultdict(lambda: defaultdict(list))
    on_buckets = defaultdict(lambda: defaultdict(list))
    for tr in trial_rows:
        cid = cluster_id(tr, mode)
        ev = tr["ev"]
        fid = tr["fid"]
        for short, r2 in tr["r"].items():
            if r2 is not None:
                r2_buckets[(ev, short, cid)][fid].append(r2)
            on = tr["on"].get(short)
            if on is not None:
                on_buckets[(ev, short, cid)][fid].append(on)
    out = []
    for (ev, short, cid), by_fid in r2_buckets.items():
        fold_means = [float(np.mean(vals)) for vals in by_fid.values() if len(vals) >= MIN_CLUSTER_N]
        if not fold_means:
            continue
        fold_ons = [
            float(np.mean(vals))
            for vals in on_buckets[(ev, short, cid)].values()
            if len(vals) >= MIN_CLUSTER_N
        ]
        out.append(
            {
                "e": ev,
                "m": short,
                "o": "trace",
                "l": "mse",
                "g": mode,
                "c": str(cid),
                "r2": stat_blob(fold_means),
                "on": stat_blob(fold_ons),
            }
        )
    return out


def build_trace_extra(trial_rows):
    """Pooled holdout-trial extras. Keys match extraMatch: e, g, c, m, l."""
    by = defaultdict(lambda: defaultdict(list))
    dist = defaultdict(lambda: defaultdict(lambda: [0, 0, 0, 0, 0]))
    n_len = defaultdict(lambda: defaultdict(int))
    on_hist = defaultdict(lambda: [0] * len(ONSET_LABELS))
    on_n = defaultdict(int)

    def add(e, g, c, short, n, r2, onset_err):
        key = (e, g, str(c), short)
        li = len_bin(n)
        by[key][li].append(r2)
        rb = r2_bin(r2)
        if rb is not None:
            dist[key][li][rb] += 1
            n_len[key][li] += 1
        ob = onset_bin(onset_err)
        if ob is not None:
            on_hist[key][ob] += 1
            on_n[key] += 1

    shorts = set()
    for tr in trial_rows:
        ev = tr["ev"]
        for short, r2 in tr["r"].items():
            shorts.add(short)
            add(ev, "all", "all", short, tr["n"], r2, tr.get("onset_err"))
            add(ev, "learned", tr["c"], short, tr["n"], r2, tr.get("onset_err"))
            add(ev, "manual", tr["cn"], short, tr["n"], r2, tr.get("onset_err"))
            add(ev, "shape", tr["sg"], short, tr["n"], r2, tr.get("onset_err"))

    med, onset = [], []
    for key in sorted(by, key=lambda k: (k[0], k[1], str(k[2]), k[3])):
        e, g, c, short = key
        meds, ns, dists = [], [], []
        for li in range(4):
            vals = by[key].get(li) or []
            ns.append(len(vals))
            meds.append(None if not vals else round(float(np.median(vals)), 4))
            tot = n_len[key][li]
            counts = dist[key][li]
            dists.append([round(100.0 * x / tot, 1) for x in counts] if tot else [0, 0, 0, 0, 0])
        med.append({"e": e, "m": short, "l": "mse", "g": g, "c": c, "med": meds, "n": ns, "dist": dists})
        tot_on = on_n[key]
        hist = on_hist[key]
        onset.append(
            {
                "e": e,
                "m": short,
                "l": "mse",
                "g": g,
                "c": c,
                "p": [round(100.0 * x / tot_on, 1) if tot_on else 0 for x in hist],
            }
        )
    return {
        "length_labels": LEN_LABELS,
        "r2_labels": R2_LABELS,
        "onset_labels": ONSET_LABELS,
        "med": med,
        "onset": onset,
    }


def build_metrics_data(data_traces, fold_scores, trial_rows, paw_names):
    across = []
    n_folds = {}
    for ev in EVALS:
        packs = (data_traces.get("evals") or {}).get(ev, {}).get("packs") or {}
        n_folds[ev] = len(packs)
        for short in [KEY_OF[(s, i)] for s in LONG for i in INPUTS]:
            across.append(metrics_row(ev, short, fold_scores.get((ev, short), [])))
    learned = cluster_across(trial_rows, "learned")
    manual = cluster_across(trial_rows, "manual")
    shape = cluster_across(trial_rows, "shape")
    names_learned = {str(i): paw_names[i] if i < len(paw_names) else f"c{i}" for i in range(len(paw_names))}
    names_learned["-1"] = "unclustered"
    names_manual = {}
    for tr in trial_rows:
        names_manual[str(tr["cn"])] = tr["cn"]
    names_shape = {"peak": "peak", "hill": "hill", "unclustered": "unclustered"}
    extra = build_trace_extra(trial_rows)
    return {
        "meta": {
            "snapshot": datetime.now(timezone.utc).strftime("%Y-%m-%d"),
            "n_folds": n_folds,
            "expected_folds": EXPECTED,
            "missing_lmo": [],
            "note": "LFP CV: spikes / LFP-only / spikes+LFP, TRACE mse, A1 GRU + A1 Transformer + Anatomical Transformer.",
        },
        "across": across,
        "clusters": {
            "min_n": MIN_CLUSTER_N,
            "names": {
                "learned": names_learned,
                "manual": names_manual,
                "shape": names_shape,
            },
            "across": learned + manual + shape,
        },
        "trace_extra": extra,
    }


def copy_traces_shell(json_text, pref):
    out_lines = []
    skipped = False
    with TRACES_SRC.open(encoding="utf-8") as fh:
        for line in fh:
            if (not skipped) and 'id="trace-json"' in line:
                out_lines.append(
                    '<script id="trace-json" type="application/json">' + json_text + "</script>\n"
                )
                skipped = True
                continue
            out_lines.append(line)
    text = "".join(out_lines)
    text = text.replace(
        "<title>ExperimentsClean traces · clusters · complete folds</title>",
        "<title>ExperimentsClean traces · LFP folds · spikes / LFP / spikes+LFP</title>",
    )
    text = text.replace(
        "<h1>ExperimentsClean traces · clusters · complete-model folds</h1>",
        "<h1>ExperimentsClean traces · LFP folds · spikes / LFP / spikes+LFP</h1>",
    )
    text = text.replace(
        'Defaults are the complete TRACE folds with the most holdout trials: LTO lto18, LSO 81a78eac, LMO CSH_ZAD_026 (hierarchical + both GNNs present). Secondary complete folds: lto05 / cef05f87 / PL033. LFP-only and spikes+LFP (MSE) are included for A1 GRU, A1 Transformer, and Anatomical Transformer; they hide under huber_move because those jobs were MSE-only. <a href="experiments_clean_metrics.html" style="color:var(--accent)">Across-fold metrics (SEM)</a>',
        'Every pack is an LFP-tested TRACE/mse fold. The trial list is the LFP-only GRU holdout, so LFP and spikes+LFP curves are present on every trial. Spike-only curves are joined on (eid, trial_index) from any fold of the same eval. Models: A1 GRU, A1 Transformer, Anatomical Transformer × spikes / LFP / spikes+LFP. Cluster filters (learned HDBSCAN, manual atlas names, rule-based shape) still apply. <a href="experiments_clean_metrics_lfp.html" style="color:var(--accent)">LFP across-fold metrics (SEM)</a>',
    )
    old_models = re.search(r"const MODELS = \[[\s\S]*?\];", text)
    if not old_models:
        raise SystemExit("MODELS array not found in traces copy")
    text = text[: old_models.start()] + MODELS_JS + text[old_models.end() :]
    text = text.replace(
        'const pref = { LTO: "lto18", LSO: "81a78eac", LMO: "CSH_ZAD_026" }[state.ev];',
        "const pref = (DATA.pref_fids && DATA.pref_fids[state.ev]) || null;",
    )
    text = text.replace(
        """  pillRow(
    document.getElementById("lossPills"),
    [
      { id: "mse", label: "mse" },
      { id: "huber", label: "huber_move" },
    ],
    state.loss,
    (id) => { state.loss = id; render(); },
  );""",
        """  const lossEl = document.getElementById("lossPills");
  if (lossEl) lossEl.style.display = "none";
  state.loss = "mse";""",
    )
    text = text.replace(
        """function presentModels(tr) {
  const keys = packModelKeys();
  return MODELS.filter((m) => {
    const k = lossKey(m);
    if (!keys.has(k)) return false;
    if (!k.includes("lfp")) return true;
    const hasR = tr && tr.r && typeof tr.r[k] === "number";
    const hasP = tr && tr.p && Array.isArray(tr.p[k]) && tr.p[k].length >= 2;
    return hasR || hasP;
  });
}""",
        """function presentModels(tr) {
  const keys = packModelKeys();
  return MODELS.filter((m) => keys.has(lossKey(m)));
}""",
    )
    # fold pills become a select when there are many LFP folds
    text = text.replace(
        '    <div class="row" id="foldPills"></div>',
        '    <div class="row" id="foldPills"></div>\n    <div class="row"><select id="foldSel" class="trial"></select></div>',
    )
    text = text.replace(
        """  const fmap = foldMap();
  const foldEl = document.getElementById("foldPills");
  if (foldEl) {
    pillRow(
      foldEl,
      availableFids().map((id) => {
        const pk = fmap ? fmap[id] : pack();
        const nfam = pk && pk.keys
          ? new Set(pk.keys.map((k) => k.replace(/_h$/, "")).filter((k) => k !== "mean" && k !== "ar")).size
          : 0;
        return { id, label: id + " · " + (pk ? pk.n : 0) + " · " + nfam + " models" };
      }),
      state.fid,
      (id) => { state.fid = id; state.idx = 0; state.seekLfp = true; state.cluster = "all"; state.hover = null; render(); },
    );
  }""",
        """  const fmap = foldMap();
  const ids = availableFids();
  const foldItems = ids.map((id) => {
    const pk = fmap ? fmap[id] : pack();
    const nfam = pk && pk.keys
      ? new Set(pk.keys.map((k) => k.replace(/_h$/, "")).filter((k) => k !== "mean" && k !== "ar")).size
      : 0;
    return { id, label: id + " · " + (pk ? pk.n : 0) + " trials · " + nfam + " models" };
  });
  const foldEl = document.getElementById("foldPills");
  const foldSel = document.getElementById("foldSel");
  const onFold = (id) => { state.fid = id; state.idx = 0; state.seekLfp = false; state.cluster = "all"; state.hover = null; render(); };
  if (foldSel) {
    foldSel.style.display = ids.length ? "" : "none";
    foldSel.innerHTML = foldItems.map((it) => `<option value="${it.id}">${it.label}</option>`).join("");
    foldSel.value = state.fid || "";
    foldSel.onchange = (e) => onFold(e.target.value);
  }
  if (foldEl) {
    foldEl.style.display = ids.length && ids.length <= 6 ? "" : "none";
    if (ids.length && ids.length <= 6) pillRow(foldEl, foldItems, state.fid, onFold);
  }""",
    )
    if "seekLfp: true" in text:
        text = text.replace("seekLfp: true", "seekLfp: false")
    if "const MODELS = [" in text and "A1 GRU (spikes)" not in text:
        raise SystemExit("MODELS replace did not stick")
    if "foldSel" not in text:
        raise SystemExit("foldSel not inserted")
    text = text.replace(
        'Concat-bin trial R² from holdout_trials.csv. Mean and AR persistence are excluded. Wheel subplot is ground-truth |ω| from ModelDataRightContra. Native 20 ms bins, not downsampled. Trailing bins with no camera support (NaN, previously drawn as 0) are omitted. Source: ExperimentsClean/paw_speed/jobs/*/holdout_curves.npz · TRACE objective.',
        'Concat-bin trial R² from LFP-fold holdout_trials.csv. Mean and AR omitted. Native 20 ms bins. Source: ExperimentsClean LFP / spikes+LFP / spikes TRACE mse jobs for A1 GRU, A1 Transformer, Anatomical Transformer.',
    )
    return text


def copy_metrics_shell(json_text):
    out_lines = []
    skipped = False
    with METRICS_SRC.open(encoding="utf-8") as fh:
        for line in fh:
            if (not skipped) and 'id="metrics-json"' in line:
                out_lines.append(
                    '<script id="metrics-json" type="application/json">' + json_text + "</script>\n"
                )
                skipped = True
                continue
            out_lines.append(line)
    text = "".join(out_lines)
    text = text.replace(
        "<title>ExperimentsClean metrics · across folds</title>",
        "<title>ExperimentsClean metrics · LFP folds</title>",
    )
    text = text.replace(
        "<h1>ExperimentsClean · across-fold metrics</h1>",
        "<h1>ExperimentsClean · LFP across-fold metrics</h1>",
    )
    text = text.replace(
        '<p class="cap"><a href="experiments_clean_traces_clusters_full.html" style="color:var(--accent)">Holdout traces · clusters · complete folds</a></p>',
        '<p class="cap"><a href="experiments_clean_traces_lfp.html" style="color:var(--accent)">LFP holdout traces (spikes / LFP / spikes+LFP)</a></p>',
    )
    text = text.replace(
        'const ORDER = ["mean", "ar", "gru", "a1tf", "anat", "cnn", "hier", "gin", "gout"];',
        'const ORDER = ["gru", "gru_lfp", "gru_slfp", "a1tf", "a1tf_lfp", "a1tf_slfp", "anat", "anat_lfp", "anat_slfp"];',
    )
    text = text.replace(
        'const DISPLAY = {"mean": "Mean", "ar": "AR persist.", "gru": "A1 GRU", "a1tf": "A1 Transformer", "anat": "Anat. Transformer", "cnn": "Aniso CNN+TF", "hier": "Hierarchical", "gin": "GNN inner", "gout": "GNN outer"};',
        'const DISPLAY = {"gru": "GRU spikes", "gru_lfp": "GRU LFP", "gru_slfp": "GRU spikes+LFP", "a1tf": "A1 TF spikes", "a1tf_lfp": "A1 TF LFP", "a1tf_slfp": "A1 TF spikes+LFP", "anat": "Anat spikes", "anat_lfp": "Anat LFP", "anat_slfp": "Anat spikes+LFP"};',
    )
    text = text.replace(
        'const SHORT = {"mean": "Mean", "ar": "AR", "gru": "GRU", "a1tf": "A1 TF", "anat": "Anat", "cnn": "CNN", "hier": "Hier", "gin": "GNN-in", "gout": "GNN-out"};',
        'const SHORT = {"gru": "GRU", "gru_lfp": "GRU-LFP", "gru_slfp": "GRU+LFP", "a1tf": "A1", "a1tf_lfp": "A1-LFP", "a1tf_slfp": "A1+LFP", "anat": "Anat", "anat_lfp": "Anat-LFP", "anat_slfp": "Anat+LFP"};',
    )
    text = text.replace(
        'const OBJS = ["trace","onset","direction"];',
        'const OBJS = ["trace"];',
    )
    text = text.replace(
        'const MCOL = {"mean":"#9aa3b2","gru":"#6ea8fe","a1tf":"#7dcea0","anat":"#c39bd3","cnn":"#e8a87c","hier":"#76d7c4","gin":"#f5b7b1","gout":"#f7dc6f"};',
        'const MCOL = {"gru":"#6ea8fe","gru_lfp":"#4fc3f7","gru_slfp":"#0277bd","a1tf":"#7dcea0","a1tf_lfp":"#a9dfbf","a1tf_slfp":"#1e8449","anat":"#c39bd3","anat_lfp":"#d7bde2","anat_slfp":"#6c3483"};',
    )
    text = text.replace(
        'const TRACE_MS = ["mean","gru","a1tf","anat","cnn","hier","gin","gout"];',
        'const TRACE_MS = ["gru","gru_lfp","gru_slfp","a1tf","a1tf_lfp","a1tf_slfp","anat","anat_lfp","anat_slfp"];',
    )
    text = text.replace(
        """  pill(document.getElementById("objPills"), OBJS.map(o=>({id:o,label:o})), state.obj, id=>{state.obj=id; render();});
  const showLoss = state.obj==="trace";
  document.getElementById("lossPills").style.display = showLoss ? "" : "none";
  if (showLoss) pill(document.getElementById("lossPills"), [{id:"mse",label:"mse"},{id:"huber_move",label:"huber_move"}], state.loss, id=>{state.loss=id; render();});""",
        """  document.getElementById("objPills").style.display = "none";
  document.getElementById("lossPills").style.display = "none";
  state.obj = "trace";
  state.loss = "mse";""",
    )
    text = text.replace(
        '[dir ? "Mean baseline acc" : onset ? "Mean baseline |onset|" : ("Baseline R² ("+r2Noun+")"),\n     fmtCenter(mean && primary(mean), 3)],',
        '[ "Models", "spikes / LFP / spikes+LFP" ],',
    )
    return text


def update_index():
    path = ROOT / "index.html"
    if not path.exists():
        return
    text = path.read_text(encoding="utf-8")
    card = """  <a class="card" href="experiments_clean_traces_lfp.html">
    <strong>LFP traces · spikes / LFP / spikes+LFP</strong>
    <span>LFP-tested folds only. Prediction curves vs true paw speed, with cluster and LTO/LSO/LMO filters.</span>
  </a>
  <a class="card" href="experiments_clean_metrics_lfp.html">
    <strong>LFP across-fold metrics (SEM)</strong>
    <span>Concat-bin R² for spikes, LFP-only, and spikes+LFP on the LFP CV folds.</span>
  </a>
"""
    if "experiments_clean_traces_lfp.html" not in text:
        text = text.replace(
            '  <a class="card" href="experiments_clean_metrics.html">',
            card + '  <a class="card" href="experiments_clean_metrics.html">',
        )
        path.write_text(text, encoding="utf-8")
        log("updated index.html")


def main():
    job_index = index_jobs()
    atlas, paw_names, wheel_names = load_atlas()
    traces, fold_scores, trial_rows = build_traces_data(job_index, atlas, paw_names, wheel_names)
    traces, traces_json, stripped = maybe_strip_curves(traces)
    html = copy_traces_shell(traces_json, traces.get("pref_fids") or {})
    TRACES_OUT.write_text(html, encoding="utf-8")
    log(f"wrote {TRACES_OUT} {TRACES_OUT.stat().st_size/1e6:.1f} MB stripped={stripped}")

    metrics = build_metrics_data(traces, fold_scores, trial_rows, paw_names)
    metrics_json = json.dumps(metrics, separators=(",", ":"), ensure_ascii=False)
    mhtml = copy_metrics_shell(metrics_json)
    METRICS_OUT.write_text(mhtml, encoding="utf-8")
    log(f"wrote {METRICS_OUT} {METRICS_OUT.stat().st_size/1e6:.1f} MB")
    update_index()
    for ev, evobj in traces["evals"].items():
        packs = evobj["packs"]
        n_curve = sum(1 for p in packs.values() if p.get("has_curves"))
        log(f"summary {ev}: folds={len(packs)} with_curves={n_curve} pref={traces['pref_fids'].get(ev)}")


if __name__ == "__main__":
    main()
