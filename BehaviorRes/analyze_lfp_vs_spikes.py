#!/usr/bin/env python3
"""LFP vs spikes vs spikes+LFP: when each wins, encoder band weights, attention.

Never glob model.pt. Never JOBS.iterdir(). Named sibling model.pt only.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import pickle
import re
import sys
import traceback
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path("/Users/Stefan/Desktop/BrainMap/BehaviorRes")
sys.path.insert(0, str(ROOT))

ec = None
mv = None
torch = None
F = None


def _need_torch():
    global ec, mv, torch, F
    if torch is not None:
        return
    import torch as _torch
    import torch.nn.functional as _F
    import ExperimentsClean as _ec
    import Modelv1 as _mv

    torch = _torch
    F = _F
    ec = _ec
    mv = _mv

JOBS = ROOT / "ExperimentsClean" / "paw_speed" / "jobs"
ATLAS = ROOT / "Modelv1Clusters" / "atlas_trial_clusters.json"
OUT_JSON = ROOT / "lfp_vs_spikes_analysis.json"

BANDS = ("delta", "theta", "alpha", "beta", "low_gamma", "high_gamma")
BAND_LABELS = ["δ 0.5–4", "θ 4–8", "α 8–13", "β 13–30", "low-γ 30–55", "high-γ 65–100"]
DT = 0.02
PAW_ON = 150.0
MAX_LAG = 30
PROBE_FOLDS = (
    ("LTO", "lto04"),
    ("LSO", "81a78eac"),
    ("LMO", "CSH_ZAD_026"),
)
ATTN_PROBE = ("LSO", "81a78eac", "a1tf")
MAX_ATTN_TRIALS = 96
ATTN_BATCH = 4

JOB_RE = re.compile(
    r"^(?P<job>\d+)_(?P<ev>LTO|LSO|LMO)_(?P<fold>\d+)_(?P<fid>.+)"
    r"_attn_(?P<model>a1_gru|a1_transformer|anat_transformer)"
    r"_trace_mse(?:_(?P<inp>lfp|spike_lfp|electrode))?$"
)
SHORT = {"a1_gru": "gru", "a1_transformer": "a1tf", "anat_transformer": "anat"}
LONG_MODEL = {
    "gru": "attn_a1_gru",
    "a1tf": "attn_a1_transformer",
    "anat": "attn_anat_transformer",
}
INPUTS = ("spikes", "lfp", "spike_lfp")
LEN_EDGES = (18, 28, 48)
LEN_LABELS = ["8–17 bins", "18–27 bins", "28–47 bins", "48+ bins"]
HILL_END_FRAC = 0.10
PEAK_RETURN_FRAC = 0.20


def log(msg):
    print(msg, flush=True)


def jf(x, nd=6):
    if x is None:
        return None
    try:
        v = float(x)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(v):
        return None
    return round(v, nd)


def mean_sem(vals):
    a = np.asarray([v for v in vals if v is not None and math.isfinite(float(v))], dtype=np.float64)
    if a.size < 1:
        return None, None, 0
    m = float(a.mean())
    sem = float(a.std(ddof=1) / math.sqrt(a.size)) if a.size > 1 else 0.0
    return m, sem, int(a.size)


def pair_key(eid, trial_index):
    try:
        ti = int(float(trial_index))
    except (TypeError, ValueError):
        return None
    e = str(eid or "").strip()
    return (e, ti) if e else None


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
        "model": SHORT[m.group("model")],
        "inp": inp,
    }


def index_probe_jobs():
    """Name-only scan of job dirs for the three probe folds. Never glob model.pt."""
    best = {}
    needles = ("_lto04_", "_81a78eac_", "_CSH_ZAD_026_")
    n_seen = 0
    with os.scandir(JOBS) as it:
        for ent in it:
            if not ent.is_dir(follow_symlinks=False):
                continue
            name = ent.name
            if not any(n in name for n in needles):
                continue
            n_seen += 1
            meta = parse_job_dir(name)
            if not meta:
                continue
            csv_path = Path(ent.path) / "holdout_trials.csv"
            if not csv_path.exists():
                continue
            key = (meta["ev"], meta["fid"], meta["model"], meta["inp"])
            npz = Path(ent.path) / "holdout_curves.npz"
            sc = Path(ent.path) / "scores.csv"
            rank = (npz.exists(), sc.exists(), meta["job"], csv_path.stat().st_mtime)
            prev = best.get(key)
            if prev is None or rank > prev[0]:
                best[key] = (rank, csv_path, meta)
    log(f"probe dirs scanned={n_seen} unique jobs={len(best)}")
    return {k: (p, m) for k, (rank, p, m) in best.items()}


def load_trace_data():
    p = ROOT / "experiments_clean_traces_lfp_data.js"
    text = p.read_text(encoding="utf-8")
    prefix = "window.TRACE_DATA="
    if not text.startswith(prefix):
        raise SystemExit(f"unexpected TRACE_DATA header in {p}")
    text = text[len(prefix) :].rstrip()
    if text.endswith(";"):
        text = text[:-1]
    log(f"loading TRACE_DATA {p.stat().st_size / 1e6:.1f} MB")
    return json.loads(text)


def load_metrics_data():
    html = (ROOT / "experiments_clean_metrics_lfp.html").read_text(encoding="utf-8")
    key = 'id="metrics-json"'
    i = html.find(key)
    if i < 0:
        raise SystemExit("metrics-json not found")
    start = html.find(">", i) + 1
    end = html.find("</script>", start)
    return json.loads(html[start:end])


def len_bin(n):
    try:
        n = int(n)
    except (TypeError, ValueError):
        return None
    if n < 8:
        return None
    if n < LEN_EDGES[0]:
        return LEN_LABELS[0]
    if n < LEN_EDGES[1]:
        return LEN_LABELS[1]
    if n < LEN_EDGES[2]:
        return LEN_LABELS[2]
    return LEN_LABELS[3]


def rt_bin(onset_s):
    if onset_s is None or not (isinstance(onset_s, (int, float)) and math.isfinite(onset_s)):
        return "none"
    x = float(onset_s)
    if x < 0.20:
        return "early <0.20s"
    if x < 0.40:
        return "mid 0.20–0.40s"
    if x < 0.80:
        return "late 0.40–0.80s"
    return "very late ≥0.80s"


def shape_bin(y):
    y = np.asarray(y, dtype=np.float32).reshape(-1)
    finite = np.isfinite(y)
    if int(finite.sum()) < 3:
        return "hill"
    mag = np.where(finite, np.abs(y), 0.0)
    peak_i = int(np.argmax(mag))
    peak = float(mag[peak_i])
    end_n = max(2, int(round(HILL_END_FRAC * mag.size)))
    if peak_i >= mag.size - end_n:
        return "hill"
    end = float(np.mean(mag[-end_n:]))
    floor = max(PEAK_RETURN_FRAC * peak, PAW_ON)
    return "peak" if end <= floor else "hill"


def load_atlas():
    if not ATLAS.exists():
        log(f"no atlas at {ATLAS}")
        return {}, []
    blob = json.loads(ATLAS.read_text())
    by = {}
    for rec in blob.get("trials") or []:
        key = pair_key(rec.get("eid"), rec.get("trial_index"))
        if key:
            by[key] = rec
    names = list(blob.get("paw_names") or [])
    log(f"atlas trials={len(by)} paw_names={names}")
    return by, names


def cluster_name(rec, names):
    if not rec:
        return "not_in_atlas"
    try:
        pc = int(rec.get("paw_c", rec.get("atlas_c", -2)))
    except (TypeError, ValueError):
        return "not_in_atlas"
    if pc == -1:
        return "unclustered"
    if 0 <= pc < len(names):
        return names[pc]
    return f"cluster_{pc}"


def load_y_map(csv_path):
    npz_path = csv_path.parent / "holdout_curves.npz"
    if not npz_path.exists():
        return {}
    z = np.load(npz_path)
    ids = np.asarray(z["trial_i"], dtype=int).reshape(-1)
    n_bins = np.asarray(z["n_bins"], dtype=int).reshape(-1)
    y = np.asarray(z["y"]).reshape(-1)
    out = {}
    off = 0
    for tid, n in zip(ids, n_bins):
        n = int(n)
        out[int(tid)] = y[off : off + n]
        off += n
    return out


def read_scores(csv_path):
    sc = csv_path.parent / "scores.csv"
    if not sc.exists():
        return {}
    df = pd.read_csv(sc)
    if df.empty:
        return {}
    row = df.iloc[0].to_dict()
    return {k: row.get(k) for k in row}


def winner_of(r_spk, r_lfp, r_slfp, tie=1e-6):
    vals = {"spikes": r_spk, "lfp": r_lfp, "spike_lfp": r_slfp}
    if any(v is None or not math.isfinite(v) for v in vals.values()):
        return None
    best = max(vals.values())
    hits = [k for k, v in vals.items() if abs(v - best) <= tie]
    if len(hits) != 1:
        return "tie"
    return hits[0]


def slice_wins(rows, key_fn, min_n=25):
    buckets = defaultdict(list)
    for r in rows:
        k = key_fn(r)
        if k is None:
            continue
        buckets[k].append(r)
    out = []
    for k in sorted(buckets, key=lambda x: (-len(buckets[x]), str(x))):
        recs = buckets[k]
        if len(recs) < min_n:
            continue
        n = len(recs)
        wins = {"spikes": 0, "lfp": 0, "spike_lfp": 0, "tie": 0}
        r2 = {inp: [] for inp in INPUTS}
        d_lfp, d_slfp, fusion = [], [], []
        for r in recs:
            w = r["winner"]
            if w in wins:
                wins[w] += 1
            for inp in INPUTS:
                r2[inp].append(r[inp])
            d_lfp.append(r["lfp"] - r["spikes"])
            d_slfp.append(r["spike_lfp"] - r["spikes"])
            fusion.append(r["spike_lfp"] - max(r["spikes"], r["lfp"]))
        item = {
            "slice": k,
            "n": n,
            "win_spikes": round(100.0 * wins["spikes"] / n, 1),
            "win_lfp": round(100.0 * wins["lfp"] / n, 1),
            "win_slfp": round(100.0 * wins["spike_lfp"] / n, 1),
            "win_tie": round(100.0 * wins["tie"] / n, 1),
        }
        for inp in INPUTS:
            m, sem, _ = mean_sem(r2[inp])
            item[f"mean_{inp}"] = jf(m, 4)
            item[f"sem_{inp}"] = jf(sem, 4)
        m, sem, _ = mean_sem(d_lfp)
        item["d_lfp_spk"] = jf(m, 4)
        item["d_lfp_spk_sem"] = jf(sem, 4)
        m, sem, _ = mean_sem(d_slfp)
        item["d_slfp_spk"] = jf(m, 4)
        m, sem, _ = mean_sem(fusion)
        item["fusion_vs_best"] = jf(m, 4)
        item["best_input"] = max(
            ("spikes", item["mean_spikes"]),
            ("lfp", item["mean_lfp"]),
            ("spike_lfp", item["mean_spike_lfp"]),
            key=lambda t: t[1] if t[1] is not None else -9,
        )[0]
        out.append(item)
    return out


MODEL_RKEYS = {
    "gru": {"spikes": "gru", "lfp": "gru_lfp", "spike_lfp": "gru_slfp"},
    "a1tf": {"spikes": "a1tf", "lfp": "a1tf_lfp", "spike_lfp": "a1tf_slfp"},
    "anat": {"spikes": "anat", "lfp": "anat_lfp", "spike_lfp": "anat_slfp"},
}
METRICS_SHORT = {
    "gru": ("gru", "spikes"),
    "gru_lfp": ("gru", "lfp"),
    "gru_slfp": ("gru", "spike_lfp"),
    "a1tf": ("a1tf", "spikes"),
    "a1tf_lfp": ("a1tf", "lfp"),
    "a1tf_slfp": ("a1tf", "spike_lfp"),
    "anat": ("anat", "spikes"),
    "anat_lfp": ("anat", "lfp"),
    "anat_slfp": ("anat", "spike_lfp"),
}


def phase_perf(jobs=None):
    traces = load_trace_data()
    metrics = load_metrics_data()
    paw_names = list(((metrics.get("clusters") or {}).get("names") or {}).get("learned") or {})
    matched_rows = []
    fold_wins = defaultdict(list)
    n_complete = 0
    evals = traces.get("evals") or {}
    for ev, evobj in evals.items():
        packs = evobj.get("packs") or {}
        for fid, pk in packs.items():
            trials = pk.get("trials") or []
            for model, keys in MODEL_RKEYS.items():
                wins = {"spikes": 0, "lfp": 0, "spike_lfp": 0, "tie": 0}
                n_here = 0
                for t in trials:
                    r = t.get("r") or {}
                    vals = {}
                    ok = True
                    for inp, rk in keys.items():
                        v = r.get(rk)
                        if not isinstance(v, (int, float)) or not math.isfinite(float(v)):
                            ok = False
                            break
                        vals[inp] = float(v)
                    if not ok:
                        continue
                    w = winner_of(vals["spikes"], vals["lfp"], vals["spike_lfp"])
                    if w is None:
                        continue
                    wins[w] += 1
                    n_here += 1
                    n_bins = t.get("n")
                    on_err = None
                    kextra = (t.get("k") or {}).get(keys["spikes"]) or {}
                    on_err = kextra.get("on")
                    matched_rows.append(
                        {
                            "ev": ev,
                            "fid": fid,
                            "model": model,
                            "spikes": vals["spikes"],
                            "lfp": vals["lfp"],
                            "spike_lfp": vals["spike_lfp"],
                            "winner": w,
                            "n_bins": n_bins,
                            "len": len_bin(n_bins),
                            "onset_err": on_err,
                            "rt": (
                                "onset err <0.08s"
                                if isinstance(on_err, (int, float)) and on_err < 0.08
                                else (
                                    "onset err 0.08–0.20s"
                                    if isinstance(on_err, (int, float)) and on_err < 0.20
                                    else (
                                        "onset err ≥0.20s"
                                        if isinstance(on_err, (int, float))
                                        else "onset err unknown"
                                    )
                                )
                            ),
                            "shape": t.get("sgn") or t.get("sg"),
                            "cluster": str(t.get("cn") or t.get("c") or "unknown"),
                            "mouse": t.get("m"),
                            "eid": t.get("e"),
                        }
                    )
                if n_here:
                    n_complete += 1
                    fold_wins[ev].append(
                        {
                            "fid": fid,
                            "model": model,
                            "n": n_here,
                            "win_spikes": wins["spikes"] / n_here,
                            "win_lfp": wins["lfp"] / n_here,
                            "win_slfp": wins["spike_lfp"] / n_here,
                        }
                    )
    log(f"complete 3-input fold packs: {n_complete}; matched trials: {len(matched_rows)}")

    concat_lookup = {}
    for rec in metrics.get("across") or []:
        pair = METRICS_SHORT.get(rec.get("m"))
        if not pair:
            continue
        model, inp = pair
        r2 = rec.get("r2") or {}
        concat_lookup[(rec.get("e"), model, inp)] = {
            "mean": jf(r2.get("mean"), 4),
            "sem": jf(r2.get("sem"), 4),
            "n": int(r2.get("n") or rec.get("nf") or 0),
            "med": jf(r2.get("med"), 4),
        }
    concat_table = []
    win_table = []
    for ev in ("LTO", "LSO", "LMO"):
        for model in ("gru", "a1tf", "anat"):
            row = {"ev": ev, "model": model}
            for inp in INPUTS:
                hit = concat_lookup.get((ev, model, inp)) or {}
                row[f"{inp}_mean"] = hit.get("mean")
                row[f"{inp}_sem"] = hit.get("sem")
                row[f"{inp}_n"] = hit.get("n")
            concat_table.append(row)
            wrecs = [r for r in fold_wins[ev] if r["model"] == model]
            if not wrecs:
                continue
            wr = {"ev": ev, "model": model, "n_folds": len(wrecs), "n_trials": int(sum(r["n"] for r in wrecs))}
            for src, dst in (
                ("win_spikes", "win_spikes"),
                ("win_lfp", "win_lfp"),
                ("win_slfp", "win_slfp"),
            ):
                m, sem, n = mean_sem([r[src] for r in wrecs])
                wr[dst] = jf(100.0 * m, 1) if m is not None else None
                wr[dst + "_sem"] = jf(100.0 * sem, 1) if sem is not None else None
            sp = concat_lookup.get((ev, model, "spikes")) or {}
            lf = concat_lookup.get((ev, model, "lfp")) or {}
            sl = concat_lookup.get((ev, model, "spike_lfp")) or {}
            if sp.get("mean") is not None and lf.get("mean") is not None:
                wr["d_concat_lfp"] = jf(lf["mean"] - sp["mean"], 4)
            if sp.get("mean") is not None and sl.get("mean") is not None:
                wr["d_concat_slfp"] = jf(sl["mean"] - sp["mean"], 4)
            win_table.append(wr)

    by_eval_model = defaultdict(list)
    for r in matched_rows:
        by_eval_model[(r["ev"], r["model"])].append(r)

    fusion_table = []
    for ev in ("LTO", "LSO", "LMO"):
        for model in ("gru", "a1tf", "anat"):
            recs = by_eval_model.get((ev, model), [])
            if not recs:
                continue
            gain = [r["spike_lfp"] - max(r["spikes"], r["lfp"]) for r in recs]
            unique = sum(1 for r in recs if r["winner"] == "spike_lfp")
            lfp_beats = sum(1 for r in recs if r["lfp"] > r["spikes"] + 1e-6)
            m, sem, n = mean_sem(gain)
            fusion_table.append(
                {
                    "ev": ev,
                    "model": model,
                    "n": n,
                    "fusion_gain": jf(m, 4),
                    "fusion_gain_sem": jf(sem, 4),
                    "pct_slfp_unique": round(100.0 * unique / max(n, 1), 1),
                    "pct_lfp_beats_spk": round(100.0 * lfp_beats / max(n, 1), 1),
                }
            )

    # Prefer anatomical transformer for slice stories (strongest LFP on LSO), plus pooled across models.
    def pooled(ev=None, model=None):
        out = []
        for r in matched_rows:
            if ev and r["ev"] != ev:
                continue
            if model and r["model"] != model:
                continue
            out.append(r)
        return out

    slices = {
        "length_LTO_anat": slice_wins(pooled("LTO", "anat"), lambda r: r["len"], min_n=40),
        "length_all_anat": slice_wins(pooled(model="anat"), lambda r: (r["ev"], r["len"]), min_n=30),
        "shape_by_eval_anat": slice_wins(pooled(model="anat"), lambda r: f"{r['ev']} {r['shape']}", min_n=40),
        "onseterr_LTO_anat": slice_wins(pooled("LTO", "anat"), lambda r: r["rt"], min_n=30),
        "cluster_LTO_anat": slice_wins(pooled("LTO", "anat"), lambda r: r["cluster"], min_n=40),
        "cluster_LSO_anat": slice_wins(pooled("LSO", "anat"), lambda r: r["cluster"], min_n=25),
        "eval_anat": slice_wins(pooled(model="anat"), lambda r: r["ev"], min_n=40),
        "length_LTO_all_models": slice_wins(pooled("LTO"), lambda r: r["len"], min_n=80),
        "shape_LTO_all_models": slice_wins(pooled("LTO"), lambda r: r["shape"], min_n=80),
    }

    # Per-session LTO: which sessions does LFP win?
    sess_rows = []
    by_sess = defaultdict(list)
    for r in pooled("LTO", "anat"):
        by_sess[r["eid"]].append(r)
    for eid, recs in by_sess.items():
        if len(recs) < 8:
            continue
        m_spk, _, n = mean_sem([x["spikes"] for x in recs])
        m_lfp, _, _ = mean_sem([x["lfp"] for x in recs])
        m_sl, _, _ = mean_sem([x["spike_lfp"] for x in recs])
        wins = {"spikes": 0, "lfp": 0, "spike_lfp": 0}
        for x in recs:
            if x["winner"] in wins:
                wins[x["winner"]] += 1
        best = max(("spikes", m_spk), ("lfp", m_lfp), ("spike_lfp", m_sl), key=lambda t: t[1] or -9)[0]
        sess_rows.append(
            {
                "eid8": str(eid)[:8],
                "mouse": recs[0]["mouse"],
                "n": n,
                "mean_spikes": jf(m_spk, 4),
                "mean_lfp": jf(m_lfp, 4),
                "mean_slfp": jf(m_sl, 4),
                "d_lfp": jf((m_lfp or 0) - (m_spk or 0), 4),
                "best": best,
                "win_lfp": round(100.0 * wins["lfp"] / max(n, 1), 1),
            }
        )
    sess_rows.sort(key=lambda r: r["d_lfp"] if r["d_lfp"] is not None else 0)

    return {
        "n_matched_trials": len(matched_rows),
        "n_complete_packs": n_complete,
        "concat": concat_table,
        "win_rates": win_table,
        "fusion": fusion_table,
        "slices": slices,
        "lto_sessions_anat": {
            "lfp_best": [s for s in sess_rows if s["best"] == "lfp"][:8],
            "spikes_best_worst_lfp": sess_rows[:8],
            "n_sessions": len(sess_rows),
            "n_lfp_best": sum(1 for s in sess_rows if s["best"] == "lfp"),
            "n_slfp_best": sum(1 for s in sess_rows if s["best"] == "spike_lfp"),
            "n_spk_best": sum(1 for s in sess_rows if s["best"] == "spikes"),
        },
        "paw_names": paw_names,
    }


def encoder_band_weights(sd, feat_dim, use_anatomy):
    _need_torch()
    W = None
    for k, v in sd.items():
        if k.endswith("enc.mlp.0.weight") or k == "enc.mlp.0.weight":
            W = np.asarray(v.detach().cpu().numpy() if torch.is_tensor(v) else v)
            break
    if W is None:
        return None
    # in_dim = feat_dim + 1  (A1) or feat_dim + 3 + area_dim + 1 (anat)
    cols = W[:, : int(feat_dim)]
    mag = np.abs(cols).mean(axis=0)
    tot = float(mag.sum()) + 1e-12
    share = (mag / tot).tolist()
    out = {"feat_dim": int(feat_dim), "in_dim": int(W.shape[1]), "d_model": int(W.shape[0]), "share": [jf(s, 4) for s in share]}
    if int(feat_dim) == 6:
        out["bands"] = {BANDS[i]: jf(share[i], 4) for i in range(6)}
        out["mag"] = {BANDS[i]: jf(float(mag[i]), 5) for i in range(6)}
    elif int(feat_dim) == 7:
        out["spike_share"] = jf(share[0], 4)
        out["lfp_share"] = jf(sum(share[1:]), 4)
        out["bands"] = {BANDS[i]: jf(share[i + 1], 4) for i in range(6)}
        out["mag"] = {"spike": jf(float(mag[0]), 5), **{BANDS[i]: jf(float(mag[i + 1]), 5) for i in range(6)}}
    elif int(feat_dim) == 1:
        out["spike_share"] = 1.0
    # rate column is last of the neural+rate block for A1
    if int(W.shape[1]) >= int(feat_dim) + 1:
        rate_mag = float(np.abs(W[:, feat_dim]).mean())
        out["rate_vs_feat"] = jf(rate_mag / (float(np.abs(cols).mean()) + 1e-12), 3)
    return out


def phase_weights(jobs):
    rows = []
    missing = []
    wanted = []
    for ev, fid in PROBE_FOLDS:
        for model in ("gru", "a1tf", "anat"):
            for inp in INPUTS:
                wanted.append((ev, fid, model, inp))
    for key in wanted:
        pack = jobs.get(key)
        if pack is None:
            missing.append({"key": list(key), "reason": "no_job"})
            continue
        path, meta = pack
        ckpt = path.parent / "model.pt"
        if not ckpt.exists():
            missing.append({"key": list(key), "reason": "no_ckpt", "dir": path.parent.name})
            continue
        try:
            blob = torch.load(ckpt, map_location="cpu", weights_only=False)
        except Exception as err:
            missing.append({"key": list(key), "reason": f"load:{err}"})
            continue
        sd = blob.get("state_dict") or blob
        feat_dim = int(blob.get("feat_dim") or ec.feat_dim_of(key[3]))
        use_anatomy = key[2] == "anat"
        w = encoder_band_weights(sd, feat_dim, use_anatomy)
        if w is None:
            missing.append({"key": list(key), "reason": "no_mlp0"})
            continue
        rows.append({"ev": key[0], "fid": key[1], "model": key[2], "inp": key[3], **w, "job": path.parent.name})
        log(f"  weights {key[0]} {key[1]} {key[2]} {key[3]} feat={feat_dim} {w.get('bands') or w.get('spike_share')}")
    # mean band share across probe folds for transformer LFP
    summary = {}
    for model in ("gru", "a1tf", "anat"):
        for inp in ("lfp", "spike_lfp"):
            recs = [r for r in rows if r["model"] == model and r["inp"] == inp and r.get("bands")]
            if not recs:
                continue
            bands = {}
            for b in BANDS:
                m, sem, n = mean_sem([r["bands"][b] for r in recs])
                bands[b] = {"mean": jf(m, 4), "sem": jf(sem, 4), "n": n}
            extra = {}
            if inp == "spike_lfp":
                m, sem, n = mean_sem([r.get("spike_share") for r in recs])
                extra["spike_share"] = {"mean": jf(m, 4), "sem": jf(sem, 4)}
                m, sem, n = mean_sem([r.get("lfp_share") for r in recs])
                extra["lfp_share"] = {"mean": jf(m, 4), "sem": jf(sem, 4)}
            summary[f"{model}_{inp}"] = {"bands": bands, **extra, "n": len(recs)}
    return {"rows": rows, "summary": summary, "missing": missing}


def load_eids(eids):
    want = set(str(e) for e in eids)
    trials = []
    for path in sorted(p for p in mv.SESS.glob("*.pkl") if not p.name.endswith(".tmp")):
        eid = path.stem
        if eid not in want and eid[:8] not in want:
            continue
        with path.open("rb") as fh:
            payload = pickle.load(fh)
        units = payload["units"]
        m = np.flatnonzero(mv.motor_mask(units).to_numpy())
        mouse = str(payload.get("mouse_id"))
        date = str(payload.get("session_date"))
        eid = str(payload["eid"])
        if m.size < mv.MIN_UNITS:
            continue
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
            n = min(spikes.shape[0], mv.MAX_BINS)
            if n < mv.MIN_BINS:
                continue
            if not mv.paw_is_wheel_paw(rec):
                continue
            behaviors = mv.trial_behaviors(rec)
            spikes = spikes[:n, m]
            packed = {}
            for name, beh in behaviors.items():
                arr = np.full(n, np.nan, dtype=np.float32)
                src = np.asarray(beh, dtype=np.float32).reshape(-1)
                mlen = min(n, src.size)
                if mlen:
                    arr[:mlen] = src[:mlen]
                packed[name] = arr
            n_keep = mv._trailing_finite_len(packed.get("paw_speed"), n)
            if n_keep < mv.MIN_BINS:
                continue
            if n_keep < n:
                n = n_keep
                spikes = spikes[:n]
                packed = {name: np.asarray(arr)[:n] for name, arr in packed.items()}
            if np.isfinite(spikes).all(axis=1).sum() < mv.MIN_BINS:
                continue
            trials.append(
                {
                    "eid": eid,
                    "mouse_id": mouse,
                    "session_date": date,
                    "trial_index": rec.get("trial_index"),
                    "used_paw": rec.get("used_paw"),
                    "wheel_paws": rec.get("wheel_paws"),
                    "spikes": np.log1p(np.clip(spikes, 0, None)).astype(np.float32),
                    "behaviors": packed,
                    "xyz": xyz,
                    "areas": areas,
                }
            )
    log(f"loaded {len(trials)} trials from {len(want)} eids")
    return trials


def explain_forward(model, batch, dev):
    x = batch["x"].to(dev)
    xyz = batch["xyz"].to(dev)
    area = batch["area"].to(dev)
    rate = batch["rate"].to(dev)
    mask_n = batch["mask_n"].to(dev)
    mask_t = batch["mask_t"].to(dev)
    lengths = batch["lengths"]
    e = model.enc(x, xyz, area, rate, mask_n)
    logits = model.pool.q(torch.tanh(model.pool.W(e))).squeeze(-1)
    logits = logits.masked_fill(~mask_n[:, :, None], -1e9)
    alpha = torch.softmax(logits, dim=1)
    h = (alpha.unsqueeze(-1) * e).sum(dim=1) * mask_t.unsqueeze(-1)
    temp = None
    if model.temporal == "transformer":
        out, temp = model.rnn(h, mask_t, return_attn=True)
    elif model.use_pack:
        packed = torch.nn.utils.rnn.pack_padded_sequence(
            h, lengths.cpu(), batch_first=True, enforce_sorted=False
        )
        out, _ = model.rnn(packed)
        T = x.shape[2]
        out, _ = torch.nn.utils.rnn.pad_packed_sequence(out, batch_first=True, total_length=T)
    else:
        out = model.rnn(h, mask_t)
    pred = model.head(out)
    if isinstance(pred, tuple):
        pred = pred[0]
    return pred, alpha, temp, x


def entropy_rows(p, axis, eps=1e-8):
    q = np.clip(p, eps, 1.0)
    h = -(q * np.log(q)).sum(axis=axis)
    n = p.shape[axis]
    return h / math.log(max(n, 2))


def lag_profile(attn, mask_t, max_lag=MAX_LAG):
    # attn: n_layers, B, H, T, T  (query, key)
    a = attn.mean(axis=(0, 2))  # B, T, T
    B, T, _ = a.shape
    lags = np.zeros(max_lag + 1, dtype=np.float64)
    cnt = np.zeros(max_lag + 1, dtype=np.float64)
    diag = []
    ent = []
    pre_mass = []
    for b in range(B):
        mt = mask_t[b]
        tb = int(mt.sum())
        if tb < 4:
            continue
        w = a[b, :tb, :tb]
        row_ent = entropy_rows(w, axis=1)
        ent.append(float(np.mean(row_ent)))
        diag.append(float(np.mean(np.diag(w))))
        for d in range(max_lag + 1):
            if tb - d < 1:
                break
            lags[d] += float(np.mean(np.diag(w, k=-d)))
            cnt[d] += 1.0
    lags = np.divide(lags, np.maximum(cnt, 1.0))
    return {
        "lag": [jf(v, 4) for v in lags.tolist()],
        "diag_mass": jf(float(np.mean(diag)) if diag else None, 4),
        "row_entropy": jf(float(np.mean(ent)) if ent else None, 4),
        "n": len(diag),
    }


def causal_half_life(lag):
    a = np.asarray([v if v is not None else 0.0 for v in lag], dtype=np.float64)
    s = a.sum()
    if s <= 0:
        return None
    c = np.cumsum(a) / s
    hit = np.flatnonzero(c >= 0.5)
    if hit.size < 1:
        return int(len(a) - 1)
    return int(hit[0])


def token_stats(alpha, mask_n, mask_t, x, input_name):
    # alpha B,N,T; x B,N,T,F
    B, N, T = alpha.shape
    F = x.shape[-1] if x.ndim == 4 else 1
    peak = []
    ent = []
    ntok = []
    band_lift = {b: [] for b in BANDS}
    spike_lift = []
    for b in range(B):
        n_obs = int(mask_n[b].sum())
        tb = int(mask_t[b].sum())
        if n_obs < 2 or tb < 4:
            continue
        al = alpha[b, :n_obs, :tb]
        peak.append(float(al.max(axis=0).mean()))
        ent.append(float(entropy_rows(al, axis=0).mean()))
        ntok.append(n_obs)
        if x.ndim != 4:
            continue
        xb = x[b, :n_obs, :tb]
        # attended mean feat vs uniform mean feat
        uni = xb.mean(axis=0)
        att = (al[:, :, None] * xb).sum(axis=0) / (al.sum(axis=0)[:, None] + 1e-8)
        if input_name == "lfp" and F >= 6:
            for i, name in enumerate(BANDS):
                d = float((att[:, i] - uni[:, i]).mean())
                band_lift[name].append(d)
        elif input_name == "spike_lfp" and F >= 7:
            spike_lift.append(float((att[:, 0] - uni[:, 0]).mean()))
            for i, name in enumerate(BANDS):
                d = float((att[:, i + 1] - uni[:, i + 1]).mean())
                band_lift[name].append(d)
    out = {
        "peak_alpha": jf(float(np.mean(peak)) if peak else None, 4),
        "token_entropy": jf(float(np.mean(ent)) if ent else None, 4),
        "n_tokens": jf(float(np.mean(ntok)) if ntok else None, 1),
    }
    if any(band_lift[b] for b in BANDS):
        out["band_lift"] = {b: jf(float(np.mean(band_lift[b])), 5) for b in BANDS if band_lift[b]}
    if spike_lift:
        out["spike_lift"] = jf(float(np.mean(spike_lift)), 5)
    return out


def concat_r2(y, p, mask):
    yt, pt = [], []
    for yi, pi, m in zip(y, p, mask):
        mm = np.asarray(m, dtype=bool)
        yt.append(np.asarray(yi)[mm].reshape(-1))
        pt.append(np.asarray(pi)[mm].reshape(-1))
    yt = np.concatenate(yt) if yt else np.zeros(0)
    pt = np.concatenate(pt) if pt else np.zeros(0)
    ok = np.isfinite(yt) & np.isfinite(pt)
    if ok.sum() < 10:
        return None
    from sklearn.metrics import r2_score

    return float(r2_score(yt[ok], pt[ok]))


def concat_r2_lists(ys, ps):
    yt = np.concatenate([np.asarray(y).reshape(-1) for y in ys]) if ys else np.zeros(0)
    pt = np.concatenate([np.asarray(p).reshape(-1) for p in ps]) if ps else np.zeros(0)
    n = min(yt.size, pt.size)
    yt, pt = yt[:n], pt[:n]
    ok = np.isfinite(yt) & np.isfinite(pt)
    if ok.sum() < 10:
        return None
    from sklearn.metrics import r2_score

    return float(r2_score(yt[ok], pt[ok]))


def average_token_stats(parts):
    if not parts:
        return {}
    out = {"n": len(parts)}
    for key in ("peak_alpha", "token_entropy", "n_tokens", "spike_lift"):
        vals = [p.get(key) for p in parts if p.get(key) is not None]
        m, sem, n = mean_sem(vals)
        if n:
            out[key] = jf(m, 4 if key != "n_tokens" else 1)
    lifts = defaultdict(list)
    for p in parts:
        for b, v in (p.get("band_lift") or {}).items():
            if v is not None:
                lifts[b].append(v)
    if lifts:
        out["band_lift"] = {b: jf(float(np.mean(v)), 5) for b, v in lifts.items()}
    return out


def average_lag_profiles(parts):
    if not parts:
        return {}
    lags = [p.get("lag") for p in parts if p.get("lag")]
    lag_mean = None
    if lags:
        L = min(len(x) for x in lags)
        arr = np.array([[(v if v is not None else 0.0) for v in x[:L]] for x in lags], dtype=np.float64)
        lag_mean = [jf(float(v), 4) for v in arr.mean(axis=0)]
    out = {"lag": lag_mean, "n": len(parts)}
    for key in ("diag_mass", "row_entropy"):
        vals = [p.get(key) for p in parts if p.get(key) is not None]
        m, sem, n = mean_sem(vals)
        if n:
            out[key] = jf(m, 4)
    if lag_mean:
        out["half_life_bins"] = causal_half_life(lag_mean)
        out["half_life_s"] = jf((out["half_life_bins"] or 0) * DT, 3)
    return out


def run_occlusion_lists(net, ds, st, dev, cols, y_list):
    from torch.utils.data import DataLoader

    collate = ec.collate_feat if st["feat_dim"] > 1 else mv.collate
    ld = DataLoader(ds, batch_size=ATTN_BATCH, shuffle=False, collate_fn=collate)
    preds = []
    with torch.no_grad():
        for batch in ld:
            x = batch["x"].clone()
            if x.dim() == 4:
                x[..., cols] = 0
            else:
                continue
            batch = dict(batch)
            batch["x"] = x
            pred, _, _, _ = explain_forward(net, batch, dev)
            p_np = pred.cpu().numpy()
            mt = batch["mask_t"].numpy()
            for j in range(int(p_np.shape[0])):
                tb = int(mt[j].sum())
                pj = p_np[j, :tb]
                if pj.ndim > 1:
                    pj = pj[..., 0]
                preds.append(pj)
    return concat_r2_lists(y_list, preds)


def phase_attn(jobs):
    ev, fid, model = ATTN_PROBE
    result = {"probe": {"eval": ev, "fold_id": fid, "model": model}, "inputs": {}, "occlusion": {}, "notes": []}
    hold = None
    for inp in INPUTS:
        pack = jobs.get((ev, fid, model, inp))
        if pack is None:
            result["notes"].append(f"missing job {inp}")
            continue
        path, meta = pack
        ckpt = path.parent / "model.pt"
        if not ckpt.exists():
            result["notes"].append(f"no checkpoint {path.parent.name}")
            continue
        if hold is None:
            df = pd.read_csv(path)
            hold = df
            test_eid = str(df["eid"].iloc[0])
    if hold is None:
        result["notes"].append("no holdout table")
        return result
    test_eid = str(hold["eid"].iloc[0])
    hold_keys = set()
    for rec in hold.to_dict("records"):
        k = pair_key(rec.get("eid"), rec.get("trial_index"))
        if k:
            hold_keys.add(k)

    # scaler sessions: a few other LFP eids
    other = []
    for p in sorted(ec.LFP_DIR.glob("*.pkl")):
        if p.stem == test_eid or p.stem[:8] == test_eid[:8]:
            continue
        other.append(p.stem)
        if len(other) >= 6:
            break
    log(f"attn probe session {test_eid[:8]} + {len(other)} scaler sessions")
    trials = load_eids([test_eid] + other)
    if not trials:
        result["notes"].append("failed to load trials")
        return result
    ec.attach_lfp(trials)
    ec.annotate_trials(trials)
    ec.bind_objective(trials, "trace")
    test_idx = np.array(
        [i for i, t in enumerate(trials) if str(t["eid"]) == test_eid and pair_key(t["eid"], t["trial_index"]) in hold_keys],
        dtype=int,
    )
    train_idx = np.array([i for i, t in enumerate(trials) if str(t["eid"]) != test_eid], dtype=int)
    test_idx = np.array([i for i in test_idx if ec.trial_has_lfp(trials[int(i)])], dtype=int)
    if test_idx.size > MAX_ATTN_TRIALS:
        rng = np.random.default_rng(0)
        test_idx = np.sort(rng.choice(test_idx, size=MAX_ATTN_TRIALS, replace=False))
    log(f"attn test trials={test_idx.size} train_scaler={train_idx.size}")
    if test_idx.size < 8:
        result["notes"].append(f"too few test trials {test_idx.size}")
        return result
    lfp_mean, lfp_std = ec.fit_lfp_scaler(trials, train_idx if train_idx.size else test_idx)
    result["n_test"] = int(test_idx.size)
    result["test_eid"] = test_eid[:8]
    result["scaler_note"] = (
        "LFP z-score fit on 6 other sessions (LSO train is all other sessions; this is a stand-in). "
        "Band ranking and attention geometry are the target, not a reprint of fold concat R²."
    )

    winner_map = {}
    # matched winners from holdout csvs
    tables = {}
    for inp in INPUTS:
        pack = jobs.get((ev, fid, model, inp))
        if pack is None:
            continue
        df = pd.read_csv(pack[0])
        tables[inp] = {}
        for rec in df.to_dict("records"):
            k = pair_key(rec.get("eid"), rec.get("trial_index"))
            if k:
                try:
                    tables[inp][k] = float(rec.get("r2"))
                except (TypeError, ValueError):
                    pass
    if all(inp in tables for inp in INPUTS):
        for k in set(tables["spikes"]) & set(tables["lfp"]) & set(tables["spike_lfp"]):
            winner_map[k] = winner_of(tables["spikes"][k], tables["lfp"][k], tables["spike_lfp"][k])

    dev = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    log(f"attn device {dev}")

    by_input = {}
    occ_rows = []
    for inp in INPUTS:
        pack = jobs.get((ev, fid, model, inp))
        if pack is None:
            continue
        path, meta = pack
        ckpt = path.parent / "model.pt"
        if not ckpt.exists():
            continue
        log(f"  forward {inp} {path.parent.name}")
        blob = torch.load(ckpt, map_location="cpu", weights_only=False)
        ec.bind_input(trials, inp, lfp_mean, lfp_std)
        st = ec.merge_eval_stats(blob.get("stats"), None, trials, test_idx)
        st["feat_dim"] = int(blob.get("feat_dim") or ec.feat_dim_of(inp))
        st["input"] = inp
        name = blob.get("model") or LONG_MODEL[model]
        net = ec.build_neural(name, st, trials, test_idx, blob.get("objective") or "trace", input_name=inp)
        net.load_state_dict(blob["state_dict"], strict=False)
        net.to(dev)
        net.eval()
        ds = mv.TrialDS(trials, test_idx, blob["y_mean"], blob["y_std"], st)
        ld = torch.utils.data.DataLoader(
            ds, batch_size=ATTN_BATCH, shuffle=False, collate_fn=ec.collate_feat if st["feat_dim"] > 1 else mv.collate
        )
        tok_parts, lag_parts = [], []
        y_list, p_list = [], []
        winner_tok = defaultdict(list)
        shape_tok = defaultdict(list)
        winner_lag = defaultdict(list)
        shape_lag = defaultdict(list)
        epoch_tok = {"pre_onset": [], "post_onset": []}
        n_trials = 0
        with torch.no_grad():
            cursor = 0
            for batch in ld:
                pred, alpha, temp, x = explain_forward(net, batch, dev)
                a_np = alpha.cpu().numpy()
                x_np = x.cpu().numpy() if x.dim() == 4 else x.cpu().numpy()[..., None]
                mn = batch["mask_n"].numpy()
                mt = batch["mask_t"].numpy()
                y_np = batch["y"].numpy()
                p_np = pred.cpu().numpy()
                temp_np = temp.cpu().numpy() if temp is not None else None
                bs = int(batch["x"].shape[0])
                for j in range(bs):
                    ti = int(test_idx[cursor + j])
                    t = trials[ti]
                    k = pair_key(t["eid"], t["trial_index"])
                    on = t.get("onset_bin")
                    wlab = winner_map.get(k)
                    yraw = np.asarray(t.get("y"), dtype=np.float32).reshape(-1)
                    slab = shape_bin(yraw)
                    n_obs = int(mn[j].sum())
                    tb = int(mt[j].sum())
                    if n_obs < 2 or tb < 4:
                        continue
                    al = a_np[j : j + 1, :n_obs, :tb]
                    xb = x_np[j : j + 1, :n_obs, :tb]
                    mnj = np.ones((1, n_obs), dtype=bool)
                    mtj = np.ones((1, tb), dtype=bool)
                    ts = token_stats(al, mnj, mtj, xb, inp)
                    tok_parts.append(ts)
                    n_trials += 1
                    if wlab:
                        winner_tok[str(wlab)].append(ts)
                    if slab:
                        shape_tok[str(slab)].append(ts)
                    if temp_np is not None:
                        tw = temp_np[:, j : j + 1, :, :tb, :tb]
                        lp = lag_profile(tw, mtj)
                        lag_parts.append(lp)
                        if wlab:
                            winner_lag[str(wlab)].append(lp)
                        if slab:
                            shape_lag[str(slab)].append(lp)
                    yj = y_np[j, :tb]
                    pj = p_np[j, :tb]
                    if yj.ndim > 1:
                        yj = yj[..., 0]
                    if pj.ndim > 1:
                        pj = pj[..., 0]
                    y_list.append(yj)
                    p_list.append(pj)
                    if isinstance(on, (int, float, np.integer, np.floating)) and np.isfinite(on):
                        on_i = int(max(1, min(int(on), tb - 1)))
                        pre = token_stats(al[:, :, :on_i], mnj, np.ones((1, on_i), dtype=bool), xb[:, :, :on_i], inp)
                        post = token_stats(al[:, :, on_i:], mnj, np.ones((1, tb - on_i), dtype=bool), xb[:, :, on_i:], inp)
                        epoch_tok["pre_onset"].append(pre)
                        epoch_tok["post_onset"].append(post)
                cursor += bs
        rec = {"token": average_token_stats(tok_parts), "n": n_trials}
        if lag_parts:
            rec["temporal"] = average_lag_profiles(lag_parts)
            rec["temporal"]["half_life_bins"] = causal_half_life(rec["temporal"]["lag"])
            rec["temporal"]["half_life_s"] = jf((rec["temporal"]["half_life_bins"] or 0) * DT, 3)
            rec["temporal_by_winner"] = {k: average_lag_profiles(v) for k, v in winner_lag.items() if len(v) >= 6}
            rec["temporal_by_shape"] = {k: average_lag_profiles(v) for k, v in shape_lag.items() if len(v) >= 6}
        rec["token_by_winner"] = {k: average_token_stats(v) for k, v in winner_tok.items() if len(v) >= 6}
        rec["token_by_shape"] = {k: average_token_stats(v) for k, v in shape_tok.items() if len(v) >= 6}
        rec["token_by_epoch"] = {k: average_token_stats(v) for k, v in epoch_tok.items() if v}
        by_input[inp] = rec

        if inp in ("lfp", "spike_lfp") and y_list:
            base_r2 = concat_r2_lists(y_list, p_list)
            occ = {"base_r2": jf(base_r2, 4)}
            ablations = []
            if inp == "lfp":
                for i, name in enumerate(BANDS):
                    ablations.append((name, [i]))
            else:
                ablations.append(("spike", [0]))
                ablations.append(("all_lfp", list(range(1, 7))))
                for i, name in enumerate(BANDS):
                    ablations.append((name, [1 + i]))
            for name, cols in ablations:
                drop = run_occlusion_lists(net, ds, st, dev, cols, y_list)
                occ[name] = {
                    "r2": jf(drop, 4),
                    "delta": jf((base_r2 - drop) if drop is not None and base_r2 is not None else None, 4),
                }
                log(f"    occlude {inp} {name}: R² {drop} Δ {occ[name]['delta']}")
            occ_rows.append({"inp": inp, **occ})
        del net
        if hasattr(torch, "mps") and torch.backends.mps.is_available():
            torch.mps.empty_cache()

    result["inputs"] = by_input
    result["occlusion"] = occ_rows
    return result


def run_occlusion(net, ds, st, dev, cols, y_all, mt):
    from torch.utils.data import DataLoader

    collate = ec.collate_feat if st["feat_dim"] > 1 else mv.collate
    ld = DataLoader(ds, batch_size=ATTN_BATCH, shuffle=False, collate_fn=collate)
    preds = []
    with torch.no_grad():
        for batch in ld:
            x = batch["x"].clone()
            if x.dim() == 4:
                x[..., cols] = 0
            else:
                continue
            batch = dict(batch)
            batch["x"] = x
            pred, _, _, _ = explain_forward(net, batch, dev)
            preds.append(pred.cpu().numpy())
    if not preds:
        return None
    p = np.concatenate(preds, 0)
    if p.ndim == 3:
        p = p[..., 0]
    return concat_r2(y_all, p, mt)


def temporal_by_onset(attn, mask_t, onsets):
    a = attn.mean(axis=(0, 2))
    buckets = {"pre_onset": [], "post_onset": []}
    B = a.shape[0]
    for b in range(B):
        tb = int(mask_t[b].sum())
        on = onsets[b]
        if tb < 6 or on is None or not (isinstance(on, (int, float, np.integer, np.floating)) and np.isfinite(on)):
            continue
        on = int(on)
        w = a[b, :tb, :tb]
        pre = np.arange(tb) < on
        post = ~pre
        if pre.sum() >= 3:
            buckets["pre_onset"].append(_lag0_long(w[pre][:, :tb], pre_keys=None))
        if post.sum() >= 3:
            buckets["post_onset"].append(_lag0_long(w[post][:, :tb], pre_keys=pre))
    out = {}
    for k, recs in buckets.items():
        if not recs:
            continue
        out[k] = {
            "diag": jf(float(np.mean([r["diag"] for r in recs])), 4),
            "lag_mean_bins": jf(float(np.mean([r["lag_mean"] for r in recs])), 3),
            "pre_key_mass": jf(float(np.nanmean([r["pre_mass"] for r in recs])), 4),
            "n": len(recs),
        }
    return out


def _lag0_long(w_rows, pre_keys=None):
    # w_rows: n_q, T
    diag = []
    lag_mean = []
    pre_mass = []
    for i, row in enumerate(w_rows):
        t = int(np.argmax(row * 0 + 1) if False else min(i, row.size - 1))
        # rows are already the query slice; approximate query index unknown.
        # Use mass-weighted lag via last-index assumption if square... skip exact diag.
        s = float(row.sum()) + 1e-8
        ks = np.arange(row.size)
        lag_mean.append(float(((row.size - 1 - ks) * row).sum() / s))  # not quite
        if pre_keys is not None and pre_keys.size == row.size:
            pre_mass.append(float(row[pre_keys].sum()))
        else:
            pre_mass.append(float("nan"))
        # peakiness of the row
        diag.append(float(row.max()))
    # Better lag: for a square matrix passed as w[queries], keys 0..T-1
    return {
        "diag": float(np.mean(diag)),
        "lag_mean": float(np.mean(lag_mean)),
        "pre_mass": float(np.nanmean(pre_mass)) if pre_mass else float("nan"),
    }


def temporal_by_group(attn, mask_t, labels):
    a = attn.mean(axis=(0, 2))
    buckets = defaultdict(list)
    for b, lab in enumerate(labels):
        if not lab:
            continue
        tb = int(mask_t[b].sum())
        if tb < 4:
            continue
        w = a[b, :tb, :tb]
        lags = [float(np.mean(np.diag(w, k=-d))) for d in range(min(MAX_LAG, tb) + 1)]
        buckets[str(lab)].append(
            {
                "diag": float(np.mean(np.diag(w))),
                "half": causal_half_life(lags),
                "ent": float(np.mean(entropy_rows(w, axis=1))),
            }
        )
    out = {}
    for k, recs in buckets.items():
        if len(recs) < 6:
            continue
        out[k] = {
            "n": len(recs),
            "diag": jf(float(np.mean([r["diag"] for r in recs])), 4),
            "half_life_bins": jf(float(np.mean([r["half"] for r in recs if r["half"] is not None])), 2),
            "row_entropy": jf(float(np.mean([r["ent"] for r in recs])), 4),
        }
    return out


def token_by_group(alpha, mask_n, mask_t, x, input_name, labels):
    buckets = defaultdict(lambda: {"peak": [], "ent": [], "lift": defaultdict(list)})
    B = alpha.shape[0]
    for b, lab in enumerate(labels):
        if not lab:
            continue
        n_obs = int(mask_n[b].sum())
        tb = int(mask_t[b].sum())
        if n_obs < 2 or tb < 4:
            continue
        al = alpha[b, :n_obs, :tb]
        buckets[str(lab)]["peak"].append(float(al.max(axis=0).mean()))
        buckets[str(lab)]["ent"].append(float(entropy_rows(al, axis=0).mean()))
        if x.ndim == 4:
            xb = x[b, :n_obs, :tb]
            uni = xb.mean(axis=0)
            att = (al[:, :, None] * xb).sum(axis=0) / (al.sum(axis=0)[:, None] + 1e-8)
            if input_name == "lfp" and xb.shape[-1] >= 6:
                for i, name in enumerate(BANDS):
                    buckets[str(lab)]["lift"][name].append(float((att[:, i] - uni[:, i]).mean()))
            if input_name == "spike_lfp" and xb.shape[-1] >= 7:
                buckets[str(lab)]["lift"]["spike"].append(float((att[:, 0] - uni[:, 0]).mean()))
                for i, name in enumerate(BANDS):
                    buckets[str(lab)]["lift"][name].append(float((att[:, i + 1] - uni[:, i + 1]).mean()))
    out = {}
    for k, rec in buckets.items():
        if len(rec["peak"]) < 6:
            continue
        item = {
            "n": len(rec["peak"]),
            "peak_alpha": jf(float(np.mean(rec["peak"])), 4),
            "token_entropy": jf(float(np.mean(rec["ent"])), 4),
        }
        if rec["lift"]:
            item["band_lift"] = {b: jf(float(np.mean(v)), 5) for b, v in rec["lift"].items() if v}
        out[k] = item
    return out


def token_by_onset(alpha, mask_n, mask_t, x, input_name, onsets):
    # split each trial's time into pre/post and pool token stats
    fake_labels = []
    # We'll compute two synthetic batches by masking time — simpler aggregate:
    pre_peak, post_peak = [], []
    pre_lift = defaultdict(list)
    post_lift = defaultdict(list)
    B = alpha.shape[0]
    for b, on in enumerate(onsets):
        n_obs = int(mask_n[b].sum())
        tb = int(mask_t[b].sum())
        if n_obs < 2 or tb < 6 or on is None or not np.isfinite(on):
            continue
        on = int(max(1, min(int(on), tb - 1)))
        al = alpha[b, :n_obs, :tb]
        xb = x[b, :n_obs, :tb] if x.ndim == 4 else None
        for name, sl in (("pre", slice(0, on)), ("post", slice(on, tb))):
            sub = al[:, sl]
            if sub.shape[1] < 3:
                continue
            pk = float(sub.max(axis=0).mean())
            (pre_peak if name == "pre" else post_peak).append(pk)
            if xb is None:
                continue
            xs = xb[:, sl]
            uni = xs.mean(axis=0)
            att = (sub[:, :, None] * xs).sum(axis=0) / (sub.sum(axis=0)[:, None] + 1e-8)
            dest = pre_lift if name == "pre" else post_lift
            if input_name == "lfp" and xs.shape[-1] >= 6:
                for i, bn in enumerate(BANDS):
                    dest[bn].append(float((att[:, i] - uni[:, i]).mean()))
            if input_name == "spike_lfp" and xs.shape[-1] >= 7:
                dest["spike"].append(float((att[:, 0] - uni[:, 0]).mean()))
                for i, bn in enumerate(BANDS):
                    dest[bn].append(float((att[:, i + 1] - uni[:, i + 1]).mean()))
    out = {}
    if pre_peak:
        out["pre_onset"] = {
            "peak_alpha": jf(float(np.mean(pre_peak)), 4),
            "band_lift": {b: jf(float(np.mean(v)), 5) for b, v in pre_lift.items() if v},
            "n": len(pre_peak),
        }
    if post_peak:
        out["post_onset"] = {
            "peak_alpha": jf(float(np.mean(post_peak)), 4),
            "band_lift": {b: jf(float(np.mean(v)), 5) for b, v in post_lift.items() if v},
            "n": len(post_peak),
        }
    return out


def merge_json(old, new):
    if not old:
        return new
    out = dict(old)
    out.update(new)
    return out


def main():
    log("analyze_lfp_vs_spikes start")
    ap = argparse.ArgumentParser()
    ap.add_argument("--phase", default="all", choices=("all", "perf", "weights", "attn"))
    args = ap.parse_args()
    prev = json.loads(OUT_JSON.read_text()) if OUT_JSON.exists() else {}
    jobs = None
    blob = {"arch": {
        "unit_pool": "AttnPool over tokens at each 20 ms bin (units for spikes / spike_lfp; motor channels for LFP).",
        "temporal": "After pooling, A1/Anat Transformer is causal self-attention over time (n_layers=1, nhead=4, d=64). GRU has no temporal attention weights.",
        "bands": "Bands are feature dimensions inside UnitEncoder, not tokens. Band focus = first-layer |W| share + occlusion ΔR² + attended-token band lift.",
        "feat": {
            "spikes": "feat_dim=1 log1p spike count per unit",
            "lfp": "feat_dim=6 log10 band power per motor channel (δ θ α β low-γ high-γ)",
            "spike_lfp": "feat_dim=7 unit rate concatenated with that electrode's 6 bands",
        },
        "note": "DualSetSequence.forward currently drops return_attn; this probe calls encoder / AttnPool / CausalTransformer directly.",
    }}
    if args.phase in ("all", "perf"):
        log("=== phase perf ===")
        blob["perf"] = phase_perf()
    if args.phase in ("all", "weights"):
        log("=== phase weights ===")
        _need_torch()
        jobs = jobs or index_probe_jobs()
        blob["weights"] = phase_weights(jobs)
    if args.phase in ("all", "attn"):
        log("=== phase attn ===")
        _need_torch()
        jobs = jobs or index_probe_jobs()
        try:
            blob["attn"] = phase_attn(jobs)
        except Exception:
            blob["attn"] = {"error": traceback.format_exc()}
            log(blob["attn"]["error"])
    out = merge_json(prev, blob) if args.phase != "all" else {**prev, **blob}
    # if phase is all, blob already complete; keep arch always
    if args.phase == "perf":
        out = merge_json(prev, {"perf": blob["perf"], "arch": blob["arch"]})
    elif args.phase == "weights":
        out = merge_json(prev, {"weights": blob["weights"], "arch": blob.get("arch") or prev.get("arch")})
    elif args.phase == "attn":
        out = merge_json(prev, {"attn": blob.get("attn"), "arch": blob.get("arch") or prev.get("arch")})
    else:
        out = blob
        if "perf" not in out and "perf" in prev:
            out["perf"] = prev["perf"]
    OUT_JSON.write_text(json.dumps(out, indent=2))
    log(f"wrote {OUT_JSON} bytes={OUT_JSON.stat().st_size}")


if __name__ == "__main__":
    main()
