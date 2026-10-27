#!/usr/bin/env python3
"""Rebuild ExperimentsClean metrics canvases + trace canvases/HTML from disk.

Never opens model.pt.
"""
from __future__ import annotations

import csv
import json
import math
import re
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

ROOT = Path("/Users/Stefan/Desktop/BrainMap/BehaviorRes")
JOBS = ROOT / "ExperimentsClean" / "paw_speed" / "jobs"
ATLAS = ROOT / "Modelv1Clusters" / "atlas_trial_clusters.json"
CANVAS_DIR = Path("/Users/Stefan/.cursor/projects/Users-Stefan-Desktop/canvases")
PROGRESS = CANVAS_DIR / "experiments-clean-progress.canvas.tsx"
TRACES = CANVAS_DIR / "experiments-clean-traces.canvas.tsx"
OVERLAY = CANVAS_DIR / "experiments-clean-traces-overlay.canvas.tsx"
HTML_TRACES = ROOT / "experiments_clean_traces.html"
HTML_WHEEL = ROOT / "experiments_clean_traces_wheel.html"
HTML_CLUSTERS = ROOT / "experiments_clean_traces_clusters.html"
SUMMARY = ROOT / "_rebuild_summary.json"

MODEL_MAP = {
    "mean": "mean",
    "ar": "ar",
    "attn_a1_gru": "gru",
    "attn_a1_transformer": "a1tf",
    "attn_anat_transformer": "anat",
    "aniso_cnn_transformer": "cnn",
    "hierarchical": "hier",
    "gnn_inner": "gin",
    "gnn_outer": "gout",
}
NEURAL_KEYS = [
    "gru", "gru_h", "a1tf", "a1tf_h", "anat", "anat_h", "cnn", "cnn_h",
    "hier", "hier_h", "gin", "gin_h", "gout", "gout_h",
]
ALL_KEYS = ["mean", "ar"] + NEURAL_KEYS
PREF = {"LTO": "lto05", "LSO": "cef05f87", "LMO": "PL033"}
FALLBACK = {"LTO": "lto00", "LSO": "03063955", "LMO": "CSHL049"}
NOTES = {
    ("LTO", "lto00"): "block 0: first 5% of each session",
    ("LTO", "lto05"): "block 5: 25–30% of each session",
    ("LSO", "03063955"): "session 03063955 of SWC_038",
    ("LSO", "cef05f87"): "session cef05f87 of PL033",
    ("LMO", "CSHL049"): "mouse CSHL049",
    ("LMO", "PL033"): "mouse PL033",
}
CANVAS_MAX_T = 48


def fnum(s):
    if s is None or s == "":
        return None
    try:
        x = float(s)
    except (TypeError, ValueError):
        return None
    return x if math.isfinite(x) else None


def rnd(x, nd):
    if not isinstance(x, (int, float, np.floating)):
        x = fnum(x)
    if x is None or not math.isfinite(float(x)):
        return None
    return round(float(x), nd)


def pct(vals, nd=3):
    x = np.asarray([v for v in vals if v is not None and math.isfinite(v)], dtype=float)
    if x.size == 0:
        return None
    qs = np.percentile(x, [10, 25, 50, 75, 90])
    out = {
        "p10": rnd(qs[0], nd),
        "p25": rnd(qs[1], nd),
        "p50": rnd(qs[2], nd),
        "p75": rnd(qs[3], nd),
        "p90": rnd(qs[4], nd),
        "mean": rnd(float(x.mean()), nd),
    }
    if nd == 3:
        out["frac_gt0"] = rnd(float(np.mean(x > 0)), 3)
    return out


def hist_r2(vals):
    x = np.asarray([v for v in vals if v is not None and math.isfinite(v)], dtype=float)
    counts = [0] * 7
    for v in x:
        if v < -0.5:
            counts[0] += 1
        elif v < -0.25:
            counts[1] += 1
        elif v < 0:
            counts[2] += 1
        elif v < 0.25:
            counts[3] += 1
        elif v < 0.5:
            counts[4] += 1
        elif v < 0.75:
            counts[5] += 1
        else:
            counts[6] += 1
    return counts


def hist_onset(vals):
    x = np.asarray([v for v in vals if v is not None and math.isfinite(v)], dtype=float)
    counts = [0] * 6
    for v in x:
        if v <= 0.04:
            counts[0] += 1
        elif v <= 0.08:
            counts[1] += 1
        elif v <= 0.12:
            counts[2] += 1
        elif v <= 0.20:
            counts[3] += 1
        elif v <= 0.40:
            counts[4] += 1
        else:
            counts[5] += 1
    return counts


def stats3(row, key, nd=3):
    return {
        "mean": rnd(row.get(f"fold_mean_{key}"), nd),
        "med": rnd(row.get(f"fold_median_{key}"), nd),
        "cat": rnd(row.get(f"fold_concat_{key}"), nd),
    }


def stats2(row, key, nd=3):
    return {
        "mean": rnd(row.get(f"fold_mean_{key}"), nd),
        "med": rnd(row.get(f"fold_median_{key}"), nd),
    }


def pred_key(model, loss):
    m = MODEL_MAP.get(model or "")
    if not m:
        return None
    if m in ("mean", "ar"):
        return m
    if loss == "huber_move":
        return m + "_h"
    return m


def collect_metrics():
    folds = []
    dist_rows = defaultdict(lambda: defaultdict(list))
    n_scored = 0
    n_npz = 0
    n_trace_npz = defaultdict(int)
    n_trace_missing = []
    n_trace = {}
    states = Counter()
    by_fold = Counter()
    for d in sorted(p for p in JOBS.iterdir() if p.is_dir()):
        sc, ht, npz, st = d / "scores.csv", d / "holdout_trials.csv", d / "holdout_curves.npz", d / "status.json"
        if st.exists():
            try:
                states[json.loads(st.read_text()).get("state")] += 1
            except Exception:
                states["bad_status"] += 1
        if npz.exists():
            n_npz += 1
        if not sc.exists():
            continue
        with sc.open(newline="") as f:
            recs = list(csv.DictReader(f))
        if not recs:
            continue
        row = recs[0]
        model = MODEL_MAP.get(row.get("model") or "")
        if not model:
            continue
        n_scored += 1
        ev = row.get("eval") or ""
        obj = row.get("objective") or ""
        loss = row.get("loss") or "none"
        fid = row.get("fold_id") or ""
        n_hold = int(float(row["n_holdout_trials"])) if fnum(row.get("n_holdout_trials")) is not None else None
        by_fold[(ev, fid)] += 1
        folds.append(
            {
                "e": ev,
                "fid": fid,
                "m": model,
                "o": obj,
                "l": loss,
                "n": n_hold,
                "nb": int(float(row["n_concat_bins"])) if fnum(row.get("n_concat_bins")) is not None else None,
                "acc": rnd(row.get("fold_direction_acc"), 3),
                "pearson": stats3(row, "pearson"),
                "r2": stats3(row, "r2"),
                "rmse": stats3(row, "rmse", 3),
                "mape": stats3(row, "mape"),
                "mae": stats3(row, "mae", 3),
                "onset_err": stats2(row, "onset_err_s"),
                "onset_abs": stats2(row, "onset_abs_s"),
                "peak_err": stats2(row, "peak_err_s"),
                "peak_abs": stats2(row, "peak_abs_s"),
                "peak_mag_err": stats2(row, "peak_mag_err", 3),
                "peak_mag_abs": stats2(row, "peak_mag_abs", 3),
            }
        )
        if obj == "trace":
            n_trace[(ev, fid)] = n_hold
            if npz.exists():
                n_trace_npz[(ev, fid)] += 1
            else:
                n_trace_missing.append(d.name)
        if not ht.exists():
            continue
        key = (ev, fid, obj, model, loss)
        with ht.open(newline="") as f:
            for rec in csv.DictReader(f):
                dist_rows[key]["r2"].append(fnum(rec.get("r2")))
                dist_rows[key]["pearson"].append(fnum(rec.get("pearson")))
                dist_rows[key]["mae"].append(fnum(rec.get("mae")))
                dist_rows[key]["rmse"].append(fnum(rec.get("rmse")))
                dist_rows[key]["mape"].append(fnum(rec.get("mape")))
                dist_rows[key]["onset_abs_s"].append(fnum(rec.get("onset_abs_s")))
                dist_rows[key]["peak_abs_s"].append(fnum(rec.get("peak_abs_s")))
                dist_rows[key]["peak_mag_abs"].append(fnum(rec.get("peak_mag_abs")))
                dt, dp = fnum(rec.get("dir_true")), fnum(rec.get("dir_pred"))
                if dt is not None and dp is not None:
                    dist_rows[key]["dir"].append(1.0 if (dt >= 0.5) == (dp >= 0.5) else 0.0)
    dist = {"trace": [], "onset": [], "mixed": [], "direction": []}
    for (ev, fid, obj, model, loss), bag in dist_rows.items():
        if obj not in dist:
            continue
        rec = {
            "e": ev,
            "fid": fid,
            "m": model,
            "l": loss,
            "n": len(bag["r2"]),
            "r2": pct(bag["r2"]),
            "pearson": pct(bag["pearson"]),
            "mae": pct(bag["mae"], 1),
            "rmse": pct(bag["rmse"], 1),
            "mape": pct(bag["mape"]),
            "onset_abs_s": pct(bag["onset_abs_s"]),
            "peak_abs_s": pct(bag["peak_abs_s"]),
            "peak_mag_abs": pct(bag["peak_mag_abs"], 1),
            "r2_hist": hist_r2(bag["r2"]),
            "onset_hist": hist_onset(bag["onset_abs_s"]),
        }
        if bag["dir"]:
            rec["dir_acc"] = rnd(float(np.mean(bag["dir"])), 3)
        dist[obj].append(rec)
    fold_ids = defaultdict(list)
    for ev, fid in sorted({(r["e"], r["fid"]) for r in folds}):
        if fid not in fold_ids[ev]:
            fold_ids[ev].append(fid)
    data = {
        "folds": folds,
        "dist": dist["trace"],
        "dist_onset": dist["onset"],
        "dist_mixed": dist["mixed"],
        "dist_dir": dist["direction"],
        "meta": {
            "n_jobs_scored": n_scored,
            "n_npz": n_npz,
            "n_missing_trace_npz": len(n_trace_missing),
            "n_trials_LTO_trace": n_trace.get(("LTO", PREF["LTO"])) or n_trace.get(("LTO", FALLBACK["LTO"])),
            "n_trials_LSO_trace": n_trace.get(("LSO", PREF["LSO"])) or n_trace.get(("LSO", FALLBACK["LSO"])),
            "n_trials_LMO_trace": n_trace.get(("LMO", PREF["LMO"])) or n_trace.get(("LMO", FALLBACK["LMO"])),
            "snapshot": "2026-10-24",
            "folds": dict(fold_ids),
            "npz_folds": {f"{e}:{fid}": n for (e, fid), n in sorted(n_trace_npz.items())},
        },
    }
    return data, {
        "states": dict(states),
        "by_fold": {f"{e}:{fid}": n for (e, fid), n in sorted(by_fold.items())},
        "n_trace": {f"{e}:{fid}": n for (e, fid), n in sorted(n_trace.items())},
        "missing_trace_npz": n_trace_missing,
        "n_trace_npz": {f"{e}:{fid}": n for (e, fid), n in sorted(n_trace_npz.items())},
    }


def splice_canvas(path: Path, blob: str):
    text = path.read_text()
    start = text.find("const DATA: Data = ")
    if start < 0:
        raise SystemExit(f"no DATA in {path}")
    end = text.find(";\nconst EVALS", start)
    if end < 0:
        end = text.find(";\n\nconst EVALS", start)
    if end < 0:
        raise SystemExit(f"no EVALS after DATA in {path}")
    path.write_text(text[: start + len("const DATA: Data = ")] + blob + text[end:])
    print("spliced canvas", path.name, "bytes", path.stat().st_size)


def downsample_idx(n, max_t):
    if n <= max_t:
        return np.arange(n)
    return np.unique(np.round(np.linspace(0, n - 1, max_t)).astype(int))


def quantize(arr, nd):
    if nd == 0:
        return [int(round(float(v))) if math.isfinite(float(v)) else 0 for v in arr]
    return [rnd(v, nd) or 0.0 for v in arr]


def load_atlas():
    blob = json.loads(ATLAS.read_text())
    by = {}
    paw_names = blob.get("paw_names") or []
    wheel_names = blob.get("wheel_names") or []
    for rec in blob["trials"]:
        tid = rec.get("trial_index")
        tid = None if tid is None else int(tid)
        by[(str(rec["eid"]), tid)] = rec
    return by, paw_names, wheel_names


def name_for(cid, names):
    try:
        c = int(cid)
    except (TypeError, ValueError):
        return "not_in_atlas"
    if c < 0:
        return "unclustered"
    if 0 <= c < len(names):
        return names[c]
    return f"c{c}"


def iter_trace_jobs():
    for d in sorted(p for p in JOBS.iterdir() if p.is_dir()):
        sc, ht = d / "scores.csv", d / "holdout_trials.csv"
        if not sc.exists() or not ht.exists():
            continue
        with sc.open(newline="") as f:
            recs = list(csv.DictReader(f))
        if not recs:
            continue
        row = recs[0]
        if row.get("objective") != "trace":
            continue
        model = row.get("model") or ""
        key = pred_key(model, row.get("loss") or "none")
        if not key:
            continue
        yield d, row, key, ht, d / "holdout_curves.npz"


def split_npz(path: Path):
    z = np.load(path, allow_pickle=False)
    y = np.asarray(z["y"], dtype=float).reshape(-1)
    p = np.asarray(z["pred"], dtype=float).reshape(-1)
    n_bins = np.asarray(z["n_bins"], dtype=int).reshape(-1)
    ids = np.asarray(z["trial_i"], dtype=int).reshape(-1)
    out = {}
    i = 0
    for n, tid in zip(n_bins, ids):
        n = int(n)
        out[int(tid)] = (y[i : i + n].copy(), p[i : i + n].copy())
        i += n
    return out, float(z["binsize"]) if "binsize" in z.files else 0.02


def build_pack(ev, fid, *, include_baselines, downsample, nd, want_curves, atlas):
    by_atlas, paw_names, wheel_names = atlas
    trials = {}
    keys_present = set()
    dt = 0.02
    n_jobs = 0
    has_curves = False
    for d, row, key, ht, npz in iter_trace_jobs():
        if (row.get("eval") or "") != ev or (row.get("fold_id") or "") != fid:
            continue
        if not include_baselines and key in ("mean", "ar"):
            continue
        n_jobs += 1
        with ht.open(newline="") as f:
            recs = list(csv.DictReader(f))
        curves = None
        if want_curves and npz.exists():
            curves, dt = split_npz(npz)
            has_curves = True
        for rec in recs:
            tid = int(float(rec["trial_i"]))
            tr = trials.get(tid)
            if tr is None:
                eid = str(rec.get("eid") or "")
                tindex = rec.get("trial_index")
                tindex = None if tindex in (None, "") else int(float(tindex))
                arec = by_atlas.get((eid, tindex))
                pc = -2 if arec is None else int(arec.get("paw_c", -2))
                wc = -2 if arec is None else int(arec.get("wheel_c", -2))
                if pc < 0:
                    pc = -1
                if wc < 0:
                    wc = -1
                cn = "not_in_atlas" if arec is None else name_for(pc, paw_names)
                wn = "not_in_atlas" if arec is None else name_for(wc, wheel_names)
                tr = {
                    "i": tid,
                    "e": eid[:8] if len(eid) >= 8 else eid,
                    "m": str(rec.get("mouse_id") or ""),
                    "n": int(float(rec["n_bins"])) if fnum(rec.get("n_bins")) is not None else 0,
                    "r": {},
                    "p": {},
                    "k": {},
                    "c": pc,
                    "cn": cn,
                    "wc": wc,
                    "wn": wn,
                }
                trials[tid] = tr
            r2 = rnd(rec.get("r2"), 3)
            if r2 is not None:
                tr["r"][key] = r2
            tr["k"][key] = {
                "pe": rnd(rec.get("pearson"), 3),
                "rm": rnd(rec.get("rmse"), 1),
                "ma": rnd(rec.get("mae"), 1),
                "mp": rnd(rec.get("mape"), 3),
                "on": rnd(rec.get("onset_abs_s"), 3),
                "pk": rnd(rec.get("peak_abs_s"), 3),
            }
            if curves is not None and tid in curves:
                y, pred = curves[tid]
                n = min(len(y), len(pred), tr["n"] or len(y))
                y, pred = y[:n], pred[:n]
                idx = downsample_idx(n, CANVAS_MAX_T) if downsample else np.arange(n)
                if "y" not in tr:
                    tr["y"] = quantize(y[idx], nd)
                    tr["n"] = int(n)
                tr["p"][key] = quantize(pred[idx], nd)
                keys_present.add(key)
    ordered = [trials[k] for k in sorted(trials)]
    if not include_baselines:
        for tr in ordered:
            tr["p"] = {k: v for k, v in tr["p"].items() if k not in ("mean", "ar")}
            tr["r"] = {k: v for k, v in tr["r"].items() if k not in ("mean", "ar")}
            tr["k"] = {k: v for k, v in tr["k"].items() if k not in ("mean", "ar")}
    keys = [k for k in (ALL_KEYS if include_baselines else NEURAL_KEYS) if k in keys_present or any(k in t["r"] for t in ordered)]
    n_max = max((t["n"] for t in ordered), default=0)
    pack = {
        "fid": fid,
        "note": NOTES.get((ev, fid), fid),
        "dt": dt,
        "keys": keys,
        "n": len(ordered),
        "n_max": int(n_max),
        "has_curves": has_curves,
        "trials": ordered,
        "n_jobs": n_jobs,
    }
    return pack


def strip_cluster_fields(pack, keep_k=False):
    out = dict(pack)
    trials = []
    for t in pack["trials"]:
        rec = {"i": t["i"], "e": t["e"], "m": t["m"], "n": t["n"], "y": t.get("y", []), "r": t["r"], "p": t.get("p", {})}
        if keep_k:
            rec["k"] = t.get("k", {})
            rec["c"] = t.get("c", -1)
            rec["cn"] = t.get("cn", "not_in_atlas")
            rec["wc"] = t.get("wc", -1)
            rec["wn"] = t.get("wn", "not_in_atlas")
            if "w" in t:
                rec["w"] = t["w"]
        trials.append(rec)
    out["trials"] = trials
    return out


def attach_wheel_from_html(packs):
    text = HTML_WHEEL.read_text()
    m = re.search(r'<script id="trace-json" type="application/json">(.*?)</script>', text, re.S)
    if not m:
        print("no wheel json")
        return
    old = json.loads(m.group(1))
    for ev, pack in packs.items():
        old_ev = (old.get("evals") or {}).get(ev) or {}
        old_fid = old_ev.get("fid")
        if old_fid != pack.get("fid"):
            continue
        by_i = {t["i"]: t.get("w") for t in old_ev.get("trials") or [] if t.get("w")}
        hit = 0
        for t in pack["trials"]:
            w = by_i.get(t["i"])
            if w:
                t["w"] = w
                hit += 1
        print("wheel copy", ev, pack["fid"], hit, "/", pack["n"])


def splice_html(path: Path, payload: dict):
    text = path.read_text()
    start = text.find('<script id="trace-json" type="application/json">')
    end = text.find("</script>", start)
    if start < 0 or end < 0:
        raise SystemExit(f"no trace-json in {path}")
    blob = json.dumps(payload, separators=(",", ":"))
    prefix = '<script id="trace-json" type="application/json">'
    path.write_text(text[: start + len(prefix)] + blob + text[end:])
    print("spliced html", path.name, "bytes", path.stat().st_size)


HTML_PACK_JS = r'''
function evalObj() {
  return DATA.evals[state.ev];
}
function foldMap() {
  const ev = evalObj();
  if (ev && ev.packs) return ev.packs;
  return null;
}
function availableFids() {
  const m = foldMap();
  if (m) return Object.keys(m);
  const p = evalObj();
  return p ? [p.fid] : [];
}
function resolveFid() {
  const ids = availableFids();
  if (!ids.length) return null;
  if (state.fid && ids.includes(state.fid)) return state.fid;
  const pref = { LTO: "lto05", LSO: "cef05f87", LMO: "PL033" }[state.ev];
  const m = foldMap();
  const hasCurves = (id) => {
    const pk = m ? m[id] : evalObj();
    return !!(pk && pk.has_curves);
  };
  if (pref && ids.includes(pref) && hasCurves(pref)) return pref;
  for (const id of ids) if (hasCurves(id)) return id;
  if (pref && ids.includes(pref)) return pref;
  return ids[0];
}
function pack() {
  const fid = resolveFid();
  state.fid = fid;
  const m = foldMap();
  if (m && fid && m[fid]) return m[fid];
  return evalObj();
}
'''


def patch_html_js(path: Path):
    text = path.read_text()
    # inject fold helpers and replace pack()
    if "function foldMap()" not in text:
        text = text.replace(
            "function pack() {\n  return DATA.evals[state.ev];\n}\n",
            HTML_PACK_JS,
        )
    if "fid: null" not in text and "const state = {" in text:
        text = text.replace(
            "const state = {\n  ev: EVALS.find((e) => DATA.evals[e]) || \"LTO\",",
            "const state = {\n  ev: EVALS.find((e) => DATA.evals[e]) || \"LTO\",\n  fid: null,",
        )
    # fold pills in setupPills
    needle = """  pillRow(
    document.getElementById("evalPills"),
    EVALS.filter((e) => DATA.evals[e]).map((e) => ({
      id: e,
      label: e + " · " + DATA.evals[e].fid + " · " + DATA.evals[e].n + " trials",
    })),
    state.ev,
    (id) => { state.ev = id; state.idx = 0; state.hover = null; render(); },
  );"""
    repl = """  pillRow(
    document.getElementById("evalPills"),
    EVALS.filter((e) => DATA.evals[e]).map((e) => ({
      id: e,
      label: e,
    })),
    state.ev,
    (id) => { state.ev = id; state.fid = null; state.idx = 0; state.cluster = "all"; state.hover = null; render(); },
  );
  const fmap = foldMap();
  const foldEl = document.getElementById("foldPills");
  if (foldEl) {
    pillRow(
      foldEl,
      availableFids().map((id) => {
        const pk = fmap ? fmap[id] : pack();
        const tag = pk && pk.has_curves ? "curves" : "metrics";
        return { id, label: id + " · " + (pk ? pk.n : 0) + " · " + tag };
      }),
      state.fid,
      (id) => { state.fid = id; state.idx = 0; state.cluster = "all"; state.hover = null; render(); },
    );
  }"""
    if needle in text:
        text = text.replace(needle, repl)
    # cluster HTML eval handler variant without cluster
    needle2 = needle.replace('state.idx = 0; state.hover = null;', 'state.idx = 0; state.cluster = "all"; state.hover = null;')
    if needle2 in text:
        text = text.replace(needle2, repl)
    if "if (!(key in (t.p || {}))) continue;" in text:
        text = text.replace(
            "if (!(key in (t.p || {}))) continue;",
            "if (typeof t.r[key] !== \"number\" && !(key in (t.p || {}))) continue;",
        )
    if "function renderCharts() {\n  const { tr } = trial();" in text and "No holdout_curves.npz" not in text:
        text = text.replace(
            "function renderCharts() {\n  const { tr } = trial();\n  const models = visibleModels(tr);\n  const host = document.getElementById(\"charts\");\n  host.innerHTML = \"\";",
            "function renderCharts() {\n  const { tr } = trial();\n  const host = document.getElementById(\"charts\");\n  host.innerHTML = \"\";\n  if (!tr || !tr.y || tr.y.length < 2) {\n    host.innerHTML = \"<p class='cap'>No holdout_curves.npz for this fold. Cluster metrics below still use holdout_trials.csv.</p>\";\n    return;\n  }\n  const models = visibleModels(tr);",
        )
    if "const { list, i, tr } = trial();\n  state.idx = i;" in text and "if (!tr) {" not in text:
        text = text.replace(
            "const { list, i, tr } = trial();\n  state.idx = i;",
            "const { list, i, tr } = trial();\n  if (!tr) {\n    document.getElementById(\"blurb\").textContent = \"No trials in this fold.\";\n    return;\n  }\n  state.idx = i;",
        )
    path.write_text(text)
    print("patched js", path.name)


def headline_rows(metrics):
    out = []
    for ev, fid in (("LTO", "lto05"), ("LSO", "cef05f87"), ("LMO", "PL033"),
                    ("LTO", "lto00"), ("LSO", "03063955"), ("LMO", "CSHL049")):
        block = {"eval": ev, "fid": fid, "trace": []}
        for row in metrics["folds"]:
            if row["e"] == ev and row["fid"] == fid and row["o"] == "trace":
                block["trace"].append(
                    {"m": row["m"], "l": row["l"], "n": row["n"],
                     "cat": row["r2"]["cat"], "med": row["r2"]["med"], "mean": row["r2"]["mean"]}
                )
        out.append(block)
    return out


def main():
    print("collect metrics")
    metrics, inv = collect_metrics()
    print("n_jobs", metrics["meta"]["n_jobs_scored"], "npz", metrics["meta"]["n_npz"],
          "missing_trace_npz", metrics["meta"]["n_missing_trace_npz"])
    print("by_fold", inv["by_fold"])
    print("npz_folds", inv["n_trace_npz"])
    splice_canvas(PROGRESS, json.dumps(metrics, separators=(",", ":")))

    print("load atlas")
    atlas = load_atlas()
    print("atlas trials", len(atlas[0]))

    evals = ["LTO", "LSO", "LMO"]
    html_packs = {}
    canvas_packs = {}
    cluster_evals = {}
    for ev in evals:
        fids = []
        for fid in (FALLBACK[ev], PREF[ev]):
            if fid not in fids:
                fids.append(fid)
        packs = {}
        for fid in fids:
            print("build", ev, fid)
            pack = build_pack(ev, fid, include_baselines=True, downsample=False, nd=2, want_curves=True, atlas=atlas)
            packs[fid] = pack
            print(" ", "n", pack["n"], "curves", pack["has_curves"], "keys", pack["keys"], "jobs", pack["n_jobs"])
        # canvas/html default: prefer pref if it has curves else fallback
        chosen = PREF[ev] if packs.get(PREF[ev], {}).get("has_curves") else FALLBACK[ev]
        if not packs.get(chosen, {}).get("has_curves"):
            for fid, pk in packs.items():
                if pk.get("has_curves"):
                    chosen = fid
                    break
        canvas_src = packs[chosen]
        canvas_packs[ev] = {
            "fid": canvas_src["fid"],
            "note": canvas_src["note"],
            "dt": canvas_src["dt"],
            "keys": canvas_src["keys"],
            "n": canvas_src["n"],
            "trials": [
                {
                    "i": t["i"], "e": t["e"], "m": t["m"], "n": t["n"],
                    "y": quantize(t["y"], 0) if t.get("y") else [],
                    "r": t["r"],
                    "p": {k: quantize(v, 0) for k, v in (t.get("p") or {}).items()},
                }
                for t in [
                    dict(t, y=(np.asarray(t["y"])[downsample_idx(len(t["y"]), CANVAS_MAX_T)].tolist() if t.get("y") else []),
                         p={k: np.asarray(v)[downsample_idx(len(v), CANVAS_MAX_T)].tolist() for k, v in (t.get("p") or {}).items()})
                    if t.get("y") else t
                    for t in canvas_src["trials"]
                ]
            ],
        }
        html_fold = {}
        for fid, pk in packs.items():
            html_fold[fid] = strip_cluster_fields(pk, keep_k=False)
            html_fold[fid]["keys"] = [k for k in NEURAL_KEYS if k in (html_fold[fid].get("keys") or [])]
            for t in html_fold[fid]["trials"]:
                t["p"] = {k: v for k, v in t["p"].items() if k not in ("mean", "ar") and v}
                t["r"] = {k: v for k, v in t["r"].items() if k not in ("mean", "ar")}
                if not t.get("y"):
                    t.pop("y", None)
        html_packs[ev] = {
            "default_fid": chosen,
            "packs": html_fold,
            # backward compatible top-level fields for any leftover pack() callers
            **{k: html_fold[chosen][k] for k in ("fid", "note", "dt", "keys", "n", "n_max", "has_curves", "trials")},
        }
        cluster_fold = {}
        for fid, pk in packs.items():
            cluster_fold[fid] = strip_cluster_fields(pk, keep_k=True)
            cluster_fold[fid]["keys"] = [k for k in NEURAL_KEYS if k in (cluster_fold[fid].get("keys") or [])]
            for t in cluster_fold[fid]["trials"]:
                t["p"] = {k: v for k, v in t["p"].items() if k not in ("mean", "ar") and v}
                t["r"] = {k: v for k, v in t["r"].items() if k not in ("mean", "ar")}
                t["k"] = {k: v for k, v in (t.get("k") or {}).items() if k not in ("mean", "ar")}
                if not t.get("y"):
                    t.pop("y", None)
        cluster_evals[ev] = {
            "default_fid": chosen,
            "packs": cluster_fold,
            **{k: cluster_fold[chosen][k] for k in ("fid", "note", "dt", "keys", "n", "n_max", "has_curves", "trials")},
        }

    # downsample canvas y/p was a bit tangled; rebuild canvas packs cleanly
    canvas_evals = {}
    n_canvas_jobs = 0
    for ev in evals:
        src = None
        packs = html_packs[ev]["packs"]
        # recover full pack from cluster (has same y)
        src = cluster_evals[ev]["packs"][html_packs[ev]["default_fid"]]
        n_canvas_jobs += src.get("n_jobs") or 0
        trials = []
        for t in src["trials"]:
            y = t.get("y") or []
            if not y:
                continue
            idx = downsample_idx(len(y), CANVAS_MAX_T)
            rec = {
                "i": t["i"], "e": t["e"], "m": t["m"], "n": t["n"],
                "y": quantize(np.asarray(y)[idx], 0),
                "r": t["r"],
                "p": {},
            }
            for k, v in (t.get("p") or {}).items():
                rec["p"][k] = quantize(np.asarray(v)[idx], 0)
            trials.append(rec)
        canvas_evals[ev] = {
            "fid": src["fid"],
            "note": src["note"],
            "dt": src["dt"],
            "keys": src["keys"],
            "n": len(trials),
            "trials": trials,
        }

    canvas_blob = json.dumps(
        {"evals": canvas_evals, "meta": {"n_jobs": n_canvas_jobs, "max_t": CANVAS_MAX_T}},
        separators=(",", ":"),
    )
    splice_canvas(TRACES, canvas_blob)
    splice_canvas(OVERLAY, canvas_blob)

    attach_wheel_from_html({ev: cluster_evals[ev]["packs"][cluster_evals[ev]["default_fid"]] for ev in evals})
    # also copy wheel onto matching fid packs
    text = HTML_WHEEL.read_text()
    m = re.search(r'<script id="trace-json" type="application/json">(.*?)</script>', text, re.S)
    if m:
        old = json.loads(m.group(1))
        for ev in evals:
            old_ev = (old.get("evals") or {}).get(ev) or {}
            old_fid = old_ev.get("fid")
            by_i = {t["i"]: t.get("w") for t in old_ev.get("trials") or [] if t.get("w")}
            for fid, pk in cluster_evals[ev]["packs"].items():
                if fid != old_fid:
                    continue
                for t in pk["trials"]:
                    w = by_i.get(t["i"])
                    if w:
                        t["w"] = w

    html_payload = {"evals": html_packs, "meta": {"snapshot": "2026-10-24", "n_jobs": metrics["meta"]["n_jobs_scored"]}}
    cluster_payload = {"evals": cluster_evals, "meta": {"snapshot": "2026-10-24", "n_jobs": metrics["meta"]["n_jobs_scored"]}}
    wheel_payload = json.loads(json.dumps(cluster_payload))
    # wheel html does not need cluster fields but w is useful; keep k off
    for ev in evals:
        for fid, pk in wheel_payload["evals"][ev]["packs"].items():
            for t in pk["trials"]:
                t.pop("k", None)
                t.pop("c", None)
                t.pop("cn", None)
                t.pop("wc", None)
                t.pop("wn", None)
        # sync top-level trials
        ch = wheel_payload["evals"][ev]["default_fid"]
        wheel_payload["evals"][ev]["trials"] = wheel_payload["evals"][ev]["packs"][ch]["trials"]

    for ev in evals:
        ch = html_payload["evals"][ev]["default_fid"]
        html_payload["evals"][ev]["trials"] = html_payload["evals"][ev]["packs"][ch]["trials"]
        ch = cluster_payload["evals"][ev]["default_fid"]
        cluster_payload["evals"][ev]["trials"] = cluster_payload["evals"][ev]["packs"][ch]["trials"]

    splice_html(HTML_TRACES, html_payload)
    splice_html(HTML_WHEEL, wheel_payload)
    splice_html(HTML_CLUSTERS, cluster_payload)
    for p in (HTML_TRACES, HTML_WHEEL, HTML_CLUSTERS):
        patch_html_js(p)

    summary = {
        "inventory": inv,
        "meta": metrics["meta"],
        "headlines": headline_rows(metrics),
        "canvas_folds": {ev: canvas_evals[ev]["fid"] for ev in evals},
        "html_default": {ev: html_packs[ev]["default_fid"] for ev in evals},
        "html_has_curves": {
            ev: {fid: pk["has_curves"] for fid, pk in html_packs[ev]["packs"].items()}
            for ev in evals
        },
    }
    SUMMARY.write_text(json.dumps(summary, indent=2))
    print("wrote", SUMMARY)
    for block in summary["headlines"]:
        print("==", block["eval"], block["fid"])
        for row in block["trace"]:
            print(f"  {row['m']:6s} {row['l']:11s} cat={row['cat']} med={row['med']} n={row['n']}")


if __name__ == "__main__":
    main()
