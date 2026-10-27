#!/usr/bin/env python3
"""Rebuild experiments-clean-progress.canvas.tsx DATA from scores + holdout_trials.

Never opens model.pt.
"""
from __future__ import annotations

import csv
import json
import math
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path("/Users/Stefan/Desktop/BrainMap/BehaviorRes/ExperimentsClean/paw_speed/jobs")
CANVAS = Path(
    "/Users/Stefan/.cursor/projects/Users-Stefan-Desktop/canvases/experiments-clean-progress.canvas.tsx"
)

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

R2_EDGES = (-0.5, -0.25, 0.0, 0.25, 0.5, 0.75)
ONSET_EDGES = (0.04, 0.08, 0.12, 0.20, 0.40)


def fnum(s):
    if s is None or s == "":
        return None
    try:
        x = float(s)
    except (TypeError, ValueError):
        return None
    return x if math.isfinite(x) else None


def rnd(x, nd):
    x = fnum(x) if not isinstance(x, (int, float, np.floating)) else float(x)
    if x is None or not math.isfinite(x):
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


def collect():
    folds = []
    dist_rows = defaultdict(lambda: defaultdict(list))
    n_scored = 0
    n_trace = defaultdict(int)
    for d in sorted(p for p in ROOT.iterdir() if p.is_dir()):
        sc, ht = d / "scores.csv", d / "holdout_trials.csv"
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
        folds.append(
            {
                "e": ev,
                "fid": fid,
                "m": model,
                "o": obj,
                "l": loss,
                "n": int(float(row["n_holdout_trials"])) if fnum(row.get("n_holdout_trials")) is not None else None,
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
            n_trace[ev] = folds[-1]["n"] or n_trace[ev]
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
        n = len(bag["r2"])
        rec = {
            "e": ev,
            "fid": fid,
            "m": model,
            "l": loss,
            "n": n,
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
    data = {
        "folds": folds,
        "dist": dist["trace"],
        "dist_onset": dist["onset"],
        "dist_mixed": dist["mixed"],
        "dist_dir": dist["direction"],
        "meta": {
            "n_jobs_scored": n_scored,
            "n_trials_LTO_trace": n_trace.get("LTO"),
            "n_trials_LSO_trace": n_trace.get("LSO"),
            "n_trials_LMO_trace": n_trace.get("LMO"),
            "snapshot": "2026-10-24",
            "folds": {
                "LTO": ["lto00", "lto05"],
                "LSO": ["03063955", "cef05f87"],
                "LMO": ["CSHL049", "PL033"],
            },
        },
    }
    return data


def main():
    data = collect()
    blob = json.dumps(data, separators=(",", ":"))
    text = CANVAS.read_text()
    start = text.find("const DATA: Data = ")
    end = text.find(";\nconst EVALS", start)
    if start < 0 or end < 0:
        # blank line variant
        end = text.find(";\n\nconst EVALS", start)
    if start < 0 or end < 0:
        raise SystemExit(f"DATA bounds {start} {end}")
    text = text[: start + len("const DATA: Data = ")] + blob + text[end:]
    CANVAS.write_text(text)
    print("n_jobs", data["meta"]["n_jobs_scored"])
    print("n_fold_rows", len(data["folds"]))
    print("dist_trace", len(data["dist"]))
    print("wrote", CANVAS, "bytes", CANVAS.stat().st_size)
    # headline TRACE concat R2
    for ev, fid in (("LTO", "lto05"), ("LSO", "cef05f87"), ("LMO", "PL033"),
                    ("LTO", "lto00"), ("LSO", "03063955"), ("LMO", "CSHL049")):
        print("==", ev, fid)
        for row in data["folds"]:
            if row["e"] == ev and row["fid"] == fid and row["o"] == "trace":
                print(
                    f"  {row['m']:6s} {row['l']:11s} cat={row['r2']['cat']} "
                    f"med={row['r2']['med']} mean={row['r2']['mean']}"
                )


if __name__ == "__main__":
    main()
