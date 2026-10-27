#!/usr/bin/env python3
"""Inventory ExperimentsClean jobs without opening model.pt."""
from __future__ import annotations

import csv
import json
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path("/Users/Stefan/Desktop/BrainMap/BehaviorRes/ExperimentsClean/paw_speed/jobs")


def main():
    dirs = sorted(p for p in ROOT.iterdir() if p.is_dir())
    print("n_dirs", len(dirs))
    n_sc = n_npz = n_ht = n_err = n_status = n_epochs = 0
    states = Counter()
    by_eval_obj = Counter()
    by_fold = Counter()
    missing_npz_trace = []
    failed = []
    starting = []
    scored_rows = []
    fids = defaultdict(set)
    for d in dirs:
        sc, npz, ht, err, st, ep = (
            d / "scores.csv",
            d / "holdout_curves.npz",
            d / "holdout_trials.csv",
            d / "error.txt",
            d / "status.json",
            d / "epochs.log",
        )
        has_sc, has_npz, has_ht = sc.exists(), npz.exists(), ht.exists()
        n_sc += int(has_sc)
        n_npz += int(has_npz)
        n_ht += int(has_ht)
        n_err += int(err.exists())
        n_status += int(st.exists())
        n_epochs += int(ep.exists())
        state = None
        if st.exists():
            try:
                state = json.loads(st.read_text()).get("state")
            except Exception:
                state = "bad_status"
            states[state] += 1
        name = d.name
        prefix = name.split("_", 1)[0]
        row = None
        if has_sc:
            with sc.open(newline="") as f:
                recs = list(csv.DictReader(f))
            if recs:
                row = recs[0]
                ev = row.get("eval") or ""
                fid = row.get("fold_id") or ""
                obj = row.get("objective") or ""
                model = row.get("model") or ""
                loss = row.get("loss") or ""
                by_eval_obj[(ev, obj)] += 1
                by_fold[(ev, fid)] += 1
                fids[ev].add(fid)
                r2 = row.get("fold_concat_r2")
                scored_rows.append(
                    {
                        "job": prefix,
                        "eval": ev,
                        "fold_id": fid,
                        "model": model,
                        "obj": obj,
                        "loss": loss,
                        "r2": r2,
                        "npz": has_npz,
                    }
                )
                if obj == "trace" and not has_npz:
                    missing_npz_trace.append(name)
        if err.exists() and not has_sc:
            failed.append(name)
        if state == "starting" and not has_sc:
            starting.append(name)
    print("n_scores", n_sc, "n_npz", n_npz, "n_holdout_trials", n_ht)
    print("n_error", n_err, "n_status", n_status, "n_epochs", n_epochs)
    print("states", dict(states))
    print("folds", {k: sorted(v) for k, v in fids.items()})
    print("by_fold_counts")
    for k, n in sorted(by_fold.items()):
        print(" ", k, n)
    print("by_eval_obj", dict(by_eval_obj))
    print("missing_trace_npz", len(missing_npz_trace))
    for x in missing_npz_trace[:20]:
        print("  ", x)
    print("failed_no_scores", failed)
    print("starting_no_scores", starting)
    print("job_prefixes", sorted({s["job"] for s in scored_rows})[:40], "... total unique", len({s["job"] for s in scored_rows}))
    # TRACE concat R2 snapshot per eval/fold/model/loss
    print("TRACE_R2")
    for s in scored_rows:
        if s["obj"] != "trace":
            continue
        print(f"  {s['eval']:3s} {s['fold_id']:12s} {s['model']:24s} {s['loss']:11s} r2={s['r2']} npz={int(s['npz'])} job={s['job']}")


if __name__ == "__main__":
    main()
