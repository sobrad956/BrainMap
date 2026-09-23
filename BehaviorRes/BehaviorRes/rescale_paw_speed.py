"""Patch every BehaviorRes dataset: right-camera x/y ×2, 2D paw speed, speedNormalized.

Does not recompute STILL/COUPLED/DECOUPLED or wheel_paws. dlc_speed / lp_speed
stay in place and become hypot(vx, vy) after the scale.
"""

from __future__ import annotations

import json
import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from enrich_and_discretize import (
    DISC_SCALAR_FIELDS,
    NORM_SCALAR_FIELDS,
    SCALAR_FIELDS,
    apply_speed_normalized,
    log,
    rescale_right_camera_trial,
    trial_scalars,
    utc_now,
)

BEH = ROOT
FOLDERS = [
    BEH / "Database",
    BEH / "DatasetFiltered",
    BEH / "DatabaseDiscretized",
    BEH / "DatasetFilteredDiscretized",
    BEH / "ModelData",
    BEH / "ModelDataLeft",
    BEH / "ModelDataRight",
    BEH / "ModelDataLeftIpsi",
    BEH / "ModelDataRightContra",
]

KINEMATICS = {
    "right_camera_scale": 2.0,
    "right_camera_note": (
        "IBL right camera is 640x512, left is 1280x1024. Near-camera x/y on "
        "right-paw trials are multiplied by 2. Disparity z is rebuilt as "
        "x_near_scaled - x_far_scaled so both cameras speak left-camera pixels. "
        "Paw-state labels are NOT recomputed."
    ),
    "speed": "2d_hypot_vx_vy",
    "speedNormalized": (
        "dlc_speedNormalized, lp_speedNormalized, and speedNormalized "
        "(Lightning Pose if present else DLC) = 2D speed divided by that "
        "session's median coupled speed for the same tracker and used_paw."
    ),
}


def already_patched(payload):
    kin = payload.get("kinematics") or {}
    return kin.get("right_camera_scale") == 2.0 and kin.get("speed") == "2d_hypot_vx_vy"


def scalar_fields_for(folder, payload):
    rec = payload["trials"][0] if payload.get("trials") else {}
    base = list(DISC_SCALAR_FIELDS if "n_bins" in rec else SCALAR_FIELDS)
    for key in NORM_SCALAR_FIELDS:
        if key not in base:
            base.append(key)
    return base


def patch_session(path, folder, fields):
    with path.open("rb") as fh:
        payload = pickle.load(fh)
    if already_patched(payload):
        return "skip"
    for rec in payload["trials"]:
        rescale_right_camera_trial(rec)
    apply_speed_normalized(payload["trials"])
    payload["kinematics"] = dict(KINEMATICS)
    payload["kinematics_patched_at"] = utc_now()
    notes = payload.get("notes")
    if isinstance(notes, dict):
        notes = dict(notes)
        notes["paw_speed"] = KINEMATICS["speed"]
        notes["speedNormalized"] = KINEMATICS["speedNormalized"]
        payload["notes"] = notes
    alignment = payload.get("alignment")
    if isinstance(alignment, dict):
        alignment = dict(alignment)
        alignment["paw_speed"] = (
            "2D hypot(vx, vy) in left-camera-equivalent pixels/s; "
            "speedNormalized = speed / session median coupled speed"
        )
        payload["alignment"] = alignment
    eid = payload["eid"]
    sess_dir = folder / "sessions"
    pkl_path = sess_dir / f"{eid}.pkl"
    tmp = pkl_path.with_suffix(".pkl.tmp")
    with tmp.open("wb") as fh:
        pickle.dump(payload, fh, protocol=pickle.HIGHEST_PROTOCOL)
    tmp.replace(pkl_path)
    pd.DataFrame([trial_scalars(rec, fields) for rec in payload["trials"]]).to_parquet(
        sess_dir / f"{eid}_trials.parquet", index=False
    )
    ok_path = sess_dir / f"{eid}.ok.json"
    ok = json.loads(ok_path.read_text()) if ok_path.exists() else {"eid": eid}
    ok["kinematics_patched_at"] = utc_now()
    ok["n_trials"] = payload["n_trials"]
    ok_path.write_text(json.dumps(ok))
    return "ok"


def merge_manifest(folder, n_ok, n_skip):
    path = folder / "manifest.json"
    man = json.loads(path.read_text()) if path.exists() else {}
    trial_parts = sorted((folder / "sessions").glob("*_trials.parquet"))
    if trial_parts:
        trials = pd.concat([pd.read_parquet(p) for p in trial_parts], ignore_index=True)
        trials.to_parquet(folder / "trials.parquet", index=False)
        man["n_mice"] = int(trials["mouse_id"].nunique())
        man["n_sessions"] = int(trials["eid"].nunique())
        man["n_trials"] = int(len(trials))
    man["kinematics"] = dict(KINEMATICS)
    man["kinematics_patched_at"] = utc_now()
    man["kinematics_sessions_ok"] = int(n_ok)
    man["kinematics_sessions_skipped"] = int(n_skip)
    path.write_text(json.dumps(man, indent=2))
    return man


def process():
    for folder in FOLDERS:
        sess_dir = folder / "sessions"
        pkls = sorted(p for p in sess_dir.glob("*.pkl") if not p.name.endswith(".tmp"))
        log(f"{folder.name}: {len(pkls)} sessions")
        n_ok = n_skip = 0
        fields = None
        for i, path in enumerate(pkls, start=1):
            if fields is None:
                with path.open("rb") as fh:
                    sample = pickle.load(fh)
                fields = scalar_fields_for(folder, sample)
            status = patch_session(path, folder, fields)
            if status == "skip":
                n_skip += 1
            else:
                n_ok += 1
            if i == 1 or i % 10 == 0 or i == len(pkls):
                log(f"  {folder.name} {i}/{len(pkls)} ok={n_ok} skip={n_skip}")
        man = merge_manifest(folder, n_ok, n_skip)
        log(
            f"  done {folder.name}: {man.get('n_trials')} trials, "
            f"patched={n_ok}, skipped={n_skip}"
        )


if __name__ == "__main__":
    process()
