"""Label each trial left / right / both for paws that actually turn the wheel.

Stored traces only contain the single paw chosen by used_paw. This script
reloads DLC and Lightning Pose for both near paws (left camera paw_r = left
paw, right camera paw_r = right paw), reuses the STILL/COUPLED/DECOUPLED
heuristic, and writes wheel_paws onto every BehaviorRes dataset folder.
"""

from __future__ import annotations

import json
import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from brainbox.io.one import SessionLoader
from one.api import ONE

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
from enrich_and_discretize import (  # noqa: E402
    COUPLED,
    DISC_SCALAR_FIELDS,
    SCALAR_FIELDS,
    interpolate_to,
    log,
    rebuild_indexes,
    segment_paw_motion,
    tracker_kinematics,
    utc_now,
    write_session,
)
from fullData import camera_arrays, load_pose_copy  # noqa: E402

BEH = ROOT
DB_DIR = BEH / "Database"
FOLDERS = [
    (BEH / "Database", SCALAR_FIELDS),
    (BEH / "DatasetFiltered", SCALAR_FIELDS),
    (BEH / "DatabaseDiscretized", DISC_SCALAR_FIELDS),
    (BEH / "DatasetFilteredDiscretized", DISC_SCALAR_FIELDS),
    (BEH / "ModelData", DISC_SCALAR_FIELDS),
]

# A paw counts as turning the wheel only if it is COUPLED (moving, wheel
# moving, not lifted) for long enough AND its image-plane speed is in the
# range of a real reach, not a body-held jitter.
MIN_COUPLED_S = 0.080
MIN_SPEED_PX_S = 100.0
MIN_COUPLED_RATIO = 0.40


def connect_one():
    return ONE(
        base_url="https://openalyx.internationalbrainlab.org",
        password="international",
    )


def camera_xy(pose, camera_name, feature="paw_r"):
    if not pose or camera_name not in pose:
        return None
    times, x, y = camera_arrays(pose[camera_name], feature)
    if times.size < 3:
        return None
    order = np.argsort(times)
    return np.asarray(times)[order], np.asarray(x)[order], np.asarray(y)[order]


def window_xy(arr, t0, t1):
    if arr is None:
        return None
    times, x, y = arr
    i0 = int(np.searchsorted(times, t0, side="left"))
    i1 = int(np.searchsorted(times, t1, side="right"))
    if i1 - i0 < 3:
        return None
    return times[i0:i1] - t0, x[i0:i1], y[i0:i1]


def paw_metrics(arr, t0, t1, wheel_t, wheel_v, first_move_s):
    win = window_xy(arr, t0, t1)
    if win is None:
        return {"coupled_s": np.nan, "speed": np.nan, "n": 0}
    t, x, y = win
    vx, vy, _vz, _speed = tracker_kinematics(t, x, y, np.full_like(x, np.nan))
    omega = interpolate_to(wheel_t, wheel_v, t, extrapolate=False)
    states = segment_paw_motion(t, x, y, vx, vy, omega, first_move_s=first_move_s)
    dt = np.diff(t)
    dt = np.append(dt, dt[-1] if dt.size else 0.02)
    dt = np.where(np.isfinite(dt) & (dt > 0), dt, 0.0)
    coupled = states == COUPLED
    spd = np.hypot(vx, vy)
    finite_spd = spd[np.isfinite(spd)]
    return {
        "coupled_s": float(dt[coupled].sum()),
        "speed": float(np.mean(finite_spd)) if finite_spd.size else np.nan,
        "n": int(t.size),
    }


def paw_contributes(coupled_s, speed):
    return (
        np.isfinite(coupled_s)
        and np.isfinite(speed)
        and coupled_s >= MIN_COUPLED_S
        and speed >= MIN_SPEED_PX_S
    )


def decide_label(left, right, used_paw):
    left_ok = paw_contributes(left["coupled_s"], left["speed"])
    right_ok = paw_contributes(right["coupled_s"], right["speed"])
    left_c = float(left["coupled_s"]) if np.isfinite(left["coupled_s"]) else 0.0
    right_c = float(right["coupled_s"]) if np.isfinite(right["coupled_s"]) else 0.0
    if left_ok and right_ok:
        stronger = max(left_c, right_c)
        weaker = min(left_c, right_c)
        if stronger > 0 and (weaker / stronger) >= MIN_COUPLED_RATIO:
            return "both"
        return "left" if left_c >= right_c else "right"
    if left_ok:
        return "left"
    if right_ok:
        return "right"
    if used_paw in ("left", "right"):
        return used_paw
    return None


def tracker_label(pose_arr, rec):
    t0 = float(rec["stimOn_times"])
    t1 = float(rec["feedback_times"])
    first_move = float(rec["actionTime"]) if rec.get("actionTime") is not None else None
    wheel_t = np.asarray(rec.get("wheel_t", []), dtype=float)
    wheel_v = np.asarray(rec.get("wheel_velocity", []), dtype=float)
    out = {}
    for tracker in ("dlc", "lp"):
        arrs = pose_arr.get(tracker) or {}
        left = paw_metrics(arrs.get("left"), t0, t1, wheel_t, wheel_v, first_move)
        right = paw_metrics(arrs.get("right"), t0, t1, wheel_t, wheel_v, first_move)
        out[tracker] = {
            "left": left,
            "right": right,
            "label": decide_label(left, right, rec.get("used_paw")),
            "usable": (left["n"] >= 3) or (right["n"] >= 3),
        }
    if out["lp"]["usable"]:
        chosen = "lp"
    elif out["dlc"]["usable"]:
        chosen = "dlc"
    else:
        chosen = None
    label = out[chosen]["label"] if chosen else rec.get("used_paw")
    return {
        "wheel_paws": label,
        "wheel_paws_source": chosen,
        "dlc_left_coupled_s": out["dlc"]["left"]["coupled_s"],
        "dlc_right_coupled_s": out["dlc"]["right"]["coupled_s"],
        "dlc_left_speed": out["dlc"]["left"]["speed"],
        "dlc_right_speed": out["dlc"]["right"]["speed"],
        "lp_left_coupled_s": out["lp"]["left"]["coupled_s"],
        "lp_right_coupled_s": out["lp"]["right"]["coupled_s"],
        "lp_left_speed": out["lp"]["left"]["speed"],
        "lp_right_speed": out["lp"]["right"]["speed"],
    }


def load_pose_arrays(one, eid):
    loader = SessionLoader(one=one, eid=eid)
    dlc = load_pose_copy(loader, "dlc")
    lp = load_pose_copy(loader, "lightningPose")
    return {
        "dlc": {
            "left": camera_xy(dlc, "leftCamera"),
            "right": camera_xy(dlc, "rightCamera"),
        },
        "lp": {
            "left": camera_xy(lp, "leftCamera"),
            "right": camera_xy(lp, "rightCamera"),
        },
    }


def apply_to_record(rec, info):
    rec["wheel_paws"] = info["wheel_paws"]
    rec["wheel_paws_source"] = info["wheel_paws_source"]
    return rec


def patch_folder(folder, scalar_fields, labels):
    sess_dir = folder / "sessions"
    pkls = sorted(p for p in sess_dir.glob("*.pkl") if not p.name.endswith(".tmp"))
    n_trials = 0
    n_missing = 0
    for path in pkls:
        with path.open("rb") as fh:
            payload = pickle.load(fh)
        changed = False
        for rec in payload["trials"]:
            key = (payload["eid"], int(rec["trial_index"]))
            info = labels.get(key)
            if info is None:
                n_missing += 1
                rec["wheel_paws"] = rec.get("wheel_paws") or rec.get("used_paw")
                changed = True
                continue
            apply_to_record(rec, info)
            changed = True
            n_trials += 1
        if changed:
            write_session(folder, payload, scalar_fields)
    manifest = rebuild_indexes(folder, scalar_fields)
    extra = json.loads((folder / "manifest.json").read_text()) if (folder / "manifest.json").exists() else {}
    extra.update(manifest)
    extra["wheel_paws"] = {
        "field": "wheel_paws",
        "values": ["left", "right", "both"],
        "min_coupled_s": MIN_COUPLED_S,
        "min_speed_px_s": MIN_SPEED_PX_S,
        "min_coupled_ratio": MIN_COUPLED_RATIO,
        "source_tracker": "lightningPose if usable else DLC",
        "updated_at": utc_now(),
        "n_trials_labeled": n_trials,
        "n_trials_without_label_row": n_missing,
    }
    (folder / "manifest.json").write_text(json.dumps(extra, indent=2))
    return extra


def main():
    one = connect_one()
    pkls = sorted(p for p in (DB_DIR / "sessions").glob("*.pkl") if not p.name.endswith(".tmp"))
    if not pkls:
        raise SystemExit(f"No Database session pickles in {DB_DIR / 'sessions'}")

    labels = {}
    rows = []
    for i, path in enumerate(pkls, start=1):
        with path.open("rb") as fh:
            payload = pickle.load(fh)
        eid = payload["eid"]
        log(f"[{i}/{len(pkls)}] {payload.get('mouse_id')} {payload.get('session_date')} {eid}")
        try:
            pose_arr = load_pose_arrays(one, eid)
        except Exception as exc:
            log(f"  pose load failed ({exc}); falling back to used_paw")
            pose_arr = {"dlc": {"left": None, "right": None}, "lp": {"left": None, "right": None}}
        n_both = 0
        for rec in payload["trials"]:
            info = tracker_label(pose_arr, rec)
            labels[(eid, int(rec["trial_index"]))] = info
            if info["wheel_paws"] == "both":
                n_both += 1
            rows.append(
                {
                    "eid": eid,
                    "mouse_id": payload.get("mouse_id"),
                    "trial_index": int(rec["trial_index"]),
                    "used_paw": rec.get("used_paw"),
                    **info,
                }
            )
        log(f"  trials={len(payload['trials'])} both={n_both}")

    table = pd.DataFrame(rows)
    table.to_parquet(BEH / "wheel_paws_labels.parquet", index=False)
    counts = table["wheel_paws"].value_counts(dropna=False).to_dict()
    log(f"Database labels: {counts}  n={len(table)}")

    for folder, fields in FOLDERS:
        if not folder.exists():
            log(f"skip missing {folder}")
            continue
        log(f"patch {folder.name}")
        extra = patch_folder(folder, fields, labels)
        log(
            f"  {folder.name}: {extra.get('n_mice')} mice, {extra.get('n_sessions')} sessions, "
            f"{extra.get('n_trials')} trials"
        )


if __name__ == "__main__":
    main()
