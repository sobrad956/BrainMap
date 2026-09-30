#!/usr/bin/env python3
"""Rewrite stored paw xyz/velocity to the paw that actually turns the wheel.

Model pickles originally stored `used_paw`: the near-camera paw with more
image-plane motion. `wheel_paws` is the paw coupled to the wheel. This script
reloads DLC and Lightning Pose from IBL, extracts the wheel paw, recomputes
vx/vy/vz/speed on the existing 20 ms bins, and sets used_paw to that paw.
Spike counts and wheel traces are left unchanged.

For wheel_paws=="both", stores the paw with longer coupled time (dominant
wheel paw) and keeps the both label.
"""

from __future__ import annotations

import argparse
import json
import pickle
import sys
from pathlib import Path

import numpy as np
from brainbox.io.one import SessionLoader
from one.api import ONE

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from enrich_and_discretize import (  # noqa: E402
    CODE_TO_STATE,
    DISC_SCALAR_FIELDS,
    NORM_SCALAR_FIELDS,
    apply_speed_normalized,
    bin_categorical,
    interp_at_right_edges,
    log,
    paw_speed,
    rebuild_indexes,
    tracker_kinematics,
    tracker_states,
    utc_now,
    write_session,
    f32,
    empty_f32,
)
from fullData import (  # noqa: E402
    load_pose_copy,
    load_wheel_arrays,
    paw_xyz_positions,
    window_pose_xyz,
    window_trace,
)
from label_wheel_paws import decide_label, paw_metrics  # noqa: E402
from label_wheel_paws import camera_xy as near_paw_xy  # noqa: E402

FOLDERS = [
    ROOT / "ModelDataRightContra",
    ROOT / "ModelDataRight",
    ROOT / "ModelData",
    ROOT / "ModelDataLeft",
    ROOT / "ModelDataLeftIpsi",
]


def connect_one():
    return ONE(
        base_url="https://openalyx.internationalbrainlab.org",
        password="international",
    )


def scalar_fields():
    fields = list(DISC_SCALAR_FIELDS)
    for key in NORM_SCALAR_FIELDS + ["used_paw_motion"]:
        if key not in fields:
            fields.append(key)
    return fields


def bin_times(rec):
    t_right = np.asarray(rec.get("t_bin_right", rec.get("wheel_t", [])), dtype=float)
    t_left = np.asarray(rec.get("t_bin_left", []), dtype=float)
    if t_left.size != t_right.size:
        t_left = t_right - 0.02
    return t_left, t_right


def empty_binned(n):
    return empty_f32() if n == 0 else np.full(n, np.nan, dtype=np.float32)


def fill_tracker(rec, pose, side, prefix, t0, t1, wheel_t, wheel_v, first_move, t_left, t_right):
    n = int(t_right.size)
    kin = paw_xyz_positions(pose, side)
    t, x, y, z = window_pose_xyz(kin, t0, t1)
    if t.size < 3:
        for axis in ("x", "y", "z", "vx", "vy", "vz", "speed"):
            rec[f"{prefix}_{axis}"] = empty_binned(n)
        rec[f"{prefix}_t"] = t_right.copy() if n else empty_f32()
        rec[f"{prefix}_paw_state_code"] = np.full(n, -1, dtype=np.int8)
        rec[f"{prefix}_paw_state"] = np.array(["none"] * n, dtype=object)
        rec[f"n_{prefix}_state_bins"] = 0
        return False
    vx, vy, vz, speed = tracker_kinematics(t, x, y, z)
    states = tracker_states(t, x, y, vx, vy, wheel_t, wheel_v, first_move)
    rec[f"{prefix}_x"] = interp_at_right_edges(t, x, t_right)
    rec[f"{prefix}_y"] = interp_at_right_edges(t, y, t_right)
    rec[f"{prefix}_z"] = interp_at_right_edges(t, z, t_right)
    rec[f"{prefix}_vx"] = interp_at_right_edges(t, vx, t_right)
    rec[f"{prefix}_vy"] = interp_at_right_edges(t, vy, t_right)
    rec[f"{prefix}_vz"] = interp_at_right_edges(t, vz, t_right)
    rec[f"{prefix}_speed"] = f32(paw_speed(rec[f"{prefix}_vx"], rec[f"{prefix}_vy"]))
    rec[f"{prefix}_t"] = t_right.copy()
    rec[f"{prefix}_paw_state_code"] = bin_categorical(t, states, t_left, t_right)
    labels = np.array(["none"] * n, dtype=object)
    code = rec[f"{prefix}_paw_state_code"]
    good = code >= 0
    labels[good] = CODE_TO_STATE[code[good]]
    rec[f"{prefix}_paw_state"] = labels
    rec[f"n_{prefix}_state_bins"] = int(np.sum(code >= 0))
    rec[f"n_{prefix}_samples"] = n
    return True


def dominant_side(pose_pack, rec, fallback):
    t0 = float(rec["stimOn_times"])
    t1 = float(rec["feedback_times"])
    first_move = float(rec["actionTime"]) if rec.get("actionTime") is not None else None
    wheel_t = np.asarray(rec.get("_wheel_t_native", []), dtype=float)
    wheel_v = np.asarray(rec.get("_wheel_v_native", []), dtype=float)
    best = fallback if fallback in ("left", "right") else "right"
    best_c = -1.0
    for tracker in ("lp", "dlc"):
        arrs = pose_pack.get(tracker) or {}
        left = paw_metrics(arrs.get("left"), t0, t1, wheel_t, wheel_v, first_move)
        right = paw_metrics(arrs.get("right"), t0, t1, wheel_t, wheel_v, first_move)
        label = decide_label(left, right, fallback)
        if label in ("left", "right"):
            return label
        lc = float(left["coupled_s"]) if np.isfinite(left["coupled_s"]) else -1.0
        rc = float(right["coupled_s"]) if np.isfinite(right["coupled_s"]) else -1.0
        if max(lc, rc) > best_c:
            best_c = max(lc, rc)
            best = "left" if lc >= rc else "right"
    return best


def side_to_store(rec, pose_pack):
    wp = rec.get("wheel_paws")
    if wp in ("left", "right"):
        return wp
    if wp == "both":
        return dominant_side(pose_pack, rec, rec.get("used_paw"))
    if rec.get("used_paw") in ("left", "right"):
        return rec.get("used_paw")
    return "right"


def pose_pack_from_loader(dlc_pose, lp_pose):
    return {
        "dlc": {
            "left": near_paw_xy(dlc_pose, "leftCamera"),
            "right": near_paw_xy(dlc_pose, "rightCamera"),
        },
        "lp": {
            "left": near_paw_xy(lp_pose, "leftCamera"),
            "right": near_paw_xy(lp_pose, "rightCamera"),
        },
    }


def patch_trial(rec, dlc_pose, lp_pose, pose_pack, wheel_times, wheel_vel):
    t0 = float(rec["stimOn_times"])
    t1 = float(rec["feedback_times"])
    t_left, t_right = bin_times(rec)
    n = int(t_right.size)
    first_move = float(rec["actionTime"]) if rec.get("actionTime") is not None else None
    wheel_t, wheel_v = window_trace(wheel_times, wheel_vel, t0, t1, require_finite_values=True)
    rec["_wheel_t_native"] = wheel_t
    rec["_wheel_v_native"] = wheel_v
    side = side_to_store(rec, pose_pack)
    rec["used_paw_motion"] = rec.get("used_paw")
    rec["used_paw"] = side
    dlc_ok = fill_tracker(
        rec, dlc_pose, side, "dlc", t0, t1, wheel_t, wheel_v, first_move, t_left, t_right
    )
    lp_ok = fill_tracker(
        rec, lp_pose, side, "lp", t0, t1, wheel_t, wheel_v, first_move, t_left, t_right
    )
    rec.pop("_wheel_t_native", None)
    rec.pop("_wheel_v_native", None)
    if n:
        rec["n_dlc_samples"] = n
        rec["n_lp_samples"] = n
    return side, dlc_ok, lp_ok


def already_patched(payload):
    kin = payload.get("kinematics") or {}
    return kin.get("paw_source") == "wheel_paws"


def patch_payload(payload, dlc_pose, lp_pose, pose_pack, wheel_times, wheel_vel):
    n_flip = 0
    for rec in payload["trials"]:
        old = rec.get("used_paw")
        side, dlc_ok, lp_ok = patch_trial(
            rec, dlc_pose, lp_pose, pose_pack, wheel_times, wheel_vel
        )
        if old != side:
            n_flip += 1
        rec["paw_trace_side"] = side
        rec["paw_trace_dlc"] = bool(dlc_ok)
        rec["paw_trace_lp"] = bool(lp_ok)
    apply_speed_normalized(payload["trials"])
    kin = dict(payload.get("kinematics") or {})
    kin["paw_source"] = "wheel_paws"
    kin["paw_source_note"] = (
        "lp_*/dlc_* xyz and velocity are the paw in used_paw, which is now "
        "the wheel-coupled paw (wheel_paws; if both, the longer-coupled paw). "
        "used_paw_motion is the old max-speed near-paw label."
    )
    payload["kinematics"] = kin
    payload["kinematics_paw_source_at"] = utc_now()
    notes = payload.get("notes")
    if isinstance(notes, dict):
        notes = dict(notes)
        notes["paw_xyz"] = (
            "wheel-coupled paw (used_paw == stored traces). signed image-plane "
            "x/y in left-camera-equivalent pixels; z = x_near - x_far after scale"
        )
        payload["notes"] = notes
    return n_flip


def session_eids(folders):
    out = {}
    for folder in folders:
        sess = folder / "sessions"
        if not sess.exists():
            continue
        for path in sorted(p for p in sess.glob("*.pkl") if not p.name.endswith(".tmp")):
            out.setdefault(path.stem, []).append(folder)
    return out


def load_pose(one, eid):
    loader = SessionLoader(one=one, eid=eid)
    wheel_times, wheel_vel = load_wheel_arrays(loader)
    dlc = load_pose_copy(loader, "dlc")
    lp = load_pose_copy(loader, "lightningPose")
    return dlc, lp, pose_pack_from_loader(dlc, lp), wheel_times, wheel_vel


def mark_manifest(folder, n_ok, n_skip, n_fail):
    path = folder / "manifest.json"
    man = json.loads(path.read_text()) if path.exists() else {}
    extra = rebuild_indexes(folder, scalar_fields())
    man.update(extra)
    kin = dict(man.get("kinematics") or {})
    kin["paw_source"] = "wheel_paws"
    man["kinematics"] = kin
    man["paw_source_patched_at"] = utc_now()
    man["paw_source_sessions_ok"] = int(n_ok)
    man["paw_source_sessions_skipped"] = int(n_skip)
    man["paw_source_sessions_failed"] = int(n_fail)
    path.write_text(json.dumps(man, indent=2))
    return man


def process(folders, only_eid=None, force=False):
    wanted = [f for f in folders if f.exists()]
    if not wanted:
        raise SystemExit("no ModelData folders found")
    eids = session_eids(wanted)
    if only_eid:
        eids = {k: v for k, v in eids.items() if k == only_eid or k.startswith(only_eid)}
        if not eids:
            raise SystemExit(f"no session {only_eid}")
    log(f"{len(eids)} unique sessions across {len(wanted)} folders")
    one = connect_one()
    fields = scalar_fields()
    stats = {f: {"ok": 0, "skip": 0, "fail": 0} for f in wanted}
    n_flip_total = 0
    for i, (eid, folders_here) in enumerate(sorted(eids.items()), start=1):
        need = []
        for folder in folders_here:
            path = folder / "sessions" / f"{eid}.pkl"
            with path.open("rb") as fh:
                payload = pickle.load(fh)
            if already_patched(payload) and not force:
                stats[folder]["skip"] += 1
                continue
            need.append((folder, payload))
        if not need:
            log(f"[{i}/{len(eids)}] {eid[:8]} skip (already wheel_paws)")
            continue
        log(f"[{i}/{len(eids)}] {eid}  folders={len(need)}")
        try:
            dlc, lp, pack, wheel_t, wheel_v = load_pose(one, eid)
        except Exception as exc:
            log(f"  pose load failed: {exc}")
            for folder, _ in need:
                stats[folder]["fail"] += 1
            continue
        for folder, payload in need:
            try:
                n_flip = patch_payload(payload, dlc, lp, pack, wheel_t, wheel_v)
                n_flip_total += n_flip
                write_session(folder, payload, fields)
                stats[folder]["ok"] += 1
                log(
                    f"  {folder.name}: trials={payload['n_trials']} "
                    f"flipped_used_paw={n_flip}"
                )
            except Exception as exc:
                log(f"  {folder.name} FAILED: {exc}")
                stats[folder]["fail"] += 1
    for folder in wanted:
        s = stats[folder]
        man = mark_manifest(folder, s["ok"], s["skip"], s["fail"])
        log(
            f"done {folder.name}: trials={man.get('n_trials')} "
            f"ok={s['ok']} skip={s['skip']} fail={s['fail']}"
        )
    log(f"used_paw reassigned on {n_flip_total} trial-writes")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--eid", default=None, help="rebuild one session (prefix ok)")
    p.add_argument("--force", action="store_true")
    p.add_argument(
        "--only-contra",
        action="store_true",
        help="only ModelDataRightContra (the Modelv1 corpus)",
    )
    args = p.parse_args()
    folders = [ROOT / "ModelDataRightContra"] if args.only_contra else FOLDERS
    process(folders, only_eid=args.eid, force=args.force)


if __name__ == "__main__":
    main()
