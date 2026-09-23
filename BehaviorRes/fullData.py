"""Build a trial-level BWM motor-cortex dataset under BehaviorRes/Database.

Each saved trial is one BWM-paper trial from the motor-cortex cohort in
MOUSELAB/BWM_2023_12_mice_with_motor_cortex_recordings.csv (39 mice, 67
sessions). Units are the well-isolated neurons used in the original
Brain-Wide Map paper (frozen clusters.pqt, QC label >= 1).

Scalar columns are written to parquet; time series and spike trains live
in per-session pickles so the ~30k-trial array payload stays resumable.
"""

from __future__ import annotations

import argparse
import json
import pickle
import traceback
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from brainbox.behavior.dlc import likelihood_threshold
from brainbox.io.one import SessionLoader, SpikeSortingLoader
from iblutil.numerical import ismember
from one.api import ONE
from scipy.interpolate import interp1d

ROOT = Path(__file__).resolve().parent
MOUSE_CSV = ROOT / "MOUSELAB" / "BWM_2023_12_mice_with_motor_cortex_recordings.csv"
BWM_RELEASE_CSV = ROOT / "2023_12_bwm_release.csv"
TRIALS_PQT = ROOT / "trials.pqt"
CLUSTERS_PQT = ROOT / "clusters.pqt"
DB_DIR = ROOT / "Database"
SESS_DIR = DB_DIR / "sessions"
MANIFEST_PATH = DB_DIR / "manifest.json"
PROGRESS_PATH = DB_DIR / "progress.jsonl"
TRIALS_INDEX_PATH = DB_DIR / "trials.parquet"
UNITS_INDEX_PATH = DB_DIR / "units.parquet"

CACHE_VERSION = 1
SPIKE_REVISION = "2024-05-06"
MIN_QC = 1.0
MIN_RT = 0.08
MAX_RT = 2.0
LIKELIHOOD_THR = 0.95
EXPECTED_N_MICE = 39
EXPECTED_N_SESSIONS = 67

CHOICE_MAP = {1: "left", -1: "right", 0: "nogo"}
OUTCOME_MAP = {1: "reward", -1: "error", 0: "no_feedback"}

SCALAR_FIELDS = [
    "mouse_id",
    "lab",
    "session_date",
    "eid",
    "trial_index",
    "block_prior",
    "stim_side",
    "choice",
    "choice_ibl",
    "brightness",
    "actionTime",
    "trialDuration",
    "reward",
    "outcome",
    "rewardVolume",
    "used_paw",
    "wheel_paws",
    "stimOn_times",
    "feedback_times",
    "firstMovement_times",
    "n_wheel_samples",
    "n_dlc_samples",
    "n_lp_samples",
    "n_spikes",
    "n_units",
]


def connect_one():
    return ONE(
        base_url="https://openalyx.internationalbrainlab.org",
        password="international",
    )


def utc_now():
    return datetime.now(timezone.utc).isoformat()


def f32(values):
    return np.asarray(values, dtype=np.float32)


def f64(values):
    return np.asarray(values, dtype=np.float64)


def i32(values):
    return np.asarray(values, dtype=np.int32)


def as_str_array(values):
    return np.asarray([str(v) for v in values], dtype=object)


def empty_f32():
    return np.array([], dtype=np.float32)


def log(msg):
    line = f"[{datetime.now().strftime('%H:%M:%S')}] {msg}"
    print(line, flush=True)
    DB_DIR.mkdir(parents=True, exist_ok=True)
    with (DB_DIR / "build.log").open("a") as fh:
        fh.write(line + "\n")


def parse_session_dates(raw):
    return [part.strip()[:10] for part in str(raw).split(",") if part.strip()]


def load_cohort_sessions():
    mice = pd.read_csv(MOUSE_CSV)
    bwm = pd.read_csv(BWM_RELEASE_CSV)
    bwm["date"] = pd.to_datetime(bwm["date"]).dt.strftime("%Y-%m-%d")
    rows = []
    for rec in mice.itertuples():
        for date in parse_session_dates(rec.motor_session_dates):
            hit = bwm[(bwm["subject"] == rec.subject) & (bwm["date"] == date)]
            if hit.empty:
                raise RuntimeError(f"No BWM eid for {rec.subject} {date}")
            eid = str(hit["eid"].iloc[0])
            probes = (
                hit[["pid", "probe_name"]]
                .drop_duplicates()
                .assign(pid=lambda d: d["pid"].astype(str), probe_name=lambda d: d["probe_name"].astype(str))
            )
            rows.append(
                {
                    "subject": rec.subject,
                    "lab": rec.lab,
                    "date": date,
                    "eid": eid,
                    "pids": probes["pid"].tolist(),
                    "probe_names": probes["probe_name"].tolist(),
                }
            )
    sessions = pd.DataFrame(rows)
    n_mice = sessions["subject"].nunique()
    n_sess = sessions["eid"].nunique()
    if n_mice != EXPECTED_N_MICE or n_sess != EXPECTED_N_SESSIONS:
        raise RuntimeError(
            f"Expected {EXPECTED_N_MICE} mice / {EXPECTED_N_SESSIONS} sessions, "
            f"found {n_mice} / {n_sess}"
        )
    return sessions


def stimulus_side(row):
    left_ok = pd.notna(row["contrastLeft"])
    right_ok = pd.notna(row["contrastRight"])
    if left_ok and not right_ok:
        return "left"
    if right_ok and not left_ok:
        return "right"
    return None


def stimulus_brightness(row, side):
    if side == "left":
        val = row.contrastLeft
    elif side == "right":
        val = row.contrastRight
    else:
        return np.nan
    return float(val) if pd.notna(val) else np.nan


def map_choice(value):
    if pd.isna(value):
        return None
    return CHOICE_MAP.get(int(value), str(int(value)))


def map_outcome(value):
    if pd.isna(value):
        return None
    return OUTCOME_MAP.get(int(value), str(int(value)))


def session_trials(all_trials, eid):
    sess = all_trials.loc[all_trials["eid"] == eid].copy().reset_index(drop=True)
    sess.insert(0, "trial_index", np.arange(len(sess), dtype=int))
    sess["stim_side"] = sess.apply(stimulus_side, axis=1)
    sess["actionTime"] = sess["firstMovement_times"] - sess["stimOn_times"]
    sess["trialDuration"] = sess["feedback_times"] - sess["stimOn_times"]
    keep = (
        sess["bwm_include"].astype(bool)
        & sess["actionTime"].between(MIN_RT, MAX_RT)
        & np.isfinite(sess["stimOn_times"])
        & np.isfinite(sess["feedback_times"])
        & np.isfinite(sess["firstMovement_times"])
        & (sess["trialDuration"] > 0)
    )
    return sess.loc[keep].reset_index(drop=True)


def interpolate_to(times_src, values_src, times_dst):
    finite = np.isfinite(times_src) & np.isfinite(values_src)
    if finite.sum() < 2:
        return np.full(np.shape(times_dst), np.nan)
    t = times_src[finite]
    v = values_src[finite]
    order = np.argsort(t)
    t = t[order]
    v = v[order]
    uniq = np.concatenate(([True], np.diff(t) > 0))
    t = t[uniq]
    v = v[uniq]
    if t.size < 2:
        return np.full(np.shape(times_dst), np.nan)
    interpolator = interp1d(t, v, kind="linear", bounds_error=False, fill_value=np.nan)
    out = interpolator(times_dst)
    out[(times_dst < t[0]) | (times_dst > t[-1])] = np.nan
    return out


def window_trace(times, values, t0, t1, require_finite_values=True):
    mask = (times >= t0) & (times <= t1) & np.isfinite(times)
    if require_finite_values:
        mask = mask & np.isfinite(values)
    if not np.any(mask):
        return empty_f32(), empty_f32()
    return f32(times[mask] - t0), f32(values[mask])


def window_xyz(times, x, y, z, t0, t1):
    mask = (times >= t0) & (times <= t1) & np.isfinite(times)
    if not np.any(mask):
        return empty_f32(), empty_f32(), empty_f32(), empty_f32()
    return (
        f32(times[mask] - t0),
        f32(x[mask]),
        f32(y[mask]),
        f32(z[mask]),
    )


def camera_arrays(pose_df, feature):
    xcol = f"{feature}_x"
    ycol = f"{feature}_y"
    if pose_df is None or xcol not in pose_df.columns or ycol not in pose_df.columns:
        return np.array([]), np.array([]), np.array([])
    return (
        pose_df["times"].to_numpy(dtype=float),
        pose_df[xcol].to_numpy(dtype=float),
        pose_df[ycol].to_numpy(dtype=float),
    )


def mean_speed(times, x, y, t0, t1):
    rel_t, x_w = window_trace(times, x, t0, t1)
    _, y_w = window_trace(times, y, t0, t1)
    n = min(rel_t.size, x_w.size, y_w.size)
    if n < 3:
        return np.nan
    rel_t, x_w, y_w = rel_t[:n], x_w[:n], y_w[:n]
    dt = np.diff(rel_t)
    disp = np.sqrt(np.diff(x_w) ** 2 + np.diff(y_w) ** 2)
    good = np.isfinite(dt) & np.isfinite(disp) & (dt > 0)
    if not np.any(good):
        return np.nan
    return float(np.nanmean(disp[good] / dt[good]))


def select_used_paw(pose, t0, t1):
    """Near-paw (`paw_r`) with larger movement: left camera = left paw."""
    if not pose:
        return None
    speeds = {}
    for cam, name in (("left", "leftCamera"), ("right", "rightCamera")):
        if name not in pose:
            speeds[cam] = np.nan
            continue
        t, x, y = camera_arrays(pose[name], "paw_r")
        speeds[cam] = mean_speed(t, x, y, t0, t1)
    left_speed = speeds.get("left", np.nan)
    right_speed = speeds.get("right", np.nan)
    if not np.isfinite(left_speed) and not np.isfinite(right_speed):
        if "leftCamera" in pose:
            return "left"
        if "rightCamera" in pose:
            return "right"
        return None
    if not np.isfinite(right_speed) or (
        np.isfinite(left_speed) and left_speed >= right_speed
    ):
        return "left"
    return "right"


RIGHT_CAMERA_SCALE = 2.0  # IBL right camera is 640x512; left is 1280x1024


def paw_xyz_positions(pose, side):
    if side is None or not pose:
        return None
    if side == "left":
        near = pose.get("leftCamera")
        far = pose.get("rightCamera")
        near_feat = "paw_r"
        far_feat = "paw_l"
        scale_near = 1.0
        scale_far = RIGHT_CAMERA_SCALE
    else:
        near = pose.get("rightCamera")
        far = pose.get("leftCamera")
        near_feat = "paw_r"
        far_feat = "paw_l"
        scale_near = RIGHT_CAMERA_SCALE
        scale_far = 1.0
    if near is None:
        return None
    t, x, y = camera_arrays(near, near_feat)
    if t.size == 0:
        return None
    x = np.asarray(x, dtype=float) * scale_near
    y = np.asarray(y, dtype=float) * scale_near
    if far is None:
        z = np.full(t.shape, np.nan)
    else:
        far_t, far_x, _far_y = camera_arrays(far, far_feat)
        if far_t.size:
            far_x = np.asarray(far_x, dtype=float) * scale_far
            z = x - interpolate_to(far_t, far_x, t)
        else:
            z = np.full(t.shape, np.nan)
    return {"times": t, "x": x, "y": y, "z": z}


def load_pose_copy(sess_loader, tracker):
    last_err = None
    for views in (["left", "right"], ["left"], ["right"]):
        try:
            sess_loader.load_pose(
                likelihood_thr=0.0,
                views=views,
                tracker=tracker,
            )
            last_err = None
            break
        except Exception as exc:
            last_err = exc
    if last_err is not None or not getattr(sess_loader, "pose", None):
        log(f"  pose tracker={tracker} unavailable: {last_err}")
        return {}
    pose = {}
    for name, df in sess_loader.pose.items():
        if df is None or df.empty or "times" not in df.columns:
            continue
        scored = df.drop(columns=["times"]).copy()
        scored = likelihood_threshold(scored, LIKELIHOOD_THR)
        out = scored.copy()
        out.insert(0, "times", df["times"].to_numpy())
        pose[name] = out
    return pose


def load_wheel_arrays(sess_loader):
    try:
        sess_loader.load_wheel(fs=1000)
        wheel = sess_loader.wheel
        return (
            wheel["times"].to_numpy(dtype=float),
            wheel["velocity"].to_numpy(dtype=float),
        )
    except Exception as exc:
        log(f"  wheel unavailable: {exc}")
        return np.array([]), np.array([])


def channel_table(channels):
    n = len(channels["x"]) if channels else 0
    if n == 0:
        return {
            "x": np.array([], dtype=float),
            "y": np.array([], dtype=float),
            "z": np.array([], dtype=float),
            "axial_um": np.array([], dtype=float),
            "lateral_um": np.array([], dtype=float),
            "atlas_id": np.array([], dtype=int),
            "acronym": np.array([], dtype=object),
        }
    acronym = channels.get("acronym")
    if acronym is None:
        acronym = np.array(["void"] * n, dtype=object)
    else:
        acronym = as_str_array(acronym)
    return {
        "x": np.asarray(channels["x"], dtype=float),
        "y": np.asarray(channels["y"], dtype=float),
        "z": np.asarray(channels["z"], dtype=float),
        "axial_um": np.asarray(channels["axial_um"], dtype=float),
        "lateral_um": np.asarray(channels["lateral_um"], dtype=float),
        "atlas_id": np.asarray(channels.get("atlas_id", np.zeros(n)), dtype=int),
        "acronym": acronym,
    }


def empty_units():
    return {
        "uuids": np.array([], dtype=object),
        "cluster_id": np.array([], dtype=int),
        "channel": np.array([], dtype=np.int32),
        "x": np.array([], dtype=float),
        "y": np.array([], dtype=float),
        "z": np.array([], dtype=float),
        "axial_um": np.array([], dtype=float),
        "lateral_um": np.array([], dtype=float),
        "acronym": np.array([], dtype=object),
        "atlas_id": np.array([], dtype=int),
    }


def unit_records(clusters, channels):
    ch_idx = np.asarray(clusters["channels"], dtype=int)
    n_ch = max(len(channels["x"]), 1)
    ch_idx = np.clip(ch_idx, 0, n_ch - 1)
    x = np.asarray(clusters["x"], dtype=float) if "x" in clusters else channels["x"][ch_idx]
    y = np.asarray(clusters["y"], dtype=float) if "y" in clusters else channels["y"][ch_idx]
    z = np.asarray(clusters["z"], dtype=float) if "z" in clusters else channels["z"][ch_idx]
    axial = (
        np.asarray(clusters["axial_um"], dtype=float)
        if "axial_um" in clusters
        else channels["axial_um"][ch_idx]
    )
    lateral = (
        np.asarray(clusters["lateral_um"], dtype=float)
        if "lateral_um" in clusters
        else channels["lateral_um"][ch_idx]
    )
    acronym = (
        as_str_array(clusters["acronym"])
        if "acronym" in clusters
        else channels["acronym"][ch_idx]
    )
    atlas_id = (
        np.asarray(clusters["atlas_id"], dtype=int)
        if "atlas_id" in clusters
        else channels["atlas_id"][ch_idx]
    )
    return {
        "uuids": as_str_array(clusters["uuids"]),
        "cluster_id": np.asarray(clusters.get("cluster_id", np.arange(len(clusters))), dtype=int),
        "channel": ch_idx.astype(np.int32),
        "x": x,
        "y": y,
        "z": z,
        "axial_um": axial,
        "lateral_um": lateral,
        "acronym": acronym,
        "atlas_id": atlas_id,
    }


def filter_to_bwm_units(spikes, clusters, keep_uuids):
    uuids = as_str_array(clusters["uuids"])
    keep = np.array([u in keep_uuids for u in uuids], dtype=bool)
    if not np.any(keep):
        empty = {
            "times": np.array([], dtype=float),
            "clusters": np.array([], dtype=np.int32),
        }
        return empty, clusters.iloc[0:0].reset_index(drop=True)
    keep_idx = np.flatnonzero(keep)
    remap = np.full(len(clusters), -1, dtype=int)
    remap[keep_idx] = np.arange(keep_idx.size, dtype=int)
    spike_new = remap[np.asarray(spikes["clusters"], dtype=int)]
    spike_ok = spike_new >= 0
    out_spikes = {
        "times": np.asarray(spikes["times"], dtype=float)[spike_ok],
        "clusters": spike_new[spike_ok].astype(np.int32),
    }
    out_clusters = clusters.iloc[keep_idx].reset_index(drop=True)
    return out_spikes, out_clusters


def bwm_unit_uuids(clusters_all, eid, pid):
    hit = clusters_all[
        (clusters_all["eid"] == eid)
        & (clusters_all["pid"] == pid)
        & (clusters_all["label"] >= MIN_QC)
    ]
    return set(as_str_array(hit["uuids"]))


def load_probe_units(one, pid, pname, eid, clusters_all):
    log(f"  loading spikes {pname} pid={pid}")
    loader = SpikeSortingLoader(pid=pid, one=one, pname=pname, eid=eid)
    spikes, clusters, channels = loader.load_spike_sorting(
        revision=SPIKE_REVISION, good_units=True
    )
    if not channels:
        spikes, clusters, channels = loader.load_spike_sorting(revision=SPIKE_REVISION)
    chan_tbl = channel_table(channels)
    keep = bwm_unit_uuids(clusters_all, eid, pid)
    if not spikes:
        log(f"    no spikes; BWM aggregate units={len(keep)}")
        return {
            "pid": pid,
            "pname": pname,
            "units": empty_units(),
            "spikes": {"times": np.array([], dtype=float), "clusters": np.array([], dtype=np.int32)},
        }
    labeled = SpikeSortingLoader.merge_clusters(spikes, clusters, channels).to_df()
    iok = (
        labeled["label"] >= MIN_QC
        if "label" in labeled.columns
        else np.ones(len(labeled), dtype=bool)
    )
    good = labeled[iok]
    spike_idx, ib = ismember(np.asarray(spikes["clusters"]), good.index.to_numpy())
    good = good.reset_index(drop=True)
    qc_spikes = {
        "times": np.asarray(spikes["times"], dtype=float)[spike_idx],
        "clusters": np.arange(len(good), dtype=np.int32)[ib],
    }
    log(f"    QC>={MIN_QC} after merge: {len(good)}; BWM aggregate uuids: {len(keep)}")
    spikes, clusters_df = filter_to_bwm_units(qc_spikes, good, keep)
    units = unit_records(clusters_df, chan_tbl) if len(clusters_df) else empty_units()
    order = np.argsort(np.asarray(spikes["times"], dtype=float), kind="stable")
    log(f"    BWM units={len(units['uuids'])} spikes={len(order)}")
    return {
        "pid": pid,
        "pname": pname,
        "units": units,
        "spikes": {
            "times": np.asarray(spikes["times"], dtype=float)[order],
            "clusters": np.asarray(spikes["clusters"], dtype=np.int32)[order],
        },
    }


def concat_session_units(probes):
    rows = []
    times = []
    uids = []
    offset = 0
    for probe in probes:
        units = probe["units"]
        n = len(units["uuids"])
        for i in range(n):
            rows.append(
                {
                    "unit_idx": offset + i,
                    "uuid": str(units["uuids"][i]),
                    "pid": probe["pid"],
                    "probe_name": probe["pname"],
                    "cluster_id": int(units["cluster_id"][i]),
                    "channel": int(units["channel"][i]),
                    "brain_area": str(units["acronym"][i]),
                    "atlas_id": int(units["atlas_id"][i]),
                    "allen_x": float(units["x"][i]),
                    "allen_y": float(units["y"][i]),
                    "allen_z": float(units["z"][i]),
                    "allen_x_um": float(units["x"][i]) * 1e6,
                    "allen_y_um": float(units["y"][i]) * 1e6,
                    "allen_z_um": float(units["z"][i]) * 1e6,
                    "axial_um": float(units["axial_um"][i]),
                    "lateral_um": float(units["lateral_um"][i]),
                }
            )
        spk = probe["spikes"]
        if spk["times"].size:
            times.append(np.asarray(spk["times"], dtype=float))
            uids.append(np.asarray(spk["clusters"], dtype=np.int32) + offset)
        offset += n
    units_df = pd.DataFrame(rows)
    if not times:
        return (
            units_df,
            np.array([], dtype=float),
            np.array([], dtype=np.int32),
        )
    all_times = np.concatenate(times)
    all_uids = np.concatenate(uids)
    order = np.argsort(all_times, kind="stable")
    return units_df, all_times[order], all_uids[order]


def window_spikes(times, uids, t0, t1):
    if times.size == 0:
        return empty_f32(), np.array([], dtype=np.int32)
    lo, hi = np.searchsorted(times, [t0, t1], side="left")
    if lo >= hi:
        return empty_f32(), np.array([], dtype=np.int32)
    return f32(times[lo:hi] - t0), i32(uids[lo:hi])


def session_pickle_path(eid):
    return SESS_DIR / f"{eid}.pkl"


def session_ok_path(eid):
    return SESS_DIR / f"{eid}.ok.json"


def session_is_complete(eid):
    ok_path = session_ok_path(eid)
    pkl_path = session_pickle_path(eid)
    if not ok_path.exists() or not pkl_path.exists():
        return False
    try:
        marker = json.loads(ok_path.read_text())
        return marker.get("version") == CACHE_VERSION and marker.get("eid") == eid
    except Exception:
        return False


def trial_scalars(rec):
    return {key: rec[key] for key in SCALAR_FIELDS}


def write_session(payload):
    SESS_DIR.mkdir(parents=True, exist_ok=True)
    eid = payload["eid"]
    pkl_path = session_pickle_path(eid)
    tmp_path = pkl_path.with_suffix(".pkl.tmp")
    with tmp_path.open("wb") as fh:
        pickle.dump(payload, fh, protocol=pickle.HIGHEST_PROTOCOL)
    tmp_path.replace(pkl_path)
    units_df = payload["units"]
    units_df.to_parquet(SESS_DIR / f"{eid}_units.parquet", index=False)
    scalars = pd.DataFrame([trial_scalars(rec) for rec in payload["trials"]])
    scalars.to_parquet(SESS_DIR / f"{eid}_trials.parquet", index=False)
    session_ok_path(eid).write_text(
        json.dumps(
            {
                "version": CACHE_VERSION,
                "eid": eid,
                "n_trials": payload["n_trials"],
                "n_units": payload["n_units"],
                "written_at": utc_now(),
            }
        )
    )
    return pkl_path


def append_progress(row):
    with PROGRESS_PATH.open("a") as fh:
        fh.write(json.dumps(row) + "\n")


def rebuild_indexes():
    trial_parts = sorted(SESS_DIR.glob("*_trials.parquet"))
    unit_parts = sorted(SESS_DIR.glob("*_units.parquet"))
    if trial_parts:
        trials = pd.concat([pd.read_parquet(p) for p in trial_parts], ignore_index=True)
        trials.to_parquet(TRIALS_INDEX_PATH, index=False)
    else:
        trials = pd.DataFrame(columns=SCALAR_FIELDS)
        trials.to_parquet(TRIALS_INDEX_PATH, index=False)
    if unit_parts:
        units = pd.concat(
            [
                pd.read_parquet(p).assign(eid=p.name.replace("_units.parquet", ""))
                for p in unit_parts
            ],
            ignore_index=True,
        )
        units.to_parquet(UNITS_INDEX_PATH, index=False)
    else:
        units = pd.DataFrame()
        units.to_parquet(UNITS_INDEX_PATH, index=False)
    manifest = {
        "version": CACHE_VERSION,
        "built_at": utc_now(),
        "n_mice": int(trials["mouse_id"].nunique()) if len(trials) else 0,
        "n_sessions": int(trials["eid"].nunique()) if len(trials) else 0,
        "n_trials": int(len(trials)),
        "n_units": int(len(units)),
        "rt_range_s": [MIN_RT, MAX_RT],
        "unit_qc": f"clusters.pqt label >= {MIN_QC}",
        "spike_revision": SPIKE_REVISION,
        "paths": {
            "sessions": str(SESS_DIR),
            "trials": str(TRIALS_INDEX_PATH),
            "units": str(UNITS_INDEX_PATH),
        },
    }
    MANIFEST_PATH.write_text(json.dumps(manifest, indent=2))
    return manifest


def window_pose_xyz(kin, t0, t1):
    if kin is None:
        return empty_f32(), empty_f32(), empty_f32(), empty_f32()
    return window_xyz(kin["times"], kin["x"], kin["y"], kin["z"], t0, t1)


def process_session(one, session, all_trials, clusters_all):
    eid = session["eid"]
    subject = session["subject"]
    lab = session["lab"]
    date = session["date"]
    trials = session_trials(all_trials, eid)
    log(
        f"{subject} {date} eid={eid}: {len(trials)} BWM trials "
        f"with RT in [{MIN_RT}, {MAX_RT}] s"
    )

    sess_loader = SessionLoader(one=one, eid=eid)
    wheel_times, wheel_vel = load_wheel_arrays(sess_loader)
    dlc_pose = load_pose_copy(sess_loader, "dlc")
    lp_pose = load_pose_copy(sess_loader, "lightningPose")

    probes = []
    for pid, pname in zip(session["pids"], session["probe_names"]):
        try:
            probes.append(load_probe_units(one, pid, pname, eid, clusters_all))
        except Exception as exc:
            log(f"  FAILED spikes {pname} {pid}: {exc}")
            traceback.print_exc()
            probes.append(
                {
                    "pid": pid,
                    "pname": pname,
                    "units": empty_units(),
                    "spikes": {
                        "times": np.array([], dtype=float),
                        "clusters": np.array([], dtype=np.int32),
                    },
                }
            )

    units_df, spike_times, spike_uids = concat_session_units(probes)
    n_units = int(len(units_df))
    records = []
    for trial in trials.itertuples():
        t0 = float(trial.stimOn_times)
        t1 = float(trial.feedback_times)
        side = select_used_paw(dlc_pose, t0, t1)
        if side is None:
            side = select_used_paw(lp_pose, t0, t1)
        dlc_kin = paw_xyz_positions(dlc_pose, side)
        lp_kin = paw_xyz_positions(lp_pose, side)
        wheel_t, wheel_v = window_trace(wheel_times, wheel_vel, t0, t1)
        dlc_t, dlc_x, dlc_y, dlc_z = window_pose_xyz(dlc_kin, t0, t1)
        lp_t, lp_x, lp_y, lp_z = window_pose_xyz(lp_kin, t0, t1)
        spk_t, spk_u = window_spikes(spike_times, spike_uids, t0, t1)
        stim_side = trial.stim_side
        rec = {
            "mouse_id": subject,
            "lab": lab,
            "session_date": date,
            "eid": eid,
            "trial_index": int(trial.trial_index),
            "block_prior": float(trial.probabilityLeft),
            "stim_side": stim_side,
            "choice": map_choice(trial.choice),
            "choice_ibl": int(trial.choice) if pd.notna(trial.choice) else None,
            "brightness": stimulus_brightness(trial, stim_side),
            "actionTime": float(trial.actionTime),
            "trialDuration": float(trial.trialDuration),
            "reward": int(trial.feedbackType) if pd.notna(trial.feedbackType) else None,
            "outcome": map_outcome(trial.feedbackType),
            "rewardVolume": float(trial.rewardVolume) if pd.notna(trial.rewardVolume) else 0.0,
            "used_paw": side,
            "stimOn_times": t0,
            "feedback_times": t1,
            "firstMovement_times": float(trial.firstMovement_times),
            "wheel_t": wheel_t,
            "wheel_velocity": wheel_v,
            "dlc_t": dlc_t,
            "dlc_x": dlc_x,
            "dlc_y": dlc_y,
            "dlc_z": dlc_z,
            "lp_t": lp_t,
            "lp_x": lp_x,
            "lp_y": lp_y,
            "lp_z": lp_z,
            "spike_times": spk_t,
            "spike_unit_idx": spk_u,
            "n_wheel_samples": int(wheel_t.size),
            "n_dlc_samples": int(dlc_t.size),
            "n_lp_samples": int(lp_t.size),
            "n_spikes": int(spk_t.size),
            "n_units": n_units,
        }
        records.append(rec)

    payload = {
        "version": CACHE_VERSION,
        "eid": eid,
        "mouse_id": subject,
        "lab": lab,
        "session_date": date,
        "pids": list(session["pids"]),
        "probe_names": list(session["probe_names"]),
        "n_units": n_units,
        "n_trials": len(records),
        "units": units_df,
        "trials": records,
        "notes": {
            "wheel_velocity": "signed angular velocity from SessionLoader, rad/s",
            "paw_xyz": (
                "signed image-plane x/y in left-camera-equivalent pixels "
                "(right camera 640x512 multiplied by 2); z = x_near - x_far "
                "after that scale; likelihood >= 0.95; times relative to stimulus onset"
            ),
            "spike_times": "seconds relative to stimulus onset, through feedback/reward",
            "allen_xyz": "IBL/CCF metres (x=ML, y=AP, z=DV)",
            "trials": "bwm_include and actionTime in [0.08, 2.0] s",
            "units": "Brain-Wide Map paper units, clusters.pqt label >= 1",
        },
    }
    path = write_session(payload)
    log(f"  wrote {path.name}: {len(records)} trials, {n_units} units")
    return payload


def load_session(eid):
    path = session_pickle_path(eid)
    with path.open("rb") as fh:
        return pickle.load(fh)


def iter_trials():
    """Yield (units_df, trial_record) for every cached trial."""
    for path in sorted(SESS_DIR.glob("*.pkl")):
        if path.name.endswith(".tmp"):
            continue
        with path.open("rb") as fh:
            payload = pickle.load(fh)
        units = payload["units"]
        for rec in payload["trials"]:
            yield units, rec


def main():
    parser = argparse.ArgumentParser(description="Build BWM motor-cortex trial database")
    parser.add_argument("--eid", help="Process a single session eid")
    parser.add_argument("--rebuild", action="store_true", help="Recompute sessions even if cached")
    parser.add_argument("--max-sessions", type=int, default=None)
    parser.add_argument("--index-only", action="store_true", help="Rebuild parquet indexes only")
    args = parser.parse_args()

    DB_DIR.mkdir(parents=True, exist_ok=True)
    SESS_DIR.mkdir(parents=True, exist_ok=True)

    if args.index_only:
        manifest = rebuild_indexes()
        log(
            f"Index only: {manifest['n_mice']} mice, {manifest['n_sessions']} sessions, "
            f"{manifest['n_trials']} trials, {manifest['n_units']} units"
        )
        return

    sessions = load_cohort_sessions()
    if args.eid:
        sessions = sessions.loc[sessions["eid"] == args.eid].copy()
        if sessions.empty:
            raise SystemExit(f"eid {args.eid} is not in the motor-cortex cohort")
    if args.max_sessions is not None:
        sessions = sessions.iloc[: args.max_sessions].copy()

    log(f"Cohort: {sessions['subject'].nunique()} mice, {len(sessions)} sessions")
    log("Loading frozen BWM trials.pqt and clusters.pqt")
    all_trials = pd.read_parquet(TRIALS_PQT)
    all_trials["eid"] = all_trials["eid"].astype(str)
    clusters_all = pd.read_parquet(CLUSTERS_PQT)
    clusters_all["eid"] = clusters_all["eid"].astype(str)
    clusters_all["pid"] = clusters_all["pid"].astype(str)

    one = None
    n_ok = 0
    n_skip = 0
    n_fail = 0
    for i, session in enumerate(sessions.to_dict("records"), start=1):
        eid = session["eid"]
        log(f"[{i}/{len(sessions)}] {session['subject']} {session['date']}")
        if not args.rebuild and session_is_complete(eid):
            log("  skip, already cached")
            n_skip += 1
            continue
        if one is None:
            one = connect_one()
        try:
            payload = process_session(one, session, all_trials, clusters_all)
            append_progress(
                {
                    "time": utc_now(),
                    "eid": eid,
                    "subject": session["subject"],
                    "date": session["date"],
                    "status": "ok",
                    "n_trials": payload["n_trials"],
                    "n_units": payload["n_units"],
                }
            )
            n_ok += 1
            rebuild_indexes()
        except Exception as exc:
            n_fail += 1
            log(f"  FAILED: {exc}")
            traceback.print_exc()
            append_progress(
                {
                    "time": utc_now(),
                    "eid": eid,
                    "subject": session["subject"],
                    "date": session["date"],
                    "status": "error",
                    "error": str(exc),
                }
            )

    manifest = rebuild_indexes()
    log(
        f"Done. processed={n_ok} skipped={n_skip} failed={n_fail}; "
        f"index {manifest['n_mice']} mice, {manifest['n_sessions']} sessions, "
        f"{manifest['n_trials']} trials, {manifest['n_units']} units"
    )
    if manifest["n_mice"] != EXPECTED_N_MICE or manifest["n_sessions"] != EXPECTED_N_SESSIONS:
        log(
            "Warning: index does not yet contain the full 39 mice / 67 sessions. "
            "Re-run fullData.py to resume remaining sessions."
        )


if __name__ == "__main__":
    main()
