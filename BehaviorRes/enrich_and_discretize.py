"""Enrich Database / DatasetFiltered with paw velocity, speed, and motion states,
then write 20 ms BWM-style discretized copies.

Native-rate additions (same timestamps as stored DLC / Lightning Pose positions)
--------------------------------------------------------------------
- dlc_vx, dlc_vy, dlc_vz, dlc_speed and the LP equivalents
- dlc_paw_state, lp_paw_state: still / coupled / decoupled from the
  ZFM-01576_video.py heuristic

Discretized copies
------------------
BehaviorRes/DatabaseDiscretized
BehaviorRes/DatasetFilteredDiscretized

Exact 20 ms bins covering [0, floor(T/0.02)*0.02) relative to stimulus
onset. Leftover time shorter than one bin is dropped so the sampling
interval is exactly 20 ms on every trial (the BWM wheel decoder used a
fixed 1.2 s window that already divided evenly; our trialDuration varies).
"""

from __future__ import annotations

import json
import pickle
import shutil
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.interpolate import interp1d
from scipy.ndimage import gaussian_filter1d

ROOT = Path(__file__).resolve().parent
DESKTOP = ROOT.parent
BEH = ROOT
DB_DIR = BEH / "Database"
FILT_DIR = BEH / "DatasetFiltered"
DB_DISC_DIR = BEH / "DatabaseDiscretized"
FILT_DISC_DIR = BEH / "DatasetFilteredDiscretized"

BINSIZE = 0.02
SMOOTH_SIGMA_S = 0.02
ENRICH_VERSION = 2
DISC_VERSION = 1

STILL, COUPLED, DECOUPLED = "still", "coupled", "decoupled"
STATE_TO_CODE = {STILL: 0, COUPLED: 1, DECOUPLED: 2}
CODE_TO_STATE = np.array([STILL, COUPLED, DECOUPLED], dtype=object)

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

DISC_SCALAR_FIELDS = SCALAR_FIELDS + [
    "n_bins",
    "leftover_s",
    "n_dlc_state_bins",
    "n_lp_state_bins",
]


def utc_now():
    return datetime.now(timezone.utc).isoformat()


def log(msg):
    print(f"[{datetime.now().strftime('%H:%M:%S')}] {msg}", flush=True)


def f32(values):
    return np.asarray(values, dtype=np.float32)


def empty_f32():
    return np.array([], dtype=np.float32)


def contiguous_segments(mask):
    idx = np.flatnonzero(mask)
    if idx.size == 0:
        return []
    breaks = np.where(np.diff(idx) > 1)[0] + 1
    return np.split(idx, breaks)


def smooth_and_velocity(times, values, sigma_s=SMOOTH_SIGMA_S):
    vel = np.full(np.shape(values), np.nan, dtype=float)
    finite = np.isfinite(times) & np.isfinite(values)
    for seg in contiguous_segments(finite):
        if len(seg) < 3:
            continue
        t = np.asarray(times[seg], dtype=float)
        y = np.asarray(values[seg], dtype=float)
        dt = np.median(np.diff(t))
        if not np.isfinite(dt) or dt <= 0:
            continue
        sigma = max(sigma_s / dt, 0.5)
        y_s = gaussian_filter1d(y, sigma=sigma, mode="nearest")
        vel[seg] = np.gradient(y_s, t)
    return vel


def interpolate_to(times_src, values_src, times_dst, extrapolate=False):
    finite = np.isfinite(times_src) & np.isfinite(values_src)
    if finite.sum() < 2:
        return np.full(np.shape(times_dst), np.nan)
    t = np.asarray(times_src, dtype=float)[finite]
    v = np.asarray(values_src, dtype=float)[finite]
    order = np.argsort(t)
    t, v = t[order], v[order]
    uniq = np.concatenate(([True], np.diff(t) > 0))
    t, v = t[uniq], v[uniq]
    if t.size < 2:
        return np.full(np.shape(times_dst), np.nan)
    fill = "extrapolate" if extrapolate else np.nan
    interpolator = interp1d(t, v, kind="linear", bounds_error=False, fill_value=fill)
    out = interpolator(times_dst)
    if not extrapolate:
        out[(times_dst < t[0]) | (times_dst > t[-1])] = np.nan
    return out


RIGHT_CAMERA_SCALE = 2.0
MIN_COUPLED_SAMPLES = 5
NORM_SCALAR_FIELDS = ["dlc_speed_norm_ref", "lp_speed_norm_ref"]


def paw_speed(vx, vy, vz=None):
    """2D image-plane speed hypot(vx, vy). vz is ignored (kept for call-site compat)."""
    vx = np.asarray(vx, dtype=float)
    vy = np.asarray(vy, dtype=float)
    return np.hypot(vx, vy)


def _same_len(*arrs):
    return all(len(a) == len(arrs[0]) for a in arrs)


def rescale_right_camera_trial(rec):
    """Put right-camera x/y (and their contribution to z) in left-camera pixels.

    IBL left camera is 1280x1024; right is 640x512. used_paw names the near
    camera. Algebra for z uses z = x_near - x_far on the stored (unscaled) traces.
    Does not recompute STILL/COUPLED/DECOUPLED.
    """
    paw = rec.get("used_paw")
    for prefix in ("dlc", "lp"):
        x = np.asarray(rec.get(f"{prefix}_x", []), dtype=float)
        y = np.asarray(rec.get(f"{prefix}_y", []), dtype=float)
        z = np.asarray(rec.get(f"{prefix}_z", []), dtype=float)
        vx = np.asarray(rec.get(f"{prefix}_vx", []), dtype=float)
        vy = np.asarray(rec.get(f"{prefix}_vy", []), dtype=float)
        vz = np.asarray(rec.get(f"{prefix}_vz", []), dtype=float)
        if x.size == 0:
            rec[f"{prefix}_speed"] = empty_f32()
            continue
        if paw == "right" and _same_len(x, y):
            if z.size == x.size:
                rec[f"{prefix}_z"] = f32(x + z)
            if vx.size == x.size and vz.size == x.size:
                rec[f"{prefix}_vz"] = f32(vx + vz)
            rec[f"{prefix}_x"] = f32(x * RIGHT_CAMERA_SCALE)
            rec[f"{prefix}_y"] = f32(y * RIGHT_CAMERA_SCALE)
            if vx.size == x.size:
                rec[f"{prefix}_vx"] = f32(vx * RIGHT_CAMERA_SCALE)
            if vy.size == y.size:
                rec[f"{prefix}_vy"] = f32(vy * RIGHT_CAMERA_SCALE)
        elif paw == "left" and z.size == x.size:
            rec[f"{prefix}_z"] = f32(RIGHT_CAMERA_SCALE * z - x)
            if vx.size == x.size and vz.size == z.size:
                rec[f"{prefix}_vz"] = f32(RIGHT_CAMERA_SCALE * vz - vx)
        vx_n = np.asarray(rec.get(f"{prefix}_vx", vx), dtype=float)
        vy_n = np.asarray(rec.get(f"{prefix}_vy", vy), dtype=float)
        rec[f"{prefix}_speed"] = f32(paw_speed(vx_n, vy_n))
    return rec


def _coupled_values(trials, prefix, paw=None):
    chunks = []
    for rec in trials:
        if paw is not None and rec.get("used_paw") != paw:
            continue
        speed = np.asarray(rec.get(f"{prefix}_speed", []), dtype=float)
        state = np.asarray(rec.get(f"{prefix}_paw_state", []), dtype=object)
        if speed.size == 0 or state.size == 0:
            continue
        n = min(speed.size, state.size)
        mask = (state[:n].astype(str) == COUPLED) & np.isfinite(speed[:n])
        if np.any(mask):
            chunks.append(speed[:n][mask])
    if not chunks:
        return np.array([], dtype=float)
    return np.concatenate(chunks)


def _median_or_nan(values, min_n=MIN_COUPLED_SAMPLES):
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if values.size < min_n:
        return np.nan
    med = float(np.median(values))
    if not np.isfinite(med) or med <= 0:
        return np.nan
    return med


def session_coupled_speed_refs(trials):
    """Typical coupled speed per tracker and near-camera (used_paw).

    Fallback: that tracker, all paws; then all finite speeds for that tracker.
    """
    refs = {}
    for prefix in ("dlc", "lp"):
        per_paw = {}
        all_coupled = _coupled_values(trials, prefix, paw=None)
        all_coupled_med = _median_or_nan(all_coupled)
        all_speed = []
        for rec in trials:
            sp = np.asarray(rec.get(f"{prefix}_speed", []), dtype=float)
            if sp.size:
                all_speed.append(sp[np.isfinite(sp)])
        all_speed_med = _median_or_nan(
            np.concatenate(all_speed) if all_speed else np.array([]), min_n=1
        )
        for paw in ("left", "right"):
            med = _median_or_nan(_coupled_values(trials, prefix, paw=paw))
            if not np.isfinite(med):
                med = all_coupled_med
            if not np.isfinite(med):
                med = all_speed_med
            per_paw[paw] = med
        per_paw[None] = all_coupled_med if np.isfinite(all_coupled_med) else all_speed_med
        refs[prefix] = per_paw
    return refs


def apply_speed_normalized(trials, refs=None):
    """speedNormalized = 2D speed / session typical coupled speed (per tracker, paw)."""
    if refs is None:
        refs = session_coupled_speed_refs(trials)
    for rec in trials:
        paw = rec.get("used_paw")
        for prefix in ("dlc", "lp"):
            speed = np.asarray(rec.get(f"{prefix}_speed", []), dtype=float)
            ref = refs[prefix].get(paw, refs[prefix].get(None))
            rec[f"{prefix}_speed_norm_ref"] = (
                float(ref) if ref is not None and np.isfinite(ref) and ref > 0 else None
            )
            if speed.size == 0:
                rec[f"{prefix}_speedNormalized"] = empty_f32()
                continue
            if rec[f"{prefix}_speed_norm_ref"] is None:
                rec[f"{prefix}_speedNormalized"] = np.full(speed.shape, np.nan, dtype=np.float32)
            else:
                rec[f"{prefix}_speedNormalized"] = f32(speed / rec[f"{prefix}_speed_norm_ref"])
        lp = np.asarray(rec.get("lp_speedNormalized", []), dtype=float)
        dlc = np.asarray(rec.get("dlc_speedNormalized", []), dtype=float)
        if lp.size:
            rec["speedNormalized"] = rec["lp_speedNormalized"]
        elif dlc.size:
            rec["speedNormalized"] = rec["dlc_speedNormalized"]
        else:
            rec["speedNormalized"] = empty_f32()
    return refs


def contact_height_baseline(y, times):
    y = np.asarray(y, dtype=float)
    times = np.asarray(times, dtype=float)
    base = np.full_like(y, np.nan)
    if y.size == 0:
        return base
    dt = float(np.nanmedian(np.diff(times))) if times.size > 1 else 0.01
    if not np.isfinite(dt) or dt <= 0:
        dt = 0.01
    a_contact = 1.0 - np.exp(-dt / 0.040)
    a_lift = 1.0 - np.exp(-dt / 0.280)
    prev = np.nan
    for i, yi in enumerate(y):
        if not np.isfinite(yi):
            base[i] = prev
            continue
        if not np.isfinite(prev):
            prev = yi
            base[i] = prev
            continue
        alpha = a_contact if yi >= prev else a_lift
        prev = (1.0 - alpha) * prev + alpha * yi
        base[i] = prev
    return base


def majority_smooth(codes, win):
    codes = np.asarray(codes, dtype=int)
    n = codes.size
    if n == 0 or win < 3:
        return codes.copy()
    half = win // 2
    out = np.empty(n, dtype=int)
    for i in range(n):
        window = codes[max(0, i - half) : min(n, i + half + 1)]
        counts = np.bincount(window, minlength=3)
        out[i] = int(np.argmax(counts))
    return out


def enforce_min_dwell(times, codes, min_s=0.045):
    times = np.asarray(times, dtype=float)
    codes = np.asarray(codes, dtype=int).copy()
    n = codes.size
    if n == 0:
        return codes
    i = 0
    while i < n:
        j = i + 1
        while j < n and codes[j] == codes[i]:
            j += 1
        t_end = times[j] if j < n else times[-1]
        if (t_end - times[i]) < min_s:
            neighbor = codes[i - 1] if i > 0 else (codes[j] if j < n else codes[i])
            codes[i:j] = neighbor
        i = j
    return codes


def segment_paw_motion(t, x, y, vx, vy, omega, first_move_s=None):
    t = np.asarray(t, dtype=float)
    y = np.asarray(y, dtype=float)
    vx = np.asarray(vx, dtype=float)
    vy = np.asarray(vy, dtype=float)
    omega = np.asarray(omega, dtype=float)
    n = t.size
    labels = np.full(n, STILL, dtype=object)
    if n == 0:
        return labels

    speed = np.hypot(vx, vy)
    speed[~np.isfinite(speed)] = 0.0
    omega_abs = np.abs(omega)
    omega_abs[~np.isfinite(omega_abs)] = 0.0

    p95_speed = float(np.nanpercentile(speed, 95)) if speed.size else 0.0
    p95_omega = float(np.nanpercentile(omega_abs, 95)) if omega_abs.size else 0.0
    v_still = float(np.clip(max(0.12 * p95_speed, 140.0), 140.0, 320.0))
    omega_move = float(np.clip(max(0.10 * p95_omega, 0.30), 0.30, 1.00))

    rest_horizon = 0.080
    if first_move_s is not None and np.isfinite(first_move_s) and first_move_s > 0:
        rest_horizon = float(np.clip(first_move_s, 0.040, 0.120))
    rest = y[t <= rest_horizon]
    if rest.size < 3:
        rest = y[: max(1, min(8, n))]
    y_rest = float(np.nanmedian(rest)) if np.any(np.isfinite(rest)) else float(np.nanmedian(y))
    y_for_base = y.copy()
    if np.isfinite(y_rest):
        y_for_base[~np.isfinite(y_for_base)] = y_rest
        y_for_base[0] = y_rest
    baseline = contact_height_baseline(y_for_base, t)
    lift = baseline - y
    lift[~np.isfinite(lift)] = 0.0
    early_std = float(np.nanstd(rest)) if np.any(np.isfinite(rest)) else 3.0
    if not np.isfinite(early_std) or early_std < 1.0:
        early_std = 3.0
    lift_thr = float(max(14.0, 4.5 * early_std))

    moving = speed >= v_still
    wheel_moving = omega_abs >= omega_move
    lifted = lift >= lift_thr
    vert_lift = (
        moving
        & (np.abs(vy) >= 0.80 * np.maximum(speed, 1.0))
        & (lift > 0.5 * lift_thr)
    )

    decoupled = lifted | (moving & ~wheel_moving) | vert_lift
    coupled = moving & wheel_moving & ~lifted
    codes = np.zeros(n, dtype=int)
    codes[coupled] = 1
    codes[decoupled] = 2

    dt = float(np.nanmedian(np.diff(t))) if n > 1 else 0.016
    if not np.isfinite(dt) or dt <= 0:
        dt = 0.016
    win = int(round(0.060 / dt))
    if win % 2 == 0:
        win += 1
    win = max(3, min(win, n if n % 2 == 1 else n - 1))
    if n >= 3 and win >= 3:
        codes = majority_smooth(codes, win)
    codes = enforce_min_dwell(t, codes, min_s=0.045)
    return CODE_TO_STATE[np.clip(codes, 0, 2)]


def tracker_kinematics(t, x, y, z):
    t = np.asarray(t, dtype=float)
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    z = np.asarray(z, dtype=float)
    vx = smooth_and_velocity(t, x)
    vy = smooth_and_velocity(t, y)
    vz = smooth_and_velocity(t, z)
    speed = paw_speed(vx, vy, vz)
    return vx, vy, vz, speed


def tracker_states(t, x, y, vx, vy, wheel_t, wheel_v, first_move_s):
    if t.size < 3:
        return np.full(t.shape, STILL, dtype=object)
    omega = interpolate_to(wheel_t, wheel_v, t, extrapolate=False)
    return segment_paw_motion(t, x, y, vx, vy, omega, first_move_s=first_move_s)


def enrich_trial(rec):
    first_move_s = float(rec["actionTime"]) if rec.get("actionTime") is not None else None
    wheel_t = np.asarray(rec.get("wheel_t", []), dtype=float)
    wheel_v = np.asarray(rec.get("wheel_velocity", []), dtype=float)

    dlc_t = np.asarray(rec.get("dlc_t", []), dtype=float)
    if dlc_t.size:
        vx, vy, vz, speed = tracker_kinematics(
            dlc_t, rec["dlc_x"], rec["dlc_y"], rec["dlc_z"]
        )
        rec["dlc_vx"] = f32(vx)
        rec["dlc_vy"] = f32(vy)
        rec["dlc_vz"] = f32(vz)
        rec["dlc_speed"] = f32(speed)
        rec["dlc_paw_state"] = tracker_states(
            dlc_t, rec["dlc_x"], rec["dlc_y"], vx, vy, wheel_t, wheel_v, first_move_s
        )
    else:
        rec["dlc_vx"] = empty_f32()
        rec["dlc_vy"] = empty_f32()
        rec["dlc_vz"] = empty_f32()
        rec["dlc_speed"] = empty_f32()
        rec["dlc_paw_state"] = np.array([], dtype=object)

    lp_t = np.asarray(rec.get("lp_t", []), dtype=float)
    if lp_t.size:
        vx, vy, vz, speed = tracker_kinematics(
            lp_t, rec["lp_x"], rec["lp_y"], rec["lp_z"]
        )
        rec["lp_vx"] = f32(vx)
        rec["lp_vy"] = f32(vy)
        rec["lp_vz"] = f32(vz)
        rec["lp_speed"] = f32(speed)
        rec["lp_paw_state"] = tracker_states(
            lp_t, rec["lp_x"], rec["lp_y"], vx, vy, wheel_t, wheel_v, first_move_s
        )
    else:
        rec["lp_vx"] = empty_f32()
        rec["lp_vy"] = empty_f32()
        rec["lp_vz"] = empty_f32()
        rec["lp_speed"] = empty_f32()
        rec["lp_paw_state"] = np.array([], dtype=object)
    return rec


def bin_grid(trial_duration):
    T = float(trial_duration)
    if not np.isfinite(T) or T <= 0:
        n_bins = 0
    else:
        n_bins = int(np.floor(T / BINSIZE))
    leftover = T - n_bins * BINSIZE if np.isfinite(T) else np.nan
    t_left = np.arange(n_bins, dtype=np.float64) * BINSIZE
    t_right = t_left + BINSIZE
    return n_bins, leftover, f32(t_left), f32(t_right)


def interp_at_right_edges(t, v, t_right):
    """BWM decoding: sample the continuous signal at the right edge of each bin.

    Unlike the paper we do not extrapolate outside the observed support.
    """
    if t_right.size == 0:
        return empty_f32()
    return f32(interpolate_to(t, v, t_right, extrapolate=False))


def bin_categorical(t, labels, t_left, t_right):
    """Majority label inside each left-closed, right-open bin.

    Ties are broken by the sample closest to the right edge. Empty bins are -1.
    """
    n_bins = t_left.size
    out = np.full(n_bins, -1, dtype=np.int8)
    if t.size == 0 or n_bins == 0:
        return out
    t = np.asarray(t, dtype=float)
    codes = np.array([STATE_TO_CODE.get(str(s), -1) for s in labels], dtype=int)
    valid = np.isfinite(t) & (codes >= 0)
    t = t[valid]
    codes = codes[valid]
    if t.size == 0:
        return out
    idx = np.floor(t / BINSIZE).astype(int)
    keep = (idx >= 0) & (idx < n_bins)
    idx = idx[keep]
    codes = codes[keep]
    t = t[keep]
    for i in range(n_bins):
        m = idx == i
        if not np.any(m):
            continue
        c = codes[m]
        counts = np.bincount(c, minlength=3)
        winners = np.flatnonzero(counts == counts.max())
        if winners.size == 1:
            out[i] = np.int8(winners[0])
        else:
            j = int(np.argmin(np.abs(t[m] - t_right[i])))
            out[i] = np.int8(c[j])
    return out


def bin_spikes(spike_times, spike_unit_idx, n_units, n_bins):
    counts = np.zeros((n_bins, n_units), dtype=np.uint16)
    if n_bins == 0 or n_units == 0 or len(spike_times) == 0:
        return counts
    times = np.asarray(spike_times, dtype=float)
    uids = np.asarray(spike_unit_idx, dtype=int)
    bin_idx = np.floor(times / BINSIZE).astype(int)
    ok = (
        np.isfinite(times)
        & (bin_idx >= 0)
        & (bin_idx < n_bins)
        & (uids >= 0)
        & (uids < n_units)
    )
    if not np.any(ok):
        return counts
    np.add.at(counts, (bin_idx[ok], uids[ok]), 1)
    return counts


def discretize_trial(rec):
    n_bins, leftover, t_left, t_right = bin_grid(rec["trialDuration"])
    n_units = int(rec.get("n_units") or 0)
    out = dict(rec)
    out["t_bin_left"] = t_left
    out["t_bin_right"] = t_right
    out["n_bins"] = int(n_bins)
    out["leftover_s"] = float(leftover) if np.isfinite(leftover) else None

    out["wheel_velocity"] = interp_at_right_edges(
        rec.get("wheel_t", []), rec.get("wheel_velocity", []), t_right
    )
    out["wheel_t"] = t_right.copy()

    for prefix in ("dlc", "lp"):
        t = rec.get(f"{prefix}_t", [])
        for axis in ("x", "y", "z", "vx", "vy", "vz", "speed"):
            key = f"{prefix}_{axis}"
            out[key] = interp_at_right_edges(t, rec.get(key, []), t_right)
        vx_b = out.get(f"{prefix}_vx", empty_f32())
        vy_b = out.get(f"{prefix}_vy", empty_f32())
        out[f"{prefix}_speed"] = f32(paw_speed(vx_b, vy_b))
        out[f"{prefix}_t"] = t_right.copy()
        states = rec.get(f"{prefix}_paw_state", np.array([], dtype=object))
        out[f"{prefix}_paw_state_code"] = bin_categorical(t, states, t_left, t_right)
        labels = np.array(["none"] * n_bins, dtype=object)
        code = out[f"{prefix}_paw_state_code"]
        good = code >= 0
        labels[good] = CODE_TO_STATE[code[good]]
        out[f"{prefix}_paw_state"] = labels

    out["spike_counts"] = bin_spikes(
        rec.get("spike_times", []),
        rec.get("spike_unit_idx", []),
        n_units,
        n_bins,
    )
    out["n_wheel_samples"] = int(n_bins)
    out["n_dlc_samples"] = int(n_bins)
    out["n_lp_samples"] = int(n_bins)
    out["n_spikes"] = int(out["spike_counts"].sum())
    out["n_dlc_state_bins"] = int(np.sum(out["dlc_paw_state_code"] >= 0))
    out["n_lp_state_bins"] = int(np.sum(out["lp_paw_state_code"] >= 0))
    return out


def session_has_pose(trials):
    return any(
        int(rec.get("n_dlc_samples") or 0) > 0 or int(rec.get("n_lp_samples") or 0) > 0
        for rec in trials
    )


def trial_scalars(rec, fields):
    return {key: rec.get(key) for key in fields}


def write_session(folder, payload, scalar_fields):
    sess_dir = folder / "sessions"
    sess_dir.mkdir(parents=True, exist_ok=True)
    eid = payload["eid"]
    pkl_path = sess_dir / f"{eid}.pkl"
    tmp = pkl_path.with_suffix(".pkl.tmp")
    with tmp.open("wb") as fh:
        pickle.dump(payload, fh, protocol=pickle.HIGHEST_PROTOCOL)
    tmp.replace(pkl_path)
    payload["units"].to_parquet(sess_dir / f"{eid}_units.parquet", index=False)
    pd.DataFrame([trial_scalars(rec, scalar_fields) for rec in payload["trials"]]).to_parquet(
        sess_dir / f"{eid}_trials.parquet", index=False
    )
    (sess_dir / f"{eid}.ok.json").write_text(
        json.dumps(
            {
                "eid": eid,
                "n_trials": payload["n_trials"],
                "n_units": payload["n_units"],
                "written_at": utc_now(),
                "enrich_version": payload.get("enrich_version"),
                "disc_version": payload.get("disc_version"),
            }
        )
    )
    return pkl_path


def rebuild_indexes(folder, scalar_fields):
    sess_dir = folder / "sessions"
    trial_parts = sorted(sess_dir.glob("*_trials.parquet"))
    unit_parts = sorted(sess_dir.glob("*_units.parquet"))
    if trial_parts:
        trials = pd.concat([pd.read_parquet(p) for p in trial_parts], ignore_index=True)
    else:
        trials = pd.DataFrame(columns=scalar_fields)
    trials.to_parquet(folder / "trials.parquet", index=False)
    if unit_parts:
        units = pd.concat(
            [
                pd.read_parquet(p).assign(eid=p.name.replace("_units.parquet", ""))
                for p in unit_parts
            ],
            ignore_index=True,
        )
    else:
        units = pd.DataFrame()
    units.to_parquet(folder / "units.parquet", index=False)
    manifest = {
        "built_at": utc_now(),
        "n_mice": int(trials["mouse_id"].nunique()) if len(trials) else 0,
        "n_sessions": int(trials["eid"].nunique()) if len(trials) else 0,
        "n_trials": int(len(trials)),
        "n_units": int(len(units)),
        "binsize_s": BINSIZE if "n_bins" in trials.columns else None,
        "paths": {
            "sessions": str(sess_dir),
            "trials": str(folder / "trials.parquet"),
            "units": str(folder / "units.parquet"),
        },
    }
    (folder / "manifest.json").write_text(json.dumps(manifest, indent=2))
    return manifest


def alignment_notes():
    return {
        "binsize_s": BINSIZE,
        "time_origin": "stimulus onset",
        "window": "[0, floor(trialDuration/0.02)*0.02)",
        "leftover": "time after the last full 20 ms bin is dropped",
        "t_bin_left": "left edge of each bin; spike-count interval is [left, right)",
        "t_bin_right": (
            "right edge of each bin. Continuous wheel and paw series are linearly "
            "interpolated here so spikes in a bin fully precede the behavior sample, "
            "matching brainwidemap.decoding.functions.process_targets "
            "(interval_begs + binsize). Unlike the paper we do not extrapolate "
            "outside the observed timestamps (NaN instead)."
        ),
        "wheel_velocity": (
            "SessionLoader signed angular velocity (rad/s) interpolated linearly "
            "onto t_bin_right. This is the BWM decoding target resampling, not the "
            "encoding-GLM previous-value difference of wheel position."
        ),
        "paw_velocity": (
            "Native vx,vy,vz from Gaussian-smoothed position (sigma=20 ms) then "
            "np.gradient, same as ZFM-01576_behavior.py. Those traces are then "
            "linearly interpolated onto t_bin_right like wheel."
        ),
        "paw_speed": (
            "2D hypot(vx, vy) in left-camera-equivalent pixels/s. Right-camera "
            "x/y (640x512) are multiplied by 2 before velocity. vz is not in speed. "
            "speedNormalized = speed / session median coupled speed, per tracker "
            "and used_paw (near camera)."
        ),
        "paw_state": (
            "STILL/COUPLED/DECOUPLED computed at native camera times with the "
            "ZFM-01576_video.py heuristic, then majority-voted inside each 20 ms "
            "bin. Ties: label of the sample closest to the right edge. Empty: none/-1. "
            "The BWM paper had no paw-state target; this rule is new."
        ),
        "spikes": (
            "Per-unit counts in [t_left, t_right), stored as spike_counts with shape "
            "(n_bins, n_units). This is the BWM multi-bin decoding representation "
            "(bincount2D / get_spike_data_per_trial) on a regular 20 ms grid."
        ),
        "paper_difference": (
            "The BWM wheel decoder used a fixed window firstMovement-0.2 to +1.0 s "
            "(exactly 60 bins of 20 ms). Here the window is stimOn to reward and "
            "length varies, so we keep only complete 20 ms bins instead of stretching "
            "linspace(0.02, T, ceil(T/0.02)) which would make dt != 20 ms."
        ),
    }


def process():
    src_dir = DB_DIR / "sessions"
    pkls = sorted(p for p in src_dir.glob("*.pkl") if not p.name.endswith(".tmp"))
    if not pkls:
        raise SystemExit(f"No session pickles in {src_dir}")

    for folder in (FILT_DIR, DB_DISC_DIR, FILT_DISC_DIR):
        if folder.exists():
            shutil.rmtree(folder)
        (folder / "sessions").mkdir(parents=True)

    n_db = n_filt = 0
    for i, path in enumerate(pkls, start=1):
        with path.open("rb") as fh:
            payload = pickle.load(fh)
        eid = payload["eid"]
        log(f"[{i}/{len(pkls)}] {payload.get('mouse_id')} {payload.get('session_date')} {eid}")

        for rec in payload["trials"]:
            enrich_trial(rec)
        apply_speed_normalized(payload["trials"])
        payload["enrich_version"] = ENRICH_VERSION
        payload["n_trials"] = len(payload["trials"])
        write_session(DB_DIR, payload, SCALAR_FIELDS + NORM_SCALAR_FIELDS)
        n_db += 1

        disc = {
            **{k: v for k, v in payload.items() if k != "trials"},
            "trials": [discretize_trial(rec) for rec in payload["trials"]],
            "disc_version": DISC_VERSION,
            "alignment": alignment_notes(),
        }
        apply_speed_normalized(disc["trials"])
        disc["n_trials"] = len(disc["trials"])
        write_session(DB_DISC_DIR, disc, DISC_SCALAR_FIELDS + NORM_SCALAR_FIELDS)

        if session_has_pose(payload["trials"]):
            kept = [rec for rec in payload["trials"] if rec.get("reward") == 1]
            if kept:
                filt = {**{k: v for k, v in payload.items() if k != "trials"}, "trials": kept}
                filt["n_trials"] = len(kept)
                filt["filter"] = {
                    "require_pose_session": True,
                    "require_reward": True,
                    "source": str(DB_DIR),
                }
                write_session(FILT_DIR, filt, SCALAR_FIELDS + NORM_SCALAR_FIELDS)
                filt_disc = {
                    **{k: v for k, v in filt.items() if k != "trials"},
                    "trials": [discretize_trial(rec) for rec in kept],
                    "disc_version": DISC_VERSION,
                    "alignment": alignment_notes(),
                }
                apply_speed_normalized(filt_disc["trials"])
                filt_disc["n_trials"] = len(filt_disc["trials"])
                write_session(FILT_DISC_DIR, filt_disc, DISC_SCALAR_FIELDS + NORM_SCALAR_FIELDS)
                n_filt += 1

        log(f"  trials={payload['n_trials']} units={payload['n_units']}")

    extra = SCALAR_FIELDS + NORM_SCALAR_FIELDS
    extra_disc = DISC_SCALAR_FIELDS + NORM_SCALAR_FIELDS
    for folder, fields, name in (
        (DB_DIR, extra, "Database"),
        (FILT_DIR, extra, "DatasetFiltered"),
        (DB_DISC_DIR, extra_disc, "DatabaseDiscretized"),
        (FILT_DISC_DIR, extra_disc, "DatasetFilteredDiscretized"),
    ):
        man = rebuild_indexes(folder, fields)
        log(
            f"{name}: {man['n_mice']} mice, {man['n_sessions']} sessions, "
            f"{man['n_trials']} trials, {man['n_units']} units"
        )
    log(f"Wrote {n_db} Database sessions; {n_filt} pose+reward sessions into filtered copies")


if __name__ == "__main__":
    process()
