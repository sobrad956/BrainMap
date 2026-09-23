"""Behavior traces for IBL Brain-Wide Map mouse ZFM-01576, session 2020-12-03.

Data sources
------------
Session identity comes from the frozen BWM release table
(`paper-brain-wide-map/brainwidemap/fixtures/2023_12_bwm_release.csv`):
    subject ZFM-01576, date 2020-12-03
    eid 9e9c6fc0-4769-4d83-9ea4-b59a1230510e

Trial events, including the paper's `bwm_include` flag, come from
`aggregates/2024_Q2_IBL_et_al_BWM/trials.pqt` on OpenAlyx/AWS, stored locally as
`BehaviorRes/trials.pqt`.

Wheel and pose are downloaded through ONE from
https://openalyx.internationalbrainlab.org (public password `international`)
and cached by ONE under its local cache directory. ALF objects:

    alf/_ibl_wheel.position.npy
    alf/_ibl_wheel.timestamps.npy
    alf/_ibl_leftCamera.dlc.pqt
    alf/_ibl_rightCamera.dlc.pqt
    alf/_ibl_leftCamera.lightningPose.pqt
    alf/_ibl_rightCamera.lightningPose.pqt
    plus matching `*Camera.times.npy`

Confidence threshold
--------------------
DLC and Lightning Pose return (x, y, likelihood) per keypoint. Any sample with
likelihood < 0.95 is set to NaN before velocity is computed, so only points
with >= 95% tracker confidence are used. Those NaNs are not interpolated across
long gaps; velocity is computed only inside contiguous high-confidence segments.

Coordinates
-----------
IBL side cameras provide 2D image coordinates, not calibrated 3D pose.
    x : near-camera horizontal pixel position (approximately A-P)
    y : near-camera vertical pixel position (height)
    z : stereo disparity of the *same anatomical paw* in the two side cameras,
        z = x_near - x_far, used as a mediolateral proxy. This is not a
        millimetre-reconstructed 3D coordinate; this session has no body-camera
        paw tracking and no camera-calibration datasets on OpenAlyx.
"""

from pathlib import Path
import pickle

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from brainbox.behavior.dlc import likelihood_threshold
from brainbox.behavior.wheel import WHEEL_DIAMETER
from brainbox.io.one import SessionLoader
from one.api import ONE
from scipy.interpolate import interp1d
from scipy.ndimage import gaussian_filter1d

ROOT = Path(__file__).resolve().parent
DESKTOP = ROOT.parent
OUT_DIR = ROOT
CACHE_PATH = OUT_DIR / "ZFM-01576_behavior_cache.pkl"
TRIALS_PQT = ROOT / "trials.pqt"
CACHE_VERSION = 3

SUBJECT = "ZFM-01576"
DATE = "2020-12-03"
EID = "9e9c6fc0-4769-4d83-9ea4-b59a1230510e"
EXPECTED_N_TRIALS = 609
MIN_RT = 0.08
MAX_RT = 2.0
LIKELIHOOD_THR = 0.95
MIN_POSE_SAMPLES = 5
DIFF_DT = 0.01
SMOOTH_SIGMA_S = 0.02
WHEEL_RADIUS_CM = WHEEL_DIAMETER / 2.0

BLUE_LEFT = (0.13850039, 0.41331206, 0.74052025)
RED_RIGHT = (0.66080672, 0.21526712, 0.23069468)


def connect_one():
    return ONE(
        base_url="https://openalyx.internationalbrainlab.org",
        password="international",
    )


def load_bwm_trials():
    trials = pd.read_parquet(TRIALS_PQT)
    sess = trials.loc[trials["eid"] == EID].copy()
    bwm = sess.loc[sess["bwm_include"]].copy().reset_index(drop=True)
    if len(bwm) != EXPECTED_N_TRIALS:
        raise RuntimeError(
            f"Expected {EXPECTED_N_TRIALS} BWM trials for {SUBJECT} {DATE}, "
            f"found {len(bwm)}"
        )
    return bwm


def stimulus_side(row):
    left = row["contrastLeft"]
    right = row["contrastRight"]
    left_ok = pd.notna(left)
    right_ok = pd.notna(right)
    if left_ok and not right_ok:
        return "left"
    if right_ok and not left_ok:
        return "right"
    return None


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


def window_trace(times, values, t0, t1):
    mask = (times >= t0) & (times <= t1) & np.isfinite(times) & np.isfinite(values)
    if not np.any(mask):
        return np.array([]), np.array([])
    return times[mask] - t0, values[mask]


def camera_arrays(pose_df, feature):
    return (
        pose_df["times"].to_numpy(dtype=float),
        pose_df[f"{feature}_x"].to_numpy(dtype=float),
        pose_df[f"{feature}_y"].to_numpy(dtype=float),
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


def select_used_paw(pose, t0, t1):
    """Near-paw (`paw_r`) with larger movement: left camera = left paw."""
    left_t, left_x, left_y = camera_arrays(pose["leftCamera"], "paw_r")
    right_t, right_x, right_y = camera_arrays(pose["rightCamera"], "paw_r")
    left_speed = mean_speed(left_t, left_x, left_y, t0, t1)
    right_speed = mean_speed(right_t, right_x, right_y, t0, t1)
    if not np.isfinite(left_speed) and not np.isfinite(right_speed):
        return None
    if not np.isfinite(right_speed) or (
        np.isfinite(left_speed) and left_speed >= right_speed
    ):
        return "left"
    return "right"


def paw_xyz_velocity(pose, side):
    if side == "left":
        near = pose["leftCamera"]
        far = pose["rightCamera"]
        near_feat = "paw_r"
        far_feat = "paw_l"
    else:
        near = pose["rightCamera"]
        far = pose["leftCamera"]
        near_feat = "paw_r"
        far_feat = "paw_l"

    t, x, y = camera_arrays(near, near_feat)
    far_t, far_x, _far_y = camera_arrays(far, far_feat)
    z = x - interpolate_to(far_t, far_x, t)
    return {
        "times": t,
        "vx": smooth_and_velocity(t, x),
        "vy": smooth_and_velocity(t, y),
        "vz": smooth_and_velocity(t, z),
    }


def load_pose_copy(sess_loader, tracker):
    sess_loader.load_pose(
        likelihood_thr=0.0,
        views=["left", "right"],
        tracker=tracker,
    )
    pose = {}
    for name, df in sess_loader.pose.items():
        dlc = df.drop(columns=["times"]).copy()
        dlc = likelihood_threshold(dlc, LIKELIHOOD_THR)
        out = dlc.copy()
        out.insert(0, "times", df["times"].to_numpy())
        pose[name] = out
    return pose


def difference_trace(t_a, v_a, t_b, v_b, dt=DIFF_DT):
    if t_a.size < 2 or t_b.size < 2:
        return np.array([]), np.array([])
    tmax = min(float(np.nanmax(t_a)), float(np.nanmax(t_b)))
    if not np.isfinite(tmax) or tmax < 2 * dt:
        return np.array([]), np.array([])
    grid = np.arange(0.0, tmax + 0.5 * dt, dt)
    da = interpolate_to(t_a, v_a, grid)
    db = interpolate_to(t_b, v_b, grid)
    diff = da - db
    finite = np.isfinite(diff)
    if finite.sum() < 2:
        return np.array([]), np.array([])
    return grid[finite], diff[finite]


def pose_xy_available(kin, t0, t1):
    tx, vx = window_trace(kin["times"], kin["vx"], t0, t1)
    ty, vy = window_trace(kin["times"], kin["vy"], t0, t1)
    return tx.size >= MIN_POSE_SAMPLES and ty.size >= MIN_POSE_SAMPLES, tx, vx, ty, vy


def split_by_stim(records, t_key, v_key):
    left, right = [], []
    for rec in records:
        pair = (rec[t_key], rec[v_key])
        if rec["stim_side"] == "left":
            left.append(pair)
        elif rec["stim_side"] == "right":
            right.append(pair)
    return left, right


def overlay_panel(ax, traces, color, title):
    plotted = 0
    for rel_t, rel_v in traces:
        if rel_t.size < 2:
            continue
        ax.plot(rel_t, rel_v, color=color, alpha=0.12, lw=0.45, rasterized=True)
        plotted += 1
    ax.axhline(0.0, color="k", lw=0.6, alpha=0.5)
    ax.set_title(f"{title}\nn = {plotted}")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    return plotted


def save_stim_split_plot(left_traces, right_traces, ylabel, title, filename):
    fig, axes = plt.subplots(1, 2, figsize=(12.5, 4.8), sharey=True)
    overlay_panel(axes[0], left_traces, BLUE_LEFT, "Stimulus left")
    overlay_panel(axes[1], right_traces, RED_RIGHT, "Stimulus right")
    axes[0].set_xlabel("Time from stimulus onset (s)")
    axes[1].set_xlabel("Time from stimulus onset (s)")
    axes[0].set_ylabel(ylabel)
    fig.suptitle(title, y=1.02)
    fig.tight_layout()
    path = OUT_DIR / filename
    fig.savefig(path, dpi=220, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {path.name}")
    return path


def build_cache():
    bwm = load_bwm_trials()
    bwm["stim_side"] = bwm.apply(stimulus_side, axis=1)
    bwm["rt"] = bwm["firstMovement_times"] - bwm["stimOn_times"]

    pre = bwm.copy()
    pre = pre.loc[pre["feedbackType"] == 1]
    pre = pre.loc[(pre["rt"] >= MIN_RT) & (pre["rt"] <= MAX_RT)]
    pre = pre.loc[pre["stim_side"].isin(["left", "right"])]
    pre = pre.reset_index(drop=True)

    print(f"BWM paper trials: {len(bwm)}")
    print(f"After dropping error feedback: {int((bwm['feedbackType'] == 1).sum())}")
    print(
        f"After also requiring RT in [{MIN_RT}, {MAX_RT}] s and known stim side: "
        f"{len(pre)}"
    )

    one = connect_one()
    sess_loader = SessionLoader(one=one, eid=EID)
    sess_loader.load_wheel(fs=1000)
    wheel_times = sess_loader.wheel["times"].to_numpy(dtype=float)
    wheel_vel_cm = (
        sess_loader.wheel["velocity"].to_numpy(dtype=float) * WHEEL_RADIUS_CM
    )

    dlc_pose = load_pose_copy(sess_loader, "dlc")
    lp_pose = load_pose_copy(sess_loader, "lightningPose")

    records = []
    kept_rows = []
    for trial in pre.itertuples():
        t0 = trial.stimOn_times
        t1 = trial.feedback_times
        side = select_used_paw(dlc_pose, t0, t1)
        if side is None:
            continue
        dlc_kin = paw_xyz_velocity(dlc_pose, side)
        lp_kin = paw_xyz_velocity(lp_pose, side)
        dlc_ok, dlc_tx, dlc_vx, dlc_ty, dlc_vy = pose_xy_available(dlc_kin, t0, t1)
        lp_ok, lp_tx, lp_vx, lp_ty, lp_vy = pose_xy_available(lp_kin, t0, t1)
        if not (dlc_ok and lp_ok):
            continue

        wheel_t, wheel_v = window_trace(wheel_times, wheel_vel_cm, t0, t1)
        dlc_tz, dlc_vz = window_trace(dlc_kin["times"], dlc_kin["vz"], t0, t1)
        lp_tz, lp_vz = window_trace(lp_kin["times"], lp_kin["vz"], t0, t1)
        diff_tx, diff_vx = difference_trace(dlc_tx, dlc_vx, lp_tx, lp_vx)
        diff_ty, diff_vy = difference_trace(dlc_ty, dlc_vy, lp_ty, lp_vy)

        rec = {
            "stim_side": trial.stim_side,
            "choice": float(trial.choice),
            "used_paw": side,
            "stimOn_times": float(t0),
            "feedback_times": float(t1),
            "firstMovement_times": float(trial.firstMovement_times),
            "rt": float(trial.rt),
            "wheel_t": wheel_t,
            "wheel_v": wheel_v,
            "dlc_tx": dlc_tx,
            "dlc_vx": dlc_vx,
            "dlc_ty": dlc_ty,
            "dlc_vy": dlc_vy,
            "dlc_tz": dlc_tz,
            "dlc_vz": dlc_vz,
            "lp_tx": lp_tx,
            "lp_vx": lp_vx,
            "lp_ty": lp_ty,
            "lp_vy": lp_vy,
            "lp_tz": lp_tz,
            "lp_vz": lp_vz,
            "diff_tx": diff_tx,
            "diff_vx": diff_vx,
            "diff_ty": diff_ty,
            "diff_vy": diff_vy,
        }
        records.append(rec)
        kept_rows.append(trial.Index)

    kept = pre.loc[kept_rows].reset_index(drop=True)
    print(
        f"After requiring both DLC and Lightning Pose paw traces: {len(kept)} "
        f"(left stim {int((kept['stim_side'] == 'left').sum())}, "
        f"right stim {int((kept['stim_side'] == 'right').sum())})"
    )

    cache = {
        "version": CACHE_VERSION,
        "eid": EID,
        "subject": SUBJECT,
        "date": DATE,
        "likelihood_thr": LIKELIHOOD_THR,
        "min_rt": MIN_RT,
        "max_rt": MAX_RT,
        "n_bwm": EXPECTED_N_TRIALS,
        "n_kept": len(kept),
        "trials": kept,
        "records": records,
        "data_sources": {
            "openalyx": "https://openalyx.internationalbrainlab.org",
            "trials_table": str(TRIALS_PQT),
            "trials_tag": "2024_Q2_IBL_et_al_BWM",
            "wheel": "alf/_ibl_wheel.{position,timestamps}.npy",
            "dlc": "alf/_ibl_{left,right}Camera.dlc.pqt",
            "lightning_pose": "alf/_ibl_{left,right}Camera.lightningPose.pqt",
        },
    }
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    with CACHE_PATH.open("wb") as f:
        pickle.dump(cache, f, protocol=pickle.HIGHEST_PROTOCOL)
    kept.to_csv(OUT_DIR / "ZFM-01576_included_trials.csv", index=False)
    print(f"Wrote cache {CACHE_PATH}")
    return cache


def load_or_build_cache():
    if CACHE_PATH.exists():
        with CACHE_PATH.open("rb") as f:
            cache = pickle.load(f)
        if (
            cache.get("version") == CACHE_VERSION
            and cache.get("eid") == EID
            and cache.get("likelihood_thr") == LIKELIHOOD_THR
        ):
            print(f"Loaded cache {CACHE_PATH} ({cache['n_kept']} trials)")
            return cache
        print("Existing cache is stale; rebuilding.")
    return build_cache()


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    cache = load_or_build_cache()
    records = cache["records"]
    n = cache["n_kept"]

    plots = [
        ("wheel_t", "wheel_v", "Signed wheel velocity (cm/s)",
         f"{SUBJECT} {DATE}: signed wheel velocity",
         "ZFM-01576_wheel_signed_velocity.png"),
        ("dlc_tx", "dlc_vx", "Paw velocity in x (px/s)",
         f"{SUBJECT} {DATE}: DeepLabCut paw x-velocity",
         "ZFM-01576_DLC_paw_velocity_x.png"),
        ("dlc_ty", "dlc_vy", "Paw velocity in y (px/s)",
         f"{SUBJECT} {DATE}: DeepLabCut paw y-velocity",
         "ZFM-01576_DLC_paw_velocity_y.png"),
        ("dlc_tz", "dlc_vz", "Paw velocity in z (px/s)",
         f"{SUBJECT} {DATE}: DeepLabCut paw z-velocity (stereo disparity)",
         "ZFM-01576_DLC_paw_velocity_z.png"),
        ("lp_tx", "lp_vx", "Paw velocity in x (px/s)",
         f"{SUBJECT} {DATE}: Lightning Pose paw x-velocity",
         "ZFM-01576_LightningPose_paw_velocity_x.png"),
        ("lp_ty", "lp_vy", "Paw velocity in y (px/s)",
         f"{SUBJECT} {DATE}: Lightning Pose paw y-velocity",
         "ZFM-01576_LightningPose_paw_velocity_y.png"),
        ("lp_tz", "lp_vz", "Paw velocity in z (px/s)",
         f"{SUBJECT} {DATE}: Lightning Pose paw z-velocity (stereo disparity)",
         "ZFM-01576_LightningPose_paw_velocity_z.png"),
        ("diff_tx", "diff_vx", "DLC − Lightning Pose x-velocity (px/s)",
         f"{SUBJECT} {DATE}: paw x-velocity difference (DLC − Lightning Pose)",
         "ZFM-01576_paw_velocity_x_DLC_minus_LP.png"),
        ("diff_ty", "diff_vy", "DLC − Lightning Pose y-velocity (px/s)",
         f"{SUBJECT} {DATE}: paw y-velocity difference (DLC − Lightning Pose)",
         "ZFM-01576_paw_velocity_y_DLC_minus_LP.png"),
    ]

    subtitle = (
        f"{n} correct BWM trials with RT 0.08–2 s and both DLC + LP paw traces; "
        "stim onset to feedback"
    )
    for t_key, v_key, ylabel, title, filename in plots:
        left, right = split_by_stim(records, t_key, v_key)
        save_stim_split_plot(
            left,
            right,
            ylabel,
            f"{title}\n{subtitle}",
            filename,
        )


if __name__ == "__main__":
    main()
