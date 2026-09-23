"""Sample videos of left-paw, right-paw, and bimanual wheel turning.

Leans on ZFM-01576_video.py: both IBL cameras, DLC/LP paw overlays, STILL /
COUPLED / DECOUPLED colors, unfolding traces, 10x slowdown, ffmpeg mp4.

Unlike the original, every clip overlays **both** anatomical paws so a
body-held other paw is visible, and trials are chosen from ModelData using
the wheel_paws label.
"""

from __future__ import annotations

import importlib.util
import json
import pickle
import sys
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
from ibllib.io.video import get_video_meta, url_from_eid

ROOT = Path(__file__).resolve().parent
DESKTOP = ROOT.parent
BEH = ROOT
OUT_DIR = BEH / "PawStrats"
MODEL_TRIALS = BEH / "ModelData" / "trials.parquet"
FILT_SESS = BEH / "DatasetFiltered" / "sessions"

sys.path.insert(0, str(ROOT))
from enrich_and_discretize import (  # noqa: E402
    COUPLED,
    DECOUPLED,
    STILL,
    interpolate_to,
    segment_paw_motion,
    tracker_kinematics,
)
from fullData import camera_arrays, load_pose_copy  # noqa: E402

spec = importlib.util.spec_from_file_location("zfm_video", ROOT / "ZFM-01576_video.py")
vid = importlib.util.module_from_spec(spec)
spec.loader.exec_module(vid)

LEFT_MOUSE = "ZFM-01576"
RIGHT_MOUSE = "ZM_2241"
BOTH_MOUSE = "UCLA034"
DUR_LO, DUR_HI = 0.28, 0.80
BINS_LO, BINS_HI = 12, 40
SLOWDOWN = vid.SLOWDOWN
OUTLINE = vid.OUTLINE
DLC_STATE_COLORS = vid.DLC_STATE_COLORS
LP_STATE_COLORS = vid.LP_STATE_COLORS


def load_behavior_module():
    spec_b = importlib.util.spec_from_file_location(
        "zfm_behavior", ROOT / "ZFM-01576_behavior.py"
    )
    module = importlib.util.module_from_spec(spec_b)
    spec_b.loader.exec_module(module)
    return module


def draw_square(frame, xy, color):
    if not np.isfinite(xy[0]) or not np.isfinite(xy[1]):
        return
    half = max(6, frame.shape[0] // 140)
    x, y = int(round(xy[0])), int(round(xy[1]))
    cv2.rectangle(frame, (x - half - 2, y - half - 2), (x + half + 2, y + half + 2), OUTLINE, 3)
    cv2.rectangle(frame, (x - half, y - half), (x + half, y + half), color, -1)


def draw_paw_legend(frame):
    vid.draw_legend(frame)
    box_h = 44
    box_w = 420
    box_top = frame.shape[0] - 92 - 8 - box_h - 4
    cv2.rectangle(frame, (8, box_top), (8 + box_w, box_top + box_h), (0, 0, 0), -1)
    y = box_top + 28
    cv2.circle(frame, (28, y - 4), 8, (220, 220, 220), -1, cv2.LINE_AA)
    cv2.putText(frame, "LEFT PAW", (44, y), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1, cv2.LINE_AA)
    cv2.rectangle(frame, (168, y - 12), (184, y + 4), (220, 220, 220), -1)
    cv2.putText(frame, "RIGHT PAW", (192, y), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1, cv2.LINE_AA)


def anatomical_feature(camera, paw):
    """Near paw is paw_r on that camera."""
    if camera == "left":
        return "paw_r" if paw == "left" else "paw_l"
    return "paw_r" if paw == "right" else "paw_l"


def window_near_paw(pose, paw, t0, t1):
    camera = "leftCamera" if paw == "left" else "rightCamera"
    if pose is None or camera not in pose:
        return np.array([]), np.array([]), np.array([])
    t, x, y = camera_arrays(pose[camera], "paw_r")
    mask = (t >= t0) & (t <= t1) & np.isfinite(t)
    if mask.sum() < 3:
        return np.array([]), np.array([]), np.array([])
    return t[mask] - t0, x[mask], y[mask]


def paw_states(pose, paw, rec, t0, t1):
    rel_t, x, y = window_near_paw(pose, paw, t0, t1)
    if rel_t.size < 3:
        return rel_t, np.full(rel_t.shape, STILL, dtype=object), np.array([]), np.array([])
    vx, vy, _vz, speed = tracker_kinematics(rel_t, x, y, np.full_like(x, np.nan))
    omega = interpolate_to(rec["wheel_t"], rec["wheel_velocity"], rel_t, extrapolate=False)
    first_move = float(rec["firstMovement_times"]) - t0
    states = segment_paw_motion(rel_t, x, y, vx, vy, omega, first_move_s=first_move)
    return rel_t, states, speed, np.hypot(vx, vy)


def candidates(df, mouse, label):
    g = df[
        (df["mouse_id"] == mouse)
        & (df["wheel_paws"] == label)
        & df["trialDuration"].between(DUR_LO, DUR_HI)
        & df["n_bins"].between(BINS_LO, BINS_HI)
    ].sort_values("trialDuration")
    if g.empty:
        raise RuntimeError(f"No {label} trials for {mouse} in duration/bin window.")
    return g


def pick_spaced(g, n):
    if len(g) < n:
        raise RuntimeError(f"Need {n} trials, found {len(g)}.")
    idxs = np.unique(np.linspace(0, len(g) - 1, n).round().astype(int))
    if idxs.size < n:
        extra = [i for i in range(len(g)) if i not in set(idxs)]
        idxs = np.concatenate([idxs, extra[: n - idxs.size]])
    return g.iloc[idxs[:n]]


def select_clips(df):
    left_rows = pick_spaced(candidates(df, LEFT_MOUSE, "left"), 2)
    right_rows = pick_spaced(candidates(df, RIGHT_MOUSE, "right"), 2)
    both_left = pick_spaced(candidates(df, BOTH_MOUSE, "left"), 1)
    both_right = pick_spaced(candidates(df, BOTH_MOUSE, "right"), 1)
    both_both = pick_spaced(candidates(df, BOTH_MOUSE, "both"), 2)
    clips = []
    for i, (_, row) in enumerate(left_rows.iterrows(), start=1):
        clips.append(("left-dominant", f"leftmouse_{LEFT_MOUSE}_left{i}", row))
    for i, (_, row) in enumerate(right_rows.iterrows(), start=1):
        clips.append(("right-dominant", f"rightmouse_{RIGHT_MOUSE}_right{i}", row))
    clips.append(("both-strategy mouse, left-only trial", f"bothmouse_{BOTH_MOUSE}_left", both_left.iloc[0]))
    clips.append(("both-strategy mouse, right-only trial", f"bothmouse_{BOTH_MOUSE}_right", both_right.iloc[0]))
    for i, (_, row) in enumerate(both_both.iterrows(), start=1):
        clips.append(("both-strategy mouse, both-paws trial", f"bothmouse_{BOTH_MOUSE}_both{i}", row))
    return clips


def load_native_trial(eid, trial_index):
    path = FILT_SESS / f"{eid}.pkl"
    with path.open("rb") as fh:
        payload = pickle.load(fh)
    for rec in payload["trials"]:
        if int(rec["trial_index"]) == int(trial_index):
            return rec, payload
    raise RuntimeError(f"Trial {trial_index} not in DatasetFiltered session {eid}")


def annotate_both_paws(
    frame,
    dlc_df,
    lp_df,
    camera,
    timestamp,
    left_dlc,
    left_lp,
    right_dlc,
    right_lp,
):
    vis = frame.copy()
    near_paw = "left" if camera == "left" else "right"
    far_paw = "right" if camera == "left" else "left"
    far_dlc = right_dlc if far_paw == "right" else left_dlc
    far_lp = right_lp if far_paw == "right" else left_lp
    near_dlc = left_dlc if near_paw == "left" else right_dlc
    near_lp = left_lp if near_paw == "left" else right_lp
    far_draw = draw_square if far_paw == "right" else vid.draw_point
    near_draw = draw_square if near_paw == "right" else vid.draw_point
    far_draw(vis, vid.nearest_xy(dlc_df, anatomical_feature(camera, far_paw), timestamp), far_dlc)
    far_draw(vis, vid.nearest_xy(lp_df, anatomical_feature(camera, far_paw), timestamp), far_lp)
    near_draw(vis, vid.nearest_xy(dlc_df, anatomical_feature(camera, near_paw), timestamp), near_dlc)
    near_draw(vis, vid.nearest_xy(lp_df, anatomical_feature(camera, near_paw), timestamp), near_lp)
    return vis


def render_trial(row, rec, label, stem, one, dlc_pose, lp_pose, urls, beh, neural=None):
    t0 = float(rec["stimOn_times"])
    t1 = float(rec["feedback_times"])
    duration = t1 - t0
    used_paw = rec.get("used_paw")
    wheel_paws = rec.get("wheel_paws")
    print(
        f"\n=== {stem}: {row.mouse_id} trial {int(row.trial_index)}  "
        f"wheel_paws={wheel_paws} used_paw={used_paw}  {duration:.3f}s ==="
    )

    paw_kin = {}
    for tracker, pose in (("dlc", dlc_pose), ("lp", lp_pose)):
        paw_kin[tracker] = {}
        for paw in ("left", "right"):
            t, states, _speed3, speed2 = paw_states(pose, paw, rec, t0, t1)
            paw_kin[tracker][paw] = {"t": t, "states": states, "speed": speed2}

    clips = {}
    for camera in ("left", "right"):
        times = dlc_pose[f"{camera}Camera"]["times"].to_numpy(dtype=float)
        clips[camera] = vid.load_camera_clip(one, camera, times, t0, t1, urls)

    master = "right" if clips["right"]["native_fps"] >= clips["left"]["native_fps"] else "left"
    master_times = clips[master]["times"]
    output_fps = clips[master]["native_fps"] / SLOWDOWN

    def series_for(paw):
        dlc = paw_kin["dlc"][paw]
        lp = paw_kin["lp"][paw]
        dlc_c = vid.colors_for_states(vid.nearest_labels(dlc["t"], dlc["states"], dlc["t"]), DLC_STATE_COLORS)
        lp_c = vid.colors_for_states(vid.nearest_labels(lp["t"], lp["states"], lp["t"]), LP_STATE_COLORS)
        return [
            (dlc["t"], dlc["speed"], dlc_c),
            (lp["t"], lp["speed"], lp_c),
        ]

    wheel_c = vid.colors_for_states(
        vid.nearest_labels(paw_kin["dlc"]["left"]["t"], paw_kin["dlc"]["left"]["states"], rec["wheel_t"]),
        DLC_STATE_COLORS,
    )
    plot_specs = [
        ("Left paw speed", "px/s", series_for("left")),
        ("Right paw speed", "px/s", series_for("right")),
        (
            "Wheel signed angular velocity",
            "rad/s",
            [(rec["wheel_t"], rec["wheel_velocity"], wheel_c)],
        ),
        (
            "Used-paw 2D speed (stored trace)",
            "px/s",
            [
                (
                    rec["dlc_t"],
                    np.hypot(rec["dlc_vx"], rec["dlc_vy"]),
                    vid.colors_for_states(
                        vid.nearest_labels(paw_kin["dlc"][used_paw or "left"]["t"], paw_kin["dlc"][used_paw or "left"]["states"], rec["dlc_t"]),
                        DLC_STATE_COLORS,
                    ),
                ),
                (
                    rec["lp_t"],
                    np.hypot(rec["lp_vx"], rec["lp_vy"]),
                    vid.colors_for_states(
                        vid.nearest_labels(paw_kin["lp"][used_paw or "left"]["t"], paw_kin["lp"][used_paw or "left"]["states"], rec["lp_t"]),
                        LP_STATE_COLORS,
                    ),
                ),
            ],
        ),
    ]

    stitched = []
    plots = None
    remainder = 0
    for timestamp in master_times:
        rel = float(timestamp) - t0
        left_dlc = vid.state_at_time(paw_kin["dlc"]["left"]["t"], paw_kin["dlc"]["left"]["states"], rel)
        left_lp = vid.state_at_time(paw_kin["lp"]["left"]["t"], paw_kin["lp"]["left"]["states"], rel)
        right_dlc = vid.state_at_time(paw_kin["dlc"]["right"]["t"], paw_kin["dlc"]["right"]["states"], rel)
        right_lp = vid.state_at_time(paw_kin["lp"]["right"]["t"], paw_kin["lp"]["right"]["states"], rel)
        panels = {}
        for camera in ("left", "right"):
            idx = vid.nearest_index(clips[camera]["times"], timestamp)
            panels[camera] = annotate_both_paws(
                clips[camera]["frames"][idx],
                dlc_pose[f"{camera}Camera"],
                lp_pose[f"{camera}Camera"],
                camera,
                float(timestamp),
                tuple(int(c) for c in DLC_STATE_COLORS[left_dlc]),
                tuple(int(c) for c in LP_STATE_COLORS[left_lp]),
                tuple(int(c) for c in DLC_STATE_COLORS[right_dlc]),
                tuple(int(c) for c in LP_STATE_COLORS[right_lp]),
            )
        combo = vid.stitch(panels["right"], panels["left"])
        draw_paw_legend(combo)
        cv2.putText(
            combo,
            f"{stem}  t={rel:.3f}s  wheel_paws={wheel_paws}  used={used_paw}  "
            f"L DLC={left_dlc.upper()}  R DLC={right_dlc.upper()}  {SLOWDOWN}x slower",
            (540, combo.shape[0] - 16),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )
        if plots is None:
            plot_w = combo.shape[1] // 2
            remainder = combo.shape[1] - 2 * plot_w
            plots = [
                vid.TracePlot(title, ylabel, (0.0, duration), vid.symmetric_ylim([s[1] for s in series]), plot_w)
                for title, ylabel, series in plot_specs
            ]
        plot_imgs = [plot.render(series, rel) for plot, (_, _, series) in zip(plots, plot_specs)]
        plot_panel = vid.stack_plots(plot_imgs)
        if remainder:
            plot_panel = np.hstack(
                [plot_panel, np.full((plot_panel.shape[0], remainder, 3), 255, dtype=np.uint8)]
            )
        left_stack = np.vstack([combo, plot_panel])
        if neural is not None:
            gap = np.full((left_stack.shape[0], 6, 3), 255, dtype=np.uint8)
            neural_panel = neural.render(stem, float(timestamp), height=left_stack.shape[0])
            frame = np.hstack([left_stack, gap, neural_panel])
        else:
            frame = left_stack
        h, w = frame.shape[:2]
        if h % 2 or w % 2:
            frame = np.pad(frame, ((0, h % 2), (0, w % 2), (0, 0)), constant_values=255)
        stitched.append(frame)

    out_path = OUT_DIR / f"{stem}.mp4"
    vid.write_mp4(out_path, stitched, output_fps)
    info = {
        "label": label,
        "stem": stem,
        "mouse_id": row.mouse_id,
        "eid": row.eid,
        "trial_index": int(row.trial_index),
        "session_date": str(row.session_date),
        "wheel_paws": wheel_paws,
        "used_paw": used_paw,
        "stim_side": rec.get("stim_side"),
        "brightness": float(row.brightness) if pd.notna(row.brightness) else None,
        "n_bins": int(row.n_bins),
        "stimOn_times": t0,
        "feedback_times": t1,
        "original_duration_s": duration,
        "slowdown": SLOWDOWN,
        "output_duration_s": duration * SLOWDOWN,
        "output_fps": output_fps,
        "n_stitched_frames": len(stitched),
        "sample": str(out_path),
        "frame_shape": list(stitched[0].shape) if stitched else None,
        "neural": neural is not None,
    }
    print(f"Wrote {out_path.name}: {duration:.3f}s -> {duration * SLOWDOWN:.3f}s at {output_fps:.2f} fps")
    return info


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    df = pd.read_parquet(MODEL_TRIALS)
    clips = select_clips(df)
    for label, stem, row in clips:
        print(
            f"Selected {stem}: {row.mouse_id} {row.eid[:8]} trial {int(row.trial_index)}  "
            f"wheel_paws={row.wheel_paws} dur={row.trialDuration:.3f}s n_bins={int(row.n_bins)}"
        )

    beh = load_behavior_module()
    neu = vid.load_neural_module()
    one = vid.connect_one()
    pose_cache = {}
    url_cache = {}
    neural_dir = OUT_DIR / "neural"
    neural_dir.mkdir(parents=True, exist_ok=True)

    windows_by_eid = {}
    rec_cache = {}
    for label, stem, row in clips:
        rec, _payload = load_native_trial(row.eid, int(row.trial_index))
        rec_cache[(row.eid, int(row.trial_index))] = rec
        windows_by_eid.setdefault(row.eid, []).append(
            (stem, float(rec["stimOn_times"]), float(rec["feedback_times"]))
        )

    renderers = {}
    atlas_ba = None
    for eid, wins in windows_by_eid.items():
        cpath = neural_dir / f"{eid}_cache.pkl"
        apath = neural_dir / f"{eid}_atlas.pkl"
        if eid == neu.EID and neu.ATLAS_ASSET_PATH.exists() and not apath.exists():
            apath.write_bytes(neu.ATLAS_ASSET_PATH.read_bytes())
        ncache = neu.load_neural_cache(
            one=one, trial_windows=wins, eid=eid, cache_path=cpath
        )
        rend = neu.NeuralRenderer(
            ncache,
            width=neu.NEURAL_WIDTH,
            height=1664,
            atlas_path=apath,
            ba=atlas_ba,
        )
        atlas_ba = rend.ba
        renderers[eid] = rend
        print(
            f"Neural {eid[:8]}: {ncache['n_units_total']} BWM units, "
            f"{len(ncache['probes'])} probe(s)"
        )

    infos = []
    for label, stem, row in clips:
        eid = row.eid
        rec = rec_cache[(eid, int(row.trial_index))]
        if eid not in pose_cache:
            sess_loader = beh.SessionLoader(one=one, eid=eid)
            pose_cache[eid] = {
                "dlc": load_pose_copy(sess_loader, "dlc"),
                "lp": load_pose_copy(sess_loader, "lightningPose"),
            }
            url_cache[eid] = {
                "left": url_from_eid(eid, label="left", one=one),
                "right": url_from_eid(eid, label="right", one=one),
            }
            for camera, url in url_cache[eid].items():
                meta = get_video_meta(url, one=one)
                print(f"{row.mouse_id} {camera} source: {meta.size / 1e9:.2f} GB")
        pose = pose_cache[eid]
        infos.append(
            render_trial(
                row,
                rec,
                label,
                stem,
                one,
                pose["dlc"],
                pose["lp"],
                url_cache[eid],
                beh,
                neural=renderers.get(eid),
            )
        )

    summary = OUT_DIR / "pawstrats_trials.json"
    summary.write_text(json.dumps(infos, indent=2, default=str))
    print(f"Wrote {summary}")


if __name__ == "__main__":
    main()
