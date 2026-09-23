"""Trim cached BWM trials from both IBL cameras, overlay paw points, and stitch.

Right camera is shown on the left, left camera on the right. DLC and Lightning
Pose paw estimates come from that camera's own tracking. After the overlays are
drawn, playback is slowed 10x (a 2 s trial becomes 20 s).
"""

from __future__ import annotations

import importlib.util
import json
import pickle
import shutil
import subprocess
from pathlib import Path

import cv2
import numpy as np
from ibllib.io.video import get_video_frames_preload, get_video_meta, url_from_eid
from one.api import ONE

ROOT = Path(__file__).resolve().parent
DESKTOP = ROOT.parent
OUT_DIR = ROOT
CACHE_PATH = OUT_DIR / "ZFM-01576_behavior_cache.pkl"
EID = "9e9c6fc0-4769-4d83-9ea4-b59a1230510e"
SLOWDOWN = 10

DLC_COLOR = (0, 255, 255)  # BGR yellow
LP_COLOR = (255, 0, 255)  # BGR magenta
OUTLINE = (0, 0, 0)

STILL, COUPLED, DECOUPLED = "still", "coupled", "decoupled"
DLC_STATE_COLORS = {
    STILL: (0, 200, 0),       # green
    COUPLED: (255, 90, 0),    # blue
    DECOUPLED: (200, 0, 180), # purple
}
LP_STATE_COLORS = {
    STILL: (0, 0, 255),       # red
    COUPLED: (0, 255, 255),   # yellow
    DECOUPLED: (0, 0, 255),   # red
}

LABEL_FONT = cv2.FONT_HERSHEY_SIMPLEX
LABEL_SCALE = 0.8
LABEL_THICKNESS = 2
LABEL_TEXT = {"left": "LEFT CAMERA", "right": "RIGHT CAMERA"}
_LABEL_SIZE, _LABEL_BASE = cv2.getTextSize(
    "RIGHT CAMERA", LABEL_FONT, LABEL_SCALE, LABEL_THICKNESS
)
LABEL_PAD_X = 14
LABEL_PAD_Y = 12
LABEL_BOX_W = _LABEL_SIZE[0] + 2 * LABEL_PAD_X
LABEL_BOX_H = _LABEL_SIZE[1] + _LABEL_BASE + 2 * LABEL_PAD_Y


def load_behavior_module():
    spec = importlib.util.spec_from_file_location(
        "zfm_behavior", ROOT / "ZFM-01576_behavior.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def load_neural_module():
    spec = importlib.util.spec_from_file_location(
        "zfm_neural", ROOT / "ZFM-01576_neural.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def connect_one():
    return ONE(
        base_url="https://openalyx.internationalbrainlab.org",
        password="international",
    )


def trial_duration(rec):
    return float(rec["feedback_times"] - rec["stimOn_times"])


def choose_four_trials(records):
    """Short, long, and two medium trials by stim-onset to reward/feedback."""
    scored = []
    for i, rec in enumerate(records):
        duration = trial_duration(rec)
        n_pts = min(rec["dlc_tx"].size, rec["lp_tx"].size)
        if n_pts >= 8 and duration > 0:
            scored.append((duration, i))
    scored.sort()
    if len(scored) < 4:
        raise RuntimeError("Need at least 4 candidate trials to build sample videos.")

    short = scored[0]
    long = scored[-1]
    remaining = scored[1:-1]
    n = len(remaining)
    med1 = remaining[max(n // 3, 0)]
    med2 = remaining[min((2 * n) // 3, n - 1)]
    chosen = [
        ("short", short[1], short[0]),
        ("medium1", med1[1], med1[0]),
        ("medium2", med2[1], med2[0]),
        ("long", long[1], long[0]),
    ]
    # Guard against accidental duplicates if the distribution is degenerate.
    seen = set()
    unique = []
    for name, idx, dur in chosen:
        if idx in seen:
            for d, j in remaining:
                if j not in seen:
                    idx, dur = j, d
                    break
        seen.add(idx)
        unique.append((name, idx, dur))
    return unique


def frame_range(times, t0, t1):
    start = int(np.searchsorted(times, t0, side="left"))
    end = int(np.searchsorted(times, t1, side="right")) - 1
    start = max(start, 0)
    end = min(end, len(times) - 1)
    if end < start:
        raise RuntimeError("No video frames fall inside the trial window.")
    return start, end


def paw_feature(camera, used_paw):
    """Map the anatomical used paw onto IBL's near/far camera labels.

    On each side view, `paw_r` is the paw closer to that camera.
    """
    if camera == "left":
        return "paw_r" if used_paw == "left" else "paw_l"
    return "paw_r" if used_paw == "right" else "paw_l"


def nearest_xy(pose_df, feature, timestamp):
    times = pose_df["times"].to_numpy(dtype=float)
    idx = int(np.argmin(np.abs(times - timestamp)))
    if abs(times[idx] - timestamp) > 0.03:
        return np.nan, np.nan
    x = float(pose_df[f"{feature}_x"].iloc[idx])
    y = float(pose_df[f"{feature}_y"].iloc[idx])
    return x, y


def draw_point(frame, xy, color):
    if not np.isfinite(xy[0]) or not np.isfinite(xy[1]):
        return
    radius = max(6, frame.shape[0] // 140)
    pt = (int(round(xy[0])), int(round(xy[1])))
    cv2.circle(frame, pt, radius + 2, OUTLINE, 3, lineType=cv2.LINE_AA)
    cv2.circle(frame, pt, radius, color, -1, lineType=cv2.LINE_AA)


def draw_panel_label(frame, text, x0=10, y0=8):
    cv2.rectangle(frame, (x0, y0), (x0 + LABEL_BOX_W, y0 + LABEL_BOX_H), (0, 0, 0), -1)
    (tw, th), base = cv2.getTextSize(text, LABEL_FONT, LABEL_SCALE, LABEL_THICKNESS)
    tx = x0 + (LABEL_BOX_W - tw) // 2
    ty = y0 + LABEL_PAD_Y + th
    cv2.putText(
        frame,
        text,
        (tx, ty),
        LABEL_FONT,
        LABEL_SCALE,
        (255, 255, 255),
        LABEL_THICKNESS,
        cv2.LINE_AA,
    )


def _legend_swatch(frame, x, y, color, text):
    cv2.circle(frame, (x, y), 6, color, -1, cv2.LINE_AA)
    cv2.putText(frame, text, (x + 12, y + 4), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (255, 255, 255), 1, cv2.LINE_AA)


def draw_legend(frame):
    box_h = 92
    box_w = 520
    box_top = frame.shape[0] - box_h - 8
    cv2.rectangle(frame, (8, box_top), (8 + box_w, box_top + box_h), (0, 0, 0), -1)
    y_dlc = box_top + 28
    y_lp = box_top + 64
    cv2.putText(frame, "DLC", (18, y_dlc + 4), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (255, 255, 255), 1, cv2.LINE_AA)
    _legend_swatch(frame, 78, y_dlc, DLC_STATE_COLORS[STILL], "STILL")
    _legend_swatch(frame, 178, y_dlc, DLC_STATE_COLORS[COUPLED], "COUPLED")
    _legend_swatch(frame, 300, y_dlc, DLC_STATE_COLORS[DECOUPLED], "DECOUPLED")
    cv2.putText(frame, "LP", (18, y_lp + 4), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (255, 255, 255), 1, cv2.LINE_AA)
    _legend_swatch(frame, 78, y_lp, LP_STATE_COLORS[STILL], "STILL")
    _legend_swatch(frame, 178, y_lp, LP_STATE_COLORS[COUPLED], "COUPLED")
    _legend_swatch(frame, 300, y_lp, LP_STATE_COLORS[DECOUPLED], "DECOUPLED")


PLOT_TICK_FONT = 0.42
PLOT_TICK_THICK = 1
PLOT_TITLE_FONT = 0.48
PLOT_HEIGHT = 320
WHEEL_COLOR = (40, 40, 40)


def symmetric_ylim(arrays, pad=0.15):
    chunks = [np.asarray(a, dtype=float).ravel() for a in arrays if a is not None and len(a)]
    if not chunks:
        return (-1.0, 1.0)
    finite = np.concatenate(chunks)
    finite = finite[np.isfinite(finite)]
    if finite.size == 0:
        return (-1.0, 1.0)
    lo, hi = np.percentile(finite, [2, 98])
    span = max(abs(float(lo)), abs(float(hi)), 1e-3)
    span *= 1.0 + pad
    return (-span, span)


def format_tick(value):
    aval = abs(value)
    if aval >= 100:
        return f"{value:.0f}"
    if aval >= 10:
        return f"{value:.1f}"
    return f"{value:.2f}"


class TracePlot:
    """Fixed-size unfolding trace plot drawn with OpenCV."""

    def __init__(self, title, ylabel, xlim, ylim, width, height=PLOT_HEIGHT):
        self.title = title
        self.ylabel = ylabel
        self.xlim = xlim
        self.ylim = ylim
        self.width = int(width)
        self.height = int(height)
        self.ml, self.mr, self.mb, self.mt = 78, 16, 48, 32

    def to_px(self, t, v):
        t0, t1 = self.xlim
        v0, v1 = self.ylim
        plot_w = self.width - self.ml - self.mr
        plot_h = self.height - self.mt - self.mb
        x = self.ml + (float(t) - t0) / max(t1 - t0, 1e-9) * plot_w
        y = self.mt + (v1 - float(v)) / max(v1 - v0, 1e-9) * plot_h
        return int(round(x)), int(round(y))

    def render(self, series, t_now):
        img = np.full((self.height, self.width, 3), 255, dtype=np.uint8)
        x_left, y_bottom = self.to_px(self.xlim[0], self.ylim[0])
        x_right, y_top = self.to_px(self.xlim[1], self.ylim[1])
        cv2.rectangle(img, (x_left, y_top), (x_right, y_bottom), (0, 0, 0), 1)
        zero = self.to_px(self.xlim[0], 0.0)[1]
        if y_top <= zero <= y_bottom:
            cv2.line(img, (x_left, zero), (x_right, zero), (180, 180, 180), 1)

        for t_tick in np.linspace(self.xlim[0], self.xlim[1], 5):
            x, _ = self.to_px(t_tick, self.ylim[0])
            cv2.line(img, (x, y_bottom), (x, y_bottom + 5), (0, 0, 0), 1)
            label = format_tick(t_tick)
            (tw, th), _ = cv2.getTextSize(
                label, cv2.FONT_HERSHEY_SIMPLEX, PLOT_TICK_FONT, PLOT_TICK_THICK
            )
            cv2.putText(
                img,
                label,
                (x - tw // 2, y_bottom + 18),
                cv2.FONT_HERSHEY_SIMPLEX,
                PLOT_TICK_FONT,
                (0, 0, 0),
                PLOT_TICK_THICK,
                cv2.LINE_AA,
            )
        for v_tick in np.linspace(self.ylim[0], self.ylim[1], 5):
            _, y = self.to_px(self.xlim[0], v_tick)
            cv2.line(img, (x_left - 5, y), (x_left, y), (0, 0, 0), 1)
            label = format_tick(v_tick)
            (tw, th), _ = cv2.getTextSize(
                label, cv2.FONT_HERSHEY_SIMPLEX, PLOT_TICK_FONT, PLOT_TICK_THICK
            )
            cv2.putText(
                img,
                label,
                (x_left - 8 - tw, y + th // 2),
                cv2.FONT_HERSHEY_SIMPLEX,
                PLOT_TICK_FONT,
                (0, 0, 0),
                PLOT_TICK_THICK,
                cv2.LINE_AA,
            )

        cv2.putText(
            img,
            self.title,
            (self.ml, 22),
            cv2.FONT_HERSHEY_SIMPLEX,
            PLOT_TITLE_FONT,
            (0, 0, 0),
            1,
            cv2.LINE_AA,
        )
        cv2.putText(
            img,
            self.ylabel,
            (6, 18),
            cv2.FONT_HERSHEY_SIMPLEX,
            PLOT_TICK_FONT,
            (0, 0, 0),
            PLOT_TICK_THICK,
            cv2.LINE_AA,
        )
        cv2.putText(
            img,
            "Time from stim (s)",
            (self.width // 2 - 70, self.height - 8),
            cv2.FONT_HERSHEY_SIMPLEX,
            PLOT_TICK_FONT,
            (0, 0, 0),
            PLOT_TICK_THICK,
            cv2.LINE_AA,
        )

        for times, values, colors in series:
            times = np.asarray(times, dtype=float)
            values = np.asarray(values, dtype=float)
            keep = np.isfinite(times) & np.isfinite(values) & (times <= t_now + 1e-9)
            if keep.sum() < 2:
                continue
            t_k = times[keep]
            v_k = values[keep]
            if isinstance(colors, (tuple, list)) and len(colors) == 3 and not np.ndim(colors) > 1:
                c_k = [tuple(int(c) for c in colors)] * t_k.size
            else:
                c_k = [tuple(int(c) for c in col) for col in np.asarray(colors)[keep]]
            run_start = 0
            for i in range(1, len(t_k) + 1):
                if i < len(t_k) and c_k[i] == c_k[run_start]:
                    continue
                pts = np.array(
                    [self.to_px(t, v) for t, v in zip(t_k[run_start:i], v_k[run_start:i])],
                    dtype=np.int32,
                )
                if len(pts) == 1:
                    cv2.circle(img, tuple(pts[0]), 2, c_k[run_start], -1, cv2.LINE_AA)
                else:
                    cv2.polylines(img, [pts], False, c_k[run_start], 2, cv2.LINE_AA)
                run_start = i

        if self.xlim[0] <= t_now <= self.xlim[1]:
            x_now, _ = self.to_px(t_now, 0.0)
            cv2.line(img, (x_now, y_top), (x_now, y_bottom), (0, 0, 220), 1)
        return img


def stack_plots(plot_imgs):
    top = np.hstack([plot_imgs[0], plot_imgs[1]])
    bottom = np.hstack([plot_imgs[2], plot_imgs[3]])
    return np.vstack([top, bottom])


def annotate_camera(frame, dlc_df, lp_df, feature, timestamp, dlc_color, lp_color):
    vis = frame.copy()
    draw_point(vis, nearest_xy(dlc_df, feature, timestamp), dlc_color)
    draw_point(vis, nearest_xy(lp_df, feature, timestamp), lp_color)
    return vis


def resize_to_height(frame, height):
    if frame.shape[0] == height:
        return frame
    scale = height / frame.shape[0]
    width = int(round(frame.shape[1] * scale))
    return cv2.resize(frame, (width, height), interpolation=cv2.INTER_AREA)


def stitch(right_frame, left_frame):
    """Place right-camera footage on the left side of the output frame."""
    height = max(right_frame.shape[0], left_frame.shape[0])
    right = resize_to_height(right_frame, height)
    left = resize_to_height(left_frame, height)
    gap = np.full((height, 6, 3), 255, dtype=np.uint8)
    combo = np.hstack([right, gap, left])
    draw_panel_label(combo, LABEL_TEXT["right"], x0=10, y0=8)
    draw_panel_label(combo, LABEL_TEXT["left"], x0=right.shape[1] + gap.shape[1] + 10, y0=8)
    return combo


def write_mp4(path, frames, fps):
    height, width = frames[0].shape[:2]
    if height % 2 or width % 2:
        frames = [
            np.pad(f, ((0, height % 2), (0, width % 2), (0, 0)), constant_values=255)
            for f in frames
        ]
        height, width = frames[0].shape[:2]
    tmp = path.with_suffix(".tmp.mp4")
    writer_fps = 30.0
    writer = cv2.VideoWriter(
        str(tmp),
        cv2.VideoWriter_fourcc(*"mp4v"),
        writer_fps,
        (width, height),
    )
    if not writer.isOpened():
        raise RuntimeError(f"Could not open VideoWriter for {tmp}")
    for frame in frames:
        writer.write(frame)
    writer.release()
    ffmpeg = shutil.which("ffmpeg")
    if not ffmpeg:
        tmp.replace(path)
        return
    subprocess.run(
        [
            ffmpeg,
            "-y",
            "-i",
            str(tmp),
            "-filter:v",
            f"setpts={writer_fps}/{float(fps)}*PTS",
            "-r",
            f"{float(fps):.6f}",
            "-c:v",
            "libx264",
            "-pix_fmt",
            "yuv420p",
            "-movflags",
            "+faststart",
            str(path),
        ],
        check=True,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    tmp.unlink(missing_ok=True)


def as_frame_list(frames):
    if isinstance(frames, np.ndarray):
        return [frames[i] for i in range(len(frames))]
    return list(frames)


def load_camera_clip(one, camera, times, t0, t1, urls):
    start, end = frame_range(times, t0, t1)
    frame_numbers = list(range(start, end + 1))
    url = urls[camera]
    print(f"{camera} camera: frames {start}-{end} ({len(frame_numbers)})")
    frames = as_frame_list(
        get_video_frames_preload(url, frame_numbers=frame_numbers, quiet=False)
    )
    if any(frame is None or not isinstance(frame, np.ndarray) for frame in frames):
        raise RuntimeError(f"Failed to stream {camera} camera frames.")
    return {
        "url": url,
        "frames": frames,
        "times": times[start : end + 1],
        "frame_numbers": frame_numbers,
        "native_fps": 1.0 / float(np.median(np.diff(times))),
    }


def nearest_index(times, timestamp):
    return int(np.argmin(np.abs(times - timestamp)))


def nearest_labels(t_src, labels, t_dst, default=STILL):
    t_src = np.asarray(t_src, dtype=float)
    t_dst = np.asarray(t_dst, dtype=float)
    labels = np.asarray(labels)
    if t_dst.size == 0:
        return labels[:0]
    if t_src.size == 0:
        return np.full(t_dst.shape, default, dtype=object)
    idx = np.searchsorted(t_src, t_dst)
    right = np.clip(idx, 0, t_src.size - 1)
    left = np.clip(idx - 1, 0, t_src.size - 1)
    choose_right = np.abs(t_src[right] - t_dst) <= np.abs(t_src[left] - t_dst)
    take = np.where(choose_right, right, left)
    return labels[take]


def colors_for_states(states, palette):
    states = np.asarray(states, dtype=object)
    out = np.zeros((states.size, 3), dtype=np.int32)
    for name, color in palette.items():
        out[states == name] = color
    return out


def contact_height_baseline(y, times):
    """Follow on-wheel image height without chasing lifts.

    Image y increases downward, so contact with the wheel is the *larger* y.
    The baseline tracks downward (contact) quickly and upward (lift) slowly.
    That lets the paw rest anywhere on the wheel's width/front-back: the
    reference is the recent contact height, not a single x/z parking spot.
    """
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
    """Replace each sample with the majority label in a symmetric window."""
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
    """Merge state runs shorter than min_s into the neighboring label."""
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
    """Label each sample STILL / COUPLED / DECOUPLED.

    STILL: paw speed is in the jitter band and the paw is not lifted.
    COUPLED: paw is moving while the wheel is turning and the paw remains
             near the recent contact height (dragging).
    DECOUPLED: the paw is lifted off that contact height, or it is moving
               while the wheel is idle (a reposition that is not turning
               the wheel).

    The paw is assumed to start on the wheel, but not at a unique x/z.
    """
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
    # A mostly-vertical swipe is a lift even if the wheel is still coasting
    # from the previous drag, and even before lift reaches the full threshold.
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

    lookup = np.array([STILL, COUPLED, DECOUPLED], dtype=object)
    return lookup[np.clip(codes, 0, 2)]


def window_near_paw_xy(pose, used_paw, t0, t1, beh):
    camera = "left" if used_paw == "left" else "right"
    feature = "paw_r"
    t, x, y = beh.camera_arrays(pose[f"{camera}Camera"], feature)
    mask = (
        (t >= t0)
        & (t <= t1)
        & np.isfinite(t)
        & np.isfinite(x)
        & np.isfinite(y)
    )
    return t[mask] - t0, x[mask], y[mask]


def motion_states_for_tracker(trial, pose, t0, t1, beh, prefix):
    rel_t, x, y = window_near_paw_xy(pose, trial["used_paw"], t0, t1, beh)
    if rel_t.size < 3:
        return rel_t, np.full(rel_t.shape, STILL, dtype=object)
    vx = beh.interpolate_to(trial[f"{prefix}_tx"], trial[f"{prefix}_vx"], rel_t)
    vy = beh.interpolate_to(trial[f"{prefix}_ty"], trial[f"{prefix}_vy"], rel_t)
    omega = beh.interpolate_to(
        trial["wheel_t"],
        np.asarray(trial["wheel_v"], dtype=float) / float(beh.WHEEL_RADIUS_CM),
        rel_t,
    )
    first_move = float(trial["firstMovement_times"]) - t0
    states = segment_paw_motion(rel_t, x, y, vx, vy, omega, first_move_s=first_move)
    return rel_t, states


def state_at_time(t_src, states, timestamp, default=STILL):
    labels = nearest_labels(t_src, states, np.array([timestamp], dtype=float), default=default)
    return str(labels[0]) if labels.size else default


def save_state_csv(path, t_dlc, dlc_states, t_lp, lp_states):
    times = t_dlc
    dlc_on = nearest_labels(t_dlc, dlc_states, times)
    lp_on = nearest_labels(t_lp, lp_states, times)
    with path.open("w") as f:
        f.write("time_from_stim_s,dlc_state,lp_state\n")
        for t, dlc, lp in zip(times, dlc_on, lp_on):
            f.write(f"{float(t):.6f},{dlc},{lp}\n")


def render_trial(trial, trial_idx, label, one, dlc_pose, lp_pose, urls, beh, neural=None):
    t0 = float(trial["stimOn_times"])
    t1 = float(trial["feedback_times"])
    used_paw = trial["used_paw"]
    duration = t1 - t0
    print(
        f"\n=== {label} trial {trial_idx}: stim {trial['stim_side']}, "
        f"used paw {used_paw}, {duration:.3f}s stim->reward ==="
    )

    dlc_t_state, dlc_states = motion_states_for_tracker(trial, dlc_pose, t0, t1, beh, "dlc")
    lp_t_state, lp_states = motion_states_for_tracker(trial, lp_pose, t0, t1, beh, "lp")
    csv_path = OUT_DIR / f"Sample_{label}_paw_states.csv"
    save_state_csv(csv_path, dlc_t_state, dlc_states, lp_t_state, lp_states)
    print(
        f"DLC states: still={np.mean(dlc_states == STILL):.2%} "
        f"coupled={np.mean(dlc_states == COUPLED):.2%} "
        f"decoupled={np.mean(dlc_states == DECOUPLED):.2%}"
    )
    print(
        f"LP  states: still={np.mean(lp_states == STILL):.2%} "
        f"coupled={np.mean(lp_states == COUPLED):.2%} "
        f"decoupled={np.mean(lp_states == DECOUPLED):.2%}"
    )
    print(f"Wrote {csv_path.name}")

    dlc_vx_c = colors_for_states(nearest_labels(dlc_t_state, dlc_states, trial["dlc_tx"]), DLC_STATE_COLORS)
    dlc_vy_c = colors_for_states(nearest_labels(dlc_t_state, dlc_states, trial["dlc_ty"]), DLC_STATE_COLORS)
    dlc_vz_c = colors_for_states(nearest_labels(dlc_t_state, dlc_states, trial["dlc_tz"]), DLC_STATE_COLORS)
    lp_vx_c = colors_for_states(nearest_labels(lp_t_state, lp_states, trial["lp_tx"]), LP_STATE_COLORS)
    lp_vy_c = colors_for_states(nearest_labels(lp_t_state, lp_states, trial["lp_ty"]), LP_STATE_COLORS)
    lp_vz_c = colors_for_states(nearest_labels(lp_t_state, lp_states, trial["lp_tz"]), LP_STATE_COLORS)
    wheel_c = colors_for_states(nearest_labels(dlc_t_state, dlc_states, trial["wheel_t"]), DLC_STATE_COLORS)

    clips = {}
    for camera in ("left", "right"):
        times = dlc_pose[f"{camera}Camera"]["times"].to_numpy(dtype=float)
        clips[camera] = load_camera_clip(one, camera, times, t0, t1, urls)

    master = "right" if clips["right"]["native_fps"] >= clips["left"]["native_fps"] else "left"
    master_times = clips[master]["times"]
    output_fps = clips[master]["native_fps"] / SLOWDOWN

    wheel_omega = np.asarray(trial["wheel_v"], dtype=float) / float(beh.WHEEL_RADIUS_CM)
    plot_specs = [
        (
            "Paw signed x-velocity",
            "px/s",
            [
                (trial["dlc_tx"], trial["dlc_vx"], dlc_vx_c),
                (trial["lp_tx"], trial["lp_vx"], lp_vx_c),
            ],
        ),
        (
            "Paw signed y-velocity",
            "px/s",
            [
                (trial["dlc_ty"], trial["dlc_vy"], dlc_vy_c),
                (trial["lp_ty"], trial["lp_vy"], lp_vy_c),
            ],
        ),
        (
            "Paw signed z-velocity",
            "px/s",
            [
                (trial["dlc_tz"], trial["dlc_vz"], dlc_vz_c),
                (trial["lp_tz"], trial["lp_vz"], lp_vz_c),
            ],
        ),
        (
            "Wheel signed angular velocity",
            "rad/s",
            [(trial["wheel_t"], wheel_omega, wheel_c)],
        ),
    ]

    sample_combo = None
    stitched = []
    for timestamp in master_times:
        rel = float(timestamp) - t0
        dlc_now = state_at_time(dlc_t_state, dlc_states, rel)
        lp_now = state_at_time(lp_t_state, lp_states, rel)
        panels = {}
        for camera in ("left", "right"):
            idx = nearest_index(clips[camera]["times"], timestamp)
            feature = paw_feature(camera, used_paw)
            panels[camera] = annotate_camera(
                clips[camera]["frames"][idx],
                dlc_pose[f"{camera}Camera"],
                lp_pose[f"{camera}Camera"],
                feature,
                float(timestamp),
                tuple(int(c) for c in DLC_STATE_COLORS[dlc_now]),
                tuple(int(c) for c in LP_STATE_COLORS[lp_now]),
            )
        combo = stitch(panels["right"], panels["left"])
        draw_legend(combo)
        cv2.putText(
            combo,
            f"{label}  t={rel:.3f}s  stim={trial['stim_side']}  paw={used_paw}  "
            f"DLC={dlc_now.upper()}  LP={lp_now.upper()}  {SLOWDOWN}x slower",
            (540, combo.shape[0] - 16),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.62,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )
        if sample_combo is None:
            sample_combo = combo
            plot_w = combo.shape[1] // 2
            remainder = combo.shape[1] - 2 * plot_w
            plots = [
                TracePlot(title, ylabel, (0.0, duration), symmetric_ylim([s[1] for s in series]), plot_w)
                for title, ylabel, series in plot_specs
            ]
        plot_imgs = [plot.render(series, rel) for plot, (_, _, series) in zip(plots, plot_specs)]
        plot_panel = stack_plots(plot_imgs)
        if remainder:
            plot_panel = np.hstack(
                [plot_panel, np.full((plot_panel.shape[0], remainder, 3), 255, dtype=np.uint8)]
            )
        left_stack = np.vstack([combo, plot_panel])
        if neural is not None:
            gap = np.full((left_stack.shape[0], 6, 3), 255, dtype=np.uint8)
            neural_panel = neural.render(label, float(timestamp), height=left_stack.shape[0])
            stitched.append(np.hstack([left_stack, gap, neural_panel]))
        else:
            stitched.append(left_stack)

    out_path = OUT_DIR / f"Sample_{label}.mp4"
    write_mp4(out_path, stitched, output_fps)
    info = {
        "label": label,
        "eid": EID,
        "trial_index_in_cache": trial_idx,
        "stim_side": trial["stim_side"],
        "used_paw": used_paw,
        "paw_feature": {
            "left": paw_feature("left", used_paw),
            "right": paw_feature("right", used_paw),
        },
        "stimOn_times": t0,
        "feedback_times": t1,
        "firstMovement_times": float(trial["firstMovement_times"]),
        "original_duration_s": duration,
        "slowdown": SLOWDOWN,
        "output_duration_s": duration * SLOWDOWN,
        "output_fps": output_fps,
        "n_stitched_frames": len(stitched),
        "sample": str(out_path),
        "frame_shape": list(stitched[0].shape) if stitched else None,
        "paw_states_csv": str(csv_path),
        "dlc_state_fraction": {
            STILL: float(np.mean(dlc_states == STILL)) if dlc_states.size else None,
            COUPLED: float(np.mean(dlc_states == COUPLED)) if dlc_states.size else None,
            DECOUPLED: float(np.mean(dlc_states == DECOUPLED)) if dlc_states.size else None,
        },
        "lp_state_fraction": {
            STILL: float(np.mean(lp_states == STILL)) if lp_states.size else None,
            COUPLED: float(np.mean(lp_states == COUPLED)) if lp_states.size else None,
            DECOUPLED: float(np.mean(lp_states == DECOUPLED)) if lp_states.size else None,
        },
    }
    print(
        f"Wrote {out_path.name}: {duration:.3f}s original -> "
        f"{duration * SLOWDOWN:.3f}s at {output_fps:.2f} fps"
    )
    return info


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    beh = load_behavior_module()
    neu = load_neural_module()

    with CACHE_PATH.open("rb") as f:
        cache = pickle.load(f)
    chosen = choose_four_trials(cache["records"])
    for name, idx, dur in chosen:
        print(f"Selected {name}: cache trial {idx}, stim->reward {dur:.3f}s")

    one = connect_one()
    trial_windows = neu.trial_windows_from_behavior_cache(cache, chosen)
    neural_cache = neu.load_neural_cache(one=one, trial_windows=trial_windows)
    neural = neu.NeuralRenderer(neural_cache, width=neu.NEURAL_WIDTH, height=1664)
    print(
        f"Neural renderer: {neural_cache['n_units_total']} BWM units, "
        f"{len(neural_cache['probes'])} probe(s)"
    )

    sess_loader = beh.SessionLoader(one=one, eid=EID)
    dlc_pose = beh.load_pose_copy(sess_loader, "dlc")
    lp_pose = beh.load_pose_copy(sess_loader, "lightningPose")
    urls = {
        "left": url_from_eid(EID, label="left", one=one),
        "right": url_from_eid(EID, label="right", one=one),
    }
    for camera, url in urls.items():
        meta = get_video_meta(url, one=one)
        print(f"{camera} source: {meta.size / 1e9:.2f} GB {url}")

    infos = []
    for name, idx, _dur in chosen:
        infos.append(
            render_trial(
                cache["records"][idx],
                idx,
                name,
                one,
                dlc_pose,
                lp_pose,
                urls,
                beh,
                neural=neural,
            )
        )

    # Keep Sample.mp4 as a copy of the first medium trial for continuity.
    medium = next(info for info in infos if info["label"] == "medium1")
    shutil.copy2(medium["sample"], OUT_DIR / "Sample.mp4")

    summary_path = OUT_DIR / "ZFM-01576_sample_trials.json"
    summary_path.write_text(json.dumps(infos, indent=2))
    print(f"Wrote {summary_path}")


if __name__ == "__main__":
    main()
