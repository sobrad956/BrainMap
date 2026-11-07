"""IBL BrainWideBench trial of the ExperimentsClean formulations.

Protocol follows Andre et al. 2026, arXiv:2609.22064
(https://arxiv.org/abs/2609.22064) and the public suite
https://www.iblcore.org/resources/benchmarking/ibl-brainwidebench/

This is a local trial, not a leaderboard submission. Official eval labels
are scored on the BWB server. Here we take ModelDataRightContra sessions,
re-extract the 1 s / 50 Hz windows around stim / first-movement / feedback,
and train the existing set-encoder / GNN / hierarchical backbones with
task-specific heads.

Task suites
  TS1  behavior decoding (frame R² / D², sequence balanced accuracy)
  TS2  forecast last 200 ms; co-smooth a held-out unit subset (Poisson D², bps)
  TS3  Cosmos-level region ID from activity, no area embedding (macro F1)

Local:
  python BrainWideBench.py --trial
  python BrainWideBench.py --trial --offline --max-sessions 1
"""

from __future__ import annotations

import argparse
import json
import math
import pickle
import sys
import traceback
import warnings
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import balanced_accuracy_score, f1_score, r2_score
from torch.utils.data import DataLoader, Dataset

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import slurm_utils

import Modelv1 as mv
import ExperimentsClean as clean
import graph as gnn
import heirarchical as hier
import fullData as fd

mv.MODEL_ROOT = ROOT / "BrainWideBench"

DATA = ROOT / "ModelDataRightContra"
SESS = DATA / "sessions"
OUT = ROOT / "BrainWideBench"
CACHE = OUT / "cache"
TRIAL_DIR = OUT / "trial"

BINSIZE = 0.02
N_BINS = 50
FORECAST_BINS = 10
COSMOOTH_FRAC = 0.20
MIN_UNITS = 8
MAX_UNITS = 80
MIN_TRIALS = 40
MAX_TRIALS = 160
BATCH = 16
LR = 1e-3
WEIGHT_DECAY = 1e-4
PATIENCE = 4
SEED = 0
SPIKE_REVISION = "2024-05-06"

CONTRAST_LEVELS = (0.0, 0.0625, 0.125, 0.25, 1.0)
COSMOS = (
    "Isocortex",
    "OLF",
    "HPF",
    "CTXsp",
    "CNU",
    "TH",
    "HY",
    "MB",
    "HB",
    "CB",
)
COSMOS_ID = {name: i for i, name in enumerate(COSMOS)}

# Prefix table for Allen acronym → IBL Cosmos (10 classes). Fibers / void drop.
_COSMOS_PREFIX = {
    "Isocortex": (
        "FRP", "MOp", "MOs", "SSp", "SSs", "SS", "GU", "VISC", "AUD", "VIS",
        "ACA", "PL", "ILA", "ORB", "AI", "RSP", "PTL", "TEa", "PERI", "ECT",
        "MO", "MOp", "MOs",
    ),
    "OLF": ("MOB", "AOB", "AON", "TT", "DP", "PIR", "NLOT", "COA", "PAA", "TR", "OT", "TTd", "TTv"),
    "HPF": ("CA1", "CA2", "CA3", "DG", "SUB", "ProS", "PRE", "POST", "HATA", "IG", "FC", "HIP", "HPF"),
    "CTXsp": ("CLA", "EP", "LA", "BLA", "BMA", "PA", "CTXsp", "EPd", "EPv"),
    "CNU": (
        "CP", "ACB", "FS", "GPe", "GPi", "SI", "MA", "LS", "MS", "NDB", "BS",
        "SF", "SH", "STR", "PAL", "sAMY", "CEA", "IA", "MEA",
    ),
    "TH": (
        "TH", "VAL", "VPM", "VPL", "PO", "LP", "LD", "MD", "PT", "PVT", "RE",
        "RH", "CM", "PCN", "CL", "PF", "SPA", "SPF", "MG", "LGd", "LGv", "RT",
        "IGL", "MH", "LH", "AD", "AV", "AM", "IAD", "IAM", "VM", "VPMpc", "VPLpc",
        "POL", "SGN", "Eth", "Xi", "IMD", "SMT", "PR", "PIL", "MGd", "MGv", "MGm",
    ),
    "HY": (
        "HY", "ZI", "LHA", "LPO", "MPO", "PVH", "VMH", "DMH", "ARH", "TU",
        "MM", "LM", "SUM", "PH", "AHN", "PVHd", "PVi", "PVp", "PVpo", "SBPV",
    ),
    "MB": (
        "SCs", "SCm", "SCig", "SCiw", "SCdg", "SCop", "PAG", "MRN", "RN", "VTA",
        "SN", "SNc", "SNr", "SNl", "IPN", "APN", "NOT", "PPT", "CUN", "III",
        "IV", "EW", "MB", "IC", "NB", "RR", "VTN", "AT", "LT", "DT", "NPC",
    ),
    "HB": (
        "P", "MY", "PRN", "GRN", "IRN", "PGR", "MARN", "MDRN", "PARN", "NTS",
        "SPVI", "SPVO", "SPVC", "HB", "PRNr", "PRNc", "CS", "LC", "PB", "SOC",
        "VII", "XII", "LIN", "LRN", "MVE", "SUV", "PARN", "IRN", "V", "VI",
    ),
    "CB": (
        "CB", "CBN", "CBX", "SIM", "CUL", "FOTU", "PYR", "NOD", "ANcr", "COPY",
        "PFL", "FL", "FN", "IP", "DN", "CENT", "DEC", "FOTU", "UVU", "NOD",
    ),
}


def _cosmos_of(acronym):
    raw = str(acronym or "").strip()
    if not raw or raw.lower() in ("void", "nan", "root", "unassigned"):
        return None
    if raw in COSMOS_ID:
        return raw
    stem = raw.split("/")[0]
    best = None
    best_n = 0
    for cosmos, prefixes in _COSMOS_PREFIX.items():
        for p in prefixes:
            if stem == p or stem.startswith(p):
                if len(p) > best_n:
                    best, best_n = cosmos, len(p)
    return best


# name, suite, kind, align, metric
# kind: frame | class | forecast | cosmooth | anatomy
TASKS = {
    "ts1-wheel_speed": {
        "suite": "TS1",
        "kind": "frame",
        "align": "move_post",
        "target": "wheel_speed",
        "n_out": 1,
        "metric": "r2",
        "zscore": True,
    },
    "ts1-right_paw_speed": {
        "suite": "TS1",
        "kind": "frame",
        "align": "move_post",
        "target": "right_paw_speed",
        "n_out": 1,
        "metric": "r2",
        "zscore": True,
    },
    "ts1-left_paw_speed": {
        "suite": "TS1",
        "kind": "frame",
        "align": "move_post",
        "target": "left_paw_speed",
        "n_out": 1,
        "metric": "r2",
        "zscore": True,
    },
    "ts1-whisker_motion_energy": {
        "suite": "TS1",
        "kind": "frame",
        "align": "move_post",
        "target": "whisker",
        "n_out": 1,
        "metric": "r2",
        "zscore": True,
    },
    "ts1-licking_rate": {
        "suite": "TS1",
        "kind": "frame",
        "align": "feedback_post",
        "target": "licking_rate",
        "n_out": 1,
        "metric": "d2",
        "zscore": False,
    },
    "ts1-choice": {
        "suite": "TS1",
        "kind": "class",
        "align": "move_pre",
        "target": "choice",
        "n_out": 2,
        "metric": "bal_acc",
        "zscore": False,
    },
    "ts1-reward": {
        "suite": "TS1",
        "kind": "class",
        "align": "feedback_post",
        "target": "reward",
        "n_out": 2,
        "metric": "bal_acc",
        "zscore": False,
    },
    "ts1-stimulus_contrast": {
        "suite": "TS1",
        "kind": "class",
        "align": "stim",
        "target": "contrast",
        "n_out": 5,
        "metric": "bal_acc",
        "zscore": False,
    },
    "ts2-forecast": {
        "suite": "TS2",
        "kind": "forecast",
        "align": "move_post",
        "target": "counts",
        "n_out": 1,
        "metric": "d2",
        "zscore": False,
    },
    "ts2-cosmooth": {
        "suite": "TS2",
        "kind": "cosmooth",
        "align": "move_post",
        "target": "counts",
        "n_out": 1,
        "metric": "d2",
        "zscore": False,
    },
    "ts3-cosmos": {
        "suite": "TS3",
        "kind": "anatomy",
        "align": "move_post",
        "target": "cosmos",
        "n_out": len(COSMOS),
        "metric": "macro_f1",
        "zscore": False,
    },
}

NEURAL = (
    "attn_a1_gru",
    "attn_a1_transformer",
    "attn_anat_transformer",
    "hierarchical",
    "gnn_inner",
    "gnn_outer",
)
MODELS = ("mean", "ar") + NEURAL
TRIAL_MODELS = (
    "mean",
    "ar",
    "attn_a1_gru",
    "attn_a1_transformer",
    "attn_anat_transformer",
    "hierarchical",
    "gnn_inner",
)
TRIAL_TASKS = (
    "ts1-wheel_speed",
    "ts1-right_paw_speed",
    "ts1-choice",
    "ts1-reward",
    "ts1-stimulus_contrast",
    "ts2-forecast",
    "ts2-cosmooth",
    "ts3-cosmos",
)


def log(msg):
    print(f"[{datetime.now().strftime('%H:%M:%S')}] {msg}", flush=True)


def set_seed(seed=SEED):
    import random

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def utc_now():
    return datetime.now(timezone.utc).isoformat()


def align_times(stim, move, feedback, align):
    stim, move, feedback = float(stim), float(move), float(feedback)
    if align == "stim":
        return stim - 0.9, stim + 0.1
    if align == "move_post":
        return move, move + 1.0
    if align == "move_pre":
        return move - 1.0, move
    if align == "feedback_post":
        return feedback, feedback + 1.0
    raise ValueError(align)


def contrast_class(brightness):
    x = float(brightness)
    if not np.isfinite(x):
        return None
    j = int(np.argmin([abs(x - c) for c in CONTRAST_LEVELS]))
    if abs(x - CONTRAST_LEVELS[j]) > 0.02:
        return None
    return j


def choice_class(rec):
    v = rec.get("choice_ibl")
    if v is None or (isinstance(v, float) and not np.isfinite(v)):
        side = str(rec.get("choice") or "")
        if side == "left":
            return 1
        if side == "right":
            return 0
        return None
    v = int(v)
    if v == 1:
        return 1
    if v == -1:
        return 0
    return None


def bin_edges(t0):
    left = t0 + np.arange(N_BINS, dtype=np.float64) * BINSIZE
    return left, left + BINSIZE


def bin_events(times, t0, scale=1.0):
    out = np.zeros(N_BINS, dtype=np.float32)
    if times is None:
        return out
    t = np.asarray(times, dtype=float).reshape(-1)
    t = t[np.isfinite(t)]
    if t.size == 0:
        return out
    rel = t - t0
    ok = (rel >= 0.0) & (rel < N_BINS * BINSIZE)
    b = np.floor(rel[ok] / BINSIZE).astype(int)
    np.add.at(out, b, 1.0)
    return out * np.float32(scale)


def bin_spikes(times, clusters, t0, n_units):
    counts = np.zeros((N_BINS, n_units), dtype=np.float32)
    if times is None or clusters is None or n_units <= 0:
        return counts
    t = np.asarray(times, dtype=float)
    u = np.asarray(clusters, dtype=np.int64)
    if t.size == 0:
        return counts
    rel = t - t0
    ok = (rel >= 0.0) & (rel < N_BINS * BINSIZE) & (u >= 0) & (u < n_units)
    if not np.any(ok):
        return counts
    b = np.floor(rel[ok] / BINSIZE).astype(np.int64)
    np.add.at(counts, (b, u[ok]), 1.0)
    return counts


def resample_trace(src_t, src_v, t0):
    _, right = bin_edges(t0)
    dst = np.full(N_BINS, np.nan, dtype=np.float32)
    t = np.asarray(src_t, dtype=float).reshape(-1)
    v = np.asarray(src_v, dtype=float).reshape(-1)
    n = min(t.size, v.size)
    if n < 2:
        return dst
    out = fd.interpolate_to(t[:n], v[:n], right)
    return np.asarray(out, dtype=np.float32)


def paw_speed_on_grid(pose, camera, feature, t0, scale=1.0):
    out = np.full(N_BINS, np.nan, dtype=np.float32)
    if not pose or camera not in pose:
        return out
    t, x, y = fd.camera_arrays(pose[camera], feature)
    if t.size < 4:
        return out
    _, right = bin_edges(t0)
    xs = fd.interpolate_to(t, np.asarray(x, dtype=float) * scale, right)
    ys = fd.interpolate_to(t, np.asarray(y, dtype=float) * scale, right)
    vx = np.gradient(xs, BINSIZE)
    vy = np.gradient(ys, BINSIZE)
    speed = np.hypot(vx, vy).astype(np.float32)
    speed[~np.isfinite(xs) | ~np.isfinite(ys)] = np.nan
    return speed


def slice_from_stim_grid(values, t0, stim, n_src):
    """Map a stimOn-aligned 20 ms series onto a BWB window, padding with NaN."""
    out = np.full(N_BINS, np.nan, dtype=np.float32)
    src = np.asarray(values, dtype=np.float32).reshape(-1)
    n = min(int(n_src), src.size)
    if n <= 0:
        return out
    src_left = float(stim) + np.arange(n, dtype=np.float64) * BINSIZE
    src_right = src_left + BINSIZE
    _, dst_right = bin_edges(t0)
    for i, tr in enumerate(dst_right):
        j = int(np.argmin(np.abs(src_right - tr)))
        if abs(src_right[j] - tr) <= BINSIZE * 0.51:
            out[i] = src[j]
    return out


def slice_counts_from_stim(counts, t0, stim):
    src = np.asarray(counts, dtype=np.float32)
    n, n_units = src.shape
    out = np.zeros((N_BINS, n_units), dtype=np.float32)
    src_right = float(stim) + (np.arange(n) + 1) * BINSIZE
    _, dst_right = bin_edges(t0)
    for i, tr in enumerate(dst_right):
        j = int(np.argmin(np.abs(src_right - tr)))
        if abs(src_right[j] - tr) <= BINSIZE * 0.51:
            out[i] = src[j]
    return out


def load_payload(path):
    with path.open("rb") as fh:
        return pickle.load(fh)


def list_session_paths():
    return sorted(p for p in SESS.glob("*.pkl") if not p.name.endswith(".tmp"))


def connect_one():
    return fd.connect_one()


def session_spikes_from_pickle(payload):
    """Lift per-trial stim-relative spike times back onto the session clock.

    Spikes exist only inside each trial's stimOn→feedback interval. BWB 1 s
    windows that extend outside that interval are zero-padded for those bins.
    """
    times = []
    clust = []
    for rec in payload.get("trials") or []:
        stim = rec.get("stimOn_times")
        st = np.asarray(rec.get("spike_times", []), dtype=float).reshape(-1)
        su = np.asarray(rec.get("spike_unit_idx", []), dtype=np.int32).reshape(-1)
        if stim is None or st.size == 0 or su.size == 0:
            continue
        n = min(st.size, su.size)
        times.append(st[:n] + float(stim))
        clust.append(su[:n])
    if not times:
        return np.array([], dtype=float), np.array([], dtype=np.int32)
    return np.concatenate(times), np.concatenate(clust)


def load_session_streams(one, payload):
    eid = str(payload["eid"])
    loader = fd.SessionLoader(one=one, eid=eid)
    wheel_t, wheel_v = fd.load_wheel_arrays(loader)
    lp = fd.load_pose_copy(loader, "lightningPose")
    dlc = fd.load_pose_copy(loader, "dlc")
    licks = None
    whisker_t = whisker_v = None
    try:
        obj = one.load_object(eid, "licks")
        licks = np.asarray(obj["times"], dtype=float)
    except Exception:
        try:
            licks = np.asarray(one.load_dataset(eid, "_ibl_licks.times"), dtype=float)
        except Exception:
            licks = None
    try:
        whisker_v = np.asarray(one.load_dataset(eid, "leftCamera.ROIMotionEnergy"), dtype=float).reshape(-1)
        whisker_t = np.asarray(one.load_dataset(eid, "leftCamera.times"), dtype=float).reshape(-1)
    except Exception:
        whisker_t = whisker_v = None
    spike_t, spike_u = session_spikes_from_pickle(payload)
    return {
        "wheel_t": wheel_t,
        "wheel_v": wheel_v,
        "lp": lp,
        "dlc": dlc,
        "licks": licks,
        "whisker_t": whisker_t,
        "whisker_v": whisker_v,
        "spike_t": spike_t,
        "spike_u": spike_u,
        "source": "one",
    }


def load_matched_spikes(one, payload):
    units = payload["units"]
    uuid_col = "uuid" if "uuid" in units.columns else "uuids"
    uuid_to_i = {str(u): i for i, u in enumerate(units[uuid_col].astype(str))}
    times = []
    clust = []
    pids = list(payload.get("pids") or [])
    names = list(payload.get("probe_names") or [])
    if len(names) < len(pids):
        names = names + [None] * (len(pids) - len(names))
    for pid, pname in zip(pids, names):
        try:
            loader = fd.SpikeSortingLoader(pid=pid, one=one, pname=pname, eid=str(payload["eid"]))
            spikes, clusters, channels = loader.load_spike_sorting(revision=SPIKE_REVISION)
            if not spikes:
                continue
            labeled = fd.SpikeSortingLoader.merge_clusters(spikes, clusters, channels).to_df()
            col = "uuids" if "uuids" in labeled.columns else "uuid"
            remap = np.full(int(np.max(labeled.index.to_numpy()) + 1) if len(labeled) else 1, -1, dtype=np.int32)
            for idx, u in zip(labeled.index.to_numpy(), labeled[col].astype(str)):
                if str(u) in uuid_to_i:
                    ii = int(idx)
                    if ii >= remap.size:
                        remap = np.pad(remap, (0, ii - remap.size + 1), constant_values=-1)
                    remap[ii] = uuid_to_i[str(u)]
            raw_c = np.asarray(spikes["clusters"], dtype=np.int64)
            ok = (raw_c >= 0) & (raw_c < remap.size)
            new_id = remap[np.clip(raw_c, 0, remap.size - 1)]
            keep = ok & (new_id >= 0)
            if not np.any(keep):
                continue
            times.append(np.asarray(spikes["times"], dtype=float)[keep])
            clust.append(new_id[keep].astype(np.int32))
            log(f"    spikes {pname or pid}: matched {int(keep.sum())} of {raw_c.size}")
        except Exception as exc:
            log(f"    spike load failed {pname or pid}: {exc}")
    if not times:
        return np.array([], dtype=float), np.array([], dtype=np.int32)
    return np.concatenate(times), np.concatenate(clust)


def cap_units(xyz, areas, cosmos, counts_ref, max_units=MAX_UNITS):
    n = int(xyz.shape[0])
    if n <= max_units:
        keep = np.arange(n)
        return xyz, areas, cosmos, keep
    rate = np.asarray(counts_ref, dtype=np.float32).mean(axis=0)
    keep = np.argsort(-rate)[:max_units]
    keep.sort()
    return xyz[keep], areas[keep], cosmos[keep], keep


def build_session_windows(payload, streams, offline=False):
    units = payload["units"]
    xyz = np.stack(
        [
            units["allen_x"].to_numpy(dtype=np.float32),
            units["allen_y"].to_numpy(dtype=np.float32),
            units["allen_z"].to_numpy(dtype=np.float32),
        ],
        axis=1,
    )
    areas = units["brain_area"].astype(str).to_numpy()
    cosmos = np.array([_cosmos_of(a) for a in areas], dtype=object)
    n_units = int(xyz.shape[0])
    rate_ref = np.zeros(n_units, dtype=np.float32)
    rows = []
    source = streams.get("source", "offline")
    for rec in payload["trials"]:
        stim = rec.get("stimOn_times")
        move = rec.get("firstMovement_times")
        fb = rec.get("feedback_times")
        if stim is None or move is None or fb is None:
            continue
        if not all(np.isfinite([stim, move, fb])):
            continue
        stim, move, fb = float(stim), float(move), float(fb)
        choice = choice_class(rec)
        contrast = contrast_class(rec.get("brightness"))
        reward = rec.get("reward")
        if reward is None:
            continue
        reward = int(reward)
        if reward not in (0, 1):
            continue
        n_src = int(rec.get("n_bins") or 0)
        src_counts = np.asarray(rec.get("spike_counts", []), dtype=np.float32)
        if src_counts.ndim != 2 or src_counts.shape[1] != n_units:
            continue
        rate_ref += src_counts.sum(axis=0)
        windows = {}
        for align in ("stim", "move_post", "move_pre", "feedback_post"):
            t0, _t1 = align_times(stim, move, fb, align)
            if source == "one" and streams.get("spike_t") is not None and streams["spike_t"].size:
                counts = bin_spikes(streams["spike_t"], streams["spike_u"], t0, n_units)
            else:
                counts = slice_counts_from_stim(src_counts, t0, stim)
            wheel = resample_trace(streams.get("wheel_t"), streams.get("wheel_v"), t0)
            if not np.isfinite(wheel).any():
                wheel = slice_from_stim_grid(
                    np.abs(np.asarray(rec.get("wheel_velocity", []), dtype=np.float32)),
                    t0,
                    stim,
                    n_src,
                )
            else:
                wheel = np.abs(wheel)
            lp, dlc = streams.get("lp") or {}, streams.get("dlc") or {}
            r_paw = paw_speed_on_grid(lp, "rightCamera", "paw_r", t0, scale=fd.RIGHT_CAMERA_SCALE)
            if not np.isfinite(r_paw).any():
                r_paw = paw_speed_on_grid(dlc, "rightCamera", "paw_r", t0, scale=fd.RIGHT_CAMERA_SCALE)
            if not np.isfinite(r_paw).any():
                r_paw = slice_from_stim_grid(rec.get("lp_speed"), t0, stim, n_src)
            l_paw = paw_speed_on_grid(lp, "leftCamera", "paw_r", t0, scale=1.0)
            if not np.isfinite(l_paw).any():
                l_paw = paw_speed_on_grid(dlc, "leftCamera", "paw_r", t0, scale=1.0)
            whisker = resample_trace(streams.get("whisker_t"), streams.get("whisker_v"), t0)
            licks = bin_events(streams.get("licks"), t0, scale=1.0 / BINSIZE)
            windows[align] = {
                "t0": t0,
                "counts": counts,
                "wheel_speed": wheel,
                "right_paw_speed": r_paw,
                "left_paw_speed": l_paw,
                "whisker": whisker,
                "licking_rate": licks,
            }
        rows.append(
            {
                "eid": str(payload["eid"]),
                "mouse_id": str(payload.get("mouse_id")),
                "trial_index": rec.get("trial_index"),
                "choice": choice,
                "contrast": contrast,
                "reward": reward,
                "windows": windows,
                "stim": stim,
                "move": move,
                "feedback": fb,
                "source": source,
            }
        )
    keep_xyz, keep_areas, keep_cosmos, keep = cap_units(xyz, areas, cosmos, rate_ref.reshape(1, -1))
    for row in rows:
        for w in row["windows"].values():
            w["counts"] = w["counts"][:, keep]
    return {
        "eid": str(payload["eid"]),
        "mouse_id": str(payload.get("mouse_id")),
        "lab": str(payload.get("lab")),
        "xyz": keep_xyz.astype(np.float32),
        "areas": keep_areas,
        "cosmos": keep_cosmos,
        "n_units": int(keep_xyz.shape[0]),
        "trials": rows,
        "source": source,
        "offline": bool(offline),
        "n_units_raw": n_units,
    }


def extract_session(path, one=None, offline=False, force=False):
    eid = path.stem
    CACHE.mkdir(parents=True, exist_ok=True)
    cache_path = CACHE / f"{eid}_{'offline' if offline else 'one'}.pkl"
    if cache_path.exists() and not force:
        with cache_path.open("rb") as fh:
            blob = pickle.load(fh)
        log(f"  cache {cache_path.name}: {len(blob['trials'])} trials, {blob['n_units']} units")
        return blob
    payload = load_payload(path)
    streams = {
        "wheel_t": None,
        "wheel_v": None,
        "lp": {},
        "dlc": {},
        "licks": None,
        "whisker_t": None,
        "whisker_v": None,
        "spike_t": None,
        "spike_u": None,
        "source": "offline",
    }
    if not offline and one is not None:
        try:
            log(f"  ONE streams {eid[:8]} {payload.get('mouse_id')}")
            streams = load_session_streams(one, payload)
        except Exception as exc:
            log(f"  ONE failed ({exc}); falling back to pickle windows")
            traceback.print_exc()
            streams["source"] = "offline"
    blob = build_session_windows(payload, streams, offline=streams["source"] != "one")
    with cache_path.open("wb") as fh:
        pickle.dump(blob, fh, protocol=4)
    log(
        f"  wrote {cache_path.name}: {len(blob['trials'])} trials, "
        f"{blob['n_units']} units, source={blob['source']}"
    )
    return blob


def causal_split(n, train=0.40, val=0.20):
    n = int(n)
    n_tr = int(math.floor(n * train))
    n_va = int(math.floor(n * val))
    n_te = n - n_tr - n_va
    if n_te < 8 or n_tr < 8:
        return None
    idx = np.arange(n)
    return idx[:n_tr], idx[n_tr : n_tr + n_va], idx[n_tr + n_va :]


def interleaved_blocks(n, block=25, train=0.40, val=0.20, seed=SEED):
    """TS2-style interleaved blocks on an ordered trial list."""
    n = int(n)
    if n < MIN_TRIALS:
        return None
    rng = np.random.RandomState(seed)
    n_blocks = max(int(math.ceil(n / block)), 3)
    labels = np.array(["train", "val", "test"] * (n_blocks // 3 + 1), dtype=object)[:n_blocks]
    rng.shuffle(labels)
    assign = np.empty(n, dtype=object)
    for i, lab in enumerate(labels):
        assign[i * block : (i + 1) * block] = lab
    tr = np.flatnonzero(assign == "train")
    va = np.flatnonzero(assign == "val")
    te = np.flatnonzero(assign == "test")
    if min(tr.size, va.size, te.size) < 6:
        return causal_split(n)
    return tr, va, te


def bind_task(session, task_name):
    spec = TASKS[task_name]
    align = spec["align"]
    xyz = session["xyz"]
    areas = session["areas"]
    cosmos = session["cosmos"]
    trials = []
    for rec in session["trials"]:
        w = rec["windows"][align]
        counts = np.asarray(w["counts"], dtype=np.float32)
        if counts.shape[0] != N_BINS or counts.shape[1] < MIN_UNITS:
            continue
        item = {
            "eid": rec["eid"],
            "mouse_id": rec["mouse_id"],
            "trial_index": rec["trial_index"],
            "counts": counts,
            "spikes": np.log1p(np.clip(counts, 0, None)).astype(np.float32),
            "xyz": xyz,
            "areas": areas,
            "cosmos": cosmos,
        }
        kind = spec["kind"]
        if kind == "frame":
            y = np.asarray(w[spec["target"]], dtype=np.float32).reshape(-1)
            if y.size != N_BINS:
                continue
            valid = np.isfinite(y)
            if int(valid.sum()) < 10:
                continue
            item["y"] = np.where(valid, y, 0.0).astype(np.float32).reshape(-1, 1)
            item["y_valid"] = valid
            item["y_class"] = None
        elif kind == "class":
            lab = rec.get(spec["target"])
            if lab is None:
                continue
            lab = int(lab)
            if lab < 0 or lab >= spec["n_out"]:
                continue
            item["y"] = np.full((N_BINS, 1), lab, dtype=np.float32)
            item["y_valid"] = np.ones(N_BINS, dtype=bool)
            item["y_class"] = lab
        elif kind in ("forecast", "cosmooth"):
            item["y"] = counts
            item["y_valid"] = np.ones(N_BINS, dtype=bool)
            item["y_class"] = None
        elif kind == "anatomy":
            labels = np.array(
                [COSMOS_ID[c] if c in COSMOS_ID else -1 for c in cosmos], dtype=np.int64
            )
            if int((labels >= 0).sum()) < 4:
                continue
            item["y"] = np.zeros((N_BINS, 1), dtype=np.float32)
            item["y_valid"] = np.ones(N_BINS, dtype=bool)
            item["y_class"] = None
            item["cosmos_id"] = labels
        else:
            continue
        trials.append(item)
    return trials


def make_stats(trials, idx):
    idx = np.asarray(idx, dtype=int)
    xyz = np.concatenate([trials[i]["xyz"] for i in idx], axis=0)
    xyz_mean = np.nanmean(xyz, axis=0).astype(np.float32)
    xyz_std = np.nanstd(xyz, axis=0).astype(np.float32)
    xyz_std = np.where(xyz_std < 1e-8, 1.0, xyz_std).astype(np.float32)
    areas = [a for i in idx for a in trials[i]["areas"]]
    vocab = {"<unk>": 0}
    for a in sorted(set(areas)):
        if a not in vocab:
            vocab[a] = len(vocab)
    buckets = {}
    for i in idx:
        t = trials[i]
        buckets.setdefault(t["eid"], []).append(t["spikes"].mean(axis=0))
    rates = {eid: np.mean(vs, axis=0).astype(np.float32) for eid, vs in buckets.items()}
    return {
        "vocab": vocab,
        "n_areas": len(vocab),
        "xyz_mean": xyz_mean,
        "xyz_std": xyz_std,
        "rates": rates,
    }


def frame_scaler(trials, idx):
    ys = []
    for i in np.asarray(idx, dtype=int):
        t = trials[int(i)]
        y = np.asarray(t["y"], dtype=np.float32).reshape(-1)
        v = np.asarray(t["y_valid"], dtype=bool).reshape(-1)
        ys.append(y[v[: y.size]])
    cat = np.concatenate(ys) if ys else np.zeros(1, dtype=np.float32)
    mean = np.float32(np.nanmean(cat)) if cat.size else np.float32(0.0)
    std = np.float32(np.nanstd(cat)) if cat.size else np.float32(1.0)
    if not np.isfinite(std) or std < 1e-6:
        std = np.float32(1.0)
    return mean.reshape(1), std.reshape(1)


class BWBDS(Dataset):
    def __init__(self, trials, idx, stats, spec, y_mean, y_std, hold_units=None):
        self.trials = trials
        self.idx = np.asarray(idx, dtype=int)
        self.stats = stats
        self.spec = spec
        self.y_mean = np.asarray(y_mean, dtype=np.float32).reshape(-1)
        self.y_std = np.asarray(y_std, dtype=np.float32).reshape(-1)
        self.hold_units = hold_units

    def __len__(self):
        return len(self.idx)

    def __getitem__(self, j):
        t = self.trials[int(self.idx[j])]
        spikes = np.asarray(t["spikes"], dtype=np.float32)
        counts = np.asarray(t["counts"], dtype=np.float32)
        n_t, n_u = spikes.shape
        xyz = (t["xyz"] - self.stats["xyz_mean"]) / self.stats["xyz_std"]
        xyz = np.nan_to_num(xyz, nan=0.0).astype(np.float32)
        areas = mv.encode_areas(t["areas"], self.stats["vocab"])
        rate = self.stats["rates"][t["eid"]]
        if rate.shape[0] != n_u:
            r = np.zeros(n_u, dtype=np.float32)
            m = min(n_u, rate.shape[0])
            r[:m] = rate[:m]
            rate = r
        kind = self.spec["kind"]
        if kind == "anatomy":
            y = np.asarray(t["cosmos_id"], dtype=np.int64)
            y_valid = y >= 0
            y_frame = np.zeros((n_t, 1), dtype=np.float32)
            y_class = -1
        elif kind in ("forecast", "cosmooth"):
            y = counts
            y_valid = np.ones(n_t, dtype=bool)
            y_frame = counts.mean(axis=1, keepdims=True).astype(np.float32)
            y_class = -1
        elif kind == "class":
            y_class = int(t["y_class"])
            y_valid = np.ones(n_t, dtype=bool)
            y_frame = np.full((n_t, 1), y_class, dtype=np.float32)
            y = y_frame
        else:
            y = np.asarray(t["y"], dtype=np.float32).reshape(n_t, -1)
            y_valid = np.asarray(t["y_valid"], dtype=bool).reshape(-1)
            if self.spec.get("zscore"):
                y = np.where(y_valid[:, None], (y - self.y_mean) / self.y_std, 0.0)
            y_frame = y
            y_class = -1
        hold = np.zeros(n_u, dtype=bool)
        if self.hold_units is not None:
            hold[: min(n_u, self.hold_units.size)] = self.hold_units[:n_u]
        return {
            "spikes": spikes,
            "counts": counts,
            "y_frame": y_frame.astype(np.float32),
            "y_valid": y_valid.astype(bool),
            "y_class": y_class,
            "xyz": xyz,
            "areas": areas,
            "rate": rate.astype(np.float32),
            "eid": str(t["eid"]),
            "mouse": str(t["mouse_id"]),
            "hold": hold,
            "cosmos": np.asarray(t.get("cosmos_id", np.full(n_u, -1)), dtype=np.int64),
        }


def collate_bwb(batch):
    B = len(batch)
    T = max(b["spikes"].shape[0] for b in batch)
    N = max(b["spikes"].shape[1] for b in batch)
    y_dim = int(batch[0]["y_frame"].shape[-1])
    x = torch.zeros(B, N, T)
    counts = torch.zeros(B, N, T)
    y = torch.zeros(B, T, y_dim)
    xyz = torch.zeros(B, N, 3)
    area = torch.zeros(B, N, dtype=torch.long)
    rate = torch.zeros(B, N)
    mask_t = torch.zeros(B, T, dtype=torch.bool)
    mask_n = torch.zeros(B, N, dtype=torch.bool)
    hold = torch.zeros(B, N, dtype=torch.bool)
    cosmos = torch.full((B, N), -1, dtype=torch.long)
    lengths = torch.zeros(B, dtype=torch.long)
    y_class = torch.zeros(B, dtype=torch.long)
    eids, mice = [], []
    for i, b in enumerate(batch):
        n, tt = b["spikes"].shape[1], b["spikes"].shape[0]
        x[i, :n, :tt] = torch.from_numpy(b["spikes"].T)
        counts[i, :n, :tt] = torch.from_numpy(b["counts"].T)
        y[i, :tt] = torch.from_numpy(b["y_frame"])
        xyz[i, :n] = torch.from_numpy(b["xyz"])
        area[i, :n] = torch.from_numpy(b["areas"])
        rate[i, :n] = torch.from_numpy(b["rate"])
        mask_t[i, :tt] = torch.from_numpy(b["y_valid"][:tt])
        mask_n[i, :n] = True
        hold[i, :n] = torch.from_numpy(b["hold"][:n])
        cosmos[i, :n] = torch.from_numpy(b["cosmos"][:n])
        lengths[i] = tt
        y_class[i] = int(b["y_class"])
        eids.append(b["eid"])
        mice.append(b["mouse"])
    return {
        "x": x,
        "counts": counts,
        "y": y,
        "xyz": xyz,
        "area": area,
        "rate": rate,
        "mask_t": mask_t,
        "mask_n": mask_n,
        "hold": hold,
        "cosmos": cosmos,
        "lengths": lengths,
        "y_class": y_class,
        "eids": eids,
        "mice": mice,
        "rho": torch.zeros(B, N, N),
    }


def resize_head(head, n_out):
    if head is None:
        return
    if isinstance(head, clean.HeadTap):
        resize_head(head.head, n_out)
        return
    net = getattr(head, "net", None)
    if net is None:
        return
    last = net[-1]
    if getattr(last, "out_features", None) == n_out:
        head.n_out = n_out
        return
    layer = nn.Linear(last.in_features, n_out)
    nn.init.xavier_uniform_(layer.weight)
    if layer.bias is not None:
        nn.init.zeros_(layer.bias)
    net[-1] = layer
    head.n_out = n_out


def build_backbone(name, stats, trials, tr_idx, n_out, classify):
    if name in ("attn_a1_gru", "attn_a1_transformer", "attn_anat_transformer"):
        model = clean.build_neural(name, stats, trials, tr_idx, "direction" if classify else "trace")
        resize_head(model.head, n_out)
        return model
    if name == "gnn_inner":
        model = clean.GraphWithHeads(
            stats["n_areas"], region_gnn=False, vocab=stats.get("vocab"), classify=classify
        )
        resize_head(model.head, n_out)
        return model
    if name == "gnn_outer":
        model = clean.GraphWithHeads(
            stats["n_areas"], region_gnn=True, vocab=stats.get("vocab"), classify=classify
        )
        resize_head(model.head, n_out)
        return model
    if name == "hierarchical":
        eids, mice = hier.ids_of(trials, tr_idx)
        inner = hier.HierarchicalGRU(
            stats["n_areas"], eids, mice, shared_encoder=True, use_anatomy=False
        )
        model = clean.HierWithHeads(inner, classify=classify)
        resize_head(model.inner.dec3, n_out)
        return model
    raise ValueError(name)


class IdentityHead(nn.Module):
    def forward(self, h):
        return h


def _swap_head(module, name, replacement):
    if name == "hierarchical":
        module.inner.dec3 = replacement
        return
    if isinstance(getattr(module, "head", None), clean.HeadTap):
        module.head.head = replacement
        return
    if hasattr(module, "head"):
        module.head = replacement


class BWBWrapper(nn.Module):
    """Task head on top of an ExperimentsClean backbone."""

    def __init__(self, inner, name, spec, d=None):
        super().__init__()
        self.inner = inner
        self.name = name
        self.spec = spec
        self.kind = spec["kind"]
        d = d or mv.RNN_HIDDEN
        if name in ("attn_anat_transformer", "attn_a1_transformer"):
            d = mv.D_MODEL
        self.d = d
        if self.kind in ("cosmooth", "anatomy"):
            _swap_head(inner, name, IdentityHead())
        if self.kind == "cosmooth":
            self.unit_mlp = nn.Sequential(nn.Linear(4, d), nn.GELU(), nn.Linear(d, d))
            self.ctx = nn.Linear(d, d)
        if self.kind == "anatomy":
            self.cls = nn.Linear(d, spec["n_out"])

    def _forward_inner(self, batch, dev, train):
        kind = "gnn" if self.name.startswith("gnn") else self.name
        drop = mv.UNIT_DROPOUT if train else 0.0
        x = batch["x"].to(dev)
        if self.kind == "forecast":
            x = x.clone()
            x[:, :, -FORECAST_BINS:] = 0.0
        if self.kind == "cosmooth":
            mask_n = batch["mask_n"].to(dev) & ~batch["hold"].to(dev)
        else:
            mask_n = batch["mask_n"].to(dev)
        if self.kind == "anatomy":
            drop = 0.0
        args = dict(
            x=x,
            xyz=batch["xyz"].to(dev),
            area=batch["area"].to(dev),
            rate=batch["rate"].to(dev),
            mask_n=mask_n,
            mask_t=batch["mask_t"].to(dev),
            lengths=batch["lengths"],
            unit_drop=drop,
        )
        if kind == "gnn":
            return self.inner(
                args["x"],
                args["xyz"],
                args["area"],
                args["rate"],
                args["mask_n"],
                args["mask_t"],
                args["lengths"],
                rho=batch["rho"].to(dev),
                unit_drop=drop,
                compute_smooth=False,
            )
        if self.name == "hierarchical":
            return self.inner(
                args["x"],
                args["xyz"],
                args["area"],
                args["rate"],
                args["mask_n"],
                args["mask_t"],
                args["lengths"],
                batch["eids"],
                batch["mice"],
                unit_drop=drop,
            )
        return self.inner(
            args["x"],
            args["xyz"],
            args["area"],
            args["rate"],
            args["mask_n"],
            args["mask_t"],
            args["lengths"],
            unit_drop=drop,
        )

    def _pred_tensor(self, out):
        if self.name.startswith("gnn"):
            pred = out[0]
        elif self.name == "hierarchical":
            pred = out[2]
        else:
            pred = out[0] if isinstance(out, tuple) else out
        return pred

    def _unit_encode(self, batch, dev, mask_n):
        x = batch["x"].to(dev)
        xyz = batch["xyz"].to(dev)
        area = batch["area"].to(dev)
        rate = batch["rate"].to(dev)
        if self.name == "hierarchical":
            e, _ = self.inner.inner.encode_units(x, xyz, area, rate, mask_n, unit_drop=0.0)
            return e
        enc = getattr(self.inner, "enc", None)
        if enc is None:
            return None
        return enc(x, xyz, area, rate, mask_n)

    def forward(self, batch, dev, train=False):
        out = self._forward_inner(batch, dev, train)
        pred = self._pred_tensor(out)
        if self.kind in ("frame", "class", "forecast"):
            return pred
        if self.kind == "cosmooth":
            ctx = self.ctx(pred)
            static = torch.cat([batch["rate"].to(dev).unsqueeze(-1), batch["xyz"].to(dev)], dim=-1)
            u = self.unit_mlp(static)
            return (ctx.unsqueeze(1) * u.unsqueeze(2)).sum(-1)
        if self.kind == "anatomy":
            mask_n = batch["mask_n"].to(dev)
            e = self._unit_encode(batch, dev, mask_n)
            if e is None:
                pooled = pred.mean(dim=1)
                logits = self.cls(pooled).unsqueeze(1).expand(-1, batch["mask_n"].size(1), -1)
                return logits
            mask_t = batch["mask_t"].to(dev)[:, None, :, None]
            e = e.masked_fill(~mask_n[:, :, None, None], 0.0)
            denom = mask_t.squeeze(-1).sum(dim=2).clamp(min=1).unsqueeze(-1)
            z = (e * mask_t).sum(dim=2) / denom
            return self.cls(z)
        return pred


def poisson_d2(y, mu):
    y = np.asarray(y, dtype=np.float64).reshape(-1)
    mu = np.clip(np.asarray(mu, dtype=np.float64).reshape(-1), 1e-8, None)
    n = min(y.size, mu.size)
    y, mu = y[:n], mu[:n]
    ok = np.isfinite(y) & np.isfinite(mu)
    y, mu = y[ok], mu[ok]
    if y.size < 8:
        return {"d2": float("nan"), "bps": float("nan")}
    ybar = float(np.mean(y))
    ybar = max(ybar, 1e-8)
    sat = np.maximum(y, 1e-8)
    l_pred = np.sum(mu - y * np.log(mu))
    l_sat = np.sum(sat - y * np.log(sat))
    l_null = np.sum(ybar - y * np.log(ybar))
    den = l_null - l_sat
    d2 = 1.0 - (l_pred - l_sat) / den if abs(den) > 1e-12 else float("nan")
    nsp = float(np.sum(y))
    bps = (l_null - l_pred) / (nsp * math.log(2.0)) if nsp > 0 else float("nan")
    return {"d2": float(d2), "bps": float(bps), "d2_clip": float(max(d2, 0.0)) if np.isfinite(d2) else float("nan")}


def score_r2(y, yhat):
    y = np.asarray(y, dtype=np.float64).reshape(-1)
    yhat = np.asarray(yhat, dtype=np.float64).reshape(-1)
    n = min(y.size, yhat.size)
    y, yhat = y[:n], yhat[:n]
    ok = np.isfinite(y) & np.isfinite(yhat)
    y, yhat = y[ok], yhat[ok]
    if y.size < 8 or np.std(y) < 1e-12:
        return {"r2": float("nan"), "r2_clip": float("nan")}
    r2 = float(r2_score(y, yhat))
    return {"r2": r2, "r2_clip": float(max(r2, 0.0))}


def invert_z(pred, mean, std):
    return pred * float(std.reshape(-1)[0]) + float(mean.reshape(-1)[0])


def run_epoch(wrapper, loader, opt, dev, spec, train):
    if wrapper is not None:
        wrapper.train(train)
    tot = 0.0
    n = 0
    ctx = torch.enable_grad() if train else torch.no_grad()
    with ctx:
        for batch in loader:
            loss = task_loss(wrapper, batch, dev, spec, train)
            if train:
                opt.zero_grad(set_to_none=True)
                loss.backward()
                nn.utils.clip_grad_norm_(wrapper.parameters(), 5.0)
                opt.step()
            bs = int(batch["x"].size(0))
            tot += float(loss.detach().cpu()) * bs
            n += bs
    return tot / max(n, 1)


def task_loss(wrapper, batch, dev, spec, train):
    kind = spec["kind"]
    pred = wrapper(batch, dev, train=train)
    mask_t = batch["mask_t"].to(dev)
    if kind == "frame":
        y = batch["y"].to(dev)
        err = (pred[..., :1] - y) ** 2
        err = err.squeeze(-1) * mask_t.to(err.dtype)
        denom = mask_t.sum().clamp(min=1).to(err.dtype)
        return err.sum() / denom
    if kind == "class":
        logits = pred.masked_fill(~mask_t.unsqueeze(-1), 0.0)
        pooled = logits.sum(dim=1) / mask_t.sum(dim=1, keepdim=True).clamp(min=1).to(pred.dtype)
        return F.cross_entropy(pooled, batch["y_class"].to(dev))
    if kind == "forecast":
        counts = batch["counts"].to(dev)
        target = counts[:, :, -FORECAST_BINS:].mean(dim=1)
        mu = F.softplus(pred[:, -FORECAST_BINS:, :1].mean(dim=-1))
        return torch.mean(mu - target * torch.log(mu.clamp(min=1e-6)))
    if kind == "cosmooth":
        counts = batch["counts"].to(dev)
        hold = batch["hold"].to(dev)
        mu = F.softplus(pred)
        m = hold[:, :, None] & mask_t[:, None, :]
        if not bool(m.any()):
            return pred.sum() * 0.0
        y = counts
        return ((mu - y * torch.log(mu.clamp(min=1e-6))) * m.to(mu.dtype)).sum() / m.to(mu.dtype).sum().clamp(min=1)
    if kind == "anatomy":
        logits = pred
        lab = batch["cosmos"].to(dev)
        m = (lab >= 0) & batch["mask_n"].to(dev)
        if not bool(m.any()):
            return logits.sum() * 0.0
        return F.cross_entropy(logits[m], lab[m])
    raise ValueError(kind)


def collect_preds(wrapper, loader, dev, spec, y_mean, y_std):
    wrapper.eval()
    y_true, y_pred = [], []
    with torch.no_grad():
        for batch in loader:
            pred = wrapper(batch, dev, train=False)
            kind = spec["kind"]
            if kind == "frame":
                mask = batch["mask_t"].numpy()
                raw = pred[..., 0].cpu().numpy()
                if spec.get("zscore"):
                    raw = invert_z(raw, y_mean, y_std)
                gt = batch["y"][..., 0].numpy()
                if spec.get("zscore"):
                    gt = invert_z(gt, y_mean, y_std)
                for i in range(raw.shape[0]):
                    m = mask[i]
                    y_pred.append(raw[i, m])
                    y_true.append(gt[i, m])
            elif kind == "class":
                mask = batch["mask_t"].to(pred.device)
                logits = pred.masked_fill(~mask.unsqueeze(-1), 0.0)
                pooled = logits.sum(dim=1) / mask.sum(dim=1, keepdim=True).clamp(min=1).to(pred.dtype)
                y_pred.append(pooled.argmax(dim=-1).cpu().numpy())
                y_true.append(batch["y_class"].numpy())
            elif kind == "forecast":
                mu = F.softplus(pred[:, -FORECAST_BINS:, :1].mean(dim=-1)).cpu().numpy()
                gt = batch["counts"][:, :, -FORECAST_BINS:].mean(dim=1).numpy()
                y_pred.append(mu.reshape(-1))
                y_true.append(gt.reshape(-1))
            elif kind == "cosmooth":
                mu = F.softplus(pred).cpu().numpy()
                gt = batch["counts"].numpy()
                hold = batch["hold"].numpy()
                mask = batch["mask_t"].numpy()
                m = hold[:, :, None] & mask[:, None, :]
                y_pred.append(mu[m])
                y_true.append(gt[m])
            elif kind == "anatomy":
                lab = batch["cosmos"].numpy()
                m = (lab >= 0) & batch["mask_n"].numpy()
                y_pred.append(pred.argmax(dim=-1).cpu().numpy()[m])
                y_true.append(lab[m])
    if not y_true:
        return np.array([]), np.array([])
    return np.concatenate([np.asarray(a).reshape(-1) for a in y_true]), np.concatenate(
        [np.asarray(a).reshape(-1) for a in y_pred]
    )


def eval_metrics(spec, y, yhat):
    if y.size == 0:
        return {"score": float("nan")}
    if spec["metric"] == "r2":
        out = score_r2(y, yhat)
        out["score"] = out["r2_clip"]
        return out
    if spec["metric"] == "d2":
        out = poisson_d2(y, np.clip(yhat, 0, None))
        out["score"] = out["d2_clip"]
        return out
    if spec["metric"] == "bal_acc":
        acc = float(balanced_accuracy_score(y.astype(int), yhat.astype(int)))
        f1 = float(f1_score(y.astype(int), yhat.astype(int), average="macro", zero_division=0))
        return {"score": acc, "bal_acc": acc, "macro_f1": f1}
    if spec["metric"] == "macro_f1":
        f1 = float(f1_score(y.astype(int), yhat.astype(int), average="macro", zero_division=0))
        acc = float(balanced_accuracy_score(y.astype(int), yhat.astype(int)))
        return {"score": f1, "macro_f1": f1, "bal_acc": acc}
    return {"score": float("nan")}


def baseline_predict(name, spec, trials, tr_idx, te_idx, hold_units=None):
    tr = [trials[int(i)] for i in tr_idx]
    te = [trials[int(i)] for i in te_idx]
    kind = spec["kind"]
    if kind == "frame":
        ys = []
        for t in tr:
            y = np.asarray(t["y"]).reshape(-1)
            v = np.asarray(t["y_valid"]).reshape(-1)
            ys.append(y[v[: y.size]])
        mu = float(np.nanmean(np.concatenate(ys))) if ys else 0.0
        y_true, yhat = [], []
        for t in te:
            y = np.asarray(t["y"]).reshape(-1)
            v = np.asarray(t["y_valid"]).reshape(-1)
            yt = y[v[: y.size]]
            if name == "mean":
                yh = np.full_like(yt, mu)
            else:
                yh = np.roll(yt, 1)
                yh[0] = mu
            y_true.append(yt)
            yhat.append(yh)
        return np.concatenate(y_true), np.concatenate(yhat)
    if kind == "class":
        labs = np.array([int(t["y_class"]) for t in tr], dtype=int)
        maj = int(np.bincount(labs).argmax()) if labs.size else 0
        y_true = np.array([int(t["y_class"]) for t in te], dtype=int)
        if name == "mean":
            return y_true, np.full_like(y_true, maj)
        return y_true, np.full_like(y_true, maj)
    if kind == "forecast":
        y_true, yhat = [], []
        mu = np.mean([t["counts"][-FORECAST_BINS:].mean() for t in tr]) if tr else 0.1
        for t in te:
            gt = t["counts"][-FORECAST_BINS:].mean(axis=1)
            if name == "mean":
                yh = np.full_like(gt, mu)
            else:
                last = t["counts"][-FORECAST_BINS - 1 : -FORECAST_BINS].mean() if t["counts"].shape[0] > FORECAST_BINS else mu
                yh = np.full_like(gt, last)
            y_true.append(gt)
            yhat.append(yh)
        return np.concatenate(y_true), np.concatenate(yhat)
    if kind == "cosmooth":
        y_true, yhat = [], []
        hold = hold_units
        rates = []
        for t in tr:
            c = t["counts"]
            if hold is not None:
                rates.append(c[:, hold[: c.shape[1]]].mean())
            else:
                rates.append(c.mean())
        mu = float(np.mean(rates)) if rates else 0.1
        for t in te:
            c = t["counts"]
            h = hold[: c.shape[1]] if hold is not None else np.ones(c.shape[1], dtype=bool)
            gt = c[:, h].reshape(-1)
            y_true.append(gt)
            yhat.append(np.full_like(gt, mu))
        return np.concatenate(y_true), np.concatenate(yhat)
    if kind == "anatomy":
        y_true, yhat = [], []
        labs = np.concatenate([t["cosmos_id"][t["cosmos_id"] >= 0] for t in tr]) if tr else np.array([0])
        maj = int(np.bincount(np.clip(labs, 0, None)).argmax()) if labs.size else 0
        for t in te:
            lab = t["cosmos_id"]
            m = lab >= 0
            y_true.append(lab[m])
            yhat.append(np.full(int(m.sum()), maj, dtype=np.int64))
        return np.concatenate(y_true), np.concatenate(yhat)
    return np.array([]), np.array([])


def holdout_mask(n_units, seed):
    rng = np.random.RandomState(seed + 17)
    n_hold = max(int(round(n_units * COSMOOTH_FRAC)), 1)
    hold = np.zeros(n_units, dtype=bool)
    hold[rng.choice(n_units, size=n_hold, replace=False)] = True
    return hold


def train_neural(name, spec, trials, splits, stats, epochs, device, seed, hold_units):
    tr_idx, va_idx, te_idx = splits
    y_mean, y_std = (np.zeros(1), np.ones(1))
    if spec["kind"] == "frame" and spec.get("zscore"):
        y_mean, y_std = frame_scaler(trials, tr_idx)
    set_seed(seed)
    classify = spec["kind"] in ("class", "anatomy")
    inner = build_backbone(name, stats, trials, tr_idx, spec["n_out"], classify)
    wrapper = BWBWrapper(inner, name, spec)
    dev = device
    wrapper.to(dev)
    ds_tr = BWBDS(trials, tr_idx, stats, spec, y_mean, y_std, hold_units)
    ds_va = BWBDS(trials, va_idx, stats, spec, y_mean, y_std, hold_units)
    ds_te = BWBDS(trials, te_idx, stats, spec, y_mean, y_std, hold_units)
    tr_ld = DataLoader(ds_tr, batch_size=BATCH, shuffle=True, collate_fn=collate_bwb)
    va_ld = DataLoader(ds_va, batch_size=BATCH, shuffle=False, collate_fn=collate_bwb)
    te_ld = DataLoader(ds_te, batch_size=BATCH, shuffle=False, collate_fn=collate_bwb)
    opt = torch.optim.AdamW(wrapper.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    best = math.inf
    best_state = None
    bad = 0
    hist = []
    for epoch in range(1, epochs + 1):
        tr_loss = run_epoch(wrapper, tr_ld, opt, dev, spec, train=True)
        va_loss = run_epoch(wrapper, va_ld, None, dev, spec, train=False)
        hist.append({"epoch": epoch, "train": tr_loss, "val": va_loss})
        log(f"      {name:24s} {epoch:02d}  train={tr_loss:.4f}  val={va_loss:.4f}")
        if va_loss + 1e-5 < best:
            best = va_loss
            best_state = {k: v.detach().cpu().clone() for k, v in wrapper.state_dict().items()}
            bad = 0
        else:
            bad += 1
            if bad >= PATIENCE:
                break
    if best_state is not None:
        wrapper.load_state_dict(best_state)
    y, yhat = collect_preds(wrapper, te_ld, dev, spec, y_mean, y_std)
    metrics = eval_metrics(spec, y, yhat)
    metrics["best_val"] = float(best) if np.isfinite(best) else float("nan")
    metrics["epochs"] = len(hist)
    metrics["n_params"] = int(sum(p.numel() for p in wrapper.parameters()))
    return metrics


def task_available(trials, spec):
    if len(trials) < MIN_TRIALS:
        return False
    if spec["kind"] == "class":
        labs = [int(t["y_class"]) for t in trials]
        if len(set(labs)) < 2:
            return False
    if spec["kind"] == "frame":
        n_ok = 0
        for t in trials:
            if int(np.asarray(t["y_valid"]).sum()) >= 10:
                n_ok += 1
        if n_ok < MIN_TRIALS:
            return False
    if spec["kind"] == "anatomy":
        labs = np.concatenate([t["cosmos_id"][t["cosmos_id"] >= 0] for t in trials])
        if np.unique(labs).size < 2:
            return False
    return True


def run_session(session, models, tasks, epochs, device, seed, max_trials=MAX_TRIALS):
    rows = []
    eid = session["eid"]
    mouse = session["mouse_id"]
    n_units = session["n_units"]
    if max_trials and len(session["trials"]) > max_trials:
        session = dict(session)
        session["trials"] = list(session["trials"][: int(max_trials)])
    log(f"session {eid[:8]}  mouse={mouse}  trials={len(session['trials'])}  units={n_units}  src={session['source']}")
    for task_name in tasks:
        spec = TASKS[task_name]
        trials = bind_task(session, task_name)
        if not task_available(trials, spec):
            log(f"  skip {task_name}: {len(trials)} usable trials")
            continue
        split_fn = interleaved_blocks if spec["suite"] == "TS2" else causal_split
        splits = split_fn(len(trials), seed=seed) if spec["suite"] == "TS2" else causal_split(len(trials))
        if splits is None:
            log(f"  skip {task_name}: split too small")
            continue
        tr_idx, va_idx, te_idx = splits
        stats = make_stats(trials, tr_idx)
        hold = holdout_mask(n_units, seed) if spec["kind"] == "cosmooth" else None
        log(
            f"  {task_name}  n={len(trials)}  split={len(tr_idx)}/{len(va_idx)}/{len(te_idx)}  "
            f"kind={spec['kind']}"
        )
        leaky = spec["kind"] == "anatomy"
        for name in models:
            if leaky and name in ("attn_anat_transformer", "gnn_inner", "gnn_outer"):
                log(f"    skip {name} on TS3 (region features would leak the label)")
                continue
            try:
                if name in ("mean", "ar"):
                    y, yhat = baseline_predict(name, spec, trials, tr_idx, te_idx, hold)
                    metrics = eval_metrics(spec, y, yhat)
                    metrics["epochs"] = 0
                    metrics["n_params"] = 0
                else:
                    metrics = train_neural(
                        name, spec, trials, splits, stats, epochs, device, seed, hold
                    )
                row = {
                    "eid": eid,
                    "mouse_id": mouse,
                    "task": task_name,
                    "suite": spec["suite"],
                    "model": name,
                    "n_train": int(len(tr_idx)),
                    "n_val": int(len(va_idx)),
                    "n_test": int(len(te_idx)),
                    "n_units": n_units,
                    "source": session["source"],
                    "metric": spec["metric"],
                    **metrics,
                }
                rows.append(row)
                log(f"    {name:24s}  {spec['metric']}={metrics.get('score', float('nan')):.3f}")
            except Exception as exc:
                log(f"    {name} FAILED {task_name}: {exc}")
                traceback.print_exc()
                rows.append(
                    {
                        "eid": eid,
                        "mouse_id": mouse,
                        "task": task_name,
                        "suite": spec["suite"],
                        "model": name,
                        "score": float("nan"),
                        "error": str(exc),
                    }
                )
    return rows


def pick_sessions(max_sessions, eids=None):
    paths = list_session_paths()
    scored = []
    for path in paths:
        payload = load_payload(path)
        n = len(payload.get("trials") or [])
        mouse = str(payload.get("mouse_id"))
        scored.append((n, mouse, path))
    scored.sort(reverse=True)
    if eids:
        want = set(eids)
        scored = [s for s in scored if s[2].stem in want or s[2].stem[:8] in want]
    picked = []
    seen_mouse = set()
    for n, mouse, path in scored:
        if n < MIN_TRIALS:
            continue
        if mouse in seen_mouse:
            continue
        picked.append(path)
        seen_mouse.add(mouse)
        if len(picked) >= max_sessions:
            break
    if len(picked) < max_sessions:
        for n, mouse, path in scored:
            if path in picked:
                continue
            picked.append(path)
            if len(picked) >= max_sessions:
                break
    return picked


def summarize(rows):
    df = pd.DataFrame(rows)
    if df.empty:
        return df, {}
    summary = {}
    for task, g in df.groupby("task"):
        pivot = {}
        for model, gg in g.groupby("model"):
            scores = pd.to_numeric(gg.get("score"), errors="coerce")
            pivot[model] = {
                "mean": float(scores.mean()) if scores.notna().any() else float("nan"),
                "n": int(scores.notna().sum()),
            }
        summary[task] = pivot
    return df, summary


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--trial", action="store_true", default=True)
    p.add_argument("--offline", action="store_true")
    p.add_argument("--force-extract", action="store_true")
    p.add_argument("--max-sessions", type=int, default=2)
    p.add_argument("--max-trials", type=int, default=MAX_TRIALS)
    p.add_argument("--epochs", type=int, default=8)
    p.add_argument("--seed", type=int, default=SEED)
    p.add_argument("--device", default="auto")
    p.add_argument("--models", nargs="*", default=list(TRIAL_MODELS))
    p.add_argument("--tasks", nargs="*", default=list(TRIAL_TASKS))
    p.add_argument("--eids", nargs="*")
    args = p.parse_args()
    set_seed(args.seed)
    slurm_utils.set_device(args.device)
    device = slurm_utils.device_of()
    OUT.mkdir(parents=True, exist_ok=True)
    TRIAL_DIR.mkdir(parents=True, exist_ok=True)
    CACHE.mkdir(parents=True, exist_ok=True)
    models = [m for m in args.models if m in MODELS]
    tasks = [t for t in args.tasks if t in TASKS]
    paths = pick_sessions(args.max_sessions, args.eids)
    log(
        f"BrainWideBench trial  sessions={len(paths)}  models={models}  "
        f"tasks={tasks}  epochs={args.epochs}  device={device}  offline={args.offline}"
    )
    one = None
    if not args.offline:
        try:
            one = connect_one()
            log("OpenAlyx connected")
        except Exception as exc:
            log(f"OpenAlyx unavailable ({exc}); offline windows")
            args.offline = True
    rows = []
    meta_sessions = []
    for path in paths:
        try:
            session = extract_session(path, one=one, offline=args.offline, force=args.force_extract)
        except Exception as exc:
            log(f"extract failed {path.name}: {exc}")
            traceback.print_exc()
            continue
        meta_sessions.append(
            {
                "eid": session["eid"],
                "mouse_id": session["mouse_id"],
                "n_trials": len(session["trials"]),
                "n_units": session["n_units"],
                "source": session["source"],
                "cosmos": sorted({c for c in session["cosmos"] if c}),
            }
        )
        rows.extend(
            run_session(
                session, models, tasks, args.epochs, device, args.seed, max_trials=args.max_trials
            )
        )
    df, summary = summarize(rows)
    TRIAL_DIR.mkdir(parents=True, exist_ok=True)
    scores_path = TRIAL_DIR / "scores.csv"
    if not df.empty:
        df.to_csv(scores_path, index=False)
    report = {
        "built_at": utc_now(),
        "paper": "https://arxiv.org/abs/2609.22064",
        "project": "https://www.iblcore.org/resources/benchmarking/ibl-brainwidebench/",
        "leaderboard": "https://bwb.iblcore.org/html/dashboard/dashboard.html",
        "epochs": args.epochs,
        "max_trials": args.max_trials,
        "offline": bool(args.offline),
        "device": str(device),
        "models": models,
        "tasks": tasks,
        "sessions": meta_sessions,
        "summary": summary,
        "n_rows": int(len(rows)),
        "note": (
            "Local single-session trial of ExperimentsClean formulations on BWB "
            "windows. Not a leaderboard submission. R² / D² are clipped at 0 "
            "for the reported score, matching Appendix E."
        ),
    }
    (TRIAL_DIR / "report.json").write_text(json.dumps(report, indent=2, default=float))
    log(f"wrote {scores_path}  rows={len(rows)}")
    print_report(summary, meta_sessions)
    return report


def print_report(summary, sessions):
    print("\n=== BrainWideBench trial ===")
    print("sessions:", ", ".join(f"{s['mouse_id']} {s['eid'][:8]} n={s['n_trials']}" for s in sessions))
    if not summary:
        print("no scores")
        return
    models = sorted({m for task in summary.values() for m in task})
    width = max(12, max(len(m) for m in models))
    header = f"{'task':28s}" + "".join(f"{m:>{width}s}" for m in models)
    print(header)
    for task, pivot in summary.items():
        line = f"{task:28s}"
        for m in models:
            val = pivot.get(m, {}).get("mean", float("nan"))
            line += f"{val:{width}.3f}" if np.isfinite(val) else f"{'—':>{width}s}"
        print(line)


if __name__ == "__main__":
    warnings.filterwarnings("ignore", category=UserWarning)
    main()
