"""Spike–LFP fusion models with ExperimentsClean LTO / LSO / LMO folds.

--grid default (pilot, what --single-fold submits)
  3 associations × 5 fusions × 5 input modes, atlas / instant / transformer,
  plus a few one-at-a-time extras (other pos, encoders, temporals, soft unnorm).
  No hyperparameter sweep. ~130 models × 3 evals.

--grid core     association × fusion only
--grid arch     full architecture cartesian × inputs (no HP)
--grid full     arch cartesian × inputs × HP three-way OATs

Input modes
  spikes | lfp | lfp_nogamma | spike_lfp | spike_lfp_nogamma

Local:
  python lfpfuse.py --self-test
  python lfpfuse.py --single-fold --list-jobs

Cluster:
  bash slurm/submit_fuse.sh --single-fold

After the array:
  python lfpfuse.py --single-fold --aggregate
"""

from __future__ import annotations

import argparse
import json
import math
import os
import pickle
import re
import signal
import sys
import traceback
from collections import defaultdict
from pathlib import Path

os.environ.setdefault("MPLBACKEND", "Agg")

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from torch.utils.data import DataLoader, Dataset

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import Modelv1 as mv
import slurm_utils
import ExperimentsClean as ec

mv.MODEL_ROOT = ROOT / "LFPFuse"

MODEL_ROOT = ROOT / "LFPFuse"
TARGET = "paw_speed"
DUMMY = False
MAX_EPOCHS_CAP = None

EVALS = ec.EVALS
ASSOCIATIONS = ("hard", "soft", "graph")
FUSIONS = ("concat", "gate", "residual", "xattn", "private")
POS_MODES = ("atlas", "none", "probe", "depth", "local")
ENCODERS = ("instant", "history", "bandconv")
TEMPORALS = ("transformer", "gru", "kalman", "recurrent", "cde")
SOFT_NORMS = ("norm", "unnorm")
POOLS = ("latent", "mean", "attn")
INPUT_MODES = ("spikes", "lfp", "lfp_nogamma", "spike_lfp", "spike_lfp_nogamma")
GRIDS = ("default", "pilot", "core", "arch", "full")

N_LFP_BANDS = ec.N_LFP_BANDS
HIGH_GAMMA_IDX = 5  # last of δ θ α β low-γ high-γ
K_NE = 8
K_EE = 6
K_NE_MAX = 16
K_EE_MAX = 12
K_QUERIES = 4
SIGMA_UM = 50.0
SPIKE_LAGS = 4
GRAPH_LAYERS = 2
TIME_CHUNK = 8
FUSE_BATCH = 16
GRAPH_BATCH = 8
LAMBDA_S = 0.1
LAMBDA_L = 0.1
LAMBDA_C = 0.1
DT_S = 0.02
LAYER_RE = re.compile(r"(\d)")

POS_DIM = {"none": 0, "atlas": 3, "probe": 2, "depth": 2, "local": 3}

DEFAULT_HP = {
    "d_model": 64,
    "n_queries": 4,
    "n_layers": 1,
    "dropout": 0.1,
    "sigma": 50.0,
    "k_ne": 8,
    "k_ee": 6,
    "graph_layers": 2,
    "spike_lags": 4,
    "kalman_dz": 16,
    "lambda_s": 0.1,
    "lambda_l": 0.1,
    "lambda_c": 0.1,
    "band_scale": "mid",
    "pool": "latent",
    "rnn_cell": "gru",
    "cde_depth": 2,
    "unit_drop": 0.15,
    "lr": 1e-3,
}

BAND_SPECS = {
    "short": ((7, 2), (5, 2), (5, 2), (5, 1), (3, 1), (3, 1)),
    "mid": ((13, 4), (9, 4), (9, 4), (7, 2), (5, 1), (5, 1)),
    "long": ((17, 8), (13, 6), (13, 6), (11, 4), (7, 2), (7, 2)),
}
HISTORY_SPEC = {"short": (7, 1), "mid": (15, 2), "long": (25, 4)}
SPIKE_CONV_K = {"short": 3, "mid": 5, "long": 9}

HP_TAG = {
    "d_model": "d",
    "n_queries": "q",
    "n_layers": "L",
    "dropout": "drop",
    "sigma": "sigma",
    "k_ne": "kne",
    "k_ee": "kee",
    "graph_layers": "g",
    "spike_lags": "lags",
    "kalman_dz": "dz",
    "lambda_s": "ls",
    "lambda_l": "ll",
    "lambda_c": "lc",
    "band_scale": "band",
    "pool": "pool",
    "rnn_cell": "cell",
    "cde_depth": "cde",
    "unit_drop": "udrop",
    "lr": "lr",
}


def log(msg):
    mv.log(msg)


def configure_out():
    global OUT, CACHE
    OUT = MODEL_ROOT / ("dummy" if DUMMY else TARGET)
    CACHE = OUT / "cache"
    OUT.mkdir(parents=True, exist_ok=True)
    CACHE.mkdir(parents=True, exist_ok=True)
    return OUT, CACHE


def device_probe(cfg):
    temporal = str(cfg.get("temporal", "transformer"))
    cell = str(cfg.get("rnn_cell", "gru"))
    if temporal == "gru" or (temporal == "recurrent" and cell == "gru"):
        return "gru"
    if temporal == "recurrent" and cell == "lstm":
        return "lstm"
    return "transformer"


def nhead_of(d):
    for h in (8, 4, 2, 1):
        if int(d) % h == 0:
            return h
    return 1


def hp_of(cfg):
    out = dict(DEFAULT_HP)
    for k in DEFAULT_HP:
        if cfg.get(k) is not None:
            out[k] = cfg[k]
    return out


def _fmt_hp(v):
    if isinstance(v, float):
        if abs(v - int(v)) < 1e-12:
            return str(int(v))
        s = f"{v:.4g}".replace(".", "p")
        return s
    return str(v)


def model_name_of(cfg, assoc=None, fuse=None, pos=None, temporal=None, soft_norm=None, encoder=None):
    if not isinstance(cfg, dict):
        cfg = {
            "assoc": assoc or cfg,
            "fuse": fuse,
            "pos": pos,
            "encoder": encoder or "instant",
            "temporal": temporal,
            "soft_norm": soft_norm or "norm",
        }
    assoc = cfg.get("assoc")
    name = (
        f"{assoc}_{cfg.get('fuse')}_{cfg.get('pos')}_{cfg.get('encoder', 'instant')}_"
        f"{cfg.get('temporal')}_{cfg.get('input', 'spike_lfp')}"
    )
    if assoc == "soft" and cfg.get("soft_norm", "norm") == "unnorm":
        name += "_unnorm"
    hp = hp_of(cfg)
    for k, default in DEFAULT_HP.items():
        v = hp[k]
        if v != default:
            name += f"_{HP_TAG[k]}{_fmt_hp(v)}"
    return name


def uses_spikes(cfg):
    return str(cfg.get("input", "spike_lfp")) in ("spikes", "spike_lfp", "spike_lfp_nogamma")


def uses_lfp(cfg):
    return str(cfg.get("input", "spike_lfp")) != "spikes"


def drop_high_gamma(cfg):
    return str(cfg.get("input", "spike_lfp")).endswith("nogamma")


def job_tag(cfg):
    return f"{cfg['eval']}_{cfg['fold']}_{cfg['model']}"


def parse_model_name(name):
    raw = str(name)
    soft_norm = "unnorm" if "_unnorm" in raw else "norm"
    parts = raw.split("_")
    if len(parts) < 5:
        raise ValueError(f"cannot parse model name {name!r}")
    return parts[0], parts[1], parts[2], parts[3], parts[4], soft_norm


# ---------------------------------------------------------------------------
# Geometry
# ---------------------------------------------------------------------------


def layer_ids(areas):
    out = np.zeros(len(areas), dtype=np.float32)
    for i, a in enumerate(areas):
        m = LAYER_RE.search(str(a))
        if m:
            out[i] = float(m.group(1))
    return out


def _finite_pair_dist(u_ax, u_lat, e_ax, e_lat, u_xyz, e_xyz):
    ax = np.asarray(u_ax, dtype=np.float32)[:, None] - np.asarray(e_ax, dtype=np.float32)[None, :]
    lat = np.asarray(u_lat, dtype=np.float32)[:, None] - np.asarray(e_lat, dtype=np.float32)[None, :]
    d = np.hypot(ax, lat)
    bad = ~np.isfinite(d)
    if np.any(bad):
        diff = np.asarray(u_xyz, dtype=np.float32)[:, None, :] - np.asarray(e_xyz, dtype=np.float32)[None, :, :]
        d_ccf = np.sqrt(np.maximum((diff ** 2).sum(-1), 0.0)) * 1e6
        d = np.where(bad, d_ccf, d)
    return np.nan_to_num(d, nan=1e6, posinf=1e6).astype(np.float32)


def knn_from_dist(dist, k, max_dist=None):
    dist = np.asarray(dist, dtype=np.float32)
    n, e = dist.shape
    k = max(1, min(int(k), e))
    part = np.argpartition(dist, kth=k - 1, axis=1)[:, :k]
    rows = np.arange(n)[:, None]
    d_k = dist[rows, part]
    order = np.argsort(d_k, axis=1)
    idx = part[rows, order].astype(np.int64)
    d_sorted = d_k[rows, order].astype(np.float32)
    valid = np.ones((n, k), dtype=bool)
    if max_dist is not None:
        valid = d_sorted <= float(max_dist)
        valid[:, 0] = True
    return idx, d_sorted, valid


def elec_probe_knn(axial, probes, k):
    axial = np.asarray(axial, dtype=np.float32).reshape(-1)
    probes = np.asarray(probes).astype(str).reshape(-1)
    e = int(axial.size)
    if e < 1:
        return np.zeros((0, 1), np.int64), np.zeros((0, 1), np.float32), np.zeros((0, 1), bool)
    dist = np.abs(axial[:, None] - axial[None, :])
    same = probes[:, None] == probes[None, :]
    dist = np.where(same, dist, dist + 1e6)
    np.fill_diagonal(dist, 1e9)
    k = max(1, min(int(k), max(e - 1, 1)))
    return knn_from_dist(dist, k)


def gaussian_w(dist, sigma=SIGMA_UM):
    sig = max(float(sigma), 1.0)
    return np.exp(-np.square(dist) / (2.0 * sig * sig)).astype(np.float32)


def stamp_geometry(t, k_ne=K_NE_MAX, k_ee=K_EE_MAX, sigma=SIGMA_UM):
    """Add knn / probe / layer fields used by every fusion model.

    Stores the widest neighborhood (K_NE_MAX / K_EE_MAX) so later jobs can
    slice k_ne / k_ee and recompute Gaussian weights at any σ.
    """
    u_xyz = np.asarray(t["unit_xyz"], dtype=np.float32)
    e_xyz = np.asarray(t["lfp_xyz"], dtype=np.float32)
    n_u, n_e = int(u_xyz.shape[0]), int(e_xyz.shape[0])
    u_ax = np.asarray(t.get("unit_axial", np.full(n_u, np.nan)), dtype=np.float32).reshape(-1)
    u_lat = np.asarray(t.get("unit_lateral", np.full(n_u, np.nan)), dtype=np.float32).reshape(-1)
    e_ax = np.asarray(t.get("lfp_axial", np.full(n_e, np.nan)), dtype=np.float32).reshape(-1)
    e_lat = np.asarray(t.get("lfp_lateral", np.full(n_e, np.nan)), dtype=np.float32).reshape(-1)
    probes = np.asarray(t.get("lfp_probe", np.array([""] * n_e))).astype(str).reshape(-1)
    areas_e = np.asarray(t.get("lfp_areas", np.array(["unk"] * n_e))).astype(str)
    to_ch = np.asarray(t.get("unit_to_ch", np.zeros(n_u, np.int32)), dtype=np.int64).reshape(-1)
    if n_u == 0 or n_e == 0:
        return t
    dist = _finite_pair_dist(u_ax, u_lat, e_ax, e_lat, u_xyz, e_xyz)
    knn, d_k, knn_ok = knn_from_dist(dist, min(k_ne, n_e), max_dist=None)
    w = gaussian_w(d_k, sigma)
    w = np.where(knn_ok, w, 0.0).astype(np.float32)
    w[:, 0] = np.maximum(w[:, 0], 1e-6)
    w_unnorm = w.copy()
    w_norm = w / np.maximum(w.sum(axis=1, keepdims=True), 1e-6)
    hard_idx = to_ch.reshape(-1, 1).astype(np.int64)
    hard_ok = (hard_idx >= 0) & (hard_idx < n_e)
    hard_idx = np.where(hard_ok, hard_idx, 0)
    ee_idx, ee_d, ee_ok = elec_probe_knn(e_ax, probes, k_ee)
    ax_min = {}
    ax_max = {}
    for p, ax in zip(probes, e_ax):
        if not np.isfinite(ax):
            continue
        ax_min[p] = ax if p not in ax_min else min(ax_min[p], ax)
        ax_max[p] = ax if p not in ax_max else max(ax_max[p], ax)
    depth = np.zeros(n_e, dtype=np.float32)
    for i, p in enumerate(probes):
        lo, hi = ax_min.get(p, 0.0), ax_max.get(p, 1.0)
        depth[i] = (e_ax[i] - lo) / max(hi - lo, 1.0) if np.isfinite(e_ax[i]) else 0.0
    centroid = np.nanmean(e_xyz, axis=0).astype(np.float32)
    local = np.stack(
        [
            e_ax - (np.nanmean(e_ax) if np.isfinite(e_ax).any() else 0.0),
            e_lat - (np.nanmean(e_lat) if np.isfinite(e_lat).any() else 0.0),
            np.linalg.norm(e_xyz - centroid[None, :], axis=1) * 1e6,
        ],
        axis=1,
    ).astype(np.float32)
    t["unit_knn"] = knn
    t["unit_knn_dist"] = d_k.astype(np.float32)
    t["unit_knn_valid"] = knn_ok
    t["unit_w_norm"] = w_norm
    t["unit_w_unnorm"] = w_unnorm
    t["hard_idx"] = hard_idx
    t["hard_valid"] = hard_ok
    t["elec_knn"] = ee_idx
    t["elec_knn_valid"] = ee_ok
    t["elec_knn_dist"] = ee_d.astype(np.float32)
    t["lfp_depth"] = depth
    t["lfp_layer"] = layer_ids(areas_e)
    t["lfp_local"] = np.nan_to_num(local, nan=0.0)
    t["lfp_probe"] = probes
    if t.get("unit_quality") is None:
        t["unit_quality"] = np.ones(n_u, dtype=np.float32)
    return t


def _unit_quality(units, m):
    cols = units.columns
    if "amp" in cols:
        q = units["amp"].to_numpy(dtype=np.float32)[m]
    elif "amplitude" in cols:
        q = units["amplitude"].to_numpy(dtype=np.float32)[m]
    elif "ks2_contamination_pct" in cols:
        c = units["ks2_contamination_pct"].to_numpy(dtype=np.float32)[m]
        q = 1.0 / (1.0 + np.clip(c, 0.0, None))
    else:
        q = np.ones(int(m.size), dtype=np.float32)
    return np.nan_to_num(q, nan=0.0).astype(np.float32)


def _fill_missing_xyz(xyz, count, axial):
    xyz = xyz.copy()
    missing = count < 1
    if not missing.any() or missing.all():
        return xyz
    order = np.argsort(np.nan_to_num(axial, nan=np.nanmean(axial) if np.isfinite(axial).any() else 0.0))
    known = ~missing
    for j in np.flatnonzero(missing):
        d = np.abs(axial[j] - axial[known]) if np.isfinite(axial[j]) else np.arange(int(known.sum()), dtype=float)
        xyz[j] = xyz[known][int(np.argmin(d))]
    return xyz


def attach_fuse(trials):
    """Join LFP bandpower plus probe geometry / knn onto spike trials."""
    by_eid = defaultdict(list)
    for i, t in enumerate(trials):
        t.setdefault("unit_spikes", t["spikes"])
        t.setdefault("unit_xyz", t["xyz"])
        t.setdefault("unit_areas", t["areas"])
        t["lfp"] = t.get("lfp")
        by_eid[str(t["eid"])].append(i)
    already = all(
        t.get("unit_knn") is not None and t.get("lfp_axial") is not None
        for t in trials
        if ec.trial_has_lfp(t)
    )
    if already and any(ec.trial_has_lfp(t) for t in trials):
        log(f"LFP fuse attach: geometry already present")
        return trials
    n_ok = 0
    n_miss = 0
    for eid, idxs in by_eid.items():
        spk_path = mv.SESS / f"{eid}.pkl"
        lfp_path = ec.LFP_DIR / f"{eid}.pkl"
        if not spk_path.exists() or not lfp_path.exists():
            n_miss += 1
            continue
        with spk_path.open("rb") as fh:
            src = pickle.load(fh)
        with lfp_path.open("rb") as fh:
            lfp_src = pickle.load(fh)
        units = src["units"]
        m = np.flatnonzero(mv.motor_mask(units).to_numpy())
        if m.size < 1:
            n_miss += 1
            continue
        unit_ch = np.asarray(units["channel"].to_numpy()[m], dtype=int)
        if "probe_name" in units.columns:
            unit_pr = units["probe_name"].astype(str).to_numpy()[m]
        else:
            unit_pr = np.array([""] * int(m.size))
        unit_xyz = np.stack(
            [
                units["allen_x"].to_numpy(dtype=np.float32)[m],
                units["allen_y"].to_numpy(dtype=np.float32)[m],
                units["allen_z"].to_numpy(dtype=np.float32)[m],
            ],
            axis=1,
        )
        ax_u = np.asarray(units["axial_um"].to_numpy()[m], dtype=float) if "axial_um" in units.columns else np.full(m.size, np.nan)
        lat_u = np.asarray(units["lateral_um"].to_numpy()[m], dtype=float) if "lateral_um" in units.columns else np.full(m.size, np.nan)
        quality = _unit_quality(units, m)
        lfp_trials = {int(r["trial_index"]): r for r in lfp_src["trials"] if r.get("trial_index") is not None}
        rec0 = lfp_src["trials"][0]
        acr = np.asarray(rec0["channel_acronym"]).astype(str)
        motor = np.asarray(rec0.get("channel_motor", np.zeros(acr.size, dtype=bool)), dtype=bool)
        axial = np.asarray(rec0["channel_axial_um"], dtype=float)
        lateral = np.asarray(rec0["channel_lateral_um"], dtype=float)
        probes = np.asarray(rec0.get("channel_probe", [""] * acr.size)).astype(str)
        offsets = ec._probe_offsets(probes)
        n_full = int(acr.size)
        keep = motor.copy()
        full_of_unit = np.full(unit_ch.size, -1, dtype=int)
        for ui, (ch, pr) in enumerate(zip(unit_ch, unit_pr)):
            off = offsets.get(str(pr), offsets.get("", 0) if "" in offsets else 0)
            fi = int(off) + int(ch)
            if 0 <= fi < n_full:
                keep[fi] = True
                full_of_unit[ui] = fi
        for ui in range(int(m.size)):
            if full_of_unit[ui] >= 0:
                continue
            d = np.abs(axial - ax_u[ui]) + np.abs(lateral - lat_u[ui])
            fi = int(np.argmin(d))
            keep[fi] = True
            full_of_unit[ui] = fi
        motor_idx = np.flatnonzero(keep)
        if motor_idx.size < 1:
            n_miss += 1
            continue
        full_to_tok = {int(fi): j for j, fi in enumerate(motor_idx.tolist())}
        unit_to_ch = np.array([full_to_tok.get(int(fi), -1) for fi in full_of_unit], dtype=np.int32)
        tok_xyz = np.zeros((motor_idx.size, 3), dtype=np.float32)
        tok_n = np.zeros(motor_idx.size, dtype=np.float32)
        for ui, j in enumerate(unit_to_ch):
            if j < 0:
                continue
            tok_xyz[j] += unit_xyz[ui]
            tok_n[j] += 1.0
        tok_xyz = _fill_missing_xyz(tok_xyz / np.maximum(tok_n, 1.0)[:, None], tok_n, axial[motor_idx])
        tok_areas = acr[motor_idx]
        geo = dict(
            unit_xyz=unit_xyz,
            unit_areas=units["brain_area"].astype(str).to_numpy()[m] if "brain_area" in units.columns else np.array(["unk"] * int(m.size)),
            unit_axial=ax_u.astype(np.float32),
            unit_lateral=lat_u.astype(np.float32),
            unit_quality=quality,
            unit_to_ch=unit_to_ch,
            lfp_xyz=tok_xyz,
            lfp_areas=tok_areas,
            lfp_axial=axial[motor_idx].astype(np.float32),
            lfp_lateral=lateral[motor_idx].astype(np.float32),
            lfp_probe=probes[motor_idx],
        )
        for i in idxs:
            t = trials[i]
            rec = lfp_trials.get(int(t["trial_index"]) if t.get("trial_index") is not None else -1)
            if rec is None:
                continue
            pow20 = rec.get("power_car_s20")
            if pow20 is None:
                continue
            lp = ec._log_lfp(pow20)
            if lp.shape[1] < n_full:
                continue
            n = min(int(t["unit_spikes"].shape[0]), int(lp.shape[0]))
            if n < mv.MIN_BINS:
                continue
            t["unit_spikes"] = np.asarray(t["unit_spikes"], dtype=np.float32)[:n]
            t["spikes"] = t["unit_spikes"]
            t["lfp"] = lp[:n, motor_idx, :N_LFP_BANDS]
            t.update(geo)
            stamp_geometry(t)
            n_ok += 1
    log(f"LFP fuse attach: {n_ok} trials with geometry, {n_miss} sessions missing LFP")
    return trials


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------


def pos_matrix(t, mode, stats):
    mode = mode or "atlas"
    e = int(np.asarray(t["lfp_xyz"]).shape[0])
    if mode == "none":
        return np.zeros((e, 0), dtype=np.float32)
    if mode == "atlas":
        xyz = (np.asarray(t["lfp_xyz"], dtype=np.float32) - stats["xyz_mean"]) / stats["xyz_std"]
        return np.nan_to_num(xyz, nan=0.0).astype(np.float32)
    if mode == "probe":
        ax = (np.asarray(t["lfp_axial"], dtype=np.float32) - stats["axial_mean"]) / stats["axial_std"]
        lat = (np.asarray(t["lfp_lateral"], dtype=np.float32) - stats["lateral_mean"]) / stats["lateral_std"]
        return np.nan_to_num(np.stack([ax, lat], axis=1), nan=0.0).astype(np.float32)
    if mode == "depth":
        depth = np.asarray(t.get("lfp_depth", np.zeros(e)), dtype=np.float32).reshape(-1)
        layer = np.asarray(t.get("lfp_layer", np.zeros(e)), dtype=np.float32).reshape(-1) / 6.0
        return np.nan_to_num(np.stack([depth, layer], axis=1), nan=0.0).astype(np.float32)
    loc = np.asarray(t.get("lfp_local", np.zeros((e, 3))), dtype=np.float32)
    loc = loc / np.maximum(stats.get("local_std", np.ones(3, np.float32)), 1e-6)
    return np.nan_to_num(loc, nan=0.0).astype(np.float32)


def fit_fuse_stats(trials, train_idx, fallback_idx):
    train_idx = np.asarray(train_idx, dtype=int)
    seen = set()
    xyz_rows, ax_rows, lat_rows, loc_rows, areas = [], [], [], [], []
    for i in train_idx:
        t = trials[int(i)]
        if t["eid"] in seen or not ec.trial_has_lfp(t):
            continue
        seen.add(t["eid"])
        xyz_rows.append(np.asarray(t["lfp_xyz"], dtype=np.float32))
        ax_rows.append(np.asarray(t["lfp_axial"], dtype=np.float32).reshape(-1))
        lat_rows.append(np.asarray(t["lfp_lateral"], dtype=np.float32).reshape(-1))
        loc_rows.append(np.asarray(t.get("lfp_local", np.zeros((xyz_rows[-1].shape[0], 3))), dtype=np.float32))
        areas.extend(np.asarray(t["lfp_areas"]).astype(str).tolist())
        areas.extend(np.asarray(t["unit_areas"]).astype(str).tolist())
    if not xyz_rows:
        xyz = np.zeros((1, 3), np.float32)
        ax = np.zeros(1, np.float32)
        lat = np.zeros(1, np.float32)
        loc = np.zeros((1, 3), np.float32)
    else:
        xyz = np.concatenate(xyz_rows, axis=0)
        ax = np.concatenate(ax_rows, axis=0)
        lat = np.concatenate(lat_rows, axis=0)
        loc = np.concatenate(loc_rows, axis=0)
    def _ms(v, n):
        v = np.asarray(v, dtype=np.float32).reshape(-1, n)
        mean = np.nanmean(v, axis=0).astype(np.float32)
        std = np.nanstd(v, axis=0).astype(np.float32)
        std = np.where(std < 1e-8, 1.0, std).astype(np.float32)
        return mean, std
    xyz_mean, xyz_std = _ms(xyz, 3)
    axial_mean, axial_std = _ms(ax, 1)
    lateral_mean, lateral_std = _ms(lat, 1)
    local_mean, local_std = _ms(loc, 3)
    vocab = {"<unk>": 0}
    for a in sorted(set(areas)):
        if a not in vocab:
            vocab[a] = len(vocab)

    def rates_from(idx):
        buckets = {}
        for i in np.asarray(idx, dtype=int):
            t = trials[int(i)]
            if not ec.trial_has_lfp(t):
                continue
            buckets.setdefault(t["eid"], []).append(np.asarray(t["unit_spikes"], dtype=np.float32).mean(axis=0))
        return {eid: np.mean(vs, axis=0).astype(np.float32) for eid, vs in buckets.items()}

    rates = rates_from(fallback_idx)
    rates.update(rates_from(train_idx))
    return {
        "vocab": vocab,
        "n_areas": len(vocab),
        "xyz_mean": xyz_mean,
        "xyz_std": xyz_std,
        "axial_mean": float(axial_mean[0]),
        "axial_std": float(axial_std[0]),
        "lateral_mean": float(lateral_mean[0]),
        "lateral_std": float(lateral_std[0]),
        "local_std": local_std,
        "rates": rates,
    }


def _slice_soft(t, cfg):
    hp = hp_of(cfg)
    knn = np.asarray(t["unit_knn"], dtype=np.int64)
    dist = np.asarray(t.get("unit_knn_dist", np.ones(knn.shape, np.float32)), dtype=np.float32)
    k = max(1, min(int(hp["k_ne"]), knn.shape[1], dist.shape[1]))
    knn, dist = knn[:, :k], dist[:, :k]
    valid = dist <= (3.0 * float(hp["sigma"]))
    valid[:, 0] = True
    w = gaussian_w(dist, hp["sigma"])
    w = np.where(valid, w, 0.0).astype(np.float32)
    w[:, 0] = np.maximum(w[:, 0], 1e-6)
    if cfg.get("soft_norm", "norm") != "unnorm":
        w = w / np.maximum(w.sum(axis=1, keepdims=True), 1e-6)
    return knn, valid, w


def _slice_elec(t, cfg):
    hp = hp_of(cfg)
    knn = np.asarray(t["elec_knn"], dtype=np.int64)
    dist = np.asarray(t["elec_knn_dist"], dtype=np.float32)
    valid = np.asarray(t["elec_knn_valid"], dtype=bool)
    k = max(1, min(int(hp["k_ee"]), knn.shape[1], dist.shape[1], valid.shape[1]))
    return knn[:, :k], valid[:, :k], dist[:, :k]


class FuseDS(Dataset):
    def __init__(self, trials, idx, y_mean, y_std, stats, cfg):
        self.trials = trials
        self.idx = np.asarray(idx, dtype=int)
        self.y_mean = np.asarray(y_mean, dtype=np.float32).reshape(-1)
        self.y_std = np.asarray(y_std, dtype=np.float32).reshape(-1)
        self.stats = stats
        self.cfg = cfg

    def __len__(self):
        return len(self.idx)

    def __getitem__(self, j):
        t = self.trials[int(self.idx[j])]
        y = np.asarray(t["y"], dtype=np.float32).reshape(t["y"].shape[0], -1)
        valid = t.get("y_valid")
        if valid is None:
            valid = np.isfinite(y).all(axis=1)
        time_valid = np.asarray(valid, dtype=bool).reshape(-1)
        y = np.where(time_valid[:, None], y, 0.0)
        y = (y - self.y_mean) / self.y_std
        spikes = np.asarray(t["unit_spikes"], dtype=np.float32)
        lfp = np.asarray(t["lfp_z"] if t.get("lfp_z") is not None else t["lfp"], dtype=np.float32)
        n = min(spikes.shape[0], lfp.shape[0], y.shape[0], time_valid.size)
        spikes, lfp, y, time_valid = spikes[:n], lfp[:n], y[:n], time_valid[:n]
        rate = np.asarray(self.stats["rates"][t["eid"]], dtype=np.float32)
        if rate.size != spikes.shape[1]:
            rate = spikes.mean(axis=0).astype(np.float32)
        knn, knn_ok, w = _slice_soft(t, self.cfg)
        ee_idx, ee_ok, ee_d = _slice_elec(t, self.cfg)
        return {
            "spikes": spikes,
            "lfp": lfp,
            "y": y.astype(np.float32),
            "time_valid": time_valid,
            "unit_xyz": np.nan_to_num(
                (np.asarray(t["unit_xyz"], dtype=np.float32) - self.stats["xyz_mean"]) / self.stats["xyz_std"],
                nan=0.0,
            ).astype(np.float32),
            "elec_xyz": np.nan_to_num(
                (np.asarray(t["lfp_xyz"], dtype=np.float32) - self.stats["xyz_mean"]) / self.stats["xyz_std"],
                nan=0.0,
            ).astype(np.float32),
            "unit_area": mv.encode_areas(t["unit_areas"], self.stats["vocab"]),
            "elec_area": mv.encode_areas(t["lfp_areas"], self.stats["vocab"]),
            "rate": rate,
            "quality": np.asarray(t.get("unit_quality", np.ones(spikes.shape[1])), dtype=np.float32),
            "pos": pos_matrix(t, self.cfg["pos"], self.stats),
            "unit_knn": knn,
            "unit_knn_valid": knn_ok,
            "unit_w": w,
            "hard_idx": np.asarray(t["hard_idx"], dtype=np.int64),
            "hard_valid": np.asarray(t["hard_valid"], dtype=bool),
            "elec_knn": ee_idx,
            "elec_knn_valid": ee_ok,
            "elec_knn_dist": ee_d,
        }


def _pad_idx(arr, n, k, fill=0):
    out = np.full((n, k), fill, dtype=arr.dtype)
    n0, k0 = arr.shape
    out[:n0, :k0] = arr
    return out


def collate_fuse(batch):
    B = len(batch)
    T = max(b["spikes"].shape[0] for b in batch)
    N = max(b["spikes"].shape[1] for b in batch)
    E = max(b["lfp"].shape[1] for b in batch)
    K_u = max(b["unit_knn"].shape[1] for b in batch)
    K_e = max(b["elec_knn"].shape[1] for b in batch)
    P = int(batch[0]["pos"].shape[-1])
    y_dim = int(batch[0]["y"].shape[-1])
    spikes = torch.zeros(B, N, T)
    lfp = torch.zeros(B, E, T, N_LFP_BANDS)
    y = torch.zeros(B, T, y_dim)
    mask_t = torch.zeros(B, T, dtype=torch.bool)
    mask_n = torch.zeros(B, N, dtype=torch.bool)
    mask_e = torch.zeros(B, E, dtype=torch.bool)
    unit_xyz = torch.zeros(B, N, 3)
    elec_xyz = torch.zeros(B, E, 3)
    unit_area = torch.zeros(B, N, dtype=torch.long)
    elec_area = torch.zeros(B, E, dtype=torch.long)
    rate = torch.zeros(B, N)
    quality = torch.zeros(B, N)
    pos = torch.zeros(B, E, max(P, 1))
    unit_knn = torch.zeros(B, N, K_u, dtype=torch.long)
    unit_knn_valid = torch.zeros(B, N, K_u, dtype=torch.bool)
    unit_w = torch.zeros(B, N, K_u)
    hard_idx = torch.zeros(B, N, 1, dtype=torch.long)
    hard_valid = torch.zeros(B, N, 1, dtype=torch.bool)
    elec_knn = torch.zeros(B, E, K_e, dtype=torch.long)
    elec_knn_valid = torch.zeros(B, E, K_e, dtype=torch.bool)
    elec_knn_dist = torch.zeros(B, E, K_e)
    lengths = torch.zeros(B, dtype=torch.long)
    for i, b in enumerate(batch):
        n, e = b["spikes"].shape[1], b["lfp"].shape[1]
        tt = b["spikes"].shape[0]
        spikes[i, :n, :tt] = torch.from_numpy(b["spikes"].T)
        lfp[i, :e, :tt] = torch.from_numpy(np.transpose(b["lfp"], (1, 0, 2)))
        y[i, :tt] = torch.from_numpy(b["y"])
        mask_t[i, :tt] = torch.from_numpy(b["time_valid"])
        mask_n[i, :n] = True
        mask_e[i, :e] = True
        unit_xyz[i, :n] = torch.from_numpy(b["unit_xyz"])
        elec_xyz[i, :e] = torch.from_numpy(b["elec_xyz"])
        unit_area[i, :n] = torch.from_numpy(b["unit_area"])
        elec_area[i, :e] = torch.from_numpy(b["elec_area"])
        rate[i, :n] = torch.from_numpy(b["rate"][:n])
        quality[i, :n] = torch.from_numpy(b["quality"][:n])
        if P:
            pos[i, :e, :P] = torch.from_numpy(b["pos"])
        unit_knn[i, :n, : b["unit_knn"].shape[1]] = torch.from_numpy(b["unit_knn"])
        unit_knn_valid[i, :n, : b["unit_knn_valid"].shape[1]] = torch.from_numpy(b["unit_knn_valid"])
        unit_w[i, :n, : b["unit_w"].shape[1]] = torch.from_numpy(b["unit_w"])
        hard_idx[i, :n] = torch.from_numpy(b["hard_idx"])
        hard_valid[i, :n] = torch.from_numpy(b["hard_valid"])
        elec_knn[i, :e, : b["elec_knn"].shape[1]] = torch.from_numpy(b["elec_knn"])
        elec_knn_valid[i, :e, : b["elec_knn_valid"].shape[1]] = torch.from_numpy(b["elec_knn_valid"])
        elec_knn_dist[i, :e, : b["elec_knn_dist"].shape[1]] = torch.from_numpy(b["elec_knn_dist"])
        lengths[i] = tt
    return {
        "spikes": spikes,
        "lfp": lfp,
        "y": y,
        "mask_t": mask_t,
        "mask_loss": mask_t,
        "mask_n": mask_n,
        "mask_e": mask_e,
        "unit_xyz": unit_xyz,
        "elec_xyz": elec_xyz,
        "unit_area": unit_area,
        "elec_area": elec_area,
        "rate": rate,
        "quality": quality,
        "pos": pos if P else pos[:, :, :0],
        "unit_knn": unit_knn,
        "unit_knn_valid": unit_knn_valid,
        "unit_w": unit_w,
        "hard_idx": hard_idx,
        "hard_valid": hard_valid,
        "elec_knn": elec_knn,
        "elec_knn_valid": elec_knn_valid,
        "elec_knn_dist": elec_knn_dist,
        "lengths": lengths,
    }


# ---------------------------------------------------------------------------
# Modules
# ---------------------------------------------------------------------------


def causal_lags(x, k):
    if k <= 0:
        return x.new_zeros(*x.shape, 0)
    parts = []
    for lag in range(1, k + 1):
        parts.append(F.pad(x, (lag, 0))[..., :-lag])
    return torch.stack(parts, dim=-1)


class MLP(nn.Module):
    def __init__(self, ins, outs, hidden=None, extra=False):
        super().__init__()
        hidden = ins if hidden is None else hidden
        layers = [nn.Linear(ins, hidden), nn.GELU(), nn.Linear(hidden, outs)]
        if extra:
            layers.extend([nn.GELU(), nn.Linear(outs, outs)])
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


class CausalConv1d(nn.Module):
    def __init__(self, c_in, c_out, kernel, dilation=1):
        super().__init__()
        self.pad = (int(kernel) - 1) * int(dilation)
        self.conv = nn.Conv1d(c_in, c_out, int(kernel), dilation=int(dilation))

    def forward(self, x):
        return self.conv(F.pad(x, (self.pad, 0)))


class SpikeEncoder(nn.Module):
    def __init__(self, n_areas, d=mv.D_MODEL, n_lags=0, use_area=True):
        super().__init__()
        self.n_lags = int(n_lags)
        self.use_area = use_area
        in_dim = 1 + self.n_lags + 2
        if use_area:
            self.area_emb = nn.Embedding(n_areas, mv.AREA_DIM, padding_idx=0)
            in_dim += mv.AREA_DIM
        else:
            self.area_emb = None
        self.mlp = MLP(in_dim, d)

    def forward(self, counts, quality, rate, area, mask_n):
        feats = [counts.unsqueeze(-1)]
        if self.n_lags:
            feats.append(causal_lags(counts, self.n_lags))
        t_len = counts.size(-1)
        q = quality.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, t_len, 1)
        r = rate.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, t_len, 1)
        parts = feats + [q, r]
        if self.area_emb is not None:
            ae = self.area_emb(area).unsqueeze(2).expand(-1, -1, t_len, -1)
            parts.append(ae)
        h = self.mlp(torch.cat(parts, dim=-1))
        return h.masked_fill(~mask_n[:, :, None, None], 0.0)


class LFPEncoder(nn.Module):
    def __init__(self, n_areas, d=mv.D_MODEL, use_area=True):
        super().__init__()
        self.use_area = use_area
        in_dim = N_LFP_BANDS
        if use_area:
            self.area_emb = nn.Embedding(n_areas, mv.AREA_DIM, padding_idx=0)
            in_dim += mv.AREA_DIM
        else:
            self.area_emb = None
        self.mlp = MLP(in_dim, d)

    def forward(self, bands, area, mask_e):
        parts = [bands]
        if self.area_emb is not None:
            t_len = bands.size(2)
            ae = self.area_emb(area).unsqueeze(2).expand(-1, -1, t_len, -1)
            parts.append(ae)
        h = self.mlp(torch.cat(parts, dim=-1))
        return h.masked_fill(~mask_e[:, :, None, None], 0.0)


class HistoryLFPEncoder(nn.Module):
    """Shared longer-support causal conv over the 6 bands, then MLP."""

    def __init__(self, n_areas, d=mv.D_MODEL, band_scale="mid"):
        super().__init__()
        kernel, dilation = HISTORY_SPEC.get(band_scale, HISTORY_SPEC["mid"])
        self.conv = CausalConv1d(N_LFP_BANDS, d, kernel, dilation=dilation)
        self.base = LFPEncoder(n_areas, d)

    def forward(self, bands, area, mask_e):
        bsz, n_e, n_t, n_b = bands.shape
        x = bands.reshape(bsz * n_e, n_t, n_b).transpose(1, 2)
        h = self.conv(x).transpose(1, 2).reshape(bsz, n_e, n_t, -1)
        base = self.base(bands, area, mask_e)
        return (h + base).masked_fill(~mask_e[:, :, None, None], 0.0)


class BandLFPEncoder(nn.Module):
    """Frequency-specific causal receptive fields (candidate windows, not tuned)."""

    def __init__(self, n_areas, d=mv.D_MODEL, band_scale="mid"):
        super().__init__()
        specs = BAND_SPECS.get(band_scale, BAND_SPECS["mid"])
        ch = max(int(d) // 4, 1)
        self.branches = nn.ModuleList([CausalConv1d(1, ch, k, dilation=dil) for k, dil in specs])
        self.mix = MLP(6 * ch + mv.AREA_DIM, d)
        self.area_emb = nn.Embedding(n_areas, mv.AREA_DIM, padding_idx=0)

    def forward(self, bands, area, mask_e):
        bsz, n_e, n_t, n_b = bands.shape
        x = bands.reshape(bsz * n_e, n_t, n_b).transpose(1, 2)
        hs = []
        for i, conv in enumerate(self.branches):
            hs.append(conv(x[:, i : i + 1]).transpose(1, 2))
        h = torch.cat(hs, dim=-1).reshape(bsz, n_e, n_t, -1)
        ae = self.area_emb(area).unsqueeze(2).expand(-1, -1, n_t, -1)
        h = self.mix(torch.cat([h, ae], dim=-1))
        return h.masked_fill(~mask_e[:, :, None, None], 0.0)


class SpikeBandEncoder(nn.Module):
    def __init__(self, n_areas, d=mv.D_MODEL, n_lags=SPIKE_LAGS, band_scale="mid"):
        super().__init__()
        self.base = SpikeEncoder(n_areas, d, n_lags=n_lags)
        self.conv = CausalConv1d(1, d, kernel=SPIKE_CONV_K.get(band_scale, 5), dilation=1)
        self.mix = nn.Linear(2 * d, d)

    def forward(self, counts, quality, rate, area, mask_n):
        base = self.base(counts, quality, rate, area, mask_n)
        bsz, n_n, n_t = counts.shape
        h = self.conv(counts.reshape(bsz * n_n, 1, n_t)).transpose(1, 2).reshape(bsz, n_n, n_t, -1)
        return self.mix(torch.cat([base, h], dim=-1)).masked_fill(~mask_n[:, :, None, None], 0.0)


def _scatter_pool(h_u, idx, w, mask_n, n_e, valid=None):
    """Permutation-invariant weighted pool of units onto electrodes.

    h_u: (B, N, T, D), idx/w: (B, N, K). Invalid indices go to a dummy bin.
    """
    bsz, n_n, n_t, d = h_u.shape
    k = idx.size(-1)
    dummy = n_e
    out = h_u.new_zeros(bsz, n_e + 1, n_t, d)
    mass = h_u.new_zeros(bsz, n_e + 1)
    count = h_u.new_zeros(bsz, n_e + 1)
    batch_ix = torch.arange(bsz, device=h_u.device)[:, None].expand(bsz, n_n).reshape(-1)
    for ki in range(k):
        e = idx[:, :, ki].clone()
        ok = mask_n & (idx[:, :, ki] >= 0) & (idx[:, :, ki] < n_e)
        if valid is not None:
            ok = ok & valid[:, :, ki]
        wk = w[:, :, ki] * ok.to(h_u.dtype)
        e = e.masked_fill(~ok, dummy)
        src = (h_u * wk[:, :, None, None]).reshape(bsz * n_n, n_t * d)
        lin = batch_ix * (n_e + 1) + e.reshape(-1)
        out.view(bsz * (n_e + 1), n_t * d).index_add_(0, lin, src)
        mass.view(bsz * (n_e + 1)).index_add_(0, lin, wk.reshape(-1))
        count.view(bsz * (n_e + 1)).index_add_(0, lin, (wk > 0).to(h_u.dtype).reshape(-1))
    pooled = out[:, :n_e]
    mass = mass[:, :n_e]
    count = count[:, :n_e]
    pooled = pooled / mass.clamp(min=1e-6)[:, :, None, None]
    extra = torch.stack([count, mass], dim=-1)
    extra = extra.unsqueeze(2).expand(-1, -1, n_t, -1)
    return pooled, extra


def _gather_knn(h_e, idx, t0, t1):
    bsz = h_e.size(0)
    n_e = h_e.size(1)
    sl = h_e[:, :, t0:t1]
    bix = torch.arange(bsz, device=h_e.device)[:, None, None]
    return sl[bix, idx.clamp(0, max(n_e - 1, 0))]


def _gather_mask(mask, idx):
    """mask: (B, M); idx: (B, N, K) → (B, N, K) bool."""
    bsz, n_m = mask.shape
    n, k = idx.shape[1], idx.shape[2]
    take = idx.clamp(0, max(n_m - 1, 0)).reshape(bsz, n * k)
    return mask.gather(1, take).view(bsz, n, k)


class CrossSiteAttention(nn.Module):
    """Neurons as queries, neighboring electrode LFP as keys/values."""

    def __init__(self, d, nhead=None):
        super().__init__()
        nhead = nhead_of(d) if nhead is None else nhead
        self.scale = d ** -0.5
        self.q = nn.Linear(d, d)
        self.k = nn.Linear(d, d)
        self.v = nn.Linear(d, d)
        self.nhead = nhead
        self.hd = d // nhead
        self.out = nn.Linear(d, d)

    def forward(self, h_u, h_e, idx, valid, mask_n, mask_e):
        bsz, n_n, n_t, d = h_u.shape
        n_e = h_e.size(1)
        k_n = idx.size(-1)
        nh, hd = self.nhead, self.hd
        outs = []
        ok = valid & mask_n.unsqueeze(-1) & _gather_mask(mask_e, idx)
        for t0 in range(0, n_t, TIME_CHUNK):
            t1 = min(n_t, t0 + TIME_CHUNK)
            q = self.q(h_u[:, :, t0:t1]).view(bsz, n_n, t1 - t0, nh, hd)
            neigh = _gather_knn(h_e, idx, t0, t1)
            k = self.k(neigh).view(bsz, n_n, k_n, t1 - t0, nh, hd)
            v = self.v(neigh).view(bsz, n_n, k_n, t1 - t0, nh, hd)
            logits = (q.unsqueeze(2) * k).sum(-1) * self.scale
            logits = logits.masked_fill(~ok[:, :, :, None, None], -1e9)
            alpha = torch.softmax(logits, dim=2)
            alpha = torch.nan_to_num(alpha, nan=0.0)
            ctx = (alpha.unsqueeze(-1) * v).sum(dim=2).reshape(bsz, n_n, t1 - t0, d)
            outs.append(self.out(ctx))
        h = torch.cat(outs, dim=2)
        return h.masked_fill(~mask_n[:, :, None, None], 0.0)


class ResidualUpdate(nn.Module):
    def __init__(self, d, dropout=mv.DROPOUT):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(2 * d, d), nn.GELU(), nn.Dropout(dropout), nn.Linear(d, d))
        self.norm = nn.LayerNorm(d)

    def forward(self, h, msg):
        return self.norm(h + self.net(torch.cat([h, msg], dim=-1)))


class BipartiteGraph(nn.Module):
    """Neuron ↔ electrode messages plus electrode–electrode probe adjacency."""

    def __init__(self, d, n_layers=GRAPH_LAYERS, dropout=mv.DROPOUT):
        super().__init__()
        self.n_layers = n_layers
        self.ne = nn.ModuleList([CrossSiteAttention(d) for _ in range(n_layers)])
        self.phi_n = nn.ModuleList([ResidualUpdate(d, dropout) for _ in range(n_layers)])
        self.phi_e = nn.ModuleList([ResidualUpdate(d, dropout) for _ in range(n_layers)])
        self.ee_psi = nn.ModuleList(
            [MLP(2 * d + 1, d) for _ in range(n_layers)]
        )

    def forward(
        self,
        h_u,
        h_e,
        unit_knn,
        unit_knn_valid,
        unit_w,
        elec_knn,
        elec_knn_valid,
        elec_knn_dist,
        mask_n,
        mask_e,
        use_spikes=True,
        use_lfp=True,
    ):
        n_e = h_e.size(1)
        for layer in range(self.n_layers):
            if use_spikes and use_lfp:
                ctx_u = self.ne[layer](h_u, h_e, unit_knn, unit_knn_valid, mask_n, mask_e)
                h_u = self.phi_n[layer](h_u, ctx_u)
            if use_spikes:
                pooled, _ = _scatter_pool(h_u, unit_knn, unit_w, mask_n, n_e, valid=unit_knn_valid)
                h_e = self.phi_e[layer](h_e, pooled)
            h_e = self._ee(h_e, elec_knn, elec_knn_valid, elec_knn_dist, mask_e, layer)
            h_u = h_u.masked_fill(~mask_n[:, :, None, None], 0.0)
            h_e = h_e.masked_fill(~mask_e[:, :, None, None], 0.0)
        return h_u, h_e

    def _ee(self, h_e, idx, valid, dist, mask_e, layer):
        bsz, n_e, n_t, d = h_e.shape
        k = idx.size(-1)
        ok = valid & mask_e.unsqueeze(-1) & _gather_mask(mask_e, idx)
        outs = []
        for t0 in range(0, n_t, TIME_CHUNK):
            t1 = min(n_t, t0 + TIME_CHUNK)
            src = h_e[:, :, t0:t1]
            neigh = _gather_knn(h_e, idx, t0, t1)
            self_exp = src.unsqueeze(2).expand_as(neigh)
            dd = dist.unsqueeze(3).unsqueeze(-1).expand(-1, -1, -1, t1 - t0, 1)
            msg = self.ee_psi[layer](torch.cat([self_exp, neigh, dd], dim=-1))
            w = ok.to(h_e.dtype).unsqueeze(3).unsqueeze(-1)
            pooled = (msg * w).sum(dim=2) / w.sum(dim=2).clamp(min=1.0)
            outs.append(pooled)
        return torch.cat(outs, dim=2)


class FuseHead(nn.Module):
    def __init__(self, d, pos_dim, kind):
        super().__init__()
        self.kind = kind
        self.pos_dim = int(pos_dim)
        extra = self.pos_dim + 2
        if kind == "concat":
            self.net = MLP(2 * d + extra, d)
        elif kind == "gate":
            self.proj_s = nn.Linear(d + extra, d)
            self.proj_l = nn.Linear(d + extra, d)
            self.gate = nn.Linear(d + extra, d)
        elif kind == "residual":
            self.proj_s = nn.Linear(d + extra, d)
            self.delta = MLP(2 * d + extra, d)
        elif kind == "private":
            self.shared = MLP(2 * d + extra, d)
            self.priv_s = MLP(d + extra, d)
            self.priv_l = MLP(d + extra, d)
            self.mix = nn.Linear(3 * d, d)
            self.dec_s = MLP(2 * d, d)
            self.dec_l = MLP(2 * d, d)
            self.cross_s = MLP(d, d)
            self.cross_l = MLP(d, d)
        else:
            self.proj_s = nn.Linear(d + extra, d)
            self.proj_l = nn.Linear(d + extra, d)
            self.mix = MLP(2 * d, d)

    def _pack(self, s, l, pos, extra):
        parts = [extra]
        if self.pos_dim:
            t_len = s.size(2)
            parts.append(pos.unsqueeze(2).expand(-1, -1, t_len, -1))
        aux = torch.cat(parts, dim=-1) if parts else s.new_zeros(*s.shape[:-1], 0)
        return aux

    def forward(self, s, l, pos, extra, mask_e):
        aux = self._pack(s, l, pos, extra)
        kind = self.kind
        if kind == "concat":
            h = self.net(torch.cat([s, l, aux], dim=-1))
            return h.masked_fill(~mask_e[:, :, None, None], 0.0), {}
        if kind == "gate":
            sa = torch.cat([s, aux], dim=-1)
            la = torch.cat([l, aux], dim=-1)
            g = torch.sigmoid(self.gate(la))
            h = g * self.proj_s(sa) + self.proj_l(la)
            return h.masked_fill(~mask_e[:, :, None, None], 0.0), {}
        if kind == "residual":
            sa = torch.cat([s, aux], dim=-1)
            h = self.proj_s(sa) + self.delta(torch.cat([s, l, aux], dim=-1))
            return h.masked_fill(~mask_e[:, :, None, None], 0.0), {}
        if kind == "private":
            zc = self.shared(torch.cat([s, l, aux], dim=-1))
            zs = self.priv_s(torch.cat([s, aux], dim=-1))
            zl = self.priv_l(torch.cat([l, aux], dim=-1))
            h = self.mix(torch.cat([zc, zs, zl], dim=-1))
            aux_loss = {
                "spike": _masked_mse(self.dec_s(torch.cat([zc, zs], dim=-1)), s.detach(), mask_e),
                "lfp": _masked_mse(self.dec_l(torch.cat([zc, zl], dim=-1)), l.detach(), mask_e),
                "cross": _masked_mse(self.cross_l(zs), zc.detach(), mask_e)
                + _masked_mse(self.cross_s(zl), zc.detach(), mask_e),
            }
            return h.masked_fill(~mask_e[:, :, None, None], 0.0), aux_loss
        sa = self.proj_s(torch.cat([s, aux], dim=-1))
        la = self.proj_l(torch.cat([l, aux], dim=-1))
        h = self.mix(torch.cat([sa, la], dim=-1))
        return h.masked_fill(~mask_e[:, :, None, None], 0.0), {}


def _masked_mse(pred, target, mask):
    w = mask[:, :, None, None].to(pred.dtype)
    return ((pred - target).pow(2) * w).sum() / w.sum().clamp(min=1.0)


class LatentQueryPool(nn.Module):
    def __init__(self, d, n_queries=K_QUERIES, nhead=None, dropout=mv.DROPOUT):
        super().__init__()
        nhead = nhead_of(d) if nhead is None else nhead
        self.n_queries = n_queries
        self.q = nn.Parameter(torch.randn(n_queries, d) * 0.02)
        self.attn = nn.MultiheadAttention(d, nhead, dropout=dropout, batch_first=True)

    def forward(self, h, mask_e):
        bsz, n_e, n_t, d = h.shape
        nq = self.n_queries
        q = self.q.unsqueeze(0).expand(bsz, -1, -1)
        pad = ~mask_e
        empty = pad.all(dim=-1)
        if empty.any():
            pad = pad.clone()
            pad[empty, 0] = False
        outs = []
        for t0 in range(0, n_t, TIME_CHUNK):
            t1 = min(n_t, t0 + TIME_CHUNK)
            tc = t1 - t0
            ht = h[:, :, t0:t1].permute(0, 2, 1, 3).reshape(bsz * tc, n_e, d)
            qt = q.unsqueeze(1).expand(bsz, tc, nq, d).reshape(bsz * tc, nq, d)
            pad_t = pad.unsqueeze(1).expand(bsz, tc, n_e).reshape(bsz * tc, n_e)
            out, _ = self.attn(qt, ht, ht, key_padding_mask=pad_t)
            outs.append(out.view(bsz, tc, nq, d))
        return torch.cat(outs, dim=1)


class MultiRateSSM(nn.Module):
    """Shared latent with separate spike / LFP updates on the decoder grid.

    Diagonal Kalman-style gain plus a learned residual, batched over trials and
    latent queries. Missing LFP (padded electrodes already pooled to 0) is
    gated by a per-time mask.
    """

    def __init__(self, d, d_z=16):
        super().__init__()
        self.d_z = d_z
        self.in_s = nn.Linear(d, d_z)
        self.in_l = nn.Linear(d, d_z)
        self.F = nn.Linear(d_z, d_z)
        self.log_q = nn.Parameter(torch.zeros(d_z))
        self.log_rs = nn.Parameter(torch.zeros(d_z))
        self.log_rl = nn.Parameter(torch.zeros(d_z))
        self.U_s = MLP(2 * d_z, d_z)
        self.U_l = MLP(2 * d_z, d_z)
        self.out = nn.Linear(d_z, d)

    def forward(self, z_s, z_l, mask_t, use_s=True, use_l=True):
        bsz, n_t, nq, d = z_s.shape
        z = z_s.new_zeros(bsz, nq, self.d_z)
        p = z_s.new_ones(bsz, nq, self.d_z)
        q = F.softplus(self.log_q) + 1e-4
        rs = F.softplus(self.log_rs) + 1e-4
        rl = F.softplus(self.log_rl) + 1e-4
        outs = []
        for t in range(n_t):
            z = torch.tanh(self.F(z))
            p = 0.9 * p + q
            if use_s:
                obs_s = self.in_s(z_s[:, t])
                ks = p / (p + rs)
                z = z + ks * self.U_s(torch.cat([z, obs_s], dim=-1))
                p = (1.0 - ks).clamp(min=1e-4) * p
            if use_l:
                obs_l = self.in_l(z_l[:, t])
                kl = p / (p + rl)
                z = z + kl * self.U_l(torch.cat([z, obs_l], dim=-1))
                p = (1.0 - kl).clamp(min=1e-4) * p
            m = mask_t[:, t].to(z.dtype)[:, None, None]
            outs.append(self.out(z) * m)
        return torch.stack(outs, dim=1)


class RecurrentDualSSM(nn.Module):
    """Recurrent state estimator with separate spike and LFP updates."""

    def __init__(self, d, d_z=16, cell="gru"):
        super().__init__()
        self.d_z = d_z
        self.cell = cell
        self.in_s = nn.Linear(d, d_z)
        self.in_l = nn.Linear(d, d_z)
        self.out = nn.Linear(d_z, d)
        if cell == "lstm":
            self.pred = nn.LSTMCell(d_z, d_z)
            self.us = nn.LSTMCell(d_z, d_z)
            self.ul = nn.LSTMCell(d_z, d_z)
        elif cell == "residual":
            self.pred = MLP(d_z, d_z)
            self.us = MLP(2 * d_z, d_z)
            self.ul = MLP(2 * d_z, d_z)
        else:
            self.pred = nn.GRUCell(d_z, d_z)
            self.us = nn.GRUCell(d_z, d_z)
            self.ul = nn.GRUCell(d_z, d_z)

    def forward(self, z_s, z_l, mask_t, use_s=True, use_l=True):
        bsz, n_t, nq, d = z_s.shape
        h = z_s.new_zeros(bsz * nq, self.d_z)
        c = z_s.new_zeros(bsz * nq, self.d_z) if self.cell == "lstm" else None
        outs = []
        for t in range(n_t):
            obs_s = self.in_s(z_s[:, t]).reshape(bsz * nq, self.d_z)
            obs_l = self.in_l(z_l[:, t]).reshape(bsz * nq, self.d_z)
            if self.cell == "lstm":
                h, c = self.pred(h, (h, c))
                if use_s:
                    h, c = self.us(obs_s, (h, c))
                if use_l:
                    h, c = self.ul(obs_l, (h, c))
            elif self.cell == "residual":
                h = h + torch.tanh(self.pred(h))
                if use_s:
                    h = h + self.us(torch.cat([h, obs_s], dim=-1))
                if use_l:
                    h = h + self.ul(torch.cat([h, obs_l], dim=-1))
            else:
                h = self.pred(h, h)
                if use_s:
                    h = self.us(obs_s, h)
                if use_l:
                    h = self.ul(obs_l, h)
            m = mask_t[:, t].to(h.dtype)[:, None, None]
            outs.append(self.out(h.view(bsz, nq, self.d_z)) * m)
        return torch.stack(outs, dim=1)


class ContinuousTimeSSM(nn.Module):
    """Learned continuous-time dynamics: z_{t+Δ} = z_t + Δ F_θ(z_t, Δ), then U_s / U_l."""

    def __init__(self, d, d_z=16, depth=2, dt=DT_S):
        super().__init__()
        self.d_z = d_z
        self.dt = float(dt)
        extra = depth >= 2
        self.F = MLP(d_z + 1, d_z, extra=extra)
        if int(depth) >= 3:
            self.F2 = MLP(d_z, d_z)
        else:
            self.F2 = None
        self.in_s = nn.Linear(d, d_z)
        self.in_l = nn.Linear(d, d_z)
        self.U_s = MLP(2 * d_z, d_z)
        self.U_l = MLP(2 * d_z, d_z)
        self.out = nn.Linear(d_z, d)

    def forward(self, z_s, z_l, mask_t, use_s=True, use_l=True):
        bsz, n_t, nq, d = z_s.shape
        z = z_s.new_zeros(bsz, nq, self.d_z)
        dt = z.new_full((bsz, nq, 1), self.dt)
        outs = []
        for t in range(n_t):
            drift = torch.tanh(self.F(torch.cat([z, dt], dim=-1)))
            if self.F2 is not None:
                drift = drift + torch.tanh(self.F2(z))
            z = z + self.dt * drift
            if use_s:
                obs_s = self.in_s(z_s[:, t])
                z = z + self.U_s(torch.cat([z, obs_s], dim=-1))
            if use_l:
                obs_l = self.in_l(z_l[:, t])
                z = z + self.U_l(torch.cat([z, obs_l], dim=-1))
            m = mask_t[:, t].to(z.dtype)[:, None, None]
            outs.append(self.out(z) * m)
        return torch.stack(outs, dim=1)


class FuseSequence(nn.Module):
    def __init__(self, n_areas, cfg):
        super().__init__()
        hp = hp_of(cfg)
        self.cfg = cfg
        self.assoc = cfg["assoc"]
        self.fuse = cfg["fuse"]
        self.pos = cfg["pos"]
        self.encoder = cfg.get("encoder", "instant")
        self.temporal = cfg["temporal"]
        self.soft_norm = cfg.get("soft_norm", "norm")
        self.use_spikes = uses_spikes(cfg)
        self.use_lfp = uses_lfp(cfg)
        self.drop_hg = drop_high_gamma(cfg)
        self.pool_name = hp["pool"]
        d = int(hp["d_model"])
        self.d = d
        nq = int(hp["n_queries"])
        n_layers = int(hp["n_layers"])
        dropout = float(hp["dropout"])
        lags = int(hp["spike_lags"]) if self.encoder in ("history", "bandconv") else 0
        scale = hp["band_scale"]
        pos_dim = POS_DIM[self.pos]
        if self.encoder == "bandconv":
            self.enc_s = SpikeBandEncoder(n_areas, d, n_lags=max(lags, 1), band_scale=scale)
            self.enc_l = BandLFPEncoder(n_areas, d, band_scale=scale)
        elif self.encoder == "history":
            self.enc_s = SpikeEncoder(n_areas, d, n_lags=max(lags, 1))
            self.enc_l = HistoryLFPEncoder(n_areas, d, band_scale=scale)
        else:
            self.enc_s = SpikeEncoder(n_areas, d, n_lags=0)
            self.enc_l = LFPEncoder(n_areas, d)
        self.xattn = CrossSiteAttention(d) if self.fuse == "xattn" or self.assoc == "graph" else None
        self.graph = (
            BipartiteGraph(d, n_layers=int(hp["graph_layers"]), dropout=dropout)
            if self.assoc == "graph"
            else None
        )
        self.fuse_head = FuseHead(d, pos_dim, "concat" if self.fuse == "xattn" else self.fuse)
        if self.pool_name == "mean":
            self.pool = mv.MeanPool(d)
            mix_in = d
        elif self.pool_name == "attn":
            self.pool = mv.AttnPool(d)
            mix_in = d
        else:
            self.pool = LatentQueryPool(d, n_queries=nq, dropout=dropout)
            mix_in = nq * d
        self.token_mix = nn.Linear(mix_in, d)
        drop_rnn = dropout if n_layers > 1 else 0.0
        if self.temporal == "gru":
            self.rnn = nn.GRU(d, d, num_layers=n_layers, batch_first=True, dropout=drop_rnn)
            self.ssm = None
            self.use_pack = True
        elif self.temporal == "kalman":
            self.rnn = None
            self.ssm = MultiRateSSM(d, d_z=int(hp["kalman_dz"]))
            self.use_pack = False
        elif self.temporal == "recurrent":
            self.rnn = None
            self.ssm = RecurrentDualSSM(d, d_z=int(hp["kalman_dz"]), cell=hp["rnn_cell"])
            self.use_pack = False
        elif self.temporal == "cde":
            self.rnn = None
            self.ssm = ContinuousTimeSSM(d, d_z=int(hp["kalman_dz"]), depth=int(hp["cde_depth"]), dt=DT_S)
            self.use_pack = False
        else:
            self.rnn = mv.CausalTransformer(d, n_layers=n_layers, nhead=nhead_of(d), dropout=dropout)
            self.ssm = None
            self.use_pack = False
        self.head = mv.Head(d, p=dropout)

    def _associate(self, h_u, h_l, batch):
        n_e = h_l.size(1)
        mask_n, mask_e = batch["mask_n"], batch["mask_e"]
        if self.assoc == "hard":
            idx = batch["hard_idx"]
            w = torch.ones(idx.shape, dtype=h_u.dtype, device=h_u.device)
            valid = batch["hard_valid"]
        else:
            idx, w, valid = batch["unit_knn"], batch["unit_w"], batch["unit_knn_valid"]
        if not self.use_spikes:
            extra = h_l.new_zeros(h_l.size(0), n_e, h_l.size(2), 2)
            s_e = h_l.new_zeros(h_l.shape)
            if self.assoc == "graph" and self.use_lfp:
                _h_u, h_e = self.graph(
                    h_u,
                    h_l,
                    batch["unit_knn"],
                    batch["unit_knn_valid"],
                    batch["unit_w"],
                    batch["elec_knn"],
                    batch["elec_knn_valid"],
                    batch["elec_knn_dist"],
                    mask_n,
                    mask_e,
                    use_spikes=False,
                    use_lfp=True,
                )
                return s_e, h_e, extra, h_u
            return s_e, h_l, extra, h_u
        if self.assoc == "graph":
            h_u, h_e = self.graph(
                h_u,
                h_l,
                batch["unit_knn"],
                batch["unit_knn_valid"],
                batch["unit_w"],
                batch["elec_knn"],
                batch["elec_knn_valid"],
                batch["elec_knn_dist"],
                mask_n,
                mask_e,
                use_spikes=True,
                use_lfp=self.use_lfp,
            )
            s_e, extra = _scatter_pool(h_u, idx, w, mask_n, n_e, valid=valid)
            return s_e, h_e, extra, h_u
        if self.fuse == "xattn" and self.use_lfp:
            h_u = self.xattn(h_u, h_l, batch["unit_knn"], batch["unit_knn_valid"], mask_n, mask_e)
        s_e, extra = _scatter_pool(h_u, idx, w, mask_n, n_e, valid=valid)
        if not self.use_lfp:
            h_l = h_l.new_zeros(h_l.shape)
        return s_e, h_l, extra, h_u

    def _pool_tokens(self, h, mask_e):
        if self.pool_name == "latent":
            return self.pool(h, mask_e)
        pop = self.pool(h, mask_e)
        return pop.unsqueeze(2)

    def forward(self, batch, unit_drop=0.0):
        mask_n = batch["mask_n"]
        if self.training and unit_drop > 0:
            mask_n = mv.apply_unit_dropout(mask_n, unit_drop)
            batch = dict(batch)
            batch["mask_n"] = mask_n
        spikes = batch["spikes"]
        lfp = batch["lfp"]
        if not self.use_spikes:
            spikes = torch.zeros_like(spikes)
        if not self.use_lfp:
            lfp = torch.zeros_like(lfp)
        elif self.drop_hg:
            lfp = lfp.clone()
            lfp[..., HIGH_GAMMA_IDX] = 0
        if self.use_spikes:
            h_u = self.enc_s(spikes, batch["quality"], batch["rate"], batch["unit_area"], mask_n)
        else:
            bsz, n_n, n_t = spikes.shape
            h_u = spikes.new_zeros(bsz, n_n, n_t, self.d)
        if self.use_lfp:
            h_l = self.enc_l(lfp, batch["elec_area"], batch["mask_e"])
        else:
            bsz, n_e, n_t, _ = lfp.shape
            h_l = lfp.new_zeros(bsz, n_e, n_t, self.d)
        s_e, l_e, extra, _h_u = self._associate(h_u, h_l, batch)
        h_e, aux = self.fuse_head(s_e, l_e, batch["pos"], extra, batch["mask_e"])
        if aux:
            if not self.use_spikes:
                aux = {**aux, "spike": aux["spike"] * 0, "cross": aux.get("cross", aux["spike"]) * 0}
            if not self.use_lfp:
                aux = {**aux, "lfp": aux["lfp"] * 0, "cross": aux.get("cross", aux["lfp"]) * 0}
        mask_t = batch["mask_t"]
        if self.ssm is not None:
            z_s = self._pool_tokens(s_e, batch["mask_e"])
            z_l = self._pool_tokens(l_e, batch["mask_e"])
            out = self.ssm(z_s, z_l, mask_t, use_s=self.use_spikes, use_l=self.use_lfp)
            out = self.token_mix(out.reshape(out.size(0), out.size(1), -1)) * mask_t.unsqueeze(-1)
        else:
            z = self._pool_tokens(h_e, batch["mask_e"])
            h = self.token_mix(z.reshape(z.size(0), z.size(1), -1)) * mask_t.unsqueeze(-1)
            if self.use_pack:
                packed = pack_padded_sequence(h, batch["lengths"].cpu(), batch_first=True, enforce_sorted=False)
                out, _ = self.rnn(packed)
                out, _ = pad_packed_sequence(out, batch_first=True, total_length=h.size(1))
            else:
                out = self.rnn(h, mask_t)
        pred = self.head(out)
        return pred, aux


# ---------------------------------------------------------------------------
# Train / score
# ---------------------------------------------------------------------------


def zscore_lfp(trials, idx, mean, std):
    mean = np.asarray(mean, dtype=np.float32).reshape(1, 1, -1)
    std = np.asarray(std, dtype=np.float32).reshape(1, 1, -1)
    for i in np.asarray(idx, dtype=int):
        t = trials[int(i)]
        if not ec.trial_has_lfp(t):
            continue
        t["lfp_z"] = ((np.asarray(t["lfp"], dtype=np.float32) - mean) / std).astype(np.float32)


def cache_dir(eval_name, fold_id, model_name):
    configure_out()
    safe = "".join(ch if ch.isalnum() or ch in "-_." else "_" for ch in str(fold_id))
    d = CACHE / eval_name / safe / model_name / "trace" / "mse"
    d.mkdir(parents=True, exist_ok=True)
    return d


def _is_oom(err):
    if isinstance(err, MemoryError):
        return True
    if type(err).__name__ in ("OutOfMemoryError", "CUDAOutOfMemoryError"):
        return True
    return "out of memory" in str(err).lower()


def _empty_cuda():
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def make_loader(trials, idx, y_mean, y_std, stats, cfg, shuffle, seed, batch):
    ds = FuseDS(trials, idx, y_mean, y_std, stats, cfg)
    if shuffle:
        sampler = ec.StrataSampler(trials, idx, n_samples=len(ds), seed=seed)
        return DataLoader(ds, batch_size=batch, sampler=sampler, collate_fn=collate_fuse)
    return DataLoader(ds, batch_size=batch, shuffle=False, collate_fn=collate_fuse)


def batch_to_device(batch, dev):
    out = {}
    for k, v in batch.items():
        out[k] = v.to(dev) if torch.is_tensor(v) and k != "lengths" else v
    return out


def aux_total(aux, cfg=None):
    if not aux:
        return None
    hp = hp_of(cfg or {})
    loss = 0.0
    loss = loss + float(hp["lambda_s"]) * aux.get("spike", 0.0)
    loss = loss + float(hp["lambda_l"]) * aux.get("lfp", 0.0)
    loss = loss + float(hp["lambda_c"]) * aux.get("cross", 0.0)
    return loss


def run_epoch(model, loader, opt, dev, train, cfg=None):
    model.train(train)
    hp = hp_of(cfg or {})
    drop = float(hp["unit_drop"]) if train else 0.0
    total, n = 0.0, 0
    for batch in loader:
        batch = batch_to_device(batch, dev)
        y = batch["y"]
        mask = batch["mask_loss"]
        if train:
            opt.zero_grad(set_to_none=True)
        pred, aux = model(batch, unit_drop=drop)
        loss = mv.trial_balanced_mse(pred, y, mask)
        extra = aux_total(aux, cfg)
        if extra is not None:
            loss = loss + extra
        if train:
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
        bs = int(y.size(0))
        total += float(loss.item()) * bs
        n += bs
    return total / max(n, 1)


def train_batch_size(assoc):
    return GRAPH_BATCH if assoc == "graph" else FUSE_BATCH


def build_model(stats, cfg):
    return FuseSequence(stats["n_areas"], cfg)


def train_neural(cfg, trials, tr_idx, va_idx, y_mean, y_std, stats, seed):
    cdir = cache_dir(cfg["eval"], cfg["fold_id"], cfg["model"])
    ckpt, hist_path = cdir / "model.pt", cdir / "history.json"
    if ckpt.exists() and hist_path.exists():
        hist = json.loads(hist_path.read_text())
        if hist.get("finished"):
            log(f"  cache hit {cfg['eval']}/{cfg['fold_id']}/{cfg['model']}")
            return torch.load(ckpt, map_location="cpu", weights_only=False), hist
    dev = slurm_utils.device_of()
    batch = train_batch_size(cfg["assoc"])
    max_ep = mv.MAX_EPOCHS
    if MAX_EPOCHS_CAP is not None:
        max_ep = min(int(max_ep), int(MAX_EPOCHS_CAP))
    model = None
    best_state = None
    hist = None
    while True:
        try:
            _empty_cuda()
            model = build_model(stats, cfg).to(dev)
            tr_ld = make_loader(trials, tr_idx, y_mean, y_std, stats, cfg, True, seed, batch)
            va_ld = make_loader(trials, va_idx, y_mean, y_std, stats, cfg, False, seed, batch)
            hp = hp_of(cfg)
            opt = torch.optim.AdamW(model.parameters(), lr=float(hp["lr"]), weight_decay=mv.WEIGHT_DECAY)
            hist = {
                "train": [],
                "val": [],
                "n_params": mv.n_params(model),
                "finished": False,
                "best_epoch": 0,
                "batch": int(batch),
            }
            log(f"  {cfg['model']} params={hist['n_params']} device={dev} batch={batch}")
            slurm_utils.write_status(state="training", epoch=0, batch=int(batch), n_params=hist["n_params"])
            best, bad, best_state = math.inf, 0, None
            for epoch in range(1, max_ep + 1):
                tr = run_epoch(model, tr_ld, opt, dev, True, cfg)
                with torch.no_grad():
                    va = run_epoch(model, va_ld, None, dev, False, cfg)
                hist["train"].append(tr)
                hist["val"].append(va)
                log(f"    {cfg['model']} {epoch:02d}  train={tr:.4f}  val={va:.4f}")
                slurm_utils.write_status(
                    state="training", epoch=epoch, train=tr, val=va, best_epoch=hist["best_epoch"], batch=int(batch)
                )
                if va + 1e-5 < best:
                    best, bad = va, 0
                    best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
                    hist["best_epoch"] = epoch
                else:
                    bad += 1
                if bad >= mv.PATIENCE:
                    log(f"    early stop at epoch {epoch}")
                    break
            break
        except (RuntimeError, MemoryError) as err:
            if not _is_oom(err) or batch <= 1:
                raise
            log(f"  OOM at batch={batch}; retrying at {max(1, batch // 2)}")
            batch = max(1, batch // 2)
            model = None
            _empty_cuda()
    if best_state is not None:
        model.load_state_dict(best_state)
    hist["finished"] = True
    blob = {
        "state_dict": {k: v.detach().cpu().clone() for k, v in model.state_dict().items()},
        "y_mean": y_mean,
        "y_std": y_std,
        "n_areas": stats["n_areas"],
        "model": cfg["model"],
        "cfg": dict(cfg),
        "stats": {
            "vocab": stats["vocab"],
            "n_areas": stats["n_areas"],
            "xyz_mean": stats["xyz_mean"],
            "xyz_std": stats["xyz_std"],
            "axial_mean": stats["axial_mean"],
            "axial_std": stats["axial_std"],
            "lateral_mean": stats["lateral_mean"],
            "lateral_std": stats["lateral_std"],
            "local_std": stats["local_std"],
            "rates": stats["rates"],
        },
        "lfp_mean": stats.get("lfp_mean"),
        "lfp_std": stats.get("lfp_std"),
    }
    torch.save(blob, ckpt)
    mv.save_json(hist_path, hist)
    return blob, hist


def predict_neural(blob, trials, idx, stats, cfg):
    dev = slurm_utils.device_of()
    st = dict(stats)
    st.update(blob.get("stats") or {})
    model = build_model(st, cfg)
    model.load_state_dict(blob["state_dict"], strict=False)
    model.to(dev)
    model.eval()
    batch = train_batch_size(cfg["assoc"])
    while True:
        ld = make_loader(trials, idx, blob["y_mean"], blob["y_std"], st, cfg, False, 0, batch)
        try:
            recs = []
            with torch.no_grad():
                for batch_d in ld:
                    batch_d = batch_to_device(batch_d, dev)
                    pred, _aux = model(batch_d, unit_drop=0.0)
                    y_z = batch_d["y"].cpu().numpy()
                    mt = batch_d["mask_t"].cpu().numpy()
                    lengths = batch_d["lengths"].numpy()
                    tr = pred.cpu().numpy()
                    for b in range(tr.shape[0]):
                        T = int(lengths[b])
                        valid = mt[b, :T]
                        y = y_z[b, :T] * blob["y_std"] + blob["y_mean"]
                        p = tr[b, :T] * blob["y_std"] + blob["y_mean"]
                        p = np.asarray(p, dtype=np.float32).reshape(T, -1)
                        y = np.asarray(y, dtype=np.float32).reshape(T, -1)
                        p[~valid] = np.nan
                        y[~valid] = np.nan
                        recs.append({"pred": p, "y": y})
            return recs
        except (RuntimeError, MemoryError) as err:
            if not _is_oom(err) or batch <= 1:
                raise
            log(f"  OOM at predict batch={batch}; retrying at {max(1, batch // 2)}")
            batch = max(1, batch // 2)
            _empty_cuda()


# ---------------------------------------------------------------------------
# Job grid
# ---------------------------------------------------------------------------


def _arch_cfg(assoc, fuse, pos, encoder, temporal, soft_norm="norm", **hp):
    rec = {
        "assoc": assoc,
        "fuse": fuse,
        "pos": pos,
        "encoder": encoder,
        "temporal": temporal,
        "soft_norm": soft_norm,
        "objective": "trace",
        "loss": "mse",
    }
    rec.update(hp)
    rec["model"] = model_name_of(rec)
    return rec


def iter_archs(assocs, fuses, poses, encoders, temporals, soft_norms):
    for assoc in assocs:
        norms = soft_norms if assoc == "soft" else ("norm",)
        for fuse in fuses:
            for pos in poses:
                for encoder in encoders:
                    for temporal in temporals:
                        for sn in norms:
                            yield _arch_cfg(assoc, fuse, pos, encoder, temporal, sn)


def iter_core(assocs, fuses, soft_norms):
    for assoc in assocs:
        sn = "norm" if assoc != "soft" else (soft_norms[0] if soft_norms else "norm")
        for fuse in fuses:
            yield _arch_cfg(assoc, fuse, "atlas", "instant", "transformer", sn)


def iter_pilot(assocs, fuses, poses, encoders, temporals, soft_norms):
    """Small screen: every assoc × fuse, then one-at-a-time extras. No HP sweep."""
    seen = set()
    for rec in iter_core(assocs, fuses, soft_norms):
        seen.add(rec["model"])
        yield rec
    if "hard" not in assocs or "concat" not in fuses:
        return
    extras = []
    for pos in poses:
        if pos != "atlas":
            extras.append(_arch_cfg("hard", "concat", pos, "instant", "transformer"))
    for enc in encoders:
        if enc != "instant":
            extras.append(_arch_cfg("hard", "concat", "atlas", enc, "transformer"))
    for temporal in temporals:
        if temporal != "transformer":
            extras.append(_arch_cfg("hard", "concat", "atlas", "instant", temporal))
    if "soft" in assocs and "unnorm" in soft_norms:
        extras.append(_arch_cfg("soft", "concat", "atlas", "instant", "transformer", "unnorm"))
    for rec in extras:
        if rec["model"] not in seen:
            seen.add(rec["model"])
            yield rec


def hp_oat_cfgs():
    """Three values of each hyperparameter, one-at-a-time on a family base.

    The default value is already present in the architecture cross, so only
    the two non-default settings are emitted here.
    """
    base_hard = dict(assoc="hard", fuse="concat", pos="atlas", encoder="instant", temporal="transformer", soft_norm="norm")
    base_soft = dict(assoc="soft", fuse="concat", pos="atlas", encoder="instant", temporal="transformer", soft_norm="norm")
    base_graph = dict(assoc="graph", fuse="concat", pos="atlas", encoder="instant", temporal="transformer", soft_norm="norm")
    base_hist = dict(assoc="hard", fuse="concat", pos="atlas", encoder="history", temporal="transformer", soft_norm="norm")
    base_band = dict(assoc="hard", fuse="concat", pos="atlas", encoder="bandconv", temporal="transformer", soft_norm="norm")
    base_kalman = dict(assoc="hard", fuse="concat", pos="atlas", encoder="instant", temporal="kalman", soft_norm="norm")
    base_rec = dict(assoc="hard", fuse="concat", pos="atlas", encoder="instant", temporal="recurrent", soft_norm="norm")
    base_cde = dict(assoc="hard", fuse="concat", pos="atlas", encoder="instant", temporal="cde", soft_norm="norm")
    base_priv = dict(assoc="hard", fuse="private", pos="atlas", encoder="instant", temporal="transformer", soft_norm="norm")
    sweeps = [
        (base_hard, "d_model", (32, 64, 128)),
        (base_hard, "n_queries", (2, 4, 8)),
        (base_hard, "n_layers", (1, 2, 3)),
        (base_hard, "dropout", (0.0, 0.1, 0.2)),
        (base_hard, "pool", ("latent", "mean", "attn")),
        (base_hard, "unit_drop", (0.0, 0.15, 0.3)),
        (base_hard, "lr", (3e-4, 1e-3, 3e-3)),
        (base_soft, "sigma", (25.0, 50.0, 100.0)),
        (base_soft, "k_ne", (4, 8, 16)),
        (base_graph, "graph_layers", (1, 2, 3)),
        (base_graph, "k_ee", (4, 6, 12)),
        (base_graph, "k_ne", (4, 8, 16)),
        (base_hist, "spike_lags", (2, 4, 8)),
        (base_band, "band_scale", ("short", "mid", "long")),
        (base_kalman, "kalman_dz", (8, 16, 32)),
        (base_rec, "rnn_cell", ("gru", "lstm", "residual")),
        (base_cde, "cde_depth", (1, 2, 3)),
        (base_priv, "lambda_s", (0.01, 0.1, 1.0)),
        (base_priv, "lambda_l", (0.01, 0.1, 1.0)),
        (base_priv, "lambda_c", (0.01, 0.1, 1.0)),
    ]
    out = []
    for base, key, values in sweeps:
        default = DEFAULT_HP[key]
        for v in values:
            if v == default:
                continue
            rec = _arch_cfg(**{**base, key: v})
            out.append(rec)
    return out


def _with_input(rec, inp):
    r = dict(rec)
    r["input"] = inp
    r["model"] = model_name_of(r)
    return r


def job_grid(
    trials,
    evals=EVALS,
    grid="default",
    assocs=ASSOCIATIONS,
    fuses=FUSIONS,
    poses=POS_MODES,
    encoders=ENCODERS,
    temporals=TEMPORALS,
    soft_norms=SOFT_NORMS,
    inputs=INPUT_MODES,
    models=None,
    fold=None,
    single_fold=False,
):
    specs = ec.keep_fold(ec.all_folds(trials, evals=evals), fold=fold, single_fold=single_fold)
    want = set(models) if models else None
    grid = grid or "default"
    inputs = tuple(inputs or INPUT_MODES)
    archs = []
    if grid in ("default", "pilot"):
        archs.extend(iter_pilot(assocs, fuses, poses, encoders, temporals, soft_norms))
    elif grid == "core":
        archs.extend(iter_core(assocs, fuses, soft_norms))
    else:
        archs.extend(iter_archs(assocs, fuses, poses, encoders, temporals, soft_norms))
        if grid == "full":
            allow = set(assocs), set(fuses), set(poses), set(encoders), set(temporals), set(soft_norms)

            def _ok(rec):
                return (
                    rec["assoc"] in allow[0]
                    and rec["fuse"] in allow[1]
                    and rec["pos"] in allow[2]
                    and rec["encoder"] in allow[3]
                    and rec["temporal"] in allow[4]
                    and rec["soft_norm"] in allow[5]
                )

            archs.extend(r for r in hp_oat_cfgs() if _ok(r))
    archs = [_with_input(r, inp) for r in archs for inp in inputs]
    configs = []
    seen = set()
    for spec in specs:
        for rec in archs:
            name = rec["model"]
            if want is not None and name not in want:
                continue
            key = (spec["eval"], spec["fold_id"], name)
            if key in seen:
                continue
            seen.add(key)
            cfg = {
                **rec,
                "eval": spec["eval"],
                "fold": int(spec["fold"]),
                "fold_id": spec["fold_id"],
            }
            cfg["model"] = name
            configs.append(cfg)
    return configs, specs


def spec_for(specs, cfg):
    for s in specs:
        if s["eval"] == cfg["eval"] and str(s["fold_id"]) == str(cfg["fold_id"]):
            return s
    raise SystemExit(f"no fold spec for {cfg}")


def run_config(cfg, trials, specs):
    spec = spec_for(specs, cfg)
    ec.bind_objective(trials, "trace")
    train_full = mv.usable_idx(trials, spec["train"])
    test_idx = mv.usable_idx(trials, spec["test"], min_bins=mv.MIN_BINS)
    train_full = ec.lfp_idx(trials, train_full)
    test_idx = ec.lfp_idx(trials, test_idx)
    if len(train_full) < 8 or len(test_idx) < 1:
        log(f"  skip {cfg}: too few trials")
        return None
    tr_idx, va_idx = mv.train_val_split(train_full, seed=mv.SEED + 17 * int(spec["fold"]))
    tr_idx, va_idx = mv.usable_idx(trials, tr_idx), mv.usable_idx(trials, va_idx)
    tr_idx, va_idx = ec.lfp_idx(trials, tr_idx), ec.lfp_idx(trials, va_idx)
    if len(tr_idx) < 4:
        log(f"  skip {cfg}: empty train after val split")
        return None
    lfp_mean, lfp_std = ec.fit_lfp_scaler(trials, tr_idx)
    zscore_lfp(trials, np.unique(np.concatenate([tr_idx, va_idx, test_idx])), lfp_mean, lfp_std)
    y_mean, y_std = mv.y_scaler_from(trials, tr_idx)
    fallback = np.unique(np.concatenate([tr_idx, va_idx, test_idx]))
    stats = fit_fuse_stats(trials, tr_idx, fallback)
    stats["lfp_mean"] = lfp_mean
    stats["lfp_std"] = lfp_std
    jdir = slurm_utils.job_dir()
    blob, hist = train_neural(cfg, trials, tr_idx, va_idx, y_mean, y_std, stats, mv.SEED + int(spec["fold"]))
    recs = predict_neural(blob, trials, test_idx, stats, cfg)
    rows, pooled = ec.score_fold(recs, trials, test_idx, "trace")
    meta = {
        **cfg,
        "target": TARGET,
        "n_train": int(len(tr_idx)),
        "n_val": int(len(va_idx)),
        "n_test": int(len(test_idx)),
        "note": spec["note"],
    }
    row = ec.write_pack(jdir, recs, rows, pooled, blob, hist, meta)
    log(
        f"  {cfg['model']:40s}  concat R²={pooled.get('fold_concat_r2', float('nan')):.3f}  "
        f"median R²={pooled.get('fold_median_r2', float('nan')):.3f}"
    )
    slurm_utils.write_status(state="done", **{k: pooled.get(k) for k in ("fold_concat_r2", "n_holdout_trials")})
    return row


def _install_term_handler():
    def _h(signum, _frame):
        try:
            slurm_utils.write_status(state="failed", signal=int(signum))
            err = slurm_utils.job_dir() / "error.txt"
            err.write_text(f"terminated by signal {signum}\n")
        except Exception:
            pass
        raise SystemExit(128 + int(signum))

    signal.signal(signal.SIGTERM, _h)
    signal.signal(signal.SIGINT, _h)


def load_trials():
    trials, _meta = mv.load_corpus()
    attach_fuse(trials)
    ec.annotate_trials(trials)
    ec.bind_objective(trials, "trace")
    return trials


def process(job=None, trials=None, **grid_kw):
    configure_out()
    mv.set_seed(mv.SEED)
    if trials is None:
        trials = load_trials()
    configs, specs = job_grid(trials, **grid_kw)
    if job is not None:
        cfg = slurm_utils.pick_config(configs, job)
        slurm_utils.set_device("auto", probe=device_probe(cfg))
        slurm_utils.begin_job(OUT, job, job_tag(cfg), **cfg)
        _install_term_handler()
        log(f"job {job}  {cfg}")
        try:
            run_config(cfg, trials, specs)
        except Exception:
            slurm_utils.write_status(state="failed")
            err = slurm_utils.job_dir() / "error.txt"
            err.write_text(traceback.format_exc())
            raise
        err = slurm_utils.job_dir() / "error.txt"
        if err.exists():
            err.unlink()
        return
    rows = []
    for i, cfg in enumerate(configs):
        slurm_utils.set_device("auto", probe=device_probe(cfg))
        slurm_utils.begin_job(OUT, i, job_tag(cfg), **cfg)
        log(f"==== job {i}/{len(configs)}  {cfg} ====")
        try:
            row = run_config(cfg, trials, specs)
            if row:
                rows.append(row)
            err = slurm_utils.job_dir() / "error.txt"
            if err.exists():
                err.unlink()
        except Exception:
            slurm_utils.write_status(state="failed")
            err = slurm_utils.job_dir() / "error.txt"
            err.write_text(traceback.format_exc())
            log(f"FAILED job {i}: {err}")
            raise
    if rows:
        pd.DataFrame(rows).to_csv(OUT / "scores.csv", index=False)


def aggregate_all():
    configure_out()
    df = slurm_utils.aggregate_scores(OUT, dest_name="scores.csv")
    log(f"aggregated {len(df)} rows -> {OUT / 'scores.csv'}")
    return df


# ---------------------------------------------------------------------------
# Dummy / self-test
# ---------------------------------------------------------------------------


def _fake_trials(n_mice=3, n_sess=2, n_per=25):
    trials = []
    rng = np.random.RandomState(0)
    for m in range(n_mice):
        ns = n_sess if m < n_mice - 1 else 1
        for s in range(ns):
            eid = f"e{m}{s}aaaaaaaa"
            n_u, n_ch = 8, 4
            xyz = np.zeros((n_u, 3), dtype=np.float32)
            xyz[:, 0] = rng.uniform(-0.0006, 0.0006, n_u)
            xyz[:, 2] = np.linspace(0.0, 0.0012, n_u)
            areas = np.array(["MOp5"] * 4 + ["MOs6a"] * 4)
            ax_u = np.linspace(0, 600, n_u).astype(np.float32)
            lat_u = np.zeros(n_u, dtype=np.float32)
            e_xyz = xyz[::2].copy()
            e_ax = ax_u[::2].copy()
            e_lat = np.zeros(n_ch, dtype=np.float32)
            for k in range(n_per):
                T = 16
                spk = rng.randn(T, n_u).astype(np.float32)
                paw = np.zeros(T, dtype=np.float32)
                on = 4 + (k % 8)
                paw[on:] = 200 + 80 * np.sin(np.linspace(0, 3, T - on))
                wheel = np.zeros(T, dtype=np.float32)
                wheel[on:] = (1 if k % 2 == 0 else -1) * 0.4
                t = {
                    "eid": eid,
                    "mouse_id": f"M{m}",
                    "trial_index": k,
                    "spikes": spk,
                    "xyz": xyz,
                    "areas": areas,
                    "unit_spikes": spk,
                    "unit_xyz": xyz,
                    "unit_areas": areas,
                    "unit_axial": ax_u,
                    "unit_lateral": lat_u,
                    "unit_quality": np.ones(n_u, dtype=np.float32),
                    "unit_to_ch": np.array([0, 0, 1, 1, 2, 2, 3, 3], dtype=np.int32),
                    "lfp": rng.randn(T, n_ch, N_LFP_BANDS).astype(np.float32) * 0.4 + 1.0,
                    "lfp_xyz": e_xyz,
                    "lfp_areas": np.array(["MOp5", "MOp5", "MOs6a", "MOs6a"]),
                    "lfp_axial": e_ax,
                    "lfp_lateral": e_lat,
                    "lfp_probe": np.array(["p0"] * n_ch),
                    "behaviors": {
                        "paw_speed": paw,
                        "wheel_signed": wheel,
                        "wheel_speed": np.abs(wheel),
                    },
                }
                stamp_geometry(t)
                trials.append(t)
    return trials


def _toy_batch(B=2, N=6, E=4, T=16, pos_dim=3):
    rng = torch.Generator().manual_seed(0)
    def rnd(*shp):
        return torch.randn(*shp, generator=rng)
    hard = torch.tensor([[0], [0], [1], [1], [2], [3]])[None].expand(B, N, 1)
    knn = torch.tensor([[0, 1], [0, 1], [1, 2], [1, 2], [2, 3], [3, 2]])[None].expand(B, N, 2)
    eknn = torch.tensor([[1, 0], [0, 2], [1, 3], [2, 1]])[None].expand(B, E, 2)
    return {
        "spikes": rnd(B, N, T).abs(),
        "lfp": rnd(B, E, T, N_LFP_BANDS),
        "y": rnd(B, T, 1),
        "mask_t": torch.ones(B, T, dtype=torch.bool),
        "mask_loss": torch.ones(B, T, dtype=torch.bool),
        "mask_n": torch.ones(B, N, dtype=torch.bool),
        "mask_e": torch.ones(B, E, dtype=torch.bool),
        "unit_xyz": rnd(B, N, 3),
        "elec_xyz": rnd(B, E, 3),
        "unit_area": torch.ones(B, N, dtype=torch.long),
        "elec_area": torch.ones(B, E, dtype=torch.long),
        "rate": rnd(B, N).abs(),
        "quality": torch.ones(B, N),
        "pos": rnd(B, E, pos_dim) if pos_dim else torch.zeros(B, E, 0),
        "unit_knn": knn,
        "unit_knn_valid": torch.ones(B, N, 2, dtype=torch.bool),
        "unit_w": torch.ones(B, N, 2) / 2,
        "hard_idx": hard,
        "hard_valid": torch.ones(B, N, 1, dtype=torch.bool),
        "elec_knn": eknn,
        "elec_knn_valid": torch.ones(B, E, 2, dtype=torch.bool),
        "elec_knn_dist": torch.ones(B, E, 2),
        "lengths": torch.full((B,), T, dtype=torch.long),
    }


def run_self_test():
    global MAX_EPOCHS_CAP
    mv.set_seed(0)
    slurm_utils.set_device("cpu", probe="transformer")
    cfgs = [
        _arch_cfg("hard", "concat", "atlas", "instant", "transformer"),
        _arch_cfg("hard", "gate", "none", "instant", "gru"),
        _arch_cfg("hard", "residual", "probe", "history", "transformer"),
        _arch_cfg("soft", "xattn", "depth", "bandconv", "transformer"),
        _arch_cfg("soft", "concat", "local", "instant", "transformer", "unnorm"),
        _arch_cfg("graph", "private", "atlas", "instant", "kalman"),
        _arch_cfg("graph", "concat", "atlas", "instant", "recurrent"),
        _arch_cfg("hard", "concat", "atlas", "instant", "cde"),
        _arch_cfg("hard", "concat", "atlas", "instant", "recurrent", rnn_cell="lstm"),
        _arch_cfg("hard", "concat", "atlas", "instant", "transformer", pool="mean"),
        _arch_cfg("hard", "concat", "atlas", "instant", "transformer", pool="attn"),
        _arch_cfg("hard", "concat", "atlas", "instant", "transformer", d_model=32, n_layers=2),
        _with_input(_arch_cfg("hard", "concat", "atlas", "instant", "transformer"), "spikes"),
        _with_input(_arch_cfg("hard", "concat", "atlas", "instant", "transformer"), "lfp"),
        _with_input(_arch_cfg("hard", "concat", "atlas", "instant", "transformer"), "lfp_nogamma"),
        _with_input(_arch_cfg("hard", "concat", "atlas", "instant", "transformer"), "spike_lfp_nogamma"),
    ]
    for cfg in cfgs:
        model = FuseSequence(4, cfg)
        batch = _toy_batch(pos_dim=POS_DIM[cfg["pos"]])
        pred, aux = model(batch)
        assert pred.shape[-1] == 1 and pred.dim() == 3, pred.shape
        loss = pred.pow(2).mean()
        extra = aux_total(aux, cfg)
        if extra is not None:
            loss = loss + extra
        loss.backward()
        log(f"self-test ok {cfg['model']}  shape={tuple(pred.shape)}")
    global DUMMY, MAX_EPOCHS_CAP
    DUMMY = True
    MAX_EPOCHS_CAP = 1
    configure_out()
    trials = _fake_trials()
    ec.annotate_trials(trials)
    ec.bind_objective(trials, "trace")
    all_cfgs, _ = job_grid(trials, evals=EVALS, grid="default", single_fold=True)
    n_models = len({c["model"] for c in all_cfgs})
    log(f"default single-fold grid: {len(all_cfgs)} jobs, {n_models} unique models")
    if not (80 <= n_models <= 400):
        raise AssertionError(f"expected a compact pilot grid, got {n_models} models")
    configs, specs = job_grid(trials, evals=("LTO",), grid="core", single_fold=True, assocs=("hard",), fuses=("concat",))
    slurm_utils.begin_job(OUT, 0, "selftest", **configs[0])
    run_config(configs[0], trials, specs)
    log("self-test dummy train ok")


def run_dummy(device="cpu"):
    global DUMMY, MAX_EPOCHS_CAP
    DUMMY = True
    MAX_EPOCHS_CAP = 2
    configure_out()
    mv.set_seed(mv.SEED)
    slurm_utils.set_device(device, probe="transformer")
    trials = _fake_trials()
    ec.annotate_trials(trials)
    ec.bind_objective(trials, "trace")
    process(trials=trials, evals=EVALS, grid="core", single_fold=True, assocs=("hard", "soft"), fuses=("concat", "gate"))


def main():
    p = argparse.ArgumentParser(description="Spike–LFP fusion LTO/LSO/LMO pipeline")
    slurm_utils.add_common_args(p)
    p.add_argument("--self-test", action="store_true")
    p.add_argument("--dummy", action="store_true")
    p.add_argument(
        "--single-fold",
        action="store_true",
        help=(
            "One representative fold per eval: LTO 25–30% block, LSO session "
            f"{ec.SINGLE_LSO_FOLD_ID}, LMO mouse {ec.SINGLE_LMO_FOLD_ID}."
        ),
    )
    p.add_argument("--fold", type=int, default=None)
    p.add_argument("--eval", nargs="*", default=list(EVALS), choices=list(EVALS))
    p.add_argument("--grid", default="default", choices=list(GRIDS))
    p.add_argument("--assoc", nargs="*", default=list(ASSOCIATIONS), choices=list(ASSOCIATIONS))
    p.add_argument("--fuse", nargs="*", default=list(FUSIONS), choices=list(FUSIONS))
    p.add_argument("--pos", nargs="*", default=list(POS_MODES), choices=list(POS_MODES))
    p.add_argument("--encoder", nargs="*", default=list(ENCODERS), choices=list(ENCODERS))
    p.add_argument("--temporal", nargs="*", default=list(TEMPORALS), choices=list(TEMPORALS))
    p.add_argument("--soft-norm", nargs="*", default=list(SOFT_NORMS), choices=list(SOFT_NORMS))
    p.add_argument(
        "--inputs",
        nargs="*",
        default=list(INPUT_MODES),
        choices=list(INPUT_MODES),
        help="spikes | lfp | lfp_nogamma | spike_lfp | spike_lfp_nogamma",
    )
    p.add_argument("--models", nargs="*", default=None, help="Subset of composed model names")
    args = p.parse_args()
    if args.self_test:
        run_self_test()
        return
    if args.dummy:
        run_dummy(device=args.device)
        return
    configure_out()
    grid_kw = dict(
        evals=tuple(args.eval),
        grid=args.grid,
        assocs=tuple(args.assoc),
        fuses=tuple(args.fuse),
        poses=tuple(args.pos),
        encoders=tuple(args.encoder),
        temporals=tuple(args.temporal),
        soft_norms=tuple(args.soft_norm),
        inputs=tuple(args.inputs),
        models=args.models,
        fold=args.fold,
        single_fold=args.single_fold,
    )
    if args.list_jobs:
        if DUMMY:
            trials = _fake_trials()
        else:
            trials, _ = mv.load_corpus()
        ec.annotate_trials(trials)
        ec.bind_objective(trials, "trace")
        configs, _ = job_grid(trials, **grid_kw)
        slurm_utils.print_jobs(configs)
        return
    if args.aggregate:
        aggregate_all()
        return
    job = slurm_utils.resolve_job_index(args.job)
    slurm_utils.set_device(args.device, probe="transformer")
    process(job=job, **grid_kw)


if __name__ == "__main__":
    main()
