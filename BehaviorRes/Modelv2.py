"""Formulation B voxel encoder on ModelDataRightContra (motor cortex only).

Two anatomical voxelizers, trained independently:

  aniso3d      Strategy 2: anisotropic CCF grid (400 µm ML/AP, 100 µm DV)
               plus a 3-D CNN spatial encoder (or a flat control).
  layer_tiles  Strategy 3: Allen MOp/MOs × layer × 400 µm ML/AP tiles
               plus an MLP spatial encoder (or a flat control).

Same holdouts, trial-balanced MSE, and GRU decoder as Modelv1.
Also writes occupancy / overlap diagnostics for the voxelizations.
"""

from __future__ import annotations

import argparse
import gc
import json
import math
import pickle
import random
import warnings
from datetime import datetime, timezone
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import mean_absolute_error, r2_score
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from torch.utils.data import DataLoader, Dataset

import slurm_utils
import Modelv1 as mv1

ROOT = Path(__file__).resolve().parent
DESKTOP = ROOT.parent
DATA = ROOT / "ModelDataRightContra"
SESS = DATA / "sessions"
OUT = ROOT / "Modelv2"
CACHE = OUT / "cache"
MODEL_ROOT = ROOT / "Modelv2"

BINSIZE = 0.02
MIN_BINS = 13
MIN_UNITS = 5
MAX_BINS = 128
VAL_FRAC = 0.12
BATCH = 32
MAX_EPOCHS = 40
PATIENCE = 5
LR = 1e-3
WEIGHT_DECAY = 1e-4
SEED = 0
D_MODEL = 64
RNN_HIDDEN = 64
DROPOUT = 0.1
VOXEL_DROPOUT = 0.15

ML_STEP = 400e-6
AP_STEP = 400e-6
DV_STEP = 100e-6
LAYERS = ("1", "2/3", "5", "6a", "6b")
REGIONS = ("MOp", "MOs")

STRATEGIES = ("aniso3d", "layer_tiles")
ENCODERS = ("spatial", "flat")
TASKS = ("trial_holdout", "session_holdout", "mouse_holdout")
TARGETS = mv1.TARGETS
TARGET_LABELS = mv1.TARGET_LABELS
TARGET_YLABELS = mv1.TARGET_YLABELS

HOLD_MOUSE = "ZM_2241"
HOLD_SESSION_MOUSE = "CSH_ZAD_026"
HOLD_SESSION_EID = "626126d5-eecf-4e9b-900e-ec29a17ece07"


def log(msg):
    print(f"[{datetime.now().strftime('%H:%M:%S')}] {msg}", flush=True)


def set_seed(seed=SEED):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def device_of():
    return slurm_utils.device_of()


def job_grid():
    return [
        {"target": beh, "task": t, "strategy": s, "encoder": e}
        for beh in TARGETS
        for t in TASKS
        for s in STRATEGIES
        for e in ENCODERS
    ]


def configure_out(target):
    global OUT, CACHE
    if target not in TARGETS:
        raise SystemExit(f"unknown target {target!r}")
    OUT = MODEL_ROOT / target
    CACHE = OUT / "cache"
    return OUT


def motor_mask(units):
    area = units["brain_area"].astype(str)
    return area.str.startswith("MOp") | area.str.startswith("MOs")


def load_corpus():
    return mv1.load_corpus()


def parse_area(area):
    a = str(area)
    region = "MOp" if a.startswith("MOp") else "MOs"
    layer = "5"
    for L in ("6b", "6a", "2/3", "5", "1"):
        if a.endswith(L):
            layer = L
            break
    return region, layer


def make_splits(trials):
    df = pd.DataFrame(
        [
            {
                "i": i,
                "eid": t["eid"],
                "mouse_id": t["mouse_id"],
                "trial_index": t["trial_index"],
            }
            for i, t in enumerate(trials)
        ]
    )
    rng = np.random.default_rng(SEED)
    eids = np.array(sorted(df.eid.unique()))
    n_hold_sess = max(1, int(round(0.10 * len(eids))))
    hold_eids = set(rng.choice(eids, size=n_hold_sess, replace=False).tolist())

    trial_test = []
    for eid in sorted(hold_eids):
        idx = df.index[df.eid == eid].to_numpy()
        n_te = max(1, int(round(0.10 * len(idx))))
        pick = rng.choice(idx, size=n_te, replace=False)
        trial_test.extend(df.loc[pick, "i"].tolist())
    trial_test = np.array(sorted(set(trial_test)), dtype=int)
    trial_train = np.array(sorted(set(df.i) - set(trial_test)), dtype=int)

    sess_test = df.loc[df.eid == HOLD_SESSION_EID, "i"].to_numpy()
    sess_train = df.loc[df.eid != HOLD_SESSION_EID, "i"].to_numpy()
    same_mouse_in_train = df.loc[
        (df.mouse_id == HOLD_SESSION_MOUSE) & (df.eid != HOLD_SESSION_EID), "i"
    ]
    if len(same_mouse_in_train) == 0:
        raise RuntimeError("session_holdout mouse has no other session in train")

    mouse_test = df.loc[df.mouse_id == HOLD_MOUSE, "i"].to_numpy()
    mouse_train = df.loc[df.mouse_id != HOLD_MOUSE, "i"].to_numpy()
    if df.loc[df.mouse_id == HOLD_MOUSE, "eid"].nunique() != 1:
        raise RuntimeError(f"{HOLD_MOUSE} is not a single-session mouse")

    splits = {
        "trial_holdout": {
            "train": trial_train,
            "test": trial_test,
            "note": (
                f"{n_hold_sess}/{len(eids)} sessions contribute held-out trials "
                f"({', '.join(e[:8] for e in sorted(hold_eids))})"
            ),
            "hold_eids": sorted(hold_eids),
        },
        "session_holdout": {
            "train": sess_train,
            "test": sess_test,
            "note": (
                f"held-out session {HOLD_SESSION_EID[:8]} of {HOLD_SESSION_MOUSE}; "
                f"{same_mouse_in_train.size} trials from other sessions of that mouse remain in train"
            ),
            "hold_eids": [HOLD_SESSION_EID],
        },
        "mouse_holdout": {
            "train": mouse_train,
            "test": mouse_test,
            "note": f"held-out mouse {HOLD_MOUSE} (one session, unseen animal)",
            "hold_eids": sorted(df.loc[df.mouse_id == HOLD_MOUSE, "eid"].unique()),
        },
    }
    for name, sp in splits.items():
        log(f"split {name}: train={len(sp['train'])}  test={len(sp['test'])}  {sp['note']}")
    return splits


def train_val_split(train_idx, seed=SEED):
    rng = np.random.default_rng(seed)
    train_idx = np.asarray(train_idx)
    rng.shuffle(train_idx)
    n_val = max(1, int(round(VAL_FRAC * len(train_idx))))
    return train_idx[n_val:], train_idx[:n_val]


def y_scaler_from(trials, idx):
    return mv1.y_scaler_from(trials, idx)


def session_records(trials):
    recs = {}
    for t in trials:
        if t["eid"] in recs:
            continue
        recs[t["eid"]] = {
            "eid": t["eid"],
            "mouse_id": t["mouse_id"],
            "xyz": t["xyz"],
            "areas": t["areas"],
        }
    return recs


def snap_origin(vals, step):
    return float(np.floor(np.min(vals) / step) * step)


def grid_axes(vals, step):
    lo = snap_origin(vals, step)
    n = int(np.floor((np.max(vals) - lo) / step)) + 1
    return lo, n


class AnisoGrid:
    """Strategy 2: 400 µm ML × 400 µm AP × 100 µm DV over the motor hull."""

    name = "aniso3d"

    def __init__(self, xyz):
        xyz = np.asarray(xyz, dtype=np.float64)
        self.x0, self.nx = grid_axes(xyz[:, 0], ML_STEP)
        self.y0, self.ny = grid_axes(xyz[:, 1], AP_STEP)
        self.z0, self.nz = grid_axes(xyz[:, 2], DV_STEP)
        self.shape = (self.nx, self.ny, self.nz)
        self.n_voxels = int(np.prod(self.shape))
        log(
            f"aniso3d grid {self.nx}×{self.ny}×{self.nz} = {self.n_voxels} "
            f"({ML_STEP*1e6:.0f}/{AP_STEP*1e6:.0f}/{DV_STEP*1e6:.0f} µm)"
        )

    def index_xyz(self, xyz):
        xyz = np.asarray(xyz, dtype=np.float64)
        ix = np.floor((xyz[:, 0] - self.x0) / ML_STEP).astype(int)
        iy = np.floor((xyz[:, 1] - self.y0) / AP_STEP).astype(int)
        iz = np.floor((xyz[:, 2] - self.z0) / DV_STEP).astype(int)
        ix = np.clip(ix, 0, self.nx - 1)
        iy = np.clip(iy, 0, self.ny - 1)
        iz = np.clip(iz, 0, self.nz - 1)
        flat = np.ravel_multi_index((ix, iy, iz), self.shape)
        return flat, np.stack([ix, iy, iz], axis=1)

    def voxelize(self, spikes, xyz, areas=None):
        T, N = spikes.shape
        flat, _ = self.index_xyz(xyz)
        G = np.zeros((T, self.n_voxels), dtype=np.float32)
        C = np.zeros(self.n_voxels, dtype=np.float32)
        for v in np.unique(flat):
            sel = flat == v
            C[v] = float(sel.sum())
            G[:, v] = spikes[:, sel].mean(axis=1)
        M = (C > 0).astype(np.float32)
        return G, M, C

    def pack_grid(self, G, M, C):
        T = G.shape[0]
        g = G.reshape(T, self.nx, self.ny, self.nz)
        m = np.broadcast_to(M.reshape(self.nx, self.ny, self.nz), g.shape)
        c = np.broadcast_to(np.log1p(C).reshape(self.nx, self.ny, self.nz), g.shape)
        return np.stack([g, m, c], axis=1).astype(np.float32)  # T, 3, X, Y, Z


class LayerTiles:
    """Strategy 3: (MOp|MOs) × Allen layer × 400 µm ML/AP tiles."""

    name = "layer_tiles"

    def __init__(self, xyz, areas):
        xyz = np.asarray(xyz, dtype=np.float64)
        self.x0, self.nx = grid_axes(xyz[:, 0], ML_STEP)
        self.y0, self.ny = grid_axes(xyz[:, 1], AP_STEP)
        self.layer_i = {L: i for i, L in enumerate(LAYERS)}
        self.region_i = {R: i for i, R in enumerate(REGIONS)}
        self.n_voxels = len(REGIONS) * len(LAYERS) * self.nx * self.ny
        log(
            f"layer_tiles {len(REGIONS)}×{len(LAYERS)}×{self.nx}×{self.ny} = {self.n_voxels} "
            f"(400 µm ML/AP)"
        )

    def _flat(self, region, layer, ix, iy):
        ri = self.region_i[region]
        li = self.layer_i[layer]
        return ((ri * len(LAYERS) + li) * self.nx + ix) * self.ny + iy

    def index_units(self, xyz, areas):
        xyz = np.asarray(xyz, dtype=np.float64)
        ix = np.clip(np.floor((xyz[:, 0] - self.x0) / ML_STEP).astype(int), 0, self.nx - 1)
        iy = np.clip(np.floor((xyz[:, 1] - self.y0) / AP_STEP).astype(int), 0, self.ny - 1)
        flat = np.empty(len(areas), dtype=int)
        keys = []
        for i, a in enumerate(areas):
            region, layer = parse_area(a)
            flat[i] = self._flat(region, layer, int(ix[i]), int(iy[i]))
            keys.append((region, layer, int(ix[i]), int(iy[i])))
        return flat, keys

    def voxelize(self, spikes, xyz, areas):
        T, N = spikes.shape
        flat, _ = self.index_units(xyz, areas)
        G = np.zeros((T, self.n_voxels), dtype=np.float32)
        C = np.zeros(self.n_voxels, dtype=np.float32)
        for v in np.unique(flat):
            sel = flat == v
            C[v] = float(sel.sum())
            G[:, v] = spikes[:, sel].mean(axis=1)
        M = (C > 0).astype(np.float32)
        return G, M, C


def jaccard(a, b):
    a, b = set(a), set(b)
    if not a and not b:
        return 1.0
    if not a or not b:
        return 0.0
    return len(a & b) / len(a | b)


def occupied_ids(vox, rec):
    if vox.name == "aniso3d":
        flat, _ = vox.index_xyz(rec["xyz"])
    else:
        flat, _ = vox.index_units(rec["xyz"], rec["areas"])
    return np.unique(flat)


def pairwise_mean_jaccard(sets):
    keys = list(sets)
    if len(keys) < 2:
        return float("nan")
    vals = []
    for i in range(len(keys)):
        for j in range(i + 1, len(keys)):
            vals.append(jaccard(sets[keys[i]], sets[keys[j]]))
    return float(np.mean(vals))


def analyze_voxels(vox, sessions, splits, trials, out_dir):
    out_dir.mkdir(parents=True, exist_ok=True)
    occ = {eid: occupied_ids(vox, rec) for eid, rec in sessions.items()}
    mouse_of = {eid: rec["mouse_id"] for eid, rec in sessions.items()}
    mice = {}
    for eid, vids in occ.items():
        mice.setdefault(mouse_of[eid], set()).update(vids.tolist())

    global_count = np.zeros(vox.n_voxels, dtype=np.int32)
    sess_count = np.zeros(vox.n_voxels, dtype=np.int32)
    unit_mult = []
    assigned = 0
    total_u = 0
    for rec in sessions.values():
        if vox.name == "aniso3d":
            flat, _ = vox.index_xyz(rec["xyz"])
        else:
            flat, _ = vox.index_units(rec["xyz"], rec["areas"])
        total_u += len(flat)
        assigned += int(((flat >= 0) & (flat < vox.n_voxels)).sum())
        for v, c in zip(*np.unique(flat, return_counts=True)):
            global_count[v] += int(c)
            unit_mult.append(int(c))
        sess_count[np.unique(flat)] += 1

    occupied = int((global_count > 0).sum())
    singleton_sess = int(((sess_count == 1) & (global_count > 0)).sum())
    shared = int((sess_count >= 2).sum())
    n_sess = len(sessions)
    n_mice = len(mice)

    within_mouse = []
    by_mouse_eids = {}
    for eid, m in mouse_of.items():
        by_mouse_eids.setdefault(m, []).append(eid)
    for m, eids in by_mouse_eids.items():
        if len(eids) < 2:
            continue
        within_mouse.append(pairwise_mean_jaccard({e: occ[e] for e in eids}))

    between_mouse = pairwise_mean_jaccard(mice)
    between_sess = pairwise_mean_jaccard(occ)

    split_overlap = {}
    for task, sp in splits.items():
        tr_eids = {trials[i]["eid"] for i in sp["train"]}
        te_eids = {trials[i]["eid"] for i in sp["test"]}
        tr_v = set().union(*[set(occ[e].tolist()) for e in tr_eids if e in occ])
        te_v = set().union(*[set(occ[e].tolist()) for e in te_eids if e in occ])
        split_overlap[task] = {
            "train_voxels": len(tr_v),
            "test_voxels": len(te_v),
            "jaccard": jaccard(tr_v, te_v),
            "test_covered": (len(te_v & tr_v) / len(te_v)) if te_v else float("nan"),
        }

    per_sess = []
    for eid, rec in sessions.items():
        vids = occ[eid]
        per_sess.append(
            {
                "eid": eid,
                "mouse_id": rec["mouse_id"],
                "n_units": int(len(rec["xyz"])),
                "n_voxels": int(len(vids)),
                "units_per_voxel": float(len(rec["xyz"]) / max(len(vids), 1)),
                "frac_of_occupied_global": float(len(vids) / max(occupied, 1)),
            }
        )
    per_mouse = []
    for m, vids in mice.items():
        n_u = int(sum(len(sessions[e]["xyz"]) for e, mm in mouse_of.items() if mm == m))
        per_mouse.append(
            {
                "mouse_id": m,
                "n_sessions": int(sum(1 for mm in mouse_of.values() if mm == m)),
                "n_units": n_u,
                "n_voxels": int(len(vids)),
                "frac_of_occupied_global": float(len(vids) / max(occupied, 1)),
            }
        )

    summary = {
        "strategy": vox.name,
        "n_voxels_catalog": vox.n_voxels,
        "n_voxels_occupied": occupied,
        "occupancy_fraction": occupied / max(vox.n_voxels, 1),
        "n_sessions": n_sess,
        "n_mice": n_mice,
        "units_assigned": int(total_u),
        "assignment_rate": float(assigned / max(total_u, 1)),
        "mean_units_per_occupied_voxel_session": float(np.mean(unit_mult)) if unit_mult else 0.0,
        "median_units_per_occupied_voxel_session": float(np.median(unit_mult)) if unit_mult else 0.0,
        "frac_occupied_voxels_unique_to_one_session": singleton_sess / max(occupied, 1),
        "n_voxels_shared_by_ge2_sessions": shared,
        "mean_jaccard_between_sessions": between_sess,
        "mean_jaccard_within_multi_session_mice": float(np.mean(within_mouse))
        if within_mouse
        else float("nan"),
        "mean_jaccard_between_mice": between_mouse,
        "median_session_voxels": float(np.median([p["n_voxels"] for p in per_sess])),
        "median_mouse_voxels": float(np.median([p["n_voxels"] for p in per_mouse])),
        "split_overlap": split_overlap,
        "grid": (
            {
                "nx": vox.nx,
                "ny": vox.ny,
                "nz": getattr(vox, "nz", None),
                "ml_um": ML_STEP * 1e6,
                "ap_um": AP_STEP * 1e6,
                "dv_um": DV_STEP * 1e6 if vox.name == "aniso3d" else None,
            }
        ),
    }
    (out_dir / f"{vox.name}_summary.json").write_text(json.dumps(summary, indent=2, default=float))
    pd.DataFrame(per_sess).to_csv(out_dir / f"{vox.name}_by_session.csv", index=False)
    pd.DataFrame(per_mouse).sort_values("n_voxels", ascending=False).to_csv(
        out_dir / f"{vox.name}_by_mouse.csv", index=False
    )
    log(
        f"{vox.name}: occupied {occupied}/{vox.n_voxels}  "
        f"sess Jaccard {between_sess:.3f}  mouse Jaccard {between_mouse:.3f}  "
        f"singleton voxels {singleton_sess/max(occupied,1):.2f}"
    )
    _plot_voxel_quality(vox, sessions, occ, global_count, sess_count, summary, per_sess, per_mouse, out_dir)
    return summary


def style_plots():
    plt.rcParams.update(
        {
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "axes.spines.top": False,
            "axes.spines.right": False,
            "font.size": 11,
        }
    )


def _plot_voxel_quality(vox, sessions, occ, global_count, sess_count, summary, per_sess, per_mouse, out_dir):
    style_plots()
    if vox.name == "aniso3d":
        gc = global_count.reshape(vox.nx, vox.ny, vox.nz)
        sc = sess_count.reshape(vox.nx, vox.ny, vox.nz)
        fig, axes = plt.subplots(2, 3, figsize=(11.2, 6.6))
        views = [
            (gc.sum(axis=1).T, "ML–DV units (sum over AP)", "ML bin", "DV bin"),
            (gc.sum(axis=0).T, "AP–DV units (sum over ML)", "AP bin", "DV bin"),
            (gc.sum(axis=2).T, "ML–AP units (sum over DV)", "ML bin", "AP bin"),
            (sc.max(axis=1).T, "ML–DV sessions (max over AP)", "ML bin", "DV bin"),
            (sc.max(axis=0).T, "AP–DV sessions (max over ML)", "AP bin", "DV bin"),
            (sc.max(axis=2).T, "ML–AP sessions (max over DV)", "ML bin", "AP bin"),
        ]
        for ax, (img, title, xl, yl) in zip(axes.ravel(), views):
            im = ax.imshow(img, origin="lower", aspect="auto", cmap="magma")
            ax.set_title(title)
            ax.set_xlabel(xl)
            ax.set_ylabel(yl)
            fig.colorbar(im, ax=ax, fraction=0.046)
        fig.suptitle("Strategy 2 anisotropic grid · aggregate occupancy")
        fig.tight_layout()
        fig.savefig(out_dir / "aniso3d_aggregate_projections.png", dpi=150)
        plt.close(fig)
    else:
        counts = np.zeros((len(REGIONS), len(LAYERS)), dtype=int)
        sess_l = np.zeros((len(REGIONS), len(LAYERS)), dtype=int)
        tile = np.zeros((len(REGIONS), len(LAYERS), vox.nx, vox.ny), dtype=int)
        for rec in sessions.values():
            _, keys = vox.index_units(rec["xyz"], rec["areas"])
            seen = set()
            for region, layer, ix, iy in keys:
                ri, li = REGIONS.index(region), LAYERS.index(layer)
                counts[ri, li] += 1
                tile[ri, li, ix, iy] += 1
                seen.add((ri, li))
            for ri, li in seen:
                sess_l[ri, li] += 1
        fig, axes = plt.subplots(2, 2, figsize=(9.6, 7.4))
        im0 = axes[0, 0].imshow(counts, origin="upper", cmap="magma")
        axes[0, 0].set_xticks(range(len(LAYERS)), LAYERS)
        axes[0, 0].set_yticks(range(len(REGIONS)), REGIONS)
        axes[0, 0].set_title("Units by region × layer (all sessions)")
        fig.colorbar(im0, ax=axes[0, 0], fraction=0.046)
        im1 = axes[0, 1].imshow(sess_l, origin="upper", cmap="viridis")
        axes[0, 1].set_xticks(range(len(LAYERS)), LAYERS)
        axes[0, 1].set_yticks(range(len(REGIONS)), REGIONS)
        axes[0, 1].set_title("Sessions covering each region × layer")
        fig.colorbar(im1, ax=axes[0, 1], fraction=0.046)
        mlap_units = tile.sum(axis=(0, 1)).T
        mlap_sess = (tile > 0).sum(axis=(0, 1)).T
        im2 = axes[1, 0].imshow(mlap_units, origin="lower", cmap="magma", aspect="auto")
        axes[1, 0].set_title("ML–AP units (sum over region/layer)")
        axes[1, 0].set_xlabel("ML tile")
        axes[1, 0].set_ylabel("AP tile")
        fig.colorbar(im2, ax=axes[1, 0], fraction=0.046)
        im3 = axes[1, 1].imshow(mlap_sess, origin="lower", cmap="viridis", aspect="auto")
        axes[1, 1].set_title("ML–AP sessions (any layer)")
        axes[1, 1].set_xlabel("ML tile")
        axes[1, 1].set_ylabel("AP tile")
        fig.colorbar(im3, ax=axes[1, 1], fraction=0.046)
        fig.suptitle("Strategy 3 layer tiles · aggregate occupancy")
        fig.tight_layout()
        fig.savefig(out_dir / "layer_tiles_aggregate.png", dpi=150)
        plt.close(fig)

    fig, axes = plt.subplots(1, 3, figsize=(11.0, 3.6))
    axes[0].hist([p["n_voxels"] for p in per_sess], bins=15, color="#4C6A92")
    axes[0].set_title("Voxels per session")
    axes[0].set_xlabel("occupied voxels")
    axes[1].hist([p["n_voxels"] for p in per_mouse], bins=12, color="#C47B3B")
    axes[1].set_title("Voxels per mouse")
    axes[1].set_xlabel("occupied voxels")
    axes[2].bar(
        ["between\nsessions", "within\nmouse", "between\nmice"],
        [
            summary["mean_jaccard_between_sessions"],
            summary["mean_jaccard_within_multi_session_mice"]
            if np.isfinite(summary["mean_jaccard_within_multi_session_mice"])
            else 0.0,
            summary["mean_jaccard_between_mice"],
        ],
        color=["#4C6A92", "#3B7A57", "#B4554A"],
    )
    axes[2].set_ylim(0, 1)
    axes[2].set_ylabel("mean Jaccard")
    axes[2].set_title("Occupancy overlap")
    fig.suptitle(f"{vox.name} · coverage and correspondence")
    fig.tight_layout()
    fig.savefig(out_dir / f"{vox.name}_overlap.png", dpi=150)
    plt.close(fig)

    eids = sorted(occ, key=lambda e: sessions[e]["mouse_id"])
    n = len(eids)
    mat = np.zeros((n, n), dtype=float)
    for i, a in enumerate(eids):
        for j, b in enumerate(eids):
            mat[i, j] = jaccard(occ[a], occ[b])
    fig, ax = plt.subplots(figsize=(7.2, 6.4))
    im = ax.imshow(mat, vmin=0, vmax=1, cmap="cividis")
    ax.set_title(f"{vox.name} session–session occupancy Jaccard")
    ax.set_xlabel("session (grouped by mouse)")
    ax.set_ylabel("session")
    fig.colorbar(im, ax=ax, fraction=0.046, label="Jaccard")
    fig.tight_layout()
    fig.savefig(out_dir / f"{vox.name}_session_jaccard.png", dpi=150)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(5.2, 3.6))
    ax.hist([c for c in global_count if c > 0], bins=20, color="#4C6A92")
    ax.set_xlabel("units assigned (all sessions stacked)")
    ax.set_ylabel("occupied voxels")
    ax.set_title(f"{vox.name} · units per occupied voxel")
    fig.tight_layout()
    fig.savefig(out_dir / f"{vox.name}_units_per_voxel.png", dpi=150)
    plt.close(fig)


def plot_voxel_comparison(summaries, out_dir):
    style_plots()
    names = [s for s in STRATEGIES if s in summaries]
    fig, axes = plt.subplots(1, 3, figsize=(11.2, 3.7))
    colors = {"aniso3d": "#4C6A92", "layer_tiles": "#C47B3B"}
    x = np.arange(len(names))
    axes[0].bar(
        x,
        [100 * summaries[n]["occupancy_fraction"] for n in names],
        color=[colors[n] for n in names],
    )
    axes[0].set_xticks(x, names)
    axes[0].set_ylabel("% of catalog occupied")
    axes[0].set_title("Catalog occupancy")
    axes[1].bar(
        x,
        [100 * summaries[n]["frac_occupied_voxels_unique_to_one_session"] for n in names],
        color=[colors[n] for n in names],
    )
    axes[1].set_xticks(x, names)
    axes[1].set_ylabel("% occupied voxels")
    axes[1].set_title("Unique to one session")
    width = 0.35
    metrics = [
        "mean_jaccard_between_sessions",
        "mean_jaccard_within_multi_session_mice",
        "mean_jaccard_between_mice",
    ]
    labels = ["sessions", "within mouse", "mice"]
    for i, n in enumerate(names):
        axes[2].bar(
            np.arange(3) + (i - 0.5) * width,
            [summaries[n][m] for m in metrics],
            width,
            color=colors[n],
            label=n,
        )
    axes[2].set_xticks(np.arange(3), labels)
    axes[2].set_ylim(0, 1)
    axes[2].set_ylabel("mean Jaccard")
    axes[2].set_title("Correspondence overlap")
    axes[2].legend(frameon=False, fontsize=8)
    fig.suptitle("Voxel strategy comparison")
    fig.tight_layout()
    fig.savefig(out_dir / "strategy_comparison.png", dpi=150)
    plt.close(fig)


def apply_voxel_dropout(grid_or_flat, mask, p=VOXEL_DROPOUT):
    """Zero a random subset of occupied voxels on the feature channels."""
    if p <= 0:
        return grid_or_flat
    drop = (torch.rand(mask.shape, device=mask.device) < p) & (mask > 0)
    out = grid_or_flat.clone()
    if out.ndim == 5:
        # B, C, X, Y, Z  (single time) — unused
        out = out.masked_fill(drop, 0)
    else:
        out = out.masked_fill(drop.unsqueeze(-1), 0)
    return out


class Head(nn.Module):
    def __init__(self, d, p=DROPOUT, n_out=mv1.N_OUT):
        super().__init__()
        self.net = nn.Sequential(nn.Dropout(p), nn.Linear(d, d), nn.GELU(), nn.Linear(d, n_out))

    def forward(self, h):
        return self.net(h)


def reduced_shape(nx, ny, nz):
    nx, ny, nz = nx // 2, ny // 2, nz // 4
    nx, ny, nz = nx // 2, ny // 2, nz // 2
    return nx, ny, nz


def _reduce3d(h):
    """MPS-safe mean-pool (no aten::avg_pool3d). 7×10×37 → 1×2×4."""
    n, c, x, y, z = h.shape
    x2, y2, z2 = (x // 2) * 2, (y // 2) * 2, (z // 4) * 4
    h = h[:, :, :x2, :y2, :z2]
    h = h.reshape(n, c, x2 // 2, 2, y2 // 2, 2, z2 // 4, 4).mean(dim=(3, 5, 7))
    n, c, x, y, z = h.shape
    x2, y2, z2 = (x // 2) * 2, (y // 2) * 2, (z // 2) * 2
    h = h[:, :, :x2, :y2, :z2]
    return h.reshape(n, c, x2 // 2, 2, y2 // 2, 2, z2 // 2, 2).mean(dim=(3, 5, 7))


class VoxelCNN3D(nn.Module):
    """3-D CNN on the (ML×AP column, DV) unfolding of the anisotropic grid.

    Shared Conv2d: 3 columns × 5 depth bins (1.2 mm ML/AP × 0.5 mm DV).
    Occupancy-weighted pool so the 86% empty voxels do not wash out G.
    MPS has no Conv3d; this is the translation-equivariant substitute.
    """

    def __init__(self, grid_shape, d=D_MODEL):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=(3, 5), padding=(1, 2)),
            nn.GELU(),
            nn.Conv2d(16, 32, kernel_size=(3, 5), padding=(1, 2)),
            nn.GELU(),
        )
        self.proj = nn.Linear(32, d)

    def forward(self, grid):
        B, T, C, X, Y, Z = grid.shape
        n = B * T
        h = grid.reshape(n, C, X * Y, Z)
        occ = h[:, 1:2]
        h = self.net(h)
        w = occ.clamp(min=0)
        h = (h * w).sum(dim=(2, 3)) / w.sum(dim=(2, 3)).clamp(min=1e-6)
        return self.proj(h).reshape(B, T, D_MODEL)


class VoxelFlat3D(nn.Module):
    """Control B1: same grid, pool+linear, no learned convolution."""

    def __init__(self, grid_shape, d=D_MODEL):
        super().__init__()
        rx, ry, rz = reduced_shape(*grid_shape)
        self.proj = nn.Linear(3 * rx * ry * rz, d)

    def forward(self, grid):
        B, T, C, X, Y, Z = grid.shape
        h = _reduce3d(grid.reshape(B * T, C, X, Y, Z)).reshape(B * T, -1)
        return self.proj(h).reshape(B, T, D_MODEL)


class VoxelMLP(nn.Module):
    def __init__(self, n_voxels, d=D_MODEL):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(3 * n_voxels, 128),
            nn.GELU(),
            nn.Linear(128, d),
        )

    def forward(self, feat):
        # feat: B, T, 3V
        B, T, F = feat.shape
        return self.net(feat.reshape(B * T, F)).reshape(B, T, D_MODEL)


class VoxelLinear(nn.Module):
    def __init__(self, n_voxels, d=D_MODEL):
        super().__init__()
        self.proj = nn.Linear(3 * n_voxels, d)

    def forward(self, feat):
        B, T, F = feat.shape
        return self.proj(feat.reshape(B * T, F)).reshape(B, T, D_MODEL)


class VoxelGRU(nn.Module):
    def __init__(self, strategy, encoder, n_voxels, grid_shape=None):
        super().__init__()
        self.strategy = strategy
        self.encoder_name = encoder
        if strategy == "aniso3d":
            if grid_shape is None:
                raise ValueError("aniso3d requires grid_shape")
            self.spatial = (
                VoxelCNN3D(grid_shape) if encoder == "spatial" else VoxelFlat3D(grid_shape)
            )
        else:
            self.spatial = VoxelMLP(n_voxels) if encoder == "spatial" else VoxelLinear(n_voxels)
        self.rnn = nn.GRU(D_MODEL, RNN_HIDDEN, num_layers=1, batch_first=True)
        self.head = Head(RNN_HIDDEN)

    def forward(self, feat, mask_t, lengths, voxel_drop=0.0):
        if self.training and voxel_drop > 0:
            feat = _dropout_features(feat, voxel_drop)
        h = self.spatial(feat) * mask_t.unsqueeze(-1)
        packed = pack_padded_sequence(h, lengths.cpu(), batch_first=True, enforce_sorted=False)
        out, _ = self.rnn(packed)
        out, _ = pad_packed_sequence(out, batch_first=True, total_length=mask_t.shape[1])
        return self.head(out)


def _dropout_features(feat, p):
    if feat.ndim == 6:
        # B, T, 3, X, Y, Z — drop occupied spatial sites (channel 1 is M)
        occ = feat[:, :, 1] > 0
        drop = (torch.rand(occ.shape, device=feat.device) < p) & occ
        feat = feat.clone()
        feat = feat.masked_fill(drop.unsqueeze(2), 0.0)
        return feat
    # B, T, 3V stacked as [G|M|logC]
    V = feat.shape[-1] // 3
    M = feat[:, :, V : 2 * V]
    drop = (torch.rand(M.shape, device=feat.device) < p) & (M > 0)
    feat = feat.clone()
    feat[:, :, :V] = feat[:, :, :V].masked_fill(drop, 0.0)
    feat[:, :, V : 2 * V] = feat[:, :, V : 2 * V].masked_fill(drop, 0.0)
    return feat


def n_params(model):
    return int(sum(p.numel() for p in model.parameters() if p.requires_grad))


def trial_balanced_mse(pred, y, mask_t):
    sq = (pred - y).pow(2).sum(dim=-1)
    w = mask_t.to(pred.dtype)
    denom = w.sum(dim=1).clamp(min=1.0)
    return ((sq * w).sum(dim=1) / denom).mean()


class VoxelDS(Dataset):
    def __init__(self, trials, idx, y_mean, y_std, cache):
        self.trials = trials
        self.idx = np.asarray(idx)
        self.y_mean = y_mean
        self.y_std = y_std
        self.cache = cache

    def __len__(self):
        return len(self.idx)

    def __getitem__(self, j):
        i = int(self.idx[j])
        t = self.trials[i]
        y = (t["y"] - self.y_mean) / self.y_std
        valid = t.get("y_valid")
        if valid is None:
            valid = np.isfinite(np.asarray(t["y"]).reshape(len(t["y"]), -1)).all(axis=1)
        y = np.where(np.asarray(valid)[:, None], y, 0.0)
        return self.cache[i], y.astype(np.float32), np.asarray(valid, dtype=bool)


def collate_grid(batch):
    feats, ys, valids = zip(*batch)
    B = len(batch)
    T = max(f.shape[0] for f in feats)
    rest = feats[0].shape[1:]
    y_dim = int(np.asarray(ys[0]).reshape(ys[0].shape[0], -1).shape[-1])
    x = torch.zeros((B, T) + rest, dtype=torch.float32)
    y = torch.zeros(B, T, y_dim, dtype=torch.float32)
    mask_t = torch.zeros(B, T, dtype=torch.bool)
    lengths = torch.zeros(B, dtype=torch.long)
    for i, (f, yi, vi) in enumerate(zip(feats, ys, valids)):
        tt = f.shape[0]
        x[i, :tt] = torch.from_numpy(np.asarray(f))
        y[i, :tt] = torch.from_numpy(np.asarray(yi, dtype=np.float32).reshape(tt, -1))
        mask_t[i, :tt] = torch.from_numpy(np.asarray(vi, dtype=bool))
        lengths[i] = tt
    return {"x": x, "y": y, "mask_t": mask_t, "lengths": lengths}


def compact_voxel(vox, spikes, xyz, areas):
    G, M, C = vox.voxelize(spikes, xyz, areas)
    occ = np.flatnonzero(M > 0).astype(np.int32)
    return {
        "occ": occ,
        "G": G[:, occ].astype(np.float32, copy=False),
        "M": M[occ].astype(np.float32, copy=False),
        "C": C[occ].astype(np.float32, copy=False),
    }


def expand_aniso(rec, shape):
    nx, ny, nz = shape
    T = rec["G"].shape[0]
    grid = np.zeros((T, 3, nx, ny, nz), dtype=np.float32)
    occ = rec["occ"]
    if occ.size == 0:
        return grid
    ix, iy, iz = np.unravel_index(occ, shape)
    grid[:, 0, ix, iy, iz] = rec["G"]
    grid[:, 1, ix, iy, iz] = rec["M"]
    grid[:, 2, ix, iy, iz] = np.log1p(rec["C"])
    return grid


def expand_tiles(rec, n_voxels):
    T = rec["G"].shape[0]
    feat = np.zeros((T, 3 * n_voxels), dtype=np.float32)
    occ = rec["occ"]
    if occ.size == 0:
        return feat
    feat[:, occ] = rec["G"]
    feat[:, n_voxels + occ] = rec["M"]
    feat[:, 2 * n_voxels + occ] = np.log1p(rec["C"])
    return feat


class FeatCache:
    """Occupied-only voxel features; expand to the catalog tensor on read."""

    def __init__(self, items, vox):
        self.items = items
        self.name = vox.name
        self.shape = getattr(vox, "shape", None)
        self.n_voxels = vox.n_voxels

    def __getitem__(self, i):
        rec = self.items[i]
        if self.name == "aniso3d":
            return expand_aniso(rec, self.shape)
        return expand_tiles(rec, self.n_voxels)


def precompute_features(trials, vox):
    items = [compact_voxel(vox, t["spikes"], t["xyz"], t["areas"]) for t in trials]
    return FeatCache(items, vox)


def free_torch():
    gc.collect()
    if torch.backends.mps.is_available():
        torch.mps.empty_cache()


def cache_dir(task, strategy, encoder):
    d = CACHE / task / f"{strategy}_{encoder}"
    d.mkdir(parents=True, exist_ok=True)
    return d


def save_json(path, obj):
    Path(path).write_text(json.dumps(obj, indent=2, default=float))


def run_epoch(model, loader, opt, dev, train=True):
    model.train(train)
    total = 0.0
    n = 0
    for batch in loader:
        x = batch["x"].to(dev)
        y = batch["y"].to(dev)
        mask_t = batch["mask_t"].to(dev)
        if train:
            opt.zero_grad(set_to_none=True)
        pred = model(x, mask_t, batch["lengths"], voxel_drop=VOXEL_DROPOUT if train else 0.0)
        loss = trial_balanced_mse(pred, y, mask_t)
        if train:
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
        total += float(loss.item()) * x.size(0)
        n += x.size(0)
    return total / max(n, 1)


def train_one(
    strategy, encoder, n_voxels, feat_cache, trials, train_idx, val_idx, y_mean, y_std, task, grid_shape=None
):
    name = f"{strategy}_{encoder}"
    cdir = cache_dir(task, strategy, encoder)
    ckpt = cdir / "model.pt"
    hist_path = cdir / "history.json"
    if ckpt.exists() and hist_path.exists():
        log(f"  cache hit {task}/{name}")
        blob = torch.load(ckpt, map_location="cpu", weights_only=False)
        hist = json.loads(hist_path.read_text())
        return blob, hist

    dev = device_of()
    model = VoxelGRU(strategy, encoder, n_voxels, grid_shape=grid_shape).to(dev)
    log(f"  {name} params={n_params(model)}  device={dev}")
    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    tr_ds = VoxelDS(trials, train_idx, y_mean, y_std, feat_cache)
    va_ds = VoxelDS(trials, val_idx, y_mean, y_std, feat_cache)
    bs = 16 if strategy == "aniso3d" else BATCH
    tr_ld = DataLoader(tr_ds, batch_size=bs, shuffle=True, collate_fn=collate_grid)
    va_ld = DataLoader(va_ds, batch_size=bs, shuffle=False, collate_fn=collate_grid)
    best_val = math.inf
    best_state = None
    bad = 0
    hist = {"train": [], "val": [], "best_epoch": 0, "n_params": n_params(model)}
    log(f"  train {name}  n_tr={len(tr_ds)} n_val={len(va_ds)}")
    for epoch in range(1, MAX_EPOCHS + 1):
        log(f"    {name} epoch {epoch:02d} start")
        tr_loss = run_epoch(model, tr_ld, opt, dev, train=True)
        with torch.no_grad():
            va_loss = run_epoch(model, va_ld, opt, dev, train=False)
        hist["train"].append(tr_loss)
        hist["val"].append(va_loss)
        log(f"    {name} epoch {epoch:02d}  train={tr_loss:.4f}  val={va_loss:.4f}")
        if va_loss + 1e-5 < best_val:
            best_val = va_loss
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            hist["best_epoch"] = epoch
            bad = 0
        else:
            bad += 1
        slurm_utils.write_status(
            state="training",
            epoch=epoch,
            train=tr_loss,
            val=va_loss,
            best_epoch=hist["best_epoch"],
            best_val=best_val,
        )
        if bad >= PATIENCE:
            log(f"    early stop at epoch {epoch}")
            break
    blob = {
        "state_dict": best_state,
        "y_mean": y_mean,
        "y_std": y_std,
        "strategy": strategy,
        "encoder": encoder,
        "n_voxels": n_voxels,
        "grid_shape": grid_shape,
        "best_val": best_val,
    }
    torch.save(blob, ckpt)
    save_json(hist_path, hist)
    return blob, hist


def predict(blob, feat_cache, trials, idx):
    dev = device_of()
    model = VoxelGRU(
        blob["strategy"], blob["encoder"], blob["n_voxels"], grid_shape=blob.get("grid_shape")
    ).to(dev)
    model.load_state_dict(blob["state_dict"])
    model.eval()
    ds = VoxelDS(trials, idx, blob["y_mean"], blob["y_std"], feat_cache)
    bs = 16 if blob["strategy"] == "aniso3d" else BATCH
    ld = DataLoader(ds, batch_size=bs, shuffle=False, collate_fn=collate_grid)
    recs = []
    with torch.no_grad():
        for batch in ld:
            pred = model(
                batch["x"].to(dev),
                batch["mask_t"].to(dev),
                batch["lengths"],
                voxel_drop=0.0,
            ).cpu().numpy()
            y_z = batch["y"].numpy()
            mt = batch["mask_t"].numpy()
            lengths = batch["lengths"].numpy()
            for b in range(pred.shape[0]):
                T = int(lengths[b])
                p = pred[b, :T] * blob["y_std"] + blob["y_mean"]
                y = y_z[b, :T] * blob["y_std"] + blob["y_mean"]
                valid = mt[b, :T]
                p = p.astype(np.float32)
                y = y.astype(np.float32)
                p[~valid] = np.nan
                y[~valid] = np.nan
                recs.append({"pred": p, "y": y})
    return recs


def score_recs(recs, target=None):
    return mv1.score_recs(recs, target=target)


def plot_training(histories, out_dir):
    style_plots()
    tasks = [t for t in TASKS if histories.get(t)]
    keys = list(dict.fromkeys(k for t in tasks for k in histories[t]))
    if not tasks or not keys:
        return
    fig, axes = plt.subplots(
        len(tasks),
        len(keys),
        figsize=(4.0 * len(keys), 2.7 * len(tasks)),
        sharex=False,
        squeeze=False,
    )
    for r, task in enumerate(tasks):
        for c, key in enumerate(keys):
            ax = axes[r, c]
            h = histories.get(task, {}).get(key)
            if not h:
                ax.set_axis_off()
                continue
            xs = np.arange(1, len(h["train"]) + 1)
            ax.plot(xs, h["train"], color="#334155", lw=1.3, label="train")
            ax.plot(xs, h["val"], color="#C47B3B", lw=1.3, label="val")
            ax.axvline(h.get("best_epoch", 1), color="0.7", ls="--", lw=0.8)
            ax.set_title(f"{task.split('_')[0]} · {key}", fontsize=8)
            ax.set_ylabel("trial MSE (z)")
            ax.set_xlabel("epoch")
            if r == 0 and c == 0:
                ax.legend(frameon=False, fontsize=7)
    fig.suptitle("Formulation B training")
    fig.tight_layout()
    fig.savefig(out_dir / "train_curves.png", dpi=140)
    plt.close(fig)


def plot_r2_bars(rows, out_dir):
    style_plots()
    df = pd.DataFrame(rows)
    keys = [f"{s}_{e}" for s in STRATEGIES for e in ENCODERS]
    colors = {
        "aniso3d_spatial": "#4C6A92",
        "aniso3d_flat": "#9AAEBF",
        "layer_tiles_spatial": "#C47B3B",
        "layer_tiles_flat": "#E0B48A",
    }
    for metric, ylabel, fname in (
        ("r2", "held-out R² (concatenated bins)", "r2_concat"),
        ("mean_trial_r2", "held-out mean trial R²", "r2_trial"),
    ):
        plot_targets = [t for t in TARGETS if t in set(df.target)] or list(set(df.target))
        n = max(len(plot_targets), 1)
        fig, axes = plt.subplots(1, n, figsize=(5.7 * n, 4.3), sharey=False)
        axes = np.atleast_1d(axes)
        for ax, target in zip(axes, plot_targets):
            x = np.arange(len(TASKS))
            width = 0.18
            for i, key in enumerate(keys):
                vals = []
                for task in TASKS:
                    sub = df[(df.task == task) & (df.model == key) & (df.target == target)]
                    vals.append(float(sub[metric].iloc[0]) if len(sub) else np.nan)
                ax.bar(x + (i - 1.5) * width, vals, width, color=colors.get(key, "#64748B"), label=key)
            ax.axhline(0.0, color="0.6", lw=0.8)
            ax.set_xticks(x)
            ax.set_xticklabels(["trial", "session", "mouse"])
            ax.set_ylabel(ylabel)
            ax.set_title(TARGET_LABELS.get(target, target))
            ax.legend(frameon=False, fontsize=7)
        fig.tight_layout()
        fig.savefig(out_dir / f"{fname}.png", dpi=150)
        plt.close(fig)


def plot_examples(example_bank, out_dir):
    style_plots()
    colors = {
        "aniso3d_spatial": "#4C6A92",
        "aniso3d_flat": "#9AAEBF",
        "layer_tiles_spatial": "#C47B3B",
        "layer_tiles_flat": "#E0B48A",
        "true": "#111827",
    }
    keys = [f"{s}_{e}" for s in STRATEGIES for e in ENCODERS]
    for task, pack in example_bank.items():
        if not pack:
            continue
        n = min(2, len(pack["true"]))
        fig, axes = plt.subplots(n, 1, figsize=(9.4, 2.4 * max(n, 1)), sharex=False)
        axes = np.atleast_1d(axes)
        ylab = TARGET_YLABELS.get(pack.get("target", ""), "signal")
        tname = pack.get("target", "")
        for j in range(n):
            ytrue = np.asarray(pack["true"][j])
            if ytrue.ndim > 1:
                ytrue = ytrue[:, 0]
            t = np.arange(len(ytrue)) * BINSIZE
            ax = axes[j]
            ax.plot(t, ytrue, color=colors["true"], lw=1.5, label="actual")
            for key in keys:
                if key in pack and j < len(pack[key]):
                    yp = np.asarray(pack[key][j])
                    if yp.ndim > 1:
                        yp = yp[:, 0]
                    ax.plot(t, yp, color=colors.get(key, "#64748B"), lw=1.0, alpha=0.9, label=key)
            ax.set_ylabel(ylab)
            ax.set_title(f"{task}  example {j + 1}  {tname}")
            if j == 0:
                ax.legend(frameon=False, ncol=3, fontsize=7)
        axes[-1].set_xlabel("time from stimOn (s)")
        fig.tight_layout()
        fig.savefig(out_dir / f"examples_{task}.png", dpi=150)
        plt.close(fig)


def pick_examples(trials, test_idx, preds, k=2):
    scored = []
    for j, i in enumerate(test_idx):
        y = trials[i]["y"]
        if len(y) < 12:
            continue
        y0 = np.asarray(y).reshape(len(y), -1)[:, 0]
        scored.append((float(np.nanstd(y0)), len(y), j))
    scored.sort(reverse=True)
    chosen = [j for _, _, j in scored[:k]]
    keys = list(preds)
    pack = {"true": [], "target": trials[int(test_idx[0])].get("target", "") if len(test_idx) else ""}
    for key in keys:
        pack[key] = []
    for j in chosen:
        pack["true"].append(preds[keys[0]][j]["y"])
        for key in keys:
            pack[key].append(preds[key][j]["pred"])
    return pack


def write_report(splits, rows, histories, voxel_summaries, path):
    df = pd.DataFrame(rows)
    lines = []
    lines.append("# Modelv2 — Formulation B voxel encoders on ModelDataRightContra")
    lines.append("")
    lines.append(f"Built {datetime.now(timezone.utc).isoformat()}")
    lines.append("")
    lines.append("## What this is")
    lines.append("")
    lines.append(
        "Formulation B: motor units are assigned to a **fixed anatomical catalog** "
        "before any learned weights see the trial. Two catalogs are trained independently. "
        "Holdouts, trial-balanced MSE, and the GRU-64 decoder match Modelv1. "
        "Each behavior is a separate 1-d model (this folder is one target)."
    )
    lines.append("")
    lines.append(
        "**Strategy 2 (`aniso3d`)** is a 400 µm ML × 400 µm AP × 100 µm DV grid over the "
        "motor CCF hull. The spatial model is a 3-D CNN: shared Conv2d on the "
        "(ML×AP column, DV) unfolding (3×400 µm × 5×100 µm), occupancy-weighted "
        "global pool, Linear to 64. The `flat` control pools the same grid without "
        "convolution (PDF control B1)."
    )
    lines.append("")
    lines.append(
        "**Strategy 3 (`layer_tiles`)** is Allen MOp/MOs × layer (1, 2/3, 5, 6a, 6b) × "
        "400 µm ML/AP tiles. Depth is the Allen layer label, not DV cubes. "
        "`spatial` is an MLP on the flattened [G; M; log(1+C)] vector; `flat` is a Linear."
    )
    lines.append("")
    lines.append("## Holdouts")
    lines.append("")
    for task in TASKS:
        sp = splits[task]
        lines.append(
            f"- **{task}**: {sp['note']}. train n={len(sp['train'])}, test n={len(sp['test'])}."
        )
    lines.append("")
    lines.append("## Voxelization quality")
    lines.append("")
    for name, s in voxel_summaries.items():
        lines.append(f"### {name}")
        lines.append("")
        lines.append(
            f"- Catalog V={s['n_voxels_catalog']}, occupied by any session "
            f"{s['n_voxels_occupied']} ({100*s['occupancy_fraction']:.1f}%)."
        )
        lines.append(
            f"- Median occupied voxels: {s['median_session_voxels']:.0f} / session, "
            f"{s['median_mouse_voxels']:.0f} / mouse."
        )
        lines.append(
            f"- Units per occupied voxel (session-level): mean "
            f"{s['mean_units_per_occupied_voxel_session']:.2f}, median "
            f"{s['median_units_per_occupied_voxel_session']:.2f}."
        )
        lines.append(
            f"- Fraction of occupied voxels unique to one session: "
            f"{s['frac_occupied_voxels_unique_to_one_session']:.2f}. "
            f"Shared by ≥2 sessions: {s['n_voxels_shared_by_ge2_sessions']}."
        )
        lines.append(
            f"- Mean occupancy Jaccard: sessions {s['mean_jaccard_between_sessions']:.3f}, "
            f"within multi-session mice {s['mean_jaccard_within_multi_session_mice']:.3f}, "
            f"between mice {s['mean_jaccard_between_mice']:.3f}."
        )
        for task, ov in s["split_overlap"].items():
            lines.append(
                f"- {task} train∩test coverage: test voxels covered by train catalog occupancy "
                f"{ov['test_covered']:.3f} (Jaccard {ov['jaccard']:.3f}; "
                f"train V={ov['train_voxels']}, test V={ov['test_voxels']})."
            )
        lines.append("")
    lines.append(
        "A high singleton-voxel fraction plus low between-session Jaccard means the "
        "occupancy mask can fingerprint a recording. That is the failure mode the PDF "
        "warns about: the model may decode *where the probe sat* rather than a shared "
        "motor representation. Tables: `voxels/<strategy>_by_session.csv` and `_by_mouse.csv`."
    )
    lines.append("")
    lines.append("### How good are the voxel representations?")
    lines.append("")
    a = voxel_summaries.get("aniso3d", {})
    b = voxel_summaries.get("layer_tiles", {})
    if a and b:
        lines.append(
            f"Neither catalog is a dense shared motor map. Strategy 2 occupies "
            f"{a['n_voxels_occupied']}/{a['n_voxels_catalog']} voxels "
            f"({100*a['occupancy_fraction']:.1f}%); strategy 3 occupies "
            f"{b['n_voxels_occupied']}/{b['n_voxels_catalog']} "
            f"({100*b['occupancy_fraction']:.1f}%). A typical session fills only "
            f"{a['median_session_voxels']:.0f} anisotropic voxels vs "
            f"{b['median_session_voxels']:.0f} layer tiles — one Neuropixels track, "
            f"not a volume."
        )
        lines.append("")
        lines.append(
            f"Correspondence is weak on both. Between-session Jaccard is "
            f"{a['mean_jaccard_between_sessions']:.3f} (aniso3d) vs "
            f"{b['mean_jaccard_between_sessions']:.3f} (layer_tiles). "
            f"Within a multi-session mouse it only rises to "
            f"{a['mean_jaccard_within_multi_session_mice']:.3f} / "
            f"{b['mean_jaccard_within_multi_session_mice']:.3f}. "
            f"{100*a['frac_occupied_voxels_unique_to_one_session']:.0f}% of occupied "
            f"aniso3d voxels and "
            f"{100*b['frac_occupied_voxels_unique_to_one_session']:.0f}% of layer tiles "
            f"belong to a single session."
        )
        lines.append("")
        sh2 = a["split_overlap"]["session_holdout"]["test_covered"]
        sh3 = b["split_overlap"]["session_holdout"]["test_covered"]
        mh2 = a["split_overlap"]["mouse_holdout"]["test_covered"]
        mh3 = b["split_overlap"]["mouse_holdout"]["test_covered"]
        lines.append(
            f"The session holdout is the stress test: test-voxel coverage by the "
            f"train occupancy is {sh2:.2f} (aniso3d) and {sh3:.2f} (layer_tiles). "
            f"The held-out insertion of CSH_ZAD_026 lands in a disjoint set of "
            f"parcels. A 3-D CNN can still apply shared kernels at those new "
            f"coordinates; a position-specific MLP/Linear on layer tiles sees "
            f"zeros on every tile the new probe occupies. Mouse-holdout coverage "
            f"is {mh2:.2f} / {mh3:.2f} — ZM_2241 overlaps the train catalog much "
            f"more than the session holdout does."
        )
        lines.append("")
        lines.append(
            f"Pooling quality favors strategy 3: median {b['median_units_per_occupied_voxel_session']:.0f} "
            f"units/tile vs {a['median_units_per_occupied_voxel_session']:.0f} units/100 µm DV bin. "
            f"Layer tiles therefore average more neurons per parcel (less Poisson noise) "
            f"but discard within-layer depth. Aniso3d keeps a 100 µm depth axis the CNN "
            f"can filter, at the cost of sparser, more session-private voxels."
        )
        lines.append("")
        lines.append(
            "**Verdict.** Both representations implement Formulation B's correspondence "
            "map, but they are occupancy-sparse fingerprints more than a shared motor "
            "volume. Strategy 3 is the better *catalog* (higher Jaccard, fewer singletons, "
            "more units per parcel). Strategy 2 is the better *geometry* for a spatial "
            "CNN (a real DV axis). Decoding below says which of those facts wins."
        )
        lines.append("")
    lines.append("## Decoding results")
    lines.append("")
    lines.append(
        "| task | model | target | R² concat | mean trial R² | median trial R² | "
        "RMSE | MAE | Pearson | n bins | n trials |"
    )
    lines.append("|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|")
    for r in rows:
        lines.append(
            f"| {r['task']} | {r['model']} | {r['target']} | "
            f"{r['r2']:.3f} | {r['mean_trial_r2']:.3f} | {r['median_trial_r2']:.3f} | "
            f"{r['rmse']:.3f} | {r['mae']:.3f} | {r['pearson']:.3f} | "
            f"{r['n_bins']} | {r['n_trials']} |"
        )
    lines.append("")
    lines.append("### Best model per task × target (concatenated-bin R²)")
    lines.append("")
    for task in TASKS:
        for target in TARGETS:
            sub = df[(df.task == task) & (df.target == target)]
            if sub.empty:
                continue
            best = sub.loc[sub.r2.idxmax()]
            lines.append(
                f"- {task} / {target}: **{best.model}** R²={best.r2:.3f} "
                f"(mean trial R²={best.mean_trial_r2:.3f})"
            )
    lines.append("")
    lines.append("## Training diagnostics")
    lines.append("")
    keys = [f"{s}_{e}" for s in STRATEGIES for e in ENCODERS]
    for task in TASKS:
        if not histories.get(task):
            continue
        lines.append(f"### {task}")
        lines.append("")
        for key in keys:
            h = histories.get(task, {}).get(key)
            if not h:
                continue
            be = h.get("best_epoch", 0)
            vtr = h["train"][be - 1] if be and be <= len(h["train"]) else float("nan")
            vva = h["val"][be - 1] if be and be <= len(h["val"]) else float("nan")
            lines.append(
                f"- {key}: {h.get('n_params', '?')} params, best @ {be}/{len(h['train'])}  "
                f"train={vtr:.4f}  val={vva:.4f}"
            )
        lines.append("")
    lines.append(
        "Plots: `train_curves.png`, `r2_concat.png`, `r2_trial.png`, `examples_<task>.png`, "
        "`voxels/` occupancy figures."
    )
    lines.append("")
    tname = rows[0]["target"] if rows else ""
    lines.append("## Interpretation")
    lines.append("")
    lines.append(
        f"This folder is a **single-target** Formulation B model for `{tname}` "
        f"({TARGET_LABELS.get(tname, tname)}). The decoder head is 1-d; other "
        "behaviors live in sibling folders under Modelv2/."
    )
    lines.append("")
    Path(path).write_text("\n".join(lines))


def process(
    strategies=STRATEGIES,
    encoders=ENCODERS,
    tasks=TASKS,
    make_plots=True,
    job_index=None,
    target="wheel_speed",
    trials=None,
    session_meta=None,
):
    warnings.filterwarnings("ignore")
    set_seed(SEED)
    configure_out(target)
    OUT.mkdir(parents=True, exist_ok=True)
    CACHE.mkdir(parents=True, exist_ok=True)
    log(f"device={device_of()}  target={target} ({TARGET_LABELS[target]})")
    vox_dir = OUT / "voxels"
    vox_dir.mkdir(exist_ok=True)

    if trials is None or session_meta is None:
        trials, session_meta = load_corpus()
    mv1.set_target(trials, target)
    splits = make_splits(trials)
    save_json(
        OUT / "splits.json",
        {
            t: {
                "note": splits[t]["note"],
                "n_train": int(len(splits[t]["train"])),
                "n_test": int(len(splits[t]["test"])),
                "hold_eids": splits[t]["hold_eids"],
            }
            for t in TASKS
        },
    )

    sess = session_records(trials)
    all_xyz = np.concatenate([r["xyz"] for r in sess.values()], axis=0)
    all_areas = np.concatenate([r["areas"] for r in sess.values()], axis=0)
    voxelizers = {}
    if "aniso3d" in strategies:
        voxelizers["aniso3d"] = AnisoGrid(all_xyz)
    if "layer_tiles" in strategies:
        voxelizers["layer_tiles"] = LayerTiles(all_xyz, all_areas)
    voxel_summaries = {}
    for name, vox in voxelizers.items():
        if name not in strategies:
            continue
        voxel_summaries[name] = analyze_voxels(vox, sess, splits, trials, vox_dir)
    if make_plots:
        plot_voxel_comparison(voxel_summaries, vox_dir)

    rows = []
    histories = {t: {} for t in TASKS}
    preds_by_task = {t: {} for t in tasks}
    test_idx_by_task = {}
    strat_order = [s for s in ("layer_tiles", "aniso3d") if s in strategies]
    n_job = 0
    for strategy in strat_order:
        vox = voxelizers[strategy]
        log(f"precomputing {strategy} features for {len(trials)} trials")
        feats = precompute_features(trials, vox)
        for task in tasks:
            log(f"==== {task} / {strategy} / {target} ====")
            train_full = mv1.usable_idx(trials, splits[task]["train"])
            test_idx = mv1.usable_idx(trials, splits[task]["test"])
            if len(train_full) < 8 or len(test_idx) < 1:
                log(f"  skip {task}: too few usable trials for {target}")
                continue
            test_idx_by_task[task] = test_idx
            tr_idx, va_idx = train_val_split(train_full)
            tr_idx, va_idx = mv1.usable_idx(trials, tr_idx), mv1.usable_idx(trials, va_idx)
            y_mean, y_std = y_scaler_from(trials, tr_idx)
            for encoder in encoders:
                key = f"{strategy}_{encoder}"
                log(f"-- {task} / {key}")
                tag = f"{target}_{task}_{key}"
                idx = job_index if job_index is not None else n_job
                jdir = slurm_utils.begin_job(
                    OUT,
                    idx,
                    tag,
                    task=task,
                    strategy=strategy,
                    encoder=encoder,
                    target=target,
                )
                blob, hist = train_one(
                    strategy,
                    encoder,
                    vox.n_voxels,
                    feats,
                    trials,
                    tr_idx,
                    va_idx,
                    y_mean,
                    y_std,
                    task,
                    grid_shape=getattr(vox, "shape", None),
                )
                histories[task][key] = hist
                recs = predict(blob, feats, trials, test_idx)
                preds_by_task[task][key] = recs
                scores = score_recs(recs, target=target)
                these = []
                for tname, sc in scores.items():
                    row = {"task": task, "model": key, "target": tname, **sc}
                    rows.append(row)
                    these.append(row)
                    log(
                        f"   {tname:12s} R²={sc['r2']:+.3f}  trialR²={sc['mean_trial_r2']:+.3f}  "
                        f"r={sc['pearson']:.3f}"
                    )
                pd.DataFrame(these).to_csv(jdir / "scores.csv", index=False)
                slurm_utils.write_status(state="done", n_rows=len(these), best_epoch=hist.get("best_epoch"))
                n_job += 1
                if make_plots:
                    pd.DataFrame(rows).to_csv(OUT / "scores.csv", index=False)
                del blob
                free_torch()
        del feats
        free_torch()

    if make_plots:
        example_bank = {
            task: pick_examples(trials, test_idx_by_task.get(task, splits[task]["test"]), preds_by_task[task])
            for task in tasks
            if preds_by_task[task]
        }

        plot_training(histories, OUT)
        plot_r2_bars(rows, OUT)
        plot_examples(example_bank, OUT)
        write_report(splits, rows, histories, voxel_summaries, OUT / "REPORT.md")
        save_json(OUT / "scores.json", rows)
        save_json(OUT / "voxel_summaries.json", voxel_summaries)
        log("wrote " + str(OUT))
        print(pd.DataFrame(rows).to_string(index=False))
    else:
        log("wrote job dir " + str(slurm_utils.job_dir() or OUT / "jobs"))
        if rows:
            print(pd.DataFrame(rows).to_string(index=False))
    return rows


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--strategies", nargs="*", default=list(STRATEGIES))
    p.add_argument("--encoders", nargs="*", default=list(ENCODERS))
    p.add_argument("--tasks", nargs="*", default=list(TASKS))
    p.add_argument("--targets", nargs="*", default=list(TARGETS), choices=list(TARGETS))
    slurm_utils.add_common_args(p)
    args = p.parse_args()
    if args.list_jobs:
        slurm_utils.print_jobs(job_grid())
        raise SystemExit(0)
    slurm_utils.set_device(args.device)
    if args.aggregate:
        for beh in args.targets:
            d = MODEL_ROOT / beh
            if (d / "jobs").exists():
                slurm_utils.aggregate_scores(d)
        raise SystemExit(0)
    job = slurm_utils.resolve_job_index(args.job)
    if job is None:
        trials, session_meta = load_corpus()
        for beh in args.targets:
            process(
                strategies=tuple(args.strategies),
                encoders=tuple(args.encoders),
                tasks=tuple(args.tasks),
                target=beh,
                trials=trials,
                session_meta=session_meta,
            )
    else:
        cfg = slurm_utils.pick_config(job_grid(), job)
        process(
            strategies=(cfg["strategy"],),
            encoders=(cfg["encoder"],),
            tasks=(cfg["task"],),
            target=cfg["target"],
            make_plots=False,
            job_index=job,
        )
