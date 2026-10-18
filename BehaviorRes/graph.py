"""Formulation D: graph-within-graph GNN on the Modelv1 corpus.

Same motor MOp/MOs trials, trial-balanced MSE, GRU temporal decoder, and CV
tasks (interpolation, LOSO, LOMO) as Modelv1. Units are a spatial / functional
graph inside each Allen region; those region tokens can optionally sit on a
directed outer graph before the GRU.

Default (recommended on this corpus) is the inner unit graph only:

    units --distance/correlation GNN--> population state --GRU--> behavior

The outer Allen-region GNN (`--region-gnn`) is off unless asked. With only
MOp and MOs present it has two nodes and is not the first experiment.

Behavioral loss is the required baseline (`--lambda-unit 0 --lambda-region 0`).
Optional Laplacian smoothness terms exist but are disabled by default: strong
projection does not mean two regions should share a latent. Static spatial
edges and train-only session correlations are precomputed; `--func-window`
adds a causal rolling correlation on the current trial.

Local:
  python graph.py --self-test
  python graph.py --pilot --list-jobs
  python graph.py --job 0 --device cpu
  python graph.py --aggregate

Slurm:
  bash slurm/submit.sh graph.py --pilot
  bash slurm/submit.sh graph.py --pilot --paw-only --region-gnn
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import warnings
from pathlib import Path

os.environ.setdefault("MPLBACKEND", "Agg")

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import Modelv1 as mv
import slurm_utils

mv.MODEL_ROOT = ROOT / "Graph"

TASKS = ("trial_repeat", "session_loso", "mouse_lomo")
TARGETS = ("wheel_speed", "paw_speed")
PILOT_LOSO = 5
PILOT_LOMO = 5
PILOT_INTERP = 1
K_NEIGHBORS = 8
UNIT_LAYERS = 2
REGION_LAYERS = 1
EDGE_DIM = 3
FUNC_WINDOW = 8
EDGE_DROPOUT = 0.1


def log(msg):
    mv.log(msg)


def coarse_region(name):
    s = str(name)
    if s.startswith("MOp"):
        return "MOp"
    if s.startswith("MOs"):
        return "MOs"
    return s[:3] if s else "unk"


def coarse_id_table(vocab, n_areas):
    """Map fine Allen vocab ids → coarse region ids (MOp/MOs first)."""
    n_areas = max(int(n_areas), 1)
    if not vocab:
        return list(range(n_areas)), [f"a{i}" for i in range(n_areas)]
    inv = [""] * n_areas
    for name, i in vocab.items():
        i = int(i)
        if 0 <= i < n_areas:
            inv[i] = str(name)
    coarse = []
    for n in inv:
        if not n or n in ("<unk>", "pad"):
            coarse.append("unk")
        else:
            coarse.append(coarse_region(n))
    order = []
    for pref in ("unk", "MOp", "MOs"):
        if pref in coarse and pref not in order:
            order.append(pref)
    for n in coarse:
        if n not in order:
            order.append(n)
    lut = {n: i for i, n in enumerate(order)}
    return [lut[n] for n in coarse], order


class EdgeMessage(nn.Module):
    """Ψ(h_n, h_q, e_qn) → message in R^d."""

    def __init__(self, d, e_dim=EDGE_DIM, dropout=mv.DROPOUT):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(2 * d + e_dim, d),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d, d),
        )

    def forward(self, h_n, h_q, e):
        return self.net(torch.cat([h_n, h_q, e], dim=-1))


class ResidualUpdate(nn.Module):
    """Φ(h, m) = h + MLP([h, m]), post-norm."""

    def __init__(self, d, dropout=mv.DROPOUT):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(2 * d, d), nn.GELU(), nn.Dropout(dropout), nn.Linear(d, d))
        self.norm = nn.LayerNorm(d)

    def forward(self, h, m):
        return self.norm(h + self.net(torch.cat([h, m], dim=-1)))


def gather_pair(mat, idx):
    """mat (B, N, N), idx (B, N, K) → (B, N, K)."""
    return torch.gather(mat, 2, idx)


def gather_nodes(h, idx):
    """h (B, N, D), idx (B, N, K) → (B, N, K, D)."""
    bsz, n_node, dim = h.shape
    k = idx.size(-1)
    take = idx.reshape(bsz, n_node * k, 1).expand(bsz, n_node * k, dim)
    return torch.gather(h, 1, take).reshape(bsz, n_node, k, dim)


def knn_from_dist(dist, valid, k):
    """valid (B, N, N) bool; returns idx (B, N, K) and keep mask."""
    k = max(1, min(int(k), dist.size(-1) - 1))
    filled = dist.masked_fill(~valid, float("inf"))
    idx = filled.topk(k, dim=-1, largest=False).indices
    keep = torch.gather(valid, 2, idx)
    return idx, keep


class UnitGNN(nn.Module):
    def __init__(self, d, n_layers=UNIT_LAYERS, k=K_NEIGHBORS, dropout=mv.DROPOUT):
        super().__init__()
        self.k = int(k)
        self.layers = nn.ModuleList(
            nn.ModuleDict({"psi": EdgeMessage(d, EDGE_DIM, dropout), "phi": ResidualUpdate(d, dropout)})
            for _ in range(n_layers)
        )

    def forward(self, h, dist, rho, same, mask_n, mask_t, area=None, train=False):
        # h: (B, N, T, D); dist/rho/same: (B, N, N)
        bsz, n_node, n_t, d = h.shape
        observed = mask_n.unsqueeze(2) & mask_n.unsqueeze(1)
        same_region = observed
        if area is not None:
            same_region = same_region & (area.unsqueeze(2) == area.unsqueeze(1))
        valid = same_region & ~torch.eye(n_node, dtype=torch.bool, device=h.device).unsqueeze(0)
        idx, keep = knn_from_dist(dist, valid, self.k)
        if train and EDGE_DROPOUT > 0:
            keep = keep & (torch.rand(keep.shape, device=h.device) >= EDGE_DROPOUT)
        e = torch.stack(
            [gather_pair(dist, idx), gather_pair(rho, idx), gather_pair(same, idx)],
            dim=-1,
        )
        w = keep.to(h.dtype).unsqueeze(-1)
        hp = h.permute(0, 2, 1, 3).reshape(bsz * n_t, n_node, d)
        idx_t = idx.unsqueeze(1).expand(bsz, n_t, n_node, idx.size(-1)).reshape(bsz * n_t, n_node, idx.size(-1))
        e_t = e.unsqueeze(1).expand(bsz, n_t, n_node, e.size(2), e.size(3))
        e_t = e_t.reshape(bsz * n_t, n_node, e.size(2), e.size(3))
        w_t = w.unsqueeze(1).expand(bsz, n_t, n_node, w.size(2), 1).reshape(bsz * n_t, n_node, w.size(2), 1)
        smooth = h.new_zeros(())
        n_smooth = 0
        for layer in self.layers:
            h_n = hp.unsqueeze(2).expand(-1, -1, idx_t.size(-1), -1)
            h_q = gather_nodes(hp, idx_t)
            msg = layer["psi"](h_n, h_q, e_t) * w_t
            pooled = msg.sum(dim=2) / w_t.sum(dim=2).clamp(min=1.0)
            hp = layer["phi"](hp, pooled)
            dh = (h_n - h_q).pow(2).sum(dim=-1, keepdim=True)
            smooth = smooth + (dh * w_t * e_t[..., 1:2].abs()).sum()
            n_smooth += int(w_t.sum().item())
        h_out = hp.reshape(bsz, n_t, n_node, d).permute(0, 2, 1, 3)
        h_out = h_out * mask_n.unsqueeze(-1).unsqueeze(-1).to(h_out.dtype)
        h_out = h_out * mask_t.unsqueeze(1).unsqueeze(-1).to(h_out.dtype)
        if n_smooth:
            smooth = smooth / n_smooth
        return h_out, smooth


class RegionGNN(nn.Module):
    def __init__(self, d, n_layers=REGION_LAYERS, e_dim=5, dropout=mv.DROPOUT):
        super().__init__()
        self.layers = nn.ModuleList(
            nn.ModuleDict(
                {
                    "psi": nn.Sequential(nn.Linear(d + e_dim, d), nn.GELU(), nn.Dropout(dropout), nn.Linear(d, d)),
                    "phi": ResidualUpdate(d, dropout),
                }
            )
            for _ in range(n_layers)
        )

    def forward(self, g, edge, present):
        # g: (B, R, T, D); edge: (R, R, E); present: (B, R)
        bsz, n_r, n_t, d = g.shape
        src_ok = present.unsqueeze(2) & present.unsqueeze(1)
        src_ok = src_ok & ~torch.eye(n_r, dtype=torch.bool, device=g.device).unsqueeze(0)
        gp = g.permute(0, 2, 1, 3)
        e = edge.to(device=g.device, dtype=g.dtype)
        for layer in self.layers:
            src = gp.unsqueeze(2).expand(bsz, n_t, n_r, n_r, d)
            e_bt = e.view(1, 1, n_r, n_r, -1).expand(bsz, n_t, n_r, n_r, -1)
            msg = layer["psi"](torch.cat([src, e_bt], dim=-1))
            w = src_ok.to(g.dtype).view(bsz, 1, n_r, n_r, 1)
            pooled = (msg * w).sum(dim=3) / w.sum(dim=3).clamp(min=1.0)
            gp = layer["phi"](gp, pooled)
            gp = gp * present.unsqueeze(1).unsqueeze(-1).to(gp.dtype)
        return gp.permute(0, 2, 1, 3)


class GraphSequence(nn.Module):
    """Inner unit GNN (+ optional region GNN) then the Modelv1 causal GRU."""

    def __init__(self, n_areas, k=K_NEIGHBORS, unit_layers=UNIT_LAYERS, region_gnn=False, vocab=None):
        super().__init__()
        self.region_gnn_on = bool(region_gnn)
        d = mv.D_MODEL
        self.enc = mv.UnitEncoder(n_areas, use_anatomy=True, extra_layer=False)
        self.unit_gnn = UnitGNN(d, n_layers=unit_layers, k=k)
        self.unit_pool = mv.AttnPool(d)
        self.region_pool = mv.AttnPool(d)
        self.global_pool = mv.AttnPool(d)
        self.region_gnn = RegionGNN(d) if self.region_gnn_on else None
        self.rnn = nn.GRU(d, mv.RNN_HIDDEN, num_layers=1, batch_first=True)
        self.head = mv.Head(mv.RNN_HIDDEN)
        cmap, names = coarse_id_table(vocab, n_areas)
        self.coarse_names = list(names)
        self.register_buffer("coarse_map", torch.tensor(cmap, dtype=torch.long), persistent=False)

    def coarse_area(self, area):
        amax = max(int(self.coarse_map.numel()) - 1, 0)
        return self.coarse_map[area.clamp(0, amax)]

    def forward(self, x, xyz, area, rate, mask_n, mask_t, lengths, rho=None, unit_drop=0.0):
        if self.training and unit_drop > 0:
            mask_n = mv.apply_unit_dropout(mask_n, unit_drop)
        h = self.enc(x, xyz, area, rate, mask_n)
        dist, same, rho_e = unit_edge_features(xyz, mask_n, rho)
        area_c = self.coarse_area(area)
        h, smooth_u = self.unit_gnn(
            h, dist, rho_e, same, mask_n, mask_t, area=area_c, train=self.training
        )
        smooth_r = h.new_zeros(())
        if self.region_gnn_on:
            g, present, rids = pool_regions(h, area_c, mask_n, self.region_pool)
            edge = region_edges(xyz, area_c, mask_n, rids, names=self.coarse_names)
            g = self.region_gnn(g, edge, present)
            z = attn_over_regions(self.global_pool, g, present, mask_t)
            if present.size(1) >= 2:
                smooth_r = region_smooth(g, present, edge)
        else:
            z = self.unit_pool(h, mask_n) * mask_t.unsqueeze(-1)
        packed = pack_padded_sequence(z, lengths.cpu(), batch_first=True, enforce_sorted=False)
        out, _ = self.rnn(packed)
        out, _ = pad_packed_sequence(out, batch_first=True, total_length=x.shape[-1])
        pred = self.head(out)
        return pred, smooth_u, smooth_r


def attn_over_regions(pool, g, present, mask_t):
    """g: (B, R, T, D) → (B, T, D). AttnPool sees regions as the set axis."""
    return pool(g, present) * mask_t.unsqueeze(-1)


def unit_edge_features(xyz, mask_n, rho):
    diff = xyz.unsqueeze(2) - xyz.unsqueeze(1)
    dist = diff.pow(2).sum(dim=-1).sqrt()
    hemi = torch.sign(xyz[..., 0])
    same = (hemi.unsqueeze(2) == hemi.unsqueeze(1)).to(xyz.dtype)
    if rho is None:
        rho_e = torch.zeros_like(dist)
    else:
        rho_e = rho.to(device=xyz.device, dtype=xyz.dtype)
        if rho_e.dim() == 4:
            rho_e = rho_e.mean(dim=-1)
    rho_e = rho_e * (mask_n.unsqueeze(2) & mask_n.unsqueeze(1)).to(rho_e.dtype)
    return dist, same, rho_e


def pool_regions(h, area, mask_n, pool):
    ids = sorted({int(v) for v in area.detach().cpu().reshape(-1).tolist() if int(v) > 0})
    if not ids:
        ids = [0]
    bsz, _, n_t, d = h.shape
    g = h.new_zeros(bsz, len(ids), n_t, d)
    present = h.new_zeros(bsz, len(ids), dtype=torch.bool)
    for j, rid in enumerate(ids):
        m = (area == rid) & mask_n
        present[:, j] = m.any(dim=1)
        if bool(present[:, j].any()):
            g[:, j] = pool(h, m)
    return g, present, ids


def _region_name(names, rid):
    if not names:
        return "unk"
    i = int(rid)
    if 0 <= i < len(names):
        return str(names[i])
    return "unk"


def region_edges(xyz, area, mask_n, ids, names=None):
    """e[r, q] = e_qr = [C_qr, C_rq, d_centroid, I_ontology, I_contralateral].

    No CCFv3 projection file is in the repo, so C is 1 on off-diagonal edges.
    Ontology is 1 when coarse names share a two-letter parent (MO for MOp/MOs).
    """
    ids = list(ids) or [0]
    n_r = len(ids)
    device, dtype = xyz.device, xyz.dtype
    cents = []
    hemi = []
    labels = []
    for rid in ids:
        m = (area == rid) & mask_n
        w = m.to(dtype).unsqueeze(-1)
        num = (xyz * w).sum(dim=(0, 1))
        den = w.sum().clamp(min=1.0)
        cents.append(num / den)
        sx = (xyz[..., 0] * m.to(dtype)).sum() / den
        hemi.append(torch.sign(sx))
        labels.append(_region_name(names, rid))
    cents = torch.stack(cents, dim=0)
    hemi = torch.stack(hemi, dim=0)
    d = torch.cdist(cents.unsqueeze(0), cents.unsqueeze(0), p=2).squeeze(0)
    contra = (hemi.unsqueeze(0) != hemi.unsqueeze(1)).to(dtype)
    ont = torch.zeros(n_r, n_r, device=device, dtype=dtype)
    for i, ni in enumerate(labels):
        for j, nj in enumerate(labels):
            if ni != "unk" and nj != "unk" and ni[:2] == nj[:2]:
                ont[i, j] = 1.0
    c_rq = torch.ones(n_r, n_r, device=device, dtype=dtype)
    eye = torch.eye(n_r, device=device, dtype=torch.bool)
    c_rq = c_rq.masked_fill(eye, 0.0)
    # Indexed [r, q] as destination, source: message uses e_qr = [C_qr, C_rq, ...]
    e = torch.stack([c_rq.T.contiguous(), c_rq, d, ont, contra], dim=-1)
    return e


def region_smooth(g, present, edge):
    # g: (B, R, T, D); weight by C_rq (channel 1 of e[r,q] = [C_qr, C_rq, ...])
    w = edge[..., 1].to(g.device, g.dtype)
    w = w * present.unsqueeze(2).to(g.dtype) * present.unsqueeze(1).to(g.dtype)
    dh = (g.unsqueeze(2) - g.unsqueeze(1)).pow(2).sum(dim=-1).mean(dim=3)
    den = w.sum().clamp(min=1.0)
    return (dh * w.unsqueeze(0)).sum() / den


def window_corr(x, mask_n, mask_t, width=FUNC_WINDOW):
    """Causal rolling Pearson over the last `width` bins. x: (B, N, T)."""
    bsz, n_node, n_t = x.shape
    w = max(2, int(width))
    pad = x.new_zeros(bsz, n_node, w - 1)
    xw = torch.cat([pad, x], dim=-1).unfold(-1, w, 1)
    # xw: B, N, T, W
    xc = xw - xw.mean(dim=-1, keepdim=True)
    num = torch.einsum("bntw,bqtw->bntq", xc, xc)
    nrm = xc.norm(dim=-1).clamp(min=1e-6)
    den = nrm.unsqueeze(3) * nrm.unsqueeze(2)
    rho = (num / den).permute(0, 1, 3, 2)
    valid = (mask_n.unsqueeze(2) & mask_n.unsqueeze(1)).unsqueeze(-1)
    rho = rho.masked_fill(~valid, 0.0)
    rho = rho * mask_t.view(bsz, 1, 1, n_t).to(rho.dtype)
    return rho.mean(dim=-1)


def session_train_corr(trials, idx):
    buckets = {}
    for i in np.asarray(idx, dtype=int):
        t = trials[int(i)]
        buckets.setdefault(str(t["eid"]), []).append(np.asarray(t["spikes"], dtype=np.float32))
    out = {}
    for eid, xs in buckets.items():
        x = np.concatenate(xs, axis=0)
        n = x.shape[1]
        if x.shape[0] < 4 or n < 2:
            out[eid] = np.eye(n, dtype=np.float32)
            continue
        c = np.corrcoef(x.T)
        c = np.nan_to_num(np.asarray(c, dtype=np.float32), nan=0.0, posinf=0.0, neginf=0.0)
        np.fill_diagonal(c, 0.0)
        out[eid] = c
    return out


class GraphDS(mv.TrialDS):
    def __getitem__(self, j):
        item = super().__getitem__(j)
        t = self.trials[int(self.idx[j])]
        return (*item, str(t["eid"]))


def make_collate(rho_by_eid):
    def _collate(batch):
        core = [b[:-1] for b in batch]
        eids = [b[-1] for b in batch]
        out = mv.collate(core)
        bsz, n_node = out["x"].shape[:2]
        rho = torch.zeros(bsz, n_node, n_node, dtype=torch.float32)
        for i, eid in enumerate(eids):
            r = rho_by_eid.get(eid)
            if r is None:
                continue
            n = min(n_node, int(r.shape[0]))
            rho[i, :n, :n] = torch.from_numpy(np.asarray(r[:n, :n], dtype=np.float32))
        out["rho"] = rho
        return out

    return _collate


def run_epoch(model, loader, opt, dev, train, lam_u, lam_r, func_window):
    model.train(train)
    tot = {k: 0.0 for k in ("loss", "beh", "u", "r")}
    n = 0
    for batch in loader:
        x = batch["x"].to(dev)
        y = batch["y"].to(dev)
        xyz = batch["xyz"].to(dev)
        area = batch["area"].to(dev)
        rate = batch["rate"].to(dev)
        mask_t = batch["mask_t"].to(dev)
        mask_loss = batch["mask_loss"].to(dev)
        mask_n = batch["mask_n"].to(dev)
        lengths = batch["lengths"]
        rho = batch["rho"].to(dev)
        if func_window:
            rho = window_corr(x, mask_n, mask_t)
        if train:
            opt.zero_grad(set_to_none=True)
        pred, su, sr = model(
            x,
            xyz,
            area,
            rate,
            mask_n,
            mask_t,
            lengths,
            rho=rho,
            unit_drop=mv.UNIT_DROPOUT if train else 0.0,
        )
        beh = mv.trial_balanced_mse(pred, y, mask_loss)
        loss = beh + lam_u * su + lam_r * sr
        if train:
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
        bs = int(x.size(0))
        tot["loss"] += float(loss.item()) * bs
        tot["beh"] += float(beh.item()) * bs
        tot["u"] += float(su.item()) * bs
        tot["r"] += float(sr.item()) * bs
        n += bs
    d = max(n, 1)
    return {k: v / d for k, v in tot.items()}


def model_tag(region_gnn, func_window, k, unit_layers):
    parts = ["graph", f"k{int(k)}", f"lu{int(unit_layers)}"]
    if region_gnn:
        parts.append("region")
    if func_window:
        parts.append("win")
    return "_".join(parts)


def cache_dir(task, fold_id, target, region_gnn, func_window, k, unit_layers, lam_u, lam_r):
    mv.configure_out(target, "cv")
    d = mv.CACHE / str(task)
    if fold_id:
        safe = "".join(ch if ch.isalnum() or ch in "-_." else "_" for ch in str(fold_id))
        d = d / safe
    tag = model_tag(region_gnn, func_window, k, unit_layers)
    if lam_u or lam_r:
        tag += f"_lu{lam_u:g}_lr{lam_r:g}"
    d = d / tag
    d.mkdir(parents=True, exist_ok=True)
    return d


def train_one(
    trials,
    tr_idx,
    va_idx,
    y_mean,
    y_std,
    stats,
    task,
    fold_id,
    target,
    bin_key,
    region_gnn,
    func_window,
    k,
    unit_layers,
    lam_u,
    lam_r,
):
    tag = model_tag(region_gnn, func_window, k, unit_layers)
    cdir = cache_dir(task, fold_id, target, region_gnn, func_window, k, unit_layers, lam_u, lam_r)
    ckpt = cdir / "model.pt"
    hist_path = cdir / "history.json"
    if ckpt.exists() and hist_path.exists():
        hist = json.loads(hist_path.read_text())
        if hist.get("finished"):
            log(f"  cache hit {task}/{fold_id}/{tag}")
            blob = torch.load(ckpt, map_location="cpu", weights_only=False)
            return blob, hist
    rho_by_eid = session_train_corr(trials, tr_idx)
    collate = make_collate(rho_by_eid)
    dev = mv.device_of()
    model = GraphSequence(
        stats["n_areas"],
        k=k,
        unit_layers=unit_layers,
        region_gnn=region_gnn,
        vocab=stats.get("vocab"),
    ).to(dev)
    log(f"  {tag} params={mv.n_params(model)}  device={dev}  region_gnn={int(region_gnn)}")
    opt = torch.optim.AdamW(model.parameters(), lr=mv.LR, weight_decay=mv.WEIGHT_DECAY)
    tr_ds = GraphDS(trials, tr_idx, y_mean, y_std, stats, bin_key=bin_key)
    va_ds = GraphDS(trials, va_idx, y_mean, y_std, stats, bin_key=bin_key)
    tr_ld = DataLoader(tr_ds, batch_size=mv.BATCH, shuffle=True, collate_fn=collate)
    va_ld = DataLoader(va_ds, batch_size=mv.BATCH, shuffle=False, collate_fn=collate)
    best_val, best_state, bad = math.inf, None, 0
    hist = {
        "train": [],
        "val": [],
        "best_epoch": 0,
        "n_params": mv.n_params(model),
        "region_gnn": bool(region_gnn),
        "func_window": bool(func_window),
        "finished": False,
    }
    log(f"  train {tag}  n_tr={len(tr_ds)} n_val={len(va_ds)}")

    def snapshot():
        return {
            "state_dict": best_state,
            "y_mean": y_mean,
            "y_std": y_std,
            "n_areas": stats["n_areas"],
            "region_gnn": bool(region_gnn),
            "func_window": bool(func_window),
            "k": int(k),
            "unit_layers": int(unit_layers),
            "best_val": best_val,
            "rho_by_eid": rho_by_eid,
            "stats": {
                "vocab": stats["vocab"],
                "n_areas": stats["n_areas"],
                "xyz_mean": stats["xyz_mean"],
                "xyz_std": stats["xyz_std"],
                "rates": stats["rates"],
            },
        }

    for epoch in range(1, mv.MAX_EPOCHS + 1):
        tr = run_epoch(model, tr_ld, opt, dev, True, lam_u, lam_r, func_window)
        with torch.no_grad():
            va = run_epoch(model, va_ld, None, dev, False, lam_u, lam_r, func_window)
        hist["train"].append(tr)
        hist["val"].append(va)
        log(
            f"    {tag} {epoch:02d}  train={tr['loss']:.4f}  val={va['loss']:.4f}  "
            f"beh={va['beh']:.4f}"
        )
        if va["loss"] + 1e-5 < best_val:
            best_val = va["loss"]
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            hist["best_epoch"] = epoch
            bad = 0
        else:
            bad += 1
        slurm_utils.write_status(
            state="training",
            epoch=epoch,
            train=tr["loss"],
            val=va["loss"],
            best_val=best_val,
            best_epoch=hist["best_epoch"],
        )
        if bad >= mv.PATIENCE:
            log(f"    early stop at epoch {epoch}")
            break
    hist["finished"] = True
    blob = snapshot()
    torch.save(blob, ckpt)
    mv.save_json(hist_path, hist)
    return blob, hist


def load_model(blob, stats):
    stats = {
        "vocab": stats["vocab"],
        "n_areas": int(stats["n_areas"]),
        "xyz_mean": np.asarray(stats["xyz_mean"], dtype=np.float32),
        "xyz_std": np.asarray(stats["xyz_std"], dtype=np.float32),
        "rates": {k: np.asarray(v, dtype=np.float32) for k, v in stats["rates"].items()},
    }
    model = GraphSequence(
        blob["n_areas"],
        k=int(blob.get("k", K_NEIGHBORS)),
        unit_layers=int(blob.get("unit_layers", UNIT_LAYERS)),
        region_gnn=bool(blob.get("region_gnn", False)),
        vocab=stats.get("vocab"),
    )
    missing, unexpected = model.load_state_dict(blob["state_dict"], strict=False)
    if missing or unexpected:
        log(f"  state_dict missing={len(missing)} unexpected={len(unexpected)}")
    return model, stats


def predict(blob, trials, idx, stats):
    stats_b = blob.get("stats", stats)
    model, stats = load_model(blob, stats_b)
    dev = mv.device_of()
    model.to(dev)
    model.eval()
    rho_by_eid = blob.get("rho_by_eid") or {}
    ds = GraphDS(trials, idx, blob["y_mean"], blob["y_std"], stats)
    ld = DataLoader(ds, batch_size=mv.BATCH, shuffle=False, collate_fn=make_collate(rho_by_eid))
    recs = []
    with torch.no_grad():
        for batch in ld:
            rho = batch["rho"].to(dev)
            if blob.get("func_window"):
                rho = window_corr(batch["x"].to(dev), batch["mask_n"].to(dev), batch["mask_t"].to(dev))
            pred, _, _ = model(
                batch["x"].to(dev),
                batch["xyz"].to(dev),
                batch["area"].to(dev),
                batch["rate"].to(dev),
                batch["mask_n"].to(dev),
                batch["mask_t"].to(dev),
                batch["lengths"],
                rho=rho,
                unit_drop=0.0,
            )
            pred = pred.cpu().numpy()
            y_z = batch["y"].numpy()
            mt = batch["mask_t"].numpy()
            lengths = batch["lengths"].numpy()
            for b in range(y_z.shape[0]):
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


def subsample_folds(folds, n_loso, n_lomo, n_interp, seed=0):
    grouped = {"trial_repeat": [], "session_loso": [], "mouse_lomo": []}
    for spec in folds:
        grouped.setdefault(spec["task"], []).append(spec)
    rng = np.random.RandomState(int(mv.SEED) + int(seed))

    def pick(task, n):
        specs = list(grouped.get(task, []))
        if n is None or int(n) >= len(specs):
            return specs
        n = int(n)
        if n <= 0:
            return []
        chosen = [specs[int(i)] for i in rng.choice(len(specs), size=n, replace=False)]
        chosen.sort(key=lambda s: (int(s["fold"]), str(s["fold_id"])))
        return chosen

    return pick("trial_repeat", n_interp) + pick("session_loso", n_loso) + pick("mouse_lomo", n_lomo)


def job_grid(
    trials=None,
    targets=None,
    tasks=None,
    pilot=False,
    loso_folds=None,
    lomo_folds=None,
    interp_folds=None,
    fold_seed=0,
    region_gnn=False,
    func_window=False,
):
    if trials is None:
        trials, _ = mv.load_corpus()
    folds = mv.cv_folds(trials, n_trial_repeats=1, tasks=tuple(tasks or TASKS))
    if pilot:
        folds = subsample_folds(
            folds,
            n_loso=PILOT_LOSO if loso_folds is None else int(loso_folds),
            n_lomo=PILOT_LOMO if lomo_folds is None else int(lomo_folds),
            n_interp=PILOT_INTERP if interp_folds is None else int(interp_folds),
            seed=fold_seed,
        )
    elif any(v is not None for v in (loso_folds, lomo_folds, interp_folds)):
        folds = subsample_folds(
            folds,
            n_loso=loso_folds,
            n_lomo=lomo_folds,
            n_interp=interp_folds,
            seed=fold_seed,
        )
    targets = list(targets or TARGETS)
    configs = []
    for beh in targets:
        for spec in folds:
            if spec["task"] not in (tasks or TASKS):
                continue
            configs.append(
                {
                    "protocol": "cv",
                    "target": beh,
                    "task": spec["task"],
                    "fold": int(spec["fold"]),
                    "fold_id": spec["fold_id"],
                    "region_gnn": bool(region_gnn),
                    "func_window": bool(func_window),
                }
            )
    return configs


def process(
    target,
    task,
    fold=None,
    fold_id=None,
    trials=None,
    session_meta=None,
    job_index=None,
    make_plots=True,
    region_gnn=False,
    func_window=False,
    k=K_NEIGHBORS,
    unit_layers=UNIT_LAYERS,
    lam_u=0.0,
    lam_r=0.0,
    n_trial_repeats=1,
    pilot=False,
    loso_folds=None,
    lomo_folds=None,
    interp_folds=None,
    fold_seed=0,
):
    warnings.filterwarnings("ignore")
    mv.set_seed(mv.SEED)
    mv.configure_out(target, "cv")
    mv.OUT.mkdir(parents=True, exist_ok=True)
    mv.CACHE.mkdir(parents=True, exist_ok=True)
    log(
        f"device={mv.device_of()}  target={target}  task={task}  "
        f"Formulation D graph  region={int(region_gnn)}  win={int(func_window)}  k={k}"
    )
    if trials is None or session_meta is None:
        trials, session_meta = mv.load_corpus()
    mv.set_target(trials, target, log1p=False)
    specs = mv.cv_folds(trials, n_trial_repeats=n_trial_repeats, tasks=TASKS)
    specs = [s for s in specs if s["task"] == task]
    if fold_id is not None:
        specs = [s for s in specs if str(s["fold_id"]) == str(fold_id)]
    else:
        if pilot or any(v is not None for v in (loso_folds, lomo_folds, interp_folds)):
            specs = subsample_folds(
                specs,
                n_loso=PILOT_LOSO if (pilot and loso_folds is None) else loso_folds,
                n_lomo=PILOT_LOMO if (pilot and lomo_folds is None) else lomo_folds,
                n_interp=PILOT_INTERP if (pilot and interp_folds is None) else interp_folds,
                seed=fold_seed,
            )
        if fold is not None:
            specs = [s for s in specs if int(s["fold"]) == int(fold)]
    rows = []
    n_job = 0
    for spec in specs:
        fold_i = int(spec["fold"])
        fid = spec["fold_id"]
        log(f"==== {task}  fold={fold_i} {fid} ====")
        mv.clear_bin_masks(trials)
        bin_mode = spec.get("bin_mode")
        if bin_mode:
            mv.apply_bin_masks(trials, spec)
            train_full = mv.usable_idx(
                trials, spec["train"], min_bins=max(4, mv.MIN_BINS // 2), mask_key="train_bins"
            )
            test_ok = mv.usable_idx(trials, spec["test"], min_bins=1, mask_key="test_bins")
            both = np.intersect1d(train_full, test_ok)
            train_full, test_idx = both, both
        else:
            train_full = mv.usable_idx(trials, spec["train"])
            test_idx = mv.usable_idx(trials, spec["test"])
        if len(train_full) < 8 or len(test_idx) < 1:
            log(f"  skip {task}/{fid}: too few trials")
            continue
        tr_idx, va_idx = mv.train_val_split(train_full, seed=mv.SEED + 17 * fold_i)
        if bin_mode:
            tr_idx = mv.usable_idx(
                trials, tr_idx, min_bins=max(4, mv.MIN_BINS // 2), mask_key="train_bins"
            )
            va_idx = mv.usable_idx(trials, va_idx, min_bins=1, mask_key="train_bins")
        else:
            tr_idx, va_idx = mv.usable_idx(trials, tr_idx), mv.usable_idx(trials, va_idx)
        if len(tr_idx) < 4:
            log(f"  skip {task}/{fid}: empty train after val split")
            continue
        y_mean, y_std = mv.y_scaler_from(trials, tr_idx)
        stats = mv.fit_unit_stats(trials, tr_idx, np.concatenate([tr_idx, va_idx, test_idx]))
        tag = f"{target}_{task}_{fid}_{model_tag(region_gnn, func_window, k, unit_layers)}"
        idx = job_index if job_index is not None else n_job
        jdir = slurm_utils.begin_job(
            mv.OUT,
            idx,
            tag,
            task=task,
            target=target,
            protocol="cv",
            fold=fold_i,
            fold_id=fid,
            formulation="D",
            region_gnn=int(region_gnn),
            func_window=int(func_window),
        )
        blob, hist = train_one(
            trials,
            tr_idx,
            va_idx,
            y_mean,
            y_std,
            stats,
            task,
            fid,
            target,
            bin_key="train_bins" if bin_mode else None,
            region_gnn=region_gnn,
            func_window=func_window,
            k=k,
            unit_layers=unit_layers,
            lam_u=lam_u,
            lam_r=lam_r,
        )
        recs = predict(blob, trials, test_idx, blob.get("stats", stats))
        min_bins = 2 if bin_mode else 8
        if bin_mode:
            recs = mv.restrict_recs(recs, trials, test_idx, "test_bins")
        sc = mv.score_recs(recs, target=target, min_trial_bins=min_bins)[target]
        row = {
            "protocol": "cv",
            "task": task,
            "target": target,
            "formulation": "D",
            "region_gnn": int(region_gnn),
            "func_window": int(func_window),
            "k": int(k),
            "unit_layers": int(unit_layers),
            "fold": fold_i,
            "fold_id": fid,
            "bin_mode": bin_mode or "",
            **sc,
        }
        rows.append(row)
        log(
            f"   R²={sc['r2']:+.3f}  mean_trial={sc['mean_trial_r2']:+.3f}  "
            f"median_trial={sc['median_trial_r2']:+.3f}  n={sc['n_trials']}"
        )
        pd.DataFrame([row]).to_csv(jdir / "scores.csv", index=False)
        slurm_utils.write_status(state="done", n_rows=1, best_epoch=hist.get("best_epoch"))
        n_job += 1
    if make_plots and rows:
        pd.DataFrame(rows).to_csv(mv.OUT / "scores.csv", index=False)
        write_report(rows, mv.OUT / "REPORT.md")
        log("wrote " + str(mv.OUT))
    return rows


def write_report(rows, path):
    df = pd.DataFrame(rows)
    lines = ["# Formulation D (graph-within-graph GNN)", ""]
    if df.empty:
        Path(path).write_text("\n".join(lines))
        return
    keys = ["task", "target", "region_gnn"]
    lines.append("| task | target | region GNN | n | concat R² | mean trial R² |")
    lines.append("|---|---|---:|---:|---:|---:|")
    for name, sub in df.groupby(keys, dropna=False):
        rec = dict(zip(keys, name if isinstance(name, tuple) else (name,)))
        lines.append(
            f"| {rec['task']} | {rec['target']} | {rec['region_gnn']} | {len(sub)} | "
            f"{sub['r2'].mean():+.3f} | {sub['mean_trial_r2'].mean():+.3f} |"
        )
    Path(path).write_text("\n".join(lines) + "\n")


def aggregate_all():
    root = mv.MODEL_ROOT
    all_rows = []
    for beh in TARGETS:
        d = root / beh / "cv"
        jobs = d / "jobs"
        if not jobs.exists():
            continue
        parts = [pd.read_csv(p) for p in sorted(jobs.glob("*/scores.csv"))]
        if not parts:
            continue
        df = pd.concat(parts, ignore_index=True)
        d.mkdir(parents=True, exist_ok=True)
        df.to_csv(d / "scores.csv", index=False)
        all_rows.append(df)
    if not all_rows:
        raise SystemExit(f"no per-job scores under {root}")
    df = pd.concat(all_rows, ignore_index=True)
    root.mkdir(parents=True, exist_ok=True)
    df.to_csv(root / "scores.csv", index=False)
    write_report(df.to_dict(orient="records"), root / "REPORT.md")
    log(f"aggregated {len(df)} rows → {root}")
    print(df.groupby(["task", "target"])[["r2", "mean_trial_r2"]].mean().to_string())


def run_self_test():
    torch.manual_seed(0)
    bsz, n, t = 2, 6, 12
    x = torch.randn(bsz, n, t)
    xyz = torch.randn(bsz, n, 3)
    xyz[:, :3, 0] = -1.0
    xyz[:, 3:, 0] = 1.0
    area = torch.tensor([[1, 1, 1, 2, 2, 2], [1, 1, 2, 2, 2, 1]])
    rate = torch.zeros(bsz, n)
    mask_n = torch.ones(bsz, n, dtype=torch.bool)
    mask_t = torch.ones(bsz, t, dtype=torch.bool)
    lengths = torch.full((bsz,), t)
    y = torch.randn(bsz, t, 1)
    rho = torch.randn(bsz, n, n)
    for region_on in (False, True):
        model = GraphSequence(3, k=3, unit_layers=1, region_gnn=region_on)
        pred, su, sr = model(x, xyz, area, rate, mask_n, mask_t, lengths, rho=rho)
        if pred.shape != y.shape:
            raise SystemExit(f"self-test shape {tuple(pred.shape)}")
        loss = mv.trial_balanced_mse(pred, y, mask_t) + 0.01 * su + 0.01 * sr
        loss.backward()
        g = sum(float(p.grad.abs().sum()) for p in model.parameters() if p.grad is not None)
        if g <= 0:
            raise SystemExit(f"self-test: no gradient region={region_on}")
    wc = window_corr(x, mask_n, mask_t, width=4)
    if wc.shape != (bsz, n, n):
        raise SystemExit(f"self-test window corr {tuple(wc.shape)}")
    cmap, names = coarse_id_table({"<unk>": 0, "MOp-ul": 1, "MOp-ll": 2, "MOs": 3}, 4)
    if names != ["unk", "MOp", "MOs"] or cmap != [0, 1, 1, 2]:
        raise SystemExit(f"self-test coarse map {cmap} {names}")
    log(f"self-test ok  grad={g:.4g}")
    return True


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--targets", nargs="*", default=list(TARGETS), choices=list(TARGETS))
    p.add_argument("--tasks", nargs="*", default=list(TASKS), choices=list(TASKS))
    p.add_argument("--protocol", default="cv", choices=("cv",))
    p.add_argument("--paw-only", action="store_true")
    p.add_argument("--pilot", action="store_true")
    p.add_argument("--loso-folds", type=int, default=None)
    p.add_argument("--lomo-folds", type=int, default=None)
    p.add_argument("--interp-folds", type=int, default=None)
    p.add_argument("--fold-seed", type=int, default=0)
    p.add_argument("--region-gnn", action="store_true", help="Outer directed region GNN (off by default on MOp/MOs)")
    p.add_argument("--func-window", action="store_true", help="Causal rolling correlation instead of train-session ρ")
    p.add_argument("--k-neighbors", type=int, default=K_NEIGHBORS)
    p.add_argument("--unit-layers", type=int, default=UNIT_LAYERS)
    p.add_argument("--lambda-unit", type=float, default=0.0)
    p.add_argument("--lambda-region", type=float, default=0.0)
    p.add_argument("--self-test", action="store_true")
    slurm_utils.add_common_args(p)
    args = p.parse_args()
    if args.paw_only:
        args.targets = ["paw_speed"]
    grid_kw = dict(
        targets=args.targets,
        tasks=args.tasks,
        pilot=args.pilot,
        loso_folds=args.loso_folds,
        lomo_folds=args.lomo_folds,
        interp_folds=args.interp_folds,
        fold_seed=args.fold_seed,
        region_gnn=args.region_gnn,
        func_window=args.func_window,
    )
    if args.list_jobs:
        slurm_utils.print_jobs(job_grid(**grid_kw))
        raise SystemExit(0)
    if args.self_test:
        run_self_test()
        raise SystemExit(0)
    slurm_utils.set_device(args.device, probe="gru")
    if args.aggregate:
        aggregate_all()
        raise SystemExit(0)
    proc_kw = dict(
        region_gnn=args.region_gnn,
        func_window=args.func_window,
        k=max(1, int(args.k_neighbors)),
        unit_layers=max(1, int(args.unit_layers)),
        lam_u=args.lambda_unit,
        lam_r=args.lambda_region,
        pilot=args.pilot,
        loso_folds=args.loso_folds,
        lomo_folds=args.lomo_folds,
        interp_folds=args.interp_folds,
        fold_seed=args.fold_seed,
    )
    job = slurm_utils.resolve_job_index(args.job)
    trials, session_meta = mv.load_corpus()
    if job is None:
        configs = job_grid(trials=trials, **grid_kw)
        for i, cfg in enumerate(configs):
            process(
                target=cfg["target"],
                task=cfg["task"],
                fold=cfg["fold"],
                fold_id=cfg["fold_id"],
                trials=trials,
                session_meta=session_meta,
                make_plots=False,
                job_index=i,
                **proc_kw,
            )
        aggregate_all()
    else:
        cfg = slurm_utils.pick_config(job_grid(trials=trials, **grid_kw), job)
        process(
            target=cfg["target"],
            task=cfg["task"],
            fold=cfg["fold"],
            fold_id=cfg["fold_id"],
            trials=trials,
            session_meta=session_meta,
            make_plots=False,
            job_index=job,
            **proc_kw,
        )
