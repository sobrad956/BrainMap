"""Formulation C: progressive trial–session–mouse hierarchy.

Same corpus, GRU trial encoder, trial-balanced MSE, and CV tasks as Modelv1
(interpolation, LOSO, LOMO). The encoder is the current attn-A1 GRU; what
changes is that behavior is decoded at three nested coordinate systems:

    X_ms  --E_ms^trial-->  H^(1)  --A_ms^session-->  H^(2)  --A_m^mouse-->  H^(3)
                              |                         |                      |
                           D_ms^(1)                  D_m^(2)                D^(3)

Stage 1 is within a session (unit identities are stable; no cross-session
correspondence). Stage 2 forces a mouse-shared decoder through a session
adapter. Stage 3 forces one global decoder through a mouse adapter.

Shared decoding without an explicit alignment term is the required baseline
(`--lambda-session 0 --lambda-mouse 0`). Optional Gaussian pair alignment on
[y, Δy] can be turned on. After the three stages, a short joint fine-tune
keeps the auxiliary heads so information loss is still measurable.

Unseen identities (LOSO / LOMO) use a shared fallback encoder plus identity
adapters. `--calibrate-frac` optionally fits the missing encoder/adapters on
a support split of the held-out session/mouse.

Local:
  python heirarchical.py --self-test
  python heirarchical.py --pilot --list-jobs
  python heirarchical.py --job 0 --device cpu
  python heirarchical.py --aggregate

Slurm:
  bash slurm/submit.sh heirarchical.py --pilot
  bash slurm/submit.sh heirarchical.py --protocol cv --pilot --paw-only
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

mv.MODEL_ROOT = ROOT / "Heirarchical"

POOL = "attn"
ABLATION = "a1"
TASKS = ("trial_repeat", "session_loso", "mouse_lomo")
TARGETS = ("wheel_speed", "paw_speed")
STAGES = ("1", "2", "3", "joint")
PILOT_LOSO = 5
PILOT_LOMO = 5
PILOT_INTERP = 1
STAGE_EPOCHS = 20
JOINT_EPOCHS = 15
LOWER_LR = 0.1
ALIGN_SIGMA = 0.5
ALIGN_CAP = 256
CALIBRATE_EPOCHS = 12
MIX_P = 0.25


def log(msg):
    mv.log(msg)


class ResidualAdapter(nn.Module):
    """Starts as identity: last layer is zero-initialized."""

    def __init__(self, d, dropout=mv.DROPOUT):
        super().__init__()
        self.norm = nn.LayerNorm(d)
        self.fc1 = nn.Linear(d, d)
        self.fc2 = nn.Linear(d, d)
        self.drop = nn.Dropout(dropout)
        nn.init.zeros_(self.fc2.weight)
        nn.init.zeros_(self.fc2.bias)

    def forward(self, h):
        x = self.drop(torch.relu(self.fc1(self.norm(h))))
        return h + self.fc2(x)


class SessionEncoder(nn.Module):
    """E_ms^trial after a shared unit MLP: attn pool + causal GRU → (B, T, d)."""

    def __init__(self, d=mv.D_MODEL, hidden=mv.RNN_HIDDEN):
        super().__init__()
        self.pool = mv.AttnPool(d)
        self.rnn = nn.GRU(d, hidden, num_layers=1, batch_first=True)
        self.out_dim = hidden

    def forward(self, e, mask_n, mask_t, lengths):
        h = self.pool(e, mask_n) * mask_t.unsqueeze(-1)
        packed = pack_padded_sequence(h, lengths.cpu(), batch_first=True, enforce_sorted=False)
        out, _ = self.rnn(packed)
        out, _ = pad_packed_sequence(out, batch_first=True, total_length=e.shape[2])
        return out * mask_t.unsqueeze(-1).to(out.dtype)


class HierarchicalGRU(nn.Module):
    """Formulation C with a Modelv1-style attn-A1 GRU trial encoder."""

    def __init__(self, n_areas, eids, mice, shared_encoder=False, use_anatomy=False):
        super().__init__()
        self.shared_encoder = bool(shared_encoder)
        self.use_anatomy = bool(use_anatomy)
        self.id_eids = list(eids)
        self.id_mice = list(mice)
        d = mv.RNN_HIDDEN
        self.unit = mv.UnitEncoder(n_areas, use_anatomy=self.use_anatomy, extra_layer=False)
        self.fallback_enc = SessionEncoder()
        self.session_enc = nn.ModuleDict()
        self.session_adapt = nn.ModuleDict()
        self.dec1 = nn.ModuleDict()
        self.mouse_adapt = nn.ModuleDict()
        self.dec2 = nn.ModuleDict()
        self.dec1_shared = mv.Head(d)
        self.dec2_shared = mv.Head(d)
        self.dec3 = mv.Head(d)
        self.session_adapt_unk = ResidualAdapter(d)
        self.mouse_adapt_unk = ResidualAdapter(d)
        if self.shared_encoder:
            self.shared_enc = SessionEncoder()
        else:
            self.shared_enc = None
            for eid in eids:
                k = _key(eid)
                self.session_enc[k] = SessionEncoder()
        for eid in eids:
            k = _key(eid)
            self.session_adapt[k] = ResidualAdapter(d)
            self.dec1[k] = mv.Head(d)
        for mouse in mice:
            k = _key(mouse)
            self.mouse_adapt[k] = ResidualAdapter(d)
            self.dec2[k] = mv.Head(d)

    def encode_units(self, x, xyz, area, rate, mask_n, unit_drop=0.0):
        if self.training and unit_drop > 0:
            mask_n = mv.apply_unit_dropout(mask_n, unit_drop)
        return self.unit(x, xyz, area, rate, mask_n), mask_n

    def _enc_for(self, eid, use_fallback=False):
        if self.shared_encoder:
            return self.shared_enc
        k = _key(eid)
        if use_fallback or k not in self.session_enc:
            return self.fallback_enc
        return self.session_enc[k]

    def trial_latent(self, e, mask_n, mask_t, lengths, eids):
        out = e.new_zeros(e.size(0), e.size(2), mv.RNN_HIDDEN)
        mix = self.training and not self.shared_encoder
        for eid in dict.fromkeys(eids):
            idx_cpu = _index(eids, eid)
            idx = idx_cpu.to(e.device)
            use_fb = mix and float(torch.rand((), device=e.device)) < MIX_P
            enc = self._enc_for(eid, use_fallback=use_fb)
            out[idx] = enc(e[idx], mask_n[idx], mask_t[idx], lengths[idx_cpu])
        return out

    def _map(self, table, unk, keys, h):
        out = torch.empty_like(h)
        mix = self.training
        for key in dict.fromkeys(keys):
            idx = _index(keys, key, h.device)
            k = _key(key)
            use_unk = mix and float(torch.rand((), device=h.device)) < MIX_P
            mod = unk if use_unk or k not in table else table[k]
            out[idx] = mod(h[idx])
        return out

    def _head(self, table, shared, keys, h):
        out = h.new_zeros(h.size(0), h.size(1), 1)
        mix = self.training
        for key in dict.fromkeys(keys):
            idx = _index(keys, key, h.device)
            k = _key(key)
            use_shared = mix and float(torch.rand((), device=h.device)) < MIX_P
            head = shared if use_shared or k not in table else table[k]
            out[idx] = head(h[idx])
        return out

    def forward(self, x, xyz, area, rate, mask_n, mask_t, lengths, eids, mice, unit_drop=0.0):
        e, mask_n = self.encode_units(x, xyz, area, rate, mask_n, unit_drop=unit_drop)
        h1 = self.trial_latent(e, mask_n, mask_t, lengths, eids)
        y1 = self._head(self.dec1, self.dec1_shared, eids, h1)
        h2 = self._map(self.session_adapt, self.session_adapt_unk, eids, h1)
        y2 = self._head(self.dec2, self.dec2_shared, mice, h2)
        h3 = self._map(self.mouse_adapt, self.mouse_adapt_unk, mice, h2)
        y3 = self.dec3(h3)
        return y1, y2, y3, h1, h2, h3

    def ensure_identity(self, eids, mice):
        """Identity-init adapters/heads for ids that appear only at eval/calibrate."""
        d = mv.RNN_HIDDEN
        ref = next(self.parameters())
        for eid in eids:
            k = _key(eid)
            if not self.shared_encoder and k not in self.session_enc:
                mod = SessionEncoder().to(device=ref.device, dtype=ref.dtype)
                mod.load_state_dict(self.fallback_enc.state_dict())
                self.session_enc[k] = mod
            if k not in self.session_adapt:
                mod = ResidualAdapter(d).to(device=ref.device, dtype=ref.dtype)
                mod.load_state_dict(self.session_adapt_unk.state_dict())
                self.session_adapt[k] = mod
            if k not in self.dec1:
                mod = mv.Head(d).to(device=ref.device, dtype=ref.dtype)
                mod.load_state_dict(self.dec1_shared.state_dict())
                self.dec1[k] = mod
            if eid not in self.id_eids:
                self.id_eids.append(eid)
        for mouse in mice:
            k = _key(mouse)
            if k not in self.mouse_adapt:
                mod = ResidualAdapter(d).to(device=ref.device, dtype=ref.dtype)
                mod.load_state_dict(self.mouse_adapt_unk.state_dict())
                self.mouse_adapt[k] = mod
            if k not in self.dec2:
                mod = mv.Head(d).to(device=ref.device, dtype=ref.dtype)
                mod.load_state_dict(self.dec2_shared.state_dict())
                self.dec2[k] = mod
            if mouse not in self.id_mice:
                self.id_mice.append(mouse)


def _key(name):
    return "".join(ch if ch.isalnum() or ch in "-_" else "_" for ch in str(name))


def _index(keys, value, device=None):
    sel = [i for i, k in enumerate(keys) if k == value]
    t = torch.as_tensor(sel, dtype=torch.long)
    return t if device is None else t.to(device)


class HierDS(mv.TrialDS):
    def __getitem__(self, j):
        item = super().__getitem__(j)
        t = self.trials[int(self.idx[j])]
        return (*item, str(t["eid"]), str(t["mouse_id"]))


def hier_collate(batch):
    core = [b[:-2] for b in batch]
    eids = [b[-2] for b in batch]
    mice = [b[-1] for b in batch]
    out = mv.collate(core)
    out["eids"] = eids
    out["mice"] = mice
    return out


def ids_of(trials, idx):
    eids, mice = [], []
    seen_e, seen_m = set(), set()
    for i in np.asarray(idx, dtype=int):
        t = trials[int(i)]
        eid, mouse = str(t["eid"]), str(t["mouse_id"])
        if eid not in seen_e:
            seen_e.add(eid)
            eids.append(eid)
        if mouse not in seen_m:
            seen_m.add(mouse)
            mice.append(mouse)
    return eids, mice


def behavior_pairs(y, h, mask, eids=None, mice=None, group="session", sigma=ALIGN_SIGMA, cap=ALIGN_CAP):
    """w_ab = exp(-||b_a-b_b||^2 / 2σ^2) on [y, Δy]; pairs from different sessions/mice."""
    y0 = y[..., 0]
    dy = torch.zeros_like(y0)
    dy[:, 1:] = y0[:, 1:] - y0[:, :-1]
    desc = torch.stack([y0, dy], dim=-1)
    bsz, n_t, d_h = h.shape
    flat_h, flat_b, grp = [], [], []
    labels = eids if group == "session" else mice
    for i in range(bsz):
        m = mask[i]
        if not bool(m.any()):
            continue
        flat_h.append(h[i, m])
        flat_b.append(desc[i, m])
        n = int(m.sum().item())
        gid = hash(_key(labels[i])) % (2**31)
        grp.append(h.new_full((n,), gid))
    if not flat_h:
        return h.new_zeros(())
    hh = torch.cat(flat_h, dim=0)
    bb = torch.cat(flat_b, dim=0)
    gg = torch.cat(grp, dim=0)
    n = hh.size(0)
    if n < 4:
        return h.new_zeros(())
    if n > cap:
        pick = torch.randperm(n, device=h.device)[:cap]
        hh, bb, gg = hh[pick], bb[pick], gg[pick]
        n = cap
    dist = torch.cdist(bb, bb, p=2).pow(2)
    w = torch.exp(-dist / (2.0 * max(float(sigma), 1e-3) ** 2))
    diff = gg.unsqueeze(0) != gg.unsqueeze(1)
    eye = torch.eye(n, dtype=torch.bool, device=h.device)
    w = w.masked_fill(~diff | eye, 0.0)
    denom = w.sum().clamp(min=1.0)
    dh = torch.cdist(hh, hh, p=2).pow(2)
    return (w * dh).sum() / denom


def stage_loss(yhat, y, mask, h=None, eids=None, mice=None, lam=0.0, group="session"):
    beh = mv.trial_balanced_mse(yhat, y, mask)
    if lam <= 0 or h is None:
        return beh, beh, yhat.new_zeros(())
    align = behavior_pairs(y, h, mask, eids=eids, mice=mice, group=group)
    return beh + lam * align, beh, align


def stage_param_tags(stage, freeze_lower):
    if stage == "1":
        return ["unit", "session_enc", "fallback_enc", "shared_enc", "dec1", "dec1_shared"]
    if stage == "2":
        tags = ["session_adapt", "dec2", "dec2_shared"]
        if not freeze_lower:
            tags += ["unit", "session_enc", "fallback_enc", "shared_enc", "dec1", "dec1_shared"]
        return tags
    if stage == "3":
        tags = ["mouse_adapt", "dec3"]
        if not freeze_lower:
            tags += [
                "unit",
                "session_enc",
                "fallback_enc",
                "shared_enc",
                "session_adapt",
                "dec1",
                "dec2",
                "dec1_shared",
                "dec2_shared",
            ]
        return tags
    return ["unit", "session", "mouse", "fallback", "shared_enc", "dec"]


def run_epoch(model, loader, opt, dev, train, stage, lam_s, lam_m, betas):
    model.train(train)
    tot = {k: 0.0 for k in ("loss", "b1", "b2", "b3", "a_s", "a_m")}
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
        eids, mice = batch["eids"], batch["mice"]
        if train:
            opt.zero_grad(set_to_none=True)
        y1, y2, y3, h1, h2, h3 = model(
            x,
            xyz,
            area,
            rate,
            mask_n,
            mask_t,
            lengths,
            eids,
            mice,
            unit_drop=mv.UNIT_DROPOUT if train else 0.0,
        )
        l1, b1, _ = stage_loss(y1, y, mask_loss)
        l2, b2, a_s = stage_loss(y2, y, mask_loss, h2, eids, mice, lam=lam_s, group="session")
        l3, b3, a_m = stage_loss(y3, y, mask_loss, h3, eids, mice, lam=lam_m, group="mouse")
        if stage == "1":
            loss = l1
        elif stage == "2":
            loss = l2
        elif stage == "3":
            loss = l3
        else:
            loss = betas[0] * l1 + betas[1] * l2 + betas[2] * l3
        if train:
            loss.backward()
            nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], 1.0)
            opt.step()
        bs = int(x.size(0))
        tot["loss"] += float(loss.item()) * bs
        tot["b1"] += float(b1.item()) * bs
        tot["b2"] += float(b2.item()) * bs
        tot["b3"] += float(b3.item()) * bs
        tot["a_s"] += float(a_s.item()) * bs
        tot["a_m"] += float(a_m.item()) * bs
        n += bs
    d = max(n, 1)
    return {k: v / d for k, v in tot.items()}


def make_loader(trials, idx, y_mean, y_std, stats, bin_key, shuffle):
    ds = HierDS(trials, idx, y_mean, y_std, stats, bin_key=bin_key)
    return DataLoader(ds, batch_size=mv.BATCH, shuffle=shuffle, collate_fn=hier_collate)


def build_optim(model, lr, freeze_lower, stage):
    tags = stage_param_tags(stage, freeze_lower)
    for n, p in model.named_parameters():
        p.requires_grad = any(tag in n for tag in tags)
    params = [p for p in model.parameters() if p.requires_grad]
    if not params:
        raise RuntimeError(f"no trainable params for stage {stage}")
    if stage == "joint":
        used = set()
        groups = []
        for tag_list, g_lr in (
            (("unit", "session_enc", "fallback", "shared_enc", "dec1"), lr * LOWER_LR),
            (("session_adapt", "dec2"), lr * 0.3),
            (("mouse_adapt", "dec3"), lr),
        ):
            params = []
            for n, p in model.named_parameters():
                if not p.requires_grad or id(p) in used:
                    continue
                if any(tag in n for tag in tag_list):
                    params.append(p)
                    used.add(id(p))
            if params:
                groups.append({"params": params, "lr": g_lr})
        leftover = [p for n, p in model.named_parameters() if p.requires_grad and id(p) not in used]
        if leftover:
            groups.append({"params": leftover, "lr": lr * 0.3})
        return torch.optim.AdamW(groups, weight_decay=mv.WEIGHT_DECAY)
    return torch.optim.AdamW(params, lr=lr, weight_decay=mv.WEIGHT_DECAY)


def cache_dir(task, fold_id, target, shared, lam_s=0.0, lam_m=0.0, calibrate_frac=0.0):
    mv.configure_out(target, "cv")
    d = mv.CACHE / str(task)
    if fold_id:
        safe = "".join(ch if ch.isalnum() or ch in "-_." else "_" for ch in str(fold_id))
        d = d / safe
    tag = "attn_a1_hier_shared" if shared else "attn_a1_hier"
    if lam_s or lam_m:
        tag += f"_ls{lam_s:g}_lm{lam_m:g}"
    if calibrate_frac:
        tag += f"_cal{calibrate_frac:g}"
    d = d / tag
    d.mkdir(parents=True, exist_ok=True)
    return d


def train_hierarchy(
    trials,
    tr_idx,
    va_idx,
    test_idx,
    y_mean,
    y_std,
    stats,
    task,
    fold_id,
    target,
    bin_key,
    stages,
    shared_encoder,
    freeze_lower,
    lam_s,
    lam_m,
    betas,
    calibrate_frac,
):
    eids, mice = ids_of(trials, tr_idx)
    tag = f"hier_{'shared' if shared_encoder else 'persess'}"
    cdir = cache_dir(task, fold_id, target, shared_encoder, lam_s, lam_m, calibrate_frac)
    ckpt = cdir / "model.pt"
    hist_path = cdir / "history.json"
    if ckpt.exists() and hist_path.exists():
        hist = json.loads(hist_path.read_text())
        if hist.get("finished"):
            log(f"  cache hit {task}/{fold_id}/{tag}")
            blob = torch.load(ckpt, map_location="cpu", weights_only=False)
            eval_idx = np.asarray(hist.get("eval_idx", test_idx), dtype=int)
            return blob, hist, eval_idx
    dev = mv.device_of()
    model = HierarchicalGRU(
        stats["n_areas"],
        eids,
        mice,
        shared_encoder=shared_encoder,
        use_anatomy=False,
    ).to(dev)
    log(f"  {tag} params={mv.n_params(model)}  device={dev}  sessions={len(eids)} mice={len(mice)}")
    tr_ld = make_loader(trials, tr_idx, y_mean, y_std, stats, bin_key, True)
    va_ld = make_loader(trials, va_idx, y_mean, y_std, stats, bin_key, False)
    hist = {"stages": {}, "n_params": mv.n_params(model), "finished": False, "best_epoch": 0}
    betas = tuple(float(b) for b in betas)

    def snapshot():
        return {
            "state_dict": {k: v.detach().cpu().clone() for k, v in model.state_dict().items()},
            "y_mean": y_mean,
            "y_std": y_std,
            "n_areas": stats["n_areas"],
            "eids": list(model.id_eids),
            "mice": list(model.id_mice),
            "shared_encoder": shared_encoder,
            "stats": {
                "vocab": stats["vocab"],
                "n_areas": stats["n_areas"],
                "xyz_mean": stats["xyz_mean"],
                "xyz_std": stats["xyz_std"],
                "rates": stats["rates"],
            },
        }

    for stage in stages:
        n_ep = JOINT_EPOCHS if stage == "joint" else STAGE_EPOCHS
        lr = mv.LR * (LOWER_LR if stage == "joint" else 1.0)
        opt = build_optim(model, lr, freeze_lower, stage)
        best_val, best_state, bad = math.inf, None, 0
        rec = {"train": [], "val": []}
        log(f"  stage {stage}  trainable={mv.n_params(model)}  lr={lr:g}")
        for epoch in range(1, n_ep + 1):
            tr = run_epoch(model, tr_ld, opt, dev, True, stage, lam_s, lam_m, betas)
            with torch.no_grad():
                va = run_epoch(model, va_ld, None, dev, False, stage, lam_s, lam_m, betas)
            rec["train"].append(tr)
            rec["val"].append(va)
            log(
                f"    {tag} {stage} {epoch:02d}  train={tr['loss']:.4f}  val={va['loss']:.4f}  "
                f"b1={va['b1']:.4f} b2={va['b2']:.4f} b3={va['b3']:.4f}"
            )
            if va["loss"] + 1e-5 < best_val:
                best_val = va["loss"]
                best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
                bad = 0
                hist["best_epoch"] = epoch
            else:
                bad += 1
            slurm_utils.write_status(
                state=f"stage_{stage}",
                epoch=epoch,
                train=tr["loss"],
                val=va["loss"],
                best_val=best_val,
            )
            if bad >= mv.PATIENCE:
                log(f"    early stop stage {stage} at epoch {epoch}")
                break
        if best_state is not None:
            model.load_state_dict(best_state)
        rec["best_val"] = best_val
        hist["stages"][stage] = rec

    eval_idx = np.asarray(test_idx, dtype=int)
    if calibrate_frac > 0 and task in ("session_loso", "mouse_lomo"):
        model, eval_idx = calibrate(
            model,
            trials,
            test_idx,
            y_mean,
            y_std,
            stats,
            bin_key,
            calibrate_frac,
            lam_s,
            lam_m,
            betas,
            task,
        )

    hist["eval_idx"] = [int(i) for i in np.asarray(eval_idx, dtype=int)]
    hist["finished"] = True
    blob = snapshot()
    torch.save(blob, ckpt)
    mv.save_json(hist_path, hist)
    return blob, hist, eval_idx


def calibrate(model, trials, test_idx, y_mean, y_std, stats, bin_key, frac, lam_s, lam_m, betas, task):
    """Fit missing encoder/adapters on a support split of the held-out identity."""
    rng = np.random.default_rng(mv.SEED + 91)
    idx = np.asarray(test_idx, dtype=int)
    rng.shuffle(idx)
    n_sup = max(4, int(round(float(frac) * len(idx))))
    if n_sup >= len(idx) - 1:
        log("  calibrate: too few held-out trials, skip")
        return model, idx
    sup, query = idx[:n_sup], idx[n_sup:]
    eids, mice = ids_of(trials, sup)
    model.ensure_identity(eids, mice)
    model.to(mv.device_of())
    new_e = {_key(e) for e in eids}
    new_m = {_key(m) for m in mice}

    def _new_id_param(name):
        for k in new_e:
            if f"session_enc.{k}." in name or f"session_adapt.{k}." in name or f"dec1.{k}." in name:
                return True
        if task == "mouse_lomo":
            for k in new_m:
                if f"mouse_adapt.{k}." in name or f"dec2.{k}." in name:
                    return True
        return False

    for n, p in model.named_parameters():
        p.requires_grad = _new_id_param(n)
    params = [p for p in model.parameters() if p.requires_grad]
    if not params:
        log("  calibrate: no new identity parameters")
        return model, query
    opt = torch.optim.AdamW(params, lr=mv.LR, weight_decay=mv.WEIGHT_DECAY)
    ld = make_loader(trials, sup, y_mean, y_std, stats, bin_key, True)
    log(f"  calibrate {task}  n_sup={len(sup)} n_query={len(query)}  ids={eids}/{mice}")
    for epoch in range(1, CALIBRATE_EPOCHS + 1):
        rec = run_epoch(model, ld, opt, mv.device_of(), True, "joint", lam_s, lam_m, betas)
        if epoch == 1 or epoch == CALIBRATE_EPOCHS:
            log(f"    calib {epoch:02d}  loss={rec['loss']:.4f}  b3={rec['b3']:.4f}")
    return model, query


def load_model(blob, stats):
    stats = {
        "vocab": stats["vocab"],
        "n_areas": int(stats["n_areas"]),
        "xyz_mean": np.asarray(stats["xyz_mean"], dtype=np.float32),
        "xyz_std": np.asarray(stats["xyz_std"], dtype=np.float32),
        "rates": {k: np.asarray(v, dtype=np.float32) for k, v in stats["rates"].items()},
    }
    model = HierarchicalGRU(
        blob["n_areas"],
        blob.get("eids", []),
        blob.get("mice", []),
        shared_encoder=bool(blob.get("shared_encoder", False)),
        use_anatomy=False,
    )
    missing, unexpected = model.load_state_dict(blob["state_dict"], strict=False)
    if missing or unexpected:
        log(f"  state_dict missing={len(missing)} unexpected={len(unexpected)}")
    return model, stats


def predict_stages(blob, trials, idx, stats):
    stats_b = blob.get("stats", stats)
    model, stats = load_model(blob, stats_b)
    # Zero-shot held-out identities use fallback encoder / unk adapters / shared heads.
    # Calibration (if requested) already installed identity modules on the loaded weights.
    dev = mv.device_of()
    model.to(dev)
    model.eval()
    ds = HierDS(trials, idx, blob["y_mean"], blob["y_std"], stats)
    ld = DataLoader(ds, batch_size=mv.BATCH, shuffle=False, collate_fn=hier_collate)
    recs = {1: [], 2: [], 3: []}
    with torch.no_grad():
        for batch in ld:
            y1, y2, y3, _, _, _ = model(
                batch["x"].to(dev),
                batch["xyz"].to(dev),
                batch["area"].to(dev),
                batch["rate"].to(dev),
                batch["mask_n"].to(dev),
                batch["mask_t"].to(dev),
                batch["lengths"],
                batch["eids"],
                batch["mice"],
                unit_drop=0.0,
            )
            preds = {1: y1.cpu().numpy(), 2: y2.cpu().numpy(), 3: y3.cpu().numpy()}
            y_z = batch["y"].numpy()
            mt = batch["mask_t"].numpy()
            lengths = batch["lengths"].numpy()
            for b in range(y_z.shape[0]):
                T = int(lengths[b])
                y = y_z[b, :T] * blob["y_std"] + blob["y_mean"]
                valid = mt[b, :T]
                y = y.astype(np.float32)
                y[~valid] = np.nan
                for k, pred in preds.items():
                    p = pred[b, :T] * blob["y_std"] + blob["y_mean"]
                    p = p.astype(np.float32)
                    p[~valid] = np.nan
                    recs[k].append({"pred": p, "y": y})
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
    shared_encoder=False,
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
                    "pool": POOL,
                    "ablation": ABLATION,
                    "fold": int(spec["fold"]),
                    "fold_id": spec["fold_id"],
                    "shared_encoder": bool(shared_encoder),
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
    shared_encoder=False,
    stages=None,
    freeze_lower=True,
    lam_s=0.0,
    lam_m=0.0,
    betas=(1.0, 1.0, 1.0),
    calibrate_frac=0.0,
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
    stages = list(stages or STAGES)
    log(
        f"device={mv.device_of()}  target={target}  task={task}  "
        f"Formulation C attn A1 GRU  shared_enc={int(shared_encoder)}  stages={stages}"
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
        tag = f"{target}_{task}_{fid}_attn_a1_hier"
        idx = job_index if job_index is not None else n_job
        jdir = slurm_utils.begin_job(
            mv.OUT,
            idx,
            tag,
            task=task,
            pool=POOL,
            ablation=ABLATION,
            target=target,
            protocol="cv",
            fold=fold_i,
            fold_id=fid,
            formulation="C",
        )
        blob, hist, eval_idx = train_hierarchy(
            trials,
            tr_idx,
            va_idx,
            test_idx,
            y_mean,
            y_std,
            stats,
            task,
            fid,
            target,
            bin_key="train_bins" if bin_mode else None,
            stages=stages,
            shared_encoder=shared_encoder,
            freeze_lower=freeze_lower,
            lam_s=lam_s,
            lam_m=lam_m,
            betas=betas,
            calibrate_frac=calibrate_frac,
        )
        recs_full = predict_stages(blob, trials, eval_idx, blob.get("stats", stats))
        min_bins = 2 if bin_mode else 8
        recs = recs_full
        if bin_mode:
            recs = {k: mv.restrict_recs(v, trials, eval_idx, "test_bins") for k, v in recs_full.items()}
        these = []
        for stage, rr in recs.items():
            sc = mv.score_recs(rr, target=target, min_trial_bins=min_bins)[target]
            hold = mv.save_holdout_pack(
                jdir,
                recs_full[stage],
                trials,
                eval_idx,
                target=target,
                min_trial_bins=min_bins,
                head="trace",
                recs_score=rr,
                stage=stage,
                job_meta={
                    "protocol": "cv",
                    "task": task,
                    "fold": fold_i,
                    "fold_id": fid,
                    "target": target,
                    "formulation": "C",
                    "stage": int(stage),
                    "bin_mode": bin_mode or "",
                },
            )
            row = {
                "protocol": "cv",
                "task": task,
                "pool": POOL,
                "target": target,
                "ablation": ABLATION,
                "formulation": "C",
                "stage": int(stage),
                "shared_encoder": int(shared_encoder),
                "fold": fold_i,
                "fold_id": fid,
                "bin_mode": bin_mode or "",
                "calibrate_frac": float(calibrate_frac),
                "n_eval": int(len(eval_idx)),
                **sc,
                **hold,
            }
            rows.append(row)
            these.append(row)
            log(
                f"   stage {stage}  R²={sc['r2']:+.3f}  mean_trial={sc['mean_trial_r2']:+.3f}  "
                f"median_trial={sc['median_trial_r2']:+.3f}  n={sc['n_trials']}"
            )
        pd.DataFrame(these).to_csv(jdir / "scores.csv", index=False)
        slurm_utils.write_status(state="done", n_rows=len(these), best_epoch=hist.get("best_epoch"))
        n_job += 1
    if make_plots and rows:
        pd.DataFrame(rows).to_csv(mv.OUT / "scores.csv", index=False)
        write_report(rows, mv.OUT / "REPORT.md")
        log("wrote " + str(mv.OUT))
    return rows


def write_report(rows, path):
    df = pd.DataFrame(rows)
    lines = ["# Formulation C (hierarchical GRU)", ""]
    if df.empty:
        Path(path).write_text("\n".join(lines))
        return
    keys = ["task", "target", "stage"]
    lines.append("| task | target | stage | n | concat R² | mean trial R² |")
    lines.append("|---|---|---:|---:|---:|---:|")
    for name, sub in df.groupby(keys, dropna=False):
        rec = dict(zip(keys, name if isinstance(name, tuple) else (name,)))
        lines.append(
            f"| {rec['task']} | {rec['target']} | {rec['stage']} | {len(sub)} | "
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
        parts = []
        for p in sorted(jobs.glob("*/scores.csv")):
            parts.append(pd.read_csv(p))
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
    print(df.groupby(["task", "target", "stage"])[["r2", "mean_trial_r2"]].mean().to_string())


def run_self_test():
    torch.manual_seed(0)
    bsz, n, t = 4, 6, 16
    x = torch.randn(bsz, n, t)
    xyz = torch.zeros(bsz, n, 3)
    area = torch.zeros(bsz, n, dtype=torch.long)
    rate = torch.zeros(bsz, n)
    mask_n = torch.ones(bsz, n, dtype=torch.bool)
    mask_t = torch.ones(bsz, t, dtype=torch.bool)
    lengths = torch.full((bsz,), t)
    y = torch.randn(bsz, t, 1)
    eids = ["s0", "s0", "s1", "s1"]
    mice = ["m0", "m0", "m0", "m1"]
    model = HierarchicalGRU(2, ["s0", "s1"], ["m0", "m1"], shared_encoder=False)
    y1, y2, y3, h1, h2, h3 = model(x, xyz, area, rate, mask_n, mask_t, lengths, eids, mice)
    if y1.shape != y.shape or h1.shape[-1] != mv.RNN_HIDDEN:
        raise SystemExit(f"self-test shape {tuple(y1.shape)} {tuple(h1.shape)}")
    loss = mv.trial_balanced_mse(y1, y, mask_t) + mv.trial_balanced_mse(y2, y, mask_t) + mv.trial_balanced_mse(y3, y, mask_t)
    loss.backward()
    g = sum(float(p.grad.abs().sum()) for p in model.parameters() if p.grad is not None)
    if g <= 0:
        raise SystemExit("self-test: no gradient")
    align = behavior_pairs(y, h2, mask_t, eids=eids, mice=mice, group="session")
    if not torch.isfinite(align):
        raise SystemExit("self-test: align not finite")
    log(f"self-test ok  grad={g:.4g}  align={float(align):.4g}")
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
    p.add_argument("--shared-encoder", action="store_true", help="One GRU for all sessions (Modelv1-like); adapters do the identity work")
    p.add_argument("--stages", nargs="*", default=list(STAGES), choices=list(STAGES))
    p.add_argument("--no-freeze-lower", action="store_true")
    p.add_argument("--lambda-session", type=float, default=0.0)
    p.add_argument("--lambda-mouse", type=float, default=0.0)
    p.add_argument("--beta1", type=float, default=1.0)
    p.add_argument("--beta2", type=float, default=1.0)
    p.add_argument("--beta3", type=float, default=1.0)
    p.add_argument("--calibrate-frac", type=float, default=0.0)
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
        shared_encoder=args.shared_encoder,
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
        shared_encoder=args.shared_encoder,
        stages=args.stages,
        freeze_lower=not args.no_freeze_lower,
        lam_s=args.lambda_session,
        lam_m=args.lambda_mouse,
        betas=(args.beta1, args.beta2, args.beta3),
        calibrate_frac=args.calibrate_frac,
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
