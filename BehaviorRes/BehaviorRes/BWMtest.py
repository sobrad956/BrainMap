"""Re-attempt BWM wheel decoding on ModelDataLeftIpsi, plus paw speed targets.

Matches the Brain-Wide Map movement decoder where we can:
    regularized linear regression, nested 5-fold interleaved CV at the
    trial level, 20 ms bins, causal lag W = 10, score = R^2 on
    concatenated held-out bins.

    The paper used Lasso (L1). Unscaled spike counts did not converge
    in a usable time here, so this file uses Ridge (L2) with a
    StandardScaler fit on each training fold. Same nested CV, same
    lag, same R^2.

Deviates where our tensors force it:
    Window is stimOn -> reward (what is stored), not firstMovement
    -0.2 to +1.0 s. Only 184 / 6819 LeftIpsi trials contain that 1.2 s
    window without extrapolation, which we do not do.
    Nulls are trial-shuffled targets (same session), not 100 imposter
    sessions. 5 circular shifts; p = (1 + n_null >= true) / 6.

Populations: MOp/MOs units (the reason this folder exists).
Targets: |wheel velocity|, 2D paw speed, session-normalized paw speed.
"""

from __future__ import annotations

import argparse
import json
import pickle
import re
import sys
import warnings
from datetime import datetime, timezone
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge, RidgeCV
from sklearn.metrics import r2_score
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler

import slurm_utils

ROOT = Path(__file__).resolve().parent
DESKTOP = ROOT.parent
DATA = ROOT / "ModelDataLeftIpsi"
OUT = ROOT / "BWMtest"
SESS = DATA / "sessions"

BINSIZE = 0.02
N_LAGS = 10  # paper W; predictors are lags 0..W inclusive
N_FOLDS = 5
N_RUNS = 1
N_NULL = 5
MIN_TRIALS = 20
MIN_UNITS = 5
MIN_BINS_AFTER_LAG = 3
ALPHAS = np.array([0.1, 1.0, 10.0, 100.0, 1000.0])
TARGETS = ("wheel_speed", "paw_speed", "paw_speed_norm")
TARGET_LABELS = {
    "wheel_speed": "wheel speed |ω|",
    "paw_speed": "paw speed (2D)",
    "paw_speed_norm": "paw speedNormalized",
}


def log(msg):
    print(f"[{datetime.now().strftime('%H:%M:%S')}] {msg}", flush=True)


def session_paths():
    return sorted(p for p in SESS.glob("*.pkl") if not p.name.endswith(".tmp"))


def job_grid():
    return [{"session": p.stem} for p in session_paths()]


def motor_mask(units):
    area = units["brain_area"].astype(str)
    return area.str.startswith("MOp") | area.str.startswith("MOs")


def build_lag_matrix(spikes, n_lags=N_LAGS):
    """Paper build_predictor_matrix: hstack of rolls 0..n_lags, crop first n_lags."""
    mat = np.hstack([np.roll(spikes, i, axis=0) for i in range(n_lags + 1)])
    return mat[n_lags:]


def trial_target(rec, name):
    if name == "wheel_speed":
        y = np.abs(np.asarray(rec.get("wheel_velocity", []), dtype=float))
    elif name == "paw_speed":
        y = np.asarray(rec.get("lp_speed", []), dtype=float)
        if y.size == 0 or not np.isfinite(y).any():
            y = np.asarray(rec.get("dlc_speed", []), dtype=float)
    else:
        y = np.asarray(rec.get("lp_speedNormalized", []), dtype=float)
        if y.size == 0 or not np.isfinite(y).any():
            y = np.asarray(rec.get("speedNormalized", []), dtype=float)
        if y.size == 0 or not np.isfinite(y).any():
            y = np.asarray(rec.get("dlc_speedNormalized", []), dtype=float)
    return y


def trial_state(rec):
    st = rec.get("lp_paw_state")
    if st is None or len(np.asarray(st)) == 0:
        st = rec.get("dlc_paw_state")
    return np.asarray(st, dtype=object) if st is not None else np.array([], dtype=object)


def pack_session(payload, target):
    units = payload["units"]
    m = motor_mask(units)
    motor_idx = np.flatnonzero(m.to_numpy())
    if motor_idx.size < MIN_UNITS:
        return None
    Xs, ys, states, meta = [], [], [], []
    for rec in payload["trials"]:
        spikes = np.asarray(rec.get("spike_counts", []), dtype=float)
        y = trial_target(rec, target)
        if spikes.ndim != 2 or spikes.shape[0] < N_LAGS + MIN_BINS_AFTER_LAG:
            continue
        n = min(spikes.shape[0], y.size)
        if n < N_LAGS + MIN_BINS_AFTER_LAG:
            continue
        spikes = spikes[:n, motor_idx]
        y = y[:n]
        X = build_lag_matrix(spikes)
        y = y[N_LAGS:]
        st = trial_state(rec)
        if st.size >= n:
            st = st[N_LAGS:n]
        else:
            st = np.array(["none"] * y.size, dtype=object)
        finite = np.isfinite(y) & np.isfinite(X).all(axis=1)
        if finite.sum() < MIN_BINS_AFTER_LAG:
            continue
        Xs.append(X[finite].astype(np.float32))
        ys.append(y[finite].astype(np.float32))
        states.append(st[finite] if st.size == finite.size else st[np.where(finite)[0]])
        meta.append(
            {
                "trial_index": rec.get("trial_index"),
                "actionTime": rec.get("actionTime"),
                "n_bins": int(n),
            }
        )
    if len(Xs) < MIN_TRIALS:
        return None
    return {
        "Xs": Xs,
        "ys": ys,
        "states": states,
        "meta": meta,
        "n_units": int(motor_idx.size),
        "n_trials": len(Xs),
    }


def _stack(idxs, Xs, ys):
    X = np.vstack([Xs[i] for i in idxs])
    y = np.concatenate([ys[i] for i in idxs])
    return X, y


def ridge_fit_predict(Xtr, ytr, Xte, alpha):
    scaler = StandardScaler()
    Xtr_s = scaler.fit_transform(Xtr)
    Xte_s = scaler.transform(Xte)
    model = Ridge(alpha=float(alpha), fit_intercept=True, solver="svd")
    model.fit(Xtr_s, ytr)
    return model.predict(Xte_s)


def predict_cv(Xs, ys, alpha, seed=0):
    n = len(Xs)
    kf = KFold(n_splits=min(N_FOLDS, n), shuffle=True, random_state=seed)
    preds = [np.full(ys[i].shape, np.nan, dtype=np.float32) for i in range(n)]
    for train_idx, test_idx in kf.split(np.arange(n)):
        Xtr, ytr = _stack(train_idx, Xs, ys)
        scaler = StandardScaler()
        Xtr_s = scaler.fit_transform(Xtr)
        model = Ridge(alpha=float(alpha), fit_intercept=True, solver="svd")
        model.fit(Xtr_s, ytr)
        for i in test_idx:
            preds[i] = model.predict(scaler.transform(Xs[i])).astype(np.float32)
    y_true = np.concatenate(ys)
    y_hat = np.concatenate(preds)
    m = np.isfinite(y_true) & np.isfinite(y_hat)
    r2 = float(r2_score(y_true[m], y_hat[m])) if m.sum() >= 10 else float("nan")
    return r2, preds


def nested_ridge(Xs, ys, seed=0):
    """Trial-level 5-fold outer CV; RidgeCV tunes alpha on the outer-train bins."""
    n = len(Xs)
    kf = KFold(n_splits=min(N_FOLDS, n), shuffle=True, random_state=seed)
    preds = [np.full(ys[i].shape, np.nan, dtype=np.float32) for i in range(n)]
    best_alphas = []
    for train_idx, test_idx in kf.split(np.arange(n)):
        Xtr, ytr = _stack(train_idx, Xs, ys)
        scaler = StandardScaler()
        Xtr_s = scaler.fit_transform(Xtr)
        model = RidgeCV(alphas=ALPHAS)
        model.fit(Xtr_s, ytr)
        best_alphas.append(float(model.alpha_))
        for i in test_idx:
            Xi = scaler.transform(Xs[i])
            preds[i] = model.predict(Xi).astype(np.float32)
    y_true = np.concatenate(ys)
    y_hat = np.concatenate(preds)
    m = np.isfinite(y_true) & np.isfinite(y_hat)
    r2 = float(r2_score(y_true[m], y_hat[m])) if m.sum() >= 10 else float("nan")
    return r2, preds, best_alphas


def regime_r2(ys, preds, states):
    out = {}
    for name in ("still", "coupled", "decoupled"):
        yt, yh = [], []
        for y, p, st in zip(ys, preds, states):
            mask = st.astype(str) == name
            if mask.size != y.size:
                continue
            yt.append(y[mask])
            yh.append(p[mask])
        if not yt:
            out[name] = float("nan")
            continue
        yt = np.concatenate(yt)
        yh = np.concatenate(yh)
        m = np.isfinite(yt) & np.isfinite(yh)
        out[name] = float(r2_score(yt[m], yh[m])) if m.sum() >= 10 else float("nan")
    return out


def shift_ys(ys, rng):
    """Circular-shift each trial's target so X/y lengths stay matched."""
    out = []
    for y in ys:
        if y.size < 4:
            out.append(y.copy())
            continue
        k = int(rng.integers(1, y.size))
        out.append(np.roll(y, k))
    return out


def decode_session(payload, target, rng):
    packed = pack_session(payload, target)
    if packed is None:
        return None
    r2, preds, alphas = nested_ridge(packed["Xs"], packed["ys"], seed=0)
    nulls = []
    for k in range(N_NULL):
        ys_n = shift_ys(packed["ys"], rng)
        r2_n, _ = predict_cv(packed["Xs"], ys_n, float(np.mean(alphas)), seed=1000 + k)
        nulls.append(r2_n)
    nulls = np.asarray(nulls, dtype=float)
    p = float((1 + np.sum(nulls >= r2)) / (len(nulls) + 1))
    regimes = regime_r2(packed["ys"], preds, packed["states"])
    return {
        "r2": r2,
        "null_median": float(np.nanmedian(nulls)),
        "r2_corrected": float(r2 - np.nanmedian(nulls)),
        "p_shuffle": p,
        "n_trials": packed["n_trials"],
        "n_units": packed["n_units"],
        "best_alpha_mean": float(np.mean(alphas)),
        "regime_r2": regimes,
        "preds": preds,
        "ys": packed["ys"],
        "states": packed["states"],
        "meta": packed["meta"],
        "nulls": nulls.tolist(),
    }


def style_plots():
    plt.rcParams.update(
        {
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "axes.spines.top": False,
            "axes.spines.right": False,
            "font.size": 11,
            "axes.titlesize": 12,
            "axes.labelsize": 11,
        }
    )


def plot_all(rows, examples, out_dir):
    style_plots()
    df = pd.DataFrame(rows)
    colors = {
        "wheel_speed": "#4C6A92",
        "paw_speed": "#C47B3B",
        "paw_speed_norm": "#3B7A57",
    }

    # 1. distributions
    fig, ax = plt.subplots(figsize=(7.2, 4.4))
    data = [df.loc[df.target == t, "r2"].dropna().to_numpy() for t in TARGETS]
    labels = [TARGET_LABELS[t] for t in TARGETS]
    try:
        bp = ax.boxplot(data, labels=labels, patch_artist=True, widths=0.55)
    except TypeError:
        bp = ax.boxplot(data, tick_labels=labels, patch_artist=True, widths=0.55)
    for patch, t in zip(bp["boxes"], TARGETS):
        patch.set_facecolor(colors[t])
        patch.set_alpha(0.7)
    ax.axhline(0.0, color="0.6", lw=0.8)
    ax.set_ylabel("held-out R²")
    ax.set_title("ModelDataLeftIpsi · motor units · Ridge decoder")
    ax.set_ylim(min(-0.2, np.nanmin(df.r2) - 0.05), max(0.5, np.nanmax(df.r2) + 0.05))
    fig.tight_layout()
    fig.savefig(out_dir / "r2_distributions.png", dpi=160)
    plt.close(fig)

    # 2. per-session dots
    fig, ax = plt.subplots(figsize=(10.5, 4.6))
    sessions = sorted(df.eid.unique())
    x = np.arange(len(sessions))
    for i, t in enumerate(TARGETS):
        sub = df[df.target == t].set_index("eid").reindex(sessions)
        ax.scatter(x + (i - 1) * 0.18, sub.r2, s=18, color=colors[t], label=TARGET_LABELS[t], alpha=0.85)
    ax.axhline(0.0, color="0.6", lw=0.8)
    ax.set_xticks(x[:: max(len(x) // 12, 1)])
    ax.set_xticklabels([s[:8] for s in sessions[:: max(len(x) // 12, 1)]], rotation=45, ha="right", fontsize=8)
    ax.set_ylabel("held-out R²")
    ax.set_xlabel("session (eid prefix)")
    ax.set_title("Per-session decoding, MOp/MOs population")
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(out_dir / "r2_by_session.png", dpi=160)
    plt.close(fig)

    # 3. pairwise scatter
    wide = df.pivot_table(index="eid", columns="target", values="r2")
    fig, axes = plt.subplots(1, 3, figsize=(10.8, 3.6))
    pairs = [
        ("wheel_speed", "paw_speed"),
        ("wheel_speed", "paw_speed_norm"),
        ("paw_speed", "paw_speed_norm"),
    ]
    for ax, (a, b) in zip(axes, pairs):
        x = wide[a]
        y = wide[b]
        ax.scatter(x, y, s=22, color="#334155", alpha=0.85)
        lo = np.nanmin([x.min(), y.min(), 0])
        hi = np.nanmax([x.max(), y.max(), 0.2])
        ax.plot([lo, hi], [lo, hi], color="0.7", lw=0.8)
        ax.set_xlabel(TARGET_LABELS[a] + " R²")
        ax.set_ylabel(TARGET_LABELS[b] + " R²")
        ax.set_aspect("equal", adjustable="box")
    fig.suptitle("Same session, different targets")
    fig.tight_layout()
    fig.savefig(out_dir / "r2_scatter_targets.png", dpi=160)
    plt.close(fig)

    # 4. regime R²
    if all(f"r2_{r}" in df.columns for r in ("still", "coupled", "decoupled")):
        fig, ax = plt.subplots(figsize=(7.4, 4.4))
        regimes = ["still", "coupled", "decoupled"]
        x = np.arange(len(regimes))
        width = 0.24
        for i, t in enumerate(TARGETS):
            vals = [df.loc[df.target == t, f"r2_{r}"].mean() for r in regimes]
            ax.bar(x + (i - 1) * width, vals, width=width, color=colors[t], label=TARGET_LABELS[t])
        ax.axhline(0.0, color="0.6", lw=0.8)
        ax.set_xticks(x)
        ax.set_xticklabels(regimes)
        ax.set_ylabel("mean session R² (test bins)")
        ax.set_title("Held-out R² split by paw regime")
        ax.legend(frameon=False)
        fig.tight_layout()
        fig.savefig(out_dir / "r2_by_regime.png", dpi=160)
        plt.close(fig)

    # 5. example traces
    if examples:
        fig, axes = plt.subplots(len(examples), 1, figsize=(8.8, 2.4 * len(examples)), sharex=False)
        if len(examples) == 1:
            axes = [axes]
        for ax, ex in zip(axes, examples):
            t = np.arange(len(ex["y"])) * BINSIZE
            ax.plot(t, ex["y"], color="0.25", lw=1.4, label="actual")
            ax.plot(t, ex["yhat"], color=colors[ex["target"]], lw=1.4, label="decoded")
            ax.set_title(
                f"{ex['mouse']} {ex['date']}  trial {ex['trial_index']}  "
                f"{TARGET_LABELS[ex['target']]}  R²={ex['r2_sess']:.2f}"
            )
            ax.set_ylabel(ex["ylabel"])
            ax.legend(frameon=False, loc="upper right")
        axes[-1].set_xlabel("time in trial after lag crop (s), origin = stimOn + 0.20 s")
        fig.tight_layout()
        fig.savefig(out_dir / "example_traces.png", dpi=160)
        plt.close(fig)

    # 6. null-corrected summary bars
    fig, ax = plt.subplots(figsize=(6.8, 4.2))
    means = [df.loc[df.target == t, "r2"].mean() for t in TARGETS]
    corr = [df.loc[df.target == t, "r2_corrected"].mean() for t in TARGETS]
    x = np.arange(3)
    ax.bar(x - 0.18, means, 0.36, color=[colors[t] for t in TARGETS], label="raw R²")
    ax.bar(x + 0.18, corr, 0.36, color=[colors[t] for t in TARGETS], alpha=0.45, label="null-corrected")
    ax.axhline(0.0, color="0.6", lw=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels([TARGET_LABELS[t] for t in TARGETS], rotation=15, ha="right")
    ax.set_ylabel("mean across sessions")
    ax.set_title("Raw vs shuffle-null-corrected R²")
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(out_dir / "r2_null_corrected.png", dpi=160)
    plt.close(fig)


def pick_examples(session_examples):
    """One decent trial per target from sessions with finite R²."""
    chosen = []
    for target in TARGETS:
        cands = [e for e in session_examples if e["target"] == target]
        cands.sort(key=lambda e: e["r2_sess"], reverse=True)
        if cands:
            chosen.append(cands[0])
    return chosen


def summarize(rows):
    df = pd.DataFrame(rows)
    lines = []
    lines.append("BWM-style decoding on ModelDataLeftIpsi (left paw, left hemisphere, MOp/MOs)")
    lines.append(
        f"sessions with a fit: {df.eid.nunique()} / 52   "
        f"method: Ridge nested {N_FOLDS}-fold, lag W={N_LAGS}, {N_NULL} circular-shift nulls"
    )
    lines.append("window: stimOn→reward (paper's firstMovement −0.2:+1.0 s does not fit these trials)")
    lines.append(
        f"null p-floor = 1/{N_NULL + 1} ≈ {1 / (N_NULL + 1):.3f}; "
        "report 'beats all nulls' instead of p<0.05"
    )
    lines.append("")
    p_floor = round(1.0 / (N_NULL + 1), 3) + 1e-9
    for t in TARGETS:
        sub = df[df.target == t]
        beats = int((sub.p_shuffle.round(3) <= p_floor).sum())
        lines.append(
            f"  {TARGET_LABELS[t]:24s}  "
            f"median R²={sub.r2.median():6.3f}  mean={sub.r2.mean():6.3f}  "
            f"null-corr median={sub.r2_corrected.median():6.3f}  "
            f"beats all {N_NULL} nulls in {beats}/{len(sub)} sessions"
        )
        for r in ("still", "coupled", "decoupled"):
            col = f"r2_{r}"
            if col in sub.columns and sub[col].notna().any():
                lines.append(f"      {r:10s} mean R²={sub[col].mean():6.3f}")
    lines.append("")
    wide = df.pivot_table(index="eid", columns="target", values="r2")
    if set(TARGETS) <= set(wide.columns):
        lines.append(
            "  corr(R² wheel, paw)     "
            f"{wide.wheel_speed.corr(wide.paw_speed):.2f}"
        )
        lines.append(
            "  corr(R² wheel, paw_norm) "
            f"{wide.wheel_speed.corr(wide.paw_speed_norm):.2f}"
        )
        lines.append(
            "  corr(R² paw, paw_norm)   "
            f"{wide.paw_speed.corr(wide.paw_speed_norm):.2f}"
        )
        paw_better = (wide.paw_speed > wide.wheel_speed).mean()
        lines.append(f"  fraction of sessions with paw R² > wheel R²: {paw_better:.2f}")
    return "\n".join(lines), df


def process(session_stems=None, make_plots=True, job_index=None):
    warnings.filterwarnings("ignore", category=UserWarning)
    warnings.filterwarnings("ignore", message="Ill-conditioned")
    OUT.mkdir(parents=True, exist_ok=True)
    pkls = session_paths()
    if session_stems is not None:
        want = set(session_stems)
        pkls = [p for p in pkls if p.stem in want]
    rng = np.random.default_rng(0)
    rows = []
    example_pool = []
    log(f"{len(pkls)} sessions in ModelDataLeftIpsi")
    for i, path in enumerate(pkls, start=0):
        with path.open("rb") as fh:
            payload = pickle.load(fh)
        mouse = payload.get("mouse_id")
        date = payload.get("session_date")
        eid = payload["eid"]
        idx = job_index if job_index is not None else i
        jdir = slurm_utils.begin_job(OUT, idx, str(eid)[:8], eid=eid, mouse=mouse)
        log(f"[{i + 1}/{len(pkls)}] {mouse} {date} {eid[:8]}")
        these = []
        for target in TARGETS:
            slurm_utils.write_status(state="fitting", target=target)
            result = decode_session(payload, target, rng)
            if result is None:
                log(f"  skip {target}: too few trials/units")
                continue
            row = {
                "eid": eid,
                "mouse_id": mouse,
                "session_date": str(date),
                "target": target,
                "r2": result["r2"],
                "r2_corrected": result["r2_corrected"],
                "null_median": result["null_median"],
                "p_shuffle": result["p_shuffle"],
                "n_trials": result["n_trials"],
                "n_units": result["n_units"],
                "best_alpha_mean": result["best_alpha_mean"],
                "r2_still": result["regime_r2"]["still"],
                "r2_coupled": result["regime_r2"]["coupled"],
                "r2_decoupled": result["regime_r2"]["decoupled"],
            }
            rows.append(row)
            these.append(row)
            log(
                f"  {target:16s} R²={result['r2']:+.3f}  "
                f"null={result['null_median']:+.3f}  p={result['p_shuffle']:.3f}  "
                f"n_tr={result['n_trials']}  n_u={result['n_units']}"
            )
            lengths = [len(y) for y in result["ys"]]
            j = int(np.argsort(lengths)[len(lengths) // 2])
            ylab = {
                "wheel_speed": "|ω| (rad/s)",
                "paw_speed": "px/s (left-cam equiv.)",
                "paw_speed_norm": "speed / coupled median",
            }[target]
            example_pool.append(
                {
                    "target": target,
                    "mouse": mouse,
                    "date": date,
                    "trial_index": result["meta"][j]["trial_index"],
                    "y": result["ys"][j],
                    "yhat": result["preds"][j],
                    "r2_sess": result["r2"],
                    "ylabel": ylab,
                }
            )
        if these:
            pd.DataFrame(these).to_csv(jdir / "scores.csv", index=False)
        slurm_utils.write_status(state="done", n_rows=len(these))
    if not rows:
        raise SystemExit("No sessions decoded.")
    if make_plots:
        examples = pick_examples(example_pool)
        return write_outputs(rows, examples)
    log("wrote job dir " + str(slurm_utils.job_dir() or OUT / "jobs"))
    print(pd.DataFrame(rows).to_string(index=False))
    return rows


def recover_rows_from_log(log_path):
    """Rebuild session_scores from a completed decode log (plot crash recovery)."""
    text = Path(log_path).read_text()
    eid_map = {p.stem[:8]: p.stem for p in SESS.glob("*.pkl") if not p.name.endswith(".tmp")}
    rows = []
    cur = None
    sess_re = re.compile(r"\[(\d+)/52\] (\S+) (\S+) (\S+)")
    tgt_re = re.compile(
        r"  (wheel_speed|paw_speed|paw_speed_norm)\s+"
        r"R²=([+-]?\d+\.\d+)\s+null=([+-]?\d+\.\d+)\s+"
        r"p=(\d+\.\d+)\s+n_tr=(\d+)\s+n_u=(\d+)"
    )
    for line in text.splitlines():
        m = sess_re.search(line)
        if m:
            cur = {"mouse": m.group(2), "date": m.group(3), "eid8": m.group(4)}
            continue
        m = tgt_re.search(line)
        if m and cur:
            r2 = float(m.group(2))
            null = float(m.group(3))
            rows.append(
                {
                    "eid": eid_map.get(cur["eid8"], cur["eid8"]),
                    "mouse_id": cur["mouse"],
                    "session_date": cur["date"],
                    "target": m.group(1),
                    "r2": r2,
                    "r2_corrected": r2 - null,
                    "null_median": null,
                    "p_shuffle": float(m.group(4)),
                    "n_trials": int(m.group(5)),
                    "n_units": int(m.group(6)),
                }
            )
    if not rows:
        raise SystemExit(f"no decode rows parsed from {log_path}")
    return rows


def _pick_trace_index(ys, preds):
    """Prefer a long, high-variance trial so traces are readable."""
    lengths = np.array([len(y) for y in ys])
    cutoff = float(np.percentile(lengths, 75)) if lengths.size else 0
    cands = [i for i, n in enumerate(lengths) if n >= cutoff]
    if not cands:
        cands = list(range(len(ys)))

    def score(i):
        y = np.asarray(ys[i], dtype=float)
        p = np.asarray(preds[i], dtype=float)
        m = np.isfinite(y) & np.isfinite(p)
        if m.sum() < 8:
            return -1.0
        return float(np.nanvar(y[m]) * m.sum())

    return max(cands, key=score)


def fetch_examples(rows, n_sessions=3):
    """Re-decode each target's best session (no nulls) and pick a long trial."""
    del n_sessions
    df = pd.DataFrame(rows)
    examples = []
    rng = np.random.default_rng(0)
    global N_NULL
    saved = N_NULL
    N_NULL = 0
    try:
        for target in TARGETS:
            sub = df[df.target == target].sort_values("r2", ascending=False)
            if sub.empty:
                continue
            eid = sub.iloc[0]["eid"]
            path = SESS / f"{eid}.pkl"
            if not path.exists():
                matches = list(SESS.glob(f"{str(eid)[:8]}*.pkl"))
                path = matches[0] if matches else path
            if not path.exists():
                continue
            with path.open("rb") as fh:
                payload = pickle.load(fh)
            mouse = payload.get("mouse_id")
            date = payload.get("session_date")
            log(f"example traces {mouse} {date} {str(eid)[:8]} {target}")
            result = decode_session(payload, target, rng)
            if result is None:
                continue
            j = _pick_trace_index(result["ys"], result["preds"])
            ylab = {
                "wheel_speed": "|ω| (rad/s)",
                "paw_speed": "px/s (left-cam equiv.)",
                "paw_speed_norm": "speed / coupled median",
            }[target]
            examples.append(
                {
                    "target": target,
                    "mouse": mouse,
                    "date": date,
                    "trial_index": result["meta"][j]["trial_index"],
                    "y": result["ys"][j],
                    "yhat": result["preds"][j],
                    "r2_sess": result["r2"],
                    "ylabel": ylab,
                }
            )
    finally:
        N_NULL = saved
    return examples


def write_outputs(rows, examples):
    warnings.filterwarnings("ignore", category=UserWarning)
    OUT.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(rows)
    df.to_csv(OUT / "session_scores.csv", index=False)
    text, df = summarize(rows)
    plot_all(rows, examples, OUT)
    summary = {
        "built_at": datetime.now(timezone.utc).isoformat(),
        "data": str(DATA),
        "n_sessions_decoded": int(df.eid.nunique()),
        "method": {
            "estimator": "Ridge + StandardScaler on each train fold (paper used Lasso; L1 did not converge on raw counts)",
            "alphas": ALPHAS.tolist(),
            "n_folds": N_FOLDS,
            "n_lags": N_LAGS,
            "n_null": N_NULL,
            "window": "stimOn to reward, leftover dropped; not paper 1.2 s firstMovement window",
            "units": "MOp or MOs, label>=1, left hemisphere",
        },
        "by_target": {},
        "text": text,
    }
    p_floor = round(1.0 / (N_NULL + 1), 3) + 1e-9
    for t in TARGETS:
        sub = df[df.target == t]
        entry = {
            "n_sessions": int(len(sub)),
            "median_r2": float(sub.r2.median()),
            "mean_r2": float(sub.r2.mean()),
            "median_r2_corrected": float(sub.r2_corrected.median()),
            "n_beats_all_nulls": int((sub.p_shuffle.round(3) <= p_floor).sum()),
        }
        for r in ("still", "coupled", "decoupled"):
            col = f"r2_{r}"
            if col in sub.columns:
                entry[f"mean_r2_{r}"] = float(sub[col].mean())
        summary["by_target"][t] = entry
    (OUT / "summary.json").write_text(json.dumps(summary, indent=2))
    (OUT / "summary.txt").write_text(text + "\n")
    log("wrote " + str(OUT))
    print("\n" + text)
    return summary


def process_from_log(log_path):
    rows = recover_rows_from_log(log_path)
    log(f"recovered {len(rows)} session/target scores from log")
    examples = fetch_examples(rows)
    return write_outputs(rows, examples)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--from-log", default=None, help="rebuild scores from a decode log")
    slurm_utils.add_common_args(p)
    args = p.parse_args()
    if args.list_jobs:
        slurm_utils.print_jobs(job_grid())
        raise SystemExit(0)
    slurm_utils.set_device(args.device)
    if args.aggregate:
        df = slurm_utils.aggregate_scores(OUT, dest_name="session_scores.csv")
        rows = df.to_dict(orient="records")
        examples = fetch_examples(rows)
        write_outputs(rows, examples)
        raise SystemExit(0)
    if args.from_log:
        process_from_log(args.from_log)
        raise SystemExit(0)
    job = slurm_utils.resolve_job_index(args.job)
    if job is None:
        process()
    else:
        cfg = slurm_utils.pick_config(job_grid(), job)
        process(session_stems=(cfg["session"],), make_plots=False, job_index=job)
