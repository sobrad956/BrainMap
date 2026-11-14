#!/usr/bin/env python3
"""Per-session LFP band-power EDA + combined QC/attention report data.

Never glob model.pt. Never JOBS.iterdir().
"""
from __future__ import annotations

import gc
import json
import math
import pickle
from pathlib import Path

import numpy as np

ROOT = Path("/Users/Stefan/Desktop/BrainMap/BehaviorRes")
LFP_DIR = ROOT / "ModelDataRightContraLFP" / "sessions"
SRC_DIR = ROOT / "ModelDataRightContra" / "sessions"
INSPECT = ROOT / "ModelDataRightContraLFP" / "inspect.json"
MANIFEST = ROOT / "ModelDataRightContraLFP" / "manifest.json"
OUT_JS = ROOT / "experiments_clean_lfp_report_data.js"
OUT_JSON = ROOT / "lfp_combined_report.json"

BANDS = ("delta", "theta", "alpha", "beta", "low_gamma", "high_gamma")
BAND_HZ = [(0.5, 4.0), (4.0, 8.0), (8.0, 13.0), (13.0, 30.0), (30.0, 55.0), (65.0, 100.0)]
EPS = 1.0
ONSET_THR = 150.0
BIN = 0.02
ALIGN_PRE = 20
ALIGN_POST = 40
ALIGN = np.arange(-ALIGN_PRE, ALIGN_POST + 1, dtype=int)
MIN_BINS = 8
MIN_ONSET_EDGE = 3
HIST_LO, HIST_HI, HIST_N = 0.0, 5.0, 20

ATTN = {
    "concat": {
        "gru": {"LTO": [0.2054, 0.1807, 0.1833], "LSO": [0.1578, 0.1111, 0.0639], "LMO": [0.1405, 0.0837, 0.1063]},
        "a1tf": {"LTO": [0.1985, 0.1778, 0.1785], "LSO": [0.1346, 0.1191, 0.0839], "LMO": [0.1341, 0.1176, 0.1045]},
        "anat": {"LTO": [0.1963, 0.1945, 0.1967], "LSO": [0.1192, 0.0779, 0.091], "LMO": [0.0999, 0.0768, 0.0853]},
    },
    "concat_sem": {
        "gru": {"LTO": [0.0067, 0.0064, 0.0079], "LSO": [0.0511, 0.0305, 0.0535], "LMO": [0.0365, 0.036, 0.0253]},
        "a1tf": {"LTO": [0.0076, 0.006, 0.0068], "LSO": [0.057, 0.0324, 0.0458], "LMO": [0.0384, 0.0263, 0.0257]},
        "anat": {"LTO": [0.0077, 0.0078, 0.0067], "LSO": [0.0274, 0.0515, 0.0339], "LMO": [0.0308, 0.0229, 0.0327]},
    },
    "wins": {
        "gru": {"LTO": [42.5, 30.7, 26.6], "LSO": [47.4, 28.7, 23.6], "LMO": [43.9, 30.7, 25.2]},
        "a1tf": {"LTO": [42.2, 28.7, 28.7], "LSO": [43.8, 28.7, 27.3], "LMO": [41.7, 31.2, 26.7]},
        "anat": {"LTO": [34.9, 31.6, 33.0], "LSO": [39.2, 30.7, 29.7], "LMO": [38.1, 30.0, 31.1]},
    },
    "d_concat": {
        "gru": {"LTO": [-0.0247, -0.0221], "LSO": [-0.0467, -0.0939], "LMO": [-0.0568, -0.0342]},
        "a1tf": {"LTO": [-0.0207, -0.02], "LSO": [-0.0155, -0.0507], "LMO": [-0.0165, -0.0296]},
        "anat": {"LTO": [-0.0018, 0.0004], "LSO": [-0.0413, -0.0282], "LMO": [-0.0231, -0.0146]},
    },
    "enc_lfp": {
        "gru": [0.1596, 0.1503, 0.1707, 0.1809, 0.1663, 0.1724],
        "a1tf": [0.1717, 0.1669, 0.1785, 0.1495, 0.1633, 0.1701],
        "anat": [0.1733, 0.1754, 0.152, 0.1355, 0.1821, 0.1817],
    },
    "enc_fused": {
        "gru": [0.1828, 0.1143, 0.1366, 0.1402, 0.134, 0.1499, 0.1422],
        "a1tf": [0.1719, 0.1516, 0.1213, 0.1275, 0.1294, 0.1427, 0.1556],
        "anat": [0.2264, 0.1411, 0.1398, 0.1177, 0.1177, 0.1168, 0.1404],
    },
    "lag_ms": ["0", "40", "80", "120", "160", "200", "240", "280", "320", "360", "400"],
    "lag_spikes": [0.1971, 0.1347, 0.1021, 0.071, 0.0511, 0.0399, 0.0364, 0.0239, 0.0161, 0.0112, 0.0061],
    "lag_lfp": [0.1865, 0.1168, 0.0903, 0.0762, 0.0692, 0.0615, 0.0608, 0.0377, 0.0225, 0.0171, 0.01],
    "lag_slfp": [0.1881, 0.1128, 0.094, 0.0797, 0.0699, 0.0613, 0.0542, 0.0335, 0.0225, 0.0145, 0.0089],
    "band_lift_lfp": [0.139, 0.083, 0.124, 0.170, -0.062, 0.017],
    "band_lift_slfp": [0.093, 0.114, 0.123, -0.089, -0.066, -0.290],
    "band_lift_pre": [0.139, 0.104, 0.133, 0.19, -0.072, 0.017],
    "band_lift_post": [0.126, 0.043, 0.103, 0.13, -0.046, 0.02],
    "occlusion_fused": {
        "cats": ["all LFP", "δ", "spike", "β", "α", "θ", "high-γ", "low-γ"],
        "delta": [0.0613, 0.0178, 0.0049, 0.0024, 0.0004, -0.0047, -0.0113, -0.026],
    },
    "token": {
        "spikes": {"n": 143, "peak": 0.041, "ent": 0.850, "lag0": 0.197, "row_ent": 0.672, "hl_s": 0.08},
        "lfp": {"n": 356, "peak": 0.008, "ent": 0.985, "lag0": 0.187, "row_ent": 0.649, "hl_s": 0.12},
        "slfp": {"n": 143, "peak": 0.067, "ent": 0.930, "lag0": 0.188, "row_ent": 0.643, "hl_s": 0.10},
    },
    "n_matched": 42324,
    "n_packs": 255,
    "lto_sess": {"n": 49, "lfp_best": 18, "slfp_best": 8, "spk_best": 23},
}


def log(msg):
    print(msg, flush=True)


def jf(x, nd=4):
    if x is None:
        return None
    try:
        v = float(x)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(v):
        return None
    return round(v, nd)


def pearson(x, y):
    x = np.asarray(x, dtype=np.float64).reshape(-1)
    y = np.asarray(y, dtype=np.float64).reshape(-1)
    n = min(x.size, y.size)
    x, y = x[:n], y[:n]
    m = np.isfinite(x) & np.isfinite(y)
    if int(m.sum()) < 12:
        return None
    x, y = x[m], y[m]
    if float(x.std()) < 1e-12 or float(y.std()) < 1e-12:
        return None
    return float(np.corrcoef(x, y)[0, 1])


def paw_is_wheel_paw(rec):
    return rec.get("used_paw") == "right" and rec.get("wheel_paws") == "right"


def paw_speed(rec):
    for k in ("lp_speed", "dlc_speed"):
        v = rec.get(k)
        if v is None:
            continue
        arr = np.asarray(v, dtype=np.float32).reshape(-1)
        if arr.size and np.isfinite(arr).any():
            return arr
    vx = rec.get("lp_vx", rec.get("dlc_vx"))
    vy = rec.get("lp_vy", rec.get("dlc_vy"))
    if vx is None or vy is None:
        return np.zeros(0, dtype=np.float32)
    n = min(np.asarray(vx).size, np.asarray(vy).size)
    if n < 1:
        return np.zeros(0, dtype=np.float32)
    return np.hypot(np.asarray(vx, dtype=np.float32).reshape(-1)[:n],
                    np.asarray(vy, dtype=np.float32).reshape(-1)[:n])


def onset_bin(y, thr=ONSET_THR, sticky=2):
    arr = np.asarray(y, dtype=np.float64).reshape(-1)
    above = np.isfinite(arr) & (np.abs(arr) >= float(thr))
    run = 0
    for i, hit in enumerate(above):
        run = run + 1 if hit else 0
        if run >= int(sticky):
            return i - int(sticky) + 1
    return None


def src_by_index(src):
    out = {}
    for rec in src.get("trials") or []:
        ti = rec.get("trial_index")
        if ti is None:
            continue
        out[int(ti)] = rec
    return out


def process_session(lfp_path):
    with lfp_path.open("rb") as fh:
        payload = pickle.load(fh)
    eid = str(payload.get("eid") or lfp_path.stem)
    src_path = SRC_DIR / f"{eid}.pkl"
    if not src_path.exists():
        log(f"  skip {eid[:8]} no spike pkl")
        return None
    with src_path.open("rb") as fh:
        src = pickle.load(fh)
    by_idx = src_by_index(src)
    mouse = str(payload.get("mouse_id") or src.get("mouse_id") or "")
    lab = payload.get("lab") or src.get("lab")
    line = int(payload.get("line_hz") or 60)
    trials = payload.get("trials") or []
    if not trials:
        return None
    motor = np.asarray(trials[0].get("channel_motor"), dtype=bool).reshape(-1)
    if motor.size < 1 or int(motor.sum()) < 1:
        acr = np.asarray(trials[0].get("channel_acronym")).astype(str)
        motor = np.array([a.startswith("MOp") or a.startswith("MOs") for a in acr])
    n_motor = int(motor.sum())
    if n_motor < 1:
        return None

    logp_chunks = [[] for _ in BANDS]
    lin_sum = np.zeros(6, dtype=np.float64)
    lin_n = 0
    pre_log = []
    post_log = []
    pre_lin = []
    post_lin = []
    pre_chunks = []
    post_chunks = []
    r_acc = [[] for _ in BANDS]
    aligned_sum = np.zeros((6, ALIGN.size), dtype=np.float64)
    aligned_cnt = np.zeros(ALIGN.size, dtype=np.float64)
    n_ok = 0
    n_onset = 0

    for rec in trials:
        ti = rec.get("trial_index")
        src_rec = by_idx.get(int(ti) if ti is not None else -1)
        if src_rec is None or not paw_is_wheel_paw(src_rec):
            continue
        pow20 = rec.get("power_car_s20")
        if pow20 is None:
            continue
        p = np.asarray(pow20, dtype=np.float32)
        if p.ndim != 3 or p.shape[-1] < 6:
            continue
        n_ch = min(p.shape[1], motor.size)
        msk = motor[:n_ch]
        if int(msk.sum()) < 1:
            continue
        p = np.clip(p[:, :n_ch, :6][:, msk, :], 0.0, None)
        T = int(p.shape[0])
        if T < MIN_BINS:
            continue
        paw = paw_speed(src_rec)
        n = min(T, int(np.asarray(paw).size))
        if n < MIN_BINS:
            continue
        p = p[:n]
        paw = np.asarray(paw, dtype=np.float32).reshape(-1)[:n]
        logp = np.log10(p + EPS).astype(np.float32)
        ch_mean = logp.mean(axis=1)
        lin = p.mean(axis=1)
        lin_sum += lin.sum(axis=0)
        lin_n += n
        for b in range(6):
            logp_chunks[b].append(logp[:, :, b].reshape(-1))
            rr = pearson(ch_mean[:, b], paw)
            if rr is not None:
                r_acc[b].append(rr)
        n_ok += 1
        on = onset_bin(paw)
        if on is None or on < MIN_ONSET_EDGE or on > n - MIN_ONSET_EDGE:
            continue
        n_onset += 1
        pre_log.append(ch_mean[:on].mean(axis=0))
        post_log.append(ch_mean[on:].mean(axis=0))
        pre_lin.append(lin[:on].mean(axis=0))
        post_lin.append(lin[on:].mean(axis=0))
        pre_chunks.append(ch_mean[:on])
        post_chunks.append(ch_mean[on:])
        for k, tau in enumerate(ALIGN):
            j = on + int(tau)
            if 0 <= j < n:
                aligned_sum[:, k] += ch_mean[j]
                aligned_cnt[k] += 1.0

    del payload, src, trials
    gc.collect()
    if n_ok < 8:
        log(f"  skip {eid[:8]} n_ok={n_ok}")
        return None

    p10, p50, p90, mean_log, hist = [], [], [], [], []
    edges = np.linspace(HIST_LO, HIST_HI, HIST_N + 1)
    for b in range(6):
        v = np.concatenate(logp_chunks[b]) if logp_chunks[b] else np.zeros(1)
        p10.append(jf(float(np.percentile(v, 10)), 4))
        p50.append(jf(float(np.percentile(v, 50)), 4))
        p90.append(jf(float(np.percentile(v, 90)), 4))
        mean_log.append(jf(float(np.mean(v)), 4))
        counts, _ = np.histogram(v, bins=edges)
        tot = float(counts.sum()) or 1.0
        hist.append([jf(float(c / tot), 4) for c in counts.tolist()])
    lin_mean = lin_sum / max(lin_n, 1)
    share = (lin_mean / (lin_mean.sum() + 1e-12)).tolist()

    def pct_stack(chunks, q):
        if not chunks:
            return [None] * 6
        arr = np.concatenate(chunks, axis=0)
        return [jf(float(np.percentile(arr[:, b], q)), 4) for b in range(6)]

    def mean_share(rows):
        if not rows:
            return [None] * 6
        m = np.mean(np.stack(rows, 0), axis=0)
        s = m / (m.sum() + 1e-12)
        return [jf(float(x), 4) for x in s.tolist()]

    if pre_log:
        pre = np.mean(np.stack(pre_log, 0), axis=0)
        post = np.mean(np.stack(post_log, 0), axis=0)
        dpp = (post - pre).tolist()
        pre_m = pre.tolist()
        post_m = post.tolist()
    else:
        pre_m = post_m = [None] * 6
        dpp = [None] * 6
    r_paw = [jf(float(np.mean(v)), 4) if v else None for v in r_acc]
    al = []
    for b in range(6):
        row = []
        for k in range(ALIGN.size):
            c = aligned_cnt[k]
            row.append(jf(float(aligned_sum[b, k] / c), 4) if c >= 8 else None)
        al.append(row)
    log(f"  {mouse} {eid[:8]} trials={n_ok} onset={n_onset} motor={n_motor}")
    return {
        "eid": eid,
        "eid8": eid[:8],
        "mouse": mouse,
        "lab": str(lab or ""),
        "line_hz": line,
        "n_trials": n_ok,
        "n_onset": n_onset,
        "n_motor": n_motor,
        "mean_log": mean_log,
        "p10": p10,
        "p50": p50,
        "p90": p90,
        "hist": hist,
        "hist_edges": [jf(float(x), 3) for x in edges.tolist()],
        "share": [jf(x, 4) for x in share],
        "share_pre": mean_share(pre_lin),
        "share_post": mean_share(post_lin),
        "pre_log": [jf(x, 4) for x in pre_m],
        "post_log": [jf(x, 4) for x in post_m],
        "pre_p10": pct_stack(pre_chunks, 10),
        "pre_p50": pct_stack(pre_chunks, 50),
        "pre_p90": pct_stack(pre_chunks, 90),
        "post_p10": pct_stack(post_chunks, 10),
        "post_p50": pct_stack(post_chunks, 50),
        "post_p90": pct_stack(post_chunks, 90),
        "d_post_pre": [jf(x, 4) for x in dpp],
        "r_paw": r_paw,
        "aligned": al,
        "aligned_n": [int(v) for v in aligned_cnt.tolist()],
    }


def median_across(rows, key, n=6):
    out = []
    for b in range(n):
        vals = [r[key][b] for r in rows if r.get(key) and r[key][b] is not None]
        out.append(jf(float(np.median(vals)), 4) if vals else None)
    return out


def median_hist(rows):
    if not rows or not rows[0].get("hist"):
        return None
    n_b = 6
    n_h = len(rows[0]["hist"][0])
    out = []
    for b in range(n_b):
        cols = []
        for k in range(n_h):
            vals = [r["hist"][b][k] for r in rows if r.get("hist") and r["hist"][b][k] is not None]
            cols.append(jf(float(np.median(vals)), 4) if vals else 0.0)
        out.append(cols)
    return out


def aligned_corpus(rows):
    """Median across sessions of within-session z-scored aligned curves."""
    empty = {
        "t": [round(float(k) * BIN, 3) for k in ALIGN.tolist()],
        "z": [[None] * ALIGN.size for _ in range(6)],
        "n": [0] * ALIGN.size,
    }
    if not rows:
        return empty
    curves = np.full((len(rows), 6, ALIGN.size), np.nan, dtype=np.float64)
    for i, r in enumerate(rows):
        al = np.array(
            [[(v if v is not None else np.nan) for v in row] for row in r["aligned"]],
            dtype=np.float64,
        )
        for b in range(6):
            v = al[b]
            m = np.isfinite(v)
            if int(m.sum()) < 8:
                continue
            mu, sd = float(np.nanmean(v)), float(np.nanstd(v))
            if sd < 1e-8:
                continue
            curves[i, b] = (v - mu) / sd
    med = np.nanmedian(curves, axis=0)
    n = np.isfinite(curves).sum(axis=0)
    return {
        "t": [round(float(k) * BIN, 3) for k in ALIGN.tolist()],
        "z": [[jf(float(med[b, k]), 4) for k in range(ALIGN.size)] for b in range(6)],
        "n": [int(n[0, k]) for k in range(ALIGN.size)],
    }


def compute_eda():
    rows = []
    paths = sorted(p for p in LFP_DIR.glob("*.pkl") if not p.name.endswith(".tmp"))
    log(f"EDA sessions on disk: {len(paths)}  dir={LFP_DIR} exists={LFP_DIR.exists()}")
    if not paths:
        oks = list(LFP_DIR.glob("*.ok.json"))
        log(f"NO PKL files; ok.json={len(oks)}")
    for i, path in enumerate(paths, 1):
        log(f"[{i}/{len(paths)}] {path.stem[:8]}")
        try:
            rec = process_session(path)
        except Exception as err:
            log(f"  FAIL {path.stem[:8]}: {err}")
            rec = None
        if rec:
            rows.append(rec)
    rows.sort(key=lambda r: (r["mouse"], r["eid8"]))
    n_up = {b: 0 for b in BANDS}
    n_dn = {b: 0 for b in BANDS}
    for r in rows:
        for i, b in enumerate(BANDS):
            d = r["d_post_pre"][i]
            if d is None:
                continue
            if d > 0:
                n_up[b] += 1
            elif d < 0:
                n_dn[b] += 1
    return {
        "bands": list(BANDS),
        "band_hz": BAND_HZ,
        "align_t": [round(float(k) * BIN, 3) for k in ALIGN.tolist()],
        "n_sessions": len(rows),
        "n_trials": int(sum(r["n_trials"] for r in rows)),
        "n_onset_trials": int(sum(r["n_onset"] for r in rows)),
        "median_share": median_across(rows, "share"),
        "median_share_pre": median_across(rows, "share_pre"),
        "median_share_post": median_across(rows, "share_post"),
        "median_mean_log": median_across(rows, "mean_log"),
        "median_pre_log": median_across(rows, "pre_log"),
        "median_post_log": median_across(rows, "post_log"),
        "median_d_post_pre": median_across(rows, "d_post_pre"),
        "median_r_paw": median_across(rows, "r_paw"),
        "median_hist": median_hist(rows),
        "n_up_after_onset": n_up,
        "n_down_after_onset": n_dn,
        "aligned_z": aligned_corpus(rows),
        "sessions": rows,
        "note": (
            "log10(power_car_s20 + 1) on motor channels, 20 ms bins. "
            "Onset = paw speed ≥ 150 px/s for 2 bins. "
            "Band share uses mean linear µV², not log. "
            "Sessions are not pooled in raw units; corpus curves are within-session z-scored then medianed."
        ),
    }


def sanitize(obj):
    if isinstance(obj, dict):
        return {k: sanitize(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [sanitize(v) for v in obj]
    if isinstance(obj, float) and not math.isfinite(obj):
        return None
    return obj


def dump_blob(blob):
    clean = sanitize(blob)
    OUT_JSON.write_text(json.dumps(clean, allow_nan=False))
    OUT_JS.write_text("window.REPORT=" + json.dumps(clean, allow_nan=False) + ";\n")


def main():
    qc = json.loads(INSPECT.read_text()) if INSPECT.exists() else {}
    man = json.loads(MANIFEST.read_text()) if MANIFEST.exists() else {}
    qc["corpus"] = {
        "n_sessions_done": man.get("n_sessions", qc.get("sessions") and len(qc["sessions"])),
        "n_sessions_total": 52,
        "n_trials": man.get("n_trials"),
        "n_mice": man.get("n_mice"),
    }
    stub = {
        "qc": qc,
        "attn": ATTN,
        "eda": {
            "bands": list(BANDS),
            "band_hz": BAND_HZ,
            "align_t": [round(float(k) * BIN, 3) for k in ALIGN.tolist()],
            "n_sessions": 0,
            "n_trials": 0,
            "n_onset_trials": 0,
            "sessions": [],
            "note": "EDA running…",
        },
    }
    dump_blob(stub)
    log(f"wrote stub {OUT_JS} ({OUT_JS.stat().st_size/1e6:.2f} MB)")
    eda = compute_eda()
    dump_blob({"qc": qc, "attn": ATTN, "eda": eda})
    log(f"wrote {OUT_JS} {OUT_JS.stat().st_size/1e6:.2f} MB  sessions={eda['n_sessions']} trials={eda['n_trials']}")


if __name__ == "__main__":
    main()
