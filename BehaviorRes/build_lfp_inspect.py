#!/usr/bin/env python3
"""Build compact LFP QC figures from ModelDataRightContraLFP pickles."""
from __future__ import annotations

import json
import pickle
from pathlib import Path

import numpy as np
from scipy.signal import butter, iirnotch, sosfreqz, spectrogram, tf2sos, welch
from scipy.stats import pearsonr

ROOT = Path(__file__).resolve().parent
SRC = ROOT / "ModelDataRightContra"
LFP = ROOT / "ModelDataRightContraLFP"
SESS = LFP / "sessions"
OUT = LFP / "inspect.json"

BANDS = [
    ("delta", 0.5, 4.0),
    ("theta", 4.0, 8.0),
    ("alpha", 8.0, 13.0),
    ("beta", 13.0, 30.0),
    ("low_gamma", 30.0, 55.0),
    ("high_gamma", 65.0, 100.0),
]
FS = 250.0
BIN = 0.02


def log10p(x):
    x = np.asarray(x, dtype=np.float64)
    return np.log10(np.maximum(x, 1e-12))


def downsample(t, y, n=180):
    t = np.asarray(t, dtype=float)
    y = np.asarray(y, dtype=float)
    if t.size == 0:
        return [], []
    if t.size <= n:
        return t.tolist(), np.nan_to_num(y, nan=0.0).tolist()
    idx = np.unique(np.round(np.linspace(0, t.size - 1, n)).astype(int))
    return t[idx].tolist(), np.nan_to_num(y[idx], nan=0.0).tolist()


def motor_mask(acr):
    a = np.asarray(acr).astype(str)
    return np.array([x.startswith("MOp") or x.startswith("MOs") for x in a])


def pick_ch(acr, axial):
    a = np.asarray(acr).astype(str)
    ax = np.asarray(axial, dtype=float)
    out = []
    for prefix, name in (("MOp", "MOp"), ("MOs", "MOs")):
        idx = np.flatnonzero(np.array([x.startswith(prefix) for x in a]))
        if idx.size:
            mid = idx[np.argsort(ax[idx])][len(idx) // 2]
            out.append((int(mid), name, str(a[mid]), float(ax[mid])))
    other = np.flatnonzero(
        ~np.array([x.startswith("MOp") or x.startswith("MOs") for x in a])
    )
    if other.size:
        mid = other[np.argsort(ax[other])][len(other) // 2]
        out.append((int(mid), "other", str(a[mid]), float(ax[mid])))
    if not out:
        out.append((int(len(a) // 2), "mid", str(a[len(a) // 2]), float(ax[len(a) // 2])))
    return out


def load_src_behav(eid):
    p = SRC / "sessions" / f"{eid}.pkl"
    with p.open("rb") as fh:
        src = pickle.load(fh)
    by_idx = {}
    for rec in src["trials"]:
        by_idx[int(rec["trial_index"])] = rec
    return src, by_idx


def epoch_slices(n_bins, stim, go, move, fb):
    """Bin indices for baseline / preparation / movement relative to stimOn=0."""
    t_right = (np.arange(n_bins) + 1) * BIN
    t_mid = t_right - 0.5 * BIN
    g = (float(go) - stim) if go is not None and np.isfinite(go) else None
    mv = (float(move) - stim) if move is not None and np.isfinite(move) else None
    fb_rel = (float(fb) - stim) if fb is not None and np.isfinite(fb) else None
    prep_end = None
    if g is not None and g > 0.04:
        prep_end = g
    elif mv is not None and mv > 0.04:
        prep_end = mv
    else:
        prep_end = 0.2
    base = t_mid < min(0.05, prep_end)
    prep = (t_mid >= 0.0) & (t_mid < prep_end)
    if mv is not None and mv > 0:
        move_m = t_mid >= mv
    elif fb_rel is not None:
        move_m = t_mid >= max(fb_rel - 0.15, prep_end)
    else:
        move_m = t_mid >= prep_end
    return base, prep, move_m


def spec_around(t, y, t_event, pre=0.4, post=0.8):
    if t_event is None or not np.isfinite(t_event):
        return None
    m = (t >= t_event - pre) & (t <= t_event + post)
    if m.sum() < 64:
        return None
    yy = np.asarray(y[m], dtype=float)
    nper = 64
    f, tt, sxx = spectrogram(
        yy, fs=FS, nperseg=nper, noverlap=nper - 16, window="hann", detrend=False
    )
    keep = f <= 100
    f = f[keep]
    sxx = sxx[keep]
    z = log10p(sxx)
    z = z - np.nanmedian(z[:, : max(2, z.shape[1] // 5)], axis=1, keepdims=True)
    # downsample grid
    fi = np.unique(np.round(np.linspace(0, f.size - 1, 24)).astype(int))
    ti = np.unique(np.round(np.linspace(0, z.shape[1] - 1, 28)).astype(int))
    grid = z[np.ix_(fi, ti)]
    tt = tt[ti] - pre
    return {
        "t": [round(float(v), 3) for v in tt],
        "f": [round(float(v), 1) for v in f[fi]],
        "z": [[round(float(v), 2) for v in row] for row in grid],
    }


def safe_r(x, y):
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    m = np.isfinite(x) & np.isfinite(y)
    if m.sum() < 12:
        return None
    if np.nanstd(x[m]) < 1e-12 or np.nanstd(y[m]) < 1e-12:
        return None
    r, p = pearsonr(x[m], y[m])
    return {"r": float(r), "p": float(p), "n": int(m.sum())}


def filt_gain_db(sos, f, fs, zero_phase=True):
    _, h = sosfreqz(np.asarray(sos), worN=np.asarray(f), fs=fs)
    mag = np.abs(h)
    if zero_phase:
        mag = mag ** 2
    return 20.0 * np.log10(np.maximum(mag, 1e-12))


def butter_sos(kind, fs, low=None, high=None, order=4):
    nyq = 0.5 * fs
    if kind == "highpass":
        return butter(order, min(max(low / nyq, 1e-6), 0.99), btype="highpass", output="sos")
    if kind == "lowpass":
        return butter(order, min(max(high / nyq, 1e-6), 0.99), btype="lowpass", output="sos")
    lo = max(low / nyq, 1e-6)
    hi = min(high / nyq, 0.99)
    return butter(order, (lo, hi), btype="bandpass", output="sos")


def notch_sos(f0, fs, Q=30):
    b, a = iirnotch(f0, Q=Q, fs=fs)
    return tf2sos(b, a)


def build_filters():
    """Zero-phase (|H|^2) gain in dB for every SOS stage used in the pipeline."""
    f = np.linspace(0.0, 125.0, 251)
    f_eval = np.maximum(f, 1e-3)
    hp = filt_gain_db(butter_sos("highpass", FS, low=0.1), f_eval, FS)
    aa = filt_gain_db(butter_sos("lowpass", 2500.0, high=120.0), f_eval, 2500.0)
    n50 = filt_gain_db(notch_sos(50.0, FS), f_eval, FS)
    n100 = filt_gain_db(notch_sos(100.0, FS), f_eval, FS)
    n60 = filt_gain_db(notch_sos(60.0, FS), f_eval, FS)
    # 120 Hz harmonic of 60 Hz is skipped at fs=250 (0.45*fs = 112.5)
    eu = hp + n50 + n100
    us = hp + n60
    bands = {}
    for name, lo, hi in BANDS:
        bands[name] = [
            round(float(v), 2)
            for v in filt_gain_db(butter_sos("bandpass", FS, low=lo, high=hi), f_eval, FS)
        ]
    return {
        "f": [round(float(v), 2) for v in f],
        "hp_01": [round(float(v), 2) for v in hp],
        "aa_120": [round(float(v), 2) for v in aa],
        "notch50": [round(float(v), 2) for v in n50],
        "notch100": [round(float(v), 2) for v in n100],
        "notch60": [round(float(v), 2) for v in n60],
        "pre_eu": [round(float(v), 2) for v in eu],
        "pre_us": [round(float(v), 2) for v in us],
        "bands": bands,
        "notes": {
            "zero_phase": "Each SOS stage is applied with filtfilt, so plotted gain is |H(f)|^2 in dB.",
            "aa": "4th-order Butterworth low-pass at 120 Hz, applied at native LF ~2500 Hz before resample to 250 Hz.",
            "hp": "4th-order Butterworth high-pass at 0.1 Hz at 250 Hz.",
            "notch": "iirnotch Q=30 at f0 and 2*f0 if 2*f0 < 0.45*fs. US 120 Hz harmonic is skipped at 250 Hz.",
            "gap": "55–65 Hz is not a designed stop band; high gamma starts at 65 Hz to sit above mains harmonics.",
        },
    }


def example_from_trial(rec, mouse, lab, line, chans, motor, acr):
    t = np.asarray(rec["t250"], dtype=float)
    car = np.asarray(rec["lfp_car"], dtype=float)
    raw = np.asarray(rec.get("lfp_unref", car), dtype=float)
    ci, region, ac, axu = chans[0]
    if raw.shape[0] < 128 or ci >= car.shape[1]:
        return None
    nper = min(512, (raw.shape[0] // 2) * 2 or 128)
    f, p_raw = welch(raw[:, ci], fs=FS, nperseg=nper)
    _, p_car = welch(car[:, ci], fs=FS, nperseg=nper)
    keep = f <= 120
    P = rec.get("power_car_s20")
    if P is None:
        return None
    P = np.asarray(P, dtype=np.float64)
    n_bins = int(rec["n_bins"])
    nch = min(P.shape[1], motor.size)
    msk = motor[:nch]
    if not np.any(msk):
        msk = np.ones(nch, dtype=bool)
    logp = log10p(P[:n_bins, :nch, :][:, msk, :]).mean(axis=1)
    t_mid = (np.arange(n_bins) + 0.5) * BIN
    tt, _ = downsample(t_mid, logp[:, 0], n=90)
    band_log = {}
    for bi, (bn, _, _) in enumerate(BANDS):
        _, yy = downsample(t_mid, logp[:, bi], n=90)
        band_log[bn] = [round(float(v), 3) for v in yy]
    stim = float(rec["stimOn_times"])

    def rel(ts):
        if ts is None:
            return None
        try:
            v = float(ts)
        except (TypeError, ValueError):
            return None
        if not np.isfinite(v):
            return None
        return round(float(v - stim), 3)

    return {
        "mouse": mouse,
        "lab": lab,
        "eid": str(rec["eid"])[:8],
        "trial_index": int(rec["trial_index"]),
        "n_bins": n_bins,
        "line_hz": line,
        "region": region,
        "acronym": ac,
        "axial_um": float(axu),
        "f": [round(float(v), 2) for v in f[keep]],
        "psd_raw": [round(float(v), 3) for v in log10p(p_raw[keep])],
        "psd_car": [round(float(v), 3) for v in log10p(p_car[keep])],
        "t": [round(float(v), 3) for v in tt],
        "band_log": band_log,
        "events": {
            "stimOn": 0.0,
            "goCue": rel(rec.get("goCue_times")),
            "firstMovement": rel(rec.get("firstMovement_times")),
            "feedback": rel(rec.get("feedback_times")),
        },
    }


def completed_sessions():
    rows = []
    for ok in sorted(SESS.glob("*.ok.json")):
        d = json.loads(ok.read_text())
        if int(d.get("n_trials") or 0) < 8:
            continue
        rows.append(d)
    return rows


def main():
    done = completed_sessions()
    if len(done) < 1:
        raise SystemExit("no completed LFP sessions yet")
    by_hz = {50: [], 60: []}
    for d in done:
        by_hz.setdefault(int(d.get("line_hz") or 60), []).append(d)
    chosen = []
    mice = set()
    for hz in (50, 60):
        n_hz = 0
        for d in by_hz.get(hz, []):
            payload = pickle.load(open(SESS / f"{d['eid']}.pkl", "rb"))
            mouse = payload.get("mouse_id")
            if mouse in mice:
                continue
            chosen.append(payload)
            mice.add(mouse)
            n_hz += 1
            print(f"  pick {mouse} {d['eid'][:8]} {hz} Hz n={payload['n_trials']}")
            if n_hz >= 2:
                break
    if not chosen:
        chosen = [pickle.load(open(SESS / f"{done[0]['eid']}.pkl", "rb"))]

    blob = {
        "bands": BANDS,
        "sessions": [],
        "traces": [],
        "psd": [],
        "spectrograms": [],
        "epoch_power": [],
        "spatial": [],
        "assoc": [],
        "ref_smooth": [],
        "example_trials": [],
        "filters": build_filters(),
    }

    epoch_acc = {
        (b[0], ep, ref, sm): []
        for b in BANDS
        for ep in ("baseline", "preparation", "movement")
        for ref in ("car", "bip")
        for sm in ("s20", "s100")
    }
    assoc_rows = []
    spatial_rows = []

    for payload in chosen:
        eid = payload["eid"]
        src, by_idx = load_src_behav(eid)
        mouse = payload.get("mouse_id")
        lab = payload.get("lab")
        line = int(payload.get("line_hz") or 60)
        blob["sessions"].append(
            {
                "eid": eid[:8],
                "mouse": mouse,
                "lab": lab,
                "line_hz": line,
                "n_trials": payload["n_trials"],
                "n_channels": payload["n_channels"],
            }
        )
        trials = sorted(
            payload["trials"],
            key=lambda r: (
                -int(r.get("n_bins") or 0),
                0 if r.get("firstMovement_times") else 1,
            ),
        )
        acr = trials[0]["channel_acronym"]
        axial = trials[0]["channel_axial_um"]
        motor = motor_mask(acr)
        chans = pick_ch(acr, axial)
        mop = np.array([str(a).startswith("MOp") for a in acr])
        mos = np.array([str(a).startswith("MOs") for a in acr])

        # traces + psd from first few trials with voltage
        n_tr_spec = 0
        n_tr_trace = 0
        n_psd = 0
        n_example = 0
        print(
            f"  {mouse} MOp={int(mop.sum())} MOs={int(mos.sum())} motor={int(motor.sum())} chans={chans}"
        )
        for rec in trials:
            if rec.get("lfp_car") is None:
                continue
            src_rec = by_idx.get(int(rec["trial_index"]))
            t = np.asarray(rec["t250"], dtype=float)
            car = np.asarray(rec["lfp_car"], dtype=float)
            raw = np.asarray(rec.get("lfp_unref", car), dtype=float)
            if car.ndim != 2 or car.shape[0] < 32:
                continue
            paw = None if src_rec is None else src_rec.get("lp_speed", src_rec.get("dlc_speed"))
            wheel = None if src_rec is None else src_rec.get("wheel_velocity")
            if n_tr_trace < 1:
                for ci, region, ac, axu in chans:
                    tt, rr = downsample(t, raw[:, ci])
                    _, cc = downsample(t, car[:, ci])
                    item = {
                        "mouse": mouse,
                        "eid": eid[:8],
                        "trial_index": int(rec["trial_index"]),
                        "region": region,
                        "acronym": ac,
                        "axial_um": axu,
                        "line_hz": line,
                        "t": [round(float(v), 3) for v in tt],
                        "raw": [round(float(v), 1) for v in rr],
                        "car": [round(float(v), 1) for v in cc],
                    }
                    if paw is not None:
                        item["paw"] = [round(float(v), 1) for v in np.nan_to_num(np.asarray(paw, dtype=float)[: rec["n_bins"]], nan=0)]
                    blob["traces"].append(item)
                n_tr_trace += 1
            if n_example < 1:
                ex = example_from_trial(rec, mouse, lab, line, chans, motor, acr)
                if ex is not None:
                    blob["example_trials"].append(ex)
                    n_example += 1
            if n_psd < 1:
                for ci, region, ac, axu in chans:
                    if raw.shape[0] < 128:
                        continue
                    f, p_raw = welch(raw[:, ci], fs=FS, nperseg=min(512, raw.shape[0] // 2 * 2 or 128))
                    _, p_car = welch(car[:, ci], fs=FS, nperseg=min(512, raw.shape[0] // 2 * 2 or 128))
                    keep = f <= 120
                    blob["psd"].append(
                        {
                            "mouse": mouse,
                            "eid": eid[:8],
                            "region": region,
                            "acronym": ac,
                            "line_hz": line,
                            "f": [round(float(v), 2) for v in f[keep]],
                            "raw": [round(float(v), 3) for v in log10p(p_raw[keep])],
                            "car": [round(float(v), 3) for v in log10p(p_car[keep])],
                        }
                    )
                n_psd += 1
            if n_tr_spec < 2:
                stim = float(rec["stimOn_times"])
                go = rec.get("goCue_times")
                mv = rec.get("firstMovement_times")
                fb = rec.get("feedback_times")
                ci = chans[0][0]
                y = car[:, ci]
                t_abs = t + stim
                for name, tev in (
                    ("stimOn", stim),
                    ("goCue", go),
                    ("firstMovement", mv),
                    ("feedback", fb),
                ):
                    spec = spec_around(t_abs, y, tev)
                    if spec is None:
                        continue
                    spec.update(
                        {
                            "event": name,
                            "mouse": mouse,
                            "eid": eid[:8],
                            "region": chans[0][1],
                            "line_hz": line,
                        }
                    )
                    blob["spectrograms"].append(spec)
                n_tr_spec += 1

            # epochs + assoc
            n_bins = int(rec["n_bins"])
            base, prep, mov = epoch_slices(
                n_bins,
                float(rec["stimOn_times"]),
                rec.get("goCue_times"),
                rec.get("firstMovement_times"),
                rec.get("feedback_times"),
            )
            paw_v = np.asarray(paw, dtype=float) if paw is not None else None
            wheel_v = np.asarray(wheel, dtype=float) if wheel is not None else None
            for ref, key20, key100 in (
                ("car", "power_car_s20", "power_car_s100"),
                ("bip", "power_bip_s20", "power_bip_s100"),
            ):
                for sm, key in (("s20", key20), ("s100", key100)):
                    P = rec.get(key)
                    if P is None:
                        continue
                    P = np.asarray(P, dtype=np.float64)
                    if P.ndim != 3:
                        continue
                    nch = min(P.shape[1], motor.size)
                    msk = motor[:nch]
                    if not np.any(msk):
                        msk = np.ones(nch, dtype=bool)
                    logp = log10p(P[:, :nch, :])
                    trial_band = np.nanmean(logp[:, msk, :], axis=1)
                    for bi, (bn, _, _) in enumerate(BANDS):
                        for ep, sl in (
                            ("baseline", base),
                            ("preparation", prep),
                            ("movement", mov),
                        ):
                            if sl is not None and np.any(sl):
                                epoch_acc[(bn, ep, ref, sm)].append(
                                    float(np.nanmean(trial_band[sl, bi]))
                                )
                    if paw_v is not None and wheel_v is not None:
                        n = min(n_bins, trial_band.shape[0], paw_v.size, wheel_v.size)
                        assoc_rows.append(
                            {
                                "eid": eid[:8],
                                "mouse": mouse,
                                "line_hz": line,
                                "ref": ref,
                                "smooth": sm,
                                "band_mean": np.nanmean(trial_band[:n], axis=0).tolist(),
                                "paw": float(np.nanmean(np.abs(paw_v[:n]))),
                                "wheel": float(np.nanmean(np.abs(wheel_v[:n]))),
                                "move_frac": float(np.mean(mov[:n])) if mov.size else 0.0,
                            }
                        )

        # spatial mean log power along probe (movement bins, CAR s20 high gamma / beta)
        spat_acc = {bn: [] for bn, _, _ in BANDS}
        n_sp = 0
        for rec in trials:
            P = rec.get("power_car_s20")
            if P is None:
                continue
            P = np.asarray(P, dtype=np.float64)
            n_bins = int(rec["n_bins"])
            _, _, mov = epoch_slices(
                n_bins,
                float(rec["stimOn_times"]),
                rec.get("goCue_times"),
                rec.get("firstMovement_times"),
                rec.get("feedback_times"),
            )
            sl = mov if np.any(mov) else np.ones(n_bins, dtype=bool)
            sl = sl[: P.shape[0]]
            spat_acc_trial = np.nanmean(log10p(P[sl]), axis=0)
            for bi, (bn, _, _) in enumerate(BANDS):
                spat_acc[bn].append(spat_acc_trial[:, bi])
            n_sp += 1
            if n_sp >= 40:
                break
        if n_sp:
            order = np.argsort(axial)
            step = max(1, order.size // 96)
            order = order[::step]
            for bn, _, _ in BANDS:
                mu = np.nanmean(np.stack(spat_acc[bn], axis=0), axis=0)
                spatial_rows.append(
                    {
                        "mouse": mouse,
                        "eid": eid[:8],
                        "band": bn,
                        "axial": [round(float(v), 0) for v in np.asarray(axial)[order]],
                        "power": [round(float(v), 3) for v in mu[order]],
                        "region": [
                            "MOp" if mop[i] else ("MOs" if mos[i] else "other")
                            for i in order
                        ],
                    }
                )

    for (bn, ep, ref, sm), vals in epoch_acc.items():
        if not vals:
            continue
        arr = np.asarray(vals, dtype=float)
        blob["epoch_power"].append(
            {
                "band": bn,
                "epoch": ep,
                "ref": ref,
                "smooth": sm,
                "mean": round(float(np.nanmean(arr)), 3),
                "sem": round(float(np.nanstd(arr) / max(np.sqrt(arr.size), 1)), 3),
                "n": int(np.isfinite(arr).sum()),
            }
        )

    blob["spatial"] = spatial_rows

    # associations across trials
    if assoc_rows:
        from collections import defaultdict

        groups = defaultdict(list)
        for row in assoc_rows:
            groups[(row["ref"], row["smooth"])].append(row)
        rng = np.random.default_rng(0)
        for (ref, sm), rows in groups.items():
            paw = np.array([r["paw"] for r in rows], dtype=float)
            wheel = np.array([r["wheel"] for r in rows], dtype=float)
            eids = np.array([r["eid"] for r in rows])
            band_mat = np.array([r["band_mean"] for r in rows], dtype=float)
            for bi, (bn, _, _) in enumerate(BANDS):
                x = band_mat[:, bi]
                r_paw = safe_r(x, paw)
                r_wheel = safe_r(x, wheel)
                shuf = []
                for _ in range(80):
                    rr = safe_r(x, rng.permutation(paw))
                    if rr:
                        shuf.append(rr["r"])
                # session identity negative control: permute by session block
                u_e = np.unique(eids)
                x_swap = x.copy()
                if u_e.size >= 2:
                    perm = rng.permutation(u_e)
                    mapping = dict(zip(u_e, perm))
                    # swap means: assign each session the band values of another
                    for i, e in enumerate(eids):
                        donor = mapping[e]
                        donors = np.flatnonzero(eids == donor)
                        x_swap[i] = x[rng.choice(donors)]
                r_sess = safe_r(x_swap, paw)
                blob["assoc"].append(
                    {
                        "band": bn,
                        "ref": ref,
                        "smooth": sm,
                        "r_paw": None if r_paw is None else round(r_paw["r"], 3),
                        "r_wheel": None if r_wheel is None else round(r_wheel["r"], 3),
                        "r_paw_shuffle_mean": None
                        if not shuf
                        else round(float(np.mean(shuf)), 3),
                        "r_paw_shuffle_abs95": None
                        if not shuf
                        else round(float(np.quantile(np.abs(shuf), 0.95)), 3),
                        "r_session_swap": None if r_sess is None else round(r_sess["r"], 3),
                        "n": int(len(rows)),
                    }
                )
            # ref/smooth comparison on high gamma vs paw
            hg = band_mat[:, -1]
            blob["ref_smooth"].append(
                {
                    "ref": ref,
                    "smooth": sm,
                    "high_gamma_paw_r": None if safe_r(hg, paw) is None else round(safe_r(hg, paw)["r"], 3),
                    "beta_paw_r": None if safe_r(band_mat[:, 3], paw) is None else round(safe_r(band_mat[:, 3], paw)["r"], 3),
                    "n": int(len(rows)),
                }
            )

    OUT.write_text(json.dumps(blob))
    print(
        "wrote",
        OUT,
        "sessions",
        len(blob["sessions"]),
        "traces",
        len(blob["traces"]),
        "psd",
        len(blob["psd"]),
        "spec",
        len(blob["spectrograms"]),
        "epoch",
        len(blob["epoch_power"]),
        "spatial",
        len(blob["spatial"]),
        "assoc",
        len(blob["assoc"]),
        "examples",
        len(blob["example_trials"]),
    )


if __name__ == "__main__":
    main()
