#!/usr/bin/env python3
"""Stream IBL LFP for ModelDataRightContra trials and pickle band-power features.

Trial index matches ModelDataRightContra (eid, trial_index, n_bins at 20 ms).
AP-band 30 kHz continuous is not stored (terabytes). LFP is the continuous
channel voltage; canonical band power is the model-ready extra input.
"""
from __future__ import annotations

import argparse
import json
import pickle
import shutil
import traceback
from datetime import datetime, timezone
from pathlib import Path
from pathlib import PurePosixPath
from urllib.parse import urlparse

import numpy as np
import pandas as pd
from scipy.signal import (
    butter,
    iirnotch,
    resample_poly,
    sosfiltfilt,
    spectrogram,
    tf2sos,
    welch,
)
from scipy.ndimage import gaussian_filter1d

ROOT = Path(__file__).resolve().parent
SRC = ROOT / "ModelDataRightContra"
OUT = ROOT / "ModelDataRightContraLFP"
SESS_OUT = OUT / "sessions"
LOG = OUT / "build.log"

BINSIZE = 0.02
FS_LFP = 250.0
FS_POW = 50.0
PAD_S = 4.0
BATCH_SPAN_S = 80.0
HP_HZ = 0.1
BUTTER_ORDER = 4
SMOOTH_SHORT_S = 0.02
SMOOTH_LONG_S = 0.10

BANDS = [
    ("delta", 0.5, 4.0),
    ("theta", 4.0, 8.0),
    ("alpha", 8.0, 13.0),
    ("beta", 13.0, 30.0),
    ("low_gamma", 30.0, 55.0),
    ("high_gamma", 65.0, 100.0),
]

# IBL Alyx `lab` → mains frequency. US 60 Hz, Europe 50 Hz.
# angelakilab: NYU (New York). churchlandlab: CSHL. churchlandlab_ucla: UCLA.
# danlab: Yang Dan, UC Berkeley. zadorlab: CSHL. steinmetzlab: UW Seattle (NR_0027 2022).
# cortexlab / hausserlab: UCL London. hoferlab / mrsicflogellab: SWC London.
# mainenlab: Champalimaud, Lisbon.
LINE_HZ = {
    "angelakilab": 60,
    "churchlandlab": 60,
    "churchlandlab_ucla": 60,
    "danlab": 60,
    "zadorlab": 60,
    "steinmetzlab": 60,
    "cortexlab": 50,
    "hausserlab": 50,
    "hoferlab": 50,
    "mrsicflogellab": 50,
    "mainenlab": 50,
}

INSPECT_EIDS = [
    "03063955-2523-47bd-ae57-f7489dd40f15",  # SWC_038 UK 50 Hz
    "8c33abef-3d3e-4d42-9f27-445e9def08f9",  # US 60 Hz
    "d2918f52-8280-43c0-924b-029b2317e62c",
    "a4000c2f-fa75-4b3e-8f06-a7cf599b87ad",
    "81a78eac-9d36-4f90-a73a-7eb3ad7f770b",
    "2c44a360-5a56-4971-8009-f469fb59de98",
    "aa3432cd-62bd-40bc-bc1c-a12d53bcbdcf",
]

INSPECT_PREFIXES = (
    "03063955",
    "8c33abef",
    "d2918f52",
    "a4000c2f",
    "81a78eac",
    "2c44a360",
    "aa3432cd",
)


def utc_now():
    return datetime.now(timezone.utc).isoformat()


def log(msg):
    OUT.mkdir(parents=True, exist_ok=True)
    line = f"[{datetime.now().strftime('%H:%M:%S')}] {msg}"
    print(line, flush=True)
    with LOG.open("a") as fh:
        fh.write(line + "\n")


def line_hz_for(lab):
    lab = str(lab or "").strip()
    if lab in LINE_HZ:
        return LINE_HZ[lab]
    low = lab.lower()
    if any(k in low for k in ("ucla", "cshl", "nyu", "princeton", "berkeley", "zador", "angelaki", "churchland", "steinmetz", "dan")):
        return 60
    if any(k in low for k in ("ucl", "swc", "hofer", "mrsic", "hausser", "cortex", "mainen", "champalimaud", "witten")):
        return 50
    return 60


def connect_one():
    from one.api import ONE

    kwargs = dict(
        base_url="https://openalyx.internationalbrainlab.org",
        password="international",
    )
    try:
        return ONE(silent=True, **kwargs)
    except TypeError:
        return ONE(**kwargs)


def is_motor(acronym):
    a = str(acronym or "")
    return a.startswith("MOp") or a.startswith("MOs")


def butter_sos(kind, fs, low=None, high=None, order=BUTTER_ORDER):
    nyq = 0.5 * fs
    if kind == "highpass":
        wn = min(max(low / nyq, 1e-6), 0.99)
        return butter(order, wn, btype="highpass", output="sos")
    if kind == "lowpass":
        wn = min(max(high / nyq, 1e-6), 0.99)
        return butter(order, wn, btype="lowpass", output="sos")
    lo = max(low / nyq, 1e-6)
    hi = min(high / nyq, 0.99)
    if not (lo < hi):
        raise ValueError(f"bad band {low}-{high} at fs={fs}")
    return butter(order, (lo, hi), btype="bandpass", output="sos")


def filtfilt_sos(sos, x):
    if x.shape[0] < 24:
        return np.full_like(x, np.nan, dtype=np.float64)
    return sosfiltfilt(sos, x, axis=0, padtype="odd")


def notch_line(x, fs, f0):
    out = np.asarray(x, dtype=np.float64)
    for harm in (f0, 2 * f0):
        if harm >= 0.45 * fs:
            continue
        b, a = iirnotch(harm, Q=30, fs=fs)
        sos = tf2sos(b, a)
        out = filtfilt_sos(sos, out)
    return out


def resample_to(x, fs_in, fs_out):
    fs_in = float(fs_in)
    fs_out = float(fs_out)
    if abs(fs_in - fs_out) < 1e-6:
        return np.asarray(x, dtype=np.float64)
    up = int(round(fs_out))
    down = int(round(fs_in))
    g = np.gcd(up, down)
    return resample_poly(np.asarray(x, dtype=np.float64), up // g, down // g, axis=0)


def hilbert_power(x):
    from scipy.signal import hilbert

    env = np.abs(hilbert(x, axis=0))
    return np.square(env)


def gaussian_smooth(x, fs, sigma_s):
    if sigma_s <= 0:
        return x
    sigma = max(sigma_s * fs, 0.5)
    return gaussian_filter1d(x, sigma=sigma, axis=0, mode="nearest")


def bin_mean(x, t, t_left, t_right):
    n = int(t_left.size)
    c = x.shape[1] if x.ndim == 2 else 1
    out = np.full((n, c) if x.ndim == 2 else (n,), np.nan, dtype=np.float32)
    if t.size == 0:
        return out
    for i in range(n):
        m = (t >= t_left[i]) & (t < t_right[i])
        if not np.any(m):
            continue
        out[i] = np.nanmean(x[m], axis=0)
    return out


def car_median(x):
    x = np.asarray(x, dtype=np.float64)
    good = np.nanstd(x, axis=0) > 1e-9
    if not np.any(good):
        return x
    med = np.nanmedian(x[:, good], axis=1, keepdims=True)
    out = x - med
    out[:, ~good] = np.nan
    return out


def bipolar_matrix(axial, lateral):
    axial = np.asarray(axial, dtype=float)
    lateral = np.asarray(lateral, dtype=float)
    pairs = []
    for lat in np.unique(np.round(lateral, 0)):
        idx = np.flatnonzero(np.abs(lateral - lat) < 1.0)
        if idx.size < 2:
            continue
        order = idx[np.argsort(axial[idx])]
        for a, b in zip(order[:-1], order[1:]):
            if 5.0 < abs(axial[b] - axial[a]) <= 45.0:
                pairs.append((int(a), int(b)))
    return pairs


def apply_bipolar(x, pairs):
    if not pairs:
        return np.zeros((x.shape[0], 0), dtype=np.float64)
    cols = []
    for a, b in pairs:
        cols.append(x[:, b] - x[:, a])
    return np.stack(cols, axis=1)


def f16(x):
    x = np.asarray(x, dtype=np.float32)
    return np.clip(x, -np.finfo(np.float16).max / 4, np.finfo(np.float16).max / 4).astype(
        np.float16
    )


def load_channel_table(ssl):
    ch = ssl.load_channels()
    n = len(ch.get("rawInd", ch.get("x", [])))
    raw = np.asarray(ch.get("rawInd", np.arange(n)), dtype=int)
    acr = np.asarray(ch.get("acronym", ["unknown"] * n)).astype(str)
    axial = np.asarray(ch.get("axial_um", ch.get("y", np.zeros(n))), dtype=float)
    lateral = np.asarray(ch.get("lateral_um", ch.get("x", np.zeros(n))), dtype=float)
    return raw, acr, axial, lateral


def stream_window(sr, t0, t1):
    fs = float(sr.fs)
    i0 = int(max(0, np.floor(t0 * fs)))
    i1 = int(min(sr.ns, np.ceil(t1 * fs)))
    if i1 - i0 < 32:
        return None, fs
    nsync = int(getattr(sr, "nsync", 1) or 0)
    data = np.asarray(sr[i0:i1, :], dtype=np.float64)
    if nsync > 0 and data.shape[1] > nsync:
        data = data[:, :-nsync]
    t = (np.arange(i0, i0 + data.shape[0]) / fs)
    return (t, data), fs


def preprocess_voltage(x, fs_in, line_hz):
    """HP-ready 250 Hz traces: anti-alias to 250 Hz, 0.1 Hz HP, line notch."""
    sos_aa = butter_sos("lowpass", fs_in, high=min(120.0, 0.45 * fs_in))
    y = filtfilt_sos(sos_aa, x)
    y = resample_to(y, fs_in, FS_LFP)
    y = filtfilt_sos(butter_sos("highpass", FS_LFP, low=HP_HZ), y)
    y = notch_line(y, FS_LFP, line_hz)
    return y * 1e6  # volts → µV so squared Hilbert power fits float16


def band_envelopes(x250):
    out = []
    for name, lo, hi in BANDS:
        sos = butter_sos("bandpass", FS_LFP, low=lo, high=hi)
        xf = filtfilt_sos(sos, x250)
        out.append(hilbert_power(xf))
    return np.stack(out, axis=-1)


def power_on_bins(env250, t250, t_left, t_right, sigma_s):
    env = gaussian_smooth(env250, FS_LFP, sigma_s)
    n_bins = int(t_left.size)
    n_ch = env.shape[1]
    n_b = env.shape[2]
    out = np.full((n_bins, n_ch, n_b), np.nan, dtype=np.float32)
    for b in range(n_b):
        out[:, :, b] = bin_mean(env[:, :, b], t250, t_left, t_right)
    return out


def trial_batches(recs, pad=PAD_S, max_span=BATCH_SPAN_S):
    items = []
    for i, rec in enumerate(recs):
        t0 = float(rec["stimOn_times"])
        n = int(rec.get("n_bins") or 0)
        t1 = t0 + max(n, 1) * BINSIZE
        items.append((t0, t1, i))
    items.sort()
    batches = []
    cur = []
    start = None
    for t0, t1, i in items:
        a, b = t0 - pad, t1 + 1.0
        if not cur:
            cur = [(a, b, i)]
            start = a
            continue
        if b - start <= max_span:
            cur.append((a, b, i))
        else:
            batches.append(cur)
            cur = [(a, b, i)]
            start = a
    if cur:
        batches.append(cur)
    return batches


def session_ok(eid):
    p = SESS_OUT / f"{eid}.ok.json"
    if not p.exists():
        return False
    try:
        d = json.loads(p.read_text())
        return int(d.get("n_trials") or 0) > 0
    except Exception:
        return False


def resolve_s3_cbin(one, eid, pname, typ="lf"):
    """OpenAlyx LFP lives on S3; FlatIron HTTP_DATA_SERVER is dead (connection refused)."""
    from one.alf.path import add_uuid_string
    from one.remote import aws

    dsets = one.list_datasets(
        eid, collection=f"*{pname}", filename=f"*{typ}.*bin", details=True
    )
    if dsets is None or len(dsets) == 0:
        raise FileNotFoundError(f"no {typ} cbin for {eid} {pname}")
    row = dsets.iloc[0]
    did = row.name[-1] if isinstance(row.name, tuple) else row.name
    rec = one.alyx.rest("datasets", "read", id=str(did))
    aws_fr = next(
        (
            x
            for x in rec["file_records"]
            if str(x.get("data_repository") or "").startswith("aws") and x.get("exists")
        ),
        None,
    )
    if aws_fr is None:
        raise FileNotFoundError(f"no aws file record for {eid} {pname} {typ}")
    key = add_uuid_string(
        PurePosixPath(aws_fr.get("data_repository_path") or "", aws_fr["relative_path"]),
        did,
    )
    s3, bucket = aws.get_s3_public()
    return s3, bucket, str(key).lstrip("/")


class S3Streamer:
    """brainbox Streamer that range-reads mtscomp chunks from public S3."""

    def __init__(self, pid, one, typ="lf", remove_cached=False):
        from brainbox.io.spikeglx import Streamer

        sr = Streamer(pid=pid, one=one, typ=typ, remove_cached=remove_cached)
        s3, bucket, key = resolve_s3_cbin(one, sr.eid, sr.pname, typ)
        sr._download_raw_partial = _s3_download_raw_partial(sr, s3, bucket, key)
        self._sr = sr
        log(f"    S3 {bucket}/{key}")

    def __getattr__(self, name):
        return getattr(self._sr, name)

    def __getitem__(self, item):
        return self._sr[item]


def _s3_download_raw_partial(sr, s3, bucket, key):
    def _download_raw_partial(first_chunk=0, last_chunk=0):
        import json as _json
        import shutil
        import time
        import string
        import random
        from one.alf.path import remove_uuid_string
        import spikeglx

        webclient = sr.one.alyx
        relpath = Path(
            sr.url_cbin.replace(webclient._par.HTTP_DATA_SERVER, ".")
        ).parents[0]
        tdir_chunk = f"chunk_{str(first_chunk).zfill(6)}_to_{str(last_chunk).zfill(6)}"
        if sr.remove_cached:
            tdir_chunk += "".join(
                [random.choice(string.ascii_letters) for _ in np.arange(10)]
            )
        sr.target_dir = Path(sr.cache_folder, relpath, tdir_chunk)
        Path(sr.target_dir).mkdir(parents=True, exist_ok=True)
        ch_file_stream = sr.target_dir.joinpath(sr.file_chunks.name).with_suffix(
            ".stream.ch"
        )
        i0 = sr.chunks["chunk_bounds"][first_chunk]
        ns_stream = sr.chunks["chunk_bounds"][last_chunk + 1] - i0
        total_samples = sr.chunks["chunk_bounds"][-1]
        meta_local_path = ch_file_stream.with_suffix(".meta")
        if not meta_local_path.exists():
            shutil.copy(sr.file_chunks.with_suffix(".meta"), meta_local_path)
        if ch_file_stream.exists() and ch_file_stream.with_suffix(".cbin").exists():
            with open(ch_file_stream, "r") as f:
                cmeta_stream = _json.load(f)
            if (
                cmeta_stream.get("chopped_first_sample", None) == i0
                and cmeta_stream.get("chopped_total_samples", None) == total_samples
            ):
                return (
                    spikeglx.Reader(
                        ch_file_stream.with_suffix(".cbin"), ignore_warnings=True
                    ),
                    ch_file_stream,
                )
        else:
            shutil.copy(sr.file_chunks, ch_file_stream)

        cmeta = sr.chunks.copy()
        cmeta["chunk_bounds"] = [
            int(_ - i0) for _ in cmeta["chunk_bounds"][first_chunk : last_chunk + 2]
        ]
        first_byte = sr.chunks["chunk_offsets"][first_chunk]
        cmeta["chunk_offsets"] = [
            _ - first_byte
            for _ in sr.chunks["chunk_offsets"][first_chunk : last_chunk + 2]
        ]
        n_bytes = cmeta["chunk_offsets"][-1]
        cmeta["sha1_compressed"] = None
        cmeta["sha1_uncompressed"] = None
        cmeta["chopped"] = True
        cmeta["chopped_first_sample"] = int(i0)
        cmeta["chopped_samples"] = int(ns_stream)
        cmeta["chopped_total_samples"] = int(total_samples)
        with open(ch_file_stream, "w") as f:
            _json.dump(cmeta, f, indent=2, sort_keys=True)

        name = Path(urlparse(sr.url_cbin).path).name
        cbin_local_path = Path(sr.target_dir) / name
        retries = 0
        while True:
            try:
                obj = s3.Object(bucket, key)
                resp = obj.get(
                    Range=f"bytes={int(first_byte)}-{int(first_byte + n_bytes - 1)}"
                )
                cbin_local_path.write_bytes(resp["Body"].read())
                break
            except Exception as e:
                retries += 1
                if retries > 5:
                    raise e
                time.sleep(min(2 ** retries, 16))
        cbin_local_path_renamed = remove_uuid_string(cbin_local_path).with_suffix(
            ".stream.cbin"
        )
        cbin_local_path.replace(cbin_local_path_renamed)
        reader = spikeglx.Reader(cbin_local_path_renamed, ignore_warnings=True)
        return reader, cbin_local_path

    return _download_raw_partial


def write_session_payload(eid, payload):
    SESS_OUT.mkdir(parents=True, exist_ok=True)
    path = SESS_OUT / f"{eid}.pkl"
    tmp = SESS_OUT / f"{eid}.pkl.tmp"
    with tmp.open("wb") as fh:
        pickle.dump(payload, fh, protocol=4)
    tmp.replace(path)
    ok = {
        "eid": eid,
        "n_trials": payload["n_trials"],
        "n_channels": payload["n_channels"],
        "written_at": utc_now(),
        "line_hz": payload["line_hz"],
    }
    (SESS_OUT / f"{eid}.ok.json").write_text(json.dumps(ok, indent=2))
    return path


def load_go_cue(one, eid, recs):
    go = {}
    try:
        tr = one.load_object(eid, "trials")
        arr = np.asarray(tr.get("goCue_times", []), dtype=float)
        for rec in recs:
            idx = rec.get("trial_index")
            if idx is None or arr.size == 0:
                continue
            i = int(idx)
            if 0 <= i < arr.size:
                go[i] = float(arr[i]) if np.isfinite(arr[i]) else None
    except Exception as exc:
        log(f"  goCue unavailable: {exc}")
    return go


def downsample_trace(t, y, n=220):
    t = np.asarray(t, dtype=float)
    y = np.asarray(y, dtype=float)
    if t.size <= n:
        return t.tolist(), np.nan_to_num(y, nan=0.0).tolist()
    idx = np.unique(np.round(np.linspace(0, t.size - 1, n)).astype(int))
    return t[idx].tolist(), np.nan_to_num(y[idx], nan=0.0).tolist()


def pick_channels(acr, axial, n_each=1):
    mop = np.flatnonzero(np.array([str(a).startswith("MOp") for a in acr]))
    mos = np.flatnonzero(np.array([str(a).startswith("MOs") for a in acr]))
    other = np.flatnonzero(~np.array([is_motor(a) for a in acr]))
    chosen = []
    for pool, name in ((mop, "MOp"), (mos, "MOs"), (other, "other")):
        if pool.size == 0:
            continue
        order = pool[np.argsort(axial[pool])]
        mid = order[len(order) // 2]
        chosen.append((int(mid), name))
    return chosen


def inspect_append(blob, rec, t250, raw250, car250, bip250, acr, axial, pairs, paw, wheel, line_hz, mouse, eid, probe):
    chans = pick_channels(acr, axial)
    t_rel = t250 - float(rec["stimOn_times"])
    keep = (t_rel >= -0.2) & (t_rel <= min(2.4, int(rec.get("n_bins") or 1) * BINSIZE + 0.2))
    if keep.sum() < 20:
        return
    t_rel = t_rel[keep]
    for ci, region in chans:
        raw_c = raw250[keep, ci]
        car_c = car250[keep, ci]
        bip_c = None
        for pi, (a, b) in enumerate(pairs):
            if a == ci or b == ci:
                bip_c = bip250[keep, pi] if bip250.size else None
                break
        tt, rr = downsample_trace(t_rel, raw_c)
        _, cc = downsample_trace(t_rel, car_c)
        item = {
            "mouse": mouse,
            "eid": eid[:8],
            "trial_index": int(rec.get("trial_index")),
            "n_bins": int(rec.get("n_bins") or 0),
            "lab_line_hz": line_hz,
            "probe": probe,
            "region": region,
            "acronym": str(acr[ci]),
            "axial_um": float(axial[ci]),
            "t": tt,
            "raw": rr,
            "car": cc,
        }
        if bip_c is not None:
            _, bb = downsample_trace(t_rel, bip_c)
            item["bipolar"] = bb
        if paw is not None and np.size(paw):
            item["paw_last"] = float(np.nanmean(np.abs(paw[-5:])))
            item["paw_peak"] = float(np.nanmax(np.abs(paw)))
        blob.setdefault("traces", []).append(item)


def inspect_psd(blob, raw250, car250, bip250, acr, axial, line_hz, mouse, eid):
    chans = pick_channels(acr, axial)
    f = None
    for ci, region in chans:
        def _w(x):
            if x.shape[0] < 256:
                return None, None
            ff, pxx = welch(x[:, ci], fs=FS_LFP, nperseg=min(1024, x.shape[0] // 2 * 2 or 256))
            return ff, np.log10(np.maximum(pxx, 1e-20))

        f, p_raw = _w(raw250)
        _, p_car = _w(car250)
        if f is None:
            continue
        blob.setdefault("psd", []).append(
            {
                "mouse": mouse,
                "eid": eid[:8],
                "region": region,
                "acronym": str(acr[ci]),
                "line_hz": line_hz,
                "f": [round(float(v), 2) for v in f if v <= 120],
                "raw": [round(float(v), 3) for v, fv in zip(p_raw, f) if fv <= 120],
                "car": [round(float(v), 3) for v, fv in zip(p_car, f) if fv <= 120],
            }
        )


def process_probe(sr, ssl, recs, go, line_hz, inspect_blob, mouse, eid, pname, units_motor_axial=None):
    raw_ind, acr, axial, lateral = load_channel_table(ssl)
    n_ch_meta = raw_ind.size
    pairs = bipolar_matrix(axial, lateral)
    batches = trial_batches(recs)
    trial_out = {i: {} for i in range(len(recs))}
    log(f"    {pname}: {n_ch_meta} channels, {len(pairs)} bipolar pairs, {len(batches)} stream batches")

    for bi, batch in enumerate(batches):
        t_lo = min(a for a, b, i in batch)
        t_hi = max(b for a, b, i in batch)
        t_lo = max(0.0, t_lo)
        try:
            streamed, fs_in = stream_window(sr, t_lo, t_hi)
        except Exception as exc:
            log(f"    stream fail {pname} batch {bi}: {exc}")
            continue
        if streamed is None:
            continue
        t_abs, volt = streamed
        n_use = min(volt.shape[1], n_ch_meta)
        volt = volt[:, :n_use]
        raw_ind_u = raw_ind[:n_use]
        acr_u = acr[:n_use]
        axial_u = axial[:n_use]
        lateral_u = lateral[:n_use]
        pairs_u = [(a, b) for a, b in pairs if a < n_use and b < n_use]

        idx = np.clip(raw_ind_u, 0, volt.shape[1] - 1)
        volt = volt[:, idx]
        raw250 = preprocess_voltage(volt, fs_in, line_hz)
        t250 = np.linspace(t_abs[0], t_abs[-1], raw250.shape[0])
        car250 = car_median(raw250)
        bip250 = apply_bipolar(raw250, pairs_u)

        env_car = band_envelopes(car250)
        env_bip = band_envelopes(bip250) if bip250.shape[1] else None

        for _, _, ti in batch:
            rec = recs[ti]
            t0 = float(rec["stimOn_times"])
            n_bins = int(rec.get("n_bins") or 0)
            if n_bins < 1:
                continue
            t_left = np.arange(n_bins, dtype=np.float64) * BINSIZE
            t_right = t_left + BINSIZE
            t_rel = t250 - t0
            m = (t_rel >= -0.05) & (t_rel <= n_bins * BINSIZE + 0.05)
            if m.sum() < 8:
                continue
            paw = rec.get("lp_speed", rec.get("dlc_speed"))
            wheel = rec.get("wheel_velocity")
            p_car_s20 = power_on_bins(env_car[m], t_rel[m], t_left, t_right, 0.0)
            p_car_s100 = power_on_bins(env_car[m], t_rel[m], t_left, t_right, SMOOTH_LONG_S)
            pack = {
                "t250": np.asarray(t_rel[m], dtype=np.float32),
                "lfp_car": f16(car250[m]),
                "lfp_unref": f16(raw250[m]),
                "power_car_s20": f16(p_car_s20),
                "power_car_s100": f16(p_car_s100),
            }
            if env_bip is not None and env_bip.size:
                pack["power_bip_s20"] = f16(
                    power_on_bins(env_bip[m], t_rel[m], t_left, t_right, 0.0)
                )
                pack["power_bip_s100"] = f16(
                    power_on_bins(env_bip[m], t_rel[m], t_left, t_right, SMOOTH_LONG_S)
                )
            trial_out[ti][pname] = pack

            if inspect_blob is not None and eid.startswith(INSPECT_PREFIXES) and ti < 6:
                inspect_append(
                    inspect_blob,
                    rec,
                    t250[m],
                    raw250[m],
                    car250[m],
                    bip250[m] if bip250.size else np.zeros((m.sum(), 0)),
                    acr_u,
                    axial_u,
                    pairs_u,
                    paw,
                    wheel,
                    line_hz,
                    mouse,
                    eid,
                    pname,
                )
                if len(inspect_blob.get("psd", [])) < 18:
                    inspect_psd(inspect_blob, raw250[m], car250[m], bip250[m] if bip250.size else raw250[m], acr_u, axial_u, line_hz, mouse, eid)

        del volt, raw250, car250, bip250, env_car, env_bip

    return {
        "probe": pname,
        "acronym": acr,
        "axial_um": axial.astype(np.float32),
        "lateral_um": lateral.astype(np.float32),
        "raw_ind": raw_ind.astype(np.int16),
        "motor": np.array([is_motor(a) for a in acr]),
        "pairs": pairs,
        "trials": trial_out,
    }


def merge_probe_trials(recs, probe_packs, go):
    trials = []
    for ti, rec in enumerate(recs):
        n_bins = int(rec.get("n_bins") or 0)
        parts = []
        acr = []
        axial = []
        lateral = []
        motor = []
        probes = []
        t250 = None
        lfp_car = []
        lfp_unref = []
        pow20 = []
        pow100 = []
        bip20 = []
        bip100 = []
        for pack in probe_packs:
            slot = pack["trials"].get(ti, {}).get(pack["probe"])
            if not slot:
                continue
            parts.append(pack["probe"])
            acr.append(pack["acronym"])
            axial.append(pack["axial_um"])
            lateral.append(pack["lateral_um"])
            motor.append(pack["motor"])
            probes.extend([pack["probe"]] * pack["acronym"].size)
            t250 = slot["t250"]
            lfp_car.append(slot["lfp_car"])
            lfp_unref.append(slot["lfp_unref"])
            pow20.append(slot["power_car_s20"])
            pow100.append(slot["power_car_s100"])
            if "power_bip_s20" in slot:
                bip20.append(slot["power_bip_s20"])
                bip100.append(slot["power_bip_s100"])
        if not pow20:
            continue

        def cat_ch(xs, ax=1):
            nT = min(x.shape[0] for x in xs)
            return np.concatenate([x[:nT] for x in xs], axis=ax)

        idx = rec.get("trial_index")
        trials.append(
            {
                "eid": rec["eid"],
                "trial_index": idx,
                "n_bins": n_bins,
                "stimOn_times": rec.get("stimOn_times"),
                "feedback_times": rec.get("feedback_times"),
                "firstMovement_times": rec.get("firstMovement_times"),
                "goCue_times": go.get(int(idx) if idx is not None else -1),
                "t_bin_right": (np.arange(n_bins, dtype=np.float32) + 1) * np.float32(BINSIZE),
                "t250": np.asarray(t250)[: min(x.shape[0] for x in lfp_car)],
                "lfp_car": cat_ch(lfp_car),
                "lfp_unref": cat_ch(lfp_unref),
                "power_car_s20": cat_ch(pow20),
                "power_car_s100": cat_ch(pow100),
                "power_bip_s20": cat_ch(bip20) if bip20 else None,
                "power_bip_s100": cat_ch(bip100) if bip100 else None,
                "channel_acronym": np.concatenate(acr),
                "channel_axial_um": np.concatenate(axial),
                "channel_lateral_um": np.concatenate(lateral),
                "channel_motor": np.concatenate(motor),
                "channel_probe": np.asarray(probes, dtype=object),
                "band_names": [b[0] for b in BANDS],
            }
        )
    return trials


def process_session(one, src_path, inspect_blob, force=False, max_trials=None):
    from brainbox.io.one import SpikeSortingLoader

    with src_path.open("rb") as fh:
        src = pickle.load(fh)
    eid = str(src["eid"])
    recs = list(src["trials"])
    if max_trials:
        recs = recs[: int(max_trials)]
    if session_ok(eid) and not force:
        try:
            have = int(json.loads((SESS_OUT / f"{eid}.ok.json").read_text()).get("n_trials") or 0)
        except Exception:
            have = 0
        if have >= len(recs):
            log(f"skip {eid[:8]} already written ({have} trials)")
            return None
        log(f"reprocess {eid[:8]} incomplete ({have} < {len(recs)})")
    lab = recs[0].get("lab") if recs else src.get("lab")
    line_hz = line_hz_for(lab)
    mouse = str(src.get("mouse_id"))
    pids = list(src.get("pids") or [])
    pnames = list(src.get("probe_names") or [])
    log(f"{mouse} {eid[:8]} lab={lab} line={line_hz} Hz  trials={len(recs)} probes={pnames}")
    go = load_go_cue(one, eid, recs)
    probe_packs = []
    want_inspect = eid.startswith(INSPECT_PREFIXES)
    for pid, pname in zip(pids, pnames):
        try:
            ssl = SpikeSortingLoader(pid=pid, one=one, pname=pname, eid=eid)
            sr = S3Streamer(pid=pid, one=one, typ="lf", remove_cached=False)
            log(f"  opened LFP {pname} fs={sr.fs:.2f} ns={sr.ns} nc={sr.nc}")
            pack = process_probe(
                sr,
                ssl,
                recs,
                go,
                line_hz,
                inspect_blob if want_inspect else None,
                mouse,
                eid,
                pname,
            )
            probe_packs.append(pack)
        except Exception as exc:
            log(f"  FAILED probe {pname}: {exc}")
            traceback.print_exc()
    trials = merge_probe_trials(recs, probe_packs, go)
    if not trials:
        log(f"  SKIP write {eid[:8]}: 0 trials with LFP")
        bad_ok = SESS_OUT / f"{eid}.ok.json"
        if bad_ok.exists():
            bad_ok.unlink()
        return None
    n_ch = int(trials[0]["channel_acronym"].size)
    payload = {
        "eid": eid,
        "mouse_id": mouse,
        "lab": lab,
        "line_hz": line_hz,
        "session_date": src.get("session_date"),
        "pids": pids,
        "probe_names": pnames,
        "n_trials": len(trials),
        "n_channels": n_ch,
        "bands": BANDS,
        "fs_lfp": FS_LFP,
        "fs_power": FS_POW,
        "refs": ["car_median", "bipolar_adjacent"],
        "smooth_s": [SMOOTH_SHORT_S, SMOOTH_LONG_S],
        "trials": trials,
        "notes": {
            "voltage": "LFP streamed from OpenAlyx SpikeGLX lf.cbin via S3 byte-range. Anti-alias, resample 250 Hz, 0.1 Hz HP, lab-line notch. Stored in µV.",
            "power": "4th-order Butterworth SOS bandpass, filtfilt, squared Hilbert envelope (µV^2), mean in 20 ms bins. s20=bin mean only; s100=Gaussian σ=100 ms then bin.",
            "refs": "car_median = common-average median; bipolar_adjacent = same-column neighbors 5-45 µm",
            "index": "same eid + trial_index + n_bins as ModelDataRightContra",
            "ap_band": "not stored; 30 kHz AP is terabyte-scale",
            "voltage_units": "microvolts",
            "power_units": "microvolts^2",
        },
        "written_at": utc_now(),
    }
    path = write_session_payload(eid, payload)
    log(f"  wrote {path.name} trials={len(trials)} channels={n_ch}")
    return payload


def rebuild_index():
    rows = []
    for pkl in sorted(SESS_OUT.glob("*.pkl")):
        if pkl.name.endswith(".tmp"):
            continue
        with pkl.open("rb") as fh:
            payload = pickle.load(fh)
        for t in payload["trials"]:
            rows.append(
                {
                    "eid": payload["eid"],
                    "mouse_id": payload["mouse_id"],
                    "lab": payload.get("lab"),
                    "line_hz": payload.get("line_hz"),
                    "trial_index": t["trial_index"],
                    "n_bins": t["n_bins"],
                    "n_channels": int(np.asarray(t["channel_acronym"]).size),
                    "n_motor_channels": int(np.asarray(t["channel_motor"]).sum()),
                    "has_lfp": True,
                    "has_bipolar": t.get("power_bip_s20") is not None,
                }
            )
    df = pd.DataFrame(rows)
    OUT.mkdir(parents=True, exist_ok=True)
    df.to_parquet(OUT / "trials.parquet", index=False)
    man = {
        "built_at": utc_now(),
        "n_trials": int(len(df)),
        "n_sessions": int(df.eid.nunique()) if len(df) else 0,
        "n_mice": int(df.mouse_id.nunique()) if len(df) else 0,
        "bands": BANDS,
        "line_hz": LINE_HZ,
        "fs_lfp": FS_LFP,
        "fs_power": FS_POW,
        "source": str(SRC),
        "index": "eid + trial_index matches ModelDataRightContra",
    }
    (OUT / "manifest.json").write_text(json.dumps(man, indent=2))
    log(f"index: {man['n_trials']} trials, {man['n_sessions']} sessions")
    return df


def src_pickles():
    return sorted(p for p in (SRC / "sessions").glob("*.pkl") if not p.name.endswith(".tmp"))


def inspect_priority(path):
    return 0 if path.stem.startswith(INSPECT_PREFIXES) else 1


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--eid", default=None)
    parser.add_argument("--max-sessions", type=int, default=None)
    parser.add_argument("--max-trials", type=int, default=None)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--index-only", action="store_true")
    parser.add_argument("--inspect-only", action="store_true")
    args = parser.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    SESS_OUT.mkdir(parents=True, exist_ok=True)
    if args.index_only:
        rebuild_index()
        return
    one = connect_one()
    inspect_blob = {"traces": [], "psd": [], "line_map": LINE_HZ, "bands": BANDS}
    paths = src_pickles()
    if args.eid:
        paths = [p for p in paths if p.stem.startswith(args.eid)]
    if args.inspect_only:
        paths = [p for p in paths if p.stem.startswith(INSPECT_PREFIXES)]
    paths = sorted(paths, key=inspect_priority)
    if args.max_sessions:
        paths = paths[: args.max_sessions]
    log(f"sessions to process: {len(paths)}")
    for i, path in enumerate(paths, start=1):
        log(f"[{i}/{len(paths)}] {path.name}")
        try:
            process_session(
                one, path, inspect_blob, force=args.force, max_trials=args.max_trials
            )
        except Exception as exc:
            log(f"SESSION FAIL {path.stem}: {exc}")
            traceback.print_exc()
        (OUT / "inspect_partial.json").write_text(json.dumps(inspect_blob, default=lambda o: None))
    rebuild_index()
    (OUT / "inspect_partial.json").write_text(json.dumps(inspect_blob, default=str))
    log("done")


if __name__ == "__main__":
    main()
