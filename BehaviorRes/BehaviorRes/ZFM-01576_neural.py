"""Download, cache, and render BWM spike-sorted units for ZFM-01576 2020-12-03.

Well-isolated neurons are those in the frozen BWM clusters table with IBL
QC label >= 1 (the same criterion used by brainwidemap.load_good_units).
Spike trains come from SpikeSortingLoader revision 2024-05-06. Probe
geometry is the resolved histology channel locations in Allen CCF / IBL xyz.
"""

from __future__ import annotations

import pickle
from pathlib import Path

import cv2
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from brainbox.io.one import SpikeSortingLoader
from iblatlas.atlas import AllenAtlas
from iblutil.numerical import ismember
from one.api import ONE

ROOT = Path(__file__).resolve().parent
DESKTOP = ROOT.parent
OUT_DIR = ROOT
NEURAL_CACHE_PATH = OUT_DIR / "ZFM-01576_neural_cache.pkl"
ATLAS_ASSET_PATH = OUT_DIR / "ZFM-01576_atlas_assets.pkl"
CLUSTERS_PQT = ROOT / "clusters.pqt"
BWM_RELEASE_CSV = ROOT / "2023_12_bwm_release.csv"
EID = "9e9c6fc0-4769-4d83-9ea4-b59a1230510e"
CACHE_VERSION = 1
ASSETS_VERSION = 2
SPIKE_REVISION = "2024-05-06"
MIN_QC = 1.0
SPIKE_TAU_S = 0.050
SPIKE_WINDOW_S = 0.180
NEURAL_WIDTH = 900
PROBE_COLORS_BGR = [
    (255, 200, 40),   # cyan-gold for probe 0
    (40, 140, 255),   # orange for probe 1
    (180, 80, 255),
    (80, 220, 120),
]


def connect_one():
    return ONE(
        base_url="https://openalyx.internationalbrainlab.org",
        password="international",
    )


def _as_str_array(values):
    return np.asarray([str(v) for v in values], dtype=object)


def discover_probes(one, eid=EID):
    """All insertions on the session, tagged with whether they are in the BWM freeze."""
    bwm = pd.read_csv(BWM_RELEASE_CSV)
    bwm_pids = set(bwm.loc[bwm["eid"] == eid, "pid"].astype(str))
    insertions = one.alyx.rest("insertions", "list", session=eid)
    probes = []
    seen = set()
    for ins in insertions:
        pid = str(ins["id"])
        seen.add(pid)
        probes.append(
            {
                "pid": pid,
                "pname": ins.get("name") or "probe",
                "model": ins.get("model", "3B2"),
                "in_bwm": pid in bwm_pids,
            }
        )
    for _, row in bwm.loc[bwm["eid"] == eid].iterrows():
        pid = str(row["pid"])
        if pid not in seen:
            probes.append(
                {
                    "pid": pid,
                    "pname": str(row["probe_name"]),
                    "model": "3B2",
                    "in_bwm": True,
                }
            )
    probes.sort(key=lambda p: p["pname"])
    return probes


def bwm_unit_uuids(eid, pid):
    df = pd.read_parquet(CLUSTERS_PQT)
    hit = df[(df["eid"] == eid) & (df["pid"] == pid) & (df["label"] >= MIN_QC)]
    return set(_as_str_array(hit["uuids"]))


def _channel_table(channels):
    n = len(channels["x"])
    atlas_id = np.asarray(channels.get("atlas_id", np.zeros(n)), dtype=int)
    acronym = channels.get("acronym")
    if acronym is None:
        acronym = np.array(["void"] * n, dtype=object)
    else:
        acronym = _as_str_array(acronym)
    return {
        "x": np.asarray(channels["x"], dtype=float),
        "y": np.asarray(channels["y"], dtype=float),
        "z": np.asarray(channels["z"], dtype=float),
        "axial_um": np.asarray(channels["axial_um"], dtype=float),
        "lateral_um": np.asarray(channels["lateral_um"], dtype=float),
        "atlas_id": atlas_id,
        "acronym": acronym,
    }


def _filter_to_bwm_units(spikes, clusters, keep_uuids):
    uuids = _as_str_array(clusters["uuids"])
    keep = np.array([u in keep_uuids for u in uuids], dtype=bool)
    if not np.any(keep):
        empty = {
            "times": np.array([], dtype=float),
            "clusters": np.array([], dtype=np.int32),
        }
        return empty, clusters.iloc[0:0].reset_index(drop=True)
    keep_idx = np.flatnonzero(keep)
    remap = np.full(len(clusters), -1, dtype=int)
    remap[keep_idx] = np.arange(keep_idx.size, dtype=int)
    spike_new = remap[np.asarray(spikes["clusters"], dtype=int)]
    spike_ok = spike_new >= 0
    out_spikes = {
        "times": np.asarray(spikes["times"], dtype=float)[spike_ok],
        "clusters": spike_new[spike_ok].astype(np.int32),
    }
    if "amps" in spikes:
        out_spikes["amps"] = np.asarray(spikes["amps"], dtype=float)[spike_ok]
    out_clusters = clusters.iloc[keep_idx].reset_index(drop=True)
    return out_spikes, out_clusters


def _unit_records(clusters, channels):
    ch_idx = np.asarray(clusters["channels"], dtype=int)
    n_ch = len(channels["x"])
    ch_idx = np.clip(ch_idx, 0, max(n_ch - 1, 0))
    x = np.asarray(clusters["x"], dtype=float) if "x" in clusters else channels["x"][ch_idx]
    y = np.asarray(clusters["y"], dtype=float) if "y" in clusters else channels["y"][ch_idx]
    z = np.asarray(clusters["z"], dtype=float) if "z" in clusters else channels["z"][ch_idx]
    axial = (
        np.asarray(clusters["axial_um"], dtype=float)
        if "axial_um" in clusters
        else channels["axial_um"][ch_idx]
    )
    lateral = (
        np.asarray(clusters["lateral_um"], dtype=float)
        if "lateral_um" in clusters
        else channels["lateral_um"][ch_idx]
    )
    acronym = (
        _as_str_array(clusters["acronym"])
        if "acronym" in clusters
        else channels["acronym"][ch_idx]
    )
    atlas_id = (
        np.asarray(clusters["atlas_id"], dtype=int)
        if "atlas_id" in clusters
        else channels["atlas_id"][ch_idx]
    )
    return {
        "uuids": _as_str_array(clusters["uuids"]),
        "cluster_id": np.asarray(clusters.get("cluster_id", np.arange(len(clusters))), dtype=int),
        "channel": ch_idx.astype(np.int32),
        "x": x,
        "y": y,
        "z": z,
        "axial_um": axial,
        "lateral_um": lateral,
        "acronym": acronym,
        "atlas_id": atlas_id,
    }


def _empty_units():
    return {
        "uuids": np.array([], dtype=object),
        "cluster_id": np.array([], dtype=int),
        "channel": np.array([], dtype=np.int32),
        "x": np.array([], dtype=float),
        "y": np.array([], dtype=float),
        "z": np.array([], dtype=float),
        "axial_um": np.array([], dtype=float),
        "lateral_um": np.array([], dtype=float),
        "acronym": np.array([], dtype=object),
        "atlas_id": np.array([], dtype=int),
    }


def load_probe_units(one, pid, pname, eid, in_bwm):
    print(f"Loading spike sorting {pname} pid={pid} in_bwm={in_bwm}")
    loader = SpikeSortingLoader(pid=pid, one=one, pname=pname, eid=eid)
    spikes, clusters, channels = loader.load_spike_sorting(
        revision=SPIKE_REVISION, good_units=bool(in_bwm)
    )
    if not channels:
        spikes, clusters, channels = loader.load_spike_sorting(revision=SPIKE_REVISION)
    chan_tbl = _channel_table(channels)
    clusters_df = None
    if in_bwm and spikes:
        labeled = SpikeSortingLoader.merge_clusters(spikes, clusters, channels).to_df()
        iok = (
            labeled["label"] >= MIN_QC
            if "label" in labeled.columns
            else np.ones(len(labeled), dtype=bool)
        )
        good = labeled[iok]
        spike_idx, ib = ismember(np.asarray(spikes["clusters"]), good.index.to_numpy())
        good = good.reset_index(drop=True)
        qc_spikes = {
            "times": np.asarray(spikes["times"], dtype=float)[spike_idx],
            "clusters": np.arange(len(good), dtype=np.int32)[ib],
        }
        keep = bwm_unit_uuids(eid, pid)
        print(f"  QC>={MIN_QC} after merge: {len(good)}; BWM aggregate uuids: {len(keep)}")
        spikes, clusters_df = _filter_to_bwm_units(qc_spikes, good, keep)
    else:
        spikes = {"times": np.array([], dtype=float), "clusters": np.array([], dtype=np.int32)}
        if not in_bwm:
            print("  Not a BWM insertion; keeping probe geometry only.")
    units = (
        _unit_records(clusters_df, chan_tbl)
        if clusters_df is not None and len(clusters_df)
        else _empty_units()
    )
    print(
        f"  channels={len(chan_tbl['x'])}  BWM units={len(units['uuids'])}  "
        f"spikes={len(spikes['times'])}"
    )
    order = np.argsort(np.asarray(spikes["times"], dtype=float), kind="stable")
    return {
        "pid": pid,
        "pname": pname,
        "in_bwm": in_bwm,
        "channels": chan_tbl,
        "units": units,
        "spikes": {
            "times": np.asarray(spikes["times"], dtype=float)[order],
            "clusters": np.asarray(spikes["clusters"], dtype=np.int32)[order],
        },
    }


def window_probe_spikes(probe, t0, t1, pad=SPIKE_WINDOW_S):
    times = probe["spikes"]["times"]
    if times.size == 0:
        return {
            "times": np.array([], dtype=float),
            "clusters": np.array([], dtype=np.int32),
        }
    lo, hi = np.searchsorted(times, [t0 - pad, t1 + pad])
    return {
        "times": times[lo:hi].copy(),
        "clusters": probe["spikes"]["clusters"][lo:hi].copy(),
    }


def cache_paths(eid, cache_path=None, atlas_path=None):
    eid = str(eid)
    if cache_path is None:
        cache_path = NEURAL_CACHE_PATH if eid == EID else OUT_DIR / f"neural_cache_{eid}.pkl"
    if atlas_path is None:
        atlas_path = ATLAS_ASSET_PATH if eid == EID else OUT_DIR / f"atlas_assets_{eid}.pkl"
    return Path(cache_path), Path(atlas_path)


def build_neural_cache(one, trial_windows, eid=EID, cache_path=None):
    """trial_windows: list of (label, t0, t1)."""
    eid = str(eid)
    cache_path, _atlas = cache_paths(eid, cache_path=cache_path)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    probes_meta = discover_probes(one, eid)
    print(f"Session {eid}: {len(probes_meta)} probe(s)")
    probes = []
    for meta in probes_meta:
        probes.append(
            load_probe_units(one, meta["pid"], meta["pname"], eid, meta["in_bwm"])
        )
    trials = {}
    for label, t0, t1 in trial_windows:
        trials[label] = {
            "t0": float(t0),
            "t1": float(t1),
            "probes": [window_probe_spikes(p, t0, t1) for p in probes],
        }
    cache = {
        "version": CACHE_VERSION,
        "eid": eid,
        "probes": [
            {k: v for k, v in p.items() if k != "spikes"}
            for p in probes
        ],
        "trials": trials,
        "n_units_total": int(sum(len(p["units"]["uuids"]) for p in probes)),
        "n_channels_total": int(sum(len(p["channels"]["x"]) for p in probes)),
    }
    with cache_path.open("wb") as f:
        pickle.dump(cache, f, protocol=pickle.HIGHEST_PROTOCOL)
    print(f"Wrote {cache_path} ({cache['n_units_total']} BWM units)")
    return cache


def load_neural_cache(one=None, trial_windows=None, rebuild=False, eid=EID, cache_path=None):
    eid = str(eid)
    cache_path, _atlas = cache_paths(eid, cache_path=cache_path)
    if not rebuild and cache_path.exists():
        with cache_path.open("rb") as f:
            cache = pickle.load(f)
        if cache.get("version") == CACHE_VERSION and cache.get("eid") == eid:
            needed = {label for label, _t0, _t1 in (trial_windows or [])}
            have = set(cache.get("trials", {}))
            if not needed or needed.issubset(have):
                print(
                    f"Loaded neural cache: {cache['n_units_total']} units, "
                    f"{len(cache['probes'])} probe(s)  eid={eid[:8]}"
                )
                return cache
            if one is None:
                one = connect_one()
            print(f"Neural cache missing trials {needed - have}; windowing from probes...")
            return _extend_neural_cache(one, cache, trial_windows, cache_path)
    if one is None:
        one = connect_one()
    if trial_windows is None:
        raise ValueError("trial_windows required to build the neural cache")
    return build_neural_cache(one, trial_windows, eid=eid, cache_path=cache_path)


def _extend_neural_cache(one, cache, trial_windows, cache_path):
    """Add new trial spike windows using already-loaded probe geometry + spikes."""
    eid = cache["eid"]
    probes_meta = discover_probes(one, eid)
    probes = []
    for meta, stored in zip(probes_meta, cache["probes"]):
        full = load_probe_units(one, meta["pid"], meta["pname"], eid, meta["in_bwm"])
        probes.append(full)
    trials = dict(cache.get("trials", {}))
    for label, t0, t1 in trial_windows:
        if label in trials:
            continue
        trials[label] = {
            "t0": float(t0),
            "t1": float(t1),
            "probes": [window_probe_spikes(p, t0, t1) for p in probes],
        }
    cache = dict(cache)
    cache["trials"] = trials
    with Path(cache_path).open("wb") as f:
        pickle.dump(cache, f, protocol=pickle.HIGHEST_PROTOCOL)
    return cache


def _region_rgb(ba, atlas_ids):
    atlas_ids = np.asarray(atlas_ids, dtype=int)
    rgb = np.full((atlas_ids.size, 3), 180, dtype=np.uint8)
    if atlas_ids.size == 0:
        return rgb
    ok, idx = ismember(atlas_ids, ba.regions.id)
    if np.any(ok):
        rgb[ok] = ba.regions.rgb[idx[ok]]
    return rgb


def _fig_to_bgr(fig):
    fig.canvas.draw()
    buf = np.asarray(fig.canvas.buffer_rgba())
    return cv2.cvtColor(buf, cv2.COLOR_RGBA2BGR)


def _ax_to_px(ax, fig, img_shape, xs, ys):
    xs = np.asarray(xs, dtype=float)
    ys = np.asarray(ys, dtype=float)
    if xs.size == 0:
        return np.zeros((0, 2), dtype=float)
    disp = ax.transData.transform(np.column_stack([xs, ys]))
    fw, fh = fig.canvas.get_width_height()
    img_h, img_w = img_shape[:2]
    px = np.empty_like(disp)
    px[:, 0] = disp[:, 0] * (img_w / max(fw, 1))
    px[:, 1] = img_h - disp[:, 1] * (img_h / max(fh, 1))
    return px


def _ax_bbox_px(ax, fig, img_shape):
    bbox = ax.get_position()
    img_h, img_w = img_shape[:2]
    x0 = int(round(bbox.x0 * img_w))
    x1 = int(round(bbox.x1 * img_w))
    y0 = int(round((1.0 - bbox.y1) * img_h))
    y1 = int(round((1.0 - bbox.y0) * img_h))
    return [x0, y0, x1, y1]


def _style_atlas_ax(ax, title):
    ax.set_title(title, color="white", fontsize=8, pad=3)
    ax.tick_params(colors="white", labelsize=6)
    ax.xaxis.label.set_color("white")
    ax.yaxis.label.set_color("white")
    for spine in ax.spines.values():
        spine.set_color("white")


def build_atlas_assets(cache, width, brain_height, ba=None, atlas_path=None):
    if ba is None:
        print("Loading Allen CCF atlas (cached after the first call)...")
        ba = AllenAtlas()
    probes = cache["probes"]
    all_xyz = []
    for probe in probes:
        ch = probe["channels"]
        all_xyz.append(np.column_stack([ch["x"], ch["y"], ch["z"]]))
        if len(probe["units"]["x"]):
            all_xyz.append(
                np.column_stack(
                    [probe["units"]["x"], probe["units"]["y"], probe["units"]["z"]]
                )
            )
    if all_xyz:
        xyz = np.concatenate(all_xyz, axis=0)
        xyz = xyz[np.isfinite(xyz).all(axis=1)]
    else:
        xyz = np.zeros((0, 3), dtype=float)
    if xyz.size == 0:
        xyz = np.zeros((1, 3), dtype=float)
    ml = float(np.nanmedian(xyz[:, 0]))
    ap = float(np.nanmedian(xyz[:, 1]))
    dv = float(np.nanmedian(xyz[:, 2]))

    dpi = 120
    fig_w = width / dpi
    fig_h = brain_height / dpi
    fig = plt.figure(figsize=(fig_w, fig_h), dpi=dpi, facecolor="black")
    gs = fig.add_gridspec(2, 2, wspace=0.18, hspace=0.28)
    ax_cor = fig.add_subplot(gs[0, 0], facecolor="black")
    ax_sag = fig.add_subplot(gs[0, 1], facecolor="black")
    ax_hor = fig.add_subplot(gs[1, 0], facecolor="black")
    ax_probe = fig.add_subplot(gs[1, 1], facecolor="black")
    ax_probe.set_xticks([])
    ax_probe.set_yticks([])
    ba.plot_cslice(ap, volume="annotation", mapping="Beryl", ax=ax_cor)
    ba.plot_sslice(ml, volume="annotation", mapping="Beryl", ax=ax_sag)
    ba.plot_hslice(dv, volume="annotation", mapping="Beryl", ax=ax_hor)
    _style_atlas_ax(ax_cor, "Coronal")
    _style_atlas_ax(ax_sag, "Sagittal")
    _style_atlas_ax(ax_hor, "Axial / horizontal")
    _style_atlas_ax(ax_probe, "Neuropixels probe")

    probe_px = []
    unit_cor, unit_sag, unit_hor = [], [], []
    for i, probe in enumerate(probes):
        color = np.array(PROBE_COLORS_BGR[i % len(PROBE_COLORS_BGR)][::-1]) / 255.0
        ch = probe["channels"]
        ax_cor.plot(ch["x"] * 1e6, ch["z"] * 1e6, color=color, lw=1.6, zorder=4)
        ax_sag.plot(ch["y"] * 1e6, ch["z"] * 1e6, color=color, lw=1.6, zorder=4)
        ax_hor.plot(ch["x"] * 1e6, ch["y"] * 1e6, color=color, lw=1.6, zorder=4)

    fig.subplots_adjust(left=0.10, right=0.98, top=0.93, bottom=0.08)
    bg = _fig_to_bgr(fig)
    probe_slot = _ax_bbox_px(ax_probe, fig, bg.shape)
    for i, probe in enumerate(probes):
        ch = probe["channels"]
        probe_px.append(
            {
                "coronal": _ax_to_px(ax_cor, fig, bg.shape, ch["x"] * 1e6, ch["z"] * 1e6),
                "sagittal": _ax_to_px(ax_sag, fig, bg.shape, ch["y"] * 1e6, ch["z"] * 1e6),
                "horizontal": _ax_to_px(ax_hor, fig, bg.shape, ch["x"] * 1e6, ch["y"] * 1e6),
            }
        )
        units = probe["units"]
        unit_cor.append(_ax_to_px(ax_cor, fig, bg.shape, units["x"] * 1e6, units["z"] * 1e6))
        unit_sag.append(_ax_to_px(ax_sag, fig, bg.shape, units["y"] * 1e6, units["z"] * 1e6))
        unit_hor.append(_ax_to_px(ax_hor, fig, bg.shape, units["x"] * 1e6, units["y"] * 1e6))
    plt.close(fig)

    unit_rgb = [_region_rgb(ba, p["units"]["atlas_id"]) for p in probes]
    chan_rgb = [_region_rgb(ba, p["channels"]["atlas_id"]) for p in probes]
    assets = {
        "version": ASSETS_VERSION,
        "eid": cache.get("eid"),
        "brain_bg": bg,
        "ml": ml,
        "ap": ap,
        "dv": dv,
        "probe_px": probe_px,
        "probe_slot": probe_slot,
        "unit_cor": unit_cor,
        "unit_sag": unit_sag,
        "unit_hor": unit_hor,
        "unit_rgb": unit_rgb,
        "chan_rgb": chan_rgb,
        "width": width,
        "brain_height": brain_height,
    }
    _, default_atlas = cache_paths(cache.get("eid", EID), atlas_path=atlas_path)
    default_atlas.parent.mkdir(parents=True, exist_ok=True)
    with default_atlas.open("wb") as f:
        pickle.dump(assets, f, protocol=pickle.HIGHEST_PROTOCOL)
    print(f"Wrote atlas assets {bg.shape[1]}x{bg.shape[0]} -> {default_atlas}")
    return assets, ba


def _scale_xy_list(items, sx, sy):
    return [np.asarray(pts, dtype=float) * np.array([sx, sy]) for pts in items]


def scale_atlas_assets(assets, width, brain_height):
    old_h, old_w = assets["brain_bg"].shape[:2]
    if old_w == width and old_h == brain_height:
        return assets
    sx = width / max(old_w, 1)
    sy = brain_height / max(old_h, 1)
    out = dict(assets)
    out["brain_bg"] = cv2.resize(assets["brain_bg"], (width, brain_height), interpolation=cv2.INTER_AREA)
    out["unit_cor"] = _scale_xy_list(assets["unit_cor"], sx, sy)
    out["unit_sag"] = _scale_xy_list(assets["unit_sag"], sx, sy)
    out["unit_hor"] = _scale_xy_list(assets.get("unit_hor", assets["unit_cor"]), sx, sy)
    out["probe_px"] = [
        {k: np.asarray(v, dtype=float) * np.array([sx, sy]) for k, v in d.items()}
        for d in assets["probe_px"]
    ]
    slot = np.asarray(assets.get("probe_slot", [0, 0, width, brain_height]), dtype=float)
    slot[0::2] *= sx
    slot[1::2] *= sy
    out["probe_slot"] = [int(round(v)) for v in slot]
    out["width"] = width
    out["brain_height"] = brain_height
    return out


def load_atlas_assets(cache, width, brain_height, ba=None, atlas_path=None):
    _cache_p, atlas_path = cache_paths(cache.get("eid", EID), atlas_path=atlas_path)
    if atlas_path.exists():
        with atlas_path.open("rb") as f:
            assets = pickle.load(f)
        if (
            assets.get("version") == ASSETS_VERSION
            and assets.get("unit_hor") is not None
            and assets.get("eid") in (None, cache.get("eid"))
        ):
            return scale_atlas_assets(assets, width, brain_height), ba
    assets, ba = build_atlas_assets(
        cache, width, brain_height, ba=ba, atlas_path=atlas_path
    )
    return scale_atlas_assets(assets, width, brain_height), ba


def unit_intensities(spike_times, spike_clusters, n_units, t_now, tau=SPIKE_TAU_S):
    inten = np.zeros(n_units, dtype=float)
    if spike_times.size == 0 or n_units == 0:
        return inten
    lo, hi = np.searchsorted(spike_times, [t_now - SPIKE_WINDOW_S, t_now + 1e-6])
    if hi <= lo:
        return inten
    dt = t_now - spike_times[lo:hi]
    val = np.exp(-np.maximum(dt, 0.0) / tau)
    cl = spike_clusters[lo:hi]
    good = (cl >= 0) & (cl < n_units)
    for c, v in zip(cl[good], val[good]):
        if v > inten[c]:
            inten[c] = v
    return inten


def _blend_circle(img, xy, radius, color, alpha):
    if alpha <= 0.02 or not np.isfinite(xy).all():
        return
    overlay = img.copy()
    pt = (int(round(xy[0])), int(round(xy[1])))
    cv2.circle(overlay, pt, int(radius), color, -1, cv2.LINE_AA)
    cv2.addWeighted(overlay, alpha, img, 1.0 - alpha, 0, dst=img)


def render_brain_overlay(assets, cache, trial_label, t_abs):
    img = assets["brain_bg"].copy()
    trial = cache["trials"][trial_label]
    n_spikes = 0
    for i, probe in enumerate(cache["probes"]):
        n_u = len(probe["units"]["uuids"])
        inten = unit_intensities(
            trial["probes"][i]["times"],
            trial["probes"][i]["clusters"],
            n_u,
            t_abs,
        )
        n_spikes += int(np.sum(inten > 0.15))
        rgb = assets["unit_rgb"][i]
        cor = assets["unit_cor"][i]
        sag = assets["unit_sag"][i]
        hor = assets.get("unit_hor", [cor] * len(cache["probes"]))[i]
        for u in range(n_u):
            rest = tuple(int(c) for c in rgb[u][::-1])  # RGB -> BGR
            a = 0.35 + 0.65 * inten[u]
            rad = 3 + 7 * inten[u]
            glow = (255, 255, 255) if inten[u] > 0.2 else rest
            for xy in (cor[u], sag[u], hor[u]):
                _blend_circle(img, xy, rad + 2, glow, min(1.0, a))
                _blend_circle(img, xy, max(2, rad - 1), rest, min(1.0, 0.55 + 0.45 * inten[u]))
    cv2.putText(
        img,
        f"BWM good units lighting up  |  active={n_spikes}",
        (14, 22),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.48,
        (255, 255, 255),
        1,
        cv2.LINE_AA,
    )
    return img


def _channel_xy(probe, width, height, pad=18):
    lat = probe["channels"]["lateral_um"]
    ax = probe["channels"]["axial_um"]
    n = lat.size
    if n == 0:
        return np.zeros((0, 2)), (0.0, 1.0, 0.0, 1.0)
    lat_min, lat_max = float(np.nanmin(lat)), float(np.nanmax(lat))
    ax_min, ax_max = float(np.nanmin(ax)), float(np.nanmax(ax))
    if lat_max <= lat_min:
        lat_max = lat_min + 48.0
    if ax_max <= ax_min:
        ax_max = ax_min + 1.0
    # Leave a legend strip on the right.
    plot_w = width * 0.62
    xs = pad + (lat - lat_min) / (lat_max - lat_min) * (plot_w - 2 * pad)
    # Tip (small axial_um) at the bottom.
    ys = height - pad - (ax - ax_min) / (ax_max - ax_min) * (height - 2 * pad)
    return np.column_stack([xs, ys]), (lat_min, lat_max, ax_min, ax_max)


def render_probe_view(cache, assets, trial_label, t_abs, width, height):
    img = np.zeros((height, width, 3), dtype=np.uint8)
    img[:] = (18, 16, 16)
    n_probes = len(cache["probes"])
    col_w = width // max(n_probes, 1)
    trial = cache["trials"][trial_label]
    for i, probe in enumerate(cache["probes"]):
        x0 = i * col_w
        sub = img[:, x0 : x0 + col_w]
        xy, _lims = _channel_xy(probe, col_w, height)
        rgb = assets["chan_rgb"][i]
        n_ch = xy.shape[0]
        n_u = len(probe["units"]["uuids"])
        inten_u = unit_intensities(
            trial["probes"][i]["times"],
            trial["probes"][i]["clusters"],
            n_u,
            t_abs,
        )
        ch_inten = np.zeros(n_ch, dtype=float)
        if n_u and n_ch:
            for u, ch in enumerate(probe["units"]["channel"]):
                if 0 <= ch < n_ch:
                    ch_inten[ch] = max(ch_inten[ch], inten_u[u])
        # Shank outline
        if n_ch:
            hull_x = [int(np.nanmin(xy[:, 0]) - 10), int(np.nanmax(xy[:, 0]) + 10)]
            cv2.rectangle(
                sub,
                (hull_x[0], 12),
                (hull_x[1], height - 12),
                PROBE_COLORS_BGR[i % len(PROBE_COLORS_BGR)],
                1,
                cv2.LINE_AA,
            )
        for c in range(n_ch):
            pt = (int(round(xy[c, 0])), int(round(xy[c, 1])))
            color = tuple(int(v) for v in rgb[c][::-1])
            a = 0.35 + 0.65 * ch_inten[c]
            rad = 3 + int(5 * ch_inten[c])
            if ch_inten[c] > 0.15:
                cv2.circle(sub, pt, rad + 3, (255, 255, 255), -1, cv2.LINE_AA)
            cv2.circle(sub, pt, rad, color, -1, cv2.LINE_AA)
            if a < 0.99 and ch_inten[c] <= 0.15:
                cv2.circle(sub, pt, 3, color, -1, cv2.LINE_AA)
        pname = probe["pname"]
        n_units = len(probe["units"]["uuids"])
        cv2.putText(
            sub,
            f"{pname}  {n_ch} ch   {n_units} BWM units",
            (8, 22),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.50,
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )
        cv2.putText(
            sub,
            "tip",
            (8, height - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.40,
            (180, 180, 180),
            1,
            cv2.LINE_AA,
        )
        acronyms = probe["channels"]["acronym"]
        uniq = []
        for ac in acronyms:
            if ac not in uniq and ac not in ("void", "root", "void0"):
                uniq.append(ac)
        legend_x = int(col_w * 0.68)
        y = 48
        for ac in uniq[:16]:
            mask = acronyms == ac
            col = tuple(int(v) for v in np.median(rgb[mask], axis=0)[::-1])
            cv2.circle(sub, (legend_x, y), 5, col, -1, cv2.LINE_AA)
            cv2.putText(
                sub,
                str(ac),
                (legend_x + 12, y + 4),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.38,
                (230, 230, 230),
                1,
                cv2.LINE_AA,
            )
            y += 16
    return img


def raster_sort_units(probe):
    """Group units by region, order those blocks by probe depth, sort within by axial_um.

    High axial_um (away from the tip / more dorsal) is drawn at the top of the raster.
    """
    acr = _as_str_array(probe["units"]["acronym"])
    axial = np.asarray(probe["units"]["axial_um"], dtype=float)
    n = acr.size
    if n == 0:
        return np.array([], dtype=int), []
    cleaned = np.array(
        ["void" if a in ("", "root", "void", "void0", "None", "nan") else a for a in acr],
        dtype=object,
    )
    regions = []
    for ac in cleaned:
        if ac not in regions:
            regions.append(ac)
    regions.sort(key=lambda ac: -float(np.nanmedian(axial[cleaned == ac])))
    order = []
    blocks = []
    for ac in regions:
        idx = np.flatnonzero(cleaned == ac)
        idx = idx[np.argsort(-axial[idx], kind="stable")]
        start = len(order)
        order.extend(idx.tolist())
        blocks.append({"acronym": str(ac), "start": start, "end": len(order)})
    return np.asarray(order, dtype=int), blocks


def prepare_trial_raster(cache, assets, trial_label):
    trial = cache["trials"][trial_label]
    t0 = float(trial["t0"])
    t1 = float(trial["t1"])
    orders, blocks_all, rgb_rows = [], [], []
    row_offset = 0
    spike_t = []
    spike_row = []
    for i, probe in enumerate(cache["probes"]):
        order, blocks = raster_sort_units(probe)
        n_u = len(probe["units"]["uuids"])
        if n_u == 0:
            continue
        inv = np.full(n_u, -1, dtype=int)
        inv[order] = np.arange(order.size, dtype=int) + row_offset
        rgb = assets["unit_rgb"][i][order]
        rgb_rows.append(rgb)
        for blk in blocks:
            blocks_all.append(
                {
                    "acronym": blk["acronym"],
                    "start": blk["start"] + row_offset,
                    "end": blk["end"] + row_offset,
                    "rgb": tuple(int(v) for v in np.median(rgb[blk["start"] : blk["end"]], axis=0)),
                }
            )
        times = np.asarray(trial["probes"][i]["times"], dtype=float)
        cl = np.asarray(trial["probes"][i]["clusters"], dtype=int)
        good = (cl >= 0) & (cl < n_u)
        spike_t.append(times[good] - t0)
        spike_row.append(inv[cl[good]])
        orders.append(order)
        row_offset += order.size
    if spike_t:
        spike_t = np.concatenate(spike_t)
        spike_row = np.concatenate(spike_row)
        keep = spike_row >= 0
        spike_t, spike_row = spike_t[keep], spike_row[keep]
        order = np.argsort(spike_t, kind="stable")
        spike_t, spike_row = spike_t[order], spike_row[order]
        rgb_rows = np.vstack(rgb_rows)
    else:
        spike_t = np.array([], dtype=float)
        spike_row = np.array([], dtype=int)
        rgb_rows = np.zeros((0, 3), dtype=np.uint8)
    return {
        "t0": t0,
        "t1": t1,
        "duration": max(t1 - t0, 1e-6),
        "n_rows": int(row_offset),
        "blocks": blocks_all,
        "rgb_rows": rgb_rows,
        "spike_t": spike_t,
        "spike_row": spike_row,
    }


def render_raster(spec, t_now_rel, width, height):
    img = np.full((height, width, 3), 255, dtype=np.uint8)
    ml, mr, mt, mb = 72, 12, 28, 36
    plot_w = max(width - ml - mr, 8)
    plot_h = max(height - mt - mb, 8)
    x0, x1 = ml, ml + plot_w
    y0, y1 = mt, mt + plot_h
    cv2.rectangle(img, (x0, y0), (x1, y1), (0, 0, 0), 1)
    cv2.putText(
        img,
        "Spike raster  (region / probe depth)",
        (ml, 18),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.45,
        (0, 0, 0),
        1,
        cv2.LINE_AA,
    )
    duration = spec["duration"]
    n_rows = max(spec["n_rows"], 1)
    row_h = plot_h / n_rows

    for blk in spec["blocks"]:
        ya = y0 + int(round(blk["start"] * row_h))
        yb = y0 + int(round(blk["end"] * row_h))
        color = blk["rgb"][::-1]  # RGB -> BGR
        cv2.rectangle(img, (8, ya), (ml - 8, max(yb, ya + 1)), color, -1)
        if yb - ya >= 12:
            label = blk["acronym"]
            (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.32, 1)
            tx = 8 + max(0, (ml - 16 - tw) // 2)
            ty = (ya + yb) // 2 + th // 2
            lum = 0.299 * blk["rgb"][0] + 0.587 * blk["rgb"][1] + 0.114 * blk["rgb"][2]
            fg = (0, 0, 0) if lum > 150 else (255, 255, 255)
            cv2.putText(
                img, label, (tx, ty), cv2.FONT_HERSHEY_SIMPLEX, 0.32, fg, 1, cv2.LINE_AA
            )
        if blk["start"] > 0:
            cv2.line(img, (x0, ya), (x1, ya), (210, 210, 210), 1)

    t_now = float(np.clip(t_now_rel, 0.0, duration))
    times = spec["spike_t"]
    rows = spec["spike_row"]
    hi = int(np.searchsorted(times, t_now + 1e-9))
    if hi > 0:
        xs = x0 + (times[:hi] / duration) * (plot_w - 1)
        ys = y0 + (rows[:hi] + 0.5) * row_h
        xs = np.clip(np.round(xs), x0, x1 - 1).astype(int)
        ys = np.clip(np.round(ys), y0, y1 - 1).astype(int)
        recent = times[:hi] >= (t_now - SPIKE_TAU_S)
        img[ys, xs] = (0, 0, 0)
        img[np.clip(ys - 1, y0, y1 - 1), xs] = (0, 0, 0)
        img[ys, np.clip(xs + 1, x0, x1 - 1)] = (0, 0, 0)
        if np.any(recent):
            img[ys[recent], xs[recent]] = (0, 0, 220)
            img[np.clip(ys[recent] - 1, y0, y1 - 1), xs[recent]] = (0, 0, 220)
            img[ys[recent], np.clip(xs[recent] + 1, x0, x1 - 1)] = (0, 0, 220)

    x_now = int(round(x0 + (t_now / duration) * plot_w))
    cv2.line(img, (x_now, y0), (x_now, y1), (0, 0, 220), 1)
    for t_tick in np.linspace(0.0, duration, 5):
        x = int(round(x0 + (t_tick / duration) * plot_w))
        cv2.line(img, (x, y1), (x, y1 + 5), (0, 0, 0), 1)
        label = f"{t_tick:.2f}" if duration < 2 else f"{t_tick:.1f}"
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.38, 1)
        cv2.putText(
            img,
            label,
            (x - tw // 2, min(height - 8, y1 + 18)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.38,
            (0, 0, 0),
            1,
            cv2.LINE_AA,
        )
    cv2.putText(
        img,
        "Time from stim (s)",
        (width // 2 - 55, height - 6),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.38,
        (0, 0, 0),
        1,
        cv2.LINE_AA,
    )
    return img


class NeuralRenderer:
    def __init__(self, cache, width=NEURAL_WIDTH, height=1664, atlas_path=None, ba=None):
        self.cache = cache
        self.width = int(width)
        self.height = int(height)
        self.brain_h = int(round(height * 0.62))
        self.raster_h = self.height - self.brain_h
        self.assets, self.ba = load_atlas_assets(
            cache, self.width, self.brain_h, ba=ba, atlas_path=atlas_path
        )
        bg_h, bg_w = self.assets["brain_bg"].shape[:2]
        if bg_w != self.width or bg_h != self.brain_h:
            self.assets = scale_atlas_assets(self.assets, self.width, self.brain_h)
        self.rasters = {
            label: prepare_trial_raster(cache, self.assets, label)
            for label in cache["trials"]
        }

    def render(self, trial_label, t_abs, height=None):
        target_h = int(height) if height else self.height
        brain = render_brain_overlay(self.assets, self.cache, trial_label, t_abs)
        if brain.shape[0] != self.brain_h or brain.shape[1] != self.width:
            brain = cv2.resize(brain, (self.width, self.brain_h), interpolation=cv2.INTER_AREA)
        slot = self.assets.get("probe_slot")
        if slot is not None:
            x0, y0, x1, y1 = [int(v) for v in slot]
            x0 = max(0, min(x0, brain.shape[1] - 2))
            x1 = max(x0 + 2, min(x1, brain.shape[1]))
            y0 = max(0, min(y0, brain.shape[0] - 2))
            y1 = max(y0 + 2, min(y1, brain.shape[0]))
            probe = render_probe_view(
                self.cache, self.assets, trial_label, t_abs, x1 - x0, y1 - y0
            )
            brain[y0:y1, x0:x1] = probe
        spec = self.rasters[trial_label]
        raster = render_raster(spec, t_abs - spec["t0"], self.width, self.raster_h)
        gap = np.full((4, self.width, 3), 255, dtype=np.uint8)
        combined = np.vstack([brain, gap, raster])
        if combined.shape[0] != target_h or combined.shape[1] != self.width:
            combined = cv2.resize(
                combined, (self.width, target_h), interpolation=cv2.INTER_AREA
            )
        return combined


def trial_windows_from_behavior_cache(beh_cache, chosen):
    windows = []
    for name, idx, _dur in chosen:
        rec = beh_cache["records"][idx]
        windows.append((name, float(rec["stimOn_times"]), float(rec["feedback_times"])))
    return windows
