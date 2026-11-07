#!/usr/bin/env python3
"""Embed inspect.json into the LFP QC canvas."""
from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent
SRC = Path(
    "/Users/Stefan/.cursor/projects/Users-Stefan-Desktop/canvases/lfp-rightcontra-qc.canvas.tsx"
)
TEMPLATE = ROOT / "lfp_qc_canvas_src.tsx"
INSPECT = ROOT / "ModelDataRightContraLFP" / "inspect.json"
OK_DIR = ROOT / "ModelDataRightContraLFP" / "sessions"
MANIFEST = ROOT / "ModelDataRightContra" / "manifest.json"


def compact(d):
    traces = []
    seen = set()
    for t in d.get("traces", []):
        key = (t["mouse"], t["region"])
        if key in seen:
            continue
        seen.add(key)
        traces.append(t)
        if len(traces) >= 9:
            break
    psd = []
    seen = set()
    for t in d.get("psd", []):
        key = (t["mouse"], t["region"])
        if key in seen:
            continue
        seen.add(key)
        psd.append(t)
        if len(psd) >= 9:
            break
    specs = []
    seen = set()
    for s in d.get("spectrograms", []):
        key = (s["mouse"], s["event"])
        if key in seen:
            continue
        seen.add(key)
        specs.append(s)
    spatial = list(d.get("spatial", []))
    n_ok = 0
    n_tr = 0
    for p in OK_DIR.glob("*.ok.json"):
        row = json.loads(p.read_text())
        n = int(row.get("n_trials") or 0)
        if n >= 8:
            n_ok += 1
            n_tr += n
    total_sessions = 52
    total_trials = 7191
    if MANIFEST.exists():
        man = json.loads(MANIFEST.read_text())
        total_sessions = int(man.get("n_sessions") or 52)
        total_trials = int(man.get("n_trials") or 7191)
    out = {
        "bands": d.get("bands", []),
        "sessions": d.get("sessions", []),
        "traces": traces,
        "psd": psd,
        "spectrograms": specs,
        "epoch_power": d.get("epoch_power", []),
        "spatial": spatial,
        "assoc": d.get("assoc", []),
        "ref_smooth": d.get("ref_smooth", []),
        "example_trials": d.get("example_trials", []),
        "filters": d.get("filters"),
        "corpus": {
            "n_sessions_done": n_ok,
            "n_sessions_total": total_sessions,
            "n_trials_done": n_tr,
            "n_trials_total": total_trials,
        },
    }
    return out


def main():
    raw = json.loads(INSPECT.read_text())
    data = compact(raw)
    src = TEMPLATE.read_text()
    payload = json.dumps(data, separators=(",", ":"))
    if "__DATA__" not in src:
        raise SystemExit("template missing __DATA__")
    out = src.replace("__DATA__", payload)
    SRC.write_text(out)
    print("wrote", SRC, "bytes", SRC.stat().st_size, "sessions", len(data["sessions"]))


if __name__ == "__main__":
    main()
