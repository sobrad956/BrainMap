"""Shared Slurm-array, device, and per-job status helpers.

Local sequential (default): run the script with no --job. All configs train
one after another on --device auto|cpu|cuda|mps.

Cluster parallel: sbatch the matching slurm/*.sbatch, which sets
SLURM_ARRAY_TASK_ID. Each array task trains one config into
<OUT>/jobs/<index>_<tag>/{status.json,epochs.log,scores.csv}.
"""

from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from pathlib import Path

_DEVICE = None
_STATUS_DIR = None
_META = {}


def resolve_device(name="auto"):
    import torch

    name = (name or "auto").lower()
    if name == "cpu":
        return torch.device("cpu")
    if name == "cuda":
        if not torch.cuda.is_available():
            raise SystemExit("CUDA requested but torch.cuda.is_available() is False")
        return torch.device("cuda")
    if name == "mps":
        if not getattr(torch.backends, "mps", None) or not torch.backends.mps.is_available():
            raise SystemExit("MPS requested but not available")
        return torch.device("mps")
    if name != "auto":
        raise SystemExit(f"unknown device {name!r}; use auto, cpu, cuda, or mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def set_device(name="auto"):
    global _DEVICE
    _DEVICE = resolve_device(name)
    return _DEVICE


def device_of():
    global _DEVICE
    if _DEVICE is None:
        _DEVICE = resolve_device("auto")
    return _DEVICE


def add_common_args(parser):
    parser.add_argument(
        "--device",
        default="auto",
        choices=("auto", "cpu", "cuda", "mps"),
        help="auto: CUDA, else MPS, else CPU. cpu forces local CPU.",
    )
    parser.add_argument(
        "--job",
        type=int,
        default=None,
        help="Run only this config index. Defaults to SLURM_ARRAY_TASK_ID if set.",
    )
    parser.add_argument(
        "--list-jobs",
        action="store_true",
        help="Print the config grid and exit (use this to set --array).",
    )
    parser.add_argument(
        "--aggregate",
        action="store_true",
        help="Merge jobs/*/scores.csv into the script's main scores file.",
    )
    return parser


def resolve_job_index(cli_job):
    if cli_job is not None:
        return int(cli_job)
    env = os.environ.get("SLURM_ARRAY_TASK_ID")
    if env not in (None, ""):
        return int(env)
    return None


def pick_config(configs, job):
    configs = list(configs)
    if job < 0 or job >= len(configs):
        raise SystemExit(f"job index {job} out of range 0..{len(configs) - 1}")
    return configs[job]


def print_jobs(configs):
    configs = list(configs)
    print(f"{len(configs)} jobs  (sbatch --array=0-{max(len(configs) - 1, 0)})")
    for i, cfg in enumerate(configs):
        parts = "  ".join(f"{k}={v}" for k, v in cfg.items())
        print(f"{i:3d}  {parts}")


def tag_of(cfg):
    return "_".join(str(v) for v in cfg.values())


def begin_job(out_root, job_index, tag, **meta):
    global _STATUS_DIR, _META
    d = Path(out_root) / "jobs" / f"{int(job_index):03d}_{tag}"
    d.mkdir(parents=True, exist_ok=True)
    _STATUS_DIR = d
    _META = {
        "job": int(job_index),
        "tag": tag,
        "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
        "slurm_array_task_id": os.environ.get("SLURM_ARRAY_TASK_ID"),
        **meta,
    }
    write_status(state="starting", device=str(device_of()))
    return d


def job_dir():
    return _STATUS_DIR


def write_status(**fields):
    if _STATUS_DIR is None:
        return
    rec = {
        **_META,
        **fields,
        "device": fields.get("device", str(device_of())),
        "updated_at": datetime.now(timezone.utc).isoformat(),
    }
    (_STATUS_DIR / "status.json").write_text(json.dumps(rec, indent=2, default=str))
    epoch = fields.get("epoch")
    if epoch is not None:
        line = (
            f"{rec['updated_at']}  epoch={epoch}  "
            f"train={fields.get('train', '')}  val={fields.get('val', '')}  "
            f"best_epoch={fields.get('best_epoch', '')}  state={fields.get('state', '')}\n"
        )
        with (_STATUS_DIR / "epochs.log").open("a") as fh:
            fh.write(line)


def aggregate_scores(out_root, dest_name="scores.csv"):
    out_root = Path(out_root)
    paths = sorted((out_root / "jobs").glob("*/scores.csv"))
    if not paths:
        raise SystemExit(f"no per-job scores under {out_root / 'jobs'}")
    import pandas as pd

    frames = [pd.read_csv(p) for p in paths]
    df = pd.concat(frames, ignore_index=True)
    dest = out_root / dest_name
    df.to_csv(dest, index=False)
    print(f"wrote {dest}  ({len(df)} rows from {len(paths)} jobs)")
    return df
