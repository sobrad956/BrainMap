#!/bin/bash
# Submit a BehaviorRes training script as a Slurm array (one config per task).
#
#   bash slurm/submit.sh Modelv1.py          # GPU if the partition provides one
#   bash slurm/submit.sh Modelv1.py --cpu    # CPU nodes, --device cpu
#   bash slurm/submit.sh Baselinev1.py
#   bash slurm/submit.sh BWMtest.py
#
# After the array finishes:
#   python Modelv1.py --aggregate
#
# Local sequential (no Slurm), CPU:
#   python Modelv1.py --device cpu

set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

if [[ $# -lt 1 ]]; then
  echo "usage: bash slurm/submit.sh Modelv1.py [--cpu] [extra sbatch args...]" >&2
  exit 1
fi

SCRIPT="$1"
shift

DEVICE="auto"
  SBATCH_GPU=(--gpus=1)
if [[ "${1:-}" == "--cpu" ]]; then
  DEVICE="cpu"
  SBATCH_GPU=()
  shift
fi

if [[ -f "$ROOT/.venv/bin/activate" ]]; then
  # shellcheck disable=SC1091
  source "$ROOT/.venv/bin/activate"
fi

mkdir -p "$ROOT/slurm/logs"
N="$(python "$SCRIPT" --list-jobs | awk 'NR==1 {print $1}')"
if [[ -z "$N" || "$N" -lt 1 ]]; then
  echo "could not read job count from: python $SCRIPT --list-jobs" >&2
  python "$SCRIPT" --list-jobs
  exit 1
fi
LAST="$((N - 1))"
echo "submitting $SCRIPT  array=0-${LAST}  device=${DEVICE}  ($N configs)"
python "$SCRIPT" --list-jobs

sbatch \
  --job-name="${SCRIPT%.py}" \
  --array="0-${LAST}" \
  --export="ALL,BEH_SCRIPT=${SCRIPT},BEH_DEVICE=${DEVICE}" \
  "${SBATCH_GPU[@]}" \
  "$@" \
  "$ROOT/slurm/train.sbatch"
