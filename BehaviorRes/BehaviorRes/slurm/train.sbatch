#!/bin/bash
# Slurm array worker for one BehaviorRes training config.
# Submit with slurm/submit.sh (sets BEH_SCRIPT and --array).
#
#   bash slurm/submit.sh Modelv1.py
#   bash slurm/submit.sh Modelv1.py --cpu
#
#SBATCH --job-name=behres
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=12:00:00
#SBATCH --output=slurm/logs/%x_%A_%a.out
#SBATCH --error=slurm/logs/%x_%A_%a.err

set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
mkdir -p "$ROOT/slurm/logs"

if [[ -f "$ROOT/.venv/bin/activate" ]]; then
  # shellcheck disable=SC1091
  source "$ROOT/.venv/bin/activate"
fi

SCRIPT="${BEH_SCRIPT:?set BEH_SCRIPT to e.g. Modelv1.py}"
DEVICE="${BEH_DEVICE:-auto}"

echo "host=$(hostname)  cuda=${CUDA_VISIBLE_DEVICES:-}  script=$SCRIPT  job=${SLURM_ARRAY_TASK_ID}  device=$DEVICE"
python -c "import torch; print('torch', torch.__version__, 'cuda', torch.cuda.is_available())" || true
python "$SCRIPT" --job "${SLURM_ARRAY_TASK_ID}" --device "$DEVICE"
