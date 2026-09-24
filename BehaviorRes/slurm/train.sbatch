#!/bin/bash
#SBATCH --job-name=behres
#SBATCH --mem=32G
#SBATCH --time=12:00:00
#SBATCH --mail-user=so504@scarletmail.rutgers.edu
#SBATCH --mail-type=END,FAIL

set -euo pipefail
ROOT="${BEH_ROOT:-${SLURM_SUBMIT_DIR:-$(pwd)}}"
cd "$ROOT"
mkdir -p "$ROOT/slurm/logs"

if [[ -f "$ROOT/.venv/bin/activate" ]]; then
  source "$ROOT/.venv/bin/activate"
fi

SCRIPT="${BEH_SCRIPT:?set BEH_SCRIPT to e.g. Modelv1.py}"
DEVICE="${BEH_DEVICE:-auto}"
PROTOCOL="${BEH_PROTOCOL:-fixed}"
EXTRA=()
if [[ "$SCRIPT" == "Modelv1.py" ]]; then
  EXTRA+=(--protocol "$PROTOCOL")
fi

OFFSET="${BEH_ARRAY_OFFSET:-0}"
GLOBAL_JOB_ID=$((SLURM_ARRAY_TASK_ID + OFFSET))

echo "host=$(hostname)  cwd=$(pwd)  cuda=${CUDA_VISIBLE_DEVICES:-}  script=$SCRIPT  local_job=${SLURM_ARRAY_TASK_ID}  global_job=${GLOBAL_JOB_ID}  offset=${OFFSET}  device=$DEVICE  protocol=$PROTOCOL"
python -c "import torch; print('torch', torch.__version__, 'cuda', torch.cuda.is_available())" || true
python "$SCRIPT" --job "${GLOBAL_JOB_ID}" --device "$DEVICE" "${EXTRA[@]}"
