#!/bin/bash
#SBATCH --job-name=expclean
#SBATCH --mem=32G
#SBATCH --time=12:00:00
#SBATCH --mail-user=so504@scarletmail.rutgers.edu
#SBATCH --mail-type=END,FAIL

set -euo pipefail
ROOT="${BEH_ROOT:-${SLURM_SUBMIT_DIR:-$(pwd)}}"
cd "$ROOT"
mkdir -p "$ROOT/slurm/logs"

if [[ -f "$ROOT/.venv/bin/activate" ]]; then
  # shellcheck disable=SC1091
  source "$ROOT/.venv/bin/activate"
fi

DEVICE="${BEH_DEVICE:-auto}"
OFFSET="${BEH_ARRAY_OFFSET:-0}"
GLOBAL_JOB_ID=$((SLURM_ARRAY_TASK_ID + OFFSET))

EXTRA=()
if [[ -n "${BEH_CLEAN_ARGS:-}" ]]; then
  IFS='+' read -r -a EXTRA <<< "${BEH_CLEAN_ARGS}"
fi

echo "host=$(hostname)  cwd=$(pwd)  cuda=${CUDA_VISIBLE_DEVICES:-}  local_job=${SLURM_ARRAY_TASK_ID}  global_job=${GLOBAL_JOB_ID}  offset=${OFFSET}  device=$DEVICE  extra=${EXTRA[*]}"
python -c "import torch; print('torch', torch.__version__, 'cuda', torch.cuda.is_available())" || true
python ExperimentsClean.py --job "${GLOBAL_JOB_ID}" --device "$DEVICE" "${EXTRA[@]}"
