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
TEMPORAL="${BEH_TEMPORAL:-gru}"
DOUBLE="${BEH_DOUBLE:-0}"
LOG1P="${BEH_LOG1P:-0}"
EXTRA=()
append_list() {
  local flag="$1"
  local val="${2:-}"
  local oldifs="$IFS"
  [[ -z "$val" ]] && return 0
  EXTRA+=("$flag")
  val="${val//,/+}"
  IFS='+'
  # shellcheck disable=SC2206
  EXTRA+=($val)
  IFS="$oldifs"
}
if [[ "$SCRIPT" == "Modelv1.py" ]]; then
  EXTRA+=(--protocol "$PROTOCOL" --temporal "$TEMPORAL")
  if [[ "$DOUBLE" == "1" ]]; then
    EXTRA+=(--double)
  fi
  if [[ "$LOG1P" == "1" ]]; then
    EXTRA+=(--log1p)
  fi
fi
if [[ "$SCRIPT" == "modelv1clusters.py" ]]; then
  if [[ "${BEH_PILOT:-0}" == "1" ]]; then
    EXTRA+=(--pilot)
  fi
  if [[ -n "${BEH_LOSO_FOLDS:-}" ]]; then
    EXTRA+=(--loso-folds "$BEH_LOSO_FOLDS")
  fi
  if [[ -n "${BEH_LOMO_FOLDS:-}" ]]; then
    EXTRA+=(--lomo-folds "$BEH_LOMO_FOLDS")
  fi
  if [[ -n "${BEH_INTERP_FOLDS:-}" ]]; then
    EXTRA+=(--interp-folds "$BEH_INTERP_FOLDS")
  fi
  if [[ -n "${BEH_FOLD_SEED:-}" ]]; then
    EXTRA+=(--fold-seed "$BEH_FOLD_SEED")
  fi
  append_list --targets "${BEH_CLUSTER_TARGETS:-}"
  if [[ -n "${BEH_TEMPORAL:-}" ]]; then
    EXTRA+=(--temporal "$BEH_TEMPORAL")
  fi
  if [[ "${BEH_ONSET_RT:-0}" == "1" ]]; then
    EXTRA+=(--onset-rt)
  fi
  if [[ "${BEH_PAW_ONLY:-0}" == "1" ]]; then
    EXTRA+=(--paw-only)
  fi
  append_list --heads "${BEH_HEADS:-}"
  append_list --subsets "${BEH_SUBSETS:-}"
  append_list --horizons "${BEH_HORIZONS:-}"
fi
if [[ "$SCRIPT" == "heirarchical.py" ]]; then
  if [[ "${BEH_PILOT:-0}" == "1" ]]; then
    EXTRA+=(--pilot)
  fi
  if [[ -n "${BEH_LOSO_FOLDS:-}" ]]; then
    EXTRA+=(--loso-folds "$BEH_LOSO_FOLDS")
  fi
  if [[ -n "${BEH_LOMO_FOLDS:-}" ]]; then
    EXTRA+=(--lomo-folds "$BEH_LOMO_FOLDS")
  fi
  if [[ -n "${BEH_INTERP_FOLDS:-}" ]]; then
    EXTRA+=(--interp-folds "$BEH_INTERP_FOLDS")
  fi
  if [[ -n "${BEH_FOLD_SEED:-}" ]]; then
    EXTRA+=(--fold-seed "$BEH_FOLD_SEED")
  fi
  if [[ "${BEH_PAW_ONLY:-0}" == "1" ]]; then
    EXTRA+=(--paw-only)
  fi
  if [[ "${BEH_SHARED_ENCODER:-0}" == "1" ]]; then
    EXTRA+=(--shared-encoder)
  fi
fi
if [[ "$SCRIPT" == "graph.py" ]]; then
  if [[ "${BEH_PILOT:-0}" == "1" ]]; then
    EXTRA+=(--pilot)
  fi
  if [[ -n "${BEH_LOSO_FOLDS:-}" ]]; then
    EXTRA+=(--loso-folds "$BEH_LOSO_FOLDS")
  fi
  if [[ -n "${BEH_LOMO_FOLDS:-}" ]]; then
    EXTRA+=(--lomo-folds "$BEH_LOMO_FOLDS")
  fi
  if [[ -n "${BEH_INTERP_FOLDS:-}" ]]; then
    EXTRA+=(--interp-folds "$BEH_INTERP_FOLDS")
  fi
  if [[ -n "${BEH_FOLD_SEED:-}" ]]; then
    EXTRA+=(--fold-seed "$BEH_FOLD_SEED")
  fi
  if [[ "${BEH_PAW_ONLY:-0}" == "1" ]]; then
    EXTRA+=(--paw-only)
  fi
  if [[ "${BEH_REGION_GNN:-0}" == "1" ]]; then
    EXTRA+=(--region-gnn)
  fi
  if [[ "${BEH_FUNC_WINDOW:-0}" == "1" ]]; then
    EXTRA+=(--func-window)
  fi
  if [[ -n "${BEH_K_NEIGHBORS:-}" ]]; then
    EXTRA+=(--k-neighbors "$BEH_K_NEIGHBORS")
  fi
  if [[ -n "${BEH_UNIT_LAYERS:-}" ]]; then
    EXTRA+=(--unit-layers "$BEH_UNIT_LAYERS")
  fi
  if [[ -n "${BEH_LAMBDA_UNIT:-}" ]]; then
    EXTRA+=(--lambda-unit "$BEH_LAMBDA_UNIT")
  fi
  if [[ -n "${BEH_LAMBDA_REGION:-}" ]]; then
    EXTRA+=(--lambda-region "$BEH_LAMBDA_REGION")
  fi
fi

OFFSET="${BEH_ARRAY_OFFSET:-0}"
GLOBAL_JOB_ID=$((SLURM_ARRAY_TASK_ID + OFFSET))

echo "host=$(hostname)  cwd=$(pwd)  cuda=${CUDA_VISIBLE_DEVICES:-}  script=$SCRIPT  local_job=${SLURM_ARRAY_TASK_ID}  global_job=${GLOBAL_JOB_ID}  offset=${OFFSET}  device=$DEVICE  protocol=$PROTOCOL  temporal=$TEMPORAL  double=$DOUBLE  log1p=$LOG1P extra=${EXTRA[*]}"
python -c "import torch; print('torch', torch.__version__, 'cuda', torch.cuda.is_available())" || true
python "$SCRIPT" --job "${GLOBAL_JOB_ID}" --device "$DEVICE" "${EXTRA[@]}"
