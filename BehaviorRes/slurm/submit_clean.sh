#!/bin/bash
# Submit ExperimentsClean.py as a Slurm array (one config per task).
#
# Entire pipeline, one fold of LTO / LSO / LMO:
#   bash slurm/submit_clean.sh --single-fold
#
# Same, CPU:
#   bash slurm/submit_clean.sh --single-fold --cpu
#
# After the array:
#   python ExperimentsClean.py --single-fold --aggregate

set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

DEVICE="auto"
SBATCH_GPU=(--gpus=1)
PY_ARGS=()
while [[ $# -gt 0 ]]; do
  case "$1" in
    --cpu)
      DEVICE="cpu"
      SBATCH_GPU=()
      shift
      ;;
    *)
      PY_ARGS+=("$1")
      shift
      ;;
  esac
done

if [[ -f "$ROOT/.venv/bin/activate" ]]; then
  # shellcheck disable=SC1091
  source "$ROOT/.venv/bin/activate"
fi

mkdir -p "$ROOT/slurm/logs"

JOB_LIST="$(python ExperimentsClean.py "${PY_ARGS[@]}" --list-jobs)"
N="$(printf '%s\n' "$JOB_LIST" | awk '$1 ~ /^[0-9]+$/ && !found {print $1; found=1}')"

if [[ -z "$N" || ! "$N" =~ ^[0-9]+$ || "$N" -lt 1 ]]; then
  echo "could not read job count from: python ExperimentsClean.py ${PY_ARGS[*]} --list-jobs" >&2
  python ExperimentsClean.py "${PY_ARGS[@]}" --list-jobs
  exit 1
fi
LAST="$((N - 1))"
echo "submitting ExperimentsClean.py  array=0-${LAST}  device=${DEVICE}  args=${PY_ARGS[*]}  ($N configs)"
printf '%s\n' "$JOB_LIST"

CLEAN_ARGS=""
if [[ ${#PY_ARGS[@]} -gt 0 ]]; then
  CLEAN_ARGS="$(IFS='+'; echo "${PY_ARGS[*]}")"
fi

MAX_ARRAY_SIZE=1001
MAX_CONCURRENT=8
GLOBAL_START=0
while (( GLOBAL_START <= LAST )); do
  REMAINING=$((LAST - GLOBAL_START + 1))
  CHUNK_SIZE=$MAX_ARRAY_SIZE
  if (( REMAINING < CHUNK_SIZE )); then
    CHUNK_SIZE=$REMAINING
  fi
  LOCAL_LAST=$((CHUNK_SIZE - 1))
  ARRAY_SPEC="0-${LOCAL_LAST}%${MAX_CONCURRENT}"
  echo "submitting Slurm chunk: global=${GLOBAL_START}-$((GLOBAL_START + LOCAL_LAST))  local=${ARRAY_SPEC}"
  sbatch \
    --job-name="expclean" \
    --chdir="$ROOT" \
    --array="$ARRAY_SPEC" \
    --mail-type=END,FAIL \
    --mail-user=so504@scarletmail.rutgers.edu \
    --output="$ROOT/slurm/logs/%x_%A_%a.out" \
    --error="$ROOT/slurm/logs/%x_%A_%a.err" \
    --export="ALL,BEH_ROOT=${ROOT},BEH_DEVICE=${DEVICE},BEH_CLEAN_ARGS=${CLEAN_ARGS},BEH_ARRAY_OFFSET=${GLOBAL_START}" \
    "${SBATCH_GPU[@]}" \
    "$ROOT/slurm/experiments_clean.sbatch"
  GLOBAL_START=$((GLOBAL_START + CHUNK_SIZE))
done
