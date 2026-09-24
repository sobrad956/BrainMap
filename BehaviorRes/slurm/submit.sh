#!/bin/bash
# Submit a BehaviorRes training script as a Slurm array (one config per task).
#
#   bash slurm/submit.sh Modelv1.py
#   bash slurm/submit.sh Modelv1.py --cpu
#   bash slurm/submit.sh Modelv1.py --protocol cv
#   bash slurm/submit.sh Modelv1.py --protocol cv --cpu
#
# Extra arguments after the flags above are passed to sbatch.
#
# After a fixed array:
#   python Modelv1.py --aggregate
# After a cv array:
#   python Modelv1.py --protocol cv --aggregate

set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

if [[ $# -lt 1 ]]; then
  echo "usage: bash slurm/submit.sh Modelv1.py [--cpu] [--protocol fixed|cv|all] [sbatch args...]" >&2
  exit 1
fi

SCRIPT="$1"
shift

DEVICE="auto"
PROTOCOL="fixed"
SBATCH_GPU=(--gpus=1)
while [[ $# -gt 0 ]]; do
  case "$1" in
    --cpu)
      DEVICE="cpu"
      SBATCH_GPU=()
      shift
      ;;
    --protocol)
      PROTOCOL="$2"
      shift 2
      ;;
    *)
      break
      ;;
  esac
done

if [[ -f "$ROOT/.venv/bin/activate" ]]; then
  # shellcheck disable=SC1091
  source "$ROOT/.venv/bin/activate"
fi

mkdir -p "$ROOT/slurm/logs"

LIST_ARGS=()
JOB_EXPORT="ALL,BEH_SCRIPT=${SCRIPT},BEH_DEVICE=${DEVICE},BEH_ROOT=${ROOT}"
if [[ "$SCRIPT" == "Modelv1.py" ]]; then
  LIST_ARGS+=(--protocol "$PROTOCOL")
  JOB_EXPORT="${JOB_EXPORT},BEH_PROTOCOL=${PROTOCOL}"
elif [[ "$PROTOCOL" != "fixed" ]]; then
  echo "$SCRIPT has no --protocol flag (only Modelv1.py does). got --protocol ${PROTOCOL}" >&2
  exit 1
fi

JOB_LIST="$(python "$SCRIPT" "${LIST_ARGS[@]}" --list-jobs)"
N="$(printf '%s\n' "$JOB_LIST" | awk '$1 ~ /^[0-9]+$/ && !found {print $1; found=1}')"

if [[ -z "$N" || ! "$N" =~ ^[0-9]+$ || "$N" -lt 1 ]]; then
  echo "could not read job count from: python $SCRIPT ${LIST_ARGS[*]} --list-jobs" >&2
  python "$SCRIPT" "${LIST_ARGS[@]}" --list-jobs
  exit 1
fi
LAST="$((N - 1))"
echo "submitting $SCRIPT  protocol=${PROTOCOL}  array=0-${LAST}  device=${DEVICE}  ($N configs)"
printf '%s\n' "$JOB_LIST"

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

  CHUNK_EXPORT="${JOB_EXPORT},BEH_ARRAY_OFFSET=${GLOBAL_START}"

  echo "submitting Slurm chunk: global=${GLOBAL_START}-$((GLOBAL_START + LOCAL_LAST))  local=${ARRAY_SPEC}"

  sbatch \
    --job-name="${SCRIPT%.py}" \
    --chdir="$ROOT" \
    --array="$ARRAY_SPEC" \
    --mail-type=END,FAIL \
    --mail-user=so504@scarletmail.rutgers.edu \
    --output="$ROOT/slurm/logs/%x_%A_%a.out" \
    --error="$ROOT/slurm/logs/%x_%A_%a.err" \
    --export="${CHUNK_EXPORT}" \
    "${SBATCH_GPU[@]}" \
    "$@" \
    "$ROOT/slurm/train.sbatch"

  GLOBAL_START=$((GLOBAL_START + CHUNK_SIZE))
done
