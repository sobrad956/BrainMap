#!/bin/bash
# Resubmit FAILED Modelv1 (or other) Slurm array tasks.
#
#   bash slurm/resubmit_failed.sh
#   bash slurm/resubmit_failed.sh 403918 403919
#   bash slurm/resubmit_failed.sh --dry-run 403919
#   bash slurm/resubmit_failed.sh --cpu 403919
#
# Pulls FAILED array tasks from sacct, reads BEH_ARRAY_OFFSET from the
# original .out log, and sbatch's only those local task IDs.

set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

SCRIPT="Modelv1.py"
PROTOCOL="cv"
DEVICE="auto"
SBATCH_GPU=(--gpus=1)
DRY=0
LOOKBACK="-14days"

usage() {
  echo "usage: bash slurm/resubmit_failed.sh [--dry-run] [--cpu] [--protocol cv] [array_jobid ...]" >&2
  exit 1
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --dry-run) DRY=1; shift ;;
    --cpu) DEVICE="cpu"; SBATCH_GPU=(); shift ;;
    --protocol) PROTOCOL="$2"; shift 2 ;;
    --script) SCRIPT="$2"; shift 2 ;;
    -h|--help) usage ;;
    --*) usage ;;
    *) break ;;
  esac
done

NAME="${SCRIPT%.py}"
mkdir -p "$ROOT/slurm/logs"

if [[ $# -gt 0 ]]; then
  ARRAY_IDS=("$@")
else
  mapfile -t ARRAY_IDS < <(
    sacct -u "$USER" -S "now${LOOKBACK}" --format=JobID,JobName,State -n -P \
      | awk -F'|' -v name="$NAME" '
          $2 == name && $3 == "FAILED" && $1 ~ /^[0-9]+_[0-9]+$/ {
            split($1, a, "_")
            print a[1]
          }' \
      | sort -u
  )
fi

if [[ ${#ARRAY_IDS[@]} -eq 0 ]]; then
  echo "no FAILED ${NAME} array jobs found (pass job ids explicitly)" >&2
  exit 1
fi

JOB_EXPORT="ALL,BEH_SCRIPT=${SCRIPT},BEH_DEVICE=${DEVICE},BEH_ROOT=${ROOT}"
if [[ "$SCRIPT" == "Modelv1.py" ]]; then
  JOB_EXPORT="${JOB_EXPORT},BEH_PROTOCOL=${PROTOCOL}"
fi

offset_from_log() {
  local array_id="$1"
  local task="$2"
  local log="$ROOT/slurm/logs/${NAME}_${array_id}_${task}.out"
  if [[ -f "$log" ]]; then
    sed -n 's/.*offset=\([0-9][0-9]*\).*/\1/p' "$log" | head -n 1
  fi
}

for array_id in "${ARRAY_IDS[@]}"; do
  mapfile -t TASKS < <(
    sacct -j "$array_id" --format=JobID,State -n -P \
      | awk -F'|' '
          $2 == "FAILED" && $1 ~ /^[0-9]+_[0-9]+$/ {
            split($1, a, "_")
            print a[2]
          }' \
      | sort -n | uniq
  )
  if [[ ${#TASKS[@]} -eq 0 ]]; then
    echo "array ${array_id}: no FAILED tasks, skip"
    continue
  fi

  OFFSET=""
  for t in "${TASKS[@]}"; do
    OFFSET="$(offset_from_log "$array_id" "$t" || true)"
    if [[ -n "${OFFSET}" ]]; then
      break
    fi
  done
  if [[ -z "${OFFSET}" ]]; then
    echo "array ${array_id}: could not read offset= from slurm/logs/${NAME}_${array_id}_*.out" >&2
    exit 1
  fi

  ARRAY_SPEC="$(IFS=,; echo "${TASKS[*]}")%8"
  echo "array ${array_id}: ${#TASKS[@]} FAILED tasks  offset=${OFFSET}  --array=${ARRAY_SPEC}"

  if [[ "$DRY" -eq 1 ]]; then
    echo "dry-run: would sbatch ${#TASKS[@]} tasks with BEH_ARRAY_OFFSET=${OFFSET}"
    continue
  fi

  sbatch \
    --job-name="$NAME" \
    --chdir="$ROOT" \
    --array="$ARRAY_SPEC" \
    --mail-type=END,FAIL \
    --mail-user=so504@scarletmail.rutgers.edu \
    --output="$ROOT/slurm/logs/%x_%A_%a.out" \
    --error="$ROOT/slurm/logs/%x_%A_%a.err" \
    --export="${JOB_EXPORT},BEH_ARRAY_OFFSET=${OFFSET}" \
    "${SBATCH_GPU[@]}" \
    "$ROOT/slurm/train.sbatch"
done
