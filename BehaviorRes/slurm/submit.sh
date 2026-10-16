#!/bin/bash
# Submit a BehaviorRes training script as a Slurm array (one config per task).
#
#   bash slurm/submit.sh Modelv1.py
#   bash slurm/submit.sh Modelv1.py --cpu
#   bash slurm/submit.sh Modelv1.py --protocol cv
#   bash slurm/submit.sh Modelv1.py --protocol cv --cpu
#   bash slurm/submit.sh Modelv1.py --protocol cv --temporal lstm
#   bash slurm/submit.sh Modelv1.py --protocol cv --temporal transformer --double --log1p
#   bash slurm/submit.sh modelv1clusters.py
#   bash slurm/submit.sh modelv1clusters.py --pilot
#   bash slurm/submit.sh modelv1clusters.py --pilot --onset-rt
#   bash slurm/submit.sh modelv1clusters.py --pilot --loso-folds 5 --lomo-folds 5
#   bash slurm/submit.sh modelv1clusters.py --cpu
#
# Extra arguments after the flags above are passed to sbatch.
#
# After a fixed array:
#   python Modelv1.py --aggregate
# After a cv array:
#   python Modelv1.py --protocol cv --aggregate
# After a variant array, pass the same architecture flags:
#   python Modelv1.py --protocol cv --temporal lstm --aggregate
#   python Modelv1.py --protocol cv --temporal transformer --double --log1p --aggregate
# After the cluster experiment (writes Modelv1ClustersV2):
#   python modelv1clusters.py --aggregate
# After a pilot array, pass the same fold flags:
#   python modelv1clusters.py --pilot --aggregate

set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

if [[ $# -lt 1 ]]; then
  echo "usage: bash slurm/submit.sh Modelv1.py|modelv1clusters.py [--cpu] [--protocol fixed|cv|all] [--temporal gru|lstm|transformer] [--double] [--log1p] [--pilot] [--loso-folds N] [--lomo-folds N] [--interp-folds N] [--fold-seed N] [sbatch args...]" >&2
  exit 1
fi

SCRIPT="$1"
shift

DEVICE="auto"
PROTOCOL="fixed"
TEMPORAL="gru"
DOUBLE=0
LOG1P=0
PILOT=0
LOSO_FOLDS=""
LOMO_FOLDS=""
INTERP_FOLDS=""
FOLD_SEED=""
CLUSTER_TARGETS=""
ONSET_RT=0
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
    --temporal)
      TEMPORAL="$2"
      shift 2
      ;;
    --double)
      DOUBLE=1
      shift
      ;;
    --log1p)
      LOG1P=1
      shift
      ;;
    --pilot)
      PILOT=1
      shift
      ;;
    --loso-folds)
      LOSO_FOLDS="$2"
      shift 2
      ;;
    --lomo-folds)
      LOMO_FOLDS="$2"
      shift 2
      ;;
    --fold-seed)
      FOLD_SEED="$2"
      shift 2
      ;;
    --interp-folds)
      INTERP_FOLDS="$2"
      shift 2
      ;;
    --targets)
      CLUSTER_TARGETS="$2"
      shift 2
      ;;
    --onset-rt)
      ONSET_RT=1
      shift
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
  LIST_ARGS+=(--protocol "$PROTOCOL" --temporal "$TEMPORAL")
  JOB_EXPORT="${JOB_EXPORT},BEH_PROTOCOL=${PROTOCOL},BEH_TEMPORAL=${TEMPORAL},BEH_DOUBLE=${DOUBLE},BEH_LOG1P=${LOG1P}"
  if [[ "$DOUBLE" == "1" ]]; then
    LIST_ARGS+=(--double)
  fi
  if [[ "$LOG1P" == "1" ]]; then
    LIST_ARGS+=(--log1p)
  fi
elif [[ "$SCRIPT" == "modelv1clusters.py" ]]; then
  if [[ "$PROTOCOL" != "fixed" || "$TEMPORAL" != "gru" || "$DOUBLE" != "0" || "$LOG1P" != "0" ]]; then
    echo "modelv1clusters.py has a fixed GRU+attn A1 grid (loss × sampling × fold). do not pass --protocol/--temporal/--double/--log1p" >&2
    exit 1
  fi
  if [[ "$PILOT" == "1" ]]; then
    LIST_ARGS+=(--pilot)
  fi
  if [[ -n "$LOSO_FOLDS" ]]; then
    LIST_ARGS+=(--loso-folds "$LOSO_FOLDS")
  fi
  if [[ -n "$LOMO_FOLDS" ]]; then
    LIST_ARGS+=(--lomo-folds "$LOMO_FOLDS")
  fi
  if [[ -n "$INTERP_FOLDS" ]]; then
    LIST_ARGS+=(--interp-folds "$INTERP_FOLDS")
  fi
  if [[ -n "$FOLD_SEED" ]]; then
    LIST_ARGS+=(--fold-seed "$FOLD_SEED")
  fi
  if [[ -n "$CLUSTER_TARGETS" ]]; then
    LIST_ARGS+=(--targets)
    IFS=',' read -r -a _tgts <<< "$CLUSTER_TARGETS"
    LIST_ARGS+=("${_tgts[@]}")
  fi
  if [[ "$ONSET_RT" == "1" ]]; then
    LIST_ARGS+=(--onset-rt)
  fi
  JOB_EXPORT="${JOB_EXPORT},BEH_PILOT=${PILOT},BEH_LOSO_FOLDS=${LOSO_FOLDS},BEH_LOMO_FOLDS=${LOMO_FOLDS},BEH_INTERP_FOLDS=${INTERP_FOLDS},BEH_FOLD_SEED=${FOLD_SEED},BEH_CLUSTER_TARGETS=${CLUSTER_TARGETS},BEH_ONSET_RT=${ONSET_RT}"
elif [[ "$PILOT" == "1" || -n "$LOSO_FOLDS" || -n "$LOMO_FOLDS" || -n "$INTERP_FOLDS" ]]; then
  echo "$SCRIPT does not take --pilot / --loso-folds / --lomo-folds; those are for modelv1clusters.py" >&2
  exit 1
elif [[ "$PROTOCOL" != "fixed" || "$TEMPORAL" != "gru" || "$DOUBLE" != "0" || "$LOG1P" != "0" ]]; then
  echo "$SCRIPT only forwards Modelv1.py architecture flags. got protocol=${PROTOCOL} temporal=${TEMPORAL} double=${DOUBLE} log1p=${LOG1P}" >&2
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
if [[ "$SCRIPT" == "modelv1clusters.py" ]]; then
  echo "submitting $SCRIPT  array=0-${LAST}  device=${DEVICE}  pilot=${PILOT}  loso=${LOSO_FOLDS:-all}  lomo=${LOMO_FOLDS:-all}  ($N configs)"
else
  echo "submitting $SCRIPT  protocol=${PROTOCOL}  temporal=${TEMPORAL}  double=${DOUBLE}  log1p=${LOG1P}  array=0-${LAST}  device=${DEVICE}  ($N configs)"
fi
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
