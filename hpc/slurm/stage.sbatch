#!/bin/bash
# Generic, requeue-safe job for one pipeline stage. Submit through
# hpc/submit_stage.sh, which sets the partition, GRES, time and memory.
# Every stage script it runs is resumable, so a requeued job continues.
#
#SBATCH --job-name=pyro-stage
#SBATCH --output=logs/%x-%A_%a.out
#SBATCH --requeue

set -euo pipefail
REPO="${SLURM_SUBMIT_DIR:-$(pwd)}"
cd "$REPO"
source hpc/config.sh
if [ -n "${PYRO_PROFILE:-}" ] && [ -f "hpc/profiles/${PYRO_PROFILE}.sh" ]; then source "hpc/profiles/${PYRO_PROFILE}.sh"; fi

echo "[job] ${SLURM_JOB_ID:-local} task ${SLURM_ARRAY_TASK_ID:-none} on $(hostname) | $*"
mkdir -p logs
SMI_PID=""
if command -v nvidia-smi >/dev/null 2>&1 && [ -e /dev/nvidia0 ]; then
  nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
  nvidia-smi --query-gpu=timestamp,index,utilization.gpu,memory.used --format=csv -l 30 \
    > "logs/gpu-${SLURM_JOB_ID:-local}-${SLURM_ARRAY_TASK_ID:-0}.csv" &
  SMI_PID=$!
fi
trap '[ -n "$SMI_PID" ] && kill $SMI_PID 2>/dev/null || true' EXIT

# Substitute the array index for {TASK} so sharded stages can use --shard {TASK}/N.
ARGS=()
for a in "$@"; do ARGS+=("${a//\{TASK\}/${SLURM_ARRAY_TASK_ID:-0}}"); done
CMD=$(printf '%q ' "${ARGS[@]}")

case "${PYRO_ENV_MODE:-container}" in
  container) in_env ro "cd '$REPO' && $CMD" ;;
  venv)
    : "${PYRO_VENV:?set PYRO_VENV for PYRO_ENV_MODE=venv}"
    source "$PYRO_VENV/bin/activate"
    bash -c "$CMD" ;;
  *) echo "unknown PYRO_ENV_MODE=$PYRO_ENV_MODE" >&2; exit 1 ;;
esac
