#!/bin/bash
# GPU job for scripts/run_ad_vlm.py on HPC Cloud Bursting. Submit through
# hpc/submit.sh, which picks the partition (GPU type/count) per model.
#
#SBATCH --job-name=pyro-ad
#SBATCH --account=csci_ga_2271-2026fa
#SBATCH --partition=g2-standard-12
#SBATCH --gres=gpu:1
#SBATCH --time=02:00:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --requeue

set -euo pipefail
REPO="${SLURM_SUBMIT_DIR}"
cd "$REPO"
source hpc/config.sh

echo "[job] $SLURM_JOB_ID on $(hostname) ($SLURM_JOB_PARTITION) | args: $*"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

# Log GPU utilisation every 30 s for the cost section of the report.
# Cloud instances with low GPU use for 20 minutes are shut down, so check this
# file first if a job dies early.
mkdir -p logs
nvidia-smi --query-gpu=timestamp,index,utilization.gpu,memory.used --format=csv -l 30 \
  > "logs/gpu-${SLURM_JOB_ID}.csv" &
SMI_PID=$!
trap 'kill $SMI_PID 2>/dev/null || true' EXIT

# run_ad_vlm.py appends results and skips finished ids, so a requeued job resumes.
ARGS=$(printf '%q ' "$@")
in_env ro "cd '$REPO' && python scripts/run_ad_vlm.py $ARGS"
