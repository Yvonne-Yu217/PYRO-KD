#!/usr/bin/env bash
#SBATCH --job-name=pyro-parallel-eval
#SBATCH --account=csci_ga_2271-2026fa
#SBATCH --partition=c12m85-a100-1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=12
#SBATCH --mem=80G
#SBATCH --time=04:00:00
#SBATCH --requeue
#SBATCH --output=/scratch/yy5919/pyro-secondary-gpu-%j.log
set -euo pipefail
cd /scratch/yy5919/pyro-revision15-formal
source hpc/config.sh
nvidia-smi --query-gpu=timestamp,name,utilization.gpu,memory.used --format=csv -l 30 > /scratch/yy5919/pyro-secondary-util-${SLURM_JOB_ID}.csv &
PYRO_MONITOR_PID=$!
trap 'kill "$PYRO_MONITOR_PID" 2>/dev/null || true' EXIT
in_env ro 'python -c "import torch; assert torch.cuda.is_available(); assert torch.cuda.get_device_properties(0).total_memory > 35*1024**3; print(torch.cuda.get_device_name(0))"'
python -u /scratch/yy5919/pyro-secondary-gpu.py
