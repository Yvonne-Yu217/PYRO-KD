#!/usr/bin/env bash
#SBATCH --job-name=pyro-ego-controller
#SBATCH --account=csci_ga_2271-2026fa
#SBATCH --partition=n2c48m24
#SBATCH --cpus-per-task=1
#SBATCH --mem=2G
#SBATCH --time=02:05:00
#SBATCH --dependency=afterany:41197
#SBATCH --output=/scratch/yy5919/pyro-ego-continuation-controller-%j.log
set -euo pipefail
cd /scratch/yy5919/pyro-revision15-formal
export CUDA_VISIBLE_DEVICES=""
python -u /scratch/yy5919/pyro-ego-continuation.py controller
