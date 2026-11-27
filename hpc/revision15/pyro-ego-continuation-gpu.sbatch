#!/usr/bin/env bash
#SBATCH --job-name=pyro-ego-continue-41197
#SBATCH --account=csci_ga_2271-2026fa
#SBATCH --partition=c12m85-a100-1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=12
#SBATCH --mem=80G
#SBATCH --time=02:30:00
#SBATCH --output=/scratch/yy5919/pyro-ego-continuation-gpu-%j.log
set -euo pipefail
cd /scratch/yy5919/pyro-revision15-formal
python -u /scratch/yy5919/pyro-ego-continuation.py worker --conditions "$@"
