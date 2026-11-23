#!/usr/bin/env bash
#SBATCH --job-name=pyro-ego-inputs
#SBATCH --account=csci_ga_2271-2026fa
#SBATCH --partition=n2c48m24
#SBATCH --cpus-per-task=4
#SBATCH --mem=12G
#SBATCH --time=02:00:00
#SBATCH --dependency=afterok:41038
#SBATCH --output=/scratch/yy5919/pyro-ego-inputs-%j.log
set -euo pipefail
cd /scratch/yy5919/pyro-revision15-formal
source hpc/config.sh
in_env ro 'CUDA_VISIBLE_DEVICES="" OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4 python -u /scratch/yy5919/pyro-build-egoschema-heldout-inputs.py'
