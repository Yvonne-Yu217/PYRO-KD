#!/usr/bin/env bash
#SBATCH --job-name=pyro-ego-heldout
#SBATCH --account=csci_ga_2271-2026fa
#SBATCH --partition=n2c48m24
#SBATCH --cpus-per-task=4
#SBATCH --mem=12G
#SBATCH --time=03:00:00
#SBATCH --array=0-3%2
#SBATCH --dependency=afterany:41027
#SBATCH --output=/scratch/yy5919/pyro-ego-heldout-%A_%a.log
set -euo pipefail
cd /scratch/yy5919/pyro-revision15-formal-prep
source hpc/config.sh
in_env ro "CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 python -u /scratch/yy5919/pyro-prepare-egoschema-heldout-v2.py --shard ${SLURM_ARRAY_TASK_ID} --shards 4 --report /scratch/yy5919/pyro-resume-results/egoschema-heldout-${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}.json"
