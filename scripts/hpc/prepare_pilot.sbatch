#!/bin/bash
# CPU job: download REFRAMED metadata + the 10 pilot videos and cache
# 16 kHz mono audio and 2 fps frames under data/cache/reframed/.
#
#   sbatch --account=<your_account> scripts/hpc/prepare_pilot.sbatch
#SBATCH --job-name=pyro-prepare-pilot
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --time=02:00:00
#SBATCH --output=logs/%x-%j.out
set -euo pipefail

cd "${SLURM_SUBMIT_DIR:-$(pwd)}"
mkdir -p logs
command -v ffmpeg > /dev/null || module load ffmpeg 2>/dev/null || {
  echo "ffmpeg not found; load an ffmpeg module or install a static build" >&2; exit 1; }

uv sync
uv run python scripts/prepare_reframed.py --download-videos --preprocess
