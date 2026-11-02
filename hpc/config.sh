# Shared settings for every HPC script, for NYU HPC Cloud Bursting (course cluster).
# Course rule: coursework runs on Cloud Bursting, NOT on Torch.
# Sourced by setup_env.sh, download_models.sh, prepare_data.sh, submit.sh and run_ad.sbatch.

# Course SLURM account (300 GPU-hours per student for Fall 2026).
ACCOUNT="${ACCOUNT:-csci_ga_2271-2026fa}"

# Partitions available to the course:
#   interactive      quick test sessions
#   n2c48m24         CPU only (setup, downloads, data prep)
#   g2-standard-12   1x L4 (24 GB)
#   g2-standard-24   2x L4
#   c12m85-a100-1    1x A100 (40 GB)
#   c24m170-a100-2   2x A100 (40 GB each)
CPU_PARTITION="n2c48m24"

# Environment overlay and model cache on scratch (the course asks for /scratch/NetID).
ENV_DIR="${ENV_DIR:-/scratch/$USER/pyro-env}"
OVERLAY="$ENV_DIR/overlay-50G-10M.ext3"
OVERLAY_SRC="/share/apps/overlay-fs-ext3/overlay-50G-10M.ext3.gz"
export HF_HOME="${HF_HOME:-/scratch/$USER/hf_cache}"   # Muse Glimmer alone is ~60 GB
export HF_HUB_ENABLE_HF_TRANSFER=1

# Jobs run offline after download_models.sh, so a network hiccup can't kill a GPU job.
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"

# CUDA base image from /share/apps/images. If the pinned one isn't there, use the
# newest cuda12 devel image. PyTorch wheels bring their own CUDA libraries, so the
# exact CUDA version in the image matters little.
SIF="${SIF:-/share/apps/images/cuda12.1.1-cudnn8.9.0-devel-ubuntu22.04.2.sif}"
if [ ! -f "$SIF" ]; then
  SIF="$(ls -1 /share/apps/images/cuda12*devel*.sif 2>/dev/null | sort -V | tail -1 || true)"
fi

# Run a command inside the container with the conda env active.
#   in_env [ro|rw] "<shell command>"
in_env() {
  local mode="$1"; shift
  local nv=""
  if [ -e /dev/nvidia0 ]; then nv="--nv"; fi
  if [ -z "$SIF" ]; then
    echo "No CUDA image found; set SIF in hpc/config.sh (ls /share/apps/images)." >&2
    return 1
  fi
  singularity exec $nv --overlay "$OVERLAY:$mode" "$SIF" \
    /bin/bash -c "source /ext3/env.sh && $*"
}
