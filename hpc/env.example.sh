# Copy to hpc/env.sh (git-ignored) and fill in. hpc/config.sh sources it.
# Nothing user- or cluster-specific is committed; every value below is yours.
#
# OPEN QUESTION FOR THE TEAM: hpc/README.md says coursework runs on HPC Cloud
# Bursting, not Torch, but the MTG-AD Globus transfer is described as landing on
# Torch scratch. Pick the cluster your allocation is on and its profile:
#   source "$REPO_ROOT/hpc/profiles/cloud_bursting.sh"   # partitions documented in hpc/README.md
#   source "$REPO_ROOT/hpc/profiles/torch.sh"            # you must fill in partitions/GRES yourself

# export PYRO_PROFILE=cloud_bursting
# export ACCOUNT=<slurm account>

# Data and caches on scratch (example only; use your own NetID path):
# export MTG_AD_ROOT=/scratch/<netid>/mtg-ad-transfer/mtg-ad
# export PYRO_CACHE=/scratch/<netid>/pyro-cache
# export PYRO_OUTPUTS=/scratch/<netid>/pyro-outputs
# export HF_HOME=/scratch/<netid>/hf_cache

# How jobs get a Python environment: "container" (Singularity overlay from
# hpc/setup_env.sh) or "venv" (activate PYRO_VENV, e.g. a uv venv on scratch).
# export PYRO_ENV_MODE=container
# export PYRO_VENV=/scratch/<netid>/pyro-venv

# API runs: keys only in the environment, never in configs or this repo.
# export OPENAI_API_KEY=...        # set interactively or from a private file outside the repo
# export PYRO_API_SPEND_CAP_USD=20
