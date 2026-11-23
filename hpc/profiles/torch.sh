# NYU Torch. Partition names, GPU types and the account are NOT filled in on
# purpose: they have not been verified for this course allocation, and the
# course rule recorded in hpc/README.md says coursework runs on Cloud Bursting.
# Confirm with the instructor before using Torch, then set these in hpc/env.sh.
export PYRO_PROFILE=torch
: "${ACCOUNT:?set ACCOUNT in hpc/env.sh for Torch}"
: "${CPU_PARTITION:?set CPU_PARTITION in hpc/env.sh for Torch}"
: "${GPU_PARTITION:?set GPU_PARTITION in hpc/env.sh for Torch}"
export GPU_GRES="${GPU_GRES:-gpu:1}"
export PYRO_ENV_MODE="${PYRO_ENV_MODE:-venv}"
