#!/bin/bash
# One-time: create a separate uv env for vLLM so its pinned torch/CUDA build
# does not fight the project's cu126 torch. Run on a GPU login/compute node.
#
#   bash scripts/hpc/setup_vllm_env.sh            # default env: $SCRATCH/envs/vllm
#   VLLM_ENV=/path/to/env bash scripts/hpc/setup_vllm_env.sh
set -euo pipefail

VLLM_ENV="${VLLM_ENV:-${SCRATCH:-$HOME}/envs/vllm}"
# Qwen3-VL needs a recent vLLM; record the installed version in the run log.
VLLM_SPEC="${VLLM_SPEC:-vllm}"

uv venv --python 3.12 "$VLLM_ENV"
VIRTUAL_ENV="$VLLM_ENV" uv pip install "$VLLM_SPEC"
"$VLLM_ENV/bin/python" -c "import vllm, torch; print('vllm', vllm.__version__, 'torch', torch.__version__, 'cuda', torch.version.cuda)"
echo "vLLM env ready at $VLLM_ENV"
