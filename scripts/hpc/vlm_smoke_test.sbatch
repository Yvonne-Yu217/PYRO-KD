#!/bin/bash
# Serve a frozen VLM with vLLM, run the smoke test against cached pilot frames,
# then shut the server down. Logs GPU, model, latency, and token usage.
#
#   sbatch --account=<your_account> scripts/hpc/vlm_smoke_test.sbatch
#   MODEL=nvidia/Muse-Glimmer-30B-NVFP4 sbatch --account=<acct> --gres=gpu:<blackwell_type>:1 \
#       scripts/hpc/vlm_smoke_test.sbatch
#
# Muse-Glimmer's NVFP4 checkpoint targets Blackwell GPUs; check `sinfo` for
# one before submitting it. Qwen3-VL-8B in bf16 fits a single 40-80 GB GPU.
#SBATCH --job-name=pyro-vlm-smoke
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH --time=01:00:00
#SBATCH --output=logs/%x-%j.out
set -euo pipefail

MODEL="${MODEL:-Qwen/Qwen3-VL-8B-Instruct}"
PORT="${PORT:-$((8000 + RANDOM % 1000))}"
VLLM_ENV="${VLLM_ENV:-${SCRATCH:-$HOME}/envs/vllm}"
CACHE="${CACHE:-}"                 # a data/cache/reframed/<video_id> dir; default: first pilot clip
MAX_MODEL_LEN="${MAX_MODEL_LEN:-16384}"
export HF_HOME="${HF_HOME:-${SCRATCH:-$HOME}/hf_cache}"

cd "${SLURM_SUBMIT_DIR:-$(pwd)}"
mkdir -p logs
if [[ -z "$CACHE" ]]; then
  first=$(python3 -c "import json;print(json.load(open('configs/reframed_pilot.json'))['clips'][0]['video_id'])")
  CACHE="data/cache/reframed/$first"
fi
[[ -f "$CACHE/manifest.json" ]] || { echo "no manifest at $CACHE; run scripts/hpc/prepare_pilot.sbatch first" >&2; exit 1; }

echo "host=$(hostname) model=$MODEL port=$PORT"
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv

"$VLLM_ENV/bin/vllm" serve "$MODEL" \
  --port "$PORT" \
  --max-model-len "$MAX_MODEL_LEN" \
  --limit-mm-per-prompt '{"image": 8}' \
  --seed 0 > "logs/vllm-${SLURM_JOB_ID:-local}.log" 2>&1 &
SERVER_PID=$!
trap 'kill $SERVER_PID 2>/dev/null || true' EXIT

for _ in $(seq 1 120); do
  if curl -sf "http://localhost:$PORT/v1/models" > /dev/null; then break; fi
  if ! kill -0 $SERVER_PID 2>/dev/null; then
    echo "vLLM exited during startup; see logs/vllm-${SLURM_JOB_ID:-local}.log" >&2; exit 1
  fi
  sleep 10
done
curl -sf "http://localhost:$PORT/v1/models" > /dev/null || { echo "vLLM did not become ready" >&2; exit 1; }

uv run --no-sync python scripts/smoke_test_vlm.py \
  --model "$MODEL" --base-url "http://localhost:$PORT/v1" --cache "$CACHE"

nvidia-smi --query-gpu=memory.used --format=csv
