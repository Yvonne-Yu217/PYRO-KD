#!/bin/bash
# Download model weights into $HF_HOME once, so GPU jobs never wait on the network.
#
# Run from the repo root on a CPU node (don't spend GPU-hours on downloads):
#   srun --account=csci_ga_2271-2026fa --partition=n2c48m24 --time=02:00:00 --pty /bin/bash
#   bash hpc/download_models.sh                 # Qwen3-VL-8B + Muse Glimmer (~80 GB)
#   bash hpc/download_models.sh Qwen/Qwen2.5-VL-7B-Instruct # anything else
#
# If a repo is gated, run `hf auth login` inside the env first (or export HF_TOKEN).
set -euo pipefail
cd "$(dirname "$0")/.."
source hpc/config.sh
export HF_HUB_OFFLINE=0

MODELS=("$@")
if [ ${#MODELS[@]} -eq 0 ]; then
  MODELS=("Qwen/Qwen3-VL-8B-Instruct" "meta-models/Muse-Glimmer-30B")
fi

for m in "${MODELS[@]}"; do
  echo "[download] $m -> $HF_HOME"
  in_env ro "hf download '$m'"
done
echo "[download] done. Next: bash hpc/prepare_data.sh"
