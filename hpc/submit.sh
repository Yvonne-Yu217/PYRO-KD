#!/bin/bash
# Submit an AD generation run (scripts/run_ad_vlm.py) with the right Cloud
# Bursting GPU per model.
#
# Usage (from the repo root, e.g. in an OOD terminal):
#   bash hpc/submit.sh <preset> [run_ad_vlm.py args...]
#
# Presets                        GPU                  GPU-hours per wall hour
#   qwen      Qwen3-VL-8B        1x L4 (24 GB)        1
#   qwen-a100 Qwen3-VL-8B        1x A100 (40 GB)      1   (faster; use for big runs)
#   muse      Muse Glimmer 30B   2x A100, bf16        2   (full precision)
#   muse-4bit Muse Glimmer 30B   1x A100, 4-bit       1   (cheaper, slight quality loss)
#
# Examples:
#   bash hpc/submit.sh qwen --limit 4                                  # smoke test
#   bash hpc/submit.sh qwen-a100 --input data/moments/mtg-ad_U_dev.jsonl
#   bash hpc/submit.sh muse --input data/moments/mtg-ad_U_dev.jsonl --limit 4
set -euo pipefail
cd "$(dirname "$0")/.."
source hpc/config.sh

if [ $# -lt 1 ]; then sed -n '2,20p' "$0"; exit 1; fi
preset="$1"; shift

case "$preset" in
  qwen)
    # ~17 GB of bf16 weights; 4 frames x batch 2 leaves headroom on a 24 GB L4.
    MODEL_ARGS=(--model qwen3-vl-8b --batch-size 2)
    SB=(--job-name=pyro-qwen --partition=g2-standard-12 --gres=gpu:1 --time=02:00:00) ;;
  qwen-a100)
    MODEL_ARGS=(--model qwen3-vl-8b --batch-size 8)
    SB=(--job-name=pyro-qwen --partition=c12m85-a100-1 --gres=gpu:1 --time=02:00:00) ;;
  muse)
    # ~60 GB of bf16 weights split across two 40 GB A100s by device_map="auto".
    # Muse reasons before answering, so it needs more new tokens than Qwen.
    MODEL_ARGS=(--model muse-glimmer-30b --batch-size 4 --max-new-tokens 512 --reasoning-strength low)
    SB=(--job-name=pyro-muse --partition=c24m170-a100-2 --gres=gpu:2 --time=03:00:00) ;;
  muse-4bit)
    MODEL_ARGS=(--model muse-glimmer-30b --quant 4bit --batch-size 4 --max-new-tokens 512 --reasoning-strength low)
    SB=(--job-name=pyro-muse4 --partition=c12m85-a100-1 --gres=gpu:1 --time=03:00:00) ;;
  *)
    echo "Unknown preset '$preset'. Use qwen, qwen-a100, muse or muse-4bit." >&2; exit 1 ;;
esac

DEFAULT_INPUT=data/moments/mtg-ad_U_dev.jsonl
if [[ " $* " != *" --input "* ]]; then
  [ -f "$DEFAULT_INPUT" ] || { echo "$DEFAULT_INPUT not found; run bash hpc/prepare_data.sh first" >&2; exit 1; }
  set -- --input "$DEFAULT_INPUT" "$@"
fi

mkdir -p logs outputs
# Later args override earlier ones in argparse, so your args win over the preset.
sbatch --account="$ACCOUNT" "${SB[@]}" hpc/run_ad.sbatch "${MODEL_ARGS[@]}" "$@"
