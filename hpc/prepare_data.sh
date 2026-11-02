#!/bin/bash
# Download MTG-AD annotations + videos (yt-dlp) and build the frame/audio cache,
# then write the uniform-selector moments for the smoke test. Needs internet,
# so run it on a CPU node, never a GPU job:
#   srun --account=csci_ga_2271-2026fa --partition=n2c48m24 --time=02:00:00 --pty /bin/bash
#   bash hpc/prepare_data.sh                    # MTG-AD dev videos (10)
#   bash hpc/prepare_data.sh mtg-ad test        # the 50 fixed MTG-AD test videos
#   bash hpc/prepare_data.sh reframed           # REFRAMED pilot clips (10)
#   bash hpc/prepare_data.sh reframed test      # the 30 fixed REFRAMED test clips
set -euo pipefail
cd "$(dirname "$0")/.."
source hpc/config.sh

DATASET="${1:-mtg-ad}"     # mtg-ad | reframed
WHICH="${2:-dev}"          # dev | test
case "$DATASET" in
  mtg-ad)   PREP="python scripts/prepare_mtg_ad.py --download-videos --preprocess --videos '$WHICH'" ;;
  reframed) PREP="python scripts/prepare_reframed.py --download-videos --preprocess --videos '$([ "$WHICH" = dev ] && echo pilot || echo test)'" ;;
  *) echo "usage: bash hpc/prepare_data.sh [mtg-ad|reframed] [dev|test]" >&2; exit 1 ;;
esac

in_env ro "$PREP" || echo "[prepare] some videos failed to download; see the list above" >&2
in_env ro "python scripts/build_moments.py --dataset '$DATASET' --selector U --videos '$WHICH'"
echo "[prepare] done. Next: bash hpc/submit.sh qwen --limit 4 --input data/moments/${DATASET}_U_${WHICH}.jsonl"
