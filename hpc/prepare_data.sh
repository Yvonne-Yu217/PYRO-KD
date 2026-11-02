#!/bin/bash
# Download MTG-AD annotations + videos (yt-dlp) and build the frame/audio cache,
# then write the uniform-selector moments for the smoke test. Needs internet,
# so run it on a CPU node, never a GPU job:
#   srun --account=csci_ga_2271-2026fa --partition=n2c48m24 --time=02:00:00 --pty /bin/bash
#   bash hpc/prepare_data.sh            # dev videos (10)
#   bash hpc/prepare_data.sh test       # the 50 fixed test videos
set -euo pipefail
cd "$(dirname "$0")/.."
source hpc/config.sh

WHICH="${1:-dev}"
in_env ro "python scripts/prepare_mtg_ad.py --download-videos --preprocess --videos '$WHICH'" \
  || echo "[prepare] some videos failed to download; see the list above" >&2
in_env ro "python scripts/build_moments.py --selector U --videos '$WHICH'"
echo "[prepare] done. Next: bash hpc/submit.sh qwen --limit 4"
