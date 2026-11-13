#!/bin/bash
# Download MTG-AD annotations + videos (yt-dlp), build the frame/audio cache and the
# VAD gap features, then write the U and A selector moments. Needs internet,
# so run it on a CPU node, never a GPU job:
#   srun --account=csci_ga_2271-2026fa --partition=n2c48m24 --time=02:00:00 --pty /bin/bash
#   bash hpc/prepare_data.sh                    # MTG-AD dev videos (10)
#   bash hpc/prepare_data.sh mtg-ad test        # the 50 fixed MTG-AD test videos
#   bash hpc/prepare_data.sh reframed           # REFRAMED pilot clips (10)
#   bash hpc/prepare_data.sh reframed test      # the 30 fixed REFRAMED test clips
set -euo pipefail
cd "$(dirname "$0")/.."
source hpc/config.sh

# SKIP_DOWNLOAD=1: use videos already in data/<dataset>/videos/ (YouTube may block cluster IPs).
DL="--download-videos"
[ "${SKIP_DOWNLOAD:-0}" = "1" ] && DL=""
DATASET="${1:-mtg-ad}"     # mtg-ad | reframed
WHICH="${2:-dev}"          # dev | test
case "$DATASET" in
  mtg-ad)   PREP="python scripts/prepare_mtg_ad.py $DL --preprocess --videos '$WHICH'" ;;
  reframed) PREP="python scripts/prepare_reframed.py $DL --preprocess --videos '$([ "$WHICH" = dev ] && echo pilot || echo test)'" ;;
  *) echo "usage: bash hpc/prepare_data.sh [mtg-ad|reframed] [dev|test]" >&2; exit 1 ;;
esac

PREP_STATUS=0
in_env ro "$PREP" || PREP_STATUS=$?
if [ "$PREP_STATUS" -ne 0 ]; then
  if [ "$DATASET" = "mtg-ad" ]; then
    echo "[prepare] some videos failed; see data/mtg-ad/preprocess_report.json" >&2
  else
    echo "[prepare] some videos failed; see the preparation output above" >&2
  fi
fi
# Silero VAD + librosa gap features: needed by selector A and the gap-based word cap.
# A failed video is reported and skipped; moments are still built for the rest.
AUDIO_STATUS=0
in_env ro "python scripts/cache_audio_features.py --cache-dir data/cache/$DATASET" || AUDIO_STATUS=$?
if [ "$AUDIO_STATUS" -ne 0 ]; then
  echo "[prepare] audio features failed for some videos; see data/cache/$DATASET/audio_features_report.json" >&2
fi
for SEL in U A; do
  in_env ro "python scripts/build_moments.py --dataset '$DATASET' --selector $SEL --videos '$WHICH'"
done
echo "[prepare] done. Next: bash hpc/submit.sh qwen --limit 4 --input data/moments/${DATASET}_U_${WHICH}.jsonl"
if [ "$PREP_STATUS" -ne 0 ]; then exit "$PREP_STATUS"; fi
exit "$AUDIO_STATUS"
