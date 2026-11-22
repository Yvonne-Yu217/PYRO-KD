#!/bin/bash
# Run on a CPU compute allocation with an authenticated SSH control master.
# Never modifies/deletes source data. Re-running resumes partial transfers.
set -euo pipefail
: "${TORCH_SOCKET:?Set TORCH_SOCKET to the authenticated SSH control socket}"
TORCH_HOST="${TORCH_HOST:-yy5919@login.torch.hpc.nyu.edu}"
TORCH_SOURCE="${TORCH_SOURCE:-/scratch/yy5919/mtg-ad-transfer}"
TRANSFER_DEST="${TRANSFER_DEST:-/scratch/yy5919/mtg-ad-transfer}"
TRANSFER_BWLIMIT="${TRANSFER_BWLIMIT:-100000}"
# Paths used in rsync's remote shell must not contain shell metacharacters.
for value in "$TORCH_SOCKET" "$TORCH_SOURCE"; do
  [[ "$value" =~ ^/[a-zA-Z0-9_./-]+$ ]] || { echo 'Unsupported SSH/source path'; exit 2; }
done
ssh -S "$TORCH_SOCKET" -O check "$TORCH_HOST"
mkdir -p "$TRANSFER_DEST"
exec 9>"$TRANSFER_DEST/.transfer.lock"
flock -n 9 || { echo 'Another transfer owns the registry lock'; exit 1; }
for dataset in mtg-ad Evaluation-datasets/egoschema Evaluation-datasets/reframed; do
  date -u
  echo "Transferring $dataset"
  mkdir -p "$TRANSFER_DEST/$dataset"
  rsync -rlt --partial --partial-dir=.rsync-partial --bwlimit="$TRANSFER_BWLIMIT" \
    --timeout=300 --stats --info=progress2 --exclude='*.partial.mp4' \
    -e "ssh -S $TORCH_SOCKET -o BatchMode=yes -o ServerAliveInterval=30" \
    "$TORCH_HOST:$TORCH_SOURCE/$dataset/" "$TRANSFER_DEST/$dataset/"
  echo "TRANSFER_COMPLETE $dataset"
done
date -u
