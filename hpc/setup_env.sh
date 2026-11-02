#!/bin/bash
# One-time environment setup on Cloud Bursting: Singularity overlay + Miniforge + packages.
#
# Run from the repo root on a CPU compute node (not the login/OOD node):
#   srun --account=csci_ga_2271-2026fa --partition=n2c48m24 --time=02:00:00 --pty /bin/bash
#   bash hpc/setup_env.sh
#
# Re-running is safe: it skips steps that are already done, and always upgrades
# the Python packages (useful when transformers adds model support).
set -euo pipefail
cd "$(dirname "$0")/.."
REPO="$(pwd)"
source hpc/config.sh

mkdir -p "$ENV_DIR" "$HF_HOME"

if [ ! -f "$OVERLAY" ]; then
  echo "[setup] copying overlay to $OVERLAY"
  cp -p "$OVERLAY_SRC" "$ENV_DIR/"
  gunzip "$OVERLAY.gz"
fi

echo "[setup] installing into overlay (takes ~10-20 min the first time)"
[ -n "$SIF" ] || { echo "No CUDA image found; set SIF in hpc/config.sh" >&2; exit 1; }
echo "[setup] image: $SIF"
singularity exec --overlay "$OVERLAY:rw" --env REPO="$REPO" "$SIF" /bin/bash -c '
set -euo pipefail
if [ ! -d /ext3/miniforge3 ]; then
  wget -q https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-Linux-x86_64.sh -O /tmp/miniforge.sh
  bash /tmp/miniforge.sh -b -p /ext3/miniforge3
  rm /tmp/miniforge.sh
fi
cat > /ext3/env.sh <<EOF
#!/bin/bash
unset -f which
source /ext3/miniforge3/etc/profile.d/conda.sh
export PATH=/ext3/miniforge3/bin:\$PATH
EOF
source /ext3/env.sh
conda install -y -q ffmpeg pip           # ffmpeg: frame/audio cache, yt-dlp merging, torchcodec
pip install -q --upgrade torch torchvision
# Project deps (selectors, evaluation, dataset loaders) from pyproject.toml.
pip install -q -e "$REPO"
pip install -q --upgrade transformers accelerate pillow "huggingface_hub[cli]" hf_transfer torchcodec bitsandbytes yt-dlp
python - <<PY
import torch, transformers
print("torch", torch.__version__, "| transformers", transformers.__version__)
print("Muse Glimmer support:", hasattr(transformers, "AutoModelForMultimodalLM"))
PY
'
echo "[setup] done. Next: bash hpc/download_models.sh"
