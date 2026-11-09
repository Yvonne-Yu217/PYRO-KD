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
singularity exec --overlay "$OVERLAY:rw" "$SIF" /bin/bash -c '
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
# Never read or write the host ~/.local: it leaks packages in and out of the overlay.
export PIP_USER=0
export PYTHONNOUSERSITE=1
EOF
source /ext3/env.sh
python - <<PY
import os, sysconfig
site_dir = sysconfig.get_paths()["purelib"]
if not os.access(site_dir, os.W_OK):
    print("WARNING: " + site_dir + " is not writable; pip will install into ~/.local instead.")
PY
echo "[setup] conda: ffmpeg"
conda install -y -q ffmpeg pip           # ffmpeg: frame/audio cache, yt-dlp merging, torchcodec
echo "[setup] pip: torch"
pip install -q --upgrade torch torchvision
# Model access first, so downloads work even if a later step fails.
echo "[setup] pip: transformers and model tooling"
pip install -q --upgrade transformers accelerate pillow "huggingface_hub[cli]" hf_transfer torchcodec bitsandbytes yt-dlp
# Project dependencies (mirrors pyproject.toml). Not `pip install -e .`: the repo src/
# packages are named datasets/utils/..., which would shadow Hugging Face datasets.
# opencv-python-headless: the container has no libGL for the GUI build.
echo "[setup] pip: project dependencies"
pip install -q --upgrade tqdm torchmetrics opencv-python-headless librosa silero-vad \
  scenedetect open-clip-torch bert-score sentence-transformers cryptography
echo "[setup] verifying imports"
python - <<PY
import importlib
for name in ["torch", "transformers", "huggingface_hub", "silero_vad", "librosa", "open_clip",
             "scenedetect", "bert_score", "sentence_transformers", "cryptography", "cv2", "yt_dlp"]:
    module = importlib.import_module(name)
    print("  ok", name, getattr(module, "__version__", ""))
import transformers
print("Muse Glimmer support:", hasattr(transformers, "AutoModelForMultimodalLM"))
PY
'
echo "[setup] done. Next: bash hpc/download_models.sh"
