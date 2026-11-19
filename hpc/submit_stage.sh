#!/bin/bash
# Submit one pipeline stage with resources chosen per stage. Nothing is
# submitted with DRY_RUN=1 (prints the sbatch command instead).
#
#   bash hpc/submit_stage.sh <stage> [extra args passed to the stage's script]
#
# Stages                 resources        script
#   tests                CPU              python -m unittest discover -s tests
#   preprocess           CPU              scripts/prepare_mtg_ad.py --offline (frames + 16 kHz WAV)
#   audio-features       CPU              scripts/cache_audio_features.py (Silero VAD + gap features)
#   audio-cues           GPU, array       scripts/cache_audio_cues.py --shard {TASK}/N (Whisper, AST, CLAP)
#   visual-features      GPU              scripts/cache_visual_features.py (OpenCLIP + PySceneDetect)
#   selector-only        CPU              scripts/run_experiment.py --stage selector-only
#   moments              CPU              scripts/run_experiment.py --stage moments
#   generate             GPU              scripts/run_experiment.py --stage generate
#   evaluate-generation  GPU (BERTScore)  scripts/run_experiment.py --stage evaluate-generation
#
# Resources come from the profile (hpc/profiles/*.sh via PYRO_PROFILE) and can be
# overridden per call: PARTITION, GRES, TIME, MEM, CPUS, ARRAY (e.g. 0-7), SHARDS.
# CONFIG selects the experiment config (default configs/experiments/mtg_ad_dev.toml).
set -euo pipefail
cd "$(dirname "$0")/.."
source hpc/config.sh
if [ -n "${PYRO_PROFILE:-}" ] && [ -f "hpc/profiles/${PYRO_PROFILE}.sh" ]; then source "hpc/profiles/${PYRO_PROFILE}.sh"; fi

if [ $# -lt 1 ]; then sed -n '2,22p' "$0"; exit 1; fi
stage="$1"; shift
CONFIG="${CONFIG:-configs/experiments/mtg_ad_dev.toml}"
CACHE="${PYRO_CACHE:-data/cache}/mtg-ad"
SHARDS="${SHARDS:-8}"
CPU=( --partition="${PARTITION:-${CPU_PARTITION:?set CPU_PARTITION}}" --cpus-per-task="${CPUS:-8}" --mem="${MEM:-32G}" )
GPU=( --partition="${PARTITION:-${GPU_PARTITION:?set GPU_PARTITION (see hpc/profiles)}}" --gres="${GRES:-${GPU_GRES:-gpu:1}}" --cpus-per-task="${CPUS:-8}" --mem="${MEM:-64G}" )
ARRAY_OPT=()

case "$stage" in
  tests)               RES=("${CPU[@]}"); T="${TIME:-00:30:00}"; CMD=(python -m unittest discover -s tests) ;;
  preprocess)          RES=("${CPU[@]}"); T="${TIME:-04:00:00}"
                       CMD=(python scripts/prepare_mtg_ad.py --offline --data-dir "${MTG_AD_ROOT:?set MTG_AD_ROOT}" --cache-dir "$CACHE" --preprocess) ;;
  audio-features)      RES=("${CPU[@]}"); T="${TIME:-04:00:00}"; CMD=(python scripts/cache_audio_features.py --cache-dir "$CACHE") ;;
  audio-cues)          RES=("${GPU[@]}"); T="${TIME:-04:00:00}"; ARRAY_OPT=(--array="${ARRAY:-0-$((SHARDS - 1))}")
                       CMD=(python scripts/cache_audio_cues.py --cache-dir "$CACHE" --device cuda --shard "{TASK}/$SHARDS") ;;
  visual-features)     RES=("${GPU[@]}"); T="${TIME:-04:00:00}"; CMD=(python scripts/cache_visual_features.py --cache-dir "$CACHE") ;;
  selector-only)       RES=("${CPU[@]}"); T="${TIME:-02:00:00}"; CMD=(python scripts/run_experiment.py --config "$CONFIG" --stage selector-only) ;;
  moments)             RES=("${CPU[@]}"); T="${TIME:-00:30:00}"; CMD=(python scripts/run_experiment.py --config "$CONFIG" --stage moments) ;;
  generate)            RES=("${GPU[@]}"); T="${TIME:-04:00:00}"; CMD=(python scripts/run_experiment.py --config "$CONFIG" --stage generate) ;;
  evaluate-generation) RES=("${GPU[@]}"); T="${TIME:-01:00:00}"; CMD=(python scripts/run_experiment.py --config "$CONFIG" --stage evaluate-generation) ;;
  *) echo "unknown stage '$stage'" >&2; exit 1 ;;
esac

mkdir -p logs
SB=(sbatch --account="${ACCOUNT:?set ACCOUNT}" --job-name="pyro-$stage" --time="$T" "${RES[@]}" ${ARRAY_OPT[@]+"${ARRAY_OPT[@]}"}
    hpc/slurm/stage.sbatch "${CMD[@]}" "$@")
if [ "${DRY_RUN:-0}" = "1" ]; then printf '%q ' "${SB[@]}"; echo; exit 0; fi
"${SB[@]}"
