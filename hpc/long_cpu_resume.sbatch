#!/bin/bash
#SBATCH --job-name=pyro-long-cpu
#SBATCH --account=csci_ga_2271-2026fa
#SBATCH --partition=n2c48m24
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --time=03:00:00
#SBATCH --requeue

# Submit from a detached source worktree with --dependency=afterany:<active CPU job>.
# First argument: absolute original outputs/verification directory, containing
# the pinned model download reports. No GPU is requested or used.
set -euo pipefail
cd "${SLURM_SUBMIT_DIR}"
source hpc/config.sh
REPORT_ROOT="${1:?absolute verification directory required}"
DERIVED_ROOT="/scratch/$USER/pyro-kd-derived"
RUN_REPORTS="$REPORT_ROOT/long-cpu-resume-${SLURM_JOB_ID}-attempt-${SLURM_RESTART_COUNT:-0}"
mkdir -p "$RUN_REPORTS"
git rev-parse HEAD > "$RUN_REPORTS/source_commit.txt"
git status --porcelain > "$RUN_REPORTS/source_status.txt"
if [[ -s "$RUN_REPORTS/source_status.txt" ]]; then
  echo 'Source worktree is dirty; refusing an unrecorded recipe change' >&2
  exit 2
fi
cp "$REPORT_ROOT/multilingual-speech-download.json" "$RUN_REPORTS/model-download.json"
python - "$RUN_REPORTS/model-download.json" <<'PIN_CHECK'
import json,sys
record=json.load(open(sys.argv[1]))
config=json.load(open('configs/multilingual_speech_model.json'))
assert record['complete']
assert {(r['repo_id'],r['revision']) for r in record['models']} == {(r['repo_id'],r['revision']) for r in config['models']}, 'model report differs from frozen pins'
PIN_CHECK
mapfile -t VIDEO_IDS < <(python -c 'import json; print("\n".join(r["video_id"] for r in json.load(open("configs/video_mme_long_subset.json"))["videos"]))')
[[ ${#VIDEO_IDS[@]} -eq 12 ]]
VIDEO_ARGS=()
for ident in "${VIDEO_IDS[@]}"; do VIDEO_ARGS+=(--video-id "$ident"); done
failures=0
run_step() {
  local name="$1"; shift
  local task_command
  printf -v task_command '%q ' "$@"
  echo "[step] $name $(date -u +%FT%TZ)"
  if in_env ro "$task_command" > "$RUN_REPORTS/$name.log" 2>&1; then
    echo "$name 0" >> "$RUN_REPORTS/status.txt"
  else
    local code=$?
    echo "$name $code" >> "$RUN_REPORTS/status.txt"
    failures=$((failures+1))
  fi
}
run_step speech env OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 \
  "/scratch/$USER/pyro-audio-env/bin/python" -u scripts/cache_speech_features.py \
  --cache-dir "$DERIVED_ROOT/cache/video-mme" \
  --model-report "$RUN_REPORTS/model-download.json" \
  --model-id Systran/faster-whisper-small --language auto
run_step visual env OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 python -u scripts/cache_visual_features.py \
  --cache-dir "$DERIVED_ROOT/cache/video-mme" --batch-size 8
run_step speech-check python scripts/verify_speech_cache.py \
  --cache-dir "$DERIVED_ROOT/cache/video-mme" "${VIDEO_ARGS[@]}" \
  --model-report "$RUN_REPORTS/model-download.json" \
  --model-id Systran/faster-whisper-small --language auto --output "$RUN_REPORTS/speech-check.json"
run_step visual-check env OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 python scripts/verify_visual_cache.py \
  --cache-dir "$DERIVED_ROOT/cache/video-mme" --subset-config configs/video_mme_long_subset.json \
  --report "$RUN_REPORTS/visual-check.json"
echo "[finish] failed steps=$failures $(date -u +%FT%TZ)"
[[ "$failures" -eq 0 ]]
