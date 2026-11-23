#!/bin/bash
#SBATCH --job-name=pyro-long-events
#SBATCH --account=csci_ga_2271-2026fa
#SBATCH --partition=n2c48m24
#SBATCH --cpus-per-task=4
#SBATCH --mem=12G
#SBATCH --time=06:00:00
#SBATCH --requeue

# Submit from a committed detached checkout, after the active CPU allocation.
set -euo pipefail
cd "${SLURM_SUBMIT_DIR}"
source hpc/config.sh
REPORT_ROOT="${1:?absolute verification directory required}"
DERIVED_ROOT="/scratch/$USER/pyro-kd-derived"
RUN_REPORTS="$REPORT_ROOT/long-events-resume-${SLURM_JOB_ID}-attempt-${SLURM_RESTART_COUNT:-0}"
mkdir -p "$RUN_REPORTS"
git rev-parse HEAD > "$RUN_REPORTS/source_commit.txt"
git status --porcelain > "$RUN_REPORTS/source_status.txt"
[[ ! -s "$RUN_REPORTS/source_status.txt" ]] || { echo 'Source worktree is dirty' >&2; exit 2; }
cp "$REPORT_ROOT/audio-model-download.json" "$RUN_REPORTS/model-download.json"
python - "$RUN_REPORTS/model-download.json" <<'PIN_CHECK'
import json,sys
record=json.load(open(sys.argv[1]))
config=json.load(open('configs/audio_models.json'))
assert record['complete']
assert {(r['repo_id'],r['revision']) for r in record['models']} == {(r['repo_id'],r['revision']) for r in config['models']}, 'model report differs from frozen pins'
PIN_CHECK
mapfile -t VIDEO_IDS < <(python -c 'import json; print("\n".join(r["video_id"] for r in json.load(open("configs/video_mme_long_subset.json"))["videos"]))')
[[ ${#VIDEO_IDS[@]} -eq 12 ]]
VIDEO_ARGS=()
for ident in "${VIDEO_IDS[@]}"; do VIDEO_ARGS+=(--video-id "$ident"); done
run_step() {
  local name="$1"; shift
  local task_command
  printf -v task_command '%q ' "$@"
  echo "[step] $name $(date -u +%FT%TZ)"
  if in_env ro "$task_command" > "$RUN_REPORTS/$name.log" 2>&1; then
    echo "$name 0" >> "$RUN_REPORTS/status.txt"
    return 0
  else
    local code=$?
    echo "$name $code" >> "$RUN_REPORTS/status.txt"
    return "$code"
  fi
}
# Require the real full-video pilot to remain valid before expanding extraction.
run_step pilot-check env OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python scripts/verify_event_cache.py \
  --cache-dir "$DERIVED_ROOT/cache/video-mme" --subset-config configs/video_mme_long_subset.json \
  --video-id 604 --model-report "$RUN_REPORTS/model-download.json" --report "$RUN_REPORTS/pilot-check.json"
failures=0
run_step events env OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 \
  "/scratch/$USER/pyro-audio-env/bin/python" -u scripts/cache_event_features.py \
  --cache-dir "$DERIVED_ROOT/cache/video-mme" "${VIDEO_ARGS[@]}" \
  --model-report "$RUN_REPORTS/model-download.json" --batch-size 4 || failures=$((failures+1))
run_step events-check env OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python scripts/verify_event_cache.py \
  --cache-dir "$DERIVED_ROOT/cache/video-mme" --subset-config configs/video_mme_long_subset.json \
  --model-report "$RUN_REPORTS/model-download.json" --report "$RUN_REPORTS/events-check.json" || failures=$((failures+1))
echo "[finish] failed steps=$failures $(date -u +%FT%TZ)"
[[ "$failures" -eq 0 ]]
