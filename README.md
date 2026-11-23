# PYRO-KD: cue-driven keyframe selection for long-video audio description and QA

This project compares uniform (U), video-only (V), audio-only (A), fused audio+video (AV) and oracle (O) selection of the moments where a frozen MLLM writes inline audio descriptions, at a fixed budget K. Salience (what is worth describing: audio events, speech cues, visual novelty) is kept separate from feasibility (where a description can go: Silero VAD gaps and the word cap floor(Δt × 3)). The spec is the revised proposal; the repo's `CV Project Proposal.md` is the earlier version. Primary data is [MTG-AD](https://github.com/google/mtg-ad); REFRAMED, EgoSchema and a long-form QA set have adapters.

- Current status, blockers and next steps: [PROJECT_STATUS.md](PROJECT_STATUS.md)
- What was done and why (append-only): [DEVELOPMENT_LOG.md](DEVELOPMENT_LOG.md)
- Readiness report and deviations from the proposal: [docs/readiness_report.md](docs/readiness_report.md)
- File formats (caches, hand labels, canonical datasets, outputs): [docs/data_formats.md](docs/data_formats.md)
- Week 1 dataset decisions: [docs/week1_spec.md](docs/week1_spec.md)

## Local setup and tests

```bash
uv venv && source .venv/bin/activate          # Python 3.12
uv pip install numpy scipy librosa soundfile silero-vad torch pillow
python -m unittest discover -s tests
```

The suite was last run (482 tests, 8 skipped) in a venv with Python 3.12, torch, numpy, scipy, librosa, soundfile, silero-vad and Pillow, and without ffmpeg, PySceneDetect, open_clip, transformers, sentence-transformers or bert-score (those belong to the HPC environment).

`uv sync` / `uv run` fail on macOS because the lockfile pins CUDA torch wheels; on HPC use `hpc/setup_env.sh`. Several test modules need numpy (and librosa/Silero for a few), so run the suite in the project venv, not the system `python3`. Tests that need ffmpeg or PySceneDetect skip when those are missing.

## Pipeline

Every stage reads the previous stage's validated cache and skips work that is already current, so any step can be rerun or resumed.

| Step | Command | Needs |
|---|---|---|
| Frames (2 fps) + 16 kHz WAV | `scripts/prepare_mtg_ad.py --offline --data-dir $MTG_AD_ROOT --cache-dir $PYRO_CACHE/mtg-ad --preprocess --videos all` | ffmpeg |
| Silero VAD gaps | `scripts/cache_audio_features.py --cache-dir $PYRO_CACHE/mtg-ad` | CPU |
| Audio cues (VAD-gated Whisper, AST, CLAP, loudness/onset, speech cues) | `scripts/cache_audio_cues.py --cache-dir $PYRO_CACHE/mtg-ad --device cuda` | GPU |
| Shot cuts + OpenCLIP embeddings | `scripts/cache_visual_features.py --cache-dir $PYRO_CACHE/mtg-ad` | GPU |
| P1 exemplars (real AD lines from the exemplar pool) | `scripts/build_p1_exemplars.py --data-dir $MTG_AD_ROOT` | annotations |
| §6 pre-commitment checks | `scripts/precommit_checks.py ...` | data per check |
| Dev-only weight tuning | `scripts/tune_weights.py --config configs/experiments/mtg_ad_dev.toml` | caches |
| Experiments | `scripts/run_experiment.py --config <toml> --stage <stage>` | see below |

`--offline` reads a read-only local copy (e.g. the Globus transfer) and keeps the committed `configs/mtg_ad_splits.json`; it never writes into the dataset directory. The original download path (`prepare_mtg_ad.py --download-videos`, yt-dlp, cookies) is unchanged; see the notes at the end.

### Experiments (`scripts/run_experiment.py`)

A TOML config (`configs/experiments/mtg_ad_dev.toml`) defines the dataset, paths (environment variables, so no user or cluster names are committed), the selectors, priors and models, and the ablations. Run IDs hash every setting that affects a result.

```bash
CFG=configs/experiments/mtg_ad_dev.toml
python scripts/run_experiment.py --config $CFG --stage plan             # list the runs
python scripts/run_experiment.py --config $CFG --stage validate-data    # adapter checks on the dataset
python scripts/run_experiment.py --config $CFG --stage selector-only    # all selectors + ablations 1,2,4,6,8; no MLLM
python scripts/run_experiment.py --config $CFG --stage moments
python scripts/run_experiment.py --config $CFG --stage generate --dry-run --limit 4   # echo backend
python scripts/run_experiment.py --config $CFG --stage generate         # the 5 x 2 x 2 = 20 runs
python scripts/run_experiment.py --config $CFG --stage ablation5        # timeline off / on / audio-native
python scripts/run_experiment.py --config $CFG --stage evaluate-generation
python scripts/run_experiment.py --config $CFG --stage report           # tables/*.md|csv, plots/*.svg
```

Outputs go to `$PYRO_OUTPUTS/experiments/<name>/` (`selections/`, `moments/`, `generations/`, `eval/`, `tables/`, `plots/`, `status.json`). Selector-only stages never build a model backend. O is always labeled an oracle upper bound and is excluded from the Holm-corrected comparison family. Summary tables compare selectors on the videos every core selector could score; a second table lists each selector on all its videos.

AD generation resumes and evaluates results only when their request fingerprints match the prepared inputs. If the inputs change or the existing results predate fingerprints, keep the old outputs, rebuild moments as needed, and use a fresh experiment directory. If interrupted JSONL output ends with a partial record, the next append saves an exact backup before repairing the tail.

Selectors: `U`, `V` and `A` are the team's implementations, called unchanged. `AV` is the proposal's two-stage selector (`src/selection/two_stage.py`): candidates from VN/NSE/SC peaks, snapping to VAD gaps long enough for the word budget, then K picks with t_min = 2 s (and optional MMR) and running-memory novelty. `A_cue` is the proposal's A (speech cues + NSE, no visual input) and `V_mem` is V with running memory. `matrix.a_selector` chooses which A fills the matrix (team decision; default `A`). All AV/A_cue weights are untuned placeholders.

Models: `hf_vlm` (Qwen2.5-VL / Qwen3-VL via transformers), `openai` / `gemini` (keys from environment variables only, with configured per-token prices and a shared spend ledger), `qwen_omni` (audio-native ablation) and `echo` (dry runs and tests only). The ledger checks the estimated cost of each API attempt, including retries, before dispatch. Actual provider charges can exceed the estimate. Temperature is fixed at 0.

AD ablation 5 uses frames in all three conditions. It compares frames alone, frames with cached timeline text, and frames with a native waveform. Raw transcript context is excluded from these prompts. A missing timeline or waveform is recorded as an unavailable input for its condition.

QA: datasets with questions (EgoSchema, or a long-form set converted to the canonical JSONL) run `--stage qa`, which asks each model with the timeline off and on and reports accuracy, accuracy by video length, and paired McNemar tests.

### HPC

```bash
cp hpc/env.example.sh hpc/env.sh     # fill in account, profile, MTG_AD_ROOT, PYRO_CACHE, PYRO_OUTPUTS (git-ignored)
DRY_RUN=1 bash hpc/submit_stage.sh audio-cues      # print the sbatch command without submitting
bash hpc/submit_stage.sh tests                      # CPU unit tests on a compute node
```

`hpc/submit_stage.sh` maps each stage to CPU or GPU resources from the cluster profile (`hpc/profiles/cloud_bursting.sh`, or `torch.sh`, which needs its partitions filled in). Jobs requeue safely because every stage resumes. **Open question:** `hpc/README.md` records that coursework runs on Cloud Bursting, while the MTG-AD Globus copy is described as landing on Torch scratch; confirm which cluster to use. The original generation scripts (`hpc/submit.sh`, `scripts/run_ad_vlm.py`) still work and are documented in [hpc/README.md](hpc/README.md).

## Selector-only evaluation with the original script

```bash
python scripts/evaluate_selectors.py --set dev --timeline-svg outputs/dev_timelines.svg
python scripts/evaluate_selectors.py --set validation --selectors U O   # all 332 validation videos
```

Temporal P/R/F1 within ±2 s (one-to-one matching), paired bootstrap CIs over videos, an F1-by-tolerance sweep, and placement diagnostics: span, share of picks in the densest tenth of the video, share in the busiest gap, share of GT starts outside speech, and speech overlap / silence utilization of ADs rendered at 1.2× with the full word budget. Some MTG-AD videos list AD starts after their declared duration; by default those starts are excluded from the ground truth for every selector and reported (`--keep-out-of-bounds` keeps them and labels the result an unrestricted upper bound). Do not tune on `--set test`.

## Layout

- `src/datasets/`: MTG-AD, REFRAMED, EgoSchema loaders; `adapters.py` (one interface, validators, canonical JSONL); `precommit.py` (§6 checks).
- `src/utils/`: media cache, K rule and U/O, Silero VAD gaps and A, visual features and V (team code).
- `src/audio/`: AudioSet superclasses, NSE salience, speech cues, VAD-gated ASR, AST/CLAP backends, prompt timeline, `audio_cues.json`.
- `src/selection/`: cue tracks on one grid, the two-stage selector, hierarchical budgets, the selector registry.
- `src/generation/`: prompts (P0, P1, QA, word cap), P1 exemplar handling, backends, the HF VLM loader, resumable output files.
- `src/evaluation/`: temporal metrics, placement metrics, audio diagnostics, significance tests, QA, BERTScore, cosine, OpenCLIP text, judge rubrics.
- `src/experiments/`: config, run matrix and ablations, stages, tuning, reporting.
- `scripts/`: entry points listed above, plus the original `build_moments.py`, `run_ad_vlm.py`, `evaluate_ad.py`.
- `hpc/`: environment setup, profiles, Slurm jobs.

## Video download notes (original MTG-AD path)

MTG-AD preparation writes `data/mtg-ad/preprocess_report.json`, including download/preprocessing failures and whether each frame/audio cache was written or reused. A cache is rebuilt if its source changes or an artifact is missing or empty. Where YouTube blocks downloads, place MP4s in `data/mtg-ad/videos/<video_id>.mp4` and run `--preprocess` (or `SKIP_DOWNLOAD=1 bash hpc/prepare_data.sh`). YouTube may require yt-dlp's [PO-token provider setup](https://github.com/yt-dlp/yt-dlp/wiki/PO-Token-Guide). Cookie files and browser profiles are credentials: never copy them to the cluster or commit them. Never commit `data/`.
