# Readiness report (2026-11-19, branch `rishabh/experiment-readiness`)

## What existed (main at 3139be9)

Media cache (frames at 2 fps, 16 kHz WAV, manifests with fingerprints); Silero VAD gaps and the word cap; selectors U, O, A (gap-quality) and V (CLIP novelty + shots); MTG-AD/REFRAMED/EgoSchema loaders and fixed splits; P0 prompt; HF VLM generation with resume and cost metadata; temporal P/R/F1, bootstrap CIs, BERTScore, sentence cosine, a 4-criterion MTG-AD judge; selector-only timing evaluation; Cloud Bursting setup and generation scripts. 394 tests passing.

## What was added or fixed

| Area | Added | Verification |
|---|---|---|
| Audio stack (§2-§3) | NSE (class onsets, CLAP change, loudness jumps, onset strength), speech cues (deictic, semantic shift, speaker change from supplied turns, long silence), VAD-gated Whisper plumbing with word timestamps, AST/CLAP windowed backends, hashed `audio_cues.json`, prompt timeline | unit-tested with injected models; real Whisper/AST/CLAP never run |
| Selection (§5.1, §5.5) | two-stage AV (candidates, gap snapping with lag/lead, GF, OP, signed per-class NSE, AV disagreement, t_min/MMR, running-memory novelty), A_cue, V_mem, hierarchical chunking and density budgets with a floor, one registry over all selectors | unit + integration tests on synthetic cues |
| Generation (§5.2, §5.4) | P1 (guidelines + 3 real exemplars from the exemplar pool, leakage check), timeline in AD and QA prompts, strict word cap, OpenAI/Gemini backends with env keys and a persistent hard spend cap, Qwen2.5-Omni backend, echo backend for dry runs | prompt/cap/ledger logic tested; no real model or API call made |
| Experiments (§5.2-§5.3) | TOML configs with env-var paths, run IDs, 5 x 2 x 2 matrix, all 8 ablations, resumable stages with failure isolation, dev-only tuning with cross-validation, tables and SVG plots | full pipeline integration test (selection -> moments -> 20 generation runs -> evaluation -> report, QA) |
| Evaluation (§7) | speech overlap at 1.2x, silence utilization, placement diagnostics (folded into `evaluate_selectors.py`), detector P/R on hand labels, trigger/suppress lift, silent-visual-event coverage, Wilcoxon (exact and normal; matches scipy), Holm, exact McNemar, QA accuracy/by length/paired timeline comparison, 5-criterion judge, OpenCLIP text similarity, common-subset summaries | unit-tested; Wilcoxon and McNemar cross-checked against scipy |
| Datasets (§1, §6) | adapter interface + validators for MTG-AD (HF or Globus layout), REFRAMED, EgoSchema, canonical JSONL; `prepare_mtg_ad.py --offline` for a read-only copy; §6 checks as pass/fail/blocked | validators tested on synthetic layouts; the real Globus copy has not been seen |
| HPC | env template (git-ignored), Cloud Bursting and Torch profiles, generic requeue-safe stage job, per-stage resources, sharded cue extraction | `bash -n` and `DRY_RUN=1`; nothing submitted |

Bugs found and fixed while building: robust (99th-percentile) normalization saturated sparse novelty tracks; NSE/SC weighted means diluted one event by the number of cached streams; pure memory novelty let a static undescribed frame outrank a shot cut (now a configurable blend); the API spend cap could be overshot when the pre-call estimate was low (now scaled by the observed actual/estimate ratio, residual bound documented); summary rows compared selectors on different video sets (now also summarized on the common subset); tuning folds with no scorable video reported F1 = -1.

## Test status

`python -m unittest discover -s tests` in the project venv: **482 tests, OK, 8 skipped** (7 need ffmpeg, 1 needs PySceneDetect). New modules: `test_selection_core` (29), `test_audio_cues` (23), `test_evaluation_extra` (18), `test_experiment_integration` (15), plus 3 diagnostics tests in `test_selector_eval`. Under the system `python3` without numpy, 7 test modules fail to import (4 of them already did before this work).

**Not tested anywhere:** real Silero on speech (only on a pure tone, where it correctly finds none), Whisper, AST, CLAP, OpenCLIP, PySceneDetect, ffmpeg extraction, Qwen-VL, the API backends against real endpoints, Qwen2.5-Omni, BERTScore/sentence-transformers scoring, any Slurm submission, any real dataset.

## Blocked (external)

1. **Data:** MTG-AD Globus transfer to scratch (not visible from this laptop). REFRAMED and EgoSchema videos not downloaded. Cue2Narrate access and the long-form QA benchmark are undecided (§6).
2. **Cluster choice:** `hpc/README.md` says Cloud Bursting only; the data is described as landing on Torch scratch. Torch partitions are deliberately not filled in.
3. **Models:** Whisper large-v3, AST, CLAP, OpenCLIP ViT-B-32, Qwen2.5-VL-7B, Qwen2.5-Omni-7B must be downloaded on HPC (`hpc/download_models.sh <repo>`).
4. **API:** key, current per-token prices and a spend cap must be set by the team.
5. **Human inputs:** hand-labeled audio events (~10 clips, §6), the peer pairwise study (not implemented), a decision on which A fills the matrix.

## Deviations from the proposal and their research implications

1. **Primary dataset is MTG-AD, not REFRAMED/Cue2Narrate** (team Week 1 decision; data is on hand). RQ1/RQ2 claims then concern user-generated video; REFRAMED stays the movie check, and RQ6 baselines depend on Cue2Narrate access.
2. **Two A selectors.** The matrix uses the team's gap-quality A unless `matrix.a_selector = "A_cue"`. The team's A is not the proposal's A (no speech cues or NSE), so H1 comparisons with it test "audio feasibility" more than "audio salience". Its known clustering and speech-free start-packing are reproduced in tests; A_cue avoids both on synthetic data.
3. **AST instead of BEATs/PANNs** (the proposal allows any of the three); chosen because it ships as a transformers checkpoint.
4. **Memory novelty is a blend by default** (`memory_blend = 0.5`); the proposal's pure definition is `memory_blend = 1.0` (ablation 8 variant `AV-memory-pure`).
5. **Gap snapping parameters** (lag 3 s, lead 0.5 s, minimum 3 words) and **grid fill candidates** (so every selector can meet K) are design choices, recorded per pick (`source`, `lag`).
6. **Overlap uses Silero speech only** and a modeled TTS rate (2.5 words/s x 1.2); the proposal also cross-checks with Whisper timestamps and MTG-AD renders real TTS. Overlap numbers are therefore approximate.
7. **Speaker change** cues need diarization turns supplied from outside; no pyannote backend is included (the proposal marks diarization optional).
8. **OpenCLIP similarity** is text-to-text with the CLIP text tower; **the judge** is text-only and cannot verify visual grounding (stated in its output).
9. **QA keyframes:** one frame per selected moment, at most 32, evenly subsampled by order; the 4-frame window applies to AD only.
10. **Not implemented:** the blinded peer pairwise study, a Cue2Narrate adapter (schema unknown; use the canonical JSONL), pyannote diarization, real-TTS rendering.

## Commands once data is ready (Cloud Bursting example)

```bash
cp hpc/env.example.sh hpc/env.sh && $EDITOR hpc/env.sh     # ACCOUNT, PYRO_PROFILE, MTG_AD_ROOT, PYRO_CACHE, PYRO_OUTPUTS, HF_HOME
bash hpc/setup_env.sh                                        # once, CPU node
bash hpc/download_models.sh Qwen/Qwen2.5-VL-7B-Instruct MIT/ast-finetuned-audioset-10-10-0.4593 \
     laion/clap-htsat-unfused laion/CLIP-ViT-B-32-laion2B-s34B-b79K Systran/faster-whisper-large-v3 \
     sentence-transformers/all-MiniLM-L6-v2 sentence-transformers/all-mpnet-base-v2
python scripts/run_experiment.py --config configs/experiments/mtg_ad_dev.toml --stage validate-data
bash hpc/submit_stage.sh tests
bash hpc/submit_stage.sh preprocess --videos dev
bash hpc/submit_stage.sh audio-features
bash hpc/submit_stage.sh visual-features
SHARDS=2 bash hpc/submit_stage.sh audio-cues
bash hpc/submit_stage.sh selector-only                      # first results: tables/selection_summary.md
python scripts/tune_weights.py --config configs/experiments/mtg_ad_dev.toml --base AV
python scripts/build_p1_exemplars.py --data-dir "$MTG_AD_ROOT"
bash hpc/submit_stage.sh moments
python scripts/run_experiment.py --config configs/experiments/mtg_ad_dev.toml --stage generate --dry-run --limit 4
bash hpc/submit_stage.sh generate --runs "ad-*-qwen-*" --limit 4   # GPU smoke test, then without --limit
bash hpc/submit_stage.sh evaluate-generation
python scripts/run_experiment.py --config configs/experiments/mtg_ad_dev.toml --stage report
```

## Recommended first HPC experiment

The selector-only run on the 10 MTG-AD dev videos: preprocess, VAD features, visual features and audio cues, then `--stage selector-only`. It costs no MLLM time, exercises every real model in the feature stack on real audio and video for the first time (Silero on real speech, Whisper, AST, CLAP, OpenCLIP, PySceneDetect), and produces RQ1(a)/RQ2 placement numbers, the selector diagnostics behind the two open A findings, and the inputs for dev-set weight tuning. Then repeat U/O/A/V/AV selector-only on all 332 validation videos before spending GPU hours on generation.
