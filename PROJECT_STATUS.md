# Project status

Snapshot as of 2026-11-19, branch `rishabh/experiment-readiness` (from main 3139be9; not pushed, no PR). Verification levels: **implemented** (code exists), **tested** (unit/integration tests with synthetic data or injected models), **verified** (run on real data/models). Nothing in this branch is *verified* yet: no real dataset, model, API or cluster job has been run from it.

## Completed

| Component | Level | Notes |
|---|---|---|
| Media cache, Silero VAD gaps, word cap, U/O/A/V selectors | verified on HPC smoke test (team, before this branch) / tested | team code, unchanged except `prepare_mtg_ad.py --offline` |
| Audio cue stack: NSE, speech cues, VAD-gated ASR plumbing, AST/CLAP backends, `audio_cues.json`, timeline | tested | real Whisper/AST/CLAP not run |
| Two-stage AV, A_cue, V_mem, gap snapping, GF/OP, signed per-class NSE, AV disagreement, MMR, memory novelty | tested | weights untuned |
| Hierarchical chunking + density budget with floor | tested | synthetic 30-minute video |
| P1 prior with real-exemplar builder and leakage check | tested | exemplars not built yet (needs annotations) |
| Backends: HF VLM, OpenAI, Gemini (spend cap), Qwen2.5-Omni, echo | spend cap and request plumbing tested with a fake transport; models not run | API prices unset |
| Experiment runner: config, run IDs, 20-run matrix, 8 ablations, resume, failure isolation, report | tested end to end on synthetic caches | |
| Evaluation: overlap at 1.2x, silence utilization, diagnostics, audio diagnostics, Wilcoxon/Holm/McNemar, QA, 5-criterion judge, CLIP text | tested (Wilcoxon/McNemar match scipy) | BERTScore/cosine/CLIP scorers not run here |
| Dataset adapters + validators, §6 checks | tested on synthetic layouts | real Globus layout unseen |
| Slurm: env template, profiles, stage job, submit wrapper | syntax + dry-run only | nothing submitted |
| Docs: README, data formats, readiness report | done | |

## In progress

Nothing is mid-edit. All work is committed on `rishabh/experiment-readiness`.

## Remaining (prioritized)

1. Run the first real selector-only experiment on MTG-AD dev (see Next actions); fix whatever the real feature extractors break.
2. Team decisions: which A fills the matrix (`matrix.a_selector`); Cloud Bursting vs Torch; API model, prices and cap.
3. Tune AV/A_cue weights on dev (`scripts/tune_weights.py`), then fix them in the config.
4. Build P1 exemplars; GPU smoke test of generation through `run_experiment.py`; then the 20-run matrix.
5. §6 checks with real data: EgoSchema audio, Cue2Narrate access, long-form QA choice, hand-labeled audio events.
6. Long-form QA converter to the canonical JSONL; ablation 7 on long videos.
7. Not implemented: blinded peer pairwise study, pyannote diarization backend, Whisper cross-check in the overlap metric, real-TTS rendering.

## Blockers

- MTG-AD videos not accessible from this laptop (Globus transfer to scratch).
- Cluster ambiguity: Cloud Bursting (course rule in `hpc/README.md`) vs Torch (data location).
- Model weights to download on HPC; API key/prices/cap; hand labels; Cue2Narrate access; long-form QA benchmark choice.

## Testing status

Project venv: `python -m unittest discover -s tests` -> 482 tests, OK, 8 skipped (ffmpeg x7, PySceneDetect x1). System `python3` without numpy cannot import 7 test modules (4 pre-existing).

## Next actions

1. Fill `hpc/env.sh`; `bash hpc/submit_stage.sh tests`.
2. `preprocess --videos dev` -> `audio-features` -> `visual-features` -> `audio-cues` -> `selector-only` (commands in `docs/readiness_report.md`).
3. Read `tables/selection_summary.md`, the timelines SVG and `streams_missing` diagnostics; log findings here and in `DEVELOPMENT_LOG.md`.
4. Bring the A-selector evidence (clustering, speech-free start packing, A vs A_cue on real dev videos) to Kiran and Prajanya before any change to `audio_selector.py`.
