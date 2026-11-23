# Development log

Append-only, chronological. Each entry: what was worked on, findings, files,
tests and outcomes, decisions with rationale, blockers and deviations. The
current snapshot lives in `PROJECT_STATUS.md`.

---

## 2026-11-19 — Session 1: audit and experiment-readiness branch

Branch `rishabh/experiment-readiness`, cut from `origin/main` at 3139be9.

### Git state found (differs from the hand-over notes)

- #16 (audio feature caching, `rishabh/audio-validation`) **merged** into main
  as 1cef9c3 today; #19 was closed and replayed as #21 (V selector), merged as
  3139be9. So this branch builds on main directly, not on the #16 branch.
- Local unpushed `rishabh/selector-validation` (cb7ce5c) holds
  `scripts/validate_selectors.py`; its diagnostics are folded into
  `scripts/evaluate_selectors.py` here (see below), and the branch is left
  untouched.
- Baseline test run on 3139be9 with the project `.venv` (torch, numpy,
  librosa, silero-vad, scipy, soundfile; no open_clip, transformers,
  sentence-transformers, bert-score, matplotlib, cv2):
  `394 tests, OK (skipped=8)`.

### Spec

The spec is the revised proposal (`~/Downloads/CV Project Proposal.md`). The
repo's tracked `CV Project Proposal.md` is the earlier version (MTG-AD primary,
3–5 minute videos, no three-stream audio, no trigger/suppress, no long-video
section). Differences that matter are recorded under Deviations.

### Audit: proposal component vs. repository (at 3139be9)

| Component (proposal §) | State | Where / notes |
|---|---|---|
| Media decode, timestamped frames, 16 kHz WAV | complete, tested | `src/utils/media.py` (frame i at i/fps, manifest, atomic rebuild) |
| Silero VAD → gaps, GF word cap floor(Δt×3) | complete, tested | `src/utils/audio_features.py`, `prompts.word_limit_for_gap` (cap 30) |
| Whisper ASR, VAD-gated, word timestamps (§2) | **missing** | MTG-AD ships ASR segments; no Whisper code |
| AudioSet tagger + CLAP (§2) | **missing** | |
| NSE: class onsets, CLAP change, loudness, onset strength (§3.1) | **missing** (only per-gap mean RMS/onset) | |
| Speech cues: speaker change, semantic shift, deictics, long silence (§3.2) | **missing** | |
| Visual: PySceneDetect + OpenCLIP novelty (§3.3) | complete (local-window novelty), tested with stand-in embeddings | `src/utils/visual_features.py`, `scripts/cache_visual_features.py` |
| Running-memory novelty (§3.3, §5.5, ablation 8) | **missing** | V uses previous-frame novelty only |
| OP overlap penalty (§3.4) | **missing** | |
| Two-stage selector: candidates, gap snapping, MMR (§5.1) | **missing** | |
| U, O selectors | complete, tested | `src/utils/selection.py` |
| A selector | complete, tested; design differs from §5.2 (gap-quality score, not SC+NSE). Open findings: clustering in the longest gap; speech-free videos → 0, 2, 4 … s | `src/utils/audio_selector.py` — **do not change without team sign-off** |
| V selector | complete, tested | `VisualCueSelector` |
| AV selector, signed/per-class b, AV disagreement (§5.1, RQ2) | **missing** | |
| Hierarchical chunking, density budget with floor (§5.5, ablation 7) | **missing** | |
| K rule, t_min 2 s, 4 frames / 2 s, temperature 0 (§5.6) | complete | K rate 3.48/min from MTG-AD validation |
| P0 prompt | complete | `src/generation/prompts.py` |
| P1 prompt (guidelines + 3 exemplars) | **missing** | |
| Audio timeline in prompt (§5.4) | **missing** | |
| Open VLM backend (Qwen-VL class) | complete, smoke-tested on HPC | `src/generation/hf_vlm.py`, `scripts/run_ad_vlm.py` |
| API backend + hard spend cap | **missing** | |
| Audio-native model (Qwen2.5-Omni / Qwen2-Audio) | **missing** | |
| QA generation (multiple choice) | **missing** (EgoSchema loader only) | `src/datasets/egoschema.py` |
| 20-run matrix, 8 ablations, run IDs, resume | **missing** (only per-step scripts) | |
| Temporal P/R/F1, bootstrap CIs | complete, tested | `src/evaluation/metrics.py`, `stats.py` |
| Selector-only timing eval | complete for U/O/A/V | `scripts/evaluate_selectors.py` |
| Speech overlap at 1.2×, silence utilisation | **missing** (only in-speech rate) | |
| Placement diagnostics (densest-decile, busiest gap, span, GT-in-gap) | on unpushed branch only | cb7ce5c |
| Detector P/R vs hand labels, trigger/suppress analysis, silent-visual-event coverage (§7B) | **missing** | |
| BERTScore, sentence cosine | complete, tested | `src/evaluation/bertscore.py`, `cosine_similarity.py` |
| OpenCLIP text similarity (§7C) | **missing** | |
| LLM judge | partial: 4 MTG-AD/VideoA11y criteria, not the proposal's 5 | `src/evaluation/rubric.py` |
| QA accuracy, accuracy vs length, timeline on/off | **missing** | |
| Wilcoxon signed-rank | **missing** | |
| MTG-AD adapter | complete for the HF layout (`data/annotations/*.jsonl` + `data/mtg-ad/videos`) | Globus layout (`/scratch/<NETID>/mtg-ad-transfer/mtg-ad/`) not supported |
| REFRAMED adapter | complete (metadata, decryption, SRT), video download unverified | |
| Cue2Narrate, long-form QA adapters | **missing**; schemas not established | |
| §6 pre-commitment checks | partially documented (`docs/week1_spec.md`), not encoded | |
| Slurm scripts | generation only, Cloud Bursting partitions hard-coded in presets | `hpc/` |
| Config system | **missing** (argparse defaults + `configs/*_splits.json`) | |

Duplicates / inconsistencies found:

1. `src/utils/visual_selector.py` (OpenCLIP wrapper with a "TODO more
   functions") is used only as the embedder inside `visual_features.clip_embedder`;
   its own `compute_visual_novelty` duplicates `visual_features.novelty_curve`.
   Left in place (still imported); not extended.
2. `src/main.py` and `src/networks/` are empty.
3. README says `uv run python -m pytest`; on macOS the lockfile's CUDA wheels
   fail, and the suite is unittest-based.
4. `hpc/README.md` says coursework runs on Cloud Bursting, not Torch; the
   hand-over mentions Torch HPC with `/scratch/<NETID>/mtg-ad-transfer/mtg-ad/`.
   **Unresolved — needs the team/instructor.** Cluster settings are made
   configurable instead of picking one.
5. `DEFAULT_RATE_PER_MINUTE = 3.5` in `selection.py` vs 3.48 in
   `configs/mtg_ad_splits.json`; every script reads the config, so the constant
   is only a fallback. Left as is.

### Decisions

- **A stays as is.** The proposal's A ("speech cues plus non-speech event
  salience, snapped to VAD gaps") is implemented as a *new* selector `A_cue`
  using the same two-stage machinery as AV with visual weights forced to 0.
  The existing gap-quality `A` keeps its name and defaults. Which one fills
  the "A" slot of the 20-run matrix is a team decision (config key
  `matrix.a_selector`, default `A`). This conflict is recorded here per the
  ground rules.
- **V stays as is** (default local novelty). Running-memory novelty is an
  option of the new two-stage selector and of `V_mem`.

### Implementation (commits a3a206a, 5ae6319, 0ef0977 and the docs commit)

- **Audio cues** (`src/audio/`): taxonomy (hand-set superclasses, speech/silence excluded), NSE salience, speech cues, VAD-gated ASR with word timestamps (Whisper ungated only for ablation 3's tag baseline), AST/CLAP windowed backends at 1 s hop, prompt timeline, `audio_cues.json` stamped with the WAV SHA-256 and the VAD hash.
- **Selection** (`src/selection/`): cue bundle on a 0.5 s grid, two-stage selector, hierarchical budgets, registry. Team A and V are called unchanged.
- **Generation**: P1 (guidelines are our paraphrase; exemplars only from real exemplar-pool lines), timeline in AD/QA prompts (P0 text byte-identical without a timeline), word-cap enforcement, API backends with a spend ledger, Qwen2.5-Omni, echo.
- **Datasets**: adapter interface + validators; `prepare_mtg_ad.py --offline/--video-dir`; §6 checks.
- **Experiments**: TOML config, run IDs, matrix and ablations, resumable stages, tuning, report.
- **Evaluation**: placement metrics and diagnostics (also folded into `scripts/evaluate_selectors.py`, replacing the separate `validate_selectors.py` on the unpushed branch), audio diagnostics, Wilcoxon/Holm/McNemar, QA, 5-criterion judge, CLIP text.
- **HPC**: `hpc/env.example.sh`, profiles, `hpc/slurm/stage.sbatch`, `hpc/submit_stage.sh`; `hpc/config.sh` sources `hpc/env.sh` and lets `CPU_PARTITION` be overridden; `setup_env.sh` installs faster-whisper.

### Bugs found and fixed during this session

1. 99th-percentile normalization saturated sparse tracks (one novelty peak made every frame 1.0) -> max normalization.
2. NSE/SC as weighted means diluted a single event by the number of cached streams -> signed sum scaled by the largest weight, clipped to [-1, 1].
3. Pure memory novelty replaced the local change signal, so an undescribed static frame outranked a shot cut -> `memory_blend` (default 0.5; 1.0 kept as ablation variant `AV-memory-pure`).
4. The API spend cap could be exceeded when the pre-call estimate was low -> estimates scaled by the worst observed actual/estimate ratio; residual overshoot bounded by one call and documented.
5. Selector summary rows were computed on different video sets (A/AV skip videos without audio) -> added `summary_common` on the videos every core selector scored, used for the main table and plot.
6. Tuning folds with no scorable video reported F1 = -1 -> None, excluded from the CV mean.
7. Selector-only scoring of QA-only datasets would score against an empty ground truth -> skipped as `no_ad_annotations`.

### Evidence on the open A findings (synthetic only)

`tests/test_selector_eval.py::PlacementDiagnosticsTests` reproduces both: A puts more picks in its busiest gap and densest tenth than U on a talky clip, and on a speech-free clip all A picks fall in the first quarter (span < 0.25). On synthetic data the two-stage `A_cue` spreads picks over > 60 % of a speech-free clip (`test_selection_core`). Real dev-set numbers are still needed before proposing any change to `audio_selector.py`.

### Tests

Project venv: 482 tests, OK, 8 skipped (ffmpeg x7, PySceneDetect x1); baseline was 394. Nothing GPU-, model-, API- or dataset-dependent was run.

### Blockers and deviations

See `docs/readiness_report.md` (deviations 1-10) and `PROJECT_STATUS.md`. Open question for the team: Cloud Bursting vs Torch.


## 2026-11-23 — Revision 1.5 execution record

The ongoing English Agent Log and experiment status now live in
`docs/AGENT_LOG.md` and `docs/EXPERIMENT_STATUS.md`. They record real smoke/pilot
verification, frozen formal runs, failures, fixes, source commits and HPC jobs.
Current research results are under `docs/results/`; reproduction commands and
artifact locations are in `docs/RESULTS_REPRODUCTION.md`. The historical entries
above remain unchanged. Paid API and human-label work is explicitly deferred.
