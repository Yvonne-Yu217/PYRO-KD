# Experiment status

## Current checkpoint — 2026-11-23 13:46 UTC

All source-available MTG, REFRAMED and EgoSchema transfers are complete.
The 12-video long Video-MME subset has complete verified media, VAD, visual,
speech, event and onset caches. Continuations40363/40408 completed successfully;
final speech60,588words/4,724windows and events27,780windows. These are cache
integrity checks, not transcription/detector accuracy.

User GPU session40880 onb-19-15 provides oneA10040GB through2026-11-24T13:26:29UTC.
The prior hold condition is satisfied by completed transfers and this newGPU
session. First four-moment MTG U/P0 smoke ran onGPU. Initial2/4wordbudgetfailures;
shared prompt clarification plus explicit failure accounting yielded3/4valid,
1remainingbudgetfailure. No token-limit failures. Rawattempts preserved. This
is onevideo only and does not pass the full smoke/evaluationgate. No grid scaling.

Next: bounded consistently applied word-budget repair, rerun/inspectsmallsmoke,
then ten-videoREFRAMEDdevelopment smoke and real semantic/renderedoverlapmetrics.
Ten-video input is prepared, not yet inferred. Full20matrix/APIcap, finalselector
weights/equal-Kpolicy, humanlabels/60peercomparisons, finalQA/ADstatistics and
report/slides remain incomplete. LongQA36questionmembership is frozen; question
content remains unopened. Cue2Narrate implementation/checkpoints still unavailable.

Earlier checkpoints and chronological updates below are retained for provenance.

## Transfer checkpoint — 2026-11-23 00:07 UTC

**Torch-to-Cloud-Bursting data transfer and verification complete.**
CPU allocation 40036 on b-10-63 was used; scheduled end remains
2026-11-23 04:22:47 UTC. GPU jobs remain on hold pending the user's GPU session.

- Authoritative proposal: Revision 1.5, tab `t.ickd261ntenp`; requirements in
  `docs/REVISION_1_5_REQUIREMENTS.md`.
- Source baseline: `3139be9be2ef52f498b5db5fecfdfa5618da03b0`.
- MTG: 413/413 source-available videos copied and SHA256 verified; 20 source failures remain.
- EgoSchema: 5031/5031 copied and manifest/ffprobe verified; all lack audio.
- REFRAMED: 1983/1983 copied and manifest/ffprobe verified; 40 source failures
  remain (all in train). No missing, unexpected, size/probe or snapshot errors.
- Transfer session 49851 and verification watcher 73417 both exited 0.
  REFRAMED transfer completed 2026-11-23 00:04:54 UTC.
- Raw destination: `/scratch/yy5919/mtg-ad-transfer`; loader views:
  `/scratch/yy5919/pyro-kd-derived/datasets`; raw Torch sources were unchanged.
- Evidence: `outputs/verification/mtg-transfer-final.json`, `mtg-sha256.log`,
  `egoschema-transfer-final.json`, `reframed-transfer-final.json`;
  transfer log `outputs/transfers/pull_torch_datasets.log`.
- Evaluation-media checks cover manifest IDs, sizes, stream types and durations;
  they do not claim full video decode or cryptographic hashes for EgoSchema/REFRAMED.
- Real MTG dev preprocessing, four-input dry run and offline Qwen2.5-VL-7B
  processor encoding passed. Exact later GPU command: `docs/REAL_DATA_SMOKE.md`.
- No real GPU inference or scientific comparison results yet. Full Revision 1.5
  implementation, GPU smoke, experiments and final reporting remain outstanding.

## Historical updates (chronological; superseded statements retained for provenance)

Source baseline: 3139be9be2ef52f498b5db5fecfdfa5618da03b0 (fetched origin/main 2026-11-22).
Proposal source: Google Doc 1dYGXJJ-MJxmJtZV02U4IrkQVXLPTdev9soelen7Ltyw, tab t.1uocwcl7xqhl.

Environment: slurm-cloud, b-19-7. Existing user Jupyter job 40006; not agent-submitted. A100 40GB idle at audit. Requested /scratch/yy5919/mtg-ad-transfer absent here. Torch access under investigation.

Tests: `python -m pytest -q` failed (default Python lacks pytest). Retry: `source hpc/config.sh && in_env ro 'python -m pytest -q'`, process session 46949.
No experiment jobs submitted. No scientific outputs yet.

Next: establish Torch access and inspect dataset manifests, finish tests, reconcile proposal against existing implementations and split sizes, run real dev smoke, then full supported matrix and statistics.
Blocked prerequisites: API spending cap; matching EgoSchema audio; private labels beyond 500; peer ratings require human contributions. Known dataset counts from user (not independently verified): REFRAMED 1983 available/40 HTTP403; EgoSchema 5031 silent videos/500 public labels.
Scope remains all five selectors × two prompts × two frozen models, REFRAMED timing/quality, EgoSchema QA, calibrated rubric, 60 peer comparisons, Wilcoxon/95% bootstrap, raw outputs, plots and reproduction commands. Existing repository documents differ from linked proposal (MTG primary vs REFRAMED primary, 30 vs 50 REFRAMED test clips); resolve explicitly without tuning on held-out outcomes.

Update: existing-container pytest retry also failed (pytest absent). Isolated test venv created at /scratch/yy5919/pyro-test-env. Baseline verification completed: **394 passed, 5911 subtests passed in 22.93s**. Command: `source hpc/config.sh && in_env ro '/scratch/yy5919/pyro-test-env/bin/python -m pytest -q'`. Output: `outputs/verification/pytest-baseline.log`. Process 21475 exited 0. These tests do not prove real model/data integration.
Torch hostname corrected to login.torch.hpc.nyu.edu using official NYU docs; SSH reports changed host key. Authenticated access requested; existing trust store preserved. Implementation gaps and full remaining requirements: `docs/PROPOSAL_1_5_AUDIT.md`. No experiment jobs submitted.

Paired-analysis implementation added: `scripts/analyze_paired.py --input METRICS.jsonl --output ANALYSIS.json --a CONDITION_A --b CONDITION_B --metric METRIC --unit cluster` (REFRAMED movie IDs) or `--unit example` (independent videos). It reports pairing exclusions, equal-weight cluster means, paired bootstrap CI and permutation Wilcoxon (SciPy required), input hash and command. P-values are explicitly unadjusted; multiplicity remains to implement before confirmatory reporting. Tests: 4 passed in 3.36s; no scientific results generated.

SSH update: following NYU's documented options reaches Microsoft device authentication. Live PTY session 70487 is waiting for user MFA; poll that handle, then send ENTER only after completion is confirmed. Do not create duplicate sessions while it is live. Torch datasets remain uninspected.

Comparison families: `python scripts/analyze_family.py --input METRICS.jsonl --family FAMILY.json --output ANALYSIS.json --unit cluster`. FAMILY.json has a `comparisons` array with `a`, `b`, `metric` fields. Freeze the full family before test scoring. Supports Holm p-value correction; CIs are explicitly pointwise. Targeted statistics verification: 41 passed, 11 subtests passed, 3.94s; log outputs/verification/pytest-statistics.log. Actual experiment analysis remains pending.

IMPORTANT SCOPE CORRECTION: authoritative Proposal Revision 1.5 is tab t.ickd261ntenp, not URL-selected t.1uocwcl7xqhl. Full source now docs/PROPOSAL_REVISION_1_5.txt; current complete inventory docs/REVISION_1_5_REQUIREMENTS.md. Earlier audit is historical and incomplete. Added scope includes long-form QA, Cue2Narrate gate, three-stream audio, eight ablations and human detector labels. No experimental scope cuts applied.
SSH 70487 closed (255) after user confirmed MFA and ENTER was sent. Replacement 62852 is live at fresh Microsoft device-auth prompt. Continue that session after user completes new authentication; do not retry the old handle.
Current priority (user): obtain data first, locate existing real MTG smoke before repeating it. 62852 terminated (255); latest SSH session 10280 awaiting device auth. Local /scratch/yy5919/pyro-kd-data/mtg-ad/{videos,incoming} empty. outputs/evaluation-smoke/manual_paraphrases.validation.evaluation.json identifies model manual-smoke/manual-paraphrases-v1; it is not proof of GPU VLM inference. Gap-snapping helper/tests added (10 passed), integration deferred until data access.

User explicitly switched to CPU-only allocation for data downloads; hold GPU submissions until data ready and user starts GPU session. Old allocation 40006 CANCELLED at 22:21 UTC, which destroyed SSH master. Current allocation 40036 on b-10-63 (CPU, six-hour limit). New SSH 45555 awaits Microsoft MFA; control path /home/yy5919/.ssh/pyro-torch-cpu-control.
Data discovery correction: rg must use --no-ignore for Git-ignored data directories. Existing /scratch/yy5919/PYRO-KD/data has REFRAMED metadata and 51 EgoSchema videos. All 51 ffprobe calls succeeded, none has audio. Evidence outputs/verification/local-egoschema-inventory.json; probe script alongside. No full decode claimed. Existing MTG dev report says 0/10 ready (HTTP403); later real GPU smoke remains unlocated. Do not redownload available data; first inspect Torch source and transfer missing files.

SSH authenticated and commands verified on torch-login-a-4; CPU master socket /home/yy5919/.ssh/pyro-torch-cpu-control, interactive session 45555. Source directories confirmed. Source records: MTG 413/433 videos, ~21GiB; EgoSchema 5031, ~99GiB; REFRAMED 1983, ~187GiB with 40 failures. Preserve source duration exceptions and missing references.
Transfer destination /scratch/yy5919/mtg-ad-transfer (Cloud Bursting). Seeded 51 local EgoSchema files via rsync -rt. Metadata transfer session 69403/PID 9347 is active. Full resumable transfer session 49851 waits for that PID, then runs outputs/transfers/pull_torch_datasets.sh, logging outputs/transfers/pull_torch_datasets.log. Script uses exclusive flock, serial MTG/EgoSchema/REFRAMED transfer, 100000 KiB/s limit, partial directory and 300s I/O timeout; no source writes/deletions. Confirm live process/session rather than blindly relaunching. Need verify coverage/size/hash and streams after completion. Current remote repo search found no prior real GPU smoke artifacts at depth <=4; broader provenance search still needed.
Metadata transfer 69403 completed successfully. Full transfer 49851 now active; MTG video bytes advancing (~70 MB/s instantaneous early reading; aggregate not stable yet). CPU job scheduled end 2026-11-23 04:22:47 UTC. Evaluation dataset verifier: `source hpc/config.sh && in_env ro 'python scripts/verify_transferred_dataset.py --root /scratch/yy5919/mtg-ad-transfer/Evaluation-datasets/egoschema --output outputs/verification/egoschema-transfer-final.json --probe'`; run similarly for reframed after transfer. Interim report explicitly incomplete; must not reuse as completion evidence.
MTG checksum verification queued as process session 25921: waits for confirmed MTG transfer completion while checking live transfer PID 10131, then runs `cd /scratch/yy5919/mtg-ad-transfer/mtg-ad && sha256sum -c SHA256SUMS` to outputs/verification/mtg-sha256.log. All 10 configured development videos are marked ready in the Torch source manifest (partial local arrival recorded in mtg-dev-source-availability.json). Remote repo matches source SHA 3139be9; no experiment outputs/cache directories found in searched repo/source depth 3. Transfer progressed to 23% of MTG at ~72MB/s instantaneous, ~53MB/s aggregate including metadata overhead.
Evaluation dataset verification watcher session 73417 is live: waits for each successful transfer stage while checking transfer PID 10131, then runs --probe verification for EgoSchema and REFRAMED sequentially. Outputs will be outputs/verification/{egoschema,reframed}-transfer-final.json. Do not launch duplicates. MTG checksum watcher remains 25921. Verifier safety tests passed 5/5 in 0.37s (pytest-transfer-verifier.log); verifies missing/size/duplicate/unexpected cases cannot report completion. MTG transfer reached 45% at ~59MB/s aggregate.
Annotation/manifest reconciliation completed: outputs/verification/annotation-coverage.json (reproduce with python outputs/verification/audit_annotation_coverage.py). MTG 332 validation/101 test annotations, 314/99 source-available; no split ID overlap. REFRAMED 1836/96/91 annotations, 1796/96/91 source-available: all 40 unavailable IDs are TRAIN; validation/test source coverage is complete. No split ID overlap or available IDs without annotations. EgoSchema 5031 unique questions/manifest videos, 500 valid public labels (0..4), no orphan labels. This establishes source mapping, not completed local transfer. MTG copied ~70%; verification processes remain pending/live.
MTG transfer and integrity verification COMPLETE: transfer stage exited successfully; checksum session 25921 exited 0; all 413 SHA256 hashes pass. Independent local manifest comparison: 413 expected/actual videos, no missing/unexpected/size mismatches. Evidence outputs/verification/mtg-sha256.log and mtg-transfer-final.json. This verifies transferred MTG media, not GPU experiment completion. EgoSchema transfer continuing (~9%); REFRAMED pending. Source unavailable MTG count 20 remains documented and is not a local transfer failure.
MTG CPU development preprocessing completed (session 49519 exit 0): all ten configured dev videos prepared using existing prepare_targets/preprocess_video, fps=2, max side 768, audio 16kHz. Raw source stayed read-only; derived cache /scratch/yy5919/pyro-kd-derived/cache/mtg-ad. Reproduce: source hpc/config.sh && in_env ro 'OMP_NUM_THREADS=4 python -u outputs/verification/prepare_verified_mtg_dev.py'. Evidence mtg-dev-preprocess.{json,log}, with code SHA and split hash; separate cache check via outputs/verification/check_mtg_dev_cache.py. GPU jobs remain on hold. EgoSchema transfer ~15%, REFRAMED queued.
MTG CPU audio baseline prepared: session 46787 exit 0, Silero/librosa computed current features for all 10 dev clips, no failures. Command: source hpc/config.sh && in_env ro 'OMP_NUM_THREADS=4 python -u scripts/cache_audio_features.py --cache-dir /scratch/yy5919/pyro-kd-derived/cache/mtg-ad'. Evidence outputs/verification/mtg-dev-audio-features.log and cache audio_features_report.json.
Existing Uniform moment builder session 29496 exit 0: 78 generated input moments across 10 videos, 111 zero-word moments excluded (189 requested). All 78 have four existing frames and positive word limits. Reproduce command in outputs/verification/mtg-dev-moments.log context: `python scripts/build_moments.py --dataset mtg-ad --selector U --videos dev --data-dir /scratch/yy5919/mtg-ad-transfer/mtg-ad --cache-dir /scratch/yy5919/pyro-kd-derived/cache/mtg-ad --output /scratch/yy5919/pyro-kd-derived/moments/mtg-ad_U_dev.jsonl`. Sidecar reports all shortfalls. These use historical builder rules (including cap30), not a claim of full Revision 1.5 fidelity; reconcile fixed-K/word-cap protocol before comparison matrix. No GPU inference yet. EgoSchema transfer ~22%; remaining pipeline still live.
Dataset gate research recorded in docs/DATASET_GATES.md; full additional-benchmark access remains unverified. Current data transfer still active at EgoSchema ~27%; no additional downloads compete with it.
Loader-compatible dataset views prepared outside raw roots at /scratch/yy5919/pyro-kd-derived/datasets/{mtg-ad,egoschema,reframed}. REFRAMED view links source encrypted annotations and videos and extracts only the two published dataset decryption keys to its own keys/ directory. Raw bundle remains unchanged. Reproduction outputs/verification/prepare_dataset_views.py; runtime loader check outputs/verification/check_dataset_loaders.py, session 29421 exit 0. Verified existing repository loaders return MTG 332/101, REFRAMED 1836/96/91, EgoSchema 5031 total/500 labeled. Evidence dataset-loader-counts.json. This is annotation compatibility, not full local media completion. EgoSchema transfer ~41%; transfer 49851 and verifier watcher 73417 remain pending/live.
EgoSchema transfer stage COMPLETED successfully: log marker TRANSFER_COMPLETE Evaluation-datasets/egoschema; 105,941,619,272 video bytes transferred, metadata already copied. Full manifest/stream audit now running as PID 20310 under watcher 73417; not complete yet. REFRAMED transfer has started (~3% copied). Transfer supervisor 49851/PID 10131 remains live. CPU allocation still has several hours remaining; throughput variable. Do not report EgoSchema verified until final JSON inspected. Current scripts/logs retain resumability and raw data remains unchanged.
EgoSchema VERIFIED: outputs/verification/egoschema-transfer-final.json reports complete=true, expected=verified=5031, zero missing/unexpected/errors/snapshot_errors. All 5031 have no audio stream. Checks cover IDs, sizes, ffprobe stream signatures and durations against manifest; no full decode/hash claim. Watcher 73417 continues waiting for REFRAMED. This passes the actual-data EgoSchema audio gate: visual-only U/V diagnostic; A/AV requires an audio-bearing benchmark, no fabricated audio. REFRAMED ~6% copied, active transfer 49851.

CPU smoke readiness: real MTG four-moment dry run passed; existing Qwen2.5-VL-7B home-cache revision cc594898137f460bfe9f0759e9844b3ce807cfb5 has five indexed shards present and offline processor/config load passed. Prepared later GPU command and evidence in docs/REAL_DATA_SMOKE.md. No model download or GPU inference performed; full smoke gate pending.

P1 CPU implementation prepared: guidelines plus exactly three frozen training AD examples with validation/test movie exclusion. Bundle /scratch/yy5919/pyro-kd-derived/prompts/reframed-train-p1.json, SHA256 a51d9076f3dfeb1bb50b1c209aba3b3e2a789843c98eb7dcb5aeaba4db724faf. Targeted 63 tests/17 subtests passed; real four-moment P1 dry run passed. Details docs/PROMPT_PRIORS.md. GPU inference remains on hold pending user session.

Revision1.5 word-budget correction: new /scratch/yy5919/pyro-kd-derived/moments/mtg-ad_U_dev_revision15.jsonl uses exact floor(gap*3), retaining 78 moments and unchanged frames/times; 53 budgets differ from historical cap30. All have known gaps, max793. Old inputs preserved; new output paths required for protocol comparisons. Targeted69 tests/794 subtests passed; P0 real-input dry run passed. See docs/REAL_DATA_SMOKE.md. Selector shortfall111 remains unresolved before full matrix.

REFRAMED development CPU preprocessing started, session56368, inside existing user CPU job40036. Command: source hpc/config.sh && in_env ro 'OMP_NUM_THREADS=4 python -u scripts/preprocess_reframed_dev.py --data-dir /scratch/yy5919/pyro-kd-derived/datasets/reframed --cache-dir /scratch/yy5919/pyro-kd-derived/cache/reframed --report outputs/verification/reframed-dev-preprocess.json'. Log: outputs/verification/reframed-dev-preprocess.log. Uses the ten frozen validation IDs only; no new sampling, downloads or held-out media preprocessing. Do not launch duplicate prep while live.

REFRAMED dev preprocessing session56368 exited0: all10 clips complete/current,2663 frames, all have audio, max duration drift0.043s. Next registered CPU step: source hpc/config.sh && in_env ro 'OMP_NUM_THREADS=4 python -u scripts/cache_audio_features.py --cache-dir /scratch/yy5919/pyro-kd-derived/cache/reframed', log outputs/verification/reframed-dev-audio-features.log. This is the existing Silero/librosa cache, not the full three-stream event stack.

REFRAMED dev CPU preparation COMPLETE: preprocessing56368, audio cache32803, U builder31194 all exited0. Frames2663 across10clips; all audio current. Derived moments /scratch/yy5919/pyro-kd-derived/moments/reframed_U_dev_revision15.jsonl has66 rows across10videos,45 zero-word exclusions/111 requested. Inputs checked for four existing frames and positive exact floor(gap*3) budgets. Reproduce builder: source hpc/config.sh && in_env ro 'python scripts/build_moments.py --dataset reframed --selector U --videos dev --data-dir /scratch/yy5919/pyro-kd-derived/datasets/reframed --cache-dir /scratch/yy5919/pyro-kd-derived/cache/reframed --output /scratch/yy5919/pyro-kd-derived/moments/reframed_U_dev_revision15.jsonl'. Full fixed-K selectors/audio stack and GPU inference remain pending.

Registered CPU audio-model preparation: pinned configs/audio_models.json (Whisper-small.en, AST AudioSet, CLAP unfused; approximately1.45GB). Download command source hpc/config.sh && in_env ro 'HF_HUB_OFFLINE=0 python -u scripts/download_audio_models.py --cache-dir /scratch/yy5919/hf_cache/hub --report outputs/verification/audio-model-download.json'; log audio-model-download.log. Separate audio venv preparation planned at /scratch/yy5919/pyro-audio-env; shared container overlay remains read-only. These are preparation artifacts, not validated detector/ASR outputs.

Audio-model preparation COMPLETE: pinned three snapshots downloaded1105, isolated faster-whisper install65832, corrected CPU offline compatibility check68780 all exit0. See docs/AUDIO_STACK.md, configs/audio_models.json and outputs/verification/audio-model-{download,offline-check}.json. AST527 scores/CLAP512 embedding and Whisper word timestamps verified on actual development audio; not full feature-cache or detector-accuracy completion. GPU jobs remain on hold.

Registered VAD-gated Whisper CPU cache run for10 prepared REFRAMED dev clips. Command source hpc/config.sh && in_env ro 'OMP_NUM_THREADS=4 /scratch/yy5919/pyro-audio-env/bin/python -u scripts/cache_speech_features.py --cache-dir /scratch/yy5919/pyro-kd-derived/cache/reframed --model-report outputs/verification/audio-model-download.json'; log outputs/verification/reframed-dev-speech.log. Uses pinned small.en CPU int8 with word timestamps, isolated VAD-only windows<=28s, no cross-window prompt conditioning. Raw files unchanged. Do not launch duplicate speech caching.

REFRAMED development speech cache22477 COMPLETE exit0:10 clips,1645 words,0 out-of-window rejections. Independent bounds/WAV-hash audit complete; repeat10686 reused all10. Cached files /scratch/yy5919/pyro-kd-derived/cache/reframed/*/speech_features.json; report speech_features_report.json. Reproduction in docs/AUDIO_STACK.md. Source scripts hashed in outputs/verification/reframed-dev-speech-check.json. Model/VAD accuracy and full audio-event stack remain unvalidated/incomplete; GPU hold unchanged.

Registered CPU event-track pilot on REFRAMED dev CxWby8RuUwU6: source hpc/config.sh && in_env ro 'OMP_NUM_THREADS=4 /scratch/yy5919/pyro-audio-env/bin/python -u scripts/cache_event_features.py --cache-dir /scratch/yy5919/pyro-kd-derived/cache/reframed --model-report outputs/verification/audio-model-download.json --video-id CxWby8RuUwU6'. Log outputs/verification/reframed-dev-events-pilot.log. Pinned AST/CLAP,10-second centered windows/1-second hop, batch4 CPU float32. Await pilot validation before full caching. These are window-classifier scores, not validated precise event timestamps.

Event pilot16928 exited0: CxWby8RuUwU6,74 windows. Independent check10674 passed NPZ hash, finite shapes74x527/74x512, normalized CLAP embeddings and zero onset scores for10 excluded speech/silence labels. Tests3 passed. Registered full10-clip CPU event caching using same command without --video-id; log outputs/verification/reframed-dev-events.log. Reuses pilot. No validated detector timing/accuracy claim; human gate remains outstanding.

Full REFRAMED dev event caching active session26247. Poll this session and outputs/verification/reframed-dev-events.log; final per-video report /scratch/yy5919/pyro-kd-derived/cache/reframed/event_features_report.json. Do not duplicate. Pilot independently passed; full run not yet complete.

Event extraction26247/PID50117 still active; pilot74 windows and HfOMXYYPG5nx160 windows completed. Verification script scripts/verify_event_cache.py reports partial coverage until all10 files exist. Aggregate output outputs/verification/reframed-dev-events-check.json; an incomplete audit is not a failed extraction. Continue polling existing process.

REFRAMED development event caching COMPLETE: extraction26247 exit0; audit72375 exit0 complete=true expected=verified10,1336 windows; reuse66969 exit0 all10 current. Evidence outputs/verification/reframed-dev-events-check.json/.log and reframed-dev-events-reuse.log. VAD/basic audio, gated Whisper1645 words and AST/CLAP tracks now cached for all10 dev clips. Detector accuracy, timeline/selector integration and GPU smoke remain pending; no GPU jobs started.

Audio timeline prompt integration prepared: scripts/attach_audio_timeline.py produced /scratch/yy5919/pyro-kd-derived/moments/reframed_U_dev_timeline_exploratory.jsonl (66 unchanged base inputs,129 speech/92 event entries). Use explicit --audio-context timeline or none; legacy remains default for historical commands. P0/P1 real-input dry runs passed;68 tests/17 subtests passed. Thresholds0.5 score/0.1 rise are exploratory only; detector validation/dev tuning and fixed-K selector work remain pending. See docs/PROMPT_PRIORS.md.

Placement feasibility audit complete (REFRAMED67458/MTG93133 exit0). Cached usable gaps and2s symmetric bounds admit76/111 and93/189 Uniform requests at minimum1-word capacity. More generous10s still does not fill K. Details docs/PLACEMENT_FEASIBILITY.md; per-video/hash evidence outputs/verification/{reframed,mtg}-dev-gap-feasibility.json. Full selector protocol remains unresolved; do not claim equal generated budgets or silently refill/reduce Uniform.

Generation provenance prepared: source/input/frame/settings hash guards resume, each prediction carries run_fingerprint, and a provenance sidecar is saved before model loading. Legacy output without fingerprints requires a new path. Targeted test session29896 exit0 (32 passed); real four-input REFRAMED timeline dry run69906 exit0, log outputs/verification/provenance-dry-run.log. Container test command: source hpc/config.sh && in_env ro '/scratch/yy5919/pyro-test-env/bin/python -m pytest -q tests/test_generation_provenance.py tests/test_generation_truncation.py'. No GPU inference. Optional fixed-K placement question pending; complete selector implementation, detector validation and actual GPU smoke still required.

Running visual-memory ranking implemented and exposed via build_moments.py --selector V --visual-novelty memory; local remains compatibility default and ablation. Mode stored in moment rows/reports. Sessions98942/80790 passed201 tests/893 subtests total. No derived real visual caches found, no model extraction/inference launched. Final two-stage pipeline must apply memory selection to feasible snapped candidates; old post-selection gap filtering remains. See docs/PLACEMENT_FEASIBILITY.md. GPU hold and optional fixed-K question unchanged.

Registered visual-model preparation: OpenCLIP ViT-B-32 laion2b_s34b_b79k pinned to official HF revision1a25a446712ba5ee05982a381eed697ef9b435cf. Download only open_clip_model.safetensors into scratch HF cache; CPU pilot planned on REFRAMED CxWby8RuUwU6 with batch8 and four threads. No GPU use. Do not duplicate the download/pilot while registered sessions are live.

Visual CPU pilot COMPLETE: model download9315 exit0, corrected audit49106 exit0. Real REFRAMED CxWby8RuUwU6 has146x512 embeddings and19 shots; local and memory ranking both produced6 exploratory moments with2s separation. Evidence outputs/verification/reframed-visual-pilot.json, repro docs/REAL_DATA_SMOKE.md and scripts/check_visual_pilot.py. Initial pilot91825 exited1 in checker after successful extraction; corrected reload reused and validated cache. All these sessions terminal; no visual job active. Remaining9 REFRAMED dev clips not yet visually cached. GPU hold unchanged.

Registered full10-clip REFRAMED dev visual caching inside user CPU job40036: source hpc/config.sh && in_env ro 'OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 python -u scripts/cache_visual_features.py --cache-dir /scratch/yy5919/pyro-kd-derived/cache/reframed --batch-size 8'. Log outputs/verification/reframed-dev-visual.log. Reuses verified pilot; pinned offline OpenCLIP, CPU node only. No duplicate visual process present at launch. Final coverage verification required before selector tuning.

Full REFRAMED development visual caches COMPLETE:93892 exit0,9 computed1 reused0 failures; audit30957 exit0 verified10/10,2663 frames386shots. Evidence outputs/verification/reframed-dev-visual-check.json. Exploratory builders19822 exit0: /scratch/yy5919/pyro-kd-derived/moments/reframed_V_{memory,local}_dev_exploratory.jsonl have82/75 rows across9videos each;29/36 of111 requested moments excluded by old post-selection speech filtering. Input audit outputs/verification/reframed-dev-v-input-check.json verifies uniqueIDs, four real frames, mode and exact positive floor(gap*3). All listed sessions terminal. Not a valid equal-K comparison; final snapped-candidate integration and dev tuning remain. No GPU inference.

Feasible-placement selection core implemented: src/utils/placed_selection.py; dev audit scripts/audit_placed_visual.py. Tests71330 exit0 (22passed); real audit67412 exit0, outputs/verification/reframed-dev-placed-visual.json. Both local/memory select97/111 with14shortfall under explicit exploratory required_words1/max_advance2/max_lag2. Independent20-case bounds/capacity/separation/accounting audit passed. This fixes memory contamination by candidates later rejected for speech, but generation builder integration, audio cue scoring, A/AV, tuning and final fixed-K policy remain unfinished. No active new processes or GPU jobs.

Registered CPU speech-cue preparation on10 frozen REFRAMED dev clips: scripts/prepare_speech_cues.py, pinned cached all-mpnet-base-v2 revision e8c3b32edf5434bc2275fc9bab85f82640a19130, four threads/batch16. Separate deictic phrase, adjacent transcript semantic-change and VAD long-silence tracks. Exploratory silence threshold5s/scale30s; no tuning/diarization claim. Output outputs/verification/reframed-dev-speech-cues.json; log reframed-dev-speech-cues.log. No model downloads or GPU use.

Speech-cue preparation COMPLETE:80086 exit0, audit86766 exit0; outputs/verification/reframed-dev-speech-cues.json and -check.json.10devclips294chunks18long-silencecues0deicticmatchedchunks. Tests84673 exit0 (8passed). New scripts/prepare_speech_cues.py uses offline pinned MPNet CPU; src/utils/speech_cues.py keeps cue components separate. Thresholds5s/30s exploratory; topic-change reliability, diarization, A/AV integration and tuning remain pending. All handles terminal. GPU hold unchanged.

Registered exploratory A/AV candidate audit: scripts/audit_fused_selectors.py, ten frozen dev clips, explicit1word/±2s gap constraints, global event trigger/suppress conditions. Separate cached speech/non-speech/visual components; no tuned weights or GF/OP score. Output outputs/verification/reframed-dev-fused-selectors.json; log reframed-dev-fused-selectors.log. CPU only, no generation.

Exploratory A/AV audit COMPLETE:33362 exit0, four conditions x10devclips; each97selected/111requested. Outputs outputs/verification/reframed-dev-fused-selectors.json and -check.json;40case independent audit passed. Tests83827 exit0 (18passed). Global trigger/suppress changes event selections on all10clips in both modes. No tuned weights, GF/OP or final matrix claim. Historical A builder remains unchanged; new pipeline currently diagnostic only. Next: missing score terms/ablation plumbing and explicit generation integration. GPU hold unchanged, all listed handles terminal.

Placed A/AV generation inputs COMPLETE for exploratory dev work: exporter56829 exit0, timeline/dryrun31569 exit0. Four97row/10video inputs (A/AV trigger/suppress), each111requested14shortfall, under /scratch/yy5919/pyro-kd-derived/moments/*placed_event_dev_exploratory*.jsonl. Eight P0/P1 first4row dry runs passed; independent388row check outputs/verification/placed-input-check.json passed. Tests70547 exit0 (35passed). Prompts/timelines use event frame-center time; prediction metadata preserves narration placement and lag. Final model smoke, missing selector score terms/ablations, tuning and fixed-K resolution remain. GPU hold unchanged, all handles terminal.

Registered exploratory placement evaluation on frozen REFRAMED dev annotations only: scripts/evaluate_placed_dev.py consumes existing fused and visual placement reports; US AD cues merged<=0.5s, exact one-to-one matching±2s, raw matches and shortfalls retained.10000resample video bootstrap requires one clip per unique movie. Output outputs/verification/reframed-dev-placement-metrics.json/.csv, log reframed-dev-placement-metrics.log. Not a tuned/final comparison; no test split or GPU use.

Exploratory dev placement metrics COMPLETE:25290 exit0; outputs/verification/reframed-dev-placement-metrics.json/.csv;60records6conditions10unique movies. Independent metrics-check.json passed; plot47100 exit0, reframed-dev-placement-f1.png inspected. Four tests91330 passed. MacroF1 Atrigger.389/Asuppress.330/AVtrigger.407/AVsuppress.377/Vlocal.413/Vmemory.461; all paired differences include0 in95% bootstrap intervals. Not final results; untuned/missing components, U/O and full matrix remain pending. GPU hold unchanged.

Hierarchical budget mechanics COMPLETE, long-form validation pending:tests4597 exit0 (25passed); short-dev diagnostic13917 exit0. outputs/verification/reframed-dev-hierarchy.json/-check.json:10clips55segments111allocated92selected19shortfall under exploratory20–60s segmentation. Global selected memory/separation preserved across segment quotas. Does not fulfill long-form benchmark or final ablation. GPU session status question pending; continue CPU-only until user supplies active session/changes hold.

Video-MME long subset frozen in configs/video_mme_long_subset.json:12held-out long videos across6domains, label-blind metadata selection; pinned dataset ead1408f75b618502df9a1d8e0950166bf0a2a0b. Raw metadata/release records /scratch/yy5919/pyro-kd-derived/video-mme. Initial range-index58689 terminal failed timeout; corrected85580 ACTIVE, log outputs/verification/video-mme-inventory-retry.log, atomic per-archive checkpoints archive-index-parts/*.json. At last check16archives760members indexed,11/12selected found; fullindex output archive-inventory.json pending. Tests6555 exit0 (4passed). No bulk101GB download or selected video extraction yet. Next poll85580; then map selected members, download only those with byte/hash/CRC checks, probe actual audio/duration. GPU hold unchanged.

Video-MME inventory85580 TERMINAL exit1:16complete archives760members81514indexbytes,4failed archive reads (17–20). Durable checkpoints and partial archive-inventory.json preserved; selected-archive-members.json maps11/12 frozen IDs,ID864unlocated. Direct17probe9504 terminal URLError; range reader now distinguishes exhausted transport from ZIP parsing errors (tests91698 exit0,5passed). No active index process; next recover remaining ranges or inspect official alternative access, then selective extraction/audio probes. No media payload downloaded. Source data location/gate details docs/DATASET_GATES.md. GPU hold unchanged.

Registered Video-MME media access pilot on frozen ID857 (smallest located selected member, not a subset replacement). scripts/download_video_mme_subset.py uses exact ranges, resumes compressed bytes, validates uncompressed size/ZIPCRC/SHA256 and ffprobes final video. Target /scratch/yy5919/pyro-kd-derived/video-mme/videos/857.mp4 (~92MB); log outputs/verification/video-mme-857-download.log. Only this pilot, no bulk release. Actual audio/duration gate pending; no model inference.

Video-MME pilot857: download13967 terminal exit1 only because host lacked ffprobe AFTER validated final file publication. Container recheck13676 exit0 reused exact file:92269189bytes SHA25634af0d3faf68073450ba019f2181b3694626a6fb54ef737d8f4f7ed4732de9b5,1988.998095s (~33.15min),H264video+AAC44.1kHzstereo. Actual audio availability gate passed for this file only. Registered full frozen subset transfer with two workers and per-file resumable compressed bytes;ID864 remains unlocated and must be reported failed, not replaced. Log outputs/verification/video-mme-subset-download.log; report /scratch/yy5919/pyro-kd-derived/video-mme/videos/download-report.json.

ACTIVE job registry: Video-MME subset transfer48497 (2workers) -> outputs/verification/video-mme-subset-download.log; final/incremental /scratch/yy5919/pyro-kd-derived/video-mme/videos/download-report.json. Current input member map has11located+864missing. ID857 already verified33.15min AACstereo+H264; other compressed partials resume by exact byte count. Missing-index refresh33675 -> video-mme-inventory-refresh.log, target archive-inventory-complete.json; reused16indexes, recovering17–20. Fresh17probe worked; range URL refresh now in source, existing48497 retains earlier imported reader until it exits. Do not duplicate live processes. If new complete map resolves864, download it separately only after confirming no process owns that ID; initial all-worker map cannot download864 because its member was null.

Index refresh33675 COMPLETE exit0:20archives900members,96834indexbytes, all12frozen IDs now located. New immutable selected-archive-members-complete.json created under scratch Video-MME root; original11member map preserved for live48497. Next use complete map to fetch864 after current subset transfer terminates (or with explicit nonduplicate ownership); initial job will still report its old null864 entry. Actual full12media/audio gate remains pending. Tests51515 exit0 (10passed).

Partial independent long-video audit92216 terminal exit1:4/12verified (808,802,616,857), all with decoded nonzero10s audio samples at10/50/90% and33.15–48.58min durations. Primary48497 remains live on other IDs. Its per-file completed-future report proves604/710/729failed transport and864failed unlocated; no writer remains for these IDs. Registered one sequential recovery worker for710,729,604,864 using complete member map and refreshed range URLs; no overlap with primary's remaining891/898/753/763. Recovery log outputs/verification/video-mme-recovery.log. Combined network concurrency at most3. Final all12audit still required.

Active transfer continuation: primary48497 still live, recovery36189 still live. Recovery710 now terminal per-file success (109574127bytes,1981.59093s,AACstereo/H264) after exact compressed-byte resume; recovery advancing729/604/864. Verifier script scripts/verify_video_mme_subset.py committed for final all12audit; partial92216 result4/12is not a failed transfer diagnosis. Poll existing handles and per-file verified metadata; do not start duplicate all-download. Final audit only after terminal/recovered coverage.

Registered second sequential recovery worker for729/891 only. Primary48497 completed891 with transport failure and now handles763; recovery36189 completed729 exit1 and now handles604 then864. These per-file terminal records establish nonoverlap. Seven verified records already exist (616,710,753,802,808,857,898). New log outputs/verification/video-mme-recovery-two.log/results.json; total active download concurrency remains at most3. Do not retry any other IDs while current writers are live.

Current transfer ownership: primary48497 live on763 (last reported260046848/327293332compressed bytes); recovery36189 live on864 after604success (last100663296/630054955); recovery26985 live,729failed again and891next/current. Seven prior verified IDs plus604now verified.729must await a new recovery after existing tasks terminate; its partial retained. Per-attempt resolver refresh added, tests89097 exit0 (10passed); already-running Python processes retain their imported reader. No whole-file restarts or duplicate writers. Final all12verification unproven.

Previous download jobs48497/36189/26985 now all TERMINAL exit1 (partial failures).891recovered successfully, leaving nine verified files and729/763/864unfinished. No writers remain. Registered one consolidated2worker recovery using complete member map and per-attempt resolver refresh; existing final files CRC/SHA rechecked/reused, compressed partials resumed. Preserved initial download report as videos/download-report-initial.json. New log outputs/verification/video-mme-final-recovery.log; do not duplicate while live.

Consolidated61556 TERMINAL exit1:11/12final files verified, only864failed transport at452984832/630054955compressed bytes. Registered single-ID864resume; no other downloader remains live. Log outputs/verification/video-mme-864-final.log. All other finals reused/preserved; final independent all12audit pending this last member.

Video-MME long data COMPLETE: all12frozen IDs downloaded; last73718 exit0, final94631audit exit0 complete=true/verified12/audio_gate_passed=true.3.221846667GB,7.715hours,30.518–50.501min pervideo;all audio+video and36decoded nonzero10s audio samples. Evidence outputs/verification/video-mme-subset-check.json and reconciled scratch videos/download-report.json. Old failed reports retained. Mapper scripts/map_video_mme_subset.py tested3cases and reproduced all12members. ALL transfer/audit handles terminal (48497,36189,26985,61556,73718,94631); do not poll/restart. Next CPU work: finish selector score/ablation plumbing and long-QA preprocessing/model-language policy without inspecting held-out answers. GPU hold and unanswered protocol question remain; no real VLM inference yet.

Onset cue cache/integration COMPLETE for10REFRAMEDdev clips:18727/63875/29845/45064exit0, tests22770passed9.1336windows source/grid/normalization verified. New opt-in --include-onsets result outputs/verification/reframed-dev-fused-with-onsets-v2.json retains source fingerprints; counts97/99/97/98 vs111requested. Historical recipes/reports preserved. No new performance metrics or protocol freeze. Remaining GF/OP terms, superclass tuning, complete ablations, human gates and GPU inference still pending. Long-video12filedata gate remains passed.

Common-placement U/O diagnostics complete:79362exit0; U76/O85 of111requested. Evaluation22564exit0 now8conditions80records with explicit macro matched/requested and coverage; independentchecks passed. Plot57893inspected; tests44678passed23. Artifacts outputs/verification/reframed-dev-placement-with-baselines.*. Onset-enhanced A/AV used in this new report; older results unchanged. Final protocol/score terms/tuning and U/O generation integration still pending. GPU hold unchanged; no active new jobs.

U/O/V generation inputs complete (2026-11-23; CPU job40036, host b-10-63). Source before edits802031d5df6bd20d2f8a6cf5037767aae8479f3b. Exporter now accepts explicit baseline conditions and visual modes with their corresponding report constraint formats; no variant fallback. Preparation38298 exit0: U_snapped76rows/9nonempty videos (10 audited), O_snapped85rows/10videos, V_local97 and V_memory97rows/10videos; each requests111, with35/26/14/14 shortfalls. Inputs and timeline variants: /scratch/yy5919/pyro-kd-derived/moments/reframed_{condition}_placed_event_dev_exploratory{,_timeline}.jsonl. All355 rows preserve source fields/frame files/context centers. Evidence outputs/verification/placed-baseline-visual-input-check.json. Eight first-four-row P0/P1 dry runs passed, logs placed_{condition}_{prior}_dry.log. Targeted tests64011 exit0:15passed. No weights loaded or inference performed; GPU hold persists.

Reproduction inside `source hpc/config.sh; in_env ro 'COMMAND'` (new output paths required):
- `python scripts/export_placed_moments.py --selection-report outputs/verification/reframed-dev-placed-baselines.json --condition U_snapped --cache-dir /scratch/yy5919/pyro-kd-derived/cache/reframed --frame-center event --output NEW_BASE.jsonl`; repeat O_snapped; V_local/V_memory use reframed-dev-placed-visual.json.
- `python scripts/attach_audio_timeline.py --input NEW_BASE.jsonl --output NEW_TIMELINE.jsonl --cache-dir /scratch/yy5919/pyro-kd-derived/cache/reframed --min-score .5 --min-rise .1`.
- `python scripts/run_ad_vlm.py --input NEW_TIMELINE.jsonl --prior P0 --audio-context timeline --limit 4 --dry-run`; repeat P1 with `--exemplars /scratch/yy5919/pyro-kd-derived/prompts/reframed-train-p1.json`.
- `/scratch/yy5919/pyro-test-env/bin/python -m pytest -q tests/test_placed_export.py tests/test_generation_provenance.py`.
All five selector families now have exploratory input plumbing; this is not the finalized20-condition protocol. Existing A/AV exports retain the earlier no-onset recipe. Remaining work includes missing score terms, tuning/protocol freeze, real GPU smoke/inference, long-form QA integration, human gates and final evaluation.

Rendered-overlap evaluator implemented (2026-11-23, CPU40036; source baseline e490939). `src/evaluation/speech_overlap.py` unions half-open detector intervals and reports per-description/summed/unique-timeline collisions separately for VAD, Whisper and their union, plus video overruns and extra AD layer duration. `scripts/evaluate_rendered_overlap.py` validates TTS text/audio hash bindings and explicit1x source playback, checks live media/VAD/Whisper provenance, materializes1.2x WAVs through ffmpeg, then measures actual PCM sample durations. Retains accelerated artifacts, hashes, implementation hashes and ffmpeg version; no word-count duration proxy. CLI usage and metric definitions are in src/evaluation/README.md. Test24655 exit0:11passed, including actual ffmpeg tempo conversion on an explicitly synthetic unit-test tone and invalid-binding rejection before output creation. CLI help53954exit0. Earlier test30822/1829 terminal0. No project narration synthesized or evaluated: no real TTS predictions exist. GF/OP selector scoring is still outstanding; rendered OP cannot honestly be claimed from gap capacity alone. GPU hold unchanged.

Optional GF selector score complete, calibration pending (2026-11-23; CPU40036, source f4c05f6). Added bounded remaining-capacity score C/(C+T), explicit target/weight, inspected per-selection component. Target15/weight1 is exploratory, not a final protocol or a word cap. Tests6554exit0:43passed (placement, hierarchy, fused candidates, export). Real dev40510exit0:392placements, each independently reconstructed from components and exact gap capacity; counts97/99/98/98 of111each. Zero-weight ablation97440exit0 exactly reproduced391 prior placements and all prior condition fields. Evidence reframed-dev-gap-score-check.json. Reproduce with scripts/audit_fused_selectors.py using --data-dir /scratch/yy5919/pyro-kd-derived/datasets/reframed --cache-dir /scratch/yy5919/pyro-kd-derived/cache/reframed --speech-cues outputs/verification/reframed-dev-speech-cues.json --required-words 1 --max-advance 2 --max-lag 2 --include-onsets --gap-weight 1 --gap-target-words 15 --output NEW.json; repeat --gap-weight0 in a new output. Container pytest command targets tests/test_placed_selection.py tests/test_hierarchical_budget.py tests/test_fused_candidates.py tests/test_placed_export.py. All handles terminal. Rendered OP selection integration, tuning, full core matrix and GPU smoke still pending; no GPU submitted.

Long-video preprocessing registered (2026-11-23; user CPU job40036, host b-10-63; source baseline202d9fa). Preparing all12 frozen Video-MME long videos sequentially at2fps/max-side768/mono16kPCM under /scratch/yy5919/pyro-kd-derived/cache/video-mme. No question/answer columns read. Each raw MP4 SHA is checked against video-mme-subset-check.json before work; current complete caches are reusable; report checkpoints per video. Command: source hpc/config.sh; in_env ro 'python scripts/preprocess_video_mme.py --verification outputs/verification/video-mme-subset-check.json --video-dir /scratch/yy5919/pyro-kd-derived/video-mme/videos --cache-dir /scratch/yy5919/pyro-kd-derived/cache/video-mme --report outputs/verification/video-mme-preprocessing.json'. Planned live log outputs/verification/video-mme-preprocessing.log. No duplicate preprocessing submission while this process is live. Completion not yet established.

Live preprocessing handle71704 confirmed running after two completed videos808/802 (10031frames total); remaining10 not yet complete. Tests43163 passed6 and expanded verifier test85352 passed7. Independent verifier implemented: `python scripts/verify_video_mme_cache.py --preprocessing-report outputs/verification/video-mme-preprocessing.json --cache-dir /scratch/yy5919/pyro-kd-derived/cache/video-mme --output outputs/verification/video-mme-cache-check.json`. Run only after preprocessing71704 is terminal and report.complete=true. It decodes all cached JPEG pixels with4threads, checks dimensions/timestamp grids/source+WAV hashes, reads full PCM payload, and checks duration coverage. No QA text or labels accessed. Do not resubmit71704 based solely on lack of log output; it may be hashing/statting NFS files between ffmpeg processes. Current preparation is real useful CPU work, not a keepalive.

Long-video timing check90548exit0: all12 source A/V streams start at0; maximum stream duration difference0.081542s, stored video-mme-stream-timing.json. Preprocess71704 remains active, seven videos complete at latest check. Optional multilingual ASR plumbing added while waiting: --model-id from supplied pinned model report, --language auto|CODE; detects per VAD window and records language probability, rejects English-only weights for auto/non-en. Existing English defaults unchanged. Tests56049passed12 plus4subtests. No multilingual ASR run yet; multilingual model pin/offline check and language-compatible semantic cues still needed. See docs/AUDIO_STACK.md.

Video-MME preprocessing71704 completed exit0 for all12 videos. Registered independent full-cache verification next: scripts/verify_video_mme_cache.py with complete preprocessing report, cache/video-mme root, output outputs/verification/video-mme-cache-check.json; live log outputs/verification/video-mme-cache-check.log. No duplicate verification while active. This verifies JPEG decoding/full PCM coverage, not QA or semantic annotation quality.

Video-MME long media cache COMPLETE and independently verified (2026-11-23). Preprocessing71704exit0; verifier96153exit0. All12 frozen videos,55549 decoded cached JPEGs at2fps/max-side768,444395013 PCM samples (7.715191h) at16k mono. Full WAV payloads read and SHA-checked, all source MP4 hashes rechecked, frame grids and<=1s duration coverage passed. Artifacts outputs/verification/video-mme-preprocessing.json and video-mme-cache-check.json, logs alongside; caches /scratch/yy5919/pyro-kd-derived/cache/video-mme. Earlier stream-timing90548exit0 established all A/V starts0. No questions/answers or video semantics inspected. All preparation/verification handles now terminal. This completes media preprocessing only: long-form VAD/visual/event/multilingual ASR features, hierarchy selection, QA integration and actual inference remain. GPU hold unchanged.

Registered long-video Silero VAD/basic gap-acoustic CPU preparation (user allocation40036; source531608d). Input12 verified caches /scratch/yy5919/pyro-kd-derived/cache/video-mme; existing default VAD threshold0.5/min speech0.25s/min silence0.1s, usable gap minimum1s/edge margin0.25s. No test-dependent tuning or QA text/answers. Command source hpc/config.sh; in_env ro 'OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 python -u scripts/cache_audio_features.py --cache-dir /scratch/yy5919/pyro-kd-derived/cache/video-mme'. Log outputs/verification/video-mme-vad.log; report in cache root/audio_features_report.json. Reuses current per-video caches on resume; no duplicate while active. Not validated language-specific detector accuracy.

Registered multilingual Whisper-small download on CPU: config configs/multilingual_speech_model.json, Systran/faster-whisper-small pinned536b0662742c02347bc0e980a01041f333bce120,486214370bytes across5files (sizes from model API). This is a separate model from existing small.en; no English cache overwrite planned. Model card confirms multilingual CTranslate2 conversion of openai/whisper-small. Command source hpc/config.sh; in_env ro 'HF_HUB_OFFLINE=0 python -u scripts/download_audio_models.py --config configs/multilingual_speech_model.json --cache-dir /scratch/yy5919/hf_cache/hub --report outputs/verification/multilingual-speech-download.json'; log multilingual-speech-download.log. Offline compatibility check to use existing REFRAMED development audio only, no held-out QA text or semantic inspection.

Multilingual Whisper preparation complete: download42077exit0, offline CPU check57147exit0. Pinned536b0662742c02347bc0e980a01041f333bce120,5files486214370bytes, all hashes in multilingual-speech-offline-check.json. Actual existing dev excerpt0.5–3.3s yielded10in-gate words, en probability0.9944057465, no rejected words. This is compatibility, not multilingual accuracy validation. Long VAD handle14618 remains live; first video604 completed (1783s detected speech/1831s,9usable gaps); monitor same handle, do not duplicate. GPU hold persists. Long ASR and multilingual semantic/lexical cue policy still pending.

VAD audit preparation: scripts/verify_vad_cache.py independently checks exact frozen subset, current WAV/settings hash, source media freshness, actual WAV duration, nonoverlapping speech intervals/summed seconds, and margin-trimmed complement gaps with finite nonnegative acoustics. Six unit tests70961exit0. Actual completed six long-video records passed interval checks37146exit0; this is partial evidence, not final12coverage. Full command after14618terminalsuccess: source hpc/config.sh; in_env ro 'python scripts/verify_vad_cache.py --subset configs/video_mme_long_subset.json --cache-dir /scratch/yy5919/pyro-kd-derived/cache/video-mme --output outputs/verification/video-mme-vad-check.json'. Detector accuracy and acoustic value re-computation are outside this audit. Long VAD14618 remains live; no duplicate submission.

Registered long-video ASR pilot604 while VAD14618 continues on other IDs. Video604 VAD is complete and independently interval-checked; no writer will modify its current feature cache. CPU4threads/int8, pinned multilingual-small536b0662742c02347bc0e980a01041f333bce120, per-VAD-window auto language, no translation/no cross-gap conditioning. Command source hpc/config.sh; in_env ro 'OMP_NUM_THREADS=4 /scratch/yy5919/pyro-audio-env/bin/python -u scripts/cache_speech_features.py --cache-dir /scratch/yy5919/pyro-kd-derived/cache/video-mme --model-report outputs/verification/multilingual-speech-download.json --model-id Systran/faster-whisper-small --language auto --video-id 604'. Log outputs/verification/video-mme-asr-pilot.log. Inspect mechanical coverage/provenance only; no transcript-based tuning or QA answers. Do not duplicate this ASR pilot while live. Full12video ASR not yet launched.

Long-video VAD preparation14618exit0 and independent full-subset audit7172exit0 COMPLETE: all12 current audio feature caches, exact default VAD protocol, speech/gap interval and duration arithmetic passed. Evidence outputs/verification/video-mme-vad-check.json/.log. Mechanical validity is not human-labeled detector accuracy. Long ASR pilot604 handle35246 remains live (fourCPUthreads, ~1.4GB RSS at last check); no completed ASR cache/report yet, do not duplicate or scale until it passes verification. New scripts/verify_speech_cache.py audits pinned model/language protocol, WAV/VAD identity, exact sample-window coverage, word/gate/count/probability validity and language metadata without printing transcript text. Tests53535exit0:13passed plus4subtests. Pilot audit command after35246success: source hpc/config.sh; in_env ro 'python scripts/verify_speech_cache.py --cache-dir /scratch/yy5919/pyro-kd-derived/cache/video-mme --video-id 604 --model-report outputs/verification/multilingual-speech-download.json --model-id Systran/faster-whisper-small --language auto --output outputs/verification/video-mme-asr-pilot-check.json'. Full12ASR not yet submitted. GPU hold unchanged.

Long ASR pilot35246exit0, independent audit59487exit0:604 has5515words/105windows/28529600 gated samples, zero rejected words. Auto language windows en102/ko2/pt1 are model estimates, not verified language labels; no transcript inspected for tuning. Registered full12longASR run with same fixed model/settings, reusing604: source hpc/config.sh; in_env ro 'OMP_NUM_THREADS=4 /scratch/yy5919/pyro-audio-env/bin/python -u scripts/cache_speech_features.py --cache-dir /scratch/yy5919/pyro-kd-derived/cache/video-mme --model-report outputs/verification/multilingual-speech-download.json --model-id Systran/faster-whisper-small --language auto'. Live log outputs/verification/video-mme-asr-full.log; cache root speech_features_report.json. Expected substantial CPU time; per-video atomic caches support resume. Do not launch duplicate ASR while this is active. No QA text/answers or final metric tuning.

EgoSchema pilot frozen question-blind using500public label membership only (no label values used): configs/egoschema_pilot.json, seed0,10pilot/490heldout explicit disjoint IDs. New freeze_egoschema_pilot.py preserves existing config via exclusive creation. Registered pilot frame preparation on sameCPU40036, serial videos2fps/max-side768, no audio expected: scripts/preprocess_egoschema_pilot.py --video-dir /scratch/yy5919/pyro-kd-derived/datasets/egoschema/videos --cache-dir /scratch/yy5919/pyro-kd-derived/cache/egoschema --report outputs/verification/egoschema-pilot-preprocessing.json. Log egoschema-pilot-preprocessing.log. Runs alongside ASR94991 (4CPUthreads); distinct dataset/cache paths. No question/answer text loaded into preprocessing or prompts. No A/AV claims for silent EgoSchema.

EgoSchema pilot media COMPLETE: preprocessing58575exit0; independent verifier37856exit0 decoded all3600JPEGs across10silent180s videos, verified2fps grid/duration coverage, fresh manifests/sourceSHA and re-probed absence of audio. Artifacts outputs/verification/egoschema-pilot-{preprocessing,cache-check}.json, cache /scratch/yy5919/pyro-kd-derived/cache/egoschema. Freeze tests57913passed4. Config explicitly lists10pilot/490heldout IDs plus source answer-file hash; freeze used membership only, no label values/questions. Reproduction scripts freeze_egoschema_pilot.py, preprocess_egoschema_pilot.py and verify_egoschema_pilot_cache.py. EgoSchema U/V QA input/runner/scoring integration remains; no QA inference performed. Long ASR94991 still active, pilot604 reused current; do not duplicate. Pilot ASR35246 and audit59487 are terminal0.

EgoSchema U QA input/runner preparation complete (source baselineb2ffa43). Builder33268exit0 writes10pilot questions only,15Uniform moments/video from existing REFRAMED rate5/min,4frames/2s per moment,60images/question and no audio. Answers file is never opened by builder; explicit field construction prevents labels in output. Input /scratch/yy5919/pyro-kd-derived/moments/egoschema_U_pilot_qa_exploratory.jsonl. Full10question dryrun40172exit0 validated all600image references and prompts without weights or prediction outputs; independent anchor/frame/split/label-exclusion audit egoschema-qa-input-check.json passed. Tests11858exit0:12passed (initial74689passed10), including strict parsing, nested label rejection, dry-run no-load/no-output and mocked successful resume. Mock predictions exist only in ephemeral test fixtures. No actual QA answers/scores or GPU inference yet. Full ASR94991 remains live;604 reused current at latest log.

Reproduce via scripts/build_egoschema_qa_inputs.py --questions /scratch/yy5919/pyro-kd-derived/datasets/egoschema/questions.json --cache-dir /scratch/yy5919/pyro-kd-derived/cache/egoschema --output NEW_INPUT.jsonl. Dry run: scripts/run_qa_vlm.py --input NEW_INPUT.jsonl --output NEW_PREDICTIONS.jsonl --dry-run. Actual future GPU smoke must use pinned local Qwen snapshot with --model, start --limit1, inspect output before scaling. Sixty images/question has not been GPU-memory-tested. Runner keeps input/frame/source/settings fingerprints, append-only outputs and per-attempt resource metadata; invalid/ambiguous answers remain errors, never label-assisted parsing. Accuracy scorer, V inputs and long-form QA integration remain outstanding. Caption P1 exemplars are not inserted into zero-shot QA.

QA scoring implemented, no actual results: src/evaluation/qa.py and scripts/evaluate_egoschema_qa.py use frozen10pilot/490heldout denominators and reject split contamination, changed label-file SHA/membership, mixed fingerprints/selectors, duplicate successes and inconsistent text/index. Missing/invalid/failed/truncated responses count incorrect; conditional valid-only accuracy remains separately named. Scorer refuses empty inputs before opening labels; no zero fabricated experiment report. Tests51222exit0:23passed across QA scoring/prompt/runner fixtures (initial63055passed20). No real label values inspected or metric files created. Reproduction usage in src/evaluation/README.md. ASR94991 remains live, process96112~409%CPU/~1.46GBRSS at check,604reused and nextvideo still computing. No restart/duplicate; userCPU40036 stilllive. GPU hold unchanged.

Registered EgoSchema pilot visual-feature CPU preparation alongside ASR94991. Uses existing pinned OpenCLIPViT-B32laion2b snapshot/default PySceneDetect config, batch8, OMP/MKL4threads, no GPU. Command source hpc/config.sh; in_env ro 'OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 python -u scripts/cache_visual_features.py --cache-dir /scratch/yy5919/pyro-kd-derived/cache/egoschema --batch-size 8'; log outputs/verification/egoschema-pilot-visual.log, report cache root/visual_features_report.json. Ten pilot-only silent videos/3600frames; no QA labels or questions used. Distinct cache from active longASR. No duplicate visual preparation while live.

EgoSchema V preparation COMPLETE: CPU visual83383exit0 computed10videos/3600embeddings, no failures; full independent cache/normalization/novelty/ranking audit54574exit0 verified10/10. Extended QA builder with explicit V_local and V_memory, preserving5/minK,4frames/2s windows,2s separation, silent visual-only prompts and question/option ordering; shortfalls explicit. Inputs at /scratch/yy5919/pyro-kd-derived/moments/egoschema_{V_local,V_memory}_pilot_qa_exploratory.jsonl: each10questions/150moments/600frame references, no shortfall. Preparation/dryruns88925exit0 for all20question inputs. Independent paired-control/selection-source audit egoschema-qa-visual-input-check.json passed. Tests80565exit0:23passed (visual memory + QA prompt/runner). No model weights loaded for QA, no predictions or accuracy. Source visual features are real CPU OpenCLIP results. ASR94991 remains active, no duplicate submitted. Actual GPU memory/QA smoke and final tuned protocol remain outstanding.

Registered long-video visual cache on userCPU40036 alongside ASR94991. Uses fixed pinned OpenCLIPViT-B32/PySceneDetect defaults, CPUbatch8/OMP4/MKL4,12label-blind frozen videos55549frames. Command source hpc/config.sh; in_env ro 'OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 python -u scripts/cache_visual_features.py --cache-dir /scratch/yy5919/pyro-kd-derived/cache/video-mme --batch-size 8'; log outputs/verification/video-mme-visual.log. Each complete video is atomically cached for resume. No new thresholds/model choices based on long-video performance; no QA question/answer access. Do not duplicate while active. Allocation40036 stilllive elapsed4h21m with6hlimit at launch; useful work does not extend Slurm wall time.

Long visual cache live handle72713/process102860 confirmed active (~206%CPU/~1.53GBRSS), concurrent longASR94991/process96112 (~409%CPU/~1.34GBRSS). Extended verify_visual_cache.py with mutually exclusive --subset-config for exact frozen long IDs; existing --pilot-config and REFRAMED defaults retained, subset SHA recorded. CLI import/help76046exit0. Once72713completes, run source hpc/config.sh; in_env ro 'python scripts/verify_visual_cache.py --cache-dir /scratch/yy5919/pyro-kd-derived/cache/video-mme --subset-config configs/video_mme_long_subset.json --report outputs/verification/video-mme-visual-check.json'. Full long visual verification still pending; no duplicate submissions. Asked asynchronously whether human event labels and60peer comparisons exist and for their locations; response pending, no assumed annotations or fabricated ratings.

Long ASR second video616 completed and audit30098exit0 verified604+616:10465words/638VADwindows total,0rejected. This is2/12coverage, not fullASR completion; per-window languages are unvalidated model estimates. Source CPU40036 ends2026-11-23T04:22:47UTC (Slurm verified), useful CPU work does not extend walltime. Prepared CPU-only continuation hpc/long_cpu_resume.sbatch:4threads/8GB/3hlimit, dependencyafterany40036 to prevent overlap, per-video cache reuse, independent speech/visual work and both final audits; step failures recorded independently. Will submit from a detached committed source worktree, preserving exact recipe. Model report copied per attempt and checked against frozen repo/revision pins. Per-restart audit directories avoid overwriting verification. GPU hold persists; no GPU requested.

CPU continuation SUBMITTED40363, verified PENDING(Dependency), afterany:40036(unfulfilled), partitionn2c48m24,4CPUs/8GB/3h, noGPU. Source detached clean worktree /scratch/yy5919/pyro-long-cpu-resume-9e20099 at9e20099a7dcae76a457da2222b18e64e8be15fe4; SlurmWorkDir/Command verified point there. Log /home/yy5919/PYRO-KD/logs/pyro-long-cpu-40363.out; audits /home/yy5919/PYRO-KD/outputs/verification/long-cpu-resume-40363-attempt-0/. Submit command from frozen worktree: sbatch --parsable --dependency=afterany:40036 --output=/home/yy5919/PYRO-KD/logs/pyro-long-cpu-%j.out hpc/long_cpu_resume.sbatch /home/yy5919/PYRO-KD/outputs/verification. No duplicate active work: currentASR94991 andvisual72713 stay within allocation40036; continuation starts only after it terminates. Verified hostnameb-10-63 and /proc/self/cgroup job_40036/step_batch, so active heavy work is on Slurm compute. Visual first604 completed3662frames/493shot boundaries; remaining visual/ASR stillactive. Queued job is not experiment completion and must be monitored through final audits.

Generation resume integrity fix COMPLETE: src/generation/jsonl_recovery.py used by both AD/QA runners validates full records, preserves exact original bytes in content-addressed backup before repairing only unterminated final records, retains valid newline-less objects, rejects complete/interior corruption and detected concurrent changes. Fingerprint checks precede repair; partial-only outputs require matching provenance sidecar. Dry-run stays read-only. Tests60437exit0:48passed across recovery/QA/truncation (initial37788passed47), including interrupted multi-byte UTF-8, preserved backup, next append readable, idempotence/concurrency rejection and mocked runner resume without lost predictions. No real experiment output modified. CPU feature jobs94991/72713 remain active, visual604+616complete at last log; queuedCPU40363 stillDependencyafterany40036. Frozen CPU checkout9e20099 needs no update because it does not use generation runners.

Registered bounded long audio-event pilot604 on CPU40036 alongside ASR94991/visual72713. Existing pinned AST/CLAP snapshots, unchanged1s hop/10s centered zero-padded windows, batch4/torch4threads; no tagger threshold tuning/QA labels. Command source hpc/config.sh; in_env ro 'OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 /scratch/yy5919/pyro-audio-env/bin/python -u scripts/cache_event_features.py --cache-dir /scratch/yy5919/pyro-kd-derived/cache/video-mme --model-report outputs/verification/audio-model-download.json --video-id 604 --batch-size 4'; log outputs/verification/video-mme-events-pilot.log. Memory checked:~19GBavailable before launch, existingASR/visual each~1.5GBRSS. Do not scale until full-video numerical/source audit passes. PendingCPU40363 resumes only ASR/visual; event work is not part of that frozen job.

Long event pilot LIVE21486/process106092 (~2.15GBRSS at check), originalASR94991/visual72713 also active; no full event scale-up. CLAP load warnings inspected: unexpected text-model/projection/logit-scale weights are from loading the audio-only submodel; no MISSING or unexpected audio weights found. Extended verify_event_cache.py supports --subset-config and explicit --video-id restricted to frozen IDs, records frozen_subset_size/full_subset_coverage/configSHA, enforces memory5 and actual16kmono/duration agreement. Tests9041exit0:3passed; stronger verifier88093exit0 passed all10existing REFRAMED caches without changing features. After21486terminalsuccess, audit command: source hpc/config.sh; in_env ro 'python scripts/verify_event_cache.py --cache-dir /scratch/yy5919/pyro-kd-derived/cache/video-mme --subset-config configs/video_mme_long_subset.json --video-id 604 --model-report outputs/verification/audio-model-download.json --report outputs/verification/video-mme-events-pilot-check.json'. This is1/12mechanics coverage, not detector accuracy. Third longASR710 independently verified28950exit0:6099words/329windows,0rejected. Total verifiedASR now3/12; no semantic transcript inspection or QA scoring.


## Cue2Narrate access re-audit — 2026-11-23

Read-only GitHub API checks found main at
`08c89ea63281b819a0fd73c060d8c954f9e696ef`, containing only `.gitignore`,
`LICENSE`, and `README.md`; the releases list is empty. No runnable implementation,
checkpoints, or paired-window annotations are supplied in that public snapshot.
Evidence: `outputs/verification/cue2narrate-access-audit.json`.
Repository: https://github.com/multimodal-ai-lab/Cue2Narrate

The paper's release plan distinguishes restricted LongLSMDC data from intended
public code/checkpoints: https://arxiv.org/html/2609.01725v1 . The official LSMDC
page routes access through MPII-MD registration and credentialed downloads:
https://sites.google.com/site/describingmovies/download . Its linked form at
https://moviedescreg.mpi-inf.mpg.de/register.php is reachable and requests an email
address. No form was submitted or message sent. Standard LSMDC access alone does
not establish availability of LongLSMDC paired windows or Cue2Narrate weights.
Direct reproduction remains pending those artifacts; the baseline is retained
in scope, and published scores are not substituted for our own evaluation.


Long visual progress audit69272 finished with exit1, explicitly incomplete:
6/12 frozen IDs verified (604,616,710,729,753,763), 26482 frames total.
The other six reported no_visual_features, consistent with the live writer;
no completed cache failed numerical/provenance checks. Evidence:
outputs/verification/video-mme-visual-progress-check.json. Audit checks current
source hashes, 512-dimensional normalized embeddings, reconstructed novelty,
shot bounds and local/memory ranking budget/separation. It is not model accuracy.
Original visual process102860, ASR96112 and event pilot106092 remain live;
no duplicate jobs were started. CPU40036 ends04:22:47UTC;40363 still pending its
afterany dependency. Full final visual audit remains required after completion.


Human event-detector scoring implementation added: src/evaluation/event_detection.py
and scripts/evaluate_event_detector.py. Requires explicit human-origin annotations,
annotator attribution, reviewed classes/intervals, unknown-region exclusions,
source WAV hashes, development protocol IDs and model/window/threshold settings.
Reports per-class/per-video/pooled window-presence P/R/F1, excludes incomplete
review coverage, and checks current source/array/protocol identity and time grid.
Exact boundary contact is not event overlap; undefined ratios are null. This is
not onset accuracy, authentication of human identity, or the completed ~10x10min
human gate. No real labels or experiment scores created. Proposed convention and
input contract appended to docs/AUDIO_STACK.md; protocol/source selection remains
to be frozen before human scoring. Tests18138exit0:19passed, including boundary,
partial/unknown coverage, invalid data, empty input, and changed-WAV rejection.
Initial tests52591exit0:15passed before adding the full CLI provenance regression.
Synthetic labels stay inside temporary tests. GPU hold and human-data question
remain pending; this preparation does not remove those gates.


Added per-batch CPU event recovery in src/utils/event_checkpoints.py and integrated
it into cache_event_features.py for subsequent runs. Each source/settings-digest
namespace stores atomic float32 probability/raw-embedding batches with hashes,
indices and shape checks. Invalid/incomplete batches recompute; final caches and
numerical recipe stay unchanged. Progress logs now report completed windows and
reused batches. Live pilot106092 predates edits and remains untouched (no duplicate
writer or restart). Tests5975exit0:28passed across checkpoint, event feature and
human scoring suites; simulated interrupted inference reuses only completed
batches and reconstructs bit-identical cue arrays. No real interrupted model run
yet. Agent mistake: initial test command76673 referenced nonexistent
 tests/test_event_cache_audit.py, exit4/no tests; corrected to actual test files.
Visual extraction reached8/12 (through808); ASR and event pilot remain active.

Registered isolated real development event-checkpoint validation at /scratch/yy5919/pyro-kd-derived/checkpoint-validation-e233432, clip CxWby8RuUwU6 (73.2s). Source manifest copied unchanged with absolute references to existing raw/frame/audio data; only new event/checkpoint outputs written to isolated root. Source e233432, pinned AST/CLAP, batch4/CPU4. No duplicate writer to production caches. Plan: complete actual extraction, preserve final outputs, replay checkpoints and compare every array; human detector scores and GPU inference are not part of this check.


Real CPU event-checkpoint validation COMPLETE on isolated development clip
CxWby8RuUwU6, source e233432: initial extraction36888exit0 produced74windows and
19batch checkpoints. Final metadata/NPZ were preserved as .first files, then
replay94626exit0 rebuilt final caches reusing19/19batches. Comparison42573exit0
confirmed all7arrays bit-identical and both NPZ SHA256 equal
 e43bf89fab55d1b6fdc6e3a1fe5b9a4a468e642814425015474e073ca2bed69c.
Independent numerical/source verifier91709exit0 passed1/1selected development
clip, explicitly not10/10coverage. Evidence outputs/verification/
event-checkpoint-real-{first,replay}.log, event-checkpoint-real-comparison.json,
and event-checkpoint-real-audit.json. Actual artifacts at
/scratch/yy5919/pyro-kd-derived/checkpoint-validation-e233432/CxWby8RuUwU6.
This exercises real model outputs and missing-final-file recovery, not an OS-kill
experiment or detector accuracy. Production caches/raw data were untouched.
Long ASR96112, visual102860 and event pilot106092 continue; visual9/12logged.


Long-video QA question membership frozen in configs/video_mme_long_questions.json:
36 unique questions, exactly3 for each of12 frozen videos. New script
scripts/freeze_video_mme_questions.py reads only video_id/videoID/duration/domain/
question_id columns after checking the original annotation-file SHA256. It
rejects mismatched video metadata, missing video coverage, duplicate IDs and any
extra content columns. No question text, options or answers inspected. This
freezes the evaluation denominator, not selection/inference settings. Tests52280
exit0:6passed; actual freeze32693exit0; independent membership/source-link checks
passed in outputs/verification/video-mme-question-membership-check.json. Agent
mistake: first attempted the data venv interpreter on the host (exit127); it
requires the container's /ext3 interpreter, and the corrected in_env invocation
succeeded. Visual extraction now10/12logged (through864); longASR/eventpilot live.


Long-video QA evaluator added in scripts/evaluate_video_mme_qa.py. Uses frozen
36-question membership/source hashes; refuses empty predictions before reading
labels, reads only selected answer rows after prediction provenance checks, and
reports per-video/domain plus overall fixed-denominator accuracy. Missing,
invalid and truncated answers count incorrect. Tests76164exit0:19passed across
new and shared QA scoring. No actual answer labels loaded or QA scores created.

Long event pilot21486exit0:604 completed1832windows. Independent audit15996exit0
verified exact pinned models, WAV/source identity, dimensions, normalization,
RMS and reconstructed cues; explicit1/12coverage, not detector accuracy. Prepared
hpc/long_events_resume.sbatch (bash syntax and model pins checked): CPU4/12GB/6h,
revalidates604pilot before extraction, resumes batch checkpoints, audits all12,
per-attempt logs/status and nonzero failures. Plan full event extraction now
insideCPU40036 plus immutable continuation afterany40036; no GPU use. All ordinary
CPU work stays on the compute allocation; no duplicate active cache writers.


Full long event extraction started handle27774 insideCPU40036 from source
65258e9458db0b8a94c468eb682f5eba984b213f, explicit12frozen IDs, batch4/CPU4,
using verified604pilot and per-batch checkpoints. Log
outputs/verification/video-mme-events-full.log. CPU-only continuation40408
SUBMITTED and Slurm-verified PENDING Dependency afterany:40036(unfulfilled),
4CPUs/12GB/6h, noGPU, immutable detached worktree
/scratch/yy5919/pyro-long-events-resume-65258e9 at aboveSHA. VerifiedWorkDir/Command
point there. Submit command from frozenworktree: sbatch --parsable
--dependency=afterany:40036 --output=/home/yy5919/PYRO-KD/logs/pyro-long-events-%j.out
hpc/long_events_resume.sbatch /home/yy5919/PYRO-KD/outputs/verification.
Logs: /home/yy5919/PYRO-KD/logs/pyro-long-events-40408.out; reports:
outputs/verification/long-events-resume-40408-attempt-0/. Pilot is revalidated
before scaling in the continuation, and final12video audit required. Queue
submission is not completion; continue monitoring. Separate pending40363 resumes
ASR/visual and does not write event caches. No duplicate active event writer.
Latest originaljobs: visual11/12logged through891; ASR729computed7361words,
619VADwindows,0rejected (independent729audit stillpending). No GPU inference.


Fourth long speech cache729 independently verified7537exit0:7361words across
619VADwindows,38575990speech samples,0rejected words. Evidence
outputs/verification/video-mme-asr-729-check.json; feature SHA256
3d419e61d58be3dfa03f144ac7fe4fdd851324145d42ea9e852f8349d3617904.
Detected language-window codes are unvalidated model estimates, not ground truth.
Verified ASR coverage now4/12. Livevisual72713/process102860 polled directly and
still active on final898;11complete visual metadata files, no restart. Full event
job27774 has begun616 with batch checkpoints;604 remains sole completed long event
cache. CPU continuation40363 and40408 still PENDING Dependency afterany40036.
Full visual audit remains next after72713terminalsuccess, not yet complete.


Long visual extraction72713exit0:12computed,0reused,0failed. Full independent
audit85259exit0 COMPLETE12/12:55549normalized512-D embeddings and6850shot boundaries,
current source/feature hashes, reconstructed novelty and local/memory ranking
budget/separation invariants passed. Evidence outputs/verification/
video-mme-visual-check.json and .log. No semantic/QA accuracy claim. Current status
header refreshed to distinguish completed preparation from pending experiments.
Long-QA resource inventory using unchanged dev5moments/min and4frames/moment
requires612–1012images/question; evidence video-mme-qa-resource-inventory.json.
This is not a frozen long budget or measured GPU limit. Determine a feasible
held-constant budget using development GPU smoke before opening held-out QA
content; do not silently truncate frames or change K across selectors.


Long QA actual-file integration check68288exit0:3tests passed under the data
venv using temporary synthetic Parquet/JSONL only (tests/test_video_mme_qa_cli.py).
The real CLI filters out an invalid answer outside frozen membership, preserves
missing-question denominators and nonzero incomplete exit status, records source
hashes, and rejects mismatched prediction provenance before attempting answer-file
access. No real benchmark answer values were opened; no experiment QA scores
were created. This resolves the previously untested Parquet filtering/CLI path.

Fifth long ASR cache753 audited45639exit0:6674words/150VADwindows,
32247997speech samples,0rejected words. Evidence
outputs/verification/video-mme-asr-753-check.json; feature hash
9fd67265d0680997dff21ee6da4af81bcdc0bb24a8a6160fdc5ec8c98186b7e9.
Per-window languages are model estimates and not semantic validation.
Long speech/event extraction continues; visual preparation remains verified12/12.


QA paired-analysis bridge implemented in scripts/export_qa_paired_metrics.py.
Exports all frozen per-question outcomes including missing/errors as0, with
video_id as cluster_id. Rejects differing question membership/source hashes,
video mappings, answer labels and inconsistent outcome/summary metrics. Adjacent
provenance records source report and export hashes. Existing paired cluster
bootstrap/Wilcoxon and predeclared-family Holm pipeline can consume it; equal-video
estimand explicit. Tests21299exit0:14passed across new exporter and existing paired
analysis. Fixtures only, no real QA statistical findings created. Documented
--unit cluster requirement, especially3questions/video for Video-MME. Current
CPU40036 elapsed5h05m of6h; continuations40363/40408 remain dependency-pending.
Long event616 progressed700/2646windows, longASR remains live, no duplicate jobs.


Actual CPU QA processor preflight84874exit0 on first EgoSchema U pilot question:
60images encoded into14075input tokens;16new-token allowance fits the pinned
Qwen2.5VL7B config's128000context. New scripts/check_qa_processor.py records source,
config and frame hashes plus tensor dimensions/bytes and explicit coverage;
local processor only, no model weights/GPU/QA predictions. Coverage1/10pilot,
not GPU memory feasibility or long-input capacity. Evidence
outputs/verification/egoschema-qa-processor-pilot.json/.log; reproduction command
in docs/REAL_DATA_SMOKE.md. Shared-template Transformers kwargs-placement warning
recorded; actual encoding passed. Initial read tried nonexistent check_vlm_inputs.py;
corrected discovery found debug_processor.py and the actual shared generation path.
Long ASR96112 and event115097 remain active; no duplicate writers or GPU request.


Processor warning audit85969exit0 inspected installed Transformers
ProcessorMixin.apply_chat_template source (outputs/verification/
processor-template-source.txt). The compatibility branch collects processing
kwargs from **kwargs, warns, then merges them into processor_kwargs before calling
self(..., **processor_kwargs). Thus padding=True is forwarded, not ignored;
the successful QA encoding is not evidence of a padding failure. No generation
code change was made. This resolves the warning concern without altering the
shared AD/QA recipe. Actual GPU feasibility remains untested. Live process check:
ASR96112 and events115097 active at~4CPUthreads each; event616 reached1100/2646
windows with batch checkpoints. CPU40036 has~51minutes remaining; dependent
CPU40363/40408 still queued. No restart, duplicate submission or GPU allocation.


Prepared real long visual hierarchy mechanics audit. VisualCueSelector gains
optional segment IDs/quotas for local and memory selection; defaults preserve
flat behavior. Memory and2s separation remain global across segments. New
scripts/audit_long_visual_hierarchy.py uses all12 verified visual caches, existing
flat audit and dev20–60s/quiet-floor1 settings; boundaries are visual shots only,
density is local visual candidate-score mass/second. Existing5/min budget is
mechanics-only, not final QA capacity. No question/answer access or heldout tuning.
Tests48463exit0:80passed/49subtests, including one-segment equivalence, cross-boundary
spacing and memory not reset; prior87238passed79before added memory regression.
Register next CPU4-or-less audit on40036 with output
outputs/verification/video-mme-visual-hierarchy.json; no feature cache writes.


Real long visual hierarchy mechanics3720exit0 COMPLETE12/12. Source24850af
(full SHA recorded in outputs/verification/video-mme-visual-hierarchy-source.txt).
Using dev20–60s bounds/floor1 and visual-shot boundaries yields1144segments;
local and global-memory hierarchy each select2314/2314requested moments with
0shortfall. Independent arithmetic audit verified contiguous full-duration
segments, <=60s spans, quota sums/floors, unique timestamp membership counts,
per-segment quota bounds and global2s spacing. Evidence
outputs/verification/video-mme-visual-hierarchy.json/.log and
video-mme-visual-hierarchy-check.json. This validates long-video mechanics only;
no QA content/labels inspected, no accuracy/quality gain claimed, and5/min remains
an exploratory resource-heavy budget rather than final GPU-validated protocol.
A/AV hierarchy still requires completed audio features and multilingual policy.


Sixth long ASR cache763 independently verified25909exit0:3959words/423VADwindows,
20860789speech samples,0rejected words. Evidence
outputs/verification/video-mme-asr-763-check.json; feature SHA256
 a22390e534a0325c0ec10bb5f83e43739082334571356cbc4e1bc78af5919c66.
Verified speech coverage6/12. Language estimates remain unvalidated.

Live jobs96112/115097 confirmed at~4CPUthreads each. Event616 has395committed
batch metadata records covering1580windows at this check (not a completed-video
audit). PendingCPU40363/40408 were revalidated through scontrol: both depend on
40036, with correct WorkDir/Command and clean immutable source checkouts at
9e20099a7dcae76a457da2222b18e64e8be15fe4 and
65258e9458db0b8a94c468eb682f5eba984b213f respectively. No duplicate submissions.
CPU40036 had~46minutes remaining; walltime is not extended by activity. Queued
work still needs monitoring through terminal status and final coverage checks.


Audio-native candidate preparation: official Qwen2.5Omni7B release API pinned
 ae9e1690543ffd5c0221dc27f79834d0294cba00. Processor-only download81765exit0 verified
13files/16175017bytes;22.36GBweights and speaker data not downloaded. Actual real
REFRAMED development audio+4image encoding69176exit0 produced2430tokens with both
modalities present; no model weights loaded. New config and check script plus
hash evidence retained. Runtime audit36991exit0/source-file inspection resolved
API difference: generation_mode='text', legacy return_audio via deprecated alias;
full wrapper still needs speaker data. Config warnings remain an actual-load
check, not silently declared safe. Agent metadata mistake: explicit signature
inspection initially named 'supports_return_audio'; corrected to 'declares' and
preserved original report, because **kwargs/decorators support the legacy option.
Audio-native ablation remains incomplete and unfrozen; GPU hold unchanged.


Live event checkpoint audit97214exit0 passed a snapshot of537committed batches /
2148contiguous prefix windows for longvideo616. Recomputed settings+currentWAV
configuration hash6d7a0fcb66d9694e2d1891d4424e34ec1bff6433b8f1341420abc1bb40f8e78d,
then verified every committed payload hash, index/count, float32 dimensions,
finite probabilities and nonzero embeddings through EventCheckpoints.load.
Evidence outputs/verification/video-mme-events-live-checkpoint-audit.json;
reproducer check_live_event_checkpoints.py beside it. Live writer was not changed
or restarted; later batches are explicitly outside this snapshot. This is not
complete616coverage or detector accuracy. Both longASR96112 and event115097 remain
live; CPU40036 had~41minutes left at the initial check, continuation jobs unchanged.


Prepared development-only per-class event/AD association analysis in
scripts/analyze_dev_event_alignment.py and src/evaluation/event_alignment.py.
Uses frozen10REFRAMEDvalidation clips, mergedUS AD starts (.5s merge), existing
AST probability>=.5/positive rise>=.1 and +/-2s AD proximity on the1sgrid.
Reports2x2counts, conditional proximity rates and phi per video/class, with
undefined correlations and support explicit; excludes predefined speech classes.
No test data, p-values, detector-accuracy interpretation or automatic weight/sign
tuning. Tests12801exit0:6passed including known+1/-1/undefined correlations and
inclusive boundaries. Next actual CPU diagnostic uses existing cached events and
output outputs/verification/reframed-dev-event-alignment.json. Initial exploratory
file reads guessed two nonexistent fused-audit names; corrected via rg discovery
to audit_fused_selectors.py before implementing the missing association path.


Development event/AD association34065exit0 COMPLETE10clips/1336windows/517included
classes, source5d0833b. Only11classes have detected onset windows at the existing
probability.5/rise.1 thresholds;10of those occur in just1clip. There are51class-window
detections total. Music appears in all10clips (31windows), equal-video mean phi
0.017814021769080805; this is descriptive, not a significance or causal result.
Sparse support does not justify fitting unconstrained per-class signs from these
clips. No thresholds/weights changed and human detector validation is still needed.
Evidence outputs/verification/reframed-dev-event-alignment.json/.log and independent
count/phi arithmetic audit reframed-dev-event-alignment-check.json. Preserves all
classes and undefined correlations, avoiding selective reporting of favorable
classes. Actual long event616 now2400/2646windows; longASR continues.


Second long event video616 completed2646windows; independent verifier34564exit0
passed pinned model/sourceWAV/array hashes, exact time grid, finite527-class
probabilities, normalized512-D embeddings, RMS and reconstructed cues. Evidence
outputs/verification/video-mme-events-616-check.json (1selected of12, explicit
partialcoverage). Combined independently verified event coverage now604+616:
4478windows,2/12videos. Live handle27774/process115097 moved to710 (1982windows);
no restart or duplicate writer. ProcessorRSS dropped after completing616, and
saved batch checkpoints are retained. Original ASR96112 remains active.
Actual full event extraction is still incomplete; final12video audit required.


Registered long-video spectral-flux onset preparation onCPU40036 using existing
scripts/cache_onset_features.py and unchanged SETTINGS(version1,16kmono,hop160,
FFT512,64mels,centerFalse,1smax bins,per-video-max normalization). Input is the
verified12video WAV cache; no event/ASR dependency and distinct onset files avoid
writer overlap. Command in_env ro 'OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1
MKL_NUM_THREADS=1 /scratch/yy5919/pyro-audio-env/bin/python -u
scripts/cache_onset_features.py --cache-dir /scratch/yy5919/pyro-kd-derived/cache/video-mme'.
Log outputs/verification/video-mme-onsets.log. No parameter tuning or label access.
Independent provenance/grid/normalization audit follows completion.


Long spectral-flux onset preparation76263exit0 COMPLETE12/12. Independent audit
66784exit0 verified27780one-second windows across all12: current source/WAV/config
hashes, exact grid, finite nonnegative values and per-video-max normalization.
Full raw-WAV recomputation for604 matched stored strength/normalized arrays;
other11received provenance/numerical checks without redundant reinference.
Evidence outputs/verification/video-mme-onsets-check.json and video-mme-onsets.log;
new reusable scripts/verify_onset_cache.py supports explicit recompute IDs.
Existing onset tests11596exit0:3passed. Same development parameters, no threshold
or normalization tuning, no QA labels inspected. This is spectral-flux consistency,
not human onset accuracy. Learned event/ASR stillactive; event710 now200/1982windows.

Registered full pinned Qwen2.5Omni7B asset download onCPU40036 after successful actual multimodal processor check and runtime API review. Same ae9e1690543ffd5c0221dc27f79834d0294cba00 revision; includes five model shards and speaker data required by the installed wrapper. Reuses processor snapshot. No GPU job, paid API or model instantiation. Config configs/qwen25_omni_model.json; log outputs/verification/qwen25-omni-model-download.log and report qwen25-omni-model-download.json. Intended to make later GPU ablation smoke offline; config warnings and real GPU feasibility remain unresolved.


Full Qwen2.5Omni7B download3293exit0 and independent file/index verifier63745exit0
COMPLETE:19assets/22379295753bytes,5safetensors shards/2448tensor names, pinned
 ae9e1690543ffd5c0221dc27f79834d0294cba00. Includes required speaker file. Every
large file matched upstream LFS SHA256 and plain files matched pinned Git blob
SHA1; all files also have recorded SHA256. Safetensors header keys exactly match
index assignments. Evidence outputs/verification/qwen25-omni-model-check.json/.log,
model-download.json/.log and upstream-file-identities.json. New reusable verifier
scripts/verify_pinned_model_files.py streams8MBchunks without instantiating a model.
Full assets now staged; earlier processor-only status is superseded. GPU loading,
configuration-warning validation and audio-native inference remain untested.

Seventh longASR802 verified32201exit0:7964words/711VADwindows,
36587194speech samples,0rejected. Feature SHA256
 e4e22eccefd41daef9dd51e58da7811dd9771154d248c640afd28d4896d89747;
evidence outputs/verification/video-mme-asr-802-check.json. Language codes remain
model estimates. Long speech/event extraction continues onCPU40036; no GPU request.


Native waveform development-input adapter validated: scripts/attach_native_audio.py
and src/generation/native_audio.py preserve existing selections and attach mono16k
waveform spans centered on frame_center_time +/-15s. Full WAV and decoded slice
SHA256, sample boundaries, declared seconds/rate and finite samples are checked.
Actual U snapped development export:76moments/9videos, all original fields including
frames and word budgets unchanged. Output /scratch/yy5919/pyro-kd-derived/moments/reframed_U_snapped_placed_event_dev_exploratory_native.jsonl.
Evidence outputs/verification/reframed-native-input-check.json records input/output,
source and split hashes; focused tests5passed. First independent audit incorrectly
assumed a strict480000sample ceiling: outward floor/ceil rounding adds one sample
for fractional centers. Corrected audit checks exact floor/ceil and <2sample rounding
tolerance; no data rewritten. No model inference or native runner integration yet.
Verified running ASR96112/event115097 on40036; continuation40363/40408 pending
Dependency. GPU hold remains. Full proposal experiments are incomplete.


Added explicit --backend omni to scripts/run_ad_vlm.py. It requires batch-size1,
no quantization and no reasoning-strength; native context requires this backend.
Existing frame/text modes remain available on the same Omni backend for later
controlled comparisons. Native rows validate full WAV/slice hashes before resume
and again before generation, reject transcript/timeline mixtures and mismatched
visual centers. Output records retain exact native sample metadata; provenance
includes backend and full input JSON. Shared error/truncation/recovery paths remain.
src/generation/omni.py uses local pinned assets, disables talker after model load,
requests generation_mode=text with thinker-specific decoding arguments, and
rejects unexpected audio-output tuples. GPU loading/inference remains untested.
Tests39460exit0:41passed (backend mocked-model contract, native spans, generation
truncation/provenance). Tests75806exit0:16passed+11subtests for recovery/audio
context/prior. Actual native CLI dry-run passed, no prediction file/model load.
Actual shared native processor check29796exit0 encoded the first real frozen dev
U moment:2390tokens, image+audio tensors present. Scope1/76moments only; noGPU
memory or quality claim. Evidence outputs/verification/reframed-native-ad-processor-check.json/.log
and native-ad-dry-run.log. Existing talker config token warnings persist and must
be evaluated in the later real GPU smoke. No GPU submission or paid API usage.


Long event710 independently verified85596exit0:1982windows, current source/model/
configuration/array checks passed. Evidence video-mme-events-710-check.json/.log,
explicit1/12auditcoverage; cumulative604+616+710verified6460windows. Live extraction
has advanced to729. ASR remains7/12verified. Scheduler recheck:40036RUNNING,
ends04:22:47UTC;40363and40408PENDINGafterany40036. Both detached source worktrees
remain clean at9e20099a7dcae76a457da2222b18e64e8be15fe4 and65258e9458db0b8a94c468eb682f5eba984b213f.
No duplicate jobs submitted; completion and full audits still pending.
Native AD runner integration test2716exit0 passed with temporary synthetic
waveform/frame and mocked inference. It verifies waveform delivery, exact output
metadata, settled resume without loading and rejection of changed WAV even when
output is already settled. No actual AD prediction generated. Added exact later
same-Omni none/native GPU smoke commands to docs/REAL_DATA_SMOKE.md; unexecuted
under GPU hold. This does not resolve GPU memory, configuration warnings or quality.


Eighth long speech808 verified64715exit0:5697words/516VADwindows,25454386speech
samples,0rejected. Feature SHA c4b0ac94b781dd7f8851c68ba5323114b8c7e0f157b43e5e6ec5455b155befa0.
Evidence outputs/verification/video-mme-asr-808-check.json/.log. Detected language
codes remain model estimates. Full long speech/event processing continues.

Core-matrix readiness inspection exposed a missing REFRAMED generated-AD evaluator:
existing evaluate_ad.py only loaded MTG. Added explicit dataset adapter using US
cues merged at<=0.5s (same placement diagnostic policy), movie cluster IDs, frozen
membership filtering and explicit missing-selected-video accounting. CLI records
selection hash and reference policy. Mixed backend/audio/selection/fingerprint
values are rejected; absent audio mode is treated as legacy for mixture checks.
Synthetic fixture integration checks merged reference text, missing-video FN,
shared movie IDs, out-of-membership rejection and mixed audio rejection. Existing
MTG schema assertions updated for explicit reference and coverage fields; first
run failed one old exact-schema assertion, then corrected the expected schema.
Final focused tests37644:29passed+27subtests. No real AD outputs or metrics created;
semantic-model freezes, OpenCLIP, rendered overlap and calibrated judges remain.
Documentation: src/evaluation/README.md. No GPU/API action.


Added REFRAMED temporal paired export scripts/export_ad_paired_metrics.py. It
requires complete frozen selected-video coverage, selection hashes and generation
fingerprints; compares perclip reference hashes, movie IDs and temporal settings;
recomputes P/R/F1 from counts. Missing predictions remain0withreference, empty
reference/prediction retainsnull. Exports clip examples/movie clusters, explicit
errors/generatedcounts and source/output hash sidecar. Existing analyze_paired
--unit cluster estimates equal-movie mean within-movie clip metrics. No pooled-F1
or semantic-quality interpretation. evaluate_ad.py now hashes input bytes and
actual merged reference timestamp/text pairs. Final corrected tests66234exit0:38passed+27subtests,
including grouped clips, missing-predictionzero, mismatched references/membership/
protocol/counts and missing generation evidence. Fixtures only, no real scores.
Full matrix, tuned/equal-Kprotocol and semantic/quality/human evaluation remain.


Correction to paired-export validation history: first test57034 exited1 with
37passed/1failed. Exporter F1 reconstruction incorrectly omitted FN in its
denominator. The missing-prediction/cluster fixture detected it. Fixed formula
to2TP/(2TP+FP+FN); rerun66234exit0 passed38tests+27subtests. Agent prematurely
wrote a passing test entry in27cd883 before inspecting the completed result;
that claim is corrected above and this history preserves the mistake. No real
evaluation export or scientific result was produced with the faulty version.


Pre-walltime checkpoint2026-11-23T04:18UTC: ninth longASR857 independently
verified96370exit0 (631words/116VADwindows/2639997speech samples/0rejected).
Feature SHA8e743e1770da419ba2de7077a4401027cb7531f7cddce57efb49fc0739cf3454;
evidence video-mme-asr-857-check.json/.log. Language labels are estimates only.
Current event729 committed-prefix audit89349exit0 verified176batches/704windows
against current WAV SHA and independently audited710recipe, including payload
hashes, contiguous indices, shapes, finite values and valid probabilities.
Digestaf10f8c942e5f352e8d2846e71f2fab9a99294bfef9f644f2fe7bc7fe9ab732f.
Evidence video-mme-events-729-checkpoint-audit.json and reproducible audit.py.
This is a prefix snapshot, not completed729features; completed event coverage
remains3/12. ASR is atomic pervideo, so its in-progress video may need repeating.
No duplicate writer or job restart.40036ends04:22:47UTC regardless ofactivity;
40363and40408remain dependency-pending withclean immutable worktrees. They still
require observed completion/full audits. GPUhold remains; finalexperimentsnotdone.


Core matrix input inventory now explicitly enumerates all20U/V/A/AV/OxP0/P1xopen/API
cells while retaining8exploratory selector variants without choosing finalVmemory
or event signs. Script scripts/audit_core_matrix_inputs.py validates frozen dev
membership, unique moments, four existing frames/timestamps, positive word budgets
and within-video requestedK consistency; records input/split hashes and perclipcounts.
Actual run passed:U76moments/9videos,O85/10,Vlocal/Vmemory/Atrigger/Asuppress/
AVtrigger/AVsuppress97/10each. Current files are inventoried by hash; they need not
represent newer exploratory recipes elsewhere. This confirms unequal coverage and
no final equal-K protocol, including one absent Uclip. No generated output scored.
Evidence outputs/verification/core-matrix-development-inputs-v2.json. Earlier
inventory version hardcoded zero completed experiments; revised tool explicitly
does not inspect execution outputs, avoiding a future stale completion claim.
NativeOmni remains ablation, not API substitution. GPUhold/APIcap/humangates remain.


2026-11-23T13:38UTC resumed in user-provided GPU session40880 onb-19-15,
A100-SXM4-40GB,24h limit ending2026-11-24T13:26:29UTC. GPU initially0MiB/0%
withno compute processes. User condition (finish transfers then obtainGPU) is now
satisfied; proceeding with authorized small GPU smoke, not fullmatrix scaling.
OldCPU40036TIMEOUT as scheduled. Continuations40363COMPLETED0:0 in36m36s and
40408COMPLETED0:0 in3h37m32s. All step statuses0. Final speech12/12,visual12/12,
events12/12auditscomplete; reports underoutputs/verification/long-cpu-resume-40363-attempt-0
andlong-events-resume-40408-attempt-0. No detector/transcriptionaccuracy claim.
Registered first real Qwen2.5VL7B smoke:4MTGdevmoments,P0,noaudio context,batch1,
128newtokens,temp0,bf16,sdpa,pinnedcc594898137f460bfe9f0759e9844b3ce807cfb5.
Output outputs/real-data-smoke/qwen25-mtg-U-P0-none.jsonl and associated metadata;
logoutputs/real-data-smoke/qwen25-mtg-U-P0-none.log. Requires actual completion,
output/frame/wordbudget inspection and evaluation before scaling. No duplicate
GPU writer. Separate CPU user session40879 remains untouched.


First real GPU smoke51403exit0 complete on40880 using source3811351:4MTGmoments
fromonevideo. Load191.8s,generate8.0s,peakallocated16.99GB,0tokenlimitfailures.
Actual output audit found2/4word-budgetviolations (3>2 and6>5), so this did NOT
pass the smoke gate despite old runner n_ok4. Agent frame review also notes
unsupported left-hand/pressing detail in guitar caption; rawdecodesunchanged.
Reviewoutputs/real-data-smoke/qwen25-mtg-U-P0-none.review.json andframes.jpg.
Evaluation61469exit1 expectedpartial:2valid/2budgetfailures,0temporalpairs among
limited4inputslotsvs5references; no semanticmodels loaded/scored. Not a selector
performance claim. Added whitespace-wordbudget failure accounting to runner and
shared outputreader, including legacyunflaggedviolations, retainingfullrawdecode.
Clarifiedstrict wordlimit insharedP0/P1prompt; no post-hoc truncation. Tests8622
firstfailed mockedmodelGPUstatreset onnewGPUhost; fixedtestCUDAisolation. Final
73407exit0:61passed+38subtests. Registered repeat same4inputs innewoutput
qwen25-mtg-U-P0-none-budget-v2.jsonl, same model/settings except sharedpromptfix.
No fullmatrix scaling. Ten-videoREFRAMEDsmoke input also preparedfromearliest
existingpositivebudgetUrowper frozenclip, provenance reframed-ten-video-smoke-inputs.json;
notfinalselectorprotocol andnotyetinferred.


Second GPU smoke47071exit0 completed source8b54a64:4decoded/3withinbudget/1budget
failure,0token-limitfailures. Actualattemptmetadata load14.0s,generate6.0s,
peakGPU17.0GB. Remaining2wordcase output"boat on water." has3words;
newrunner correctly records error and preserves fullrawdecode. Other outputs:
"Person plays guitar with fingers on strings."(7/23),"man closes eyes mouth open hand on face."(8/11),
"man looks at camera."(4/5). Agent frame inspection supports broadactions but
is not humanqualityvalidation. No all-clear smokeclaim/gridexpansion. Nextneeded
is bounded shared word-budget repair with allattempts/cost preserved, then
actualten-video development evaluation including semantic andrenderedoverlap.


Registered optional bounded word-budget repair (--word-budget-retries1), shared
acrossP0/P1 andVLM/Omni backends. One extra same-model/frame-context call only for
nontruncatedoverbudgetoutput; bothprompts/rawdrafts/token counts/errors retained in
generation_attempts, total_new_tokens/generation_calls/repair_latency_s recorded.
Failure remainsfailure, no stringtruncation/no unboundedloop. Policy enters run
fingerprint. Tests46861exit0:12passed covering repair/failure/exception/no-repair,
native runner and provenance. Registered third same4MTGsmoke on40880,output
outputs/real-data-smoke/qwen25-mtg-U-P0-none-budget-repair.jsonl, batch1/temp0/
128tokens/P0/none/sdpa/quantnone/pinnedQwen. No gridscaling. Requires actualinspection.


Bounded-repair GPU smoke71886exit0:3/4valid,1remainingwordbudgetfailure. Five
modelcalls acrossfourinputs; failed2wordcase repaired to"boat on water" (still3),
bothdrafts preserved. No further prompt adjustment based on this individualcase.
Registered ten-video REFRAMED development diagnostic using same fixedsharedprompt/
one-repairpolicy, source9a5c834, earliestexistingpositivebudgetUinputper frozenclip,
P0/none/batch1/temp0/128tokens/sdpa/bf16/Qwenpinnedcc5948. Output
outputs/real-data-smoke/qwen25-reframed-ten-U-P0-repair.jsonl and .log. Purpose:
measure failure prevalence and exercise actualevaluationacross10clips; not final
selector/coregridscaling orclaim ofwordbudgetgatepassing. Failures remain denominator
coverage. Neededsemantic/renderedoverlapverification remains beforelargerexperiments.


2026-11-23 new user sessions40891GPU/b-19-17 and40892CPU/b-10-1; prior40880
CANCELLED after24m32s. Generation13177hadcompleted10REFRAMEDrows, but evaluation8112
handle/process disappeared withno report. Restarted evaluation28180 CPU-only after
verifying no livewriter; logs evaluation-resume.log. User explicitly requests
active task management and useful GPUutilization. Registered independent first
EgoSchemaGPUsmoke:one frozenpilot Uquestion/60frames/16maxnewtokens/temp0/pinned
Qwen2.5VL7B/sdpa/bf16 on40891. Outputoutputs/real-data-smoke/egoschema-U-first.jsonl.
No labelsprovidedtoinference; evaluationfollows. No duplicateGPUworker. CPU AD
semantic evaluation overlapsGPU QA. Fullgridstillheldpendinggates.


First EgoSchema GPU smoke37075 completed one60framequestion withvalid"A"answer,
2newtokens,noerror,peakallocated19862140928bytes(~19.9GB). Validinference/input
parsing, not yet accuracyclaim. Registered full10question U pilot innewfile
outputs/real-data-smoke/egoschema-U-ten.jsonl under currentruntimecode; previous
first-questionoutput preserved (sourcefingerprint changed by TTSimplementation).
SamepinnedQwen/bf16/sdpa/16tokens/temp0. Noheldout490inference orlabelinspection.
Piper1.8.0 isolatedTTSenv installed, voiceen_US-ljspeech-medium pinned toe9c8a39444359388adaa94255becb37a5a2b8e51,
3assets63536868bytes verifiedupstreamLFS/Git hashes. Modelcard sayspublicdomain
dataset; noLessaclicenseformsubmitted. Configconfigs/piper_tts.json, noise0,
normalrate1.0,CPU2threads. Realrender21213exit0:9validREFRAMEDdescriptionsrendered,
1budgetfailureexcludedandexplicitlyreported. WAVPCMnonempty/mono/16bit/22050Hz
validated. Evidence /scratch/yy5919/pyro-kd-derived/tts/reframed-ten-smoke/report.json.
Actual1.2xoverlapaudit37593running whileGPUQAproceeds. TTSqualitynotyetlistened.
Sources: https://github.com/OHF-Voice/piper1-gpl/blob/main/docs/API_PYTHON.md and
https://huggingface.co/rhasspy/piper-voices/blob/e9c8a39444359388adaa94255becb37a5a2b8e51/en/en_US/ljspeech/medium/MODEL_CARD.


Ten-question EgoSchema U pilot49979completed10/10validanswers,noerrors. Actual
accuracy evaluation now running againstfrozenpilotlabels only; noheldoutlabels.
Registered next bounded10question V_local thenV_memory pilot GPUqueue on40891,
samefrozeninputs/60frames/Qwen/bf16/sdpa/16tokens/temp0; outputs
outputs/real-data-smoke/egoschema-V_local-ten.jsonl andegoschema-V_memory-ten.jsonl.
One GPUwriter atatime. Stopsqueueif inferencefails. CPUevaluation overlaps.
REFRAMEDsemantic evaluation28180exit1 dueonebudgetfailure:9valid,3matches,
BERTScoreF1.85656785965over3pairs,cosine.26599191647. Limitedone-moment-per-clip
coverage against110references meansnotfinalselectorquality. Renderedoverlap37593
exit0:9WAVs,48.7391875renderedseconds,8fit,one0.052875sunionVAD/Whispercollision,
0out-of-video. Failedonewordcaseexcludedexplicitly. TTSintelligibilitynotyetreviewed.


REFRAMED ten-clip visual inspection savedreframed-ten.visual-review.json plus2frame
contact sheets. Broadscenecontent oftenvisible, but repetition and unsupported
motion/details remain; no invented calibrated ratings. This is agentinspection,
not60humanpeercomparisons. Actualsemanticmetrics9valid/1budgetfailure,3matches,
BERTScoreF1.85657 andsentencecosine.26599 over3pairs only. Rendered overlap8/9fit,
0.052875scollision; one excludedbudgetfailure keptincoverage. Nofinalbenchmarkclaim.
EgoSchema U tenpilot accuracy58284exit0:10/10, independently checked frozenIDs,
strictletter mapping anddirectpilotanswercomparisons. Predictions vary acrossA-E;
top-levelgoldfields absent ininput. Evidence egoschema-U-ten.independent-check.json.
Noheldout490accuracy. V_localcompleted10answers; CPUevaluationlaunched. V_memory
GPUqueue44349continues sequentially, no simultaneousGPUmodels. Userrequest for
usefulGPUactivity handled with independentQAwork whileCPUAD/TTS auditsrun.

## 2026-11-23 isolated integration checkpoint

Resume worktree: /scratch/yy5919/pyro-revision15-resume; branch
work/revision15-resume-20261123 combines local 7bfe411 with PR22 head8148fe0.
Original checkout preserved. Four runner safety fixes and affected integration
checks pass (36 tests,11subtests); pre-final full suite789passed/1failed/3skipped,
remaining failure then fixed and rerun. Draft20-cell plan and detailed audit:
docs/RESUME_2026_11_23.md. No paid calls or formal grid launches. API authorization,
human annotation gates, final protocol and cache adapter remain pending.
