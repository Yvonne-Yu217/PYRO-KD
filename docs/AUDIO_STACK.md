# Three-stream audio preparation

The current container already provides Silero VAD and Transformers. A separate
system-site-packages venv at `/scratch/yy5919/pyro-audio-env` adds
`faster-whisper==1.2.1`; the shared overlay is unchanged. Installed dependencies
and download logs are under `outputs/verification/audio-*`.

Pinned model repositories/revisions and expected file sizes are in
`configs/audio_models.json`. Downloads total about 1.45GB (AST's duplicate
PyTorch weights are omitted in favor of safetensors). The official
[faster-whisper documentation](https://github.com/SYSTRAN/faster-whisper)
describes CPU int8 and word timestamps; the
[AST model card](https://huggingface.co/MIT/ast-finetuned-audioset-10-10-0.4593)
identifies its AudioSet classifier, and the
[CLAP documentation](https://huggingface.co/docs/transformers/model_doc/clap)
describes projected audio embeddings.

Whisper-small.en is the initial frozen English speech checkpoint; CPU int8 is
recorded explicitly. AST and CLAP run float32 in the CPU check. This is a
preparation choice, not a quality comparison or a passed human-label gate.

```bash
source hpc/config.sh
in_env ro 'python -m venv --system-site-packages /scratch/yy5919/pyro-audio-env && /scratch/yy5919/pyro-audio-env/bin/python -m pip install faster-whisper==1.2.1'
in_env ro 'HF_HUB_OFFLINE=0 python scripts/download_audio_models.py --cache-dir /scratch/yy5919/hf_cache/hub --report outputs/verification/audio-model-download.json'
in_env ro 'OMP_NUM_THREADS=4 /scratch/yy5919/pyro-audio-env/bin/python scripts/check_audio_models.py --cache-dir /scratch/yy5919/pyro-kd-derived/cache/reframed --model-report outputs/verification/audio-model-download.json --report outputs/verification/audio-model-offline-check.json'
```

Actual offline CPU checks passed: first 10seconds of development clip
l5lLQICPOcnb produced finite AST 527-class scores and CLAP 512-dimensional
embeddings. CLAP embeddings are not unit-normalized by this audio-only model;
normalize or use explicit cosine distance when computing novelty. Loading the
audio-only CLAP class reports unused text-model weights; no missing audio
weights were reported. For Whisper, development clip CxWby8RuUwU6's VAD interval
[0.5,3.3] produced 10 word timestamps within its boundaries. No non-VAD samples
were passed to Whisper. Evidence: audio-model-offline-check.json/.log.

The sections below record subsequent development-cache implementation.
Remaining work includes timeline/selector integration and human-labeled
detector evaluation; the compatibility check alone proves neither.

## VAD-gated speech cache

Implemented scripts/cache_speech_features.py and src/utils/speech_features.py.
Speech intervals are unioned only when they overlap or touch; non-speech gaps
are never bridged. Each interval is split into at most28-second windows,
with inward sample rounding. Whisper uses English, beam5, temperature0,
word timestamps, and no conditioning on prior windows. Predictions outside
a supplied window are kept as rejected diagnostics rather than moved/clamped.
Content hashes cover the exact WAV bytes, VAD config and intervals, pinned
model revision, backend versions and decoding settings. Caches publish after
checking that audio/VAD did not change during the run.

```bash
source hpc/config.sh
in_env ro 'OMP_NUM_THREADS=4 /scratch/yy5919/pyro-audio-env/bin/python scripts/cache_speech_features.py --cache-dir /scratch/yy5919/pyro-kd-derived/cache/reframed --model-report outputs/verification/audio-model-download.json'
```

Completed all10 REFRAMED development caches with1645 retained words and no
out-of-window words. One clip has zero VAD-positive intervals and therefore
zero words; this is not evidence that it truly contains no speech. Independent
checks verified every saved word/window lies within a VAD interval and each
WAV hash matches. A repeat run reused all10 caches. Gating tests:4 passed,
4 subtests. Evidence: outputs/verification/reframed-dev-speech{,-reuse}.log,
reframed-dev-speech-check.json, and pytest-speech-features.log. This does not
measure ASR accuracy or validate the VAD detector; human checks remain pending.

## AudioSet/CLAP event tracks

scripts/cache_event_features.py caches ten-second centered audio windows at
one-second hops, padding only beyond the clip boundaries with zeros. AST
produces527 sigmoid class scores; these are uncalibrated window-classifier
outputs, not precise frame-level event labels. CLAP audio embeddings are
unit-normalized; novelty compares each embedding to the mean of the previous
five normalized embeddings. Positive class-score differences produce onset
cues. One-second RMS windows supply positive dB changes. First differences at
the first time point are zero because there is no previous observation.

All class scores are retained. Onset scoring excludes the AudioSet Speech
class and its descendants, Whispering, and Silence (10 labels), retaining
laughter and music. The class hierarchy was checked against the official
[AudioSet ontology](https://github.com/audioset/ontology/blob/master/ontology.json).
Window overlap can smear timing; detector P/R against actual human event
labels must still establish whether this representation is usable at the
proposal's temporal tolerance. Do not report these scores as confirmed events.

```bash
source hpc/config.sh
in_env ro 'OMP_NUM_THREADS=4 /scratch/yy5919/pyro-audio-env/bin/python scripts/cache_event_features.py --cache-dir /scratch/yy5919/pyro-kd-derived/cache/reframed --model-report outputs/verification/audio-model-download.json'
```

Each video gets event_features.npz and event_features.json, with WAV hash,
model revisions, dependency versions, window settings, label order and NPZ
hash. Interrupted or mismatched array/metadata pairs do not pass cache reuse.
Pilot CxWby8RuUwU6 completed74 windows and passed independent shape/finite,
normalization, class-mask and NPZ-hash checks. Three numerical tests passed.
Full ten-clip CPU caching subsequently completed and passed the audit below.

Full-cache audit (run after extraction, or as an explicitly incomplete check):

```bash
source hpc/config.sh
in_env ro 'python scripts/verify_event_cache.py --cache-dir /scratch/yy5919/pyro-kd-derived/cache/reframed --model-report outputs/verification/audio-model-download.json --report outputs/verification/reframed-dev-events-check.json'
```

The audit uses frozen development IDs, checks current source/WAV hashes,
model revisions and vocabulary, one-second coverage, normalized embeddings,
finite/range-valid scores, NPZ/config hashes, and recomputes RMS and derived
cues. It is not model re-inference or an accuracy test. Missing unfinished
files make complete=false and exit1; do not interpret that as extraction
failure while the registered extraction process remains live.

Completion checkpoint (2026-11-23): extraction26247 exited0. Audit72375
verified all10 development clips and1336 one-second time points. Repeat66969
reused all10 caches. Evidence: outputs/verification/reframed-dev-events-check.json,
reframed-dev-events-check.log, reframed-dev-events-reuse.log. These checks prove
artifact consistency and coverage, not event detection accuracy. VAD/ASR/event
features are now present for this development subset; downstream integration,
human detector labels and full experiments remain outstanding.

Speech-cue development preparation

`src/utils/speech_cues.py` derives separate lexical deictic and adjacent-transcript semantic-change tracks plus long-silence opportunities from validated VAD gaps. Chunks stop at ASR gate boundaries, sentence punctuation or24words; they never bridge separate transcription windows. The phrase list is frozen in code (look at this/that, over there, watch this/that); matches can be figurative and are not labels. Semantic distance is `(1-cosine)/2` between successive chunks, first chunk zero. Short VAD chunks can make this a noisy proxy for topic change. Speaker-change diarization remains unimplemented.

CPU preparation uses the existing pinned sentence-transformers/all-mpnet-base-v2 revision e8c3b32edf5434bc2275fc9bab85f82640a19130 with four threads/batch16; model files and library versions are fingerprinted in the output. Run inside the read-only environment:
`OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 python -u scripts/prepare_speech_cues.py --cache-dir /scratch/yy5919/pyro-kd-derived/cache/reframed --min-silence 5 --silence-scale 30 --output outputs/verification/reframed-dev-speech-cues.json`.
Use a new output path when repeating; records are not overwritten. The5s threshold and30s scale are exploratory, not dev-tuned. No ASR words does not imply silence: only current VAD gaps define silence cues.

Ten development clips produced294 chunks,18long-silence cues and0phrase-matching chunks. Output includes embeddings for rechecking semantic differences. Audit outputs/verification/reframed-dev-speech-cues-check.json verified frozen IDs, ASR source hashes, gate bounds,768-dimensional normalized embeddings and reconstructed cosine changes. This is preparation, not detector accuracy or final audio-selector scoring. A/AV integration and speech/non-speech ablations remain outstanding.

Spectral-onset cue checkpoint

`cache_onset_features.py` computes librosa spectral-flux strength from16kHzmono WAV with hop160/n_fft512/n_mels64/centerFalse, takes the maximum within each half-open1s bin, and records raw strength plus video-maximum normalization (silent tracks remain0). Separate onset_features.json records source WAV hash, settings and library versions; AST/CLAP outputs remain unchanged. This is an uncalibrated full-audio cue, including possible speech transients, not a validated non-speech event label.

All10REFRAMEDdev clips cached:1336windows. Independent audit verified source hashes, event-grid alignment and normalization. Nine unit tests passed, including actual silent-wave extraction. `audit_fused_selectors.py --include-onsets` explicitly adds normalized strength under the global trigger/suppress sign; omission preserves the earlier recipe. New source hashes are included for downstream export validation. Exploratory totals97/99/97/98 of111requested for Atrigger/Asuppress/AVtrigger/AVsuppress. Selected event sets changed in9/8/9/9clips; no accuracy or superiority claim. Outputs outputs/verification/reframed-dev-onsets-check.json and reframed-dev-fused-with-onsets-v2.json. Earlier reports retained. Threshold/weight tuning and fixed-K resolution remain pending.

Multilingual preparation support (not yet run on long videos)

`cache_speech_features.py --model-id <pinned multilingual repo in model report>
--language auto` now passes `language=None` separately for each existing
VAD-positive window and records detected language/probability per window.
Speech stays in the detected language; there is no translation or concatenation
across silent gaps. A forced language code is also supported. Auto/non-English
requests fail with English-only repo names or loaded English-only weights.
Language policy enters the cache fingerprint; default small.en/en settings and
existing English feature schema are preserved.

Local faster-whisper implementation was inspected to confirm its optional
language argument and returned language metadata. Unit tests cover two different
window languages, forced-code pass-through, invalid detection probabilities and
unchanged VAD isolation. This is plumbing, not validated multilingual accuracy:
short speech windows can be poor language-detection evidence. Multilingual
weights still need a frozen revision and an actual offline compatibility check;
no Video-MME ASR has been run. The English-only MPNet semantic-shift and lexical
cue policy also requires a multilingual decision before long-form audio-aware
selection. Do not silently describe current English-only caches as multilingual.

Multilingual model cached and offline compatibility passed

Pinned Systran/faster-whisper-small revision536b0662742c02347bc0e980a01041f333bce120
in configs/multilingual_speech_model.json, downloaded486214370bytes/5files.
Official model card https://huggingface.co/Systran/faster-whisper-small identifies
this as the CTranslate2 conversion of openai/whisper-small. The separate
English-only snapshot remains unchanged. Offline CPU int8 test used the fixed
REFRAMED development clip CxWby8RuUwU6, VAD interval0.5–3.3s:10words, zero rejected
out-of-gate words, detected en probability0.9944057465. This English excerpt checks
loading, word timing and language metadata only; non-English ASR accuracy is not
validated. All five downloaded files have SHA256 recorded in the check report.

Reproduce with `scripts/check_multilingual_speech.py --cache-dir
/scratch/yy5919/pyro-kd-derived/cache/reframed --model-report
outputs/verification/multilingual-speech-download.json --output NEW_CHECK.json`
using /scratch/yy5919/pyro-audio-env/bin/python inside the container, OMP_NUM_THREADS=4.
Artifacts multilingual-speech-{download,offline-check}.json/log are in
outputs/verification. No long held-out ASR output or QA answers inspected.

## Human detector scoring interface (labels still pending)

`src/evaluation/event_detection.py` and `scripts/evaluate_event_detector.py`
provide window-level class-presence precision/recall/F1, per class, per video,
and pooled across videos. This is a proposed scoring convention to freeze before
collecting/scoring labels, not an onset-localization metric or a completed human
gate. A class is present when a labeled interval overlaps a tagger window for
positive duration; exact boundary contact does not count. The real, unpadded
window must be fully covered by reviewed intervals. Any overlap with an unknown
interval excludes that window. Unreviewed audio is never a negative label.
Overlapping 10-second windows are correlated; do not treat them as independent
samples for confidence intervals. Undefined precision/recall is JSON null.

The annotation JSON must have `origin: "human"` and nonempty `videos`. Each video
has `video_id`, a nonempty `annotator_id`, `audio_sha256` (the listened-to cached
WAV), `duration`, explicit `reviewed_classes`, `reviewed_intervals` as [start,end]
pairs, optional `unknown_intervals`, and `events` objects with `label`, `start`,
`end`. Reviewing an interval means checking every declared class throughout it;
an empty event list represents reviewed negatives, not unfinished annotation.
Annotators should listen without model suggestions. An ID records attribution;
the evaluator cannot authenticate that the labels were actually made by a human.

The separate protocol JSON specifies `split: "development"`, the exact
`video_ids`, `models` mapping of repository IDs to pinned revisions,
`window_seconds: 10.0`, `hop_seconds: 1.0`, and explicit per-class `thresholds`.
Use the exact AST class names from the cache. Source selection must be checked
against official splits before freezing this protocol; a split field alone is
not proof of development membership. No actual annotation set or thresholds have
been frozen by adding this interface. About ten ten-minute movie-audio clips and
real human labels are still required by the proposal.

Run after annotations and a protocol exist:

```bash
python scripts/evaluate_event_detector.py --annotations <human-json> \
  --protocol <frozen-protocol-json> --cache-dir <event-cache-root> \
  --output <new-score-json>
```

The CLI requires all protocol IDs, preserves existing outputs, verifies current
source/WAV and feature-array hashes, model/window provenance and the full time
grid, and records annotation/protocol/cache hashes. `complete` describes supplied
protocol coverage only; it does not certify the proposal's annotation volume,
label quality, or human gate. Run `verify_event_cache.py` first for the broader
numerical cache audit. No experiment detector scores have been produced: unit
tests use temporary synthetic fixtures only.

CPU event extraction now checkpoints each completed inference batch under
`<video-cache>/.event-checkpoints/<source-and-settings-digest>/`. Resume verifies
batch indices, dimensions, float32 values, nonzero embeddings and payload SHA256.
A changed WAV/model/settings digest uses a separate directory; missing, corrupt,
or incompletely written batches are recomputed. Metadata is committed after the
atomic payload replacement. Final event arrays and their existing provenance
format remain unchanged, and completed final caches are reused first. Checkpoints
are retained for recovery and require a single writer per video, as do final
caches. Progress prints after the first batch, every 25 batches, and the last.
The already-running long pilot predates this code and still saves per video;
subsequent runs gain batch recovery. Synthetic interruption tests prove exact
cue-array equality after reconstruction; a real interrupted model run has not
yet been exercised.

Real-data recovery follow-up: isolated 73.2s development clip CxWby8RuUwU6 produced 74 windows and 19 checkpoints; rebuilding without final cache files reused all 19 batches, yielded seven bit-identical arrays and an identical NPZ SHA256, and passed the independent event cache verifier. Evidence: outputs/verification/event-checkpoint-real-comparison.json and event-checkpoint-real-audit.json. No OS-kill test or human detector-accuracy claim.
