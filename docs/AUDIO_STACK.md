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
