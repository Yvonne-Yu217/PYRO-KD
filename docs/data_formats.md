# Data formats

Every cache file lives in the video's cache directory `$PYRO_CACHE/<dataset>/<video_id>/`, next to `manifest.json`. Consumers validate before use and report a reason code instead of using stale or malformed files.

## Per-video caches

| File | Writer | Validated by | Invalidated when |
|---|---|---|---|
| `manifest.json`, `frames/%06d.jpg`, `audio_16k.wav` | `src/utils/media.py:preprocess_video` | `cache_is_complete`, `cache_is_current` | source path/size/mtime or extraction settings change |
| `audio_features.json` (Silero speech, usable gaps) | `scripts/cache_audio_features.py` | `cache_checks.load_audio_features` | WAV bytes, VAD config or feature version change |
| `visual_features.json` + `visual_embeddings.<digest>.npy` | `scripts/cache_visual_features.py` | `visual_features.load_validated_features` | manifest hash, CLIP/shot config or frames change |
| `audio_cues.json` | `scripts/cache_audio_cues.py` | `audio.cues_io.load_cues_for_selection` | WAV bytes or the VAD result change (`audio_sha256`, `vad_hash`); the producer also re-runs on config or backend changes |

Frame *i* is at *i / fps* seconds of the stream (2 fps). All times are seconds on that clock.

### `audio_cues.json` (version 1)

```json
{
  "version": 1, "duration": 300.0, "config_hash": "...", "audio_sha256": "...", "vad_hash": "...",
  "backends": {"transcriber": "faster-whisper:large-v3:float16", "tagger": "MIT/ast-...", "clap": "laion/clap-htsat-unfused", ...},
  "streams": {"asr": "ok", "tagger": "ok", "clap": "ok", "acoustic": "ok", "semantic": "ok",
              "diarization": "missing: no diarization (optional in the proposal)"},
  "asr": {"source": "whisper-vad-gated", "segments": [{"start", "end", "text", "words": [{"start", "end", "word", "probability"}]}]},
  "tagger": {"model": "...", "times": [0.0, 1.0, ...], "labels": ["Music", ...], "posteriors": [[...], ...]},
  "clap": {"model": "...", "times": [...], "change": [...]},
  "acoustic": {"times": [...], "loudness_db": [...], "loudness_jump": [...], "onset_strength": [...]},
  "events": [{"time", "label", "superclass", "strength", "source": "tagger" | "whisper_tag"}],
  "speech_cues": [{"time", "kind": "deictic" | "semantic_shift" | "speaker_change" | "long_silence", "strength", "text", "timing"}]
}
```

A stream that did not run is recorded as `"missing: <reason>"`, never as an empty result. `asr.source = "dataset"` means MTG-AD's released ASR (segment-level timing; deictic cue times are interpolated and marked `timing: "interpolated"`). Ablation 3's Whisper-tag cue set is a second file, `audio_cues.whisper_tags.json`, with the same schema.

## Hand-labeled audio events (§6 detector check)

JSONL, one onset per line. Not shipped; the team creates it.

```json
{"video_id": "abc123", "time": 42.0, "label": "Applause"}
{"video_id": "abc123", "time": 63.5, "superclass": "music"}
```

`label` is an AudioSet display name (mapped to a superclass by `src/audio/taxonomy.py`) or give `superclass` directly. Matching is one-to-one within ±1 s by default.

## Canonical dataset JSONL (Cue2Narrate, long-form QA, anything else)

One object per video, validated by `src/datasets/adapters.py:validate_canonical_row`:

```json
{"video_id": "v1", "split": "test", "video_path": "videos/v1.mp4", "duration": 1834.2,
 "ad": [{"start": 12.0, "end": 14.5, "text": "..."}],
 "transcript": [{"start": 0.0, "end": 3.1, "text": "..."}],
 "qa": [{"question_id": "v1_q1", "question": "...", "options": ["...", "..."], "answer": 2}],
 "group": "movie-17"}
```

`ad`, `transcript`, `qa` and `group` are optional. `answer` is an option index or null. `group` is the independence cluster (e.g. the movie) for statistics. Write a small converter per benchmark once its release files are in hand; this project does not assume their field names.

## P1 exemplars (`configs/p1_exemplars.json`)

Built by `scripts/build_p1_exemplars.py` from MTG-AD validation videos outside dev, never test:

```json
{"version": "mtg-ad-exemplar-pool-seed0", "guidelines_version": "p1-guidelines-v1",
 "exemplars": [{"video_id": "...", "time": 31.2, "text": "...", "category": "Cooking"}, ...]}
```

Generation refuses P1 if an exemplar's `video_id` is among the evaluated videos.

## Experiment outputs (`$PYRO_OUTPUTS/experiments/<name>/`)

- `selections/<run_id>.jsonl`: `{"video_id", "selector", "run_id", "k", "duration", "times", "status", "available", "details", "diagnostics", "oracle", "ablation"}`. Status is `ok`, `ok_degraded` (ran without some cue streams; see `diagnostics.streams_missing`), `missing_vad`, `missing_audio_features`, `missing_visual_features`, `missing_ground_truth` or `error`.
- `moments/<run_id>.jsonl`: one generation input per selected time (frames, frame times, transcript context, `word_limit`, `audio_timeline`, `audio_clip`); `.report.json` lists times with no feasible gap (zero budget), which are never sent to a model.
- `generations/<run_id>.jsonl`: append-only; `model_output` is the model's text, `output` the text after the word-cap policy, plus `words`, `over_budget`, `truncated`, cost and latency. Later rows supersede earlier errors for the same id.
- `eval/selection_metrics.json`, `eval/generation_<run_id>.json`, `eval/qa_metrics.json`, `eval/tuning_<base>.json`, `eval/dataset_validation.json`.
- `tables/*.md|csv`, `plots/*.svg`, `status.json`, `config.resolved.json`.
