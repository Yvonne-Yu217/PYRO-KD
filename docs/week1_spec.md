# Week 1 spec: data, pilot, and compute

Status as of 2026-11-02. Items marked **TEAM** still need a person to make the call or do the work.

## Dataset decision

**Primary: REFRAMED** (`igorsterner/reframed`, CC BY-NC 4.0; source-video rights are separate, so videos are never committed).

- Verified locally: the validation and test JSONL files load, and the test split decrypts with the published `keys.zip`. All SRT tracks parse.
- Validation has 96 clips from 10 movies, about 229 minutes in total. Test has 91 clips from 10 movies.
- All AD, subtitle and SDH times are **clip-relative**: AD end times never exceed `video_duration` by more than 0.11 s. `video_start` is the clip's offset within the full movie.
- Every clip has US AD. Two validation clips have no subtitles, and 79 of 96 validation clips have a screenplay (no test clips do).
- **Not verified yet:** whether videos download. The dataset's own recipe is an ffmpeg pull from theplatform.com, implemented in `download_video`. Run `scripts/hpc/prepare_pilot.sbatch` and check the reported duration drift.
- Mind the Gap is still the fallback if REFRAMED videos are unavailable (**TEAM:** Kiran's author email). The MTG-AD parser in `src/evaluation/parsing.py` stays for that path.

**Downstream: EgoSchema**, using only the public subset of 500 labeled questions. `scripts/prepare_egoschema.py` fetches the annotations and fixes a 10-item pilot. The pilot is excluded from the reported evaluation. Videos follow the EgoSchema README (**TEAM:** download onto HPC scratch).

## Pilot and dev split

`configs/reframed_pilot.json` holds 10 validation clips, one per validation movie. The pick is a seeded shuffle (seed 0) among clips with at least 3 US AD cues and subtitles. These 10 clips are also the **dev set** used to tune α, prompts and the judge. The test sample stays untouched until Week 4.

Because several excerpts come from the same movie, report results per movie or with a cluster bootstrap by movie.

## Choosing K

The proposal estimated about 16 descriptions per 5-minute video. REFRAMED is denser:

| Validation, US track | Median per minute |
|---|---|
| Raw SRT cues | 12.6 |
| Descriptions (cues ≤ 0.5 s apart merged) | **5.0** |
| Descriptions (≤ 1.0 s merged) | 4.5 |
| Descriptions (≤ 2.0 s merged) | 3.6 |

The UK track is sparser, at about 3.4 descriptions per minute with the 0.5 s merge.

**Proposed rule:** `K = max(1, round(5.0 × duration_min))`, which gives K ≈ 11–12 for a typical 2.3-minute clip. The rule comes from validation only. **TEAM:** confirm the rule, or pick a lower rate if the UK track counts equally.

Raw cue counts overstate how many descriptions there are, because REFRAMED splits single sentences across cues. Use `merge_cues` before any count-based metric.

## Cache schema (`src/utils/media.py`)

`data/cache/<dataset>/<video_id>/` contains:

- `manifest.json` with `media_info` (from ffprobe), `settings`, `config_hash`, `audio_path` and `frames[{index, timestamp, path}]`.
- `audio_16k.wav`: mono 16-bit PCM for Silero VAD, Whisper and librosa. It is `null` when the video has no audio.
- `frames/%06d.jpg`: frames at 2 fps (1 fps for EgoSchema), long side ≤ 768 px. Frame *i* is at *i/fps* seconds.

Re-running with the same settings reuses the cache. Changing the settings rebuilds it.

Shot detection must still run on the original video, not on the sparse frames (see the V5 caution in the proposal).

## Models and compute

- Both backbones are served through vLLM's OpenAI-compatible API (`src/generation/vlm_client.py`). Swapping models is a name change, and every call logs latency and token counts.
- Qwen3-VL-8B-Instruct is the primary model. Muse-Glimmer-30B-NVFP4 is optional, because its NVFP4 path targets Blackwell GPUs. **TEAM:** confirm which GPU types the HPC allocation has.
- `scripts/hpc/setup_vllm_env.sh` installs vLLM once, in a separate env.
- `scripts/hpc/vlm_smoke_test.sbatch` serves a model, describes 3 windows of a pilot clip, logs GPU memory, and writes `outputs/smoke/`.
- **TEAM:** HPC allocation and Slurm account (faculty sponsor), API spend cap, and the novelty or literature check.
