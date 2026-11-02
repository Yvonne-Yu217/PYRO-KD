# Week 1 spec: data, splits, and compute

Status as of 2026-11-02. Items marked **TEAM** still need a person to make the call or do the work.

## Dataset decision

**Primary: Mind the Gap (MTG-AD).** The dataset is `google/mind-the-gap-inline-ad` on Hugging Face (CC BY 4.0), with code at [google/mtg-ad](https://github.com/google/mtg-ad). It's the proposal's original primary dataset, and it's now public, so no access request is needed.

- **Size:** 433 YouTube videos, about 36 hours, with professional inline AD. 332 are validation and 101 are test (the paper's evaluation set).
- **Fields:** the GT AD has **start times only**, with a median of 16 segments per video. Each video also has a timestamped ASR transcript (empty for 94 validation and 20 test videos with no speech) and Gemini-generated dense visual info.
- **Loader:** `src/datasets/mtg_ad.py` reads `data/annotations/*.jsonl` and sorts the AD and transcript segments by time. A few validation scripts list them out of order, and two validation videos have one AD start past the stated duration.
- **Videos:** downloaded from YouTube with yt-dlp at ≤720p (`hpc/prepare_data.sh`). **Not verified yet:** whether every video is still online. The script lists any unavailable ones.
- **Evaluation:** our temporal metrics, BERTScore, cosine similarity and rubric (`src/evaluation/`) already follow MTG-AD's format and judge definitions. The official MTG-AD-Eval (TTS-rendered timing and Quality-RF) is Gemini-based. **TEAM:** decide whether to run it as the primary timing check, as the proposal suggests.
**Secondary: REFRAMED** (`igorsterner/reframed`, CC BY-NC 4.0), used as a movie-domain check of the MTG-AD findings (`src/datasets/reframed.py`, `configs/reframed_splits.json`).

- **Loader:** the validation and test metadata load and the test split decrypts. AD, subtitle and SDH times are clip-relative, and the clips are shorter (median 2.3 minutes, up to 8.7).
- **Splits:** a 10-clip pilot from validation (one clip per movie) and a fixed 30-clip test sample spread across the 10 test movies. Report results per movie, because excerpts from one movie are not independent.
- **K rate:** 5.0 descriptions per minute after merging AD cues less than 0.5 s apart (validation median). The raw cue rate of 12.6 per minute overstates how many separate descriptions there are.
- **Videos:** downloaded by ffmpeg from the dataset's source. **Not verified yet:** whether the downloads work (the script prints each clip's duration drift).

**Downstream: EgoSchema** (public 500-question subset), unchanged; see `scripts/prepare_egoschema.py`.

All three datasets share the same frame cache, moment format and generation script. The proposal is updated to match (`CV Project Proposal.md`).

## Fixed splits (`configs/mtg_ad_splits.json`, seed 0)

| Set | Source | Size | Rule |
|---|---|---|---|
| Dev | validation | 10 videos | Category-stratified among videos with ≥ 3 AD segments. Used to tune α, prompts and the judge. |
| Exemplar pool | validation | the rest | Any validation video not in dev; never test. |
| Test | test | 50 of 101 | Category-stratified. Don't evaluate on it until Week 4. |

The 10 dev videos are 5 vlog, 2 education, and 1 each of cooking, nature and travel. Their durations run from 214 to 517 s, and 3 of them have no speech.

If compute allows, run all 101 test videos to compare with the paper's 12 released model configurations.

## Choosing K

We use one duration-based rule for every selector, set from validation only:

`K = max(1, round(rate × duration_s / 60))`

| Validation statistic | Value |
|---|---|
| GT AD segments per minute, median | **3.48** |
| GT AD segments per video, median | 16 |
| K/human ratio at rate 3.5 | 1.00 |
| K/human ratio at rate 5.0 | 1.45 |

The configured rate is **3.48**. The earlier 5.0 per minute came from REFRAMED, where film AD is denser. On MTG-AD, a rate of 5.0 would select about 45% more moments than the human describers wrote, which lowers temporal precision for every selector equally.

**Confirmed 2026-11-02:** the rate stays at 3.48. It is a single value in `configs/mtg_ad_splits.json`.

## Cache and moment schema

- `data/cache/mtg-ad/<video_id>/` holds `manifest.json`, `audio_16k.wav` (mono) and `frames/%06d.jpg`. Frames are at 2 fps with the long side ≤ 768 px, and frame *i* is at *i/fps* seconds (`src/utils/media.py`).
- `data/moments/<selector>_<set>.jsonl` holds one row per selected moment: `time`, 4 frames over 2 s, transcript within ±15 s, and the word limit (`scripts/build_moments.py`). Every selector writes the same fields, so the generator's inputs stay identical.
- The word limit is 15 words until the VAD gap cache exists. After that it becomes `floor(gap × 3)`.

## Models and compute (NYU HPC Cloud Bursting)

- **Account:** `csci_ga_2271-2026fa`, 300 GPU-hours per student. Setup and data prep run on the CPU partition `n2c48m24`. See `hpc/README.md`.
- **Qwen3-VL-8B (primary):** runs on 1× L4 with the `qwen` preset, or 1× A100 with `qwen-a100`.
- **Muse Glimmer 30B (`meta-models/Muse-Glimmer-30B`):** runs in bf16 on 2× A100 (`muse`) or 4-bit on 1× A100 (`muse-4bit`). Report the quantization alongside any Muse results.
- **Generation:** transformers in-process (`src/generation/hf_vlm.py`), at temperature 0. Each run writes a `.meta.json` with GPU, peak memory, load time and seconds per item, and requeued jobs resume.
- **TEAM:** set the API spending cap, and do the novelty and literature check.

## Audio branch (selector A and the word cap)

- **Features** (`src/utils/audio_features.py`, `scripts/cache_audio_features.py`): Silero VAD gives speech segments, which are inverted into non-speech gaps. Gaps shorter than 1 s are dropped, and edges that touch speech are pulled in by 0.25 s. librosa then adds the mean RMS energy and mean onset strength inside each gap. Everything is cached as `audio_features.json` beside each video's manifest, and videos without audio are listed rather than skipped silently.
- **A selector** (`src/utils/audio_selector.py`): `score = 1.0 × duration − 0.5 × energy − 0.0 × onset`, each term divided by its maximum over the video's gaps. The moment sits at the gap start. Top-K uses a 2 s minimum separation. If a video has fewer valid gaps than K, the selector returns fewer moments and the shortfall is reported, with no padding. The weights and the sign of the onset term are dev-set questions for Week 3.
- **Word cap:** `floor(gap × 3)` words, between 3 and 30, where gap is the time from the moment to the next speech onset (0 inside speech). The rule is the same for every selector, so gap fit is comparable.
- **Checked on:** synthetic speech made with macOS `say` and silence in between. VAD recovered the speech at 3.0–7.4 s, 13.5–16.1 s and 19.2–21.6 s, and found the four expected gaps. **Not yet checked:** real MTG-AD and REFRAMED audio (music, noise), and how well the gaps line up with human AD starts.
