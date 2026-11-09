# PYRO-KD: cue-driven keyframe selection for MLLM audio description

This project compares uniform, video-only, audio-only, fused audio+video and oracle selection of the moments where a frozen VLM writes inline audio descriptions. It evaluates against professional AD in [Mind the Gap (MTG-AD)](https://github.com/google/mtg-ad) (primary), checks the findings on movie excerpts from REFRAMED, and tests downstream QA on EgoSchema. Week 1 decisions are in [docs/week1_spec.md](docs/week1_spec.md).

## Local setup

```bash
uv sync                      # Python 3.12; torch from the cu126 index (Linux)
python3 -m unittest discover -s tests
```

On macOS, the CUDA torch wheels don't install. The tests use only the standard library, Pillow and FFmpeg, so `python3` is enough to run them.

## Data (never commit `data/`)

```bash
python scripts/prepare_mtg_ad.py                     # annotations + configs/mtg_ad_splits.json
python scripts/prepare_mtg_ad.py --download-videos --preprocess   # dev videos (needs yt-dlp)
python scripts/prepare_mtg_ad.py --download-videos --cookies-from-browser chrome  # run locally
python scripts/prepare_mtg_ad.py --preprocess                     # reuse manually supplied videos
python scripts/cache_audio_features.py                            # Silero VAD + librosa (HPC env)
python scripts/build_moments.py --selector U --videos dev         # generation inputs (U or A)
python scripts/prepare_reframed.py                   # REFRAMED metadata + configs/reframed_splits.json
python scripts/build_moments.py --dataset reframed --videos dev
python scripts/prepare_egoschema.py                  # EgoSchema annotations + QA pilot
```

MTG-AD preparation writes `data/mtg-ad/preprocess_report.json`, including
download/preprocessing failures and whether each complete frame/audio cache was
written or reused. A cache is rebuilt if its source changes or a recorded
artifact is missing. On clusters where YouTube blocks downloads, place MP4s in
`data/mtg-ad/videos/<video_id>.mp4` and use the second command above (or run
`SKIP_DOWNLOAD=1 bash hpc/prepare_data.sh`). YouTube may require yt-dlp's
[recommended PO-token provider setup](https://github.com/yt-dlp/yt-dlp/wiki/PO-Token-Guide);
keep provider credentials/tokens outside the repository.

If a cloud/datacenter IP is blocked even with a PO token, run the browser-cookie
command on the same local machine and network where YouTube plays normally.
Copy `data/mtg-ad/videos/` to the cluster, then use `--preprocess` or
`SKIP_DOWNLOAD=1`. Cookie files and browser profiles are credentials: never copy
them to the cluster or commit them. `--cookies /safe/path/cookies.txt` and
`--yt-dlp-extractor-args` are also available for advanced local setups.

## Models on NYU HPC

See [hpc/README.md](hpc/README.md). In short:

```bash
bash hpc/setup_env.sh && bash hpc/download_models.sh && bash hpc/prepare_data.sh   # CPU node
bash hpc/submit.sh qwen --limit 4                                                    # GPU smoke test
```

## Layout

- `src/datasets/`: MTG-AD (primary), REFRAMED (movie check), and EgoSchema loaders.
- `src/utils/media.py`: ffprobe, 16 kHz audio, timestamped frame cache, frame windows.
- `src/utils/selection.py`: shared K rule, uniform selector (U), and top-K with minimum separation.
- `src/utils/audio_features.py`, `audio_selector.py`: Silero VAD gaps, librosa gap features, and the audio-only selector (A).
- `src/utils/visual_selector.py`: CLIP novelty scoring for V (in progress).
- `src/generation/`: transformers VLM loader and generator, and the P0/P1 prompts.
- `src/evaluation/`: temporal P/R/F1, BERTScore, sentence cosine, LLM-judge rubric.
- `scripts/`: data preparation, moment building, and AD generation (`run_ad_vlm.py`).
- `hpc/`: Cloud Bursting setup, model download, data prep, and job submission.
