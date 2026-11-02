# PYRO-KD: cue-driven keyframe selection for MLLM audio description

This project compares uniform, video-only, audio-only, fused audio+video and oracle selection of the moments where a frozen VLM writes audio descriptions. It evaluates against professional AD on REFRAMED, with a downstream QA check on EgoSchema. Week 1 decisions are in [docs/week1_spec.md](docs/week1_spec.md).

## Setup

```bash
uv sync                      # Python 3.12; torch from the cu126 index
uv run python -m unittest discover -s tests
```

On macOS, the CUDA torch wheels don't install. Run the standard-library tests with `python3 -m unittest discover -s tests`. FFmpeg must be on `PATH`.

## Data (never commit `data/`)

```bash
uv run python scripts/prepare_reframed.py                                # metadata + pilot list
uv run python scripts/prepare_reframed.py --download-videos --preprocess # pilot videos -> cache
uv run python scripts/prepare_egoschema.py                               # annotations + QA pilot
```

## Models on HPC

```bash
bash scripts/hpc/setup_vllm_env.sh                                   # once
sbatch --account=<acct> scripts/hpc/prepare_pilot.sbatch
sbatch --account=<acct> scripts/hpc/vlm_smoke_test.sbatch            # Qwen3-VL-8B
MODEL=nvidia/Muse-Glimmer-30B-NVFP4 sbatch --account=<acct> scripts/hpc/vlm_smoke_test.sbatch
```

## Layout

- `src/datasets/`: REFRAMED (decryption, SRT tracks, pilot) and EgoSchema loaders.
- `src/utils/media.py`: ffprobe, 16 kHz audio, timestamped frame cache, frame windows.
- `src/utils/visual_selector.py`, `audio_selector.py`: the selectors (in progress).
- `src/generation/vlm_client.py`: frozen VLM client for any vLLM / OpenAI-compatible server.
- `src/evaluation/`: temporal P/R/F1, BERTScore, sentence cosine, LLM-judge rubric.
