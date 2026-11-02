# Running PYRO-KD on NYU HPC Cloud Bursting

These scripts are adapted from the team's VLM smoke-test bundle. They generate audio descriptions for selected video moments with frozen open VLMs.

Per the course rules, coursework runs on **HPC Cloud Bursting**, not Torch. The account is `csci_ga_2271-2026fa`, with 300 GPU-hours per student.

| File | What it does |
|---|---|
| `hpc/config.sh` | Account, partitions, paths, and the `in_env` container helper |
| `hpc/setup_env.sh` | One-time setup: Singularity overlay, Miniforge, PyTorch/transformers, and the project's own dependencies |
| `hpc/download_models.sh` | Pre-downloads Qwen3-VL-8B and Muse Glimmer 30B to `$HF_HOME` |
| `hpc/prepare_data.sh` | Downloads MTG-AD annotations and videos (yt-dlp), builds the frame/audio cache, and writes the uniform-selector moments |
| `hpc/submit.sh` | Submits a generation run on the right GPU for each model |
| `hpc/run_ad.sbatch` | The SLURM job (called by `submit.sh`) |
| `scripts/run_ad_vlm.py` | Model-agnostic AD generation; resumes after requeue |

## Models and GPUs

| Preset | Model | Partition | GPU-hours / hour |
|---|---|---|---|
| `qwen` | Qwen3-VL-8B (primary) | `g2-standard-12` (1× L4, 24 GB) | 1 |
| `qwen-a100` | Qwen3-VL-8B | `c12m85-a100-1` (1× A100 40 GB) | 1 |
| `muse` | Muse Glimmer 30B, bf16 | `c24m170-a100-2` (2× A100 40 GB) | 2 |
| `muse-4bit` | Muse Glimmer 30B, 4-bit | `c12m85-a100-1` (1× A100 40 GB) | 1 |

To swap models, pass `--model qwen2.5-vl-7b` or any Hugging Face repo id. Each run's `.meta.json` records the quantization, GPU, peak memory and seconds per item. Report the quantization next to any Muse numbers.

## First-time setup (each teammate, or once in a shared folder)

Open a terminal from the OOD portal (https://ood.burst.hpc.nyu.edu/; NYU VPN required off campus), then:

```bash
cd /scratch/$USER && git clone https://github.com/Yvonne-Yu217/PYRO-KD.git && cd PYRO-KD

# CPU node for setup, downloads and data prep, so no GPU-hours are spent
srun --account=csci_ga_2271-2026fa --partition=n2c48m24 --time=02:00:00 --pty /bin/bash
bash hpc/setup_env.sh          # overlay + conda + packages (~20 min)
bash hpc/download_models.sh    # Qwen3-VL-8B + Muse Glimmer (~80 GB)
bash hpc/prepare_data.sh       # MTG-AD dev videos -> data/cache/mtg-ad, data/moments/mtg-ad_U_dev.jsonl
bash hpc/prepare_data.sh reframed   # optional: REFRAMED pilot clips
exit
```

`prepare_data.sh` prints each video's duration drift against the annotations. It also lists any YouTube videos that are no longer available. Report those, because the test sample may need a fixed replacement rule.

## Running

```bash
bash hpc/submit.sh qwen --limit 4        # smoke test: 4 moments
bash hpc/submit.sh muse-4bit --limit 4

bash hpc/submit.sh qwen-a100             # all U moments on the dev videos (~190)

squeue --me
tail -f logs/pyro-qwen-<jobid>.out
```

Outputs go to `outputs/<model>/<moments>_<prior>.jsonl`, plus a `.meta.json` file. `logs/gpu-<jobid>.csv` logs GPU utilisation every 30 s, for the cost section.

To check prompts and frames without a GPU:

```bash
srun --account=csci_ga_2271-2026fa --partition=interactive --time=00:30:00 --pty /bin/bash
source hpc/config.sh && in_env ro "python scripts/run_ad_vlm.py --dry-run"
```

## Moment input format

`scripts/build_moments.py` writes one JSON object per selected moment:

```json
{"id": "CUF3onU96-M_U_000", "video_id": "CUF3onU96-M", "selector": "U", "time": 16.25,
 "frames": ["data/cache/mtg-ad/CUF3onU96-M/frames/000032.jpg", "..."], "frame_times": [15.0, 16.0, 16.5, 17.0],
 "transcript": "nearby ASR text", "word_limit": null}
```

Every selector writes the same fields. That keeps the frame window, transcript context and word limit identical across U, V, A, AV and O. A `null` word limit falls back to 15 words until the VAD gap cache exists.

## Cloud Bursting rules that affect these scripts

- **20-minute idle shutdown.** Instances with low GPU use for 20 minutes are shut down. Do setup, downloads and data prep on `n2c48m24`, and use real batch sizes. If a job dies early, check `logs/gpu-<jobid>.csv` first.
- **Budget.** Each student has 300 GPU-hours. Smoke-test with `--limit`. The two-GPU `muse` preset uses hours twice as fast.
- **Requeue.** Jobs use `#SBATCH --requeue`, and `run_ad_vlm.py` skips finished ids, so a requeued job resumes.
- **Offline jobs.** Jobs set `HF_HUB_OFFLINE=1`, so run `download_models.sh` first. YouTube downloads happen only in `prepare_data.sh`.
