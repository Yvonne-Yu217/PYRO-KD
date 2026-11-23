# Real-data GPU smoke preparation

GPU submission is on hold until the current data transfer finishes and the user
obtains their GPU session. No GPU inference has run during this CPU preparation.

## Verified CPU prerequisites

- MTG available media: 413 files passed source SHA256 checks.
- Ten development clips have current frame/audio caches under
  `/scratch/yy5919/pyro-kd-derived/cache/mtg-ad`.
- Existing U builder produced 78 positive-budget moments (111 exclusions).
  The word budget now follows floor(gap*3) exactly. The selector shortfall still makes this a pipeline smoke input, not the final comparison protocol.
- Four real moments passed `run_ad_vlm.py --dry-run`, opening their images and
  constructing P0 prompts. Log: `outputs/verification/mtg-real-dev-dry-run.log`.
- Existing Qwen2.5-VL-7B snapshot has all five indexed weight shards present
  (16,584,414,560 bytes); CPU-only offline AutoConfig/AutoProcessor load succeeded.
  Evidence: `outputs/verification/existing-vlm-cache.json` and
  `outputs/verification/qwen25-offline-processor.log`.
  Presence is not a full weight integrity or GPU loading test.

CPU multimodal encoding also passed for the same four real inputs in batches
of two, using the generation path's `apply_chat_template` options and left
padding. Report: `outputs/verification/mtg-real-processor.json`; reproducer:
`outputs/verification/check_real_mtg_processor.py`. This checks image encoding,
chat formatting and padded tensors, but does not exercise model generation.

## Prepared command for the later GPU session

Run from the repository root on an allocated GPU compute node after the hold
is lifted. This command has not been executed with GPU inference:

```bash
source hpc/config.sh
in_env ro 'python scripts/run_ad_vlm.py --model /home/yy5919/.cache/huggingface/hub/models--Qwen--Qwen2.5-VL-7B-Instruct/snapshots/cc594898137f460bfe9f0759e9844b3ce807cfb5 --input /scratch/yy5919/pyro-kd-derived/moments/mtg-ad_U_dev_revision15.jsonl --output outputs/real-data-smoke/qwen25-mtg-U-P0.jsonl --prior P0 --limit 4 --batch-size 2 --max-new-tokens 128 --quant none'
```

The exact local snapshot matches the proposal's Qwen2.5-VL-7B open-model class
and avoids the default scratch cache, which currently lacks Qwen3/Muse weights.
Do not download Muse as a substitute for the proposal's API-model condition.
Record code SHA, GPU/job details, elapsed time and memory for the actual run.
Inspect all four descriptions against their frames and word limits; evaluate
outputs against the appropriate development references before scaling.
A successful dry run or model load alone does not pass the end-to-end smoke gate.
The shortfall in U moments and wide transcript contexts require protocol review
before using these inputs for confirmatory selector comparisons.

Word-budget correction (2026-11-23): the new revision15 input preserves the
78 IDs, timestamps and frame windows but removes the historical cap30.
53 budgets changed; the largest upper limit is 793 words. Keep generations
concise and inspect token-limit failures; this upper limit is not a target
length. The old input is preserved for provenance. Use fresh output files
when comparing budgets; do not resume historical outputs under a new protocol.
Evidence: outputs/verification/mtg-revision15-word-budget.json and
mtg-revision15-p0-dry-run.log. Unknown-gap fallback remains historical behavior
and is not proof of gap fit. All 78 rebuilt inputs have known gap lengths.

REFRAMED development inputs are also prepared on CPU. Reproduce:

```bash
source hpc/config.sh
in_env ro 'OMP_NUM_THREADS=4 python scripts/preprocess_reframed_dev.py --data-dir /scratch/yy5919/pyro-kd-derived/datasets/reframed --cache-dir /scratch/yy5919/pyro-kd-derived/cache/reframed --report outputs/verification/reframed-dev-preprocess.json'
in_env ro 'OMP_NUM_THREADS=4 python scripts/cache_audio_features.py --cache-dir /scratch/yy5919/pyro-kd-derived/cache/reframed'
in_env ro 'python scripts/build_moments.py --dataset reframed --selector U --videos dev --data-dir /scratch/yy5919/pyro-kd-derived/datasets/reframed --cache-dir /scratch/yy5919/pyro-kd-derived/cache/reframed --output /scratch/yy5919/pyro-kd-derived/moments/reframed_U_dev_revision15.jsonl'
```

Ten frozen validation clips yield2663 frames and current basic audio features.
The current U builder yields66 inputs and45 zero-budget exclusions. These are
smoke inputs; full fixed-K comparison readiness has not been established.
