# Real-data GPU smoke preparation

GPU submission is on hold until the current data transfer finishes and the user
obtains their GPU session. No GPU inference has run during this CPU preparation.

## Verified CPU prerequisites

- MTG available media: 413 files passed source SHA256 checks.
- Ten development clips have current frame/audio caches under
  `/scratch/yy5919/pyro-kd-derived/cache/mtg-ad`.
- Existing U builder produced 78 positive-budget moments (111 exclusions).
  This is a pipeline smoke input, not the final Revision 1.5 comparison protocol.
- Four real moments passed `run_ad_vlm.py --dry-run`, opening their images and
  constructing P0 prompts. Log: `outputs/verification/mtg-real-dev-dry-run.log`.
- Existing Qwen2.5-VL-7B snapshot has all five indexed weight shards present
  (16,584,414,560 bytes); CPU-only offline AutoConfig/AutoProcessor load succeeded.
  Evidence: `outputs/verification/existing-vlm-cache.json` and
  `outputs/verification/qwen25-offline-processor.log`.
  Presence is not a full weight integrity or GPU loading test.

## Prepared command for the later GPU session

Run from the repository root on an allocated GPU compute node after the hold
is lifted. This command has not been executed with GPU inference:

```bash
source hpc/config.sh
in_env ro 'python scripts/run_ad_vlm.py --model /home/yy5919/.cache/huggingface/hub/models--Qwen--Qwen2.5-VL-7B-Instruct/snapshots/cc594898137f460bfe9f0759e9844b3ce807cfb5 --input /scratch/yy5919/pyro-kd-derived/moments/mtg-ad_U_dev.jsonl --output outputs/real-data-smoke/qwen25-mtg-U-P0.jsonl --prior P0 --limit 4 --batch-size 2 --max-new-tokens 128 --quant none'
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
