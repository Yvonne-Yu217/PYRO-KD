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

Generation provenance checkpoint:
`run_ad_vlm.py` hashes the full input JSONL, every referenced frame, runtime Python/JSON files under src/scripts/configs, and generation settings. Each prediction carries `run_fingerprint`; `<output>.jsonl.provenance.json` is atomically saved before model loading, and completed attempt metadata includes the record and Git SHA/dirty status. Documentation-only commits do not invalidate matching runtime bytes. Resume rejects mismatched or missing fingerprints, so historical output files require a separate new output path. This does not hash model weights or freeze a moving remote model alias: use the documented pinned snapshot for actual runs. Batch size, limit and token cap remain attempt settings to support OOM recovery and truncation retries; row-level token limits must still be checked when reporting comparisons. All input frames are hashed even for a limited run so expanding the limit remains compatible.

CPU validation: 32 tests passed across test_generation_provenance.py and test_generation_truncation.py. Four real REFRAMED timeline inputs passed dry-run (outputs/verification/provenance-dry-run.log). No inference or GPU smoke claim.

CPU visual pilot verified on real REFRAMED dev clip CxWby8RuUwU6:146 frame embeddings x512 dimensions and19 shot boundaries. OpenCLIP weights are pinned to revision1a25a446712ba5ee05982a381eed697ef9b435cf; file SHA256 ac4f8c4b88af6d963118cbf40ad93176d092abbedfcb752601ae1866352656e6. Revision enters the cache config hash, invalidating earlier unpinned feature caches. Result and exploratory K=6 local/memory timestamps: outputs/verification/reframed-visual-pilot.json. This verifies extraction and ranking behavior, not accuracy or final placement.

Reproduce: `source hpc/config.sh` then `in_env ro 'OMP_NUM_THREADS=4 python -u scripts/check_visual_pilot.py --cache /scratch/yy5919/pyro-kd-derived/cache/reframed/CxWby8RuUwU6 --output outputs/verification/reframed-visual-pilot.json'`. Uses four CPU threads, batch8, no GPU; reuses validated cache. Download was explicitly online; subsequent use is offline. The reported elapsed time for a reused cache is verification time, not original extraction latency.

Full development visual audit command: `source hpc/config.sh` then `in_env ro 'OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 python scripts/verify_visual_cache.py --cache-dir /scratch/yy5919/pyro-kd-derived/cache/reframed --report outputs/verification/reframed-dev-visual-check.json'`. Expected IDs come from the frozen REFRAMED split configuration. Checks include source freshness, feature config/time alignment, embedding file hash, finite512-dimensional normalized vectors, reconstructed adjacent-frame novelty (0.001 tolerance for float16 storage), shot bounds, and both rankers' K/separation/timestamp constraints. Audit K uses media duration and the frozen rate; it is a mechanical check, not a finalized experimental input builder. Incomplete coverage exits1 and is normal while extraction is still running.

Final REFRAMED visual coverage:10/10 frozen dev videos,2663 embeddings and386 shots, audit complete=true. Exploratory `build_moments.py --dataset reframed --selector V --visual-novelty memory` (or `local`) with the derived dataset/cache paths produced82 (or75) positive-budget rows across9videos. Both requested111 and lost29/36 to the old post-selection zero-word filter. These files are explicitly named `reframed_V_{memory,local}_dev_exploratory.jsonl`; they do not establish an equal-budget comparison or a finalized two-stage method. The independent row audit is outputs/verification/reframed-dev-v-input-check.json.

Placed A/AV generation inputs prepared (exploratory)

`scripts/export_placed_moments.py` exports the four diagnostic conditions with mandatory `--frame-center event|placement`. For this dev preparation `event` was used: `time` is narration placement, `event_time` is the selected event, and `frame_center_time`/policy plus lag are explicit. Four frames span2s around the event; prompts now display this frame-center time instead of incorrectly labelling them with delayed narration time. Timeline context uses the same frame center; placement remains available for overlap evaluation. Generation prediction rows retain these fields plus selection provenance and shortfall.

Inputs: `/scratch/yy5919/pyro-kd-derived/moments/reframed_{A_trigger,A_suppress,AV_trigger,AV_suppress}_placed_event_dev_exploratory.jsonl` and corresponding `_timeline.jsonl` files. Each has97 rows across10clips,111requested and14shortfall. Shifted narration rows:25/16/25/18 respectively. Cached timelines use existing exploratory0.5score/0.1rise thresholds; no annotation text is injected. Eight real-input dry runs (four conditions x P0/P1, first4rows each) passed, log outputs/verification/placed-input-preparation.log and per-condition logs. Independent all-row audit outputs/verification/placed-input-check.json checks base preservation, source frames, exact budgets and distinct event/playback timing. This is not VLM inference or a finalized experimental protocol.

Example export: `python scripts/export_placed_moments.py --selection-report outputs/verification/reframed-dev-fused-selectors.json --condition AV_trigger --cache-dir /scratch/yy5919/pyro-kd-derived/cache/reframed --frame-center event --output <new-output.jsonl>`. Export verifies selection-source hashes, live VAD gaps, bounds and media freshness; it refuses to overwrite an input. Attach timeline with scripts/attach_audio_timeline.py as documented earlier; P1 uses the frozen train-only exemplar bundle.
