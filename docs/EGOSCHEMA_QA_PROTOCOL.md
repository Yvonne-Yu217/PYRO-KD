# Frozen EgoSchema visual QA diagnostic

`configs/egoschema_qa_frozen.json` freezes three conditions before heldout
inference/scoring: U, V_local and V_memory. Each uses the same490 heldout videos,
excluding all ten pilot IDs. All distributed EgoSchema clips lack audio, so
these are visual-only diagnostics. No A/AV, timeline or native-audio condition
is inferred from missing audio.

The completed ten-item pilots produced10 valid answers per condition; U was
10/10 correct and both V variants9/10. These small pilot scores are not heldout
results. They establish real-data runtime viability at768 pixels, four frames
per moment, K=5/min, minimum separation2s, the pinned Qwen2.5-VL7B snapshot,
bfloat16/SDPA, temperature0 and16 output tokens. Those settings remain fixed.
No AD exemplars enter the zero-shot QA prompt. Input preparation records any
realized-K shortfall and does not silently subsample frames.

The protocol records exact membership, budget and question-source hashes and
pilot artifact hashes. Heldout input hashes are recorded in per-condition
`.jsonl.preparation.json` receipts after validated atomic publication, before
inference. The CPU builder checks490 unique IDs, pilot exclusion, source hashes,
question/option ordering, current silent media, selected times and frame paths.
It never opens the answer file. Generation retains its own input/frame/source
fingerprint and every invalid or failed prediction.

## Queue and costs

Original media/visual extraction41027 is followed by corrected cache validation
41038, then input preparation41198. The initial extraction driver's validation
mistake is recorded separately; its useful caches are retained and revalidated.
Secondary GPU41197 owns U and V_local; primary40964 owns V_memory. Output files
are disjoint. Jobs skip unavailable CPU inputs instead of keeping a GPU idle.
Each condition is bounded at90 minutes. Pilot timings extrapolate to roughly
105 total GPU minutes for1470 requests, excluding loading and filesystem work.
No paid API calls are authorized or needed for these conditions.

## Evaluation and predeclared comparisons

Retain all490 questions in each condition's denominator; missing, invalid and
generation-error predictions count incorrect. Labels are opened only by the
dedicated evaluator after actual predictions exist. Use the frozen config's
answer-file hash; never change membership or generation settings after observing
heldout correctness.

The diagnostic family in `configs/egoschema_qa_comparisons.json` is directly
compatible with `scripts/analyze_family.py`:

- V_local minus U.
- V_memory minus U.
- V_memory minus V_local.

Use video-cluster paired analysis (one question per video),10000 bootstrap
resamples, seed0, Wilcoxon tests and Holm correction across these three declared
comparisons. Confidence intervals are pointwise95%, not simultaneous intervals.
This is an explicitly separate diagnostic family from long-form QA; do not claim
global multiplicity correction across both families.

After generation finishes, run the following from the records checkout for each
condition, with a new output path if a report already exists:

```bash
source hpc/config.sh
in_env ro 'CUDA_VISIBLE_DEVICES="" python scripts/evaluate_egoschema_qa.py --input /scratch/yy5919/pyro-resume-results/egoschema-heldout/U.jsonl --answers /scratch/yy5919/pyro-kd-derived/datasets/egoschema/subset_answers.json --pilot-config configs/egoschema_pilot.json --split heldout --output /scratch/yy5919/pyro-resume-results/egoschema-heldout/U.evaluation.json'
# Repeat with V_local and V_memory after their outputs complete.
in_env ro 'CUDA_VISIBLE_DEVICES="" python scripts/export_qa_paired_metrics.py --dataset egoschema --report U /scratch/yy5919/pyro-resume-results/egoschema-heldout/U.evaluation.json --report V_local /scratch/yy5919/pyro-resume-results/egoschema-heldout/V_local.evaluation.json --report V_memory /scratch/yy5919/pyro-resume-results/egoschema-heldout/V_memory.evaluation.json --output /scratch/yy5919/pyro-resume-results/egoschema-heldout/paired.jsonl'
in_env ro 'CUDA_VISIBLE_DEVICES="" python scripts/analyze_family.py --input /scratch/yy5919/pyro-resume-results/egoschema-heldout/paired.jsonl --family configs/egoschema_qa_comparisons.json --unit cluster --resamples 10000 --seed 0 --output /scratch/yy5919/pyro-resume-results/egoschema-heldout/comparisons.json'
```

Agent contribution: read-only pilot and scheduler audit, frozen protocol and
comparison specification. No heldout answers were read and no jobs were submitted
by this documentation task. Parent agent coordinates execution and final reporting.
