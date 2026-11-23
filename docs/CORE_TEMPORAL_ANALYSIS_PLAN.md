# Core temporal statistical analysis

This family was declared on 2026-11-23 while core U outputs were already
available. It is newly declared, not retrospectively described as preregistered.
The contrasts were chosen structurally without inspecting their effects.

Scope: all ten Qwen core cells (U, V, A_cue, AV, O crossed with P0/P1), using
all 30 frozen REFRAMED test clips. Export retains zero-output clips, generation
failures and selection/preparation shortfalls. The inferential metric is temporal
F1 only. Matched-pair BERTScore/OpenCLIP scores have conditional denominators and
are not used for significance claims by this pipeline.

The fixed family contains all six unordered deployable-selector pairs within
each prompt (12 contrasts), followed by P1 minus P0 within each selector
(5 contrasts). The O prompt contrast is an oracle upper-bound diagnostic and
must not be interpreted as a deployable-method comparison. All 17 p-values
receive one Holm correction.

The analysis unit is the movie: average clip metrics within each movie, then
use equal-weight movie means for paired 95% bootstrap confidence intervals
(10,000 resamples, seed 0) and two-sided paired Wilcoxon tests. Existing
`analyze_paired.py` uses a permutation-based Wilcoxon implementation and reports
its method/version. No per-moment independence assumption is made. Precision,
recall and temporal counts are exported as descriptive checks only.

Before export, all ten cells must be complete. The exporter checks exact frozen
membership, raw-generation IDs and timestamps, complete moment attempts, final
success/error semantics, and original frozen source/input hashes. It recomputes
per-video temporal metrics from the adapter's references and checks them against
every report plus its macro F1. Zero-output clips remain zero when they have
references; undefined empty-reference/empty-output values are retained in the
export and cause complete-family inference to stop rather than silently omit
clips.

Source hashes bind the config, original experiment identity, generation files,
moments, reports, raw test annotation file, manifests, adapter-derived reference
records, and exporter implementation. Historical evaluation JSON lacks its own
reference digest: matching the recomputation cannot retroactively prove that
references never changed before export. This limitation is recorded explicitly.
No raw predictions or existing reports are rewritten.

Run only after all ten reports and their raw outputs are complete:

```bash
cd /scratch/yy5919/pyro-revision15-records
source hpc/config.sh
in_env ro 'CUDA_VISIBLE_DEVICES="" OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 python scripts/export_core_temporal_metrics.py --config /scratch/yy5919/pyro-revision15-formal/configs/experiments/reframed_test_local.toml --experiment-dir /scratch/yy5919/pyro-resume-results/experiments/reframed_test_local_v1 --output /scratch/yy5919/pyro-resume-results/formal-core-temporal/paired.jsonl'
in_env ro 'CUDA_VISIBLE_DEVICES="" OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 python scripts/analyze_core_temporal.py --input /scratch/yy5919/pyro-resume-results/formal-core-temporal/paired.jsonl --plan configs/core_temporal_contrasts.json --output /scratch/yy5919/pyro-resume-results/formal-core-temporal/contrasts.json'
```

Both commands refuse to overwrite prior exports/results. No formal statistical
execution was performed while preparing this plan. Synthetic regression tests
exercise reference/report mismatches, truncated files, missing attempts,
superseded errors, zero-output membership, and movie-level pairing.
