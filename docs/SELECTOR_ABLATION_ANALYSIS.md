# Selector-ablation analysis

The 16-contrast family was newly declared on 2026-11-23 after the descriptive
31-run selector table was available. Contrasts follow the experimental factors,
not observed effect sizes. This is not described as prospective preregistration.

The family comprises two speech/non-speech versus combined contrasts; global
and five superclass trigger versus suppress contrasts; tagger versus gated
Whisper events; snap versus event-time placement; three novelty-memory
contrasts; and three class-only versus A_cue contrasts. The latter compare
complete selector variants, including their documented speech-stream restriction;
they are not isolated event-detector accuracy measurements. The Whisper arm
removes AST/CLAP while retaining common speech/acoustic controls. All settings
and original raw results remain unchanged.

The exporter checked all 31 frozen run IDs and their complete 30-clip membership.
It recomputed temporal TP/FP/FN, precision, recall and F1 from raw selection
timestamps and current adapter references, checking every report value, requested
K, realized count, shortfall and movie grouping. Clips with zero selections
remain included. Reference records and raw annotations are bound by hashes;
annotation text is not copied into the committed result artifacts. Historical
selection reports did not store their own reference hashes, so this recomputation
cannot prove retrospectively that references never changed before the audit.

Paired inference averages clips within each movie and gives each of the ten
movies equal weight. It uses 10,000 paired bootstrap samples (seed 0) for ordinary
95% confidence intervals, two-sided permutation-based Wilcoxon tests, and one
Holm adjustment across all 16 comparisons. Intervals are not multiplicity-adjusted.
No oracle comparison, matched-pair text score, rendered-speech measurement, or
human event-detector accuracy claim is made.

All 16 Holm-adjusted p-values were 1.0 in this sample. This provides no evidence
for a difference after family correction; it does not establish equivalence.
For example, snap minus event-time placement has mean F1 difference 0.0369 and
ordinary bootstrap interval [0.0005, 0.0857], but its Holm-adjusted p-value is 1.0.
The interval and family-adjusted test address different inferential summaries.

Results: [table](results/reframed-selector-ablations/selector-ablation-stats.md),
[CSV](results/reframed-selector-ablations/selector-ablation-stats.csv),
[full statistics/provenance](results/reframed-selector-ablations/selector-ablation-stats.json),
[figure](results/reframed-selector-ablations/selector-ablation-stats.png).

Reproduce into a fresh directory (existing analyses are never overwritten):

```bash
cd /scratch/yy5919/pyro-revision15-records
source hpc/config.sh
in_env ro 'CUDA_VISIBLE_DEVICES="" OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 MKL_NUM_THREADS=2 python scripts/analyze_selector_ablations.py --config /scratch/yy5919/pyro-revision15-formal/configs/experiments/reframed_test_local.toml --experiment-dir /scratch/yy5919/pyro-resume-results/experiments/reframed_test_local_v1 --plan configs/selector_ablation_contrasts.json --output-dir /scratch/yy5919/pyro-resume-results/selector-ablation-reproduction'
```

Verification: four synthetic regression tests passed, including zero-selection
coverage, changed references/groups, missing clips, changed selections and movie
pairing. Real execution verified 31 selectors, 30 clips, ten movies and all 16
contrasts. Logs: `/scratch/yy5919/pyro-selector-ablation-stats-tests.log` and
`/scratch/yy5919/pyro-selector-ablation-stats-real.log`. This was CPU-only work.
