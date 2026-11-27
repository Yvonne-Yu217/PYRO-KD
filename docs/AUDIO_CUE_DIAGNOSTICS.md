# Audio-cue diagnostics on frozen REFRAMED clips

The existing experiment CLI has no trigger/suppress diagnostics stage. This
exporter reuses `src/evaluation/audio_diagnostics.py` directly and checks its
low-change coverage against the already computed selector metrics. It does not
change weights, thresholds, caches, selections or predictions.

The parameters remain the implementation's existing defaults: detected-onset
association within ±2 seconds; low modeled audio change strictly below 0.2
within ±2 seconds; timeline chance sampled at 0.1-second intervals. The audio
change signal is the maximum across modeled non-speech-event components,
including normalized CLAP change, loudness and onset tracks. It is not a physical
silence measurement, a manually labeled event set or an event-detector accuracy
estimate. Optional diarization remains unavailable and is recorded in provenance.

All 30 frozen clips and five core selectors passed reference and raw-selection
checks. There are 315 detected onsets across 61 labels. The superclass summaries
report pooled reference-onset hit rates, duration-weighted timeline chance, and
their descriptive ratio. Differences in weighting and the observational design
mean that lift must not be interpreted as causal triggering or suppression.
Music lift is 1.317 and impulsive-event lift is 1.258; these values do not establish
whether the events themselves were correctly detected.

The fixed low-change proxy selects nine of 548 reference starts (1.64%) across
six clips. U, V, A_cue, AV and O match 5, 1, 0, 1 and 5 of those starts,
respectively. This denominator is very small. Realized selector budgets differ,
and O is a budget-capped oracle. These are descriptive coverage results only;
human event labels remain necessary for detector precision/recall.

Results: [summary](results/reframed-audio-cue-diagnostics/audio-cue-diagnostics.md),
[superclass CSV](results/reframed-audio-cue-diagnostics/superclass-association.csv),
[detected-label CSV](results/reframed-audio-cue-diagnostics/detected-label-association.csv),
and [per-clip results/provenance](results/reframed-audio-cue-diagnostics/audio-cue-diagnostics.json).
Source hashes bind the config, frozen implementation, raw references, selectors,
media manifests, VAD, bridged cues and upstream cache dependencies. No raw
annotation text or human detector labels are added to Git.

Reproduce into a fresh directory:

```bash
cd /scratch/yy5919/pyro-revision15-records
source hpc/config.sh
in_env ro 'CUDA_VISIBLE_DEVICES="" OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 MKL_NUM_THREADS=2 python scripts/export_audio_cue_diagnostics.py --config /scratch/yy5919/pyro-revision15-formal/configs/experiments/reframed_test_local.toml --experiment-dir /scratch/yy5919/pyro-resume-results/experiments/reframed_test_local_v1 --output-dir /scratch/yy5919/pyro-resume-results/audio-cue-diagnostics-reproduction'
```

Verification: 22 relevant tests passed, including threshold-boundary behavior,
empty proxy subsets, zero-selection coverage, chance-rate calculation, missing
selectors and existing diagnostic routines. Real execution checked all 30 clips
and reproduced existing low-change counts and recalls exactly. Logs:
`/scratch/yy5919/pyro-audio-cue-export-tests.log` and
`/scratch/yy5919/pyro-audio-cue-export-real.log`. CPU only; no paid API calls.
