# Selector ablation statistics

Newly declared 2026-11-23 after descriptive31-run results were available; structural contrasts, not selected by effects; not prospectively preregistered. All30clips,10movies; movie-mean differences; 95% paired bootstrap CIs; two-sided Wilcoxon; Holm across16. Selector placement only, not human detector accuracy. RequestedK may exceed realizedK.

| Ablation | A − B | Mean F1 difference | 95% CI | Holm p |
|---|---|---:|---|---:|
| 1 | A_cue-speech − A_cue-both | -0.0005 | [-0.0424, 0.0383] | 1.0000 |
| 1 | A_cue-nonspeech − A_cue-both | 0.0110 | [-0.0061, 0.0282] | 1.0000 |
| 2 | AV-trigger − AV-suppress | 0.0244 | [-0.0198, 0.0658] | 1.0000 |
| 2 | AV-music-trig − AV-music-supp | 0.0030 | [-0.0136, 0.0228] | 1.0000 |
| 2 | AV-impulsive-trig − AV-impulsive-supp | -0.0049 | [-0.0218, 0.0084] | 1.0000 |
| 2 | AV-vocal_nonspeech-trig − AV-vocal_nonspeech-supp | -0.0013 | [-0.0089, 0.0051] | 1.0000 |
| 2 | AV-ambient-trig − AV-ambient-supp | -0.0017 | [-0.0051, 0.0000] | 1.0000 |
| 2 | AV-other-trig − AV-other-supp | 0.0083 | [-0.0142, 0.0360] | 1.0000 |
| 3 | A_cue-tagger − A_cue-whispertags | 0.0166 | [-0.0180, 0.0491] | 1.0000 |
| 4 | AV-snap − AV-event | 0.0369 | [0.0005, 0.0857] | 1.0000 |
| 8 | AV-memory − AV-local | 0.0159 | [-0.0257, 0.0516] | 1.0000 |
| 8 | AV-memory-pure − AV-local | 0.0043 | [-0.0319, 0.0432] | 1.0000 |
| 8 | V_mem − V | -0.0220 | [-0.0916, 0.0252] | 1.0000 |
| 6 | A_cue-music-only − A_cue | 0.0205 | [-0.0239, 0.0598] | 1.0000 |
| 6 | A_cue-impulsive-only − A_cue | -0.0052 | [-0.0577, 0.0440] | 1.0000 |
| 6 | A_cue-ambient-only − A_cue | -0.0121 | [-0.0533, 0.0258] | 1.0000 |
