# QA results

12audio-bearing long videos,36frozen questions percondition. A/AV are constrained-budget diagnostics; frames differ fromvisualcontrols. No whole-benchmark or equal-budget claim.

Pointwise95% percentile bootstrap over videos,10000 resamples,seed0; not simultaneous intervals. Failed/missing/invalid predictions count incorrect.

| Condition | Correct / expected | Accuracy | 95% CI | Generation error / invalid / missing |
|---|---:|---:|---|---|
| U-none | 18/36 | 0.500 | [0.333, 0.667] | 0 / 0 / 0 |
| V-local-none | 17/36 | 0.472 | [0.278, 0.667] | 0 / 0 / 0 |
| U-flat-timeline | 16/36 | 0.444 | [0.250, 0.639] | 0 / 0 / 0 |
| V_local-flat-timeline | 16/36 | 0.444 | [0.278, 0.639] | 0 / 0 / 0 |
| A_cue-flat-timeline | 20/36 | 0.556 | [0.361, 0.750] | 0 / 0 / 0 |
| A_cue-hierarchical-timeline | 20/36 | 0.556 | [0.361, 0.750] | 0 / 0 / 0 |
| AV-flat-timeline | 18/36 | 0.500 | [0.306, 0.694] | 0 / 0 / 0 |
| AV-hierarchical-timeline | 18/36 | 0.500 | [0.306, 0.694] | 0 / 0 / 0 |
| A_cue-flat-none | 21/36 | 0.583 | [0.389, 0.778] | 0 / 0 / 0 |
| AV-flat-none | 20/36 | 0.556 | [0.361, 0.750] | 0 / 0 / 0 |
