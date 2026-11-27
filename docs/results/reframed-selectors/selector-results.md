# REFRAMED selector and ablation results

Frozen30clip test subset. Descriptive clip-macro temporal metrics at ±2s, before caption preparation/generation. Requested K need not equal realized K. Picks in speech and zero-budget picks remain visible; selector overlap estimates are not actual rendered speech. O uses reference timestamps and is an oracle upper bound. No detector accuracy or human-quality claim.

| selector | ablation | requested | selected | shortfall_videos | picks_in_speech | zero_budget_picks | precision | recall | f1 |
|---|---|---|---|---|---|---|---|---|---|
| U | core | 388 | 388 | 0 | 79 | 105 | 0.404 | 0.269 | 0.311 |
| V | core | 388 | 388 | 0 | 63 | 83 | 0.467 | 0.382 | 0.389 |
| A_cue | core | 388 | 384 | 2 | 0 | 0 | 0.588 | 0.493 | 0.505 |
| AV | core | 388 | 384 | 2 | 0 | 0 | 0.585 | 0.470 | 0.494 |
| O | core | 388 | 366 | 8 | 18 | 23 | 1.000 | 0.750 | 0.839 |
| A_cue-speech | ablation1 | 388 | 384 | 2 | 0 | 0 | 0.589 | 0.490 | 0.505 |
| A_cue-nonspeech | ablation1 | 388 | 384 | 2 | 0 | 0 | 0.597 | 0.504 | 0.516 |
| A_cue-both | ablation1 | 388 | 384 | 2 | 0 | 0 | 0.588 | 0.493 | 0.505 |
| AV-trigger | ablation2 | 388 | 384 | 2 | 0 | 0 | 0.585 | 0.470 | 0.494 |
| AV-suppress | ablation2 | 388 | 384 | 2 | 0 | 0 | 0.552 | 0.456 | 0.470 |
| AV-music-trig | ablation2 | 388 | 384 | 2 | 0 | 0 | 0.539 | 0.443 | 0.459 |
| AV-music-supp | ablation2 | 388 | 384 | 2 | 0 | 0 | 0.536 | 0.440 | 0.456 |
| AV-impulsive-trig | ablation2 | 388 | 384 | 2 | 0 | 0 | 0.533 | 0.439 | 0.454 |
| AV-impulsive-supp | ablation2 | 388 | 384 | 2 | 0 | 0 | 0.540 | 0.443 | 0.459 |
| AV-vocal_nonspeech-trig | ablation2 | 388 | 384 | 2 | 0 | 0 | 0.539 | 0.442 | 0.458 |
| AV-vocal_nonspeech-supp | ablation2 | 388 | 384 | 2 | 0 | 0 | 0.540 | 0.443 | 0.459 |
| AV-ambient-trig | ablation2 | 388 | 384 | 2 | 0 | 0 | 0.539 | 0.443 | 0.459 |
| AV-ambient-supp | ablation2 | 388 | 384 | 2 | 0 | 0 | 0.542 | 0.444 | 0.461 |
| AV-other-trig | ablation2 | 388 | 384 | 2 | 0 | 0 | 0.553 | 0.450 | 0.468 |
| AV-other-supp | ablation2 | 388 | 384 | 2 | 0 | 0 | 0.539 | 0.444 | 0.460 |
| A_cue-tagger | ablation3 | 388 | 384 | 2 | 0 | 0 | 0.588 | 0.493 | 0.505 |
| A_cue-whispertags | ablation3 | 388 | 384 | 2 | 0 | 0 | 0.573 | 0.473 | 0.489 |
| AV-snap | ablation4 | 388 | 384 | 2 | 0 | 0 | 0.585 | 0.470 | 0.494 |
| AV-event | ablation4 | 388 | 388 | 0 | 12 | 14 | 0.526 | 0.476 | 0.457 |
| A_cue-music-only | ablation6 | 388 | 384 | 2 | 0 | 0 | 0.609 | 0.510 | 0.526 |
| A_cue-impulsive-only | ablation6 | 388 | 384 | 2 | 0 | 0 | 0.578 | 0.487 | 0.500 |
| A_cue-ambient-only | ablation6 | 388 | 384 | 2 | 0 | 0 | 0.571 | 0.481 | 0.493 |
| AV-memory | ablation8 | 388 | 384 | 2 | 0 | 0 | 0.585 | 0.470 | 0.494 |
| AV-memory-pure | ablation8 | 388 | 384 | 2 | 0 | 0 | 0.564 | 0.469 | 0.482 |
| AV-local | ablation8 | 388 | 384 | 2 | 0 | 0 | 0.561 | 0.460 | 0.478 |
| V_mem | ablation8 | 388 | 388 | 0 | 64 | 78 | 0.450 | 0.348 | 0.367 |
