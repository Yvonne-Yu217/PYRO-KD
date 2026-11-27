## Selector-only placement (tolerance +/-2.0 s), on the videos every core selector scored

| selector | ablation | videos | macro_F1 [95% CI] | P | R | overlap_s [95% CI] | silence_util | span | busiest_gap | recall_silent_GT | shortfall_videos | not_run |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| A_cue | core | 10 | 0.514 [0.428, 0.608] | 0.559 | 0.492 | 0.00 [0.00, 0.00] | 0.961 | 0.749 | 0.623 | 0.000 | 3 |  |
| A_cue-tagger | ablation3 | 10 | 0.514 [0.428, 0.608] | 0.559 | 0.492 | 0.00 [0.00, 0.00] | 0.961 | 0.749 | 0.623 | 0.000 | 3 |  |
| A_cue-whispertags | ablation3 | 10 | 0.512 [0.415, 0.616] | 0.555 | 0.491 | 0.00 [0.00, 0.00] | 0.959 | 0.753 | 0.637 | 0.000 | 3 |  |
