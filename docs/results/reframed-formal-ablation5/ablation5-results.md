# Audio-context ablation

Frozen30clip REFRAMED test subset, A_cue/P0, same384 prepared moments. Native condition changes model as well as input modality. Text similarity conditions on valid temporal matches; all failures remain visible.

| selector | prior | model | timeline | status | requested_moments | selected_moments | valid | expected | failed | budget_overruns | budget_known | f1 | bertscore | openclip | tts_fit | tts_rendered |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| A_cue | P0 | qwen | False | evaluated with failures | 388 | 384 | 349 | 384 | 35 | 8 | 357 | 0.469 | 0.860 | 0.411 | 345 | 349 |
| A_cue | P0 | qwen | True | evaluated with failures | 388 | 384 | 372 | 384 | 12 | 6 | 378 | 0.521 | 0.867 | 0.409 | 370 | 372 |
| A_cue | P0 | omni | False | evaluated with failures | 388 | 384 | 326 | 384 | 58 | 4 | 330 | 0.465 | 0.871 | 0.442 | 324 | 326 |
