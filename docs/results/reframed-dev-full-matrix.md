# Planned core matrix

Development only:10 frozen clips. Paid API conditions deferred. Requested, selected and prepared budgets differ; no formal test claims.

| selector | prior | model | status | requested_moments | selected_moments | valid | expected | failed | budget_overruns | budget_known | f1 | bertscore | openclip | tts_fit | tts_rendered |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| U | P0 | qwen | evaluated with failures | 111 | 111 | 54 | 66 | 12 | 8 | 62 | 0.223 | 0.866 | 0.445 | 53 | 54 |
| U | P0 | api | deferred: no paid authorization | — | — | — | — | — | — | — | — | — | — | — | — |
| U | P1 | qwen | evaluated with failures | 111 | 111 | 55 | 66 | 11 | 10 | 65 | 0.213 | 0.859 | 0.405 | 55 | 55 |
| U | P1 | api | deferred: no paid authorization | — | — | — | — | — | — | — | — | — | — | — | — |
| V | P0 | qwen | evaluated with failures | 111 | 111 | 66 | 75 | 9 | 7 | 73 | 0.291 | 0.865 | 0.450 | 64 | 66 |
| V | P0 | api | deferred: no paid authorization | — | — | — | — | — | — | — | — | — | — | — | — |
| V | P1 | qwen | evaluated with failures | 111 | 111 | 61 | 75 | 14 | 12 | 73 | 0.278 | 0.859 | 0.441 | 58 | 61 |
| V | P1 | api | deferred: no paid authorization | — | — | — | — | — | — | — | — | — | — | — | — |
| A_cue | P0 | qwen | evaluated with failures | 111 | 95 | 85 | 95 | 10 | 6 | 91 | 0.477 | 0.864 | 0.435 | 85 | 85 |
| A_cue | P0 | api | deferred: no paid authorization | — | — | — | — | — | — | — | — | — | — | — | — |
| A_cue | P1 | qwen | evaluated with failures | 111 | 95 | 88 | 95 | 7 | 7 | 95 | 0.510 | 0.863 | 0.421 | 88 | 88 |
| A_cue | P1 | api | deferred: no paid authorization | — | — | — | — | — | — | — | — | — | — | — | — |
| AV | P0 | qwen | evaluated with failures | 111 | 95 | 83 | 95 | 12 | 11 | 94 | 0.462 | 0.866 | 0.445 | 81 | 83 |
| AV | P0 | api | deferred: no paid authorization | — | — | — | — | — | — | — | — | — | — | — | — |
| AV | P1 | qwen | evaluated with failures | 111 | 95 | 82 | 95 | 13 | 13 | 95 | 0.476 | 0.862 | 0.424 | 77 | 82 |
| AV | P1 | api | deferred: no paid authorization | — | — | — | — | — | — | — | — | — | — | — | — |
| O | P0 | qwen | evaluated with failures | 111 | 92 | 65 | 71 | 6 | 5 | 70 | 0.626 | 0.866 | 0.451 | 64 | 65 |
| O | P0 | api | deferred: no paid authorization | — | — | — | — | — | — | — | — | — | — | — | — |
| O | P1 | qwen | evaluated with failures | 111 | 92 | 66 | 71 | 5 | 5 | 71 | 0.631 | 0.862 | 0.419 | 63 | 66 |
| O | P1 | api | deferred: no paid authorization | — | — | — | — | — | — | — | — | — | — | — | — |
