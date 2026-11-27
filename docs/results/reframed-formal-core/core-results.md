# Planned core matrix

Frozen30clip REFRAMED test subset; all20planned cells listed,10paidAPI cells deferred. Textmetrics are conditional on valid temporal matches; failures remain in coverage.

| selector | prior | model | timeline | status | requested_moments | selected_moments | valid | expected | failed | budget_overruns | budget_known | f1 | bertscore | openclip | tts_fit | tts_rendered |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| U | P0 | qwen | True | evaluated with failures | 388 | 388 | 255 | 283 | 28 | 22 | 277 | 0.275 | 0.863 | 0.408 | 246 | 255 |
| U | P0 | api | True | deferred: no paid authorization | — | — | — | — | — | — | — | — | — | — | — | — |
| U | P1 | qwen | True | evaluated with failures | 388 | 388 | 251 | 283 | 32 | 32 | 283 | 0.276 | 0.862 | 0.380 | 243 | 251 |
| U | P1 | api | True | deferred: no paid authorization | — | — | — | — | — | — | — | — | — | — | — | — |
| V | P0 | qwen | True | evaluated with failures | 388 | 388 | 266 | 305 | 39 | 31 | 297 | 0.351 | 0.866 | 0.400 | 258 | 266 |
| V | P0 | api | True | deferred: no paid authorization | — | — | — | — | — | — | — | — | — | — | — | — |
| V | P1 | qwen | True | evaluated with failures | 388 | 388 | 264 | 305 | 41 | 41 | 305 | 0.315 | 0.862 | 0.381 | 249 | 264 |
| V | P1 | api | True | deferred: no paid authorization | — | — | — | — | — | — | — | — | — | — | — | — |
| A_cue | P0 | qwen | True | evaluated with failures | 388 | 384 | 372 | 384 | 12 | 6 | 378 | 0.521 | 0.867 | 0.409 | 370 | 372 |
| A_cue | P0 | api | True | deferred: no paid authorization | — | — | — | — | — | — | — | — | — | — | — | — |
| A_cue | P1 | qwen | True | evaluated with failures | 388 | 384 | 372 | 384 | 12 | 12 | 384 | 0.516 | 0.864 | 0.378 | 365 | 372 |
| A_cue | P1 | api | True | deferred: no paid authorization | — | — | — | — | — | — | — | — | — | — | — | — |
| AV | P0 | qwen | True | evaluated with failures | 388 | 384 | 347 | 384 | 37 | 28 | 375 | 0.441 | 0.867 | 0.394 | 336 | 347 |
| AV | P0 | api | True | deferred: no paid authorization | — | — | — | — | — | — | — | — | — | — | — | — |
| AV | P1 | qwen | True | evaluated with failures | 388 | 384 | 341 | 384 | 43 | 43 | 384 | 0.441 | 0.865 | 0.378 | 331 | 341 |
| AV | P1 | api | True | deferred: no paid authorization | — | — | — | — | — | — | — | — | — | — | — | — |
| O | P0 | qwen | True | evaluated with failures | 388 | 366 | 322 | 343 | 21 | 16 | 338 | 0.669 | 0.867 | 0.411 | 315 | 322 |
| O | P0 | api | True | deferred: no paid authorization | — | — | — | — | — | — | — | — | — | — | — | — |
| O | P1 | qwen | True | evaluated with failures | 388 | 366 | 318 | 343 | 25 | 25 | 343 | 0.696 | 0.864 | 0.384 | 311 | 318 |
| O | P1 | api | True | deferred: no paid authorization | — | — | — | — | — | — | — | — | — | — | — | — |
