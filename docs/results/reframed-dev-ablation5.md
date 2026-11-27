# Development audio-context ablation

A_cue/P0, same95 selected moments;30-second text/native context. Dev-only, no human ratings. Native model change is part of the condition. Text metrics use temporal matches of valid predictions; failed outputs stay in coverage.

| Condition | Valid / expected | F1 | BERTScore | OpenCLIP | TTS fit / rendered | Speech overlap (s) |
|---|---:|---:|---:|---:|---:|---:|
| native audio (Omni) | 62/95 | 0.408 | 0.867 | 0.459 | 62/62 | 0.000 |
| no audio context | 76/95 | 0.433 | 0.861 | 0.460 | 76/76 | 0.000 |
| text timeline | 85/95 | 0.477 | 0.864 | 0.435 | 85/85 | 0.000 |
