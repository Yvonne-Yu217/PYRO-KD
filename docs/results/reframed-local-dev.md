# Local development pilot

Development diagnostics only. Failed predictions and zero-output clips remain in coverage. BERTScore/OpenCLIP use temporal matches; TTS is measured at 1.2x.

| Selector | Prompt | Valid / expected | F1 | BERTScore | OpenCLIP | TTS fit / rendered | Speech overlap (s) |
|---|---|---:|---:|---:|---:|---:|---:|
| AV | P0 | 83/95 | 0.462 | 0.866 | 0.445 | 81/83 | 0.322 |
| AV | P1 | 82/95 | 0.476 | 0.862 | 0.424 | 77/82 | 0.618 |
| A_cue | P0 | 85/95 | 0.477 | 0.864 | 0.435 | 85/85 | 0.000 |
| A_cue | P1 | 88/95 | 0.510 | 0.863 | 0.421 | 88/88 | 0.000 |
| O | P0 | 65/71 | 0.626 | 0.866 | 0.451 | 64/65 | 0.026 |
| O | P1 | 66/71 | 0.631 | 0.862 | 0.419 | 63/66 | 0.602 |
| U | P0 | 54/66 | 0.223 | 0.866 | 0.445 | 53/54 | 0.039 |
| U | P1 | 55/66 | 0.213 | 0.859 | 0.405 | 55/55 | 0.000 |
| V | P0 | 66/75 | 0.291 | 0.865 | 0.450 | 64/66 | 0.435 |
| V | P1 | 61/75 | 0.278 | 0.859 | 0.441 | 58/61 | 0.374 |
