# Measured gap utilization

Scope: dev; 10 frozen videos in every cell. Unique measured narration occupancy of all VAD/Whisper-union gaps; unused gaps and zero-output videos remain in denominators. Failed captions contribute no narration. This is a machine-defined speech-free measure, not human-validated silence.

| Selector | Prior | Model | Timeline | Renders | Zero-render videos | Unique gap use | All-gap seconds | Excess in-gap layers (s) |
|---|---|---|---|---:|---:|---:|---:|---:|
| U | P0 | qwen | True | 54 | 2 | 0.4464 | 892.825 | 65.851 |
| U | P1 | qwen | True | 55 | 2 | 0.5309 | 892.825 | 171.360 |
| V | P0 | qwen | True | 66 | 1 | 0.3762 | 892.825 | 153.519 |
| V | P1 | qwen | True | 61 | 1 | 0.4346 | 892.825 | 201.086 |
| A_cue | P0 | qwen | True | 85 | 0 | 0.5193 | 892.825 | 191.371 |
| A_cue | P1 | qwen | True | 88 | 0 | 0.6006 | 892.825 | 334.828 |
| AV | P0 | qwen | True | 83 | 0 | 0.4483 | 892.825 | 170.073 |
| AV | P1 | qwen | True | 82 | 0 | 0.5270 | 892.825 | 341.069 |
| O | P0 | qwen | True | 65 | 1 | 0.4374 | 892.825 | 105.185 |
| O | P1 | qwen | True | 66 | 1 | 0.5397 | 892.825 | 184.284 |
