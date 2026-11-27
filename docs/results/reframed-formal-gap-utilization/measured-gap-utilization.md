# Measured gap utilization

Scope: test_sample; 30 frozen videos in every cell. Unique measured narration occupancy of all VAD/Whisper-union gaps; unused gaps and zero-output videos remain in denominators. Failed captions contribute no narration. This is a machine-defined speech-free measure, not human-validated silence.

| Selector | Prior | Model | Timeline | Renders | Zero-render videos | Unique gap use | All-gap seconds | Excess in-gap layers (s) |
|---|---|---|---|---:|---:|---:|---:|---:|
| U | P0 | qwen | True | 255 | 3 | 0.4655 | 3739.112 | 407.325 |
| U | P1 | qwen | True | 251 | 3 | 0.5418 | 3739.112 | 550.197 |
| V | P0 | qwen | True | 266 | 4 | 0.3961 | 3739.112 | 675.126 |
| V | P1 | qwen | True | 264 | 5 | 0.4475 | 3739.112 | 978.582 |
| A_cue | P0 | qwen | True | 372 | 0 | 0.5456 | 3739.112 | 729.971 |
| A_cue | P1 | qwen | True | 372 | 0 | 0.6146 | 3739.112 | 1021.798 |
| AV | P0 | qwen | True | 347 | 1 | 0.4721 | 3739.112 | 843.719 |
| AV | P1 | qwen | True | 341 | 1 | 0.5393 | 3739.112 | 1242.269 |
| O | P0 | qwen | True | 322 | 3 | 0.5214 | 3739.112 | 491.101 |
| O | P1 | qwen | True | 318 | 2 | 0.5844 | 3739.112 | 711.851 |
