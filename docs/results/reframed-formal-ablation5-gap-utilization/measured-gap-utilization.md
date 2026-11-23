# Measured gap utilization

Scope: test_sample; 30 frozen videos in every cell. Unique measured narration occupancy of all VAD/Whisper-union gaps; unused gaps and zero-output videos remain in denominators. Failed captions contribute no narration. This is a machine-defined speech-free measure, not human-validated silence.

| Selector | Prior | Model | Timeline | Renders | Zero-render videos | Unique gap use | All-gap seconds | Excess in-gap layers (s) |
|---|---|---|---|---:|---:|---:|---:|---:|
| A_cue | P0 | qwen | False | 349 | 1 | 0.4655 | 3739.112 | 435.994 |
| A_cue | P0 | qwen | True | 372 | 0 | 0.5456 | 3739.112 | 729.971 |
| A_cue | P0 | omni | False | 326 | 1 | 0.3822 | 3739.112 | 332.352 |
