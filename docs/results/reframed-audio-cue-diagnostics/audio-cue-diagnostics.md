# Detector-defined audio-cue diagnostics

Detector-defined descriptive associations only; not causation, actual silence, human detector precision/recall, or rendered speech.

Frozen parameters: onset window ±2 s; low modeled change <0.2 within ±2 s; timeline chance sampled every0.1 s. No threshold tuning. All30clips retained.

| Superclass | Detected onsets | Clips with onset | AD-onset hit rate | Timeline chance | Lift |
|---|---:|---:|---:|---:|---:|
| music | 121 | 27 | 0.1241 | 0.0942 | 1.3170 |
| impulsive | 28 | 12 | 0.0310 | 0.0247 | 1.2583 |
| vocal_nonspeech | 27 | 15 | 0.0164 | 0.0183 | 0.8996 |
| ambient | 14 | 6 | 0.0091 | 0.0088 | 1.0361 |
| other | 125 | 23 | 0.0894 | 0.0856 | 1.0442 |

Low modeled audio-change subset: 9/548 reference starts (0.0164); this is not a count of genuinely silent events.

| Selector | Matched low-change references | Pooled recall | Clip-macro recall | Eligible clips |
|---|---:|---:|---:|---:|
| U | 5 | 0.5556 | 0.5000 | 6 |
| V | 1 | 0.1111 | 0.1667 | 6 |
| A_cue | 0 | 0.0000 | 0.0000 | 6 |
| AV | 1 | 0.1111 | 0.1667 | 6 |
| O | 5 | 0.5556 | 0.6667 | 6 |

O is an oracle upper bound. Pooled association hit rates weight reference starts; chance rates weight clip duration. Lift is descriptive and does not establish trigger/suppress causation. Human event annotations remain unavailable.
