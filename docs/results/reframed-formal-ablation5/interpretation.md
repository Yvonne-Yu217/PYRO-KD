# Formal audio-context interpretation

All30 frozen test clips remain in temporal evaluation. The selector was chosen on dev, not these results. Timeline-present predictions reuse the identical core A_cue/P0 run. No inference was duplicated.

| Context | Valid / attempted | Temporal F1 | Individual TTS fit / rendered | Speech collision seconds | Excess overlapping narration seconds |
|---|---:|---:|---:|---:|---:|
| No audio context (Qwen) | 349/384 | 0.469 | 345/349 | 0.468 | 435.994 |
| Text timeline (Qwen) | 372/384 | 0.521 | 370/372 | 0.271 | 729.971 |
| Native audio (Omni) | 326/384 | 0.465 | 324/326 | 0.407 | 332.352 |

Temporal F1 measures placement of valid outputs, not their semantic correctness. Text similarities are conditional on valid temporal matches. Native contrasts change the model as well as its inputs. Individual speech fit does not prevent narration self-overlap. Movie-clustered significance analysis will follow complete core reference validation; no significance claim is made here.
