# Long-video selection budget audit

Read-only audit before audio-aware QA inference. All 12 media/VAD/cue/timeline
source checks passed. Four frames per selected moment and minimum separation
of two seconds are retained. Target K is5/minute, but safe placement limits
achieved K. These are constrained-budget diagnostics; they cannot establish
an equal-realized-frame-budget comparison with U/V.

| Video | Requested K | A flat selected | Maximum eligible candidate capacity |
|---|---:|---:|---:|
|604|153|14|14|
|616|220|220|432|
|710|165|17|18|
|729|230|63|67|
|753|174|19|19|
|763|161|161|206|
|802|243|157|163|
|808|175|140|142|
|857|166|166|862|
|864|201|201|336|
|891|253|166|168|
|898|173|173|761|

Total1497/2314 moments,5988frames versus9256forU. VAD marks97.4%of604 as
speech, leaving nine usable gaps totaling34.5seconds;753 is96.3%speech.
Safe-gap capacity dominates; greedy packing explains only15additional missing
selections across the seven deficient videos. These are algorithmic VAD outputs,
not human speech annotations. No arbitrary filling, altered feasibility settings,
or held-out outcome tuning was used. The global timeline is20speech+20event
entries for every video. Optional diarization remains unavailable; A uses no
visual features. Human detector validation remains deferred.
