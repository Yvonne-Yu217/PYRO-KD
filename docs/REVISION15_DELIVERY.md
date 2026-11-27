# Revision 1.5 local experiment delivery

This record covers the authorized local work. It does **not** certify completion of the paid-model, human-validation, or unavailable Cue2Narrate requirements. API spending is $0. No human annotations were invented.

## Scope and source

REFRAMED is the primary evaluation: the frozen 30-clip subset across ten movies, not all downloaded videos. Twenty core conditions remain in the matrix; ten local Qwen conditions ran and ten API conditions are deferred. Additional no-context Qwen and native-audio Omni conditions reuse the frozen A_cue/P0 selection. Proposal A means A_cue.

Formal AD and EgoSchema use immutable source `09abb6c0512e5e97549ad4e66d3bae05a68c42c8`. Later long-video conditions use immutable `a7e0e1a`; original visual controls retain `1f5fb60` provenance. Reporting and verified fixes are on `work/revision15-results`. No running inference source was edited.

A_cue and AV used bridged speech cues, whose semantic-shift and deictic methods
differ from the native ones; see [the audio stack notes](AUDIO_STACK.md) for the
method and the affected results.

## Completed AD results

- Ten local core cells: 3,398 unique attempts, 3,108 valid captions, 290 failures (256 word-budget violations; 34 token-limit failures), zero missing IDs.
- Two additional context/native cells: 768 attempts, 675 valid captions, 93 explicit failures.
- Total formal AD: 4,166 attempts, 3,783 valid captions and 383 failures. Every valid caption has an actual 1.2x Piper render; 3,693 individually fit the machine-estimated speech-free bounds.
- Individual fit is not joint narration schedulability. Core conditions accumulate 7,651.939 seconds of excess narration self-overlap and 17.807 seconds of original-speech union collision. These totals span separate experimental conditions.
- All 31 selector configurations cover the frozen 30 clips. Four of 17 newly declared core temporal contrasts survive Holm correction; no P1 minus P0 contrast does. The contrast family was declared after early U output existed and is not described as preregistered.
- Under identical A_cue/P0 prepared moments, timeline versus no context improves mean temporal F1 by 0.0519 (pointwise 95% movie-bootstrap CI [0.0107, 0.1189], Holm p=0.0410). This measures valid-output availability at fixed times, not semantic superiority. Native audio also changes model architecture.

Tables: [core](results/reframed-formal-core/core-results.md), [context/native](results/reframed-formal-ablation5/ablation5-results.md), [paired core contrasts](results/reframed-core-temporal/paired-contrasts.md), [selector ablations](results/reframed-selector-ablations/), and [measured gap use](results/reframed-formal-gap-utilization/).

## QA

Long-video QA is complete: 12 audio-bearing Video-MME videos, 36 frozen questions in each of ten conditions, 360 valid predictions and no missing/invalid/generation failures. All 13 paired contrasts have Holm p=1.0. Matched-frame timeline controls yield A_cue 20/36 versus no-timeline 21/36, and AV 18/36 versus no-timeline 20/36. Audio-selected conditions have lower realized frame counts than visual controls; these are constrained-budget diagnostics, not equal-budget causal comparisons.

[Long-QA table](results/video-mme-long/qa-results.md), [paired comparisons](results/video-mme-long/paired-contrasts.md), and [accuracy by length](results/video-mme-long/qa-by-length.pdf).

EgoSchema is complete: 1,470/1,470 valid predictions, with no missing, invalid or generation failures. On 490 heldout questions, U scores 319/490 (65.10%), V_local 311/490 (63.47%), and V_memory 309/490 (63.06%). The three Holm-adjusted p-values are 0.5443, 0.5003 and 0.8337; no visual novelty advantage is detected. Ten pilot questions are excluded. All videos are silent, so this supports visual-only conclusions. See the [EgoSchema table](results/egoschema-heldout/qa-results.md) and [paired contrasts](results/egoschema-heldout/paired-contrasts.md).

## Verification and reproducibility

The immutable formal source passed 817 tests before execution. The accumulated results branch passed 847 tests, with three skips, two deprecation warnings and 5,974 subtests. Full log: `/scratch/yy5919/pyro-results-final-tests.log`.

The [completion inventory](verification/revision15-completion.json) passes with zero errors and 175 bound artifacts. The [cached-table reproduction](verification/reproduction.json) completed all 12 commands and generated 51 artifacts. Nine canonical/reproduced CSV comparisons are byte-identical, including all QA scores and paired contrasts ([comparison evidence](verification/reproduction-comparison.json)). Commands and guards are described in [completion verification](verification/REVISION15_COMPLETION.md) and [table reproduction](verification/REPRODUCE_REVISION15_TABLES.md). Inference commands and frozen configs are linked from [results reproduction](RESULTS_REPRODUCTION.md).

Raw predictions and per-example metrics are rooted at `/scratch/yy5919/pyro-resume-results`; derived inputs and rendered audio at `/scratch/yy5919/pyro-kd-derived`. The compact retention target is `/home/yy5919/pyro-kd-artifacts/revision15-20261123/artifacts.tar.gz`; publication is verified separately in `docs/verification/retention.json` after every archive member passes checksum validation. Models, media, caches and large outputs are excluded from Git.

Primary user-owned allocation: 40964. Agent-owned secondary: 41197. CPU recovery and preparation: 41038_{0,2,3}, 41216_1, 41217. Conditional CPU continuation controller: 41245 completed successfully and submitted no GPU because both secondary outputs were complete. Secondary 41197 completed successfully and released its GPU; user-owned 40964 was left intact after inference ended. [Compute accounting](results/compute-accounting.md) separates allocation hours, process wall time and model-call latency.

## Remaining requirements and limits

Paid API cells and rubric judging/calibration await budget authorization. Human audio-event labels, detector precision/recall and 60 blinded peer comparisons await real annotations. Cue2Narrate access/comparable baseline remains unavailable. Machine detector proxies and agent qualitative reviews do not replace these requirements.

Global signed cue weights have verified five-fold development CV across ten distinct movies. Superclass variants are fixed component ablations, not learned/CV-selected superclass weights; VN and overlap-penalty weights were not grid axes. No tuning was performed on test results. See [development evidence](verification/development-cv-evidence.json) and [requirements audit](verification/final-requirements-audit.md).

The [paper](report/main.pdf) and [12-slide deck](slides/main.pdf) are research artifacts for team review. Human contribution attribution, the team code walkthrough and course submission/presentation remain team responsibilities.
