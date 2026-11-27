# Authoritative Revision 1.5 requirements

Source: Google document 1dYGXJJ-MJxmJtZV02U4IrkQVXLPTdev9soelen7Ltyw,
tab **t.ickd261ntenp**, titled **Revision 1.5**. Retrieved 2026-11-22.
Full text: PROPOSAL_REVISION_1_5.txt. An earlier agent audit read an older
tab by mistake; the current user URL correctly selects Revision1.5. The older
audit is retained as an error record, not the current specification. No
experiments were run under that mistaken scope.

The numbered list below preserves the original requirements inventory; its
historical progress notes are not current execution status. See
[final requirements audit](verification/final-requirements-audit.md) and
[experiment status](EXPERIMENT_STATUS.md) for completed evidence and remaining
work. Current execution uses the proposal-permitted30-clip REFRAMED scope, ten
local core cells with ten paid API cells explicitly deferred, and ongoing
EgoSchema completion. Passing unit tests is not experiment evidence.

1. Pre-commitment gates: inspect EgoSchema audio; verify Cue2Narrate access and
   comparable official baselines; verify audio, license and download size for
   a long-form QA benchmark (Video-MME long or comparable); obtain real human
   event labels on about ten ten-minute movie clips and measure detector P/R.
   Apply documented fallback only after its gate fails, recording evidence.
2. Three audio streams: Silero VAD; word-timestamped Whisper restricted to
   VAD-positive regions; AudioSet tagger plus CLAP embeddings. Cache features
   by content hash. Existing gap-energy A is only a historical baseline.
3. Cue tracks: non-speech posterior onsets excluding speech/silence classes,
   CLAP changes, loudness deltas/onset strength; speech semantic/deictic cues
   (optional diarization); scene transitions and running-memory OpenCLIP
   novelty; gap feasibility and speech-overlap penalty.
4. Two-stage candidates and gap snapping with bounded event lag; shared K,
   2-second separation, four frames over two seconds, floor(gap*3) word cap,
   identical timeline format and temperature 0. Tune five score weights on
   dev only, with cross-validation for signed/per-superclass event weights.
5. Core matrix: U/V/A/AV/O x P0/P1 x open Qwen-class/API model = 20.
   P1 uses guidelines and three professional exemplars (the source does not specify a split). API needs
   user spending cap. Preserve full local matrix while API is blocked.
6. Eight ablations: speech vs events vs both; trigger/suppress global and
   per-class; Whisper event baseline vs tagger/CLAP; event-time vs gap-snap;
   timeline absent/present/audio-native; per-class event effects; flat vs
   hierarchical budgets; running vs local novelty. Whisper remains VAD-gated.
7. Long videos: scene/silence/music chunking, event-density budget allocation
   with quiet-segment floor, running selected-moment memory; evaluate about
   30–60 minute audio-bearing videos and report accuracy against length.
8. REFRAMED/Cue2Narrate temporal P/R within 2 seconds, baseline comparability,
   actual rendered TTS overlap at 1.2x using VAD and Whisper, silence usage,
   OpenCLIP and BERTScore, calibrated five-axis judge with audio redundancy.
9. Diagnostics: hand-label detector P/R, audio-onset/AD correlation per class,
   signed-weight effects, silent-visual-event coverage, failure analyses.
10. EgoSchema U/V visual-only if silent; audio-bearing long QA with timeline
    on/off; audio-native ablation on dev-selected top selector.
11. Human blinded peer ratings: 60 real comparisons. Do not fabricate ratings
    or detector labels. Prepare review packets and retain anonymized evidence.
12. Paired Wilcoxon and 95% bootstrap, with REFRAMED movie dependence accounted
    for. Software helpers tested (41 tests/11 subtests); real scores pending.
13. Deliver report, figures, slides, raw predictions, per-example scores,
    labeled audio subset, peer logs, coverage/failure accounting and one-step
    reproduction; freeze source/model/data provenance.

Execution order: verify data/gates and environment, implement/cache audio and
visual tracks, real ten-development-video U/P0 smoke with evaluated output,
finish selectors and freeze dev-tuned protocol, full supported matrix and
ablations, human inputs, statistics and deliverables. This was the planned execution order at the time
of the requirements inventory. The current30-clip evaluation and deferred
API/human work are explicitly recorded in the frozen protocol and final audit;
do not interpret this historical snapshot as a claim that no scope reduction
occurred. Existing official splits remain intact. New evaluation subsets must
be chosen before inspecting held-out outcomes.
