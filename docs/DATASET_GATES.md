# Revision 1.5 dataset gates — 2026-11-22

Current transfer priority is MTG, EgoSchema and REFRAMED on the CPU allocation.
No additional bulk dataset download has been started.

## Cue2Narrate

The [paper](https://arxiv.org/html/2609.01725v1) names the benchmark LongLSMDC
and the method Cue2Narrate. It uses paired visual/spoken windows and trained
components, so published localization mAP is not interchangeable with our
fixed-budget ±2-second placement P/R. This requires an explicit comparable
protocol before reporting baseline gains.

The paper links [multimodal-ai-lab/Cue2Narrate](https://github.com/multimodal-ai-lab/Cue2Narrate).
GitHub API inspection currently lists only .gitignore, LICENSE and README.md;
README blob eab5ab8365bd534db74b18ff7a23aa8e5db78475 contains only the title.
Data/code access through the official linked repository is therefore not yet
established. Further release-plan inspection is pending; do not claim that
all possible access routes were exhausted or silently drop this comparison.

## Long-form QA candidate

[Video-MME's official README](https://github.com/MME-Benchmarks/Video-MME/blob/main/README.md)
describes long videos of 30–60 minutes, with audio among supported modalities.
It specifies academic-only use and restrictions on redistribution/copying.
Its subtitle evaluation uses subtitles corresponding to selected frames, so
a full audio timeline condition must be separately labeled rather than
presented as the unchanged official subtitle protocol.

Pending before acquisition: inspect official linked release and exact file
sizes, choose and freeze a small independent subset, resolve applicable use
terms, and verify audio on actual downloaded media. Paper statements alone
do not pass the audio-availability gate. Existing three-dataset transfer takes
precedence; no fallback has been invoked.

## Existing datasets

MTG: 413 videos transferred with SHA256 checks passing; 20 source failures.
EgoSchema: full 5031-video source manifest, 500 public labels; 51 preexisting
local videos independently probed without audio. Full transfer/verification
pending. Use as visual-only diagnostic unless actual source audio exists.
REFRAMED: 1983 source-available videos; 40 failures all in training. Source
validation/test coverage 96/91. Local full transfer/verification pending.

Human audio-detector labels and peer ratings remain required external inputs;
none have been synthesized or represented as collected.
