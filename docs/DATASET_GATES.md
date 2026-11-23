# Revision 1.5 dataset gates — 2026-11-22

MTG, EgoSchema and REFRAMED transfer to Cloud Bursting is complete.
Long-form access inspection is now proceeding on the CPU allocation.

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
EgoSchema: all5031 videos transferred and probed without audio;500public labels.
Size/stream/duration checks passed for all files. Use as visual-only diagnostic unless actual source audio exists.
REFRAMED: 1983 source-available videos; 40 failures all in training. Source
validation/test coverage 96/91. Local transfer/size/stream/duration verification complete for all1983 files.

Human audio-detector labels and peer ratings remain required external inputs;
none have been synthesized or represented as collected.

## Video-MME pinned access and label-blind subset — 2026-11-23

The official README links the LMMS release, now redirected to
[lmms-eval/Video-MME](https://huggingface.co/datasets/lmms-eval/Video-MME).
Pinned dataset revision: ead1408f75b618502df9a1d8e0950166bf0a2a0b.
The20 video ZIPs total100,994,322,505bytes; annotation Parquet is405,167bytes.
Only metadata and archive-index byte ranges are being acquired at this step,
not the entire101GB release. Exact HTTP206 ranges verified on ZIP01. Range
reader rejects ignored/mismatched ranges before reading a whole archive.
Academic local research use follows the official access route; do not publish
or redistribute raw videos/annotations.

Frozen subset: configs/video_mme_long_subset.json. Select2videos/domain across
6domains from300long videos by ascending SHA256(seed0:video_id). Selection
reads only video_id/videoID/duration/domain columns, never questions/options/
answers. This gives12held-out evaluation videos; no performance-based or
availability-based replacement. Retain failures explicitly. This is a small
long-form subset, not a claim to evaluate the full benchmark. The web dataset
viewer incidentally displayed short-video example questions/answers before
column-restricted selection; those short videos are not eligible for this
long-only subset and no long question/answer content was inspected.

Metadata and release inventory live under
/scratch/yy5919/pyro-kd-derived/video-mme. A separate system-site-packages
pyro-data-env supplies pyarrow26.0.1; shared overlay unchanged. Archive indexing
is in progress; selective video extraction and actual audio/duration checks
remain required before this gate passes. This inspection does not establish
30–60minute duration or usable audio for any selected file yet.

Archive inventory checkpoint:16/20archives successfully indexed,760members,
81,514bytes of index transfer. Eleven frozen IDs mapped to2,581,429,970compressed
bytes; ID864 remains unlocated. archives17–20 raised BadZipFile in ZipFile after
transport trouble; a direct archive17 range probe also failed with URLError.
Do not infer corrupted release files: Python ZipFile can swallow OSError during
end-record reads and surface BadZipFile. The reader now wraps exhausted network
failures explicitly as transport errors. Complete parts are durable; the partial
inventory and selected-archive-members.json remain under the scratch metadata
root. No selected media payload has been downloaded, and audio gate remains open.
