# PYRO-KD human review workspace

An English, browser-based review form for the five-person team. Review the original clip, watch anonymous versions X and Y, score four criteria, flag overlapping speech, and export independent ratings. Progress is saved in the current browser; **Export reviews is the submission step**. There is no account, database, analytics, or automatic upload.

## Try the interface

From the repository root:

```bash
python3 -m http.server 8000 --bind 127.0.0.1 --directory review/web
```

Open **http://localhost:8000** and select **Explore a practice example**. The eight-second practice video and its spoken descriptions are synthetic. Practice exports are rejected by the study summarizer. No real test media or human labels are committed.

## Assignment design

| Reviewer | Fusion | Visual selection | Prompting | Total |
| --- | ---: | ---: | ---: | ---: |
| Yvonne | 8 | 8 | 8 | 24 |
| Daksh | 8 | 8 | 8 | 24 |
| Rishabh | 8 | 8 | 8 | 24 |
| Kiran | 8 | 8 | 8 | 24 |
| Prajanya | 8 | 8 | 8 | 24 |

There are **60 distinct comparisons and 120 independent ratings**:

- Fusion: AV/P0 versus A_cue/P0, 20 items.
- Visual selection: V/P0 versus U/P0, 20 items.
- Prompting: A_cue/P1 versus A_cue/P0, 20 items.

Use the existing 30 REFRAMED test clips from ten movies. Each clip receives two different contrasts; each movie contributes two items per contrast. Every item receives two reviewers, and each reviewer pair receives two items per contrast. X/Y orientation is balanced within each contrast and fixed for both reviewers of the same item. Assignment and orientation are seeded and reproducible.

These are **new exploratory human comparisons**, not a preregistered evaluation. Repeated ratings and clips from the same movie are dependent. The built-in summarizer reports descriptive preferences and raw reviewer agreement, not significance tests.

## Coordinator: build the real packages on HPC

The frozen videos and Piper WAVs are outside GitHub. Run these commands on a machine with the existing artifacts and `ffmpeg`/`ffprobe`. The preparation steps are CPU-only and do not regenerate model outputs or spend API credits. Full video rendering may take time; use an appropriate CPU allocation on HPC.

### 1. Export the frozen inputs

```bash
python3 review/export_hpc_inputs.py \
  --output review/private/narrations.json
```

Default artifact locations:

- Video cache: `/scratch/yy5919/pyro-kd-derived/cache/reframed-test-whisper-gated-v1`
- TTS root: `/scratch/yy5919/pyro-kd-derived/tts/formal-local-v1`

Override them with `--cache-root` and `--tts-root` if needed. Original media paths come from the frozen media manifests. The exporter verifies manifest, rendered-JSONL, and WAV hashes against the recorded artifacts, verifies valid-render counts and IDs, and takes movie IDs from the published paired-metrics file. Missing or changed inputs cause an explicit error.

The original Piper WAVs are unaccelerated. The exporter records the factor needed to reach **1.2x playback**; do not apply another speed change afterward. Every valid render is included. Failures and infeasible moments remain missing narration, including clips with zero valid renders.

### 2. Create full-clip narration versions

```bash
python3 review/render_tracks.py \
  --input review/private/narrations.json \
  --output review/generated/rendered-v1
```

This overlays each existing WAV at its original timestamp, preserving dialogue collisions and narration self-overlap. It does not repair scheduling or shorten captions. Source audio and narration each receive a fixed 0.5 gain, followed by the same 0.95 peak limiter in every version; no per-version loudness normalization is used. Narration tails beyond the clip end are trimmed and this is recorded in the render receipt. Rate the presented full-clip playback; use the original metrics for out-of-clip duration claims.

Before distribution, check a sample of the real mixes for audio levels, correct source media, and timing. These are playback composites, not new model results. Record any changes to the mix recipe and regenerate all versions consistently if the pilot reveals a problem.

### 3. Create blinded folders

```bash
python3 review/review_tool.py prepare \
  --input review/generated/rendered-v1/clips.json \
  --output review/generated/study-v1 \
  --study-id pyro-revision15-human-v1 \
  --seed 20261123
```

The builder validates frozen clip membership, copies required media into each reviewer folder under opaque hash filenames, and produces:

```text
study-v1/
  coordinator-only/
    study-key.json       # Method mapping, assignment key, and source hashes
    allocation.json
  reviewers/
    Yvonne/              # App, 24 anonymous items, and required media
    Daksh/
    Rishabh/
    Kiran/
    Prajanya/
```

**Send each person only their own folder. Never serve or share the parent directory containing the coordinator key.** Real packages contain no method names, source clip IDs, or movie IDs in their manifests. Do not give reviewers the key before they submit. A coordinator who inspects the key should record that their own ratings were not fully blinded; preferably have a non-reviewing helper handle the key, or build and distribute without opening it.

All five people can review independently. Real packages deliberately omit the synthetic practice files; do calibration using the repository practice page or separate development examples first. Package generation does not submit or send anything to teammates.

## Teammate instructions

1. Open a terminal in your assigned folder and run:

   ```bash
   python3 -m http.server 8000 --bind 127.0.0.1
   ```

2. Open **http://localhost:8000**. Your name and 24 assigned items load automatically.
3. Watch the original, then X and Y. Only one player runs at a time.
4. Rate each version from 1–5 on **accuracy, usefulness, non-redundancy, and clarity/timing**. The on-page rubric explains the scale.
5. Mark dialogue overlap and narration-on-narration overlap separately.
6. Choose **X, Y, Tie, or Cannot assess**, and explain with timestamps where relevant.
7. Confirm independent review and select **Save completed review**. Editing a completed review returns it to draft until you save it again.
8. Select **Export reviews** after each session. When all 24 are complete, send the final JSON file to the coordinator.

Use **Tie** for similarly useful outputs. Use **Cannot assess** when media or evidence is unavailable; a reason is required and numeric scores are optional in that case. Missing narration is part of the system output, not automatically a broken asset. Replay as needed and do not compare ratings with others before submission.

Local progress is isolated by study, package and reviewer. Clearing browser storage or switching devices can remove local progress; keep exported backups. **Restore saved export** checks the study/package/reviewer and asks before replacing conflicting local records. Different packages or practice exports cannot be restored into your real assignment.

There are no automated human-quality judgments. This form covers description preference, not the separate audio-event labeling/precision–recall requirement or a blind/low-vision user study.

## Collect results

Keep real exports under `review/private/` (ignored by Git). Supply one final export per person:

```bash
python3 review/review_tool.py summarize \
  --key review/generated/study-v1/coordinator-only/study-key.json \
  --exports review/private/Yvonne-reviews.json \
            review/private/Daksh-reviews.json \
            review/private/Rishabh-reviews.json \
            review/private/Kiran-reviews.json \
            review/private/Prajanya-reviews.json \
  --output review/private/summary-v1.json
```

The tool checks package identity, item assignments, duplicate reviews, required scores and overlap flags, and practice contamination. Drafts do not count as complete. It reports outstanding assignments, preferences by contrast, item-level disagreements, and agreement among paired assessable reviews. Partial collection is allowed and labeled through explicit counts. Original exports retain criterion scores, timestamps, and qualitative notes for further analysis.

## Existing mixed videos or custom source layout

You can skip rendering if complete mixed videos already exist. Supply an input JSON with a `clips` array containing all 30 frozen IDs:

```json
{
  "clips": [{
    "clip_id": "<frozen clip ID>",
    "movie_id": "<verified movie ID>",
    "original": "/absolute/path/original.mp4",
    "variants": {
      "U_P0": "/absolute/path/U-P0.mp4",
      "V_P0": "/absolute/path/V-P0.mp4",
      "A_cue_P0": "/absolute/path/A-P0.mp4",
      "AV_P0": "/absolute/path/AV-P0.mp4",
      "A_cue_P1": "/absolute/path/A-P1.mp4"
    }
  }]
}
```

Paths may also be relative to the JSON file. Files must be nonempty MP4/WebM assets. Full media decoding is checked separately by the coordinator. The builder refuses to overwrite an existing study output. Copying full videos into five portable folders requires additional disk space; estimate space before running.

For custom narration sources, `render_tracks.py` accepts `narrations` instead of `variants`: each condition maps to a list of `{"time": 12.5, "wav": "/path/line.wav", "playback_speed": 1.2}`. An empty list explicitly represents no valid narration. Use a factor of 1.0 for WAVs that already have the desired speed.

## Verification

```bash
python3 -m unittest discover -s review/tests -v
node --test review/tests/core.test.cjs
node --check review/web/app.js
```

Tests cover exact allocation balance, seeded orientation, frozen membership, opaque reviewer packages, export validation, duplicate rejection, partial completion, draft handling, and exclusion of practice data. A synthetic video/WAV smoke exercises the ffmpeg mixer. The local preview serves successfully; interactive browser QA was unavailable in the authoring environment and remains to be checked on a teammate's browser before study distribution.
