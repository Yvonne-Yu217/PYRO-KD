# Prompt priors

P0 retains the original plain current-video instruction. P1 prepends a
researcher-written guideline summary (observable facts, objective present
tense, concision, no unsupported identities, no speech/audio repetition,
and gap word-limit compliance), followed by exactly three professional AD
examples. No claim is made that this wording is a quoted industry standard.
The current-video portion is identical between P0 and P1.

Examples come from REFRAMED training US AD, merging cues separated by at most
0.5 seconds. Before any model results, the deterministic rule selects clips
by ascending video ID, then the first merged cue with 6–25 whitespace words
and terminal punctuation, taking one example per movie. All validation/test
movie IDs are excluded. Held-out text is not used for selection. The full
source-file hashes, selection rule, movie/video IDs, track, intervals and
texts are saved in the derived bundle. Existing bundles cannot be overwritten.
This is a fixed style-example selection rule, not a claim that examples are
optimal or that P1 improves model quality.

```bash
source hpc/config.sh
in_env ro 'python scripts/prepare_prompt_exemplars.py --data-dir /scratch/yy5919/pyro-kd-derived/datasets/reframed --output /scratch/yy5919/pyro-kd-derived/prompts/reframed-train-p1.json'
```

The current frozen bundle SHA256 is
`a51d9076f3dfeb1bb50b1c209aba3b3e2a789843c98eb7dcb5aeaba4db724faf`.
For P1, add these flags to a generation command:

```bash
--prior P1 --exemplars /scratch/yy5919/pyro-kd-derived/prompts/reframed-train-p1.json
```

The generator requires the bundle for P1 and rejects it for P0. Each output
row records its exemplar hash; the attempt metadata retains the path, texts
and hash. Resuming into another prior or exemplar hash is rejected. Use
separate output files for distinct conditions.

CPU validation: 63 tests and 17 subtests passed across prompt, pipeline and
generation truncation checks. Four actual MTG inputs passed the P1 dry run,
opening frames and building prompts. Evidence is in
`outputs/verification/{pytest-prompt-priors,prepare-p1-exemplars,mtg-real-dev-p1-dry-run}.log`.
No P1 model inference or quality comparison has run.

Correction: the earlier requirements inventory said “validation exemplars.”
The authoritative Revision 1.5 source only specifies three professional
examples. Training-only examples are the implementation choice; the source
requirement was not changed.

## Audio context controls

`run_ad_vlm.py --audio-context` accepts `legacy` (existing annotation transcript
behavior), `none` (no audio text), or `timeline` (derived ASR/event estimates).
The backward-compatible default is legacy; new timeline experiments must set
the mode explicitly. P0 and P1 share the same selected audio context. Timeline
mode ignores the old annotation transcript field and rejects missing timeline
data before loading a model. Empty detected context is explicit and is not a
substitute for missing audio. Separate outputs are required for different
modes or changed timeline hashes; resume checks enforce this.

Prepare a development-only exploratory timeline input:

```bash
source hpc/config.sh
in_env ro 'python scripts/attach_audio_timeline.py --input /scratch/yy5919/pyro-kd-derived/moments/reframed_U_dev_revision15.jsonl --output /scratch/yy5919/pyro-kd-derived/moments/reframed_U_dev_timeline_exploratory.jsonl --cache-dir /scratch/yy5919/pyro-kd-derived/cache/reframed --min-score 0.5 --min-rise 0.1'
```

The tool requires both current speech and event caches from the same WAV and
VAD gates. It uses class-rise local maxima, excludes speech/silence labels,
and attaches nearby detected speech/events within15s each side of a moment.
Speech words are grouped into at most12-word phrases, breaking at gaps>0.7s.
At most40 entries nearest the moment are retained, then sorted chronologically;
omissions and feature/config hashes are recorded. Events are model estimates,
not ground truth. The explicit0.5 score/0.1 rise thresholds are initial
exploratory settings, not tuned or detector-validated final settings.

Actual development attachment preserved all original fields of66 moments:
129 speech entries and92 event estimates, no omitted entries. Both P0/P1
four-input CPU dry runs passed with `--audio-context timeline`. Targeted
verification passed68 tests/17 subtests. Evidence:
outputs/verification/reframed-dev-timeline-check.json,
reframed-timeline-{p0,p1}-dry-run.log, and pytest-audio-timeline.log.
No caption quality, QA utility or detector-accuracy result is claimed.

## Development variants (not part of the frozen matrix)

P0 and P1 are unchanged. To find which part of P1 drives its longer, more often
over-budget captions, `run_ad_vlm.py` also accepts, on development inputs only:

| Prior | Guidelines | Examples | Question |
|---|---|---|---|
| P1G | P1's | none | Do the guidelines alone lengthen captions? |
| P1E | none | three | Do the examples alone change length or leak content? |
| P1S | one-short-sentence instruction | three | Does a length target recover the loss? |

P1E and P1S need `--exemplars`; P1G rejects it. Use one output file per prior, then
compare moment by moment:

```bash
python scripts/compare_prior_outputs.py --baseline <p0.jsonl> --variant <p1.jsonl>
```
