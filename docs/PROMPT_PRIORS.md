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
