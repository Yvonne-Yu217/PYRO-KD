# Evaluation components

Report the components separately; there is no combined project-wide score.

| Component | Question / meaning | Implementation |
| --- | --- | --- |
| Temporal Precision/Recall/F1 | Did we choose the right moment? One-to-one matching of professional AD start times, primarily at +/-2 seconds. | `metrics.py` |
| BERTScore | Token-level semantic correspondence between temporally matched generated and professional AD. | `bertscore.py` |
| Sentence cosine | Sentence-level semantic correspondence for the same matched pairs. | `cosine_similarity.py` |
| Rubric | Qualitative AD quality on four MTG-AD / VideoA11y dimensions, each rated 1-5. | `rubric.py` |

Dialogue redundancy (MTG-style **does-not-restate**) and **TTS overlap / gap fit**
must be reported as separate metrics. They are not implemented by this rubric
and must not be folded into its four ratings.

## Evaluate a generation run

```bash
uv run --no-sync python scripts/evaluate_ad.py \
  --input outputs/qwen3-vl-8b/mtg-ad_U_dev_P0.jsonl \
  --data-dir data/mtg-ad --split validation \
  --output outputs/qwen3-vl-8b/mtg-ad_U_dev_P0.evaluation.json
```

Use `--split test` for test annotations. `--tolerance` defaults to `2.0`
seconds (inclusive). `--lang` and `--bert-model` configure BERTScore;
`--cosine-model` defaults to `sentence-transformers/all-mpnet-base-v2`.
The script does not download datasets, run selectors/generation, or call rubric
providers. Real semantic scoring requires the declared model dependencies and
may download model weights on first use.

The callable `scripts.evaluate_ad.evaluate_ad(generated_path, data_dir, split,
...)` returns the same JSON-compatible report. Tests and library callers may
inject `bert_scorer` and `encoder` to avoid model loading. Annotations are loaded
through `mtg_ad.load_split`; its existing `MTGVideo.ad` items are used directly.
Each video is matched once. Both semantic functions receive the identical
`TemporalMatch` list/objects; only those pairs enter the semantic scorers.
The default BERTScore model and sentence encoder are loaded lazily on the first
matched pair and reused across every video in the run.

The report contains `videos`, `aggregate`, input/model metadata, scope/coverage,
and generation/semantic errors. Generated-item order and row metadata are
retained so each match index identifies its source entry. Successful rows need
a nonblank string `video_id`, finite nonnegative numeric `time`, and nonblank
string `output`. Rows containing an `error` key still need `video_id`/`time`;
they are counted/reported but excluded from predictions and semantic scoring.
An error followed by a successful retry is allowed and its error attempt is
still reported. Repeated successful IDs are rejected to avoid double counting.
Blank JSONL lines are ignored; malformed rows fail with a file/line diagnostic.

By default, evaluate only IDs represented in the input, including error-only
videos. This supports dev subsets of the larger validation split. Other split
IDs are listed in `coverage.split_video_ids_without_input`; use `--all-videos`
to include them as empty predictions (their GT entries become false negatives).
Unknown input video IDs are rejected before scoring. One experiment is expected
per file: conflicting selector/model/model_id/prior/quant values are rejected
instead of pooling different runs.

Dataset temporal TP/FP/FN are sums over evaluated videos; Precision/Recall/F1
are micro scores calculated from those totals. `total_matched_pairs` is summed
TP, and mean temporal error pools all matches. BERTScore P/R/F1 means and cosine
mean/median pool scored matched pairs, rather than averaging per-video means.
Unavailable scores remain `null`; genuine zero scores remain zero. Each metric
reports its scored-pair count, which can be below total matches if model calls
fail. Both-empty temporal inputs have undefined (`null`) P/R/F1, as in the
existing metrics; empty predictions with nonempty GT have zero P/R/F1.

Semantic failures are recorded per video/metric while preserving temporal and
other successful metric results. The CLI saves the report and returns status 1
if any generation-error rows or semantic failures exist; invalid inputs return
status 2. A completely empty input is labeled `empty_input`. Failed generations
are excluded by design, so these temporal counts describe successfully generated
AD; isolated selector evaluation still needs the complete selector output.
There is no mAP, combined project score, or rubric networking in this script.

## Rubric definitions

These definitions preserve the terminology and intent of Google's
[MTG-AD VideoA11y-Ref evaluator](https://github.com/google/mtg-ad/blob/98854450bee6241f3e1c17eb819a9d5f6ad4ff12/evaluation/metrics.py#L378)
and [Quality-RF evaluator](https://github.com/google/mtg-ad/blob/98854450bee6241f3e1c17eb819a9d5f6ad4ff12/evaluation/reference_free.py).
The definitions are also stored in `rubric.RUBRIC_DEFINITIONS`:

- **Descriptive:** Whether the AD provides useful and sufficiently specific visual information about important people, objects, actions, settings, and relevant properties.
- **Objective:** Whether the AD reports observable visual information without speculation, unsupported interpretation, opinions, or invented relationships/attributes.
- **Accurate:** Whether described entities, actions, properties, spatial relations, and other visual facts are correct and grounded.
- **Clear:** Whether the AD is understandable, coherent, uses unambiguous references, and is easy for a blind or low-vision listener to follow.

## Calling the rubric

```python
from src.evaluation.rubric import evaluate_rubric

# my_judge is supplied by the caller: judge(prompt: str) -> str.
result = evaluate_rubric(
    "A woman waves at the camera.",
    ground_truth_description="A woman waves.",
    transcript=None,
    judge=my_judge,
)
```

The injected judge receives a versioned prompt and returns text containing a
single JSON object, for example:

```json
{"descriptive": 4, "objective": 5, "accurate": 4, "clear": 5, "rationale": "Explain the evidence and its limits."}
```

Raw JSON, markdown-fenced JSON, and prose surrounding one JSON object are
accepted. Title-case field names and MTG-AD's `Reason` alias are normalized.
Each score must be a finite JSON number in [1, 5]; fractional values are
allowed. Numeric strings, booleans, missing scores, duplicate keys, multiple
objects, and malformed JSON are rejected. A nonempty rationale is required.
Judge exceptions, explicit error responses, and invalid output raise
`RubricEvaluationError`; they never become zero ratings. `mean_score` is
calculated locally as convenience metadata over the four ratings only.

## Modes and evidence limits

- **Reference-based:** Supplying nonblank professional GT AD selects
  `reference_based`. Ratings follow VideoA11y-Ref's comparison with the
  professional description. The reference may be incomplete, and agreement
  with it does not independently verify every visual fact.
- **Reference-free / transcript-aware:** Omitting GT (or supplying blank GT)
  selects `reference_free`. A transcript is optional; it supplies spoken
  context. Transcript alone cannot establish visual factual accuracy. All
  four ratings are provisional text-only judgments, particularly Accurate.
  The judge should return an error if it cannot justify provisional ratings,
  rather than guessing or treating missing evidence as a correct/incorrect fact.

Unlike MTG-AD's video-grounded Quality-RF evaluator, this module does not pass
video to the judge. Both modes return `accuracy_verified: false` and an
explicit `limitations` field. These modes should not be pooled or labeled as
video-grounded accuracy measurements.

The response includes `mode` and `rubric_version`. The prompt is deterministic
for fixed inputs, but judge responses need not be. For reproducible experiments,
the caller should record inputs, prompt/version, judge model/version, generation
settings, and raw responses, and cache/reuse responses as appropriate. Provider
adapters, API calls, caching, and the final CLI are outside this module.

## Offline tests

```bash
uv run --offline --no-sync python -m unittest discover -s tests -v
```

All normal tests use injected/mock judges and encoders, with no network calls
or pretrained-model downloads. `--no-sync` permits these standard-library tests
to run in the existing macOS environment despite the project's CUDA PyTorch
wheel incompatibility; it does not validate real model/provider integration.
