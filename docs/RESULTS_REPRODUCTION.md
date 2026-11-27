# Revision 1.5 results and reproduction

This guide describes the frozen local experiment. It does not authorize paid API
calls. Human event labels, blinded ratings and rubric judging remain deferred.
Formal local caption generation, metrics, TTS, long QA and EgoSchema scoring are complete. Consult
[EXPERIMENT_STATUS.md](EXPERIMENT_STATUS.md) and artifact coverage before calling
any stage complete. Do not rerun a live job or alter its source worktree.

## Frozen sources and environment

| Stage | Immutable source | Configuration |
|---|---|---|
| Formal REFRAMED and Ego QA | `09abb6c` | `configs/experiments/reframed_test_local.toml`, `configs/egoschema_qa_frozen.json` |
| Long visual controls | `1f5fb60` | `configs/long_visual_qa_frozen.json` |
| Long timeline and matching A/AV no-timeline controls | `a7e0e1a` | `configs/long_audio_qa_frozen.json`, `configs/long_no_timeline_qa_frozen.json` |
| Report exports | `work/revision15-results` | scripts and frozen comparison families in that branch |

These sources are in `/scratch/yy5919/pyro-revision15-formal`,
`pyro-revision15-next`, and `pyro-revision15-long-prep` respectively. Report code
is in `pyro-revision15-records`. The original checkout and unrelated edits remain
preserved. Fetch and use separate clean worktrees for reproduction, keeping
existing experiment outputs read-only. Experiment identity checks reject source
or configuration changes under the same output directory; use a new experiment
name/output root for a genuinely new run rather than bypassing that guard.

The existing read-only Singularity environment is entered through:

```bash
source hpc/config.sh
in_env ro '/scratch/yy5919/pyro-test-env/bin/python -m pytest -q'
```

The exact formal-source suite passed817tests,3skips and5974subtests. The log is
`/scratch/yy5919/pyro-formal-source-full-tests.log`. Native audio uses
`/scratch/yy5919/pyro-audio-env/bin/python`; Video-MME metadata evaluation uses
`/scratch/yy5919/pyro-data-env/bin/python`; actual speech rendering uses
`/scratch/yy5919/pyro-tts-env/bin/python`. Environments, data and model weights
are not committed. Model snapshot paths and metric revisions are pinned in the
frozen configs; preserve or explicitly relocate those paths before new runs.

## Reproduce tables and plots from existing artifacts

Run from the records checkout after the corresponding artifact set is complete.
This does not launch inference or call APIs:

```bash
source hpc/config.sh
in_env ro 'CUDA_VISIBLE_DEVICES="" OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 python scripts/export_selector_results.py --input /scratch/yy5919/pyro-resume-results/experiments/reframed_test_local_v1/eval/selection_metrics.json --output-dir docs/results/reframed-selectors'
in_env ro 'CUDA_VISIBLE_DEVICES="" OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 python scripts/export_core_results.py --config configs/experiments/reframed_local_dev_pilot.toml --tts-root /scratch/yy5919/pyro-kd-derived/tts/local-dev-pilot-v1 --metric-overlay /scratch/yy5919/pyro-resume-results/experiments/reframed_local_dev_pilot_clip --output-dir /scratch/yy5919/pyro-resume-results/local-dev-report --deferred-model api --scope "Development pilot; not heldout results"'
in_env ro 'python scripts/plot_core_coverage.py --input /scratch/yy5919/pyro-resume-results/local-dev-report/core-results.json --output docs/results/reframed-dev-coverage --title "REFRAMED development: coverage across 10 local cells"'
```

For the formal core table, use `reframed_test_local.toml`, TTS root
`/scratch/yy5919/pyro-kd-derived/tts/formal-local-v1`, a distinct formal report
output directory, and omit the development metric overlay. Preserve
`--deferred-model api`. Check all ten local cells have complete expected IDs,
all30video denominators, and finished metric/TTS reports before publishing the
formal table. The exporter preserves the ten deferred API rows in the20-cell
matrix, and stores hashes plus failed/missing IDs in its JSON evidence.

## Inference commands and output ownership

The primary formal run uses one Qwen model across ten cells:

```bash
# From the immutable formal checkout, with a legitimate GPU allocation.
source hpc/config.sh
in_env ro 'OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 python -u scripts/run_experiment.py --config configs/experiments/reframed_test_local.toml --stage generate --runs "ad-*-qwen-*"'
```

This was bounded by a four-hour watchdog and the primary GPU lock
`/scratch/yy5919/pyro-resume-gpu.lock`. The actual core preparation counts are
U283,V305,A_cue384,AV384,O343 perprompt, or3398requests. The paid model patterns
must not be selected. Preparation and feature jobs precede inference; do not
allocate a GPU merely to wait for downloads or CPU features.

Audio-context ablation 5 uses the same frozen config and two disjoint patterns:

```bash
in_env ro 'OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 python -u scripts/run_experiment.py --config configs/experiments/reframed_test_local.toml --stage ablation5 --runs "ad-A_cue-P0-qwen-tl0-*"'
in_env ro 'OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 /scratch/yy5919/pyro-audio-env/bin/python -u scripts/run_experiment.py --config configs/experiments/reframed_test_local.toml --stage ablation5 --runs "ad-A_cue-P0-omni-*"'
```

The timeline-present condition is the identical existing coreA_cue/P0run; it is
reused, not generated twice. Native audio changes the model as well as modality.
The no-context and native outputs each expect384moments.

QA uses `scripts/run_qa_vlm.py --input INPUT --output OUTPUT --model PINNED_MODEL
--max-side SIZE --max-new-tokens 16`. SIZE is224for long QA and768for EgoSchema.
Inputs are published under `/scratch/yy5919/pyro-kd-derived/moments/` with source
hash receipts; each output has generation provenance. Long inputs are named
`video-mme-long-CONDITION-v1.jsonl`; Ego inputs are
`egoschema-heldout-{U,V_local,V_memory}-v1.jsonl`. Exact conditions and model paths
are listed in the frozen JSONs. Match original input/frame hashes for resume.

## Evaluation and statistical records

Core CPU workers evaluate finished cells while generation advances; actual TTS
uses the formal cache `/scratch/yy5919/pyro-kd-derived/cache/reframed-test`.
Development caches are not valid substitutes. Text similarity is conditional on
temporally matched valid outputs. TTS fit is perdescription; inspect narration
self-overlap separately before making playback claims.

[EGOSCHEMA_QA_PROTOCOL.md](EGOSCHEMA_QA_PROTOCOL.md) gives the exact scoring,
paired export and Holm-family commands. Its three490-question conditions retain
missing/invalid/error predictions as incorrect. Official labels are opened only
in the dedicated evaluator after prediction readiness.

Long QA uses `scripts/evaluate_video_mme_qa.py` with
`/scratch/yy5919/pyro-kd-derived/video-mme/annotations.parquet`, followed by
`scripts/export_qa_paired_metrics.py` and video-cluster analysis. The plan in
`configs/long_qa_comparisons.json` contains ten conditions and thirteen contrasts.
The scratch scoring driver `/scratch/yy5919/pyro-score-long-qa.py` records
percondition coverage, paired rows, comparison statistics and the plan hash.
Every condition expects36questions from12videos. A/AV realized frame deficits
must remain explicit in interpretation.

## Artifacts and jobs

| Evidence | Path |
|---|---|
| Formal core and ablations | `/scratch/yy5919/pyro-resume-results/experiments/reframed_test_local_v1` |
| Formal actual TTS | `/scratch/yy5919/pyro-kd-derived/tts/formal-local-v1` |
| Development core | `/scratch/yy5919/pyro-resume-results/experiments/reframed_local_dev_pilot_v1` |
| Development audio-context ablation | `/scratch/yy5919/pyro-resume-results/experiments/reframed_dev_ablation5_v1` |
| Long QA and scoring | `/scratch/yy5919/pyro-resume-results/long-qa-frozen` |
| EgoSchema heldout and scoring | `/scratch/yy5919/pyro-resume-results/egoschema-heldout` |
| Human-label requirements | Deferred; no fabricated artifact |

Historical primary allocation40964 ran formal core and its ready queue. Secondary41197
ran independent prepared tasks with separate output ownership and a four-hour
allocation cap. Ego CPU preparation recovered through41038_{0,2,3},41216_1
and41217; the earlier41198dependency was replaced. These are recorded historical job IDs, not instructions
to duplicate submission. Inspect `squeue`, `sacct`, logs, provenance and actual
coverage to determine current state. No dummy keepalive workloads are permitted.

## Report PDF

The report contains completed local-result tables and remains for team review
until human authorship/contributions and original track declaration are confirmed. Build with the existing container TeX:

```bash
source hpc/config.sh
in_env ro 'cd docs/report && pdflatex -interaction=nonstopmode -halt-on-error main.tex && pdflatex -interaction=nonstopmode -halt-on-error main.tex'
```

See [report/README.md](report/README.md) for verification and remaining edits.

## Completed formal AD exports

Canonical local core results are `docs/results/reframed-formal-core/core-results.json`
and its CSV/Markdown/coverage figure. There are20planned rows,10local evaluated
and10APIdeferred;3398attempts,3108valid,3024individualspeechfits. The complete
movie-clustered core comparison family is
`docs/results/reframed-core-temporal/contrasts.json`; the context family is
`docs/results/reframed-ablation5-temporal/contrasts.json`. Both are newly declared
structural families after descriptive results, not prospective preregistrations.
The context comparison holds selected timestamps fixed, so its temporalF1
change reflects valid-output availability, not improved placement selection.

Measured all-gap occupancy is recorded in
`docs/results/reframed-formal-gap-utilization/measured-gap-utilization.json`;
all30clips, unused gaps, and zero-output clips remain in each denominator.
Detector-defined diagnostics and traceable development frame/playback examples
are separate artifacts; neither substitutes for human labels.

Refresh paper and deck AD tables from the canonical artifacts with
`python docs/report/refresh_tables.py`, then compile each LaTeX source twice.
The generator also refreshes completed long QA and EgoSchema tables; no local-result slots remain pending.

## Completed long-video QA exports

`docs/results/video-mme-long/qa-results.json` contains10conditions with36/36valid
answers each,360total, no invalid/missing/generation-error rows. The same directory
contains13paired contrasts, coverage, per-video frame/duration analysis, and plots.
All13Holm-adjusted p-values are1.0. Same-frame timeline A_cue scores20/36versus
21/36without; AV18/36versus20/36without. This is not evidence that timelines help,
and the small sample does not prove no effect. A/AV versus U/V remains a
constrained-realized-budget diagnostic. The frozen subset is12videos, not the
wholeVideo-MME benchmark. EgoSchema aggregation is also complete.

## Verified tuning and implementation scope

`docs/verification/development-cv-evidence.json` binds the existing five-fold
cross-validation on ten distinct development movies, each held out exactly once
and disjoint from test. The eight-point grid varied global signed NSE±0.5,
SC0.5/1.0 and GF0/0.5; mean heldout-fold F1 was0.495162 for A_cue and0.472262
for AV. Superclass variants are fixed±0.5 component ablations, not learned
per-superclass CV weights. VN and OP were not grid axes. Do not claim all five
weights or class-specific coefficients were cross-validated, and do not retune
on heldout results.

The frozen inference source passed817tests; the later integrated report branch
passed847tests,3skips and5974subtests. The latter is bound in
`docs/verification/final-test-results.json` to tested commit7fe412d and its log
hash. Reporting checks do not change historical inference provenance or replace
human validation. Team names, original track declaration, individual human
contributions, code walkthrough and submission remain human responsibilities.

## Completed EgoSchema exports

`docs/results/egoschema-heldout/qa-results.json` contains three490-question
conditions, all490answers valid percondition and zero missing/invalid/generation
errors. U scores319/490 (65.10%), V_local311/490 (63.47%), and V_memory309/490
(63.06%). The predeclared three-contrast family gives Holm p-values0.5443,
0.5003 and0.8337 for V_local−U, V_memory−U, and V_memory−V_local. None supports
a detected visual-selection advantage; absence of significance is not equivalence.
All distributed clips lack audio, so there is no audio-aware claim. Combined
QA coverage is1830/1830valid answers across the frozen subsets.

See the final delivery record and completion/reproduction manifests for the
final verifier outcome, artifact retention location, current commit and job
accounting. These manifests, not historical queue prose, define final operational
status. No new API authorization or human annotations are implied by local
experiment completion.
