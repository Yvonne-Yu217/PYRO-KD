# Reproduce local result tables and plots

This CPU-only entry point rebuilds summaries and plots from **verified cached metrics and QA scoring rows**. It is not end-to-end inference reproduction: it does not download data, select/tune configurations, run generation, render speech, call paid APIs, read raw QA labels, or collect human annotations. Model/data preparation and inference commands remain in the experiment status and frozen-protocol records.

After the final completion inventory passes, run with the existing analysis environment (Python3.11+, NumPy, SciPy and Matplotlib):

```bash
python scripts/reproduce_revision15_tables.py \
  --completion-manifest /scratch/yy5919/pyro-resume-results/revision15-completion.json \
  --output-root /scratch/yy5919/pyro-resume-results/reproduced-tables-v1
```

The output root must not already exist. Defaults use the frozen formal config and current HPC results/derived/TTS roots; flags permit alternative paths only when bound by the completion inventory. A false/incomplete manifest, changed artifact, unbound input, or incomplete contrast family stops execution. QA scoring records are not opened until the complete-manifest hash gate passes. The same gate runs again after reproduction to catch concurrent input changes. `reproduction.json` records commands, exporter hashes, source revision, completion-manifest hash, output hashes and partial-failure progress.

Outputs include the core and ablation5 AD tables and coverage plots (Qwen and native Omni separately); all ten long-QA and three EgoSchema condition summaries; long-QA duration plots; saved paired families containing17 core,3 ablation,13 long and3 EgoSchema contrasts; and all31 selector configurations. Summaries are rebuilt by the existing exporters, with fixed-denominator QA validation and failed outcomes retained. Paired plots use the saved frozen-family statistics rather than rerunning statistical resampling. Paid API and human validation remain deferred.

The completion inventory must bind every direct input, including AD generation/moment/selection/metric/overlap records, `selection_metrics.json`, frozen long comparison protocol and question-input JSONL files, all QA evaluation files, and the four contrast artifacts (`formal-core-temporal/contrasts.json`, `formal-ablation5-temporal/contrasts.json`, and each QA evaluation directory's `comparisons.json`). The wrapper does not add unverified inputs to the inventory.

Agent verification (2026-11-23): four metadata-only guard tests cover false manifests without artifact reads, changed hashes before command construction, existing output roots, unbound inputs, and a passing metadata gate. No live QA scores were opened and no production reproduction was attempted before the final completion manifest existed. Root runs the end-to-end cached-results reproduction after completion. Contribution: independent wrapper, guard tests and documentation; no live source, GPU job or paid service changes.
