# Local experiment completion inventory

The completion verifier is CPU-only and reads metadata, predictions and scoring artifacts. It never loads a model, reads raw QA answers, or rehashes weights, frame caches or rendered audio. It reuses detailed temporal and measured-gap exporter evidence, verifying that their bound generation/evaluation/TTS records have not changed. Run those exporters first after all relevant jobs finish.

Example (substitute the actual final exporter artifact paths):

```bash
/scratch/yy5919/miniforge3/bin/python scripts/verify_revision15_completion.py \
  --gap-evidence docs/results/reframed-formal-gap-utilization/measured-gap-utilization.json \
  --gap-evidence docs/results/reframed-formal-ablation5-gap-utilization/measured-gap-utilization.json \
  --temporal-evidence /scratch/yy5919/pyro-resume-results/formal-core-temporal/paired.jsonl.provenance.json \
  --output /scratch/yy5919/pyro-resume-results/revision15-completion.json
```

Exit zero means the inventoried **local** experiments and required evidence are complete. Exit one means not ready; inspect `errors`, per-condition coverage, failed IDs and artifact hashes. All expected final attempts must exist: 3,398 core AD plus 768 additional AD, ten long QA conditions with 36 questions each, and three EgoSchema conditions with 490 questions each. Duplicate final/success IDs, unexpected IDs, malformed/truncated JSONL and missing IDs fail. Historical superseded errors are counted separately. Failed final generations are permitted and enumerated, never silently promoted to successes. TTS coverage is required for all valid captions and failure accounting must agree with generation. Whole-dataset raw QA coverage and provenance must pass before any QA scoring artifacts are opened; scoring uses the fixed denominator and validates failures as incorrect.

The manifest binds the verifier revision, frozen source identities, protocols, small input hashes, prediction/provenance records, metrics and exporter evidence. Historical AD metric reports have no original raw-input digest; this verifier checks their counts and relies on the detailed temporal export for recomputed reference-backed temporal metrics, rather than claiming nonexistent provenance. Large waveform/reference validation remains the detailed exporters' responsibility.

`all_proposal_complete` is always false. Paid API experiments and rubric judging, human audio-event labels and blinded peer ratings, and the gated Cue2Narrate condition remain explicitly deferred.

Agent contribution and verification (2026-11-23): added the independent completion inventory and four synthetic tests for retry/failure accounting, missing/unexpected/duplicate IDs, truncated JSONL, and the pre-scoring QA gate. Ran against current partial artifacts: returned not ready, identified unfinished AD/TTS and QA conditions, and did not open QA scores. No GPU work, labels, live source changes or paid calls were involved. A first invocation used a compute-container Python symlink unavailable on this host; reran successfully with the existing scratch Miniforge Python. Partial audit artifact: `/scratch/yy5919/pyro-resume-results/revision15-partial-inventory.json`.
