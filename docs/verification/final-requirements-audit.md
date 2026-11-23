# Final requirements audit — EgoSchema ongoing

Audit date: 2026-11-23. Read the locally archived `docs/PROPOSAL_REVISION_1_5.txt`, `docs/REVISION_1_5_REQUIREMENTS.md`, and `/scratch/yy5919/pyro-course-project.html` before comparing current artifacts. The historical older-tab audit is explicitly superseded. This is a requirements review, not a completion certificate. No inference settings, labels, paid services or jobs were changed.

## Supported evidence

- REFRAMED is primary: frozen30 test clips, all31 selector configurations, ten local core AD cells plus two additional context/native cells. Twenty planned core cells remain visible, with ten API cells deferred. All AD attempts and valid-caption TTS coverage pass the current completion audit; failed generation remains explicit.
- Proposal A is `A_cue`, not the historical gap-only A. Source/config/model/split identities and training-only P1 exemplars are recorded. Development diagnostics precede the formal freeze.
- Selection ablations, audio-onset/reference associations and silent-event diagnostics have artifacts. Detector-defined association is not human detector precision/recall. Native Omni changes architecture as well as modality.
- All ten audio-bearing long-QA conditions have completed and have fixed-denominator scores,13 paired contrasts and accuracy-by-length plots. Hierarchical/flat allocation evidence is retained. Realized audio-condition budgets differ from visual controls; the report correctly labels these constrained-budget diagnostics.
- EgoSchema remains ongoing:490 heldout questions per U/V_local/V_memory condition, visual-only. Do not publish its scores until the whole-dataset completion gate passes.
- Actual1.2x TTS overlap, measured gap utilization, conditional BERTScore/OpenCLIP and paired temporal statistics are published. Individual speech fit is not joint narration schedulability, and semantic metrics conditioned on successful temporal matches do not establish complete-output quality.
- LaTeX report and slides exist. Current report verification records8 total pages,6 before the Agent Log appendix, no undefined references or overfull boxes; the course asks for4–9 research-report pages excluding references, plus the Agent Log appendix outside the limit. Agent Track and research format are stated. Twelve slides have compile verification. Incremental commits and an English chronological Agent Log are present.

## Feasible remaining CPU-only closeout

1. When Ego finishes, run fixed-denominator scoring/statistics and the completion verifier, then run the one-command cached-table reproduction wrapper into a fresh directory. Its reproduction manifest must finish successfully; a tested wrapper alone is not an end-to-end reproduction result.
2. Refresh report/slides/README/status with the verified Ego results and final failure totals; rebuild PDFs, inspect their rendered output, and refresh hashes/page counts. Replace stale “ongoing/pending” language only for work actually completed. Preserve deferred proposal requirements.
3. Link the completion and reproduction manifests, final commit/source SHAs, job IDs, frozen configs, and raw prediction/per-example metric paths in the final delivery. Scratch paths establish location, not archival durability; provide a small checksum inventory and an explicit retention/export location for reproducible raw artifacts without committing caches/weights/data.
4. Cite the verified [development CV evidence](development-cv-evidence.json): A_cue and AV used five folds on ten unique development clips/movies, two held out per fold and each held out exactly once, disjoint from test. The eight-point grid varied global signed NSE ±0.5, SC0.5/1.0 and GF0/0.5. Mean heldout-fold F1 was0.495162 for A_cue and0.472262 for AV; final full-dev choices were A_cue(nse0.5,sc0.5,gf0.5) and AV(nse0.5,sc1,gf0). This establishes global signed-weight CV, not per-superclass weight learning: class variants in `src/experiments/matrix.py` are fixed ±component ablations at magnitude0.5. No cross-validated per-superclass parameter selection was found; VN and OP were not axes of this grid. The earlier all-zero solution and explicit nonzero rerun remain documented in Agent Log around line835 as development-only decisions. Do not retune on test.
5. Optionally prepare blank annotation/rating instructions and a blinded, reproducibly sampled peer packet for the user's later human work. This is preparation only, not the missing60 ratings or detector validation. The user has deferred human labels; no fabricated completions or ratings belong in current results.

## Requirements that remain explicitly deferred or human-owned

- Paid API model cells and calibrated five-axis rubric judging: zero authorized spend.
- Human detector sanity-check labels/precision-recall and60 blinded peer comparisons: no supplied annotations. Machine diagnostics and agent qualitative inspection are not substitutes. The original detector pre-commitment gate therefore remains unmet; current runs must not be described as human-validated.
- Cue2Narrate comparison: failed access/comparability gate; use the documented REFRAMED fallback and omit Cue2Narrate performance claims.
- Team member names, individual human contributions/manual diagnoses, and confirmation of the originally declared track: require truthful team attribution. The report already marks these as pending; agents must not invent them.
- Course code walkthrough and final submission/presentation: human responsibilities. The course emphasizes failure diagnosis and verification in the Agent Log; the appendix covers agent failures, but cannot establish what a human personally diagnosed until the team supplies that attribution.

## Completion language

Use “authorized local experiments complete” only after the final verifier passes. Never “all Proposal Revision1.5 requirements complete.” Preserve the distinction between20 planned core cells and10 executed local cells;30 frozen REFRAMED clips and the downloaded corpus; known decoded word-cap violations and unknown truncated decodes; pre-scoring QA contrast declarations and post-descriptive AD contrast declarations. Current report wording largely preserves these distinctions. The full integrated test result reported by the coordinator is847 passed,3 skipped,5974 subtests; unit success is verification of implementation, not replacement for experimental or human evidence.
