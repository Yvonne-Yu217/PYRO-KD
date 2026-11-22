# Proposal 1.5 implementation and verification audit

Audited 2026-11-22 against the user-linked Google Docs tab and main
3139be9be2ef52f498b5db5fecfdfa5618da03b0. This is a requirements inventory,
not evidence of completed experiments. The linked proposal takes precedence
over the older week1 specification. Existing split files remain preserved.

| Requirement | Current evidence / remaining work |
|---|---|
| Primary REFRAMED evaluation | Loader exists; media unavailable on this Cloud Bursting workspace. Verify 1,983 files and 40 documented failures on Torch. |
| Official splits / fixed 50 test clips | Existing config selects 30 test clips. Preserve it as legacy; derive a separate versioned 50-clip config from binary availability only, without inspecting test outcomes. Group inference by movie. |
| Five selectors U,V,A,AV,O | U,A,V moment generation exists; O exists only for timing; AV absent. Implement and verify O generation and fused selection. |
| Constant K / separation / four frames over 2 seconds | Shared duration-based K and frame windows exist. Oracle explicitly bypasses separation; resolve in proposal-specific protocol and distinguish historical unconstrained upper bound. Report all shortfalls. |
| P0 and P1 | Only P0 exists. P1 requires guidelines plus three professional exemplars from validation, excluding dev/test videos; persist provenance and freeze before evaluation. |
| Frozen open model and API model | Existing generator supports Qwen models. Proposal requests Qwen2.5-VL class; historical smoke used Qwen3. Pin model and revision for new runs. API condition blocked without explicit spending cap; do not replace it with Muse and call matrix complete. |
| Real development end-to-end smoke | Historical synthetic smoke is insufficient. Must inspect real inputs, predictions and evaluation before scaling. |
| Visual/audio feature caches | Implemented; verify current manifests and feature hashes on actual media. No missing audio treated as silence. |
| Fusion weight tuning | Missing; tune using development only, freeze before held-out inference. |
| Timing precision / recall | Existing MTG selector evaluator; extend to REFRAMED and all selectors. |
| Rendered speech collision at 1.2x and silence utilization | Must implement/verify actual TTS durations and VAD comparison; word-count estimates cannot satisfy rendered timing. |
| OpenCLIP cosine and BERTScore | Existing cosine implementation uses sentence embeddings; audit and implement proposal OpenCLIP measure explicitly. |
| Calibrated five-criterion judge | Rubric helpers exist; calibration, actual scores and provenance unverified. |
| EgoSchema QA | Annotation preparation only; implement QA inputs, inference, parsing and accuracy. Audit 5,031 videos and 500 public labels. A/AV blocked absent matching audio; O unavailable without AD timestamps. Exclude pilot from held-out local accuracy. |
| Blinded 60 peer comparisons | Requires real human ratings; prepare blinded packet, never generate ratings on behalf of peers. |
| Paired Wilcoxon / 95% bootstrap | Bootstrap helpers exist; Wilcoxon and movie-clustered REFRAMED analysis missing. Use paired examples and report exclusions. |
| Raw predictions, per-example metrics, tables, plots, failure analysis | Pending actual runs; preserve failures and coverage denominators. |
| Exact reproduction / code freeze | Record source SHA, environment, model revision, input hashes, commands, Slurm IDs, outputs and frozen protocol per run. |

Some equations are omitted by the Google Docs text export. Retrieve the full
native document or export before claiming numerical protocol fidelity. No
proposal cut-list item has been silently invoked.

Course Agent Track requirements were read from
https://www.sainingxie.com/cv-fall2026/project/ (curl retrieval after web tool
failure): log tooling/contributions, agent failures, human diagnosis and
verification. Agent Log is an appendix deliverable.
