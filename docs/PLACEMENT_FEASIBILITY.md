# Development placement feasibility

The fixed Uniform candidate formula and bounded placement in VAD gaps do not
necessarily permit K generated descriptions. An audit of the frozen ten-clip
development subsets quantified this before changing the selector protocol.
No test outcomes or model generations were used. The audit changes no inputs.

| Dataset | Requested Uniform moments | Minimum word capacity | Advance/lag bound | Individually snappable | Nearest placements after 2s separation |
|---|---:|---:|---:|---:|---:|
| REFRAMED | 111 | 1 | 0s | 60 | 60 |
| REFRAMED | 111 | 1 | 2s | 76 | 76 |
| REFRAMED | 111 | 1 | 10s | 94 | 89 |
| REFRAMED | 111 | 15 | 2s | 62 | 62 |
| MTG | 189 | 1 | 0s | 71 | 71 |
| MTG | 189 | 1 | 2s | 93 | 93 |
| MTG | 189 | 1 | 10s | 135 | 134 |
| MTG | 189 | 15 | 2s | 65 | 65 |

Bounds are symmetric and exploratory; no final bound or minimum capacity has
been selected by this audit. It uses current cached usable gaps, including
the existing minimum-gap duration and speech-boundary margins. That differs
from the old builder's untrimmed remaining-time-to-speech test, so the0s counts
are not expected to match the old66/78 emitted inputs. A one-word requirement
is a feasibility diagnostic, not a recommendation for description quality.

The individual count is exact for each candidate under the specified gaps and
bounds. The separated count greedily removes collisions among nearest snapped
placements; it is not an optimal joint placement algorithm or a proof of the
maximum feasible cardinality. Missing individual placements cannot be repaired
by a better collision solver under the same bounds.

Maximum absolute annotation/media duration drift on these development clips
was0.043s for REFRAMED and0.852s for MTG. Those checks do not prove transcript
or AD alignment across the entire dataset.

Implications for implementation:

- Keep the shared requested K explicit. Do not reduce it separately per selector.
- Preserve infeasible requests and reasons in coverage/failure accounting.
- Do not silently move Uniform events arbitrarily far or add replacement events
  while continuing to call the condition the fixed Uniform baseline.
- Distinguish requested candidate budget from feasible/generated descriptions;
  a claim of equal generated counts is not currently supported.
- Complete dev-only placement/protocol reconciliation before the full matrix.
  This audit does not freeze a replacement protocol or resolve the requirement.

Reproduce with scripts/audit_gap_feasibility.py, passing --dataset, --data-dir,
--cache-dir, --splits and --output. Actual commands and outputs:

```bash
source hpc/config.sh
in_env ro 'python scripts/audit_gap_feasibility.py --dataset reframed --data-dir /scratch/yy5919/pyro-kd-derived/datasets/reframed --cache-dir /scratch/yy5919/pyro-kd-derived/cache/reframed --splits configs/reframed_splits.json --output outputs/verification/reframed-dev-gap-feasibility.json'
in_env ro 'python scripts/audit_gap_feasibility.py --dataset mtg-ad --data-dir /scratch/yy5919/pyro-kd-derived/datasets/mtg-ad --cache-dir /scratch/yy5919/pyro-kd-derived/cache/mtg-ad --splits configs/mtg_ad_splits.json --output outputs/verification/mtg-dev-gap-feasibility.json'
```

Reports include all15 combinations of capacity1/6/15 and bounds0/1/2/5/10,
per-video counts and split hashes. No generation or quality metric was run.

Running visual memory implementation checkpoint

`build_moments.py --selector V --visual-novelty memory` now enables greedy novelty against every already-selected embedding, recomputed after each choice. The first selection uses the historical adjacent-frame score; subsequent novelty is `(1 - max cosine to selected frames) / 2`, plus the existing shot-boundary bonus. Embeddings are normalized at use, validated against the committed cache, and invalid/missing embeddings fail explicitly. Minimum separation remains 2 seconds; ties choose earlier times and output is chronological. `--visual-novelty local` preserves the adjacent-frame baseline (still the compatibility default). Mode is recorded in inputs and reports.

This is ranking integration, not completion of the two-stage selector: the builder still filters unspeakable moments after ranking, and the placement/fixed-K protocol remains unresolved. Memory currently refers to selected candidates, including candidates subsequently filtered by the old builder; the final snapped-candidate pipeline must rank feasible placements before generation. Shot/novelty weights remain untuned. There are no real development visual caches in the derived cache directory yet, so validation currently covers synthetic recurrence/separation cases and existing pipeline tests, not real visual feature extraction or scientific results. Per-label event weights, speech cues and AV integration remain outstanding.
