# Experiment status

Source baseline: 3139be9be2ef52f498b5db5fecfdfa5618da03b0 (fetched origin/main 2026-11-22).
Proposal source: Google Doc 1dYGXJJ-MJxmJtZV02U4IrkQVXLPTdev9soelen7Ltyw, tab t.1uocwcl7xqhl.

Environment: slurm-cloud, b-19-7. Existing user Jupyter job 40006; not agent-submitted. A100 40GB idle at audit. Requested /scratch/yy5919/mtg-ad-transfer absent here. Torch access under investigation.

Tests: `python -m pytest -q` failed (default Python lacks pytest). Retry: `source hpc/config.sh && in_env ro 'python -m pytest -q'`, process session 46949.
No experiment jobs submitted. No scientific outputs yet.

Next: establish Torch access and inspect dataset manifests, finish tests, reconcile proposal against existing implementations and split sizes, run real dev smoke, then full supported matrix and statistics.
Blocked prerequisites: API spending cap; matching EgoSchema audio; private labels beyond 500; peer ratings require human contributions. Known dataset counts from user (not independently verified): REFRAMED 1983 available/40 HTTP403; EgoSchema 5031 silent videos/500 public labels.
Scope remains all five selectors × two prompts × two frozen models, REFRAMED timing/quality, EgoSchema QA, calibrated rubric, 60 peer comparisons, Wilcoxon/95% bootstrap, raw outputs, plots and reproduction commands. Existing repository documents differ from linked proposal (MTG primary vs REFRAMED primary, 30 vs 50 REFRAMED test clips); resolve explicitly without tuning on held-out outcomes.

Update: existing-container pytest retry also failed (pytest absent). Isolated test venv created at /scratch/yy5919/pyro-test-env. Baseline verification completed: **394 passed, 5911 subtests passed in 22.93s**. Command: `source hpc/config.sh && in_env ro '/scratch/yy5919/pyro-test-env/bin/python -m pytest -q'`. Output: `outputs/verification/pytest-baseline.log`. Process 21475 exited 0. These tests do not prove real model/data integration.
Torch hostname corrected to login.torch.hpc.nyu.edu using official NYU docs; SSH reports changed host key. Authenticated access requested; existing trust store preserved. Implementation gaps and full remaining requirements: `docs/PROPOSAL_1_5_AUDIT.md`. No experiment jobs submitted.

Paired-analysis implementation added: `scripts/analyze_paired.py --input METRICS.jsonl --output ANALYSIS.json --a CONDITION_A --b CONDITION_B --metric METRIC --unit cluster` (REFRAMED movie IDs) or `--unit example` (independent videos). It reports pairing exclusions, equal-weight cluster means, paired bootstrap CI and permutation Wilcoxon (SciPy required), input hash and command. P-values are explicitly unadjusted; multiplicity remains to implement before confirmatory reporting. Tests: 4 passed in 3.36s; no scientific results generated.

SSH update: following NYU's documented options reaches Microsoft device authentication. Live PTY session 70487 is waiting for user MFA; poll that handle, then send ENTER only after completion is confirmed. Do not create duplicate sessions while it is live. Torch datasets remain uninspected.

Comparison families: `python scripts/analyze_family.py --input METRICS.jsonl --family FAMILY.json --output ANALYSIS.json --unit cluster`. FAMILY.json has a `comparisons` array with `a`, `b`, `metric` fields. Freeze the full family before test scoring. Supports Holm p-value correction; CIs are explicitly pointwise. Targeted statistics verification: 41 passed, 11 subtests passed, 3.94s; log outputs/verification/pytest-statistics.log. Actual experiment analysis remains pending.

IMPORTANT SCOPE CORRECTION: authoritative Proposal Revision 1.5 is tab t.ickd261ntenp, not URL-selected t.1uocwcl7xqhl. Full source now docs/PROPOSAL_REVISION_1_5.txt; current complete inventory docs/REVISION_1_5_REQUIREMENTS.md. Earlier audit is historical and incomplete. Added scope includes long-form QA, Cue2Narrate gate, three-stream audio, eight ablations and human detector labels. No experimental scope cuts applied.
SSH 70487 closed (255) after user confirmed MFA and ENTER was sent. Replacement 62852 is live at fresh Microsoft device-auth prompt. Continue that session after user completes new authentication; do not retry the old handle.
Current priority (user): obtain data first, locate existing real MTG smoke before repeating it. 62852 terminated (255); latest SSH session 10280 awaiting device auth. Local /scratch/yy5919/pyro-kd-data/mtg-ad/{videos,incoming} empty. outputs/evaluation-smoke/manual_paraphrases.validation.evaluation.json identifies model manual-smoke/manual-paraphrases-v1; it is not proof of GPU VLM inference. Gap-snapping helper/tests added (10 passed), integration deferred until data access.

User explicitly switched to CPU-only allocation for data downloads; hold GPU submissions until data ready and user starts GPU session. Old allocation 40006 CANCELLED at 22:21 UTC, which destroyed SSH master. Current allocation 40036 on b-10-63 (CPU, six-hour limit). New SSH 45555 awaits Microsoft MFA; control path /home/yy5919/.ssh/pyro-torch-cpu-control.
Data discovery correction: rg must use --no-ignore for Git-ignored data directories. Existing /scratch/yy5919/PYRO-KD/data has REFRAMED metadata and 51 EgoSchema videos. All 51 ffprobe calls succeeded, none has audio. Evidence outputs/verification/local-egoschema-inventory.json; probe script alongside. No full decode claimed. Existing MTG dev report says 0/10 ready (HTTP403); later real GPU smoke remains unlocated. Do not redownload available data; first inspect Torch source and transfer missing files.
