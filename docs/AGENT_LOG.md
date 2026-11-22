# Agent Log

## 2026-11-22 — initial audit
Human supplied the experiment scope, proposal, repository and Torch data paths; authorized ordinary Slurm GPU jobs, but no paid API budget.
Agent read the attachment, repository README, week1 specification and HPC documentation; fetched origin/main; inspected Slurm, GPU and filesystem; read the linked proposal tab through Google Drive and course page through curl after web retrieval failed.
Verified clean starting worktree and HEAD = origin/main = 3139be9be2ef52f498b5db5fecfdfa5618da03b0.
Mistake: initially called this allocation Torch before checking cluster identity. Corrected: scontrol reports slurm-cloud (Cloud Bursting), node b-19-7; supplied Torch directory is absent. Initial default-Python test failed because pytest is absent; retrying existing Singularity environment.
No experiments or measured scientific findings yet. No data or user jobs modified.

## 2026-11-22 — baseline tests and scope audit
Existing container Python also lacked pytest. Created an isolated venv with system-site-packages at /scratch/yy5919/pyro-test-env using the existing read-only Singularity overlay; installed pytest 9.2.0 and iniconfig 2.3.1. No shared overlay mutation. Started baseline tests with output persisted under outputs/verification/pytest-baseline.log.
NYU official connection documentation identified login.torch.hpc.nyu.edu (earlier guessed hostname was incorrect). Correct hostname responds, but SSH rejects a changed host key. Did not delete or bypass existing trusted keys; requested authenticated Torch access. User dataset counts remain unverified.
Audited implementation against proposal and wrote PROPOSAL_1_5_AUDIT.md; identified absent AV, P1 and QA inference plus statistical/rendered-timing gaps rather than treating existing baseline code as proposal completion.
Baseline suite exited 0: 394 tests and 5911 subtests passed in 22.93s. Verified persisted log. This does not establish real-data GPU smoke success.

Implemented paired JSONL analysis with movie aggregation, exclusions and provenance. Consulted official SciPy Wilcoxon documentation for ties/zeros and permutation method. Four tests passed, including analytically known 2/32 two-sided probability and unequal cluster-size weighting. No real metrics analyzed. User subsequently directed retrying SSH using NYU instructions and NetID yy5919.

## 2026-11-22 — SSH authentication and comparison families
Read NYU's official connection instructions in full after the user's direction. Their documented host-key handling is for multiple login nodes. Retried using per-command options (no persistent SSH configuration changes). One node offered only GSSAPI; another attempt reached keyboard-interactive Microsoft device authentication. Session 70487 is awaiting human MFA; no credentials saved in logs. Previous assertion that host-key mismatch necessarily blocked further progress was overly conservative given the official instructions.
Added scripts/analyze_family.py and Holm correction to preserve declared family size even when comparisons lack data. Distinguishes pointwise confidence intervals from family-corrected p-values. Verification: 41 tests and 11 subtests passed in 3.94s, outputs/verification/pytest-statistics.log. These are software checks, not experiment results.

## 2026-11-22 — material proposal correction
Full Google Docs structure revealed a separate tab titled Revision 1.5 (t.ickd261ntenp). The URL points to the older Reframed and EgoSchema tab. Agent mistake: assumed the supplied URL tab was Revision 1.5 and failed to inspect other tabs initially. This omitted long-video QA, Cue2Narrate, three audio streams, signed event salience, running novelty memory, hierarchy, eight ablations and human detector sanity labels. Corrected by saving full authoritative text and writing REVISION_1_5_REQUIREMENTS.md; marked previous audit historical. No experiment outputs exist, so no measured findings are contaminated by this mistake.
User confirmed MFA complete, but sending ENTER to session 70487 returned connection closed (exit 255). Started replacement session 62852 and reached a fresh device prompt; user must authenticate that session. Authentication codes are not persisted in repo.
Added standalone gap-snapping primitive with explicit advance/lag and word capacity, tested 10 cases successfully (0.08s); not yet integrated or real-data verified. User directed prioritizing data access and locating prior MTG smoke. Session 62852 closed on ENTER; new session 10280 started with connection multiplexing for reuse after authentication. Local checked MTG media directories are empty; found manual-paraphrase evaluation smoke, not yet evidence of real VLM/video smoke.
