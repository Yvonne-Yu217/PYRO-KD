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
