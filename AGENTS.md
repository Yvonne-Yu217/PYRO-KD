# Experiment workflow

Follow the user's Proposal Revision 1.5 experiment request. Preserve existing work and official splits. Record exact source SHA for each run. Shared raw datasets are read-only; do not commit data, credentials, weights or caches.

Maintain docs/AGENT_LOG.md in English with tools, human/agent contributions, mistakes, corrections and verification evidence. Maintain docs/EXPERIMENT_STATUS.md with commands, job IDs, outputs and next actions; register jobs before duplicate submission.

Run relevant tests and inspect a real development-data GPU smoke test before scaling. Execute all five selectors, both prompt conditions, frozen models, REFRAMED, EgoSchema and paired statistical analyses; explicitly record unsupported conditions. Never fabricate audio, labels or results. Tune only on development data. Paid APIs require an explicit spending cap.

Use bounded resumable Slurm jobs for heavy work, prepare CPU inputs ahead of GPU work, monitor scheduler/GPU/log evidence and verify coverage after completion. No dummy keepalive workloads. Do not cancel user-owned interactive sessions without authorization; release agent-owned idle allocations. Preserve raw predictions, per-example scores, summary tables, plots, confidence intervals, failure analyses and reproduction commands. Commit incremental code/documentation only.

## Revision 1.5 resumption (2026-11-23)

Use the isolated work/revision15-resume-20261123 integration branch while PR #22
is open. Preserve /home/yy5919/PYRO-KD and its uncommitted logs. Read
 docs/RESUME_2026_11_23.md before launching. Proposal A maps to A_cue, not legacy
A. Draft configs are not formal freeze evidence. API cap remains zero without
explicit user authorization. Do not rebuild compatible cached model features
just to satisfy a second schema; validate an adapter first. Never modify source
files underneath live inference. Keep human detector labels and peer ratings
explicitly pending until real annotations are supplied.

## Current execution freeze

Formal caption and EgoSchema inference use the immutable09abb6c worktree
`/scratch/yy5919/pyro-revision15-formal`. Remaining long-video QA uses immutable
a7e0e1a in `pyro-revision15-long-prep`; original visual controls retain their
recorded1f5fb60 provenance. Reporting changes belong in `pyro-revision15-records`
on `work/revision15-results`. These isolated branches contain the earlier PR22
integration and validated fixes; do not switch their source during live jobs.

Read the latest experiment status and inspect Slurm/process/output evidence
before submitting anything. Primary GPU queues share one lock; secondary tasks
own disjoint output files. Conditional continuation must verify prior jobs are
terminal and expected outputs are unfinished before requesting another GPU.
Never allocate a GPU just to wait for CPU inputs or run a dummy keepalive.
Paid API budget remains zero. Human annotations must come from the user/team;
agent qualitative inspection must be labeled as such.
