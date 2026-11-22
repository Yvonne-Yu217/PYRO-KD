# Experiment workflow

Follow the user's Proposal Revision 1.5 experiment request. Preserve existing work and official splits. Record exact source SHA for each run. Shared raw datasets are read-only; do not commit data, credentials, weights or caches.

Maintain docs/AGENT_LOG.md in English with tools, human/agent contributions, mistakes, corrections and verification evidence. Maintain docs/EXPERIMENT_STATUS.md with commands, job IDs, outputs and next actions; register jobs before duplicate submission.

Run relevant tests and inspect a real development-data GPU smoke test before scaling. Execute all five selectors, both prompt conditions, frozen models, REFRAMED, EgoSchema and paired statistical analyses; explicitly record unsupported conditions. Never fabricate audio, labels or results. Tune only on development data. Paid APIs require an explicit spending cap.

Use bounded resumable Slurm jobs for heavy work, prepare CPU inputs ahead of GPU work, monitor scheduler/GPU/log evidence and verify coverage after completion. No dummy keepalive workloads. Do not cancel user-owned interactive sessions without authorization; release agent-owned idle allocations. Preserve raw predictions, per-example scores, summary tables, plots, confidence intervals, failure analyses and reproduction commands. Commit incremental code/documentation only.
