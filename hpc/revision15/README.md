# Revision1.5 execution drivers

Exact small driver snapshots from the2026-11-23 HPC run. These are deployment
records with explicit scratch paths, not portable automatic-launch defaults.
They contain no modelweights, data, credentials or predictions. Source code is
loaded from the frozen worktrees named in each driver; do not edit those while
jobs run. `driver_manifest.json` records original paths and hashes.

Before resuming, inspect livejobs and predictions. Never submit a second writer
to a run. The secondary andprimary queues own disjointconditions; the primary
queues share `/scratch/yy5919/pyro-resume-gpu.lock`. PaidAPI calls are not used.
The user-owned interactive allocation must not be cancelled automatically.

Scheduler correction: original41038_1 was held before execution, and41198 was
cancelled while pending. The actual replacement submissions were:

```bash
env -u SLURM_GET_USER_ENV SLURM_EXPORT_ENV=ALL sbatch --export=ALL --array=1 --dependency=afterok:41038_0:41038_2:41038_3 /scratch/yy5919/pyro-egoschema-heldout-v2.sbatch
# returned41216
env -u SLURM_GET_USER_ENV SLURM_EXPORT_ENV=ALL sbatch --export=ALL --dependency=afterok:41216 /scratch/yy5919/pyro-egoschema-heldout-inputs.sbatch
# returned41217
```

The dependency directives embedded in the copied sbatchfiles document initial
submissions; CLI overrides above document the repaired execution. On a fresh
run, use actual newly returnedjobIDs rather than copying historical IDs. Existing
completedcompatiblecaches should be validated/reused, never regenerated solely
because a scheduler jobID changed.

CPUscorers wait for complete expectedattemptcoverage. LongQA waits all10frozen
conditions; Ego waits all3heldoutconditions and stable predictionhashes before
reading answers. TTSuses thetestcache and preserves alreadyrendered audio.
See `docs/RESULTS_REPRODUCTION.md` for scientific settings and report commands.
