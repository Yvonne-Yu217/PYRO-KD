# Experiment status

Source baseline: 3139be9be2ef52f498b5db5fecfdfa5618da03b0 (fetched origin/main 2026-11-22).
Proposal source: Google Doc 1dYGXJJ-MJxmJtZV02U4IrkQVXLPTdev9soelen7Ltyw, tab t.1uocwcl7xqhl.

Environment: slurm-cloud, b-19-7. Existing user Jupyter job 40006; not agent-submitted. A100 40GB idle at audit. Requested /scratch/yy5919/mtg-ad-transfer absent here. Torch access under investigation.

Tests: `python -m pytest -q` failed (default Python lacks pytest). Retry: `source hpc/config.sh && in_env ro 'python -m pytest -q'`, process session 46949.
No experiment jobs submitted. No scientific outputs yet.

Next: establish Torch access and inspect dataset manifests, finish tests, reconcile proposal against existing implementations and split sizes, run real dev smoke, then full supported matrix and statistics.
Blocked prerequisites: API spending cap; matching EgoSchema audio; private labels beyond 500; peer ratings require human contributions. Known dataset counts from user (not independently verified): REFRAMED 1983 available/40 HTTP403; EgoSchema 5031 silent videos/500 public labels.
Scope remains all five selectors × two prompts × two frozen models, REFRAMED timing/quality, EgoSchema QA, calibrated rubric, 60 peer comparisons, Wilcoxon/95% bootstrap, raw outputs, plots and reproduction commands. Existing repository documents differ from linked proposal (MTG primary vs REFRAMED primary, 30 vs 50 REFRAMED test clips); resolve explicitly without tuning on held-out outcomes.
