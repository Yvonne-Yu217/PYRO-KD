# Research report draft

`main.tex` is a conference-style Agent Track research report draft; `main.pdf`
is the compiled snapshot. It contains verified development and frozen selector
results, verified final local caption metrics and explicit pending EgoSchema slots, and an Agent
Log appendix. It is not a finished submission or evidence of completed jobs.

Human author names and individual contributions must be supplied by the team.
Paid-model results, human event labels, peer ratings and rubric judgments remain
deferred. The draft deliberately makes no human-quality claim from automated
metrics. Long A/AV QA is labeled a constrained-budget diagnostic because realized
frame counts differ from visual controls.

The two small TeX tables were generated from checked-in
`docs/results/reframed-local-dev.csv` and
`docs/results/reframed-selectors/selector-results.json`. The coverage figure
comes from `docs/results/reframed-dev-coverage.pdf`. Run `python docs/report/refresh_tables.py` from the repository root to refresh
those tables and their source-hash receipt. This does not fill pending EgoSchema
result slots.

The host lacks a TeX engine, but the existing read-only container has
`/usr/bin/pdflatex`; no package installation is needed. From the repository root:

```bash
source hpc/config.sh
in_env ro 'cd docs/report && pdflatex -interaction=nonstopmode -halt-on-error main.tex && pdflatex -interaction=nonstopmode -halt-on-error main.tex'
```

Compile intermediates are ignored. Main-text page count and the appendix start
are recorded in `verification.json`. Before final submission, fill the
pending EgoSchema result slot from verified formal exports, update conclusions using the
fixed comparison families, check all captions/denominators, confirm human roles,
and inspect the rendered PDF. The complete chronological Agent Log remains
`docs/AGENT_LOG.md`; the appendix summarizes major failures and verification.

Formal caption refresh: all20matrix rows remain visible (10local,10APIdeferred).
The formal coverage totals3398attempts,3108valid captions and3024individual
TTSfits. Core17-contrast and audio-context3-contrast families were declared after
descriptive results, not preregistered. Timeline-vs-none uses identical selected
timestamps, so temporalF1 differences reflect valid-output availability rather
than a selection change or semantic-quality proof. EgoSchema remains pending.

Long QA refresh: all10conditions completed360valid answers, with no missing,
invalid or generation-error rows. All13Holm-adjusted comparison p-values are1.0.
Matched-frame timeline controls scoreA_cue20/36versus21/36without andAV18/36versus
20/36without. This does not establish absence of an effect on a12-video subset.
Cross-selector A/AV results remain constrained-budget diagnostics. Canonical
sources: `docs/results/video-mme-long/qa-results.json` and `comparisons.json`.
Only EgoSchema results remain pending in the local quantitative report.
