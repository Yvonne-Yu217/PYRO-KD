# Revision 1.5 presentation draft

`main.tex` and `main.pdf` contain a12-slide Beamer draft. The deck includes
verified development, formal selector, and formal audio-context ablation results.
The formal local core and audio-context metrics are complete; Long QA is complete; EgoSchema aggregation remains explicitly pending. Human authorship and
contribution attribution require team confirmation; paid APIs and human labels
remain deferred. No human-quality, Cue2Narrate, or equal-realized-budget long-QA
claim is made.

Build with the existing read-only container (no software installation):

```bash
source hpc/config.sh
in_env ro 'cd docs/slides && pdflatex -interaction=nonstopmode -halt-on-error main.tex && pdflatex -interaction=nonstopmode -halt-on-error main.tex'
```

Slide evidence:

- Development coverage: `docs/results/reframed-dev-coverage.pdf` and the
  development core export.
- Selector means: `docs/results/reframed-selectors/selector-results.json`.
- Selector contrast family: `docs/results/reframed-selector-ablations/selector-ablation-stats.json`.
  This family was declared after descriptive results, not preregistered.
- Formal context ablation: `docs/results/reframed-formal-ablation5/ablation5-results.json`
  and its interpretation file.
- Playback examples: `docs/results/qualitative-agent-review.md`; targeted agent
  inspection, not human labels or a frequency estimate.
- Long budget: `docs/results/long-selection-budget-audit.md` and frozen protocols.

`verification.json` records source/PDF hashes, page count and compile checks.
Rendered slides7 and11 were visually inspected for table readability. The first
compile exposed incorrectly escaped line-break commands in the coverage slide;
those were corrected before the successful two-pass build. Final compile has no
overfull boxes or undefined references.

Before presentation, replace the pending-QA slide only with completed,
validated formal exports; update conclusions and status consistently with the
paper; add confirmed human author names/contributions; rehearse the code
walkthrough and the distinctions between selection, caption validity, individual
speech fit and combined playback quality.

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
