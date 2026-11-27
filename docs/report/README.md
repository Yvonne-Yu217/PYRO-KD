# Research report for team review

`main.tex` and `main.pdf` contain the completed authorized local experiment
results: development diagnostics, all31selector configurations, the20-cell core
matrix (10local evaluated and10APIdeferred), formal audio-context ablations,
ten long-QA conditions, and three EgoSchema conditions. There are no pending
local-result slots. The English Agent Log appendix summarizes failures and
verification; the chronological log remains `docs/AGENT_LOG.md`.

Human author names, individual contributions, and the original track declaration
still require truthful team confirmation. Paid-model/rubric comparisons, human
detector labels and blinded peer ratings remain deferred. Cue2Narrate is
unavailable. Long A/AV comparisons have unequal realized frame budgets and are
explicitly diagnostics, not equal-budget causal evidence. The report is for team
review, not a claim that every original proposal requirement is complete.

`python docs/report/refresh_tables.py` regenerates all paper/deck tables from
checked-in canonical JSON/CSV sources and records their hashes. It verifies
3398local core attempts,3108valid captions,3024individual speech fits,360valid
long-QA answers and three fixed490-question EgoSchema conditions. Correctness,
confidence intervals and comparisons are taken from canonical exports; no new
inference, human annotation or statistical tuning occurs during typesetting.

Build from the repository root with existing read-only container TeX:

```bash
source hpc/config.sh
in_env ro 'cd docs/report && pdflatex -interaction=nonstopmode -halt-on-error main.tex && pdflatex -interaction=nonstopmode -halt-on-error main.tex'
```

The host lacks TeX; the existing container has `/usr/bin/pdflatex`. No software
installation is needed. Compile intermediates are ignored. `verification.json`
records source/PDF hashes, page counts and layout checks. Before submission,
confirm team attribution, inspect the PDF, rehearse the code walkthrough, and
preserve the documented scope limits and deferred human/API requirements.

Development CV is specifically five folds on ten movies over the eight-point
global NSE/SC/GF grid. Superclass variants are fixed±0.5 ablations; learned
superclass coefficients and VN/OP grid tuning were not completed. The frozen
inference source passed817tests; the later integrated report branch passed847,
with3skips and5974subtests. Tests do not replace scientific or human evidence.
