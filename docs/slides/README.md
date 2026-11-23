# Revision 1.5 presentation for team review

`main.tex` and `main.pdf` contain12Beamer slides with completed authorized local
caption and QA results. All local-result placeholders have been removed. The
deck reports the formal core, audio-context comparison, complete360-answer long
QA and1470-answer EgoSchema diagnostic, with failures and comparison families
explicit. Human attribution and original track declaration require confirmation.

Build from the repository root using the existing read-only container:

```bash
python docs/report/refresh_tables.py
source hpc/config.sh
in_env ro 'cd docs/slides && pdflatex -interaction=nonstopmode -halt-on-error main.tex && pdflatex -interaction=nonstopmode -halt-on-error main.tex'
```

No software installation or GPU is needed. `verification.json` records hashes,
slide count, clean two-pass compile checks and visual inspection. Tables are
generated from the same canonical sources as the paper. Formal core and context
contrast families were declared after descriptive results; QA families were
specified before correctness scoring. Pointwise bootstrap intervals are not
simultaneous family intervals; Holm correction is within each stated family.

Keep these distinctions when presenting:

- TemporalF1 availability is not semantic quality. The timeline ablation uses
  identical selected timestamps, so different valid-output coverage changes F1.
- Individual TTS fit is not joint narration schedulability.
- All13long-QA Holm p-values are1.0; this small subset does not establish no effect.
- EgoSchema is visual-only; none of its three corrected contrasts supports a
  visual-selection advantage over uniform.
- Long A/AV use fewer realized frames than U/V: diagnostic, not equal-budget.
- Human event precision/recall and preferences are unmeasured. Agent examples
  and detector-defined associations do not substitute for human labels.
- Global NSE/SC/GF weights have five-fold dev CV; superclass coefficients are
  fixed ablations and VN/OP were not grid axes.

Paid APIs/rubric judging, human labels/peer ratings, Cue2Narrate comparison and
team submission responsibilities remain outside the completed local work.
