"""Regenerate the draft's completed development and selector tables only."""
import csv, hashlib, json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parent
sources = [ROOT/'docs/results/reframed-local-dev.csv',
           ROOT/'docs/results/reframed-selectors/selector-results.json']
rows = list(csv.DictReader(sources[0].open()))
assert len(rows) == 10 and sum(int(r['valid']) for r in rows) == 705
lines = [r'\begin{tabular}{llrrr}', r'\toprule Selector & Prior & Valid/input & F1 & BERT \\', r'\midrule']
for row in sorted(rows, key=lambda r: (['U', 'V', 'A_cue', 'AV', 'O'].index(r['selector']), r['prior'])):
    name = row['selector'].replace('_', r'\_')
    lines.append(f"{name} & {row['prior']} & {row['valid']}/{row['expected']} & {float(row['placement_f1']):.3f} & {float(row['bertscore']):.3f} " + r'\\')
lines.extend([r'\bottomrule', r'\end{tabular}'])
(OUT/'dev_table.tex').write_text('\n'.join(lines)+'\n')
data = json.loads(sources[1].read_text())
core = [r for r in data['rows'] if r['ablation'] == 'core']
assert len(core) == 5 and all(r['videos'] == 30 for r in core)
lines = [r'\begin{tabular}{lrrr}', r'\toprule Selector & Requested & Selected & F1 \\', r'\midrule']
for row in core:
    name = row['selector'].replace('_', r'\_')
    lines.append(f"{name} & {row['requested']} & {row['selected']} & {row['f1']:.3f} " + r'\\')
lines.extend([r'\bottomrule', r'\end{tabular}'])
(OUT/'selector_table.tex').write_text('\n'.join(lines)+'\n')
(OUT/'table_provenance.json').write_text(json.dumps({
    'sources': {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest() for p in sources},
    'scope': 'completed development and selector results; no formal caption or QA result substitution',
    'development_cells': 10, 'selector_core_cells': 5,
}, indent=2)+'\n')
print('Generated10development and5selector rows; pending formal slots untouched.')
