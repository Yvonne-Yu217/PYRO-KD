"""Regenerate the draft's completed development, selector and formal caption tables."""
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
print('Generated10development and5selector rows; QA slots untouched.')

formal_path = ROOT/'docs/results/reframed-formal-core/core-results.json'
if formal_path.exists():
    data = json.loads(formal_path.read_text()); formal = data['rows']
    local = [r for r in formal if r['model'] == 'qwen']
    deferred = [r for r in formal if r['model'] == 'api']
    assert len(formal) == 20 and len(local) == len(deferred) == 10
    assert all(r['missing'] == 0 and r['tts_rendered'] == r['valid'] for r in local)
    assert sum(r['expected'] for r in local) == 3398
    assert sum(r['valid'] for r in local) == 3108
    assert sum(r['tts_fit'] for r in local) == 3024
    lines = [r'\begin{tabular}{lllrrrrrr}', r'\toprule Selector & Prior & Model & Valid/input & Failed & F1 & BERT & TTS fit/rendered & Zero-output clips \\', r'\midrule']
    for r in formal:
        name = r['selector'].replace('_', r'\_')
        if r['model'] == 'api':
            lines.append(f"{name} & {r['prior']} & API & \\multicolumn{{6}}{{l}}{{Deferred: no paid authorization}} " + r'\\')
        else:
            lines.append(f"{name} & {r['prior']} & Qwen & {r['valid']}/{r['expected']} & {r['failed']} & {r['f1']:.3f} & {r['bertscore']:.3f} & {r['tts_fit']}/{r['tts_rendered']} & {r['zero_output_videos']} " + r'\\')
    lines.extend([r'\bottomrule', r'\end{tabular}'])
    (OUT/'formal_core_table.tex').write_text('\n'.join(lines)+'\n')
    lines = [r'\begin{tabular}{lrrrr}', r'\toprule Selector & P0 valid/input & P0 F1 & P1 valid/input & P1 F1 \\', r'\midrule']
    for name in ['U', 'V', 'A_cue', 'AV', 'O']:
        a = next(r for r in local if r['selector'] == name and r['prior'] == 'P0')
        b = next(r for r in local if r['selector'] == name and r['prior'] == 'P1')
        texname = name.replace('_', r'\_')
        lines.append(f"{texname} & {a['valid']}/{a['expected']} & {a['f1']:.3f} & {b['valid']}/{b['expected']} & {b['f1']:.3f} " + r'\\')
    lines.extend([r'\bottomrule', r'\end{tabular}'])
    (ROOT/'docs/slides/core_table.tex').write_text('\n'.join(lines)+'\n')
    provenance = json.loads((OUT/'table_provenance.json').read_text())
    provenance['sources'][str(formal_path.relative_to(ROOT))] = hashlib.sha256(formal_path.read_bytes()).hexdigest()
    provenance.update(formal_core_cells=20, formal_local_cells=10, formal_deferred_cells=10, scope='completed development, selector and formal caption results; QA pending')
    (OUT/'table_provenance.json').write_text(json.dumps(provenance, indent=2)+'\n')
    print('Generated20formal core rows and5paired slide rows;3108valid/3024TTSfit verified.')

long_path = ROOT/'docs/results/video-mme-long/qa-results.json'
if long_path.exists():
    qa = json.loads(long_path.read_text())['rows']
    assert len(qa) == 10 and all(r['expected'] == r['valid'] == 36 for r in qa)
    assert all(r['missing'] == r['invalid_answers'] == r['generation_errors'] == 0 for r in qa)
    labels = {'U-none':'U / none', 'V-local-none':'V / none', 'U-flat-timeline':'U / timeline', 'V_local-flat-timeline':'V / timeline', 'A_cue-flat-timeline':r'A\_cue flat / timeline', 'A_cue-hierarchical-timeline':r'A\_cue hier. / timeline', 'AV-flat-timeline':'AV flat / timeline', 'AV-hierarchical-timeline':'AV hier. / timeline', 'A_cue-flat-none':r'A\_cue flat / none', 'AV-flat-none':'AV flat / none'}
    lines = [r'\begin{tabular}{lrrl}', r'\toprule Condition & Correct/36 & Accuracy & 95\% CI \\', r'\midrule']
    for r in qa:
        lines.append(f"{labels[r['condition']]} & {r['correct']} & {r['accuracy']:.3f} & [{r['ci_low']:.3f},{r['ci_high']:.3f}] " + r'\\')
    lines.extend([r'\bottomrule', r'\end{tabular}'])
    (OUT/'long_qa_table.tex').write_text('\n'.join(lines)+'\n')
    by = {r['condition']:r for r in qa}
    lines = [r'\begin{tabular}{lrrr}', r'\toprule Selector & No timeline & Flat + timeline & Hier. + timeline \\', r'\midrule']
    for name, off, on, hier in [('U','U-none','U-flat-timeline',None),('V','V-local-none','V_local-flat-timeline',None),(r'A\_cue','A_cue-flat-none','A_cue-flat-timeline','A_cue-hierarchical-timeline'),('AV','AV-flat-none','AV-flat-timeline','AV-hierarchical-timeline')]:
        cells = [str(by[x]['correct'])+'/36' if x else '---' for x in [off,on,hier]]
        lines.append(name+' & '+' & '.join(cells)+' '+r'\\')
    lines.extend([r'\bottomrule', r'\end{tabular}'])
    (ROOT/'docs/slides/long_qa_table.tex').write_text('\n'.join(lines)+'\n')
    provenance = json.loads((OUT/'table_provenance.json').read_text())
    provenance['sources'][str(long_path.relative_to(ROOT))] = hashlib.sha256(long_path.read_bytes()).hexdigest()
    provenance.update(long_qa_conditions=10,long_qa_attempts=360,scope='completed development, selector, formal caption and long QA results; EgoSchema pending')
    (OUT/'table_provenance.json').write_text(json.dumps(provenance,indent=2)+'\n')
    print('Generated10long QA conditions;360valid answers andzero missing/errors verified.')
