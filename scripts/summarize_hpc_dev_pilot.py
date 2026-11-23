"""Consolidate completed dev metrics without modifying any producer's artifacts."""
import json,csv,hashlib
from pathlib import Path
root=Path('/scratch/yy5919/pyro-resume-results/experiments/reframed_local_dev_pilot_v1')
clip=root.with_name('reframed_local_dev_pilot_clip')
tts=Path('/scratch/yy5919/pyro-kd-derived/tts/local-dev-pilot-v1')
out=Path('/scratch/yy5919/pyro-resume-results/local-dev-report');out.mkdir(exist_ok=True)
rows=[];details=[]
for p in sorted((root/'eval').glob('generation_*.json')):
 d=json.loads(p.read_text());run=d['run_id'];cp=clip/'eval'/p.name;tp=tts/run/'overlap/report.json'
 if not cp.exists() or not tp.exists():raise RuntimeError('still pending '+run)
 c=json.loads(cp.read_text());t=json.loads(tp.read_text())
 for k in ['run_id','expected_moments','macro_f1','missing_moment_ids']:assert d[k]==c[k],(run,k)
 assert t['complete'] and t['playback_speed']==1.2
 preds={x['id']:x for x in map(json.loads,(root/'generations'/f'{run}.jsonl').read_text().splitlines())}
 assert len(preds)==d['expected_moments'] and not d['missing_moment_ids']
 tr=[r for v in t['videos'] for r in v['rows']]
 valid=sum('error' not in x for x in preds.values());assert len(tr)==valid,(run,len(tr),valid)
 d['text_metrics']['clip_text']=c['text_metrics']['clip_text']
 row={'selector':d['selector'],'prior':d['prior'],'expected':len(preds),'valid':valid,'failed':len(preds)-valid,'zero_prediction_videos':len(d['missing_video_ids']),'placement_f1':d['macro_f1']['mean'],'bertscore':d['text_metrics']['bertscore']['per_video_mean']['mean'],'clip_text':d['text_metrics']['clip_text']['per_video_mean']['mean'],'tts_fit':sum(x['fits_speech_free_video'] for x in tr),'tts_rendered':len(tr),'tts_speech_overlap_seconds':sum(v['speech_overlap_seconds_timeline']['union'] for v in t['videos']),'tts_narration_excess_seconds':sum(v['narration_excess_layer_seconds'] for v in t['videos'])}
 rows.append(row);details.append({'generation':d,'tts':t,'failed_ids':[k for k,x in preds.items() if 'error'in x],'source_hashes':{str(x):hashlib.sha256(x.read_bytes()).hexdigest() for x in [p,cp,tp]}})
assert len(rows)==10
(out/'dev-results.json').write_text(json.dumps({'scope':'10 dev clips only; not formal test results. Text metrics are temporally matched valid predictions only; F1 includes all ten clips. TTS covers valid rendered predictions only. Zero-prediction videos include selection shortfalls. No human judgment.', 'rows':rows,'details':details},indent=2)+'\n')
with (out/'dev-results.csv').open('w') as f:w=csv.DictWriter(f,fieldnames=rows[0]);w.writeheader();w.writerows(rows)
lines=['# Local development pilot','', 'Development diagnostics only. Failed predictions and zero-output clips remain in coverage. BERTScore/OpenCLIP use temporal matches; TTS is measured at 1.2x.','', '| Selector | Prompt | Valid / expected | F1 | BERTScore | OpenCLIP | TTS fit / rendered | Speech overlap (s) |','|---|---|---:|---:|---:|---:|---:|---:|']
for r in rows:lines.append(f"| {r['selector']} | {r['prior']} | {r['valid']}/{r['expected']} | {r['placement_f1']:.3f} | {r['bertscore']:.3f} | {r['clip_text']:.3f} | {r['tts_fit']}/{r['tts_rendered']} | {r['tts_speech_overlap_seconds']:.3f} |")
(out/'dev-results.md').write_text('\n'.join(lines)+'\n')
print(json.dumps({'complete':True,'cells':len(rows),'expected':sum(r['expected'] for r in rows),'valid':sum(r['valid'] for r in rows),'out':str(out)}))
