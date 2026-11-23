"""Export every planned core cell, including deferred models and incomplete coverage."""
import argparse,csv,hashlib,json,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from src.experiments.config import load_config
from src.experiments.matrix import core_generation_runs
from src.experiments.generation_stage import generations_path,moments_path,budget_compliance
from src.experiments.io import read_jsonl
from src.generation.outputs import final_rows,is_success
from src.experiments.selection_stage import selections_path


def main():
 p=argparse.ArgumentParser(description=__doc__)
 p.add_argument('--config',required=True,type=Path)
 p.add_argument('--tts-root',required=True,type=Path)
 p.add_argument('--output-dir',required=True,type=Path)
 p.add_argument('--metric-overlay',type=Path)
 p.add_argument('--deferred-model',action='append',default=[])
 p.add_argument('--scope',required=True)
 a=p.parse_args();cfg=load_config(a.config);root=Path(cfg['paths']['output_root'])/cfg['experiment']['name']
 rows=[];evidence=[]
 for run in core_generation_runs(cfg):
  row=dict(selector=run.selection.name,prior=run.prior,model=run.model_label,status='pending',requested_moments=None,selected_moments=None,valid=None,expected=None,failed=None,missing=None,zero_output_videos=None,budget_overruns=None,budget_known=None,budget_unknown=None,f1=None,bertscore=None,openclip=None,tts_fit=None,tts_rendered=None,tts_overlap_seconds=None,tts_narration_excess_seconds=None,generation_latency_seconds=None)
  if run.model_label in a.deferred_model:
   row['status']='deferred: no paid authorization';rows.append(row);continue
  expected={x['id'] for x in read_jsonl(moments_path(root,run.selection))};final=final_rows(read_jsonl(generations_path(root,run)));preds={x['id']:x for x in final}
  if len(preds)!=len(final):raise ValueError('duplicate successful prediction IDs')
  selection_rows={x['video_id']:x for x in read_jsonl(selections_path(root,run.selection))}
  row.update(requested_moments=sum(x['k'] for x in selection_rows.values()),selected_moments=sum(len(x.get('times',[])) for x in selection_rows.values()))
  budget=budget_compliance(list(preds.values()),len(expected-set(preds)))
  row.update(budget_overruns=budget['n_over_budget'],budget_known=budget['n_known'],budget_unknown=budget['n_unknown_generated'])
  unexpected=set(preds)-expected
  if unexpected:raise ValueError(f'{run.run_id}: unexpected prediction IDs')
  row.update(expected=len(expected),valid=sum(is_success(x) for x in preds.values()),failed=sum(not is_success(x) for x in preds.values()),missing=len(expected-set(preds)),generation_latency_seconds=sum(x.get('latency_s',0) for x in preds.values()))
  ep=root/'eval'/f'generation_{run.run_id}.json';sources=[]
  if ep.exists():
   e=json.loads(ep.read_text());sources.append(ep)
   assert not e['missing_moment_ids'] and len(preds)==e['expected_moments']==len(expected)
   if a.metric_overlay:
    cp=a.metric_overlay/'eval'/ep.name
    if cp.exists():
     c=json.loads(cp.read_text());sources.append(cp)
     for key in ['run_id','expected_moments','macro_f1']:assert c[key]==e[key]
     for metric,value in c['text_metrics'].items():
      if value.get('status')=='ok':e['text_metrics'][metric]=value
   row.update(f1=e['macro_f1']['mean'],zero_output_videos=sum(v['n_ok']==0 for v in e['videos']))
   for metric,key in [('bertscore','bertscore'),('clip_text','openclip')]:
    result=e['text_metrics'].get(metric,{})
    if result.get('status')=='ok':row[key]=result['per_video_mean']['mean']
   row['status']='evaluated with failures' if row['failed'] or row['zero_output_videos'] else 'evaluated'
  tp=a.tts_root/run.run_id/'overlap/report.json'
  if tp.exists():
   t=json.loads(tp.read_text());assert t['complete'] and t['playback_speed']==1.2;sources.append(tp)
   tr=[x for v in t['videos'] for x in v['rows']];assert len(tr)==row['valid']
   row.update(tts_fit=sum(x['fits_speech_free_video'] for x in tr),tts_rendered=len(tr),tts_overlap_seconds=sum(v['speech_overlap_seconds_timeline']['union'] for v in t['videos']),tts_narration_excess_seconds=sum(v['narration_excess_layer_seconds'] for v in t['videos']))
  evidence.append({'run_id':run.run_id,'failed_ids':[k for k,x in preds.items() if not is_success(x)],'missing_ids':sorted(expected-set(preds)),'source_sha256':{str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in sources}})
  rows.append(row)
 a.output_dir.mkdir(parents=True,exist_ok=True)
 (a.output_dir/'core-results.json').write_text(json.dumps({'scope':a.scope,'rows':rows,'evidence':evidence,'notes':'Text metrics condition on temporally matched valid outputs. F1 includes expected videos. TTS covers rendered valid outputs only; individual speech fit does not exclude narration self-overlap, reported separately. Latency sums model calls, not allocated GPU wall time.'},indent=2)+'\n')
 with (a.output_dir/'core-results.csv').open('w') as f:
  writer=csv.DictWriter(f,fieldnames=rows[0]);writer.writeheader();writer.writerows(rows)
 def val(x):return '—' if x is None else f'{x:.3f}' if isinstance(x,float) else str(x)
 fields=['selector','prior','model','status','requested_moments','selected_moments','valid','expected','failed','budget_overruns','budget_known','f1','bertscore','openclip','tts_fit','tts_rendered']
 lines=['# Planned core matrix','',a.scope,'','| '+' | '.join(fields)+' |','|'+'---|'*len(fields)]
 lines+=['| '+' | '.join(val(r[k]) for k in fields)+' |' for r in rows]
 (a.output_dir/'core-results.md').write_text('\n'.join(lines)+'\n')
 print(json.dumps({'cells':len(rows),'deferred':sum(r['status'].startswith('deferred') for r in rows),'output':str(a.output_dir)}))

if __name__=='__main__':main()
