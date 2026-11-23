"""Evaluate only formal ablation5 no-context/native cells on CPU."""
import json,sys,time
from pathlib import Path
repo=Path('/scratch/yy5919/pyro-revision15-formal');sys.path.insert(0,str(repo))
from src.experiments.config import load_config
from src.experiments.data import load_records
from src.experiments.matrix import ablation5_runs
from src.experiments.io import read_jsonl
from src.experiments.generation_stage import evaluate_generation, moments_path, generations_path, _optional_scorer
from src.experiments.report import make_report
cfg=load_config(repo/'configs/experiments/reframed_test_local.toml')
exp=Path(cfg['paths']['output_root'])/cfg['experiment']['name']
records=load_records(cfg);runs=[r for r in ablation5_runs(cfg) if not r.timeline]
def initialize_scorers():
 scorers={};metadata={}
 try:
  from bert_score import BERTScorer
  bert=BERTScorer(lang='en',device='cpu',batch_size=8,idf=False,rescale_with_baseline=False,use_fast_tokenizer=False)
  scorers['bertscore']=(lambda c,r:[float(x) for x in bert.score(c,r)[2]],None)
  metadata['bertscore']={'hash':bert.hash,'device':'cpu','batch_size':8}
 except Exception as e:scorers['bertscore']=(None,'unavailable: '+repr(e))
 for name in ['sentence_cosine','clip_text']:
  scorers[name]=_optional_scorer(name,cfg)
  metadata[name]={'status':'ready' if scorers[name][0] else scorers[name][1]}
 (exp/'eval').mkdir(exist_ok=True)
 (exp/'eval'/'ablation5_text_scorer_provenance.json').write_text(json.dumps(metadata,indent=2)+'\n')
 return scorers
scorers=None
print(json.dumps({'watching':[r.run_id for r in runs],'scope':'ablation5 no-context/native only','device':'cpu'}),flush=True)
completed=set();deadline=time.monotonic()+21600
while len(completed)<len(runs) and time.monotonic()<deadline:
 for run in runs:
  if run.run_id in completed:continue
  expected={r['id'] for r in read_jsonl(moments_path(exp,run.selection))}
  rows=read_jsonl(generations_path(exp,run))
  if not expected or not expected <= {r['id'] for r in rows}:continue
  if scorers is None:scorers=initialize_scorers()
  result=evaluate_generation(cfg,run,records,exp,scorers)
  completed.add(run.run_id)
  print(json.dumps({'run':run.run_id,'videos':result['n_videos'],'macro_f1':result['macro_f1'],'text_metrics':result['text_metrics']}),flush=True)
  # Parent consolidates reports; no shared report writer here.
 if len(completed)<len(runs):time.sleep(20)
print(json.dumps({'completed_cells':len(completed),'expected_cells':len(runs)}),flush=True)
raise SystemExit(0 if len(completed)==len(runs) else 2)
