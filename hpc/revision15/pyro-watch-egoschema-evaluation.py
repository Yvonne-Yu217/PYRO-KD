"""CPU watcher: never open answer labels before all three heldout outputs settle."""
import argparse,fcntl,hashlib,json,os,subprocess,sys,time
from pathlib import Path
REPO=Path('/scratch/yy5919/pyro-revision15-formal')
RECORDS=Path('/scratch/yy5919/pyro-revision15-records')
COMMIT='09abb6c0512e5e97549ad4e66d3bae05a68c42c8'
sys.path.insert(0,str(REPO))
from src.datasets.frozen_membership import egoschema_ids
from src.generation.outputs import final_rows
CONDITIONS=['U','V_local','V_memory']

def digest(raw):return hashlib.sha256(raw).hexdigest()
def read_ready(raw,expected,condition):
 rows=[json.loads(line) for line in raw.splitlines() if line.strip()]
 if any(not isinstance(r,dict) or r.get('id') not in expected for r in rows):raise ValueError('unknown prediction IDs')
 if any(r.get('dataset')!='egoschema' or r.get('split')!='heldout' or r.get('selection_condition')!=condition or not r.get('run_fingerprint') for r in rows):raise ValueError('prediction condition/provenance mismatch')
 if len({r['run_fingerprint'] for r in rows})>1:raise ValueError('mixed generation fingerprints')
 final=final_rows(rows)
 if len({r['id'] for r in final})!=len(final):raise ValueError('duplicate successful predictions')
 return len(final)==len(expected) and {r['id'] for r in final}==expected

def self_test():
 from src.evaluation.qa import score_qa
 from scripts.export_qa_paired_metrics import export_reports
 from scripts.analyze_family import analyze_family
 base={'dataset':'egoschema','split':'heldout','selection_condition':'U','selector':'U','run_fingerprint':'synthetic'}
 rows=[{**base,'id':'a','video_id':'a','output':'A','predicted_index':0},
       {**base,'id':'b','video_id':'b','output':'A','predicted_index':0,'error':'generate: synthetic failure'}]
 encode=lambda r:('\n'.join(json.dumps(x) for x in r)+'\n').encode()
 assert read_ready(encode(rows),{'a','b'},'U')
 assert not read_ready(encode(rows[:1]),{'a','b'},'U')
 try:read_ready(encode(rows+[rows[0]]),{'a','b'},'U')
 except ValueError:pass
 else:raise AssertionError('duplicate successful prediction accepted')
 reports=[]
 for condition in CONDITIONS:
  report=score_qa(rows,{'a':0,'b':0},5)
  assert report['correct']==1 and report['accuracy']==.5 and not report['rows'][1]['correct']
  report.update(config_sha256='synthetic-config',answers_sha256='synthetic-answers',split='heldout')
  reports.append((condition,report))
 exported=export_reports(reports,'egoschema')
 family=json.loads((RECORDS/'configs/egoschema_qa_comparisons.json').read_text())
 result=analyze_family(exported,family['comparisons'],'cluster',10,0)
 assert len(result['comparisons'])==3 and all(c['wilcoxon']['family_size']==3 for c in result['comparisons'])
 print(json.dumps({'self_test':'passed','synthetic_failure_counted_incorrect':True,'readiness_gate':True,'Holm_family_size':3,'real_answer_file_opened':False}))

def main():
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('--self-test',action='store_true');a=p.parse_args()
 if a.self_test:self_test();return
 if os.environ.get('CUDA_VISIBLE_DEVICES')!='':raise ValueError('CPU-only watcher required')
 if subprocess.check_output(['git','-C',str(REPO),'rev-parse','HEAD'],text=True).strip()!=COMMIT:raise ValueError('formal source changed')
 root=Path('/scratch/yy5919/pyro-resume-results/egoschema-heldout');out=root/'evaluation';out.mkdir(parents=True,exist_ok=True)
 family_source=RECORDS/'configs/egoschema_qa_comparisons.json';family_bytes=family_source.read_bytes();family_hash=digest(family_bytes)
 membership=REPO/'configs/egoschema_pilot.json';membership_bytes=membership.read_bytes();config=json.loads(membership_bytes)
 expected=set(egoschema_ids(config,'heldout'));assert len(expected)==490
 paths={c:root/(c+'.jsonl') for c in CONDITIONS};commands=[];status_path=out/'watcher-status.json'
 def write_json(path,data):
  tmp=path.with_suffix('.tmp');tmp.write_text(json.dumps(data,indent=2)+'\n');tmp.replace(path)
 def save(state,**extra):write_json(status_path,{'state':state,'pid':os.getpid(),'updated_utc':time.strftime('%FT%TZ',time.gmtime()),'source_commit':COMMIT,'driver_sha256':digest(Path(__file__).read_bytes()),'family_sha256':family_hash,'membership_sha256':digest(membership_bytes),'commands':commands,**extra})
 def run(args,allowed=(0,)):
  cmd=[sys.executable,'-u',*args];started=time.monotonic()
  result=subprocess.run(cmd,cwd=REPO)
  commands.append({'argv':cmd,'cwd':str(REPO),'exit_code':result.returncode,'seconds':round(time.monotonic()-started,3)})
  save('evaluating')
  if result.returncode not in allowed:raise RuntimeError(f'command failed: {result.returncode}')
 with (out/'.watcher.lock').open('a+') as lock:
  fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
  previous=None;deadline=time.monotonic()+8*3600;raws={}
  while time.monotonic()<deadline:
   readiness={};raws={}
   for condition,path in paths.items():
    if not path.exists():readiness[condition]='missing file';continue
    raw=path.read_bytes()
    try:ready=read_ready(raw,expected,condition)
    except json.JSONDecodeError:readiness[condition]='incomplete JSONL tail';continue
    if ready:raws[condition]=raw;readiness[condition]='490 unique expected IDs'
    else:readiness[condition]='partial coverage'
   hashes={c:digest(raw) for c,raw in raws.items()}
   save('waiting; labels unopened',readiness=readiness)
   if len(raws)==3 and hashes==previous:break
   previous=hashes if len(raws)==3 else None
   time.sleep(30)
  else:raise TimeoutError('8h watcher bound reached before stable complete coverage; no labels read')
  if membership.read_bytes()!=membership_bytes or family_source.read_bytes()!=family_bytes:raise ValueError('frozen evaluation specification changed')
  frozen_family=out/'comparisons.frozen.json'
  if frozen_family.exists() and frozen_family.read_bytes()!=family_bytes:raise ValueError('previous family differs')
  if not frozen_family.exists():frozen_family.write_bytes(family_bytes)
  snapshot_root=out/'prediction-snapshots';snapshot_root.mkdir(exist_ok=True)
  snapshots={}
  for condition,raw in raws.items():
   path=snapshot_root/(condition+'.jsonl')
   if path.exists() and path.read_bytes()!=raw:raise ValueError('new predictions differ from existing evaluation snapshot')
   if not path.exists():
    with path.open('xb') as f:f.write(raw)
   snapshots[condition]=path
  save('all three outputs stable; label-reading gate passed',input_sha256=hashes)
  answers=Path('/scratch/yy5919/pyro-kd-derived/datasets/egoschema/subset_answers.json')
  report_paths={c:out/(c+'.evaluation.json') for c in CONDITIONS}
  for condition,report_path in report_paths.items():
   if not report_path.exists():
    run(['scripts/evaluate_egoschema_qa.py','--input',str(snapshots[condition]),'--answers',str(answers),'--pilot-config',str(membership),'--split','heldout','--output',str(report_path)],allowed=(0,1))
   report=json.loads(report_path.read_text())
   if report.get('expected')!=490 or report.get('attempted')!=490 or report.get('input_sha256')!=hashes[condition] or report.get('config_sha256')!=digest(membership_bytes) or report.get('answers_sha256')!=config['answers_file_sha256']:
    raise ValueError('evaluation coverage/provenance mismatch')
   if len(report['rows'])!=490 or {r['id'] for r in report['rows']}!=expected:raise ValueError('evaluation denominator mismatch')
   if any(r['correct'] for r in report['rows'] if r['status']!='valid'):raise ValueError('failed prediction counted correct')
  paired=out/'paired.jsonl';paired_prov=paired.with_suffix('.jsonl.provenance.json')
  if not paired.exists():
   args=['scripts/export_qa_paired_metrics.py','--dataset','egoschema']
   for c in CONDITIONS:args+=['--report',c,str(report_paths[c])]
   run(args+['--output',str(paired)])
  provenance=json.loads(paired_prov.read_text())
  if provenance['output_sha256']!=digest(paired.read_bytes()) or {s['condition']:s['sha256'] for s in provenance['sources']}!={c:digest(path.read_bytes()) for c,path in report_paths.items()}:
   raise ValueError('paired export changed or belongs to different reports')
  comparison=out/'comparisons.json'
  if not comparison.exists():run(['scripts/analyze_family.py','--input',str(paired),'--family',str(frozen_family),'--unit','cluster','--resamples','10000','--seed','0','--output',str(comparison)])
  stats=json.loads(comparison.read_text())
  if stats['input_sha256']!=digest(paired.read_bytes()) or stats['family_sha256']!=family_hash or len(stats['comparisons'])!=3:raise ValueError('comparison provenance mismatch')
  if any(len(c['units'])!=490 or c['excluded'] or c['wilcoxon']['family_size']!=3 for c in stats['comparisons']):raise ValueError('paired coverage/correction mismatch')
  reports={c:json.loads(path.read_text()) for c,path in report_paths.items()}
  summary={'scope':'frozen490-video visual-only EgoSchema diagnostic','source_commit':COMMIT,'family_sha256':family_hash,'input_sha256':hashes,
           'conditions':{c:{k:r[k] for k in ('expected','attempted','valid_predictions','correct','accuracy','status_counts')} for c,r in reports.items()},'comparisons_path':str(comparison)}
  write_json(out/'summary.json',summary);save('complete',summary=summary)
  print(json.dumps(summary),flush=True)

if __name__=='__main__':main()
