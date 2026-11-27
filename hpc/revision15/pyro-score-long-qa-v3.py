"""Score all frozen QA cells once, after inference finishes for all conditions."""
import json,time,subprocess,sys,hashlib
from pathlib import Path
repo=Path('/scratch/yy5919/pyro-revision15-formal');sys.path.insert(0,str(repo))
from scripts.export_qa_paired_metrics import export_reports
from scripts.analyze_paired import analyze
from src.evaluation.stats import holm_adjust
planpath=Path('/scratch/yy5919/pyro-revision15-records/configs/long_qa_comparisons.json');plan=json.loads(planpath.read_text())
root=Path('/scratch/yy5919/pyro-resume-results/long-qa-frozen');out=root/'evaluation';out.mkdir(exist_ok=True)
import fcntl
lock=(out/'scoring.lock').open('a+');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
membership_path=repo/'configs/video_mme_long_questions.json'
membership=json.loads(membership_path.read_text());expected={r['question_id']:r['video_id'] for r in membership['questions']};assert len(expected)==36
plan_bytes=planpath.read_bytes();membership_bytes=membership_path.read_bytes()
def complete_rows(rows):
 return (len(rows)==36 and len({r.get('id') for r in rows})==36
         and {r.get('id'):r.get('video_id') for r in rows}==expected
         and len({r.get('run_fingerprint') for r in rows})==1
         and all(r.get('run_fingerprint') for r in rows))
previous=None;stable_hashes=None;deadline=time.monotonic()+21600
while time.monotonic()<deadline:
 current={}
 for name in plan['conditions']:
  p=root/(name+'-224.jsonl')
  try:
   raw=p.read_bytes()
   if not raw.endswith(b'\n'):continue
   rows=[json.loads(x) for x in raw.splitlines()]
   if complete_rows(rows):current[name]=hashlib.sha256(raw).hexdigest()
  except (FileNotFoundError,ValueError):pass
 if len(current)==len(plan['conditions']) and current==previous:
  stable_hashes=current;break
 previous=current;time.sleep(30)
else:raise RuntimeError('exact stable condition coverage not ready after6hours')
assert planpath.read_bytes()==plan_bytes and membership_path.read_bytes()==membership_bytes
(out/'readiness.json').write_text(json.dumps({'expected':expected,'prediction_sha256':stable_hashes,'conditions':len(plan['conditions']),'plan_sha256':hashlib.sha256(plan_bytes).hexdigest(),'driver_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),'policy':'exact36ID/video mappings, onefingerprint, completeJSONL, stableacross2polls30sapart beforelabels'},indent=2)+'\n')
reports=[]
for name in plan['conditions']:
 inp=root/(name+'-224.jsonl');score=out/(name+'.json')
 assert hashlib.sha256(inp.read_bytes()).hexdigest()==stable_hashes[name]
 if not score.exists():
  subprocess.run([sys.executable,'scripts/evaluate_video_mme_qa.py','--input',str(inp),'--annotations','/scratch/yy5919/pyro-kd-derived/video-mme/annotations.parquet','--output',str(score)],cwd=repo,check=False)
 report=json.loads(score.read_text());assert report['input_sha256']==hashlib.sha256(inp.read_bytes()).hexdigest();reports.append((name,report))
 print(json.dumps({'condition':name,'correct':report['correct'],'expected':report['expected'],'status_counts':report['status_counts']}),flush=True)
rows=export_reports(reports,'video-mme');(out/'paired-metrics.jsonl').write_text(''.join(json.dumps(r)+'\n'for r in rows))
comparisons=[analyze(rows,a,b,'accuracy','cluster',plan['resamples'],plan['seed']) for a,b in plan['contrasts']]
for comparison,p in zip(comparisons,holm_adjust([c['wilcoxon'].get('pvalue') for c in comparisons])):comparison['wilcoxon']['p_holm']=p
(out/'comparisons.json').write_text(json.dumps({'plan_sha256':hashlib.sha256(planpath.read_bytes()).hexdigest(),'interpretation':plan['interpretation'],'comparisons':comparisons},indent=2)+'\n')
print(json.dumps({'complete':True,'conditions':len(reports),'comparisons':len(comparisons),'out':str(out)}),flush=True)
