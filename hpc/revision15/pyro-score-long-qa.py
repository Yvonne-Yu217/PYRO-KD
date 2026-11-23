"""Score all frozen QA cells once, after inference finishes for all conditions."""
import json,time,subprocess,sys,hashlib
from pathlib import Path
repo=Path('/scratch/yy5919/pyro-revision15-formal');sys.path.insert(0,str(repo))
from scripts.export_qa_paired_metrics import export_reports
from scripts.analyze_paired import analyze
from src.evaluation.stats import holm_adjust
planpath=Path('/scratch/yy5919/pyro-revision15-records/configs/long_qa_comparisons.json');plan=json.loads(planpath.read_text())
root=Path('/scratch/yy5919/pyro-resume-results/long-qa-frozen');out=root/'evaluation';out.mkdir(exist_ok=True)
deadline=time.monotonic()+21600
while time.monotonic()<deadline:
 ready=True
 for name in plan['conditions']:
  p=root/(name+'-224.jsonl')
  try:r=[json.loads(x) for x in p.read_text().splitlines()];ready &= len({x['id'] for x in r})==36
  except (FileNotFoundError,ValueError):ready=False
 if ready:break
 time.sleep(30)
else:raise RuntimeError('all-condition prediction coverage not ready after6hours')
reports=[]
for name in plan['conditions']:
 inp=root/(name+'-224.jsonl');score=out/(name+'.json')
 if not score.exists():
  subprocess.run([sys.executable,'scripts/evaluate_video_mme_qa.py','--input',str(inp),'--annotations','/scratch/yy5919/pyro-kd-derived/video-mme/annotations.parquet','--output',str(score)],cwd=repo,check=False)
 report=json.loads(score.read_text());assert report['input_sha256']==hashlib.sha256(inp.read_bytes()).hexdigest();reports.append((name,report))
 print(json.dumps({'condition':name,'correct':report['correct'],'expected':report['expected'],'status_counts':report['status_counts']}),flush=True)
rows=export_reports(reports,'video-mme');(out/'paired-metrics.jsonl').write_text(''.join(json.dumps(r)+'\n'for r in rows))
comparisons=[analyze(rows,a,b,'accuracy','cluster',plan['resamples'],plan['seed']) for a,b in plan['contrasts']]
for comparison,p in zip(comparisons,holm_adjust([c['wilcoxon'].get('pvalue') for c in comparisons])):comparison['wilcoxon']['p_holm']=p
(out/'comparisons.json').write_text(json.dumps({'plan_sha256':hashlib.sha256(planpath.read_bytes()).hexdigest(),'interpretation':plan['interpretation'],'comparisons':comparisons},indent=2)+'\n')
print(json.dumps({'complete':True,'conditions':len(reports),'comparisons':len(comparisons),'out':str(out)}),flush=True)
